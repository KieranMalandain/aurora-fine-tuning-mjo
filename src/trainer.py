# src/trainer.py
"""
Trainer for Aurora MJO fine-tuning.

Implements:
  - train_epoch: single forward/backward pass over one DataLoader epoch
  - validate:    evaluation loop (no grad) with same loss reporting

Design principles:
  - All hyper-parameters come from the config dict; no hidden constants.
  - Dummy/real dataset selection is controlled by config['data']['use_dummy'].
  - Loss terms are individually gated by config flags so they can be toggled
    without touching this file.
  - Autoregressive rollout is enabled via config['training']['rollout']['enabled'].
    The curriculum starts at `start_steps` and grows by one every
    `step_increase_every_n_epochs`, capped at `max_steps`.  Phase 1 configs
    leave rollout disabled and behaviour is identical to the original single-step
    path.
  - Modular so LoRA and rollout curriculum can be layered on top cleanly.
"""

import os
import math
import logging
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.loss import TropicalWeightedL1Loss, SpectralLoss, MoistureBudgetLoss

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    opt_cfg = cfg["training"]["optimizer"]
    name = opt_cfg.get("name", "adamw").lower()
    lr = opt_cfg["lr"]
    wd = opt_cfg.get("weight_decay", 1e-5)
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))

    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name!r}")


def _build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    sched_cfg = cfg["training"]["scheduler"]
    name = sched_cfg.get("name", "none").lower()
    total_epochs = cfg["training"]["epochs"]
    warmup = sched_cfg.get("warmup_steps", 0)
    eta_min = sched_cfg.get("eta_min", 0.0)

    if name == "none" or name == "constant":
        return None

    if name == "cosine":
        total_steps = total_epochs * steps_per_epoch
        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return eta_min / optimizer.param_groups[0]["initial_lr"] + \
                   0.5 * (1.0 - eta_min / optimizer.param_groups[0]["initial_lr"]) * \
                   (1.0 + math.cos(math.pi * progress))
        # Store initial_lr before making the scheduler
        for g in optimizer.param_groups:
            g.setdefault("initial_lr", g["lr"])
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unknown scheduler: {name!r}")


def _extract_batch_outputs(pred_batch, target_dict, device):
    """
    Given an Aurora output Batch and a target dict, return stacked
    (pred_tensor, target_tensor) for the grid loss.

    Aurora's output Batch has .surf_vars and .atmos_vars as dicts of tensors.
    We only compare keys that exist in both pred and target.
    """
    pred_parts, tgt_parts = [], []

    if hasattr(pred_batch, "surf_vars"):
        for k, v in pred_batch.surf_vars.items():
            if k in target_dict:
                pred_parts.append(v.to(device))
                tgt_parts.append(target_dict[k].to(device))

    if hasattr(pred_batch, "atmos_vars"):
        for k, v in pred_batch.atmos_vars.items():
            if k in target_dict:
                pred_parts.append(v.to(device))
                tgt_parts.append(target_dict[k].to(device))

    if not pred_parts:
        return None, None

    pred_tensor = torch.cat([t.reshape(t.shape[0], -1) for t in pred_parts], dim=-1)
    tgt_tensor  = torch.cat([t.reshape(t.shape[0], -1) for t in tgt_parts],  dim=-1)
    return pred_tensor, tgt_tensor


def _advance_batch(in_batch, pred_batch, step_index: int):
    """Build the next input Batch from the previous prediction.

    Aurora keeps a rolling 2-timestep history window in surf_vars / atmos_vars
    (shape ``(B, t, ...)`` where ``t = 2``).  To advance one step we:

    1. Drop the oldest timestep (index 0) and append the new prediction at the
       back, giving shape ``(B, 2, ...)`` again.
    2. Advance ``metadata.time`` by 6 h (one Aurora step) for every element in
       the batch.
    3. Increment ``metadata.rollout_step`` so Aurora selects the correct LoRA
       adapter when LoRA is enabled.

    Args:
        in_batch: The Batch passed into the *current* step's forward pass.
        pred_batch: The Batch returned by the *current* step's forward pass.
        step_index: 0-indexed rollout step (used to set rollout_step).

    Returns:
        A new Batch ready to be fed into the next rollout step.
    """
    from aurora.batch import Batch, Metadata

    dt = timedelta(hours=6)

    # Advance the surface variables: roll history window.
    # pred_batch.surf_vars has shape (B, 1, H, W) -- the single predicted step.
    # in_batch.surf_vars has shape (B, 2, H, W) -- [t-1, t].
    # Next input window = [t, pred] i.e. drop oldest, append prediction.
    new_surf = {}
    for k in in_batch.surf_vars:
        history = in_batch.surf_vars[k]          # (B, 2, H, W)
        pred_k  = pred_batch.surf_vars.get(k)    # (B, 1, H, W) or None
        if pred_k is not None:
            new_surf[k] = torch.cat([history[:, 1:, ...], pred_k], dim=1)
        else:
            # Variable not in prediction; carry forward the last known state.
            new_surf[k] = torch.cat([history[:, 1:, ...], history[:, -1:, ...]], dim=1)

    new_atmos = {}
    for k in in_batch.atmos_vars:
        history = in_batch.atmos_vars[k]         # (B, 2, C, H, W)
        pred_k  = pred_batch.atmos_vars.get(k)   # (B, 1, C, H, W) or None
        if pred_k is not None:
            new_atmos[k] = torch.cat([history[:, 1:, ...], pred_k], dim=1)
        else:
            new_atmos[k] = torch.cat([history[:, 1:, ...], history[:, -1:, ...]], dim=1)

    # Advance time tags for each batch element.
    new_time = tuple(t + dt for t in in_batch.metadata.time)

    new_metadata = Metadata(
        lat=in_batch.metadata.lat,
        lon=in_batch.metadata.lon,
        time=new_time,
        atmos_levels=in_batch.metadata.atmos_levels,
        rollout_step=step_index + 1,  # step_index is 0-based; next step is +1
    )

    return Batch(
        surf_vars=new_surf,
        static_vars=in_batch.static_vars,   # static fields do not change
        atmos_vars=new_atmos,
        metadata=new_metadata,
    )


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataloader(cfg: dict, split: str) -> DataLoader:
    """
    Construct the appropriate dataloader for `split` in {"train", "val"}.
    Respects config['data']['use_dummy'] flag.
    """
    data_cfg = cfg["data"]
    use_dummy = data_cfg.get("use_dummy", True)

    if use_dummy:
        # --- Dummy dataset (Yale Bouchet one-month ERA5) ---
        from src.dummy_dataset import MJODataset, load_and_combine_files

        dummy_cfg = data_cfg.get("dummy", {})
        surface_files = dummy_cfg.get("surface_files", [])
        pressure_files = dummy_cfg.get("pressure_files", [])
        static_file = dummy_cfg.get("static_file", "")

        if not surface_files or not pressure_files or not static_file:
            raise ValueError(
                "Dummy dataset paths are not set in config['data']['dummy']. "
                "Please fill in surface_files, pressure_files, and static_file."
            )

        surface_ds = load_and_combine_files(surface_files)
        pressure_ds = load_and_combine_files(pressure_files)
        dataset = MJODataset(surface_ds, pressure_ds, static_file)
        collate = MJODataset.collate_fn

    else:
        # --- Real dataset (NERSC LANL) ---
        from src.dataset import LANLMJODataset

        real_cfg = data_cfg.get("real", {})
        if split == "train":
            years = real_cfg.get("train_years", [1980, 2015])
        elif split == "val":
            years = real_cfg.get("val_years", [2016, 2019])
        else:
            raise ValueError(f"Unknown split: {split!r}")

        dataset = LANLMJODataset(
                    start_year=years[0],
                    end_year=years[1],
                    root_dir=data_cfg.get("root"),
                )
        from src.dataset import collate_fn as collate

    loader = DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 1),
        shuffle=(split == "train"),
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=data_cfg.get("pin_memory", True),
        collate_fn=collate,
    )
    return loader


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Trains an Aurora-based MJO model for one or more epochs.

    Args:
        model:   The Aurora wrapper returned by src.model.load_model().
        cfg:     Config dict (loaded from a YAML, e.g. configs/phase1_baseline.yaml).
        device:  torch.device to use.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        device: torch.device,
        train_loader=None,
        val_loader=None,
    ):
        """
        Args:
            train_loader / val_loader: Optional pre-built DataLoaders.  When
                supplied (e.g. for smoke-tests) the Trainer skips calling
                build_dataloader() entirely, avoiding any file-path validation.
        """
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        # ------------------------------------------------------------------
        # Build losses
        # ------------------------------------------------------------------
        loss_cfg = cfg.get("loss", {})

        # Grid loss -- always constructed; disabled via weight = 0 if needed
        grid_cfg = loss_cfg.get("grid", {})
        # Aurora native resolution: lat from 90 to -90, 720 points
        lat_coords = torch.linspace(90, -90, 720)
        self.grid_loss = TropicalWeightedL1Loss(
            lat_coords=lat_coords,
            tropics_bbox=grid_cfg.get("tropics_bbox", [-20, 20]),
            tropics_weight=grid_cfg.get("tropics_weight", 1.0),
            extratropics_weight=grid_cfg.get("extratropics_weight", 0.1),
        ).to(device)
        self.use_grid_loss = grid_cfg.get("enabled", True)

        # Spectral loss (Phase 1: typically disabled)
        spec_cfg = loss_cfg.get("spectral", {})
        self.spectral_loss = SpectralLoss().to(device) if spec_cfg.get("enabled", False) else None
        self.spectral_weight = float(spec_cfg.get("weight", 0.0))

        # MJO head loss (Phase 1: disabled until head is wired)
        mjo_cfg = loss_cfg.get("mjo_head", {})
        self.use_mjo_head_loss = mjo_cfg.get("enabled", False)
        self.mjo_head_weight   = float(mjo_cfg.get("weight", 0.0))

        # Moisture-budget physics loss (Phase 2 only)
        phys_cfg = loss_cfg.get("moisture_budget", {})
        self.use_moisture_budget = phys_cfg.get("enabled", False)
        self.moisture_budget_weight = float(phys_cfg.get("weight", 0.0))
        if self.use_moisture_budget and self.moisture_budget_weight > 0:
            # Aurora native 0.25° grid: 720 lat × 1440 lon
            pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500,
                               600, 700, 850, 925, 1000]  # hPa
            lat_coords = torch.linspace(90, -90, 720).tolist()
            lon_coords = torch.linspace(0, 359.75, 1440).tolist()
            self.moisture_budget_loss = MoistureBudgetLoss(
                pressure_levels=pressure_levels,
                latitudes=lat_coords,
                longitudes=lon_coords,
                dt_seconds=phys_cfg.get("dt_seconds", 21600),
                tropics_bbox=tuple(phys_cfg.get("tropics_bbox", [-20, 20])),
            ).to(device)
            log.info("Moisture-budget physics loss enabled (weight=%.4f)",
                     self.moisture_budget_weight)
        else:
            self.moisture_budget_loss = None

        # ------------------------------------------------------------------
        # Optimiser & data
        # ------------------------------------------------------------------
        self.optimizer = _build_optimizer(model, cfg)

        # Accept pre-built loaders (e.g. synthetic smoke-test loaders) so that
        # no file I/O occurs during construction when they are injected.
        if train_loader is not None and val_loader is not None:
            log.info("Using pre-built DataLoaders (injected by caller).")
            self.train_loader = train_loader
            self.val_loader   = val_loader
        else:
            log.info("Building train dataloader…")
            self.train_loader = build_dataloader(cfg, split="train")
            log.info("Building val dataloader…")
            self.val_loader   = build_dataloader(cfg, split="val")

        self.scheduler = _build_scheduler(
            self.optimizer, cfg, steps_per_epoch=len(self.train_loader)
        )

        # ------------------------------------------------------------------
        # Misc training knobs
        # ------------------------------------------------------------------
        train_cfg = cfg.get("training", {})
        self.epochs           = train_cfg.get("epochs", 10)
        self.grad_accum_steps = max(1, train_cfg.get("grad_accum_steps", 1))
        self.max_grad_norm    = train_cfg.get("max_grad_norm", 1.0)
        self.log_every        = cfg.get("logging", {}).get("log_every_n_steps", 10)
        self.val_every        = cfg.get("logging", {}).get("val_every_n_epochs", 1)

        ckpt_cfg = cfg.get("checkpointing", {})
        self.save_dir       = Path(ckpt_cfg.get("save_dir", "checkpoints/phase1_baseline"))
        self.save_every     = ckpt_cfg.get("save_every_n_epochs", 1)
        self.keep_last_n    = ckpt_cfg.get("keep_last_n", 3)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Rollout curriculum config
        # ------------------------------------------------------------------
        rollout_cfg = train_cfg.get("rollout", {})
        self.rollout_enabled    = rollout_cfg.get("enabled", False)
        self.rollout_start      = max(1, rollout_cfg.get("start_steps", 1))
        self.rollout_max        = max(self.rollout_start, rollout_cfg.get("max_steps", 1))
        self.rollout_incr_every = max(1, rollout_cfg.get("step_increase_every_n_epochs", 2))
        self.rollout_weighting  = rollout_cfg.get("step_loss_weighting", "uniform")

        self._step = 0          # global optimiser step counter
        self._saved_ckpts = []  # for keep_last_n bookkeeping

    # ------------------------------------------------------------------
    # Rollout helpers
    # ------------------------------------------------------------------

    def _current_rollout_steps(self, epoch: int) -> int:
        """Return the active rollout horizon for this epoch.

        The curriculum starts at ``rollout_start`` and grows by 1 every
        ``rollout_incr_every`` epochs, up to ``rollout_max``.

        Examples (start=1, max=4, incr_every=2)::

            epoch 1-2  -> 1 step
            epoch 3-4  -> 2 steps
            epoch 5-6  -> 3 steps
            epoch 7+   -> 4 steps
        """
        if not self.rollout_enabled:
            return 1
        increments = (epoch - 1) // self.rollout_incr_every
        return min(self.rollout_start + increments, self.rollout_max)

    def _step_weights(self, k: int) -> list:
        """Return a list of per-step loss weights that sums to 1.

        Strategy is controlled by ``self.rollout_weighting``:

        - ``"uniform"``     : equal weight ``1/k`` at every step.
        - ``"final_heavy"`` : last step = 0.5; earlier steps share the rest
          equally.
        """
        if k == 1:
            return [1.0]
        if self.rollout_weighting == "final_heavy":
            early_w = 0.5 / (k - 1)
            return [early_w] * (k - 1) + [0.5]
        # default: uniform
        return [1.0 / k] * k

    # ------------------------------------------------------------------
    # Forward + loss
    # ------------------------------------------------------------------

    def _compute_loss(self, in_batch, target_dict, epoch: int = 1) -> dict:
        """
        Run one or more forward passes and compute the composite loss.

        When rollout is disabled (Phase 1) this is identical to the original
        single-step forward pass.  When enabled, the model is called k times
        autoregressively; the predicted state is used as input for the next
        step via :func:`_advance_batch`.  The per-step grid/spectral/mjo_head
        losses are accumulated with the weighting strategy specified in the
        config.

        Args:
            in_batch:    Initial Aurora Batch (from the DataLoader).
            target_dict: Dict of target tensors (single-step, from the
                         DataLoader).  Only used to compute losses against the
                         *step-1* target; for k > 1 steps we still compare
                         every prediction against this same target because the
                         dataset emits only one-step targets.
            epoch:       Current epoch number (1-indexed) used to look up the
                         curriculum step.

        Returns:
            Dict with keys: 'total', 'grid', 'spectral', 'mjo_head'.
            All values are scalar tensors.
        """
        k = self._current_rollout_steps(epoch)
        weights = self._step_weights(k)

        # Accumulate per-component losses weighted across rollout steps.
        acc = {"grid": 0.0, "spectral": 0.0, "mjo_head": 0.0, "moisture_budget": 0.0}

        current_batch = in_batch

        for step_idx in range(k):
            w = weights[step_idx]

            # ---- Forward pass ----
            model_out = self.model(current_batch)

            # Handle dual-head output (mjo_head enabled) vs plain Batch.
            if isinstance(model_out, tuple):
                pred_batch, mjo_pred = model_out
            else:
                pred_batch = model_out
                mjo_pred   = None

            # ---- Grid loss ----
            if self.use_grid_loss:
                pred_t, tgt_t = _extract_batch_outputs(
                    pred_batch, target_dict, self.device
                )
                if pred_t is not None:
                    acc["grid"] = acc["grid"] + w * self.grid_loss(pred_t, tgt_t)
            # (if not use_grid_loss, acc["grid"] stays 0.0)

            # ---- Spectral loss ----
            if self.spectral_loss is not None and self.spectral_weight > 0:
                pred_t, tgt_t = _extract_batch_outputs(
                    pred_batch, target_dict, self.device
                )
                if pred_t is not None:
                    acc["spectral"] = (
                        acc["spectral"]
                        + w * self.spectral_weight * self.spectral_loss(pred_t, tgt_t)
                    )

            # ---- MJO head loss ----
            if (
                self.use_mjo_head_loss
                and mjo_pred is not None
                and "mjo_targets" in target_dict
            ):
                mjo_l1 = nn.functional.l1_loss(
                    mjo_pred.to(self.device),
                    target_dict["mjo_targets"].to(self.device),
                )
                acc["mjo_head"] = acc["mjo_head"] + w * self.mjo_head_weight * mjo_l1

            # ---- Moisture-budget physics loss ----
            if self.moisture_budget_loss is not None:
                mb_loss = self.moisture_budget_loss(current_batch, pred_batch)
                acc["moisture_budget"] = acc["moisture_budget"] + w * self.moisture_budget_weight * mb_loss

            # ---- Advance state for next rollout step ----
            if step_idx < k - 1:
                current_batch = _advance_batch(current_batch, pred_batch, step_idx)

        # Convert accumulated floats / tensors to scalar tensors.
        def _to_tensor(v):
            if isinstance(v, torch.Tensor):
                return v
            return torch.tensor(v, device=self.device)

        losses = {k_: _to_tensor(v) for k_, v in acc.items()}
        losses["total"] = sum(losses.values())
        return losses

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        """
        Run one full training epoch.

        Returns:
            Mean total loss over the epoch.
        """
        self.model.train()
        self.optimizer.zero_grad()

        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            # Unpack -- dummy_dataset returns (in_batch, target_dict)
            # real dataset returns   (in_batch, surf_out, atmos_out)
            if len(batch) == 2:
                in_batch, target_dict = batch
            elif len(batch) == 3:
                in_batch, surf_out, atmos_out = batch
                # Merge surface and atmos targets into one dict
                target_dict = {**surf_out, **atmos_out}
            else:
                raise ValueError(f"Unexpected batch tuple length: {len(batch)}")

            losses = self._compute_loss(in_batch, target_dict, epoch=epoch)
            loss = losses["total"] / self.grad_accum_steps

            loss.backward()
            epoch_loss += losses["total"].item()

            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self._step += 1

            if batch_idx % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                k = self._current_rollout_steps(epoch)
                log.info(
                    f"Epoch {epoch:03d} | step {batch_idx:05d}/{num_batches} "
                    f"| rollout_k={k} "
                    f"| loss={losses['total'].item():.4f} "
                    f"(grid={losses['grid'].item():.4f}, "
                    f"spec={losses['spectral'].item():.4f}, "
                    f"mjo={losses['mjo_head'].item():.4f}, "
                    f"phys={losses['moisture_budget'].item():.4f}) "
                    f"| lr={lr:.2e}"
                )

        return epoch_loss / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """
        Run one full validation epoch (no gradient computation).

        Returns:
            Mean total loss over the validation set.
        """
        self.model.eval()

        val_loss = 0.0
        num_batches = len(self.val_loader)

        for batch in self.val_loader:
            if len(batch) == 2:
                in_batch, target_dict = batch
            elif len(batch) == 3:
                in_batch, surf_out, atmos_out = batch
                target_dict = {**surf_out, **atmos_out}
            else:
                raise ValueError(f"Unexpected batch tuple length: {len(batch)}")

            losses = self._compute_loss(in_batch, target_dict, epoch=epoch)
            val_loss += losses["total"].item()

        mean_val = val_loss / max(num_batches, 1)
        log.info(f"Epoch {epoch:03d} | VAL loss={mean_val:.4f}")
        return mean_val

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float):
        fname = self.save_dir / f"epoch_{epoch:03d}_val{val_loss:.4f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.cfg,
            },
            fname,
        )
        log.info(f"Saved checkpoint: {fname}")
        self._saved_ckpts.append(fname)

        # Prune to keep_last_n
        while len(self._saved_ckpts) > self.keep_last_n:
            old = self._saved_ckpts.pop(0)
            if old.exists():
                old.unlink()
                log.info(f"Removed old checkpoint: {old}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self):
        """
        Full training loop: run self.epochs epochs of train + optional val.
        """
        best_val = float("inf")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            log.info(f"Epoch {epoch:03d} | TRAIN loss={train_loss:.4f}")

            if epoch % self.val_every == 0:
                val_loss = self.validate(epoch)
                if val_loss < best_val:
                    best_val = val_loss
                    self.save_checkpoint(epoch, val_loss)
            elif epoch % self.save_every == 0:
                self.save_checkpoint(epoch, val_loss=float("nan"))

        log.info(f"Training complete. Best val loss: {best_val:.4f}")
