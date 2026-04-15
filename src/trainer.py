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
  - Autoregressive rollout is scaffolded but disabled by default (Phase 1).
  - Modular so LoRA and rollout curriculum can be layered on top cleanly.
"""

import os
import math
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.loss import TropicalWeightedL1Loss, SpectralLoss

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

        self._step = 0          # global optimiser step counter
        self._saved_ckpts = []  # for keep_last_n bookkeeping

    # ------------------------------------------------------------------
    # Forward + loss
    # ------------------------------------------------------------------

    def _compute_loss(self, in_batch, target_dict) -> dict:
        """
        Run a single forward pass and compute the composite loss.

        Returns a dict with keys: 'total', 'grid', 'spectral', 'mjo_head'.
        All values are scalar tensors.
        """
        # Move Aurora Batch metadata to device where needed
        in_batch = in_batch  # Batch is a dataclass; tensors move with .to() below

        # Forward (single step -- rollout disabled in Phase 1)
        pred_batch = self.model(in_batch)

        losses = {}

        # ---- Grid loss ----
        if self.use_grid_loss:
            pred_t, tgt_t = _extract_batch_outputs(pred_batch, target_dict, self.device)
            if pred_t is not None:
                # Reshape to (B, Lat, Lon) for the spatial weighting
                # TropicalWeightedL1Loss broadcasts over (B, T, Lat, Lon)
                losses["grid"] = self.grid_loss(pred_t, tgt_t)
            else:
                losses["grid"] = torch.tensor(0.0, device=self.device)
        else:
            losses["grid"] = torch.tensor(0.0, device=self.device)

        # ---- Spectral loss ----
        if self.spectral_loss is not None and self.spectral_weight > 0:
            pred_t, tgt_t = _extract_batch_outputs(pred_batch, target_dict, self.device)
            losses["spectral"] = self.spectral_weight * self.spectral_loss(pred_t, tgt_t)
        else:
            losses["spectral"] = torch.tensor(0.0, device=self.device)

        # ---- MJO head loss ----
        # Disabled in Phase 1.  When enabled, the model must return an 'mjo_preds'
        # attribute from the forward pass (RMM1, RMM2, Amplitude).
        if self.use_mjo_head_loss and hasattr(pred_batch, "mjo_preds") and "mjo_targets" in target_dict:
            mjo_l1 = nn.functional.l1_loss(
                pred_batch.mjo_preds.to(self.device),
                target_dict["mjo_targets"].to(self.device),
            )
            losses["mjo_head"] = self.mjo_head_weight * mjo_l1
        else:
            losses["mjo_head"] = torch.tensor(0.0, device=self.device)

        losses["total"] = sum(v for v in losses.values())
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

            losses = self._compute_loss(in_batch, target_dict)
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
                log.info(
                    f"Epoch {epoch:03d} | step {batch_idx:05d}/{num_batches} "
                    f"| loss={losses['total'].item():.4f} "
                    f"(grid={losses['grid'].item():.4f}, "
                    f"spec={losses['spectral'].item():.4f}, "
                    f"mjo={losses['mjo_head'].item():.4f}) "
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

            losses = self._compute_loss(in_batch, target_dict)
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
