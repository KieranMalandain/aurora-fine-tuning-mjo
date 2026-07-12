# src/trainer.py
"""
Trainer for Aurora MJO fine-tuning — Perlmutter production version.

What changed vs. the previous trainer (each maps to an observed failure):

  * STEP-LEVEL CHECKPOINTING.  One LANL epoch is ~13.5k steps/rank (>6 h);
    epoch-only saves meant a 12 h job could die checkpoint-less.  We now save
    every `checkpointing.save_every_n_steps` (default 500) optimizer steps,
    on loss plateau, at every epoch end / best-val, when the wall-clock guard
    trips, and on SIGUSR1 from SLURM.
  * TRUE RESUME.  epoch / global_step / batch_in_epoch / scheduler / AMP
    scaler / RNG are all restored; the DistributedSampler shuffling is
    deterministic in (seed, epoch), so we regenerate the same permutation and
    skip the already-consumed indices WITHOUT touching the netCDF files.
  * WALL-CLOCK GUARD.  `training.time_limit_hours` (set to 11.0 for a 11.5 h
    SLURM allocation): when exceeded, all ranks agree via all_reduce, save,
    and exit with code 99 so the chained SLURM job resumes cleanly.
  * OOM / cuBLAS HANDLING.  Job 52464118 died with
    CUBLAS_STATUS_EXECUTION_FAILED inside a bf16 backward — on A100 that is
    an OOM/workspace failure in disguise.  Fixes: gradient checkpointing is
    forced ON whenever rollout is enabled (enabling it mid-run on a live DDP
    graph, as before, is unsafe), physics loss runs in fp32, and any
    OOM/cuBLAS error triggers an emergency checkpoint before re-raising so
    the job chain restarts from the last good state.
  * GRAD-ACCUM + DDP no_sync().  Gradients are only all-reduced on the
    boundary micro-step.
  * JSONL METRICS every `logging.log_every_n_steps` (set 100), append-only →
    resumable across jobs.
"""

import math
import contextlib
import logging
import signal
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from src.loss import TropicalWeightedL1Loss, SpectralLoss, MoistureBudgetLoss
from src.checkpoint import CheckpointManager, MetricsLogger

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def _build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    opt_cfg = cfg["training"]["optimizer"]
    name = opt_cfg.get("name", "adamw").lower()
    lr = opt_cfg["lr"]
    wd = opt_cfg.get("weight_decay", 1e-5)
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters — check freeze_backbone/use_lora config.")
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name!r}")


def _build_scheduler(optimizer, cfg: dict, optim_steps_per_epoch: int):
    """NOTE: total steps are OPTIMIZER steps = ceil(batches / grad_accum),
    not raw batch count — the old code overestimated by grad_accum×."""
    sched_cfg = cfg["training"]["scheduler"]
    name = sched_cfg.get("name", "none").lower()
    total_epochs = cfg["training"]["epochs"]
    warmup = sched_cfg.get("warmup_steps", 0)
    eta_min = sched_cfg.get("eta_min", 0.0)

    if name in ("none", "constant"):
        return None

    if name == "cosine":
        total_steps = max(1, total_epochs * optim_steps_per_epoch)
        for g in optimizer.param_groups:
            g.setdefault("initial_lr", g["lr"])
        base_lr = optimizer.param_groups[0]["initial_lr"]

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = min(1.0, (step - warmup) / max(total_steps - warmup, 1))
            floor = eta_min / base_lr
            return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unknown scheduler: {name!r}")


# ---------------------------------------------------------------------------
# Batch plumbing helpers (unchanged behaviour, kept from original)
# ---------------------------------------------------------------------------

def _extract_batch_outputs(pred_batch, target_dict, device):
    pred_parts, tgt_parts = [], []
    for attr in ("surf_vars", "atmos_vars"):
        if hasattr(pred_batch, attr):
            for k, v in getattr(pred_batch, attr).items():
                if k in target_dict:
                    pred_parts.append(v.to(device))
                    tgt_parts.append(target_dict[k].to(device))
    if not pred_parts:
        return None, None
    pred_tensor = torch.cat([t.reshape(t.shape[0], -1) for t in pred_parts], dim=-1)
    tgt_tensor = torch.cat([t.reshape(t.shape[0], -1) for t in tgt_parts], dim=-1)
    return pred_tensor, tgt_tensor


def _align_shapes(pred_t, tgt_t):
    while pred_t.ndim > tgt_t.ndim:
        if pred_t.shape[1] == 1:
            pred_t = pred_t.squeeze(1)
        else:
            break
    while tgt_t.ndim > pred_t.ndim:
        if tgt_t.shape[0] == 1 or tgt_t.shape[1] == 1:
            tgt_t = tgt_t.squeeze(0) if tgt_t.shape[0] == 1 else tgt_t.squeeze(1)
        else:
            break
    return pred_t, tgt_t


def _upsample_batch_gpu(in_batch, surf_targets_list, atmos_targets_list, device):
    """Upsample a 1° batch + targets to 0.25° (720×1440) on GPU."""
    from aurora import Batch
    TARGET_SIZE = (720, 1440)

    def _up(t):
        orig = t.shape
        if orig[-2:] == TARGET_SIZE:
            return t
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            t = t.unsqueeze(0)
        elif t.ndim == 5:
            B, T, C, H, W = t.shape
            t = t.reshape(B * T, C, H, W)
        elif t.ndim == 4:
            pass
        else:
            return t
        t = F.interpolate(t, size=TARGET_SIZE, mode='bilinear', align_corners=False)
        if len(orig) == 2:
            return t.squeeze(0).squeeze(0)
        elif len(orig) == 3:
            return t.squeeze(0)
        elif len(orig) == 5:
            return t.reshape(orig[0], orig[1], orig[2], *TARGET_SIZE)
        return t

    new_surf = {k: _up(v.to(device)) for k, v in in_batch.surf_vars.items()}
    new_atmos = {k: _up(v.to(device)) for k, v in in_batch.atmos_vars.items()}
    new_static = {k: v.to(device) for k, v in in_batch.static_vars.items()}

    up_batch = Batch(
        surf_vars=new_surf, atmos_vars=new_atmos,
        static_vars=new_static, metadata=in_batch.metadata,
    )

    target_dict_list = []
    for surf_targets, atmos_targets in zip(surf_targets_list, atmos_targets_list):
        target_dict = {}
        for k, v in surf_targets.items():
            target_dict[k] = _up(v.to(device))
        for k, v in atmos_targets.items():
            target_dict[k] = _up(v.to(device))
        target_dict_list.append(target_dict)

    return up_batch, target_dict_list


def _advance_batch(in_batch, pred_batch, step_index: int):
    """Roll the 2-step history window forward with the model prediction."""
    from datetime import timedelta
    from aurora.batch import Batch, Metadata

    dt = timedelta(hours=6)

    new_surf = {}
    for k in in_batch.surf_vars:
        history = in_batch.surf_vars[k]
        pred_k = pred_batch.surf_vars.get(k)
        if pred_k is not None:
            new_surf[k] = torch.cat([history[:, 1:, ...], pred_k], dim=1)
        else:
            new_surf[k] = torch.cat([history[:, 1:, ...], history[:, -1:, ...]], dim=1)

    new_atmos = {}
    for k in in_batch.atmos_vars:
        history = in_batch.atmos_vars[k]
        pred_k = pred_batch.atmos_vars.get(k)
        if pred_k is not None:
            new_atmos[k] = torch.cat([history[:, 1:, ...], pred_k], dim=1)
        else:
            new_atmos[k] = torch.cat([history[:, 1:, ...], history[:, -1:, ...]], dim=1)

    new_time = tuple(t + dt for t in in_batch.metadata.time)
    new_metadata = Metadata(
        lat=in_batch.metadata.lat,
        lon=in_batch.metadata.lon,
        time=new_time,
        atmos_levels=in_batch.metadata.atmos_levels,
        rollout_step=step_index + 1,
    )
    return Batch(
        surf_vars=new_surf,
        static_vars=in_batch.static_vars,
        atmos_vars=new_atmos,
        metadata=new_metadata,
    )


# ---------------------------------------------------------------------------
# Resumable sampler + dataloader construction
# ---------------------------------------------------------------------------

class ResumableDistributedSampler(DistributedSampler):
    """DistributedSampler whose per-epoch permutation can be fast-forwarded.

    DistributedSampler's shuffle depends ONLY on (seed, epoch), so after a
    restart we regenerate the identical permutation and simply skip the first
    `skip_samples` indices — zero wasted netCDF I/O, exact data continuity.
    Works single-GPU too (num_replicas=1, rank=0).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_samples = 0

    def __iter__(self):
        indices = list(super().__iter__())
        return iter(indices[self.skip_samples:])

    def __len__(self):
        return max(0, super().__len__() - self.skip_samples)


def build_dataloader(cfg: dict, split: str) -> DataLoader:
    """Construct the dataloader for `split` in {"train", "val"}."""
    data_cfg = cfg["data"]
    use_dummy = data_cfg.get("use_dummy", True)

    rollout_cfg = cfg.get("training", {}).get("rollout", {})
    max_rollout_steps = rollout_cfg.get("max_steps", 1) if rollout_cfg.get("enabled", False) else 1

    if use_dummy:
        from src.dummy_dataset import MJODataset, load_and_combine_files
        dummy_cfg = data_cfg.get("dummy", {})
        surface_files = dummy_cfg.get("surface_files", [])
        pressure_files = dummy_cfg.get("pressure_files", [])
        static_file = dummy_cfg.get("static_file", "")
        if not surface_files or not pressure_files or not static_file:
            raise ValueError("Dummy dataset paths not set in config['data']['dummy'].")
        surface_ds = load_and_combine_files(surface_files)
        pressure_ds = load_and_combine_files(pressure_files)
        dataset = MJODataset(surface_ds, pressure_ds, static_file,
                             max_rollout_steps=max_rollout_steps)
        collate = MJODataset.collate_fn
    else:
        from src.dataset import LANLMJODataset, collate_fn as collate
        real_cfg = data_cfg.get("real", {})
        years = real_cfg.get("train_years", [1980, 2015]) if split == "train" \
            else real_cfg.get("val_years", [2016, 2019])
        dataset = LANLMJODataset(
            start_year=years[0], end_year=years[1],
            root_dir=data_cfg.get("root"),
            slt_path=data_cfg.get("slt_path"),
            max_rollout_steps=max_rollout_steps,
        )

    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    sampler = ResumableDistributedSampler(
        dataset,
        num_replicas=world,
        rank=rank,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        seed=cfg.get("experiment", {}).get("seed", 42),
    )

    num_workers = data_cfg.get("num_workers", 4)
    return DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 1),
        shuffle=False,                      # sampler owns the shuffle
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=data_cfg.get("pin_memory", True),
        collate_fn=collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(data_cfg.get("prefetch_factor", 4) if num_workers > 0 else None),
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Trains an Aurora-based MJO model with production checkpoint/resume."""

    # exit code the SLURM chain script interprets as "timed out cleanly,
    # checkpoint written, please resume in the next job"
    EXIT_TIMEOUT = 99

    def __init__(self, model, cfg, device, train_loader=None, val_loader=None,
                 is_main: bool = True):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.is_main = is_main

        # ------------------------------------------------------------------
        # Losses
        # ------------------------------------------------------------------
        loss_cfg = cfg.get("loss", {})

        grid_cfg = loss_cfg.get("grid", {})
        lat_coords = torch.linspace(90, -90, 720)
        self.grid_loss = TropicalWeightedL1Loss(
            lat_coords=lat_coords,
            tropics_bbox=grid_cfg.get("tropics_bbox", [-20, 20]),
            tropics_weight=grid_cfg.get("tropics_weight", 1.0),
            extratropics_weight=grid_cfg.get("extratropics_weight", 0.1),
        ).to(device)
        self.use_grid_loss = grid_cfg.get("enabled", True)

        spec_cfg = loss_cfg.get("spectral", {})
        self.spectral_loss = SpectralLoss().to(device) if spec_cfg.get("enabled", False) else None
        self.spectral_weight = float(spec_cfg.get("weight", 0.0))

        mjo_cfg = loss_cfg.get("mjo_head", {})
        self.use_mjo_head_loss = mjo_cfg.get("enabled", False)
        self.mjo_head_weight = float(mjo_cfg.get("weight", 0.0))

        phys_cfg = loss_cfg.get("moisture_budget", {})
        self.use_moisture_budget = phys_cfg.get("enabled", False)
        self.moisture_budget_weight = float(phys_cfg.get("weight", 0.0))
        if self.use_moisture_budget and self.moisture_budget_weight > 0:
            pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500,
                               600, 700, 850, 925, 1000]
            self.moisture_budget_loss = MoistureBudgetLoss(
                pressure_levels=pressure_levels,
                latitudes=torch.linspace(90, -90, 720).tolist(),
                longitudes=torch.linspace(0, 359.75, 1440).tolist(),
                dt_seconds=phys_cfg.get("dt_seconds", 21600),
                tropics_bbox=tuple(phys_cfg.get("tropics_bbox", [-20, 20])),
            ).to(device)
            log.info("Moisture-budget physics loss enabled (weight=%.4f)",
                     self.moisture_budget_weight)
        else:
            self.moisture_budget_loss = None

        # ------------------------------------------------------------------
        # Optimizer & data
        # ------------------------------------------------------------------
        self.optimizer = _build_optimizer(model, cfg)

        if train_loader is not None and val_loader is not None:
            log.info("Using pre-built DataLoaders (injected by caller).")
            self.train_loader = train_loader
            self.val_loader = val_loader
        else:
            log.info("Building train dataloader…")
            self.train_loader = build_dataloader(cfg, split="train")
            log.info("Building val dataloader…")
            self.val_loader = build_dataloader(cfg, split="val")

        # ------------------------------------------------------------------
        # Training knobs
        # ------------------------------------------------------------------
        train_cfg = cfg.get("training", {})
        self.epochs = train_cfg.get("epochs", 10)
        self.grad_accum_steps = max(1, train_cfg.get("grad_accum_steps", 1))
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.log_every = cfg.get("logging", {}).get("log_every_n_steps", 100)
        self.val_every = cfg.get("logging", {}).get("val_every_n_epochs", 1)
        # cap epoch length so one epoch fits the SLURM window (None = full)
        self.max_steps_per_epoch = train_cfg.get("max_steps_per_epoch", None)
        self.max_val_batches = train_cfg.get("max_val_batches", None)
        # wall-clock guard: 11.0 h inside a 11.5 h allocation → 30 min buffer
        self.time_limit_s = float(train_cfg.get("time_limit_hours", 0) or 0) * 3600.0
        self._t_start = time.time()

        optim_steps_per_epoch = math.ceil(
            (self.max_steps_per_epoch or len(self.train_loader)) / self.grad_accum_steps
        )
        self.scheduler = _build_scheduler(self.optimizer, cfg, optim_steps_per_epoch)

        # ------------------------------------------------------------------
        # Checkpointing (step-level) + metrics
        # ------------------------------------------------------------------
        ckpt_cfg = cfg.get("checkpointing", {})
        self.save_dir = Path(ckpt_cfg.get("save_dir", "checkpoints/run"))
        self.save_every_steps = int(ckpt_cfg.get("save_every_n_steps", 500))
        self.plateau_window = int(ckpt_cfg.get("plateau_window", 200))
        self.plateau_rel_delta = float(ckpt_cfg.get("plateau_rel_delta", 0.01))
        self.ckpt = CheckpointManager(
            self.save_dir, keep_last_n=ckpt_cfg.get("keep_last_n", 3), is_main=is_main
        )
        self.metrics = MetricsLogger(self.save_dir / "metrics.jsonl", is_main=is_main)

        # ------------------------------------------------------------------
        # Rollout curriculum
        # ------------------------------------------------------------------
        rollout_cfg = train_cfg.get("rollout", {})
        self.rollout_enabled = rollout_cfg.get("enabled", False)
        self.rollout_start = max(1, rollout_cfg.get("start_steps", 1))
        self.rollout_max = max(self.rollout_start, rollout_cfg.get("max_steps", 1))
        self.rollout_incr_every = max(1, rollout_cfg.get("step_increase_every_n_epochs", 2))
        self.rollout_weighting = rollout_cfg.get("step_loss_weighting", "uniform")

        # ------------------------------------------------------------------
        # Resume counters (populated by load_checkpoint / train.py)
        # ------------------------------------------------------------------
        self.global_step = 0        # optimizer steps completed, all epochs
        self.start_epoch = 1        # epoch to (re)start in
        self.resume_batch_offset = 0  # batches already consumed in start_epoch
        self.best_val = float("inf")
        self._loss_hist: list[float] = []   # for plateau detection
        self._steps_since_save = 0

        # ------------------------------------------------------------------
        # AMP
        # ------------------------------------------------------------------
        self.use_amp = train_cfg.get("use_amp", False)
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.grad_scaler = torch.amp.GradScaler(
            'cuda', enabled=(self.use_amp and self.amp_dtype == torch.float16)
        )
        if self.use_amp:
            log.info(f"AMP enabled with dtype={self.amp_dtype}")

        # ------------------------------------------------------------------
        # SIGUSR1 → graceful save+exit (backup to the wall-clock guard;
        # SLURM sends it via `#SBATCH --signal=B:USR1@1800`).
        # ------------------------------------------------------------------
        self._usr1 = False
        try:
            signal.signal(signal.SIGUSR1, self._on_usr1)
        except (ValueError, OSError):
            pass  # non-main thread / unsupported platform

    # ------------------------------------------------------------------
    def _on_usr1(self, signum, frame):
        log.warning("SIGUSR1 received (SLURM timeout warning) — will checkpoint & exit.")
        self._usr1 = True

    def _should_stop_for_time(self) -> bool:
        """Local decision only; must be all-reduced before acting (DDP)."""
        if self._usr1:
            return True
        if self.time_limit_s > 0 and (time.time() - self._t_start) > self.time_limit_s:
            return True
        return False

    def _sync_stop_flag(self, local_stop: bool) -> bool:
        """All ranks must agree on stopping or DDP collectives deadlock."""
        if not dist.is_initialized():
            return local_stop
        t = torch.tensor([1.0 if local_stop else 0.0], device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item() > 0)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def load_checkpoint(self, path, weights_only: bool = False):
        """Restore full trainer state from `path` (see CheckpointManager.load)."""
        counters = CheckpointManager.load(
            path, self.model, self.optimizer, self.scheduler, self.grad_scaler,
            weights_only_model=weights_only,
        )
        if not weights_only:
            self.global_step = counters["global_step"]
            self.best_val = counters["best_val"]
            self.start_epoch = max(1, counters["epoch"])
            self.resume_batch_offset = counters["batch_in_epoch"]
            # If we stopped exactly at an epoch boundary, advance.
            epoch_len = self.max_steps_per_epoch or len(self.train_loader)
            if self.resume_batch_offset >= epoch_len:
                self.start_epoch += 1
                self.resume_batch_offset = 0

    # ------------------------------------------------------------------
    # Rollout helpers
    # ------------------------------------------------------------------

    def _current_rollout_steps(self, epoch: int) -> int:
        if not self.rollout_enabled:
            return 1
        increments = (epoch - 1) // self.rollout_incr_every
        return min(self.rollout_start + increments, self.rollout_max)

    def _step_weights(self, k: int) -> list:
        if k == 1:
            return [1.0]
        if self.rollout_weighting == "final_heavy":
            early_w = 0.5 / (k - 1)
            return [early_w] * (k - 1) + [0.5]
        return [1.0 / k] * k

    # ------------------------------------------------------------------
    # Forward + loss (logic unchanged; physics loss internally fixed in loss.py)
    # ------------------------------------------------------------------

    def _compute_loss(self, in_batch, target_dict_list, epoch: int = 1) -> dict:
        k = self._current_rollout_steps(epoch)
        weights = self._step_weights(k)
        acc = {"grid": 0.0, "spectral": 0.0, "mjo_head": 0.0, "moisture_budget": 0.0}
        current_batch = in_batch

        for step_idx in range(k):
            w = weights[step_idx]
            if isinstance(target_dict_list, dict):
                target_dict = target_dict_list
            else:
                target_dict = target_dict_list[min(step_idx, len(target_dict_list) - 1)]

            model_out = self.model(current_batch)
            if isinstance(model_out, tuple):
                pred_batch, mjo_pred = model_out
            else:
                pred_batch, mjo_pred = model_out, None

            if self.use_grid_loss:
                var_losses = []
                for attr in ("surf_vars", "atmos_vars"):
                    if hasattr(pred_batch, attr):
                        for var_name, pred_t in getattr(pred_batch, attr).items():
                            if var_name in target_dict:
                                tgt_t = target_dict[var_name].to(self.device).float()
                                p = pred_t.to(self.device).float()
                                p, tgt_t = _align_shapes(p, tgt_t)
                                var_losses.append(self.grid_loss(p, tgt_t))
                if var_losses:
                    acc["grid"] = acc["grid"] + w * (sum(var_losses) / len(var_losses))

            if self.spectral_loss is not None and self.spectral_weight > 0:
                pred_t, tgt_t = _extract_batch_outputs(pred_batch, target_dict, self.device)
                if pred_t is not None:
                    acc["spectral"] = acc["spectral"] + \
                        w * self.spectral_weight * self.spectral_loss(pred_t, tgt_t)

            if self.use_mjo_head_loss and mjo_pred is not None and "mjo_targets" in target_dict:
                mjo_l1 = nn.functional.l1_loss(
                    mjo_pred.to(self.device), target_dict["mjo_targets"].to(self.device))
                acc["mjo_head"] = acc["mjo_head"] + w * self.mjo_head_weight * mjo_l1

            if self.moisture_budget_loss is not None:
                mb = self.moisture_budget_loss(current_batch, pred_batch)
                acc["moisture_budget"] = acc["moisture_budget"] + \
                    w * self.moisture_budget_weight * mb

            if step_idx < k - 1:
                current_batch = _advance_batch(current_batch, pred_batch, step_idx)

        def _to_tensor(v):
            return v if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device)

        losses = {k_: _to_tensor(v) for k_, v in acc.items()}
        losses["total"] = sum(losses.values())
        return losses

    # ------------------------------------------------------------------
    # Batch normalisation helper
    # ------------------------------------------------------------------

    def _prep_batch(self, batch):
        """Unpack, upsample if needed, move to device, force fp32+contiguous."""
        if len(batch) == 2:
            in_batch, target_dict_list = batch
        elif len(batch) == 3:
            in_batch, surf_out_list, atmos_out_list = batch
            if isinstance(surf_out_list, dict):
                surf_out_list = [surf_out_list]
                atmos_out_list = [atmos_out_list]
            needs_upsample = next(iter(surf_out_list[0].values())).shape[-1] < 1440
            if needs_upsample:
                in_batch, target_dict_list = _upsample_batch_gpu(
                    in_batch, surf_out_list, atmos_out_list, self.device)
            else:
                target_dict_list = [{**s, **a} for s, a in zip(surf_out_list, atmos_out_list)]
        else:
            raise ValueError(f"Unexpected batch tuple length: {len(batch)}")

        for group in (in_batch.surf_vars, in_batch.atmos_vars, in_batch.static_vars):
            for k_var in group:
                group[k_var] = group[k_var].to(self.device).float().contiguous()
        return in_batch, target_dict_list

    # ------------------------------------------------------------------
    # Checkpoint decision + emit
    # ------------------------------------------------------------------

    def _maybe_step_checkpoint(self, epoch: int, batch_in_epoch: int):
        """Save every `save_every_steps` optimizer steps OR on loss plateau."""
        if not self.is_main:
            return
        due_steps = self._steps_since_save >= self.save_every_steps
        due_plateau = False
        # Plateau: relative change of mean loss between the two most recent
        # half-windows below plateau_rel_delta → the model has settled;
        # snapshot it so a crash costs nothing.
        n = self.plateau_window
        if not due_steps and len(self._loss_hist) >= 2 * n and self._steps_since_save >= n:
            prev = sum(self._loss_hist[-2 * n:-n]) / n
            recent = sum(self._loss_hist[-n:]) / n
            if prev > 0 and abs(prev - recent) / prev < self.plateau_rel_delta:
                due_plateau = True
        if due_steps or due_plateau:
            self._save(f"step_{self.global_step:07d}", epoch, batch_in_epoch,
                       reason="plateau" if due_plateau else "interval")

    def _save(self, tag: str, epoch: int, batch_in_epoch: int,
              val_loss: float = float("nan"), is_best: bool = False, reason: str = ""):
        self.ckpt.save(
            tag, self.model, self.optimizer, self.scheduler, self.grad_scaler,
            epoch=epoch, global_step=self.global_step, batch_in_epoch=batch_in_epoch,
            best_val=self.best_val, val_loss=val_loss, config=self.cfg, is_best=is_best,
        )
        if reason:
            log.info(f"[ckpt] trigger: {reason}")
        self._steps_since_save = 0

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int, skip_batches: int = 0):
        """One training epoch. Returns (mean_loss, stopped_early: bool)."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loader = self.train_loader
        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)  # per-epoch reshuffle
        if isinstance(sampler, ResumableDistributedSampler):
            # CRITICAL RESUME STEP: same (seed, epoch) → same permutation;
            # skip exactly the batches already consumed before the restart.
            sampler.skip_samples = skip_batches * loader.batch_size

        epoch_len = len(loader) + skip_batches           # nominal full length
        if self.max_steps_per_epoch is not None:
            epoch_len = min(epoch_len, self.max_steps_per_epoch)

        epoch_loss, seen = 0.0, 0
        batch_in_epoch = skip_batches
        stopped = False
        is_ddp = hasattr(self.model, "no_sync")

        for batch in loader:
            if batch_in_epoch >= epoch_len:
                break
            batch_in_epoch += 1
            micro = (batch_in_epoch - 1) % self.grad_accum_steps
            is_boundary = (micro == self.grad_accum_steps - 1) or (batch_in_epoch == epoch_len)

            try:
                in_batch, target_dict_list = self._prep_batch(batch)

                # DDP no_sync on non-boundary micro-steps: skip the (very
                # expensive at 0.25°) gradient all-reduce until we step.
    
                sync_ctx = self.model.no_sync() if (is_ddp and not is_boundary) \
                    else contextlib.nullcontext()
                with sync_ctx:
                    with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                        losses = self._compute_loss(in_batch, target_dict_list, epoch=epoch)
                        loss = losses["total"] / self.grad_accum_steps
                    
                    if not torch.isfinite(losses["total"]):
                        ts = getattr(in_batch.metadata, "time", ("?",))[0]
                        log.warning(f"[nan-guard] non-finite loss at epoch {epoch} batch {batch_in_epoch} "
                                    f"time={ts} (grid={losses['grid'].item():.4g}); skipping this micro-batch.")
                        continue

                    self.grad_scaler.scale(loss).backward()

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                msg = str(e)
                if "out of memory" in msg.lower() or "cublas" in msg.lower() or "illegal memory access" in msg.lower():
                    log.error(f"GPU memory/cuBLAS failure at epoch {epoch} "
                            f"batch {batch_in_epoch}: {msg[:300]}")
                    try:
                        self._save(f"emergency_step_{self.global_step:07d}",
                                epoch, batch_in_epoch - 1, reason="OOM/CUBLAS")
                    except Exception as save_err:
                        log.error(f"Emergency checkpoint save ALSO failed: {save_err}")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                raise

            step_loss = losses["total"].item()
            epoch_loss += step_loss
            seen += 1
            self._loss_hist.append(step_loss)
            if len(self._loss_hist) > 4 * self.plateau_window:
                self._loss_hist = self._loss_hist[-2 * self.plateau_window:]

            if is_boundary:
                self.grad_scaler.unscale_(self.optimizer)
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self._steps_since_save += 1

                # ---- step-based / plateau checkpoint --------------------
                self._maybe_step_checkpoint(epoch, batch_in_epoch)

                # ---- synchronized timeout check (once per optim step) ----
                if self._sync_stop_flag(self._should_stop_for_time()):
                    log.warning("Wall-clock guard tripped — saving final state.")
                    self._save(f"step_{self.global_step:07d}", epoch, batch_in_epoch,
                               reason="time-limit")
                    if dist.is_initialized():
                        dist.barrier()
                    stopped = True
                    break

            if (batch_in_epoch % self.log_every == 0 or batch_in_epoch == 1) and self.is_main:
                lr = self.optimizer.param_groups[0]["lr"]
                k = self._current_rollout_steps(epoch)
                elapsed = time.time() - self._t_start
                log.info(
                    f"Epoch {epoch:03d} | batch {batch_in_epoch:05d}/{epoch_len} "
                    f"| step {self.global_step} | rollout_k={k} "
                    f"| loss={step_loss:.4f} "
                    f"(grid={losses['grid'].item():.4f}, "
                    f"spec={losses['spectral'].item():.4f}, "
                    f"mjo={losses['mjo_head'].item():.4f}, "
                    f"phys={losses['moisture_budget'].item():.4f}) "
                    f"| lr={lr:.2e} | t={elapsed/3600:.2f}h"
                )
                self.metrics.log({
                    "split": "train", "epoch": epoch, "batch": batch_in_epoch,
                    "step": self.global_step, "loss": step_loss,
                    "grid": losses["grid"].item(),
                    "spectral": losses["spectral"].item(),
                    "mjo_head": losses["mjo_head"].item(),
                    "moisture_budget": losses["moisture_budget"].item(),
                    "lr": lr, "rollout_k": self._current_rollout_steps(epoch),
                })

        # reset skip so later epochs iterate fully
        if isinstance(sampler, ResumableDistributedSampler):
            sampler.skip_samples = 0

        epoch_avg = epoch_loss / max(seen, 1)
        if dist.is_initialized():
            t = torch.tensor(epoch_avg, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            epoch_avg = t.item()
        return epoch_avg, stopped

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        val_loss, n, skipped = 0.0, 0, 0
        for batch in self.val_loader:
            if self.max_val_batches is not None and n >= self.max_val_batches:
                break
            in_batch, target_dict_list = self._prep_batch(batch)
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                losses = self._compute_loss(in_batch, target_dict_list, epoch=epoch)
            if not torch.isfinite(losses["total"]):
                ts = getattr(in_batch.metadata, "time", ("?",))[0]
                log.warning(f"[nan-guard] non-finite VAL loss at time={ts}; skipping.")
                skipped += 1
                continue
            val_loss += losses["total"].item()
            n += 1

        mean_val = val_loss / max(n, 1)
        if dist.is_initialized():
            t = torch.tensor(mean_val, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            mean_val = t.item()
        if self.is_main:
            log.info(f"Epoch {epoch:03d} | VAL loss={mean_val:.4f} ({n} batches, {skipped} skipped)")
            self.metrics.log({"split": "val", "epoch": epoch, "step": self.global_step,
                            "loss": mean_val, "skipped": skipped})
        return mean_val

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def fit(self) -> int:
        """Run training.  Returns process exit code:
        0   = all epochs complete (writes DONE marker),
        99  = clean timeout stop, checkpoint written → SLURM chain resumes.
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            skip = self.resume_batch_offset if epoch == self.start_epoch else 0
            if skip and self.is_main:
                log.info(f"Resuming epoch {epoch} at batch offset {skip}")
            train_loss, stopped = self.train_epoch(epoch, skip_batches=skip)
            if self.is_main:
                log.info(f"Epoch {epoch:03d} | TRAIN loss={train_loss:.4f}")
            if stopped:
                return self.EXIT_TIMEOUT

            if epoch % self.val_every == 0:
                val_loss = self.validate(epoch)
                is_best = val_loss < self.best_val
                if is_best:
                    self.best_val = val_loss
                # ALWAYS save at epoch end (old code skipped non-improving
                # epochs entirely → a whole epoch of work could be lost).
                self._save(f"epoch_{epoch:03d}", epoch + 1, 0,
                           val_loss=val_loss, is_best=is_best, reason="epoch-end")
            else:
                self._save(f"epoch_{epoch:03d}", epoch + 1, 0, reason="epoch-end")

            if dist.is_initialized():
                dist.barrier()  # nobody starts the next epoch mid-save

        if self.is_main:
            log.info(f"Training complete. Best val loss: {self.best_val:.4f}")
            (self.save_dir / "DONE").write_text(f"best_val={self.best_val}\n")
        return 0
