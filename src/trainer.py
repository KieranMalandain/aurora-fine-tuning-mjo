# src/trainer.py
"""
Trainer for Aurora MJO fine-tuning  - Perlmutter production version (v3).

v3 changes, each mapped to a finding in campaign-debugging-handoff.md:

  * DETACHED ROLLOUT (handoff §2  - the blocking illegal-memory-access).
    `training.rollout.backprop: "detached"` trains multi-step rollout with
    per-step backward + detached state advance ("pushforward" training).
    Activation memory is O(1) in the rollout horizon k  - the same footprint
    as single-step training (~40 GB observed)  - so ACTIVATION CHECKPOINTING
    IS NOT NEEDED AT ALL, which sidesteps the unresolved checkpointing IMA
    entirely. Gradients do not flow *through* time (each step treats its
    input state as a constant), but every step is still trained on the
    model's own predictions, which is the property that fixes rollout drift
    and is standard practice for autoregressive weather-model fine-tuning.
    `backprop: "full"` keeps the old whole-chain BPTT for when/if the
    checkpointing bug is fixed (see tools/repro_ima_matrix.py).

  * DDP-SAFE NON-FINITE POLICY (handoff §3a  - the `continue` guard was a
    latent deadlock).  Skipping backward on ONE rank while others run a
    gradient-syncing backward hangs the NCCL all-reduce.  Policy now:
      - non-sync (no_sync / accumulation) backwards: per-rank skip is safe,
        because only the final sync backward all-reduces the ACCUMULATED
        gradient buckets, and rank-asymmetric contributions just average.
      - the sync backward is guarded COLLECTIVELY: ranks all_reduce a
        finiteness flag; a rank holding a non-finite loss backwards a
        zero-valued surrogate that still touches every trainable parameter,
        so DDP's reducer fires on all ranks and no one deadlocks or desyncs.

  * VALIDATION FIXES (handoff §3b):
      - all-batches-skipped now returns NaN + ERROR log (was: 0.0, which
        `0.0 < inf` would have crowned "best" forever);
      - `is_best` requires math.isfinite(val_loss) (NaN < best is False, so
        best-checkpointing silently never fired);
      - first non-finite val batch dumps per-variable input/target min/max  -
        pinpoints WHICH variable is poisoned without a separate run;
      - metrics.jsonl records n_ok / n_skipped;
      - validation coverage: when training.max_val_batches is set, the val
        loader is built over EVENLY-SPACED indices across the whole val
        range instead of the first N chronological samples of 2016.

Everything else (step-level checkpointing, resumable sampler, wall-clock
guard, SIGUSR1, JSONL metrics, OOM emergency save) is unchanged from v2.
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
from torch.utils.data import DataLoader, DistributedSampler, Subset

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
        raise ValueError("No trainable parameters  - check freeze_backbone/use_lora config.")
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name!r}")


def _build_scheduler(optimizer, cfg: dict, optim_steps_per_epoch: int):
    """Total steps are OPTIMIZER steps = ceil(batches / grad_accum)."""
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
# Batch plumbing helpers
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
    """Upsample a 1deg batch + targets to 0.25deg (720x1440) on GPU."""
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


# FIX 4 (AURORA_MJO_GAMEPLAN §Finding 3 / FIX 4, phase-3 LoRA rollout only):
# `_advance_batch` splices the model's own prediction straight into the next
# step's input history with no clamping, bypassing Aurora's built-in
# `apply_rollout_input_clipping`. From a cold start the freshly-initialized
# ttr/tcwv (and, post-FIX-2/6, msl) heads emit garbage; that garbage becomes
# an OOD input for step 2, which goes non-finite. Clamp every fed-back
# prediction to a generous physical range before it re-enters the model.
# Bounds are physical (pre-normalization) units; tune if your convention
# differs. Only variables present in this table are clamped - everything
# else passes through unchanged.
_ROLLOUT_CLAMP = {
    "msl": (3.0e4, 1.1e5), "2t": (150., 350.), "10u": (-120., 120.), "10v": (-120., 120.),
    "ttr": (-600., 50.), "tcwv": (0., 120.), "q": (0., 0.1),
}


def _clamp_fed(name, t):
    """Clamp a fed-back rollout prediction to a sane physical range, if configured."""
    b = _ROLLOUT_CLAMP.get(name)
    return t.clamp(*b) if (b is not None and isinstance(t, torch.Tensor)) else t


def _advance_batch(in_batch, pred_batch, step_index: int, detach: bool = False):
    """Roll the 2-step history window forward with the model prediction.

    detach=True severs the autograd graph at the step boundary  - required by
    the detached-rollout mode so each step's graph can be freed immediately
    after its own backward().
    """
    from datetime import timedelta
    from aurora.batch import Batch, Metadata

    dt = timedelta(hours=6)

    def _maybe_detach(t):
        return t.detach() if (detach and isinstance(t, torch.Tensor)) else t

    new_surf = {}
    for k in in_batch.surf_vars:
        history = _maybe_detach(in_batch.surf_vars[k])
        pred_k = pred_batch.surf_vars.get(k)
        if pred_k is not None:
            fed = _clamp_fed(k, _maybe_detach(pred_k))
            new_surf[k] = torch.cat([history[:, 1:, ...], fed], dim=1)
        else:
            new_surf[k] = torch.cat([history[:, 1:, ...], history[:, -1:, ...]], dim=1)

    new_atmos = {}
    for k in in_batch.atmos_vars:
        history = _maybe_detach(in_batch.atmos_vars[k])
        pred_k = pred_batch.atmos_vars.get(k)
        if pred_k is not None:
            fed = _clamp_fed(k, _maybe_detach(pred_k))
            new_atmos[k] = torch.cat([history[:, 1:, ...], fed], dim=1)
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

    Shuffle depends ONLY on (seed, epoch), so after a restart we regenerate
    the identical permutation and skip the first `skip_samples` indices  -
    zero wasted netCDF I/O, exact data continuity.  Works single-GPU too.
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

    # v3: representative validation coverage.  With max_val_batches set, the
    # old sequential loader evaluated the SAME first ~N chronological samples
    # (~50 days of early 2016) every epoch.  Subsample evenly across the
    # whole val range instead.
    if split == "val":
        world = dist.get_world_size() if dist.is_initialized() else 1
        max_val = cfg.get("training", {}).get("max_val_batches", None)
        if max_val is not None and len(dataset) > max_val * world:
            idx = torch.linspace(0, len(dataset) - 1, steps=max_val * world).round().long()
            idx = torch.unique(idx)
            dataset = Subset(dataset, idx.tolist())

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
        shuffle=False,
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
        self.max_steps_per_epoch = train_cfg.get("max_steps_per_epoch", None)
        self.max_val_batches = train_cfg.get("max_val_batches", None)
        self.time_limit_s = float(train_cfg.get("time_limit_hours", 0) or 0) * 3600.0
        self._t_start = time.time()

        optim_steps_per_epoch = math.ceil(
            (self.max_steps_per_epoch or len(self.train_loader)) / self.grad_accum_steps
        )
        self.scheduler = _build_scheduler(self.optimizer, cfg, optim_steps_per_epoch)

        # ------------------------------------------------------------------
        # Checkpointing + metrics
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
        # v3: "detached" (per-step backward, O(1) memory, no checkpointing
        # needed) or "full" (whole-chain BPTT, requires checkpointing at k>=2).
        self.rollout_backprop = rollout_cfg.get("backprop", "full").lower()
        if self.rollout_enabled:
            log.info(f"Rollout enabled: backprop={self.rollout_backprop} "
                     f"k={self.rollout_start}->{self.rollout_max}")

        # ------------------------------------------------------------------
        # Resume counters
        # ------------------------------------------------------------------
        self.global_step = 0
        self.start_epoch = 1
        self.resume_batch_offset = 0
        self.best_val = float("inf")
        self._loss_hist: list[float] = []
        self._steps_since_save = 0
        self._nonfinite_train_batches = 0
        # FIX 1 (AURORA_MJO_GAMEPLAN §Finding 2 / FIX 1): counts optimizer
        # steps skipped because gradients were non-finite even though the
        # loss itself was finite (the bf16-attention corruption path that
        # otherwise poisons Adam's moment buffers forever).
        self._nonfinite_grad_steps = 0

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

        self._usr1 = False
        try:
            signal.signal(signal.SIGUSR1, self._on_usr1)
        except (ValueError, OSError):
            pass

    # ------------------------------------------------------------------
    def _on_usr1(self, signum, frame):
        log.warning("SIGUSR1 received (SLURM timeout warning)  - will checkpoint & exit.")
        self._usr1 = True

    def _should_stop_for_time(self) -> bool:
        if self._usr1:
            return True
        if self.time_limit_s > 0 and (time.time() - self._t_start) > self.time_limit_s:
            return True
        return False

    def _sync_stop_flag(self, local_stop: bool) -> bool:
        if not dist.is_initialized():
            return local_stop
        t = torch.tensor([1.0 if local_stop else 0.0], device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item() > 0)

    # ------------------------------------------------------------------
    # DDP-safe non-finite machinery (v3)
    # ------------------------------------------------------------------

    def _collective_all_finite(self, local_finite: bool) -> bool:
        """True iff EVERY rank's loss is finite.  Must be called by all ranks
        the same number of times (we call it exactly once per sync backward)."""
        if not dist.is_initialized():
            return local_finite
        t = torch.tensor([1.0 if local_finite else 0.0], device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item() > 0.5)

    def _surrogate_sync_backward(self):
        """Zero-valued backward that touches every trainable parameter.

        Used on a rank whose loss is non-finite during the GRADIENT-SYNCING
        backward: DDP's reducer needs a backward pass on every rank to fire
        its bucket hooks, or the other ranks deadlock in all-reduce.  This
        surrogate contributes exactly zero gradient while keeping the
        collective aligned, and the all-reduced (accumulated) gradients stay
        identical on all ranks afterwards.
        """
        z = None
        for p in self.model.parameters():
            if p.requires_grad:
                term = p.float().sum()
                z = term if z is None else z + term
        if z is not None:
            (z * 0.0).backward()

    def _guarded_backward(self, loss: torch.Tensor, is_sync: bool,
                          context: str, epoch: int, batch_in_epoch: int) -> bool:
        """Backward `loss` with the v3 non-finite policy.

        Returns True if a real (non-surrogate) backward ran.
        """
        finite = bool(torch.isfinite(loss).item())
        if is_sync:
            if self._collective_all_finite(finite):
                self.grad_scaler.scale(loss).backward()
                return True
            # Someone (possibly us) is non-finite on the sync pass.
            if not finite:
                self._nonfinite_train_batches += 1
                log.warning(f"[nan-guard] non-finite loss at epoch {epoch} "
                            f"batch {batch_in_epoch} ({context}, sync pass)  - "
                            f"surrogate backward on this rank.")
                self._surrogate_sync_backward()
            else:
                # Our loss is fine, another rank's is not: run the real
                # backward  - asymmetric contributions are safe because the
                # all-reduce output is identical on every rank.
                self.grad_scaler.scale(loss).backward()
            return finite
        # Non-sync (no_sync/accumulation) pass: per-rank skip is safe.
        if finite:
            self.grad_scaler.scale(loss).backward()
            return True
        self._nonfinite_train_batches += 1
        log.warning(f"[nan-guard] non-finite loss at epoch {epoch} "
                    f"batch {batch_in_epoch} ({context}, no-sync pass)  - skipped.")
        return False

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def load_checkpoint(self, path, weights_only: bool = False):
        counters = CheckpointManager.load(
            path, self.model, self.optimizer, self.scheduler, self.grad_scaler,
            weights_only_model=weights_only,
        )
        if not weights_only:
            self.global_step = counters["global_step"]
            self.best_val = counters["best_val"]
            self.start_epoch = max(1, counters["epoch"])
            self.resume_batch_offset = counters["batch_in_epoch"]
            self._nonfinite_train_batches = counters.get("nonfinite_train_batches", 0)
            self._nonfinite_grad_steps = counters.get("nonfinite_grad_steps", 0)
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
    # Per-step loss (shared by full-BPTT, detached, and validation paths)
    # ------------------------------------------------------------------

    def _single_step_losses(self, current_batch, target_dict) -> tuple[dict, object]:
        """One forward pass + all component losses for ONE rollout step.

        Returns (losses_dict_of_tensors_unweighted, pred_batch).
        """
        model_out = self.model(current_batch)
        if isinstance(model_out, tuple):
            pred_batch, mjo_pred = model_out
        else:
            pred_batch, mjo_pred = model_out, None

        zero = torch.zeros((), device=self.device)
        out = {"grid": zero, "spectral": zero, "mjo_head": zero, "moisture_budget": zero}

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
                out["grid"] = sum(var_losses) / len(var_losses)

        if self.spectral_loss is not None and self.spectral_weight > 0:
            pred_t, tgt_t = _extract_batch_outputs(pred_batch, target_dict, self.device)
            if pred_t is not None:
                out["spectral"] = self.spectral_weight * self.spectral_loss(pred_t, tgt_t)

        if self.use_mjo_head_loss and mjo_pred is not None and "mjo_targets" in target_dict:
            out["mjo_head"] = self.mjo_head_weight * nn.functional.l1_loss(
                mjo_pred.to(self.device), target_dict["mjo_targets"].to(self.device))

        if self.moisture_budget_loss is not None:
            out["moisture_budget"] = self.moisture_budget_weight * \
                self.moisture_budget_loss(current_batch, pred_batch)

        return out, pred_batch

    def _select_target(self, target_dict_list, step_idx: int) -> dict:
        if isinstance(target_dict_list, dict):
            return target_dict_list
        return target_dict_list[min(step_idx, len(target_dict_list) - 1)]

    # ------------------------------------------------------------------
    # Full-BPTT composite loss (validation + rollout.backprop == "full")
    # ------------------------------------------------------------------

    def _compute_loss(self, in_batch, target_dict_list, epoch: int = 1) -> dict:
        k = self._current_rollout_steps(epoch)
        weights = self._step_weights(k)
        acc = {"grid": 0.0, "spectral": 0.0, "mjo_head": 0.0, "moisture_budget": 0.0}
        current_batch = in_batch

        for step_idx in range(k):
            w = weights[step_idx]
            target_dict = self._select_target(target_dict_list, step_idx)
            step_losses, pred_batch = self._single_step_losses(current_batch, target_dict)
            for name in acc:
                acc[name] = acc[name] + w * step_losses[name]
            if step_idx < k - 1:
                current_batch = _advance_batch(current_batch, pred_batch, step_idx)

        def _to_tensor(v):
            return v if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device)

        losses = {k_: _to_tensor(v) for k_, v in acc.items()}
        losses["total"] = sum(losses.values())
        return losses

    # ------------------------------------------------------------------
    # Detached rollout: per-step backward, O(1) activation memory in k.
    # ------------------------------------------------------------------

    def _detached_rollout_step(self, in_batch, target_dict_list, epoch: int,
                               is_boundary: bool, batch_in_epoch: int) -> dict:
        """Train one micro-batch with detached ("pushforward") rollout.

        For each rollout step j:
          forward -> per-step loss -> BACKWARD IMMEDIATELY (freeing step j's
          graph) -> advance state with DETACHED predictions.

        DDP context discipline: every backward except the LAST rollout step
        of a BOUNDARY micro-batch runs under no_sync(); the last one is the
        gradient-syncing pass and is guarded collectively.  Because DDP
        reduces the ACCUMULATED per-parameter gradients on the sync pass,
        the k per-step backwards accumulate exactly like grad-accum
        micro-steps do.

        Returns a dict of float loss components (for logging).
        """
        k = self._current_rollout_steps(epoch)
        weights = self._step_weights(k)
        is_ddp = hasattr(self.model, "no_sync")
        items = {"grid": 0.0, "spectral": 0.0, "mjo_head": 0.0,
                 "moisture_budget": 0.0, "total": 0.0}

        current_batch = in_batch
        for step_idx in range(k):
            is_sync = is_boundary and (step_idx == k - 1)
            sync_ctx = self.model.no_sync() if (is_ddp and not is_sync) \
                else contextlib.nullcontext()
            with sync_ctx:
                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                    step_losses, pred_batch = self._single_step_losses(
                        current_batch, self._select_target(target_dict_list, step_idx))
                    step_total = sum(step_losses.values())
                    scaled = step_total * (weights[step_idx] / self.grad_accum_steps)
                self._guarded_backward(scaled, is_sync,
                                       context=f"detached k={step_idx+1}/{k}",
                                       epoch=epoch, batch_in_epoch=batch_in_epoch)

            # Logging accumulation (weighted, NaN-sanitized for display only).
            for name, v in step_losses.items():
                val = float(v.detach().item())
                if math.isfinite(val):
                    items[name] += weights[step_idx] * val
            tot = float(step_total.detach().item())
            if math.isfinite(tot):
                items["total"] += weights[step_idx] * tot

            # Advance with the graph CUT  - this is what frees step j's
            # activations and keeps memory flat in k.
            if step_idx < k - 1:
                current_batch = _advance_batch(current_batch, pred_batch,
                                               step_idx, detach=True)
        return items

    # ------------------------------------------------------------------
    # Batch prep
    # ------------------------------------------------------------------

    def _prep_batch(self, batch):
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
        if not self.is_main:
            return
        due_steps = self._steps_since_save >= self.save_every_steps
        due_plateau = False
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
            nonfinite_train_batches=self._nonfinite_train_batches,
            nonfinite_grad_steps=self._nonfinite_grad_steps,
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
            sampler.set_epoch(epoch)
        if isinstance(sampler, ResumableDistributedSampler):
            sampler.skip_samples = skip_batches * loader.batch_size

        epoch_len = len(loader) + skip_batches
        if self.max_steps_per_epoch is not None:
            epoch_len = min(epoch_len, self.max_steps_per_epoch)

        k = self._current_rollout_steps(epoch)
        use_detached = (self.rollout_enabled and self.rollout_backprop == "detached")

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

                if use_detached:
                    # Detached path manages its own no_sync per rollout step.
                    items = self._detached_rollout_step(
                        in_batch, target_dict_list, epoch, is_boundary, batch_in_epoch)
                    losses_view = items
                    step_loss = items["total"]
                else:
                    sync_ctx = self.model.no_sync() if (is_ddp and not is_boundary) \
                        else contextlib.nullcontext()
                    with sync_ctx:
                        with torch.amp.autocast('cuda', enabled=self.use_amp,
                                                dtype=self.amp_dtype):
                            losses = self._compute_loss(in_batch, target_dict_list,
                                                        epoch=epoch)
                            loss = losses["total"] / self.grad_accum_steps
                        self._guarded_backward(loss, is_boundary, context="full",
                                               epoch=epoch, batch_in_epoch=batch_in_epoch)
                    losses_view = {n: float(v.detach().item()) for n, v in losses.items()}
                    step_loss = losses_view["total"]

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                msg = str(e)
                if "out of memory" in msg.lower() or "cublas" in msg.lower() \
                        or "illegal memory access" in msg.lower():
                    log.error(f"GPU memory/CUDA failure at epoch {epoch} "
                              f"batch {batch_in_epoch}: {msg[:300]}")
                    try:
                        self._save(f"emergency_step_{self.global_step:07d}",
                                   epoch, batch_in_epoch - 1, reason="OOM/CUDA")
                    except Exception as save_err:
                        log.error(f"Emergency checkpoint save ALSO failed: {save_err}")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                raise

            if math.isfinite(step_loss):
                epoch_loss += step_loss
                seen += 1
                self._loss_hist.append(step_loss)
                if len(self._loss_hist) > 4 * self.plateau_window:
                    self._loss_hist = self._loss_hist[-2 * self.plateau_window:]

            if is_boundary:
                self.grad_scaler.unscale_(self.optimizer)  # no-op under bf16, harmless

                # --- FIX 1: reject non-finite gradients before they poison
                # Adam's moment buffers. A finite loss can still produce
                # non-finite gradients (saturated bf16 attention softmax ->
                # finite activation, NaN in its Jacobian); clip_grad_norm_
                # alone does NOT catch this because a NaN total norm just
                # propagates through the clip without raising. Without this
                # guard, one bad batch corrupts Adam's m/v buffers forever
                # (they update *before* the lr multiply, so this happens
                # even at lr≈0 during warmup) and every subsequent step and
                # forward pass goes NaN.
                local_finite = True
                for p in self.model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        local_finite = False
                        break
                # Collective so all DDP ranks step-or-skip together.
                grads_finite = self._collective_all_finite(local_finite)

                if grads_finite:
                    if self.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    if self.scheduler is not None:
                        self.scheduler.step()
                else:
                    self._nonfinite_grad_steps += 1
                    if self.is_main:
                        log.warning(
                            f"[grad-guard] non-finite grads at step {self.global_step} "
                            f"(epoch {epoch} batch {batch_in_epoch}) - optimizer step "
                            f"skipped (total={self._nonfinite_grad_steps})."
                        )
                    # Do NOT step optimizer or scheduler; just clear the
                    # poisoned grads below and move on.

                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self._steps_since_save += 1

                self._maybe_step_checkpoint(epoch, batch_in_epoch)

                if self._sync_stop_flag(self._should_stop_for_time()):
                    log.warning("Wall-clock guard tripped  - saving final state.")
                    self._save(f"step_{self.global_step:07d}", epoch, batch_in_epoch,
                               reason="time-limit")
                    if dist.is_initialized():
                        dist.barrier()
                    stopped = True
                    break

            if (batch_in_epoch % self.log_every == 0 or batch_in_epoch == 1) and self.is_main:
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - self._t_start
                log.info(
                    f"Epoch {epoch:03d} | batch {batch_in_epoch:05d}/{epoch_len} "
                    f"| step {self.global_step} | rollout_k={k}"
                    f"{'(det)' if use_detached else ''} "
                    f"| loss={step_loss:.4f} "
                    f"(grid={losses_view['grid']:.4f}, "
                    f"spec={losses_view['spectral']:.4f}, "
                    f"mjo={losses_view['mjo_head']:.4f}, "
                    f"phys={losses_view['moisture_budget']:.4f}) "
                    f"| lr={lr:.2e} | t={elapsed/3600:.2f}h "
                    f"| nan_skipped={self._nonfinite_train_batches} "
                    f"| grad_skipped={self._nonfinite_grad_steps}"
                )
                self.metrics.log({
                    "split": "train", "epoch": epoch, "batch": batch_in_epoch,
                    "step": self.global_step, "loss": step_loss,
                    "grid": losses_view["grid"],
                    "spectral": losses_view["spectral"],
                    "mjo_head": losses_view["mjo_head"],
                    "moisture_budget": losses_view["moisture_budget"],
                    "lr": lr, "rollout_k": k,
                    "detached": use_detached,
                    "nan_skipped_total": self._nonfinite_train_batches,
                    "nonfinite_grad_steps_total": self._nonfinite_grad_steps,
                })

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
    def _log_bad_val_batch(self, in_batch, target_dict_list):
        """Per-variable input/target/PREDICTION ranges for the first non-finite
        val batch  - identifies WHICH variable is poisoned without a separate
        debug run.

        FIX 3 (AURORA_MJO_GAMEPLAN §Finding 4): the previous version logged
        only surf inputs, q/t atmos inputs, and targets  - never z/u/v, never
        statics, and critically never the model's own prediction, which is
        the tensor that actually goes NaN. We now run one extra forward pass
        here and log every surf/atmos prediction variable so FIX 1-2 can be
        *confirmed* (expect: no PRED.* non-finite) rather than assumed.
        """
        def rng(t):
            t = t.float()
            fin = torch.isfinite(t)
            tag = "" if fin.all() else f" NON-FINITE({int((~fin).sum())})"
            v = t[fin]
            return (f"[{v.min():.3e}, {v.max():.3e}]{tag}" if v.numel() else "[empty]")

        lines = []
        # Inputs: ALL surface + ALL atmos (not just q, t).
        for name, t in in_batch.surf_vars.items():
            lines.append(f"in.surf.{name}={rng(t)}")
        for name, t in in_batch.atmos_vars.items():
            lines.append(f"in.atmos.{name}={rng(t)}")

        # The missing measurement: the model's own prediction, per variable.
        self.model.eval()
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            out = self.model(in_batch)
        pred = out[0] if isinstance(out, tuple) else out
        for attr in ("surf_vars", "atmos_vars"):
            for name, t in getattr(pred, attr).items():
                lines.append(f"PRED.{attr[:4]}.{name}={rng(t)}")

        tgt = target_dict_list[0] if isinstance(target_dict_list, list) else target_dict_list
        for name, t in list(tgt.items())[:6]:
            lines.append(f"tgt.{name}={rng(t)}")

        log.error("[val-diag] first non-finite val batch: " + " | ".join(lines))

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        val_loss, n, skipped = 0.0, 0, 0
        logged_bad = False
        for batch in self.val_loader:
            if self.max_val_batches is not None and (n + skipped) >= self.max_val_batches:
                break
            in_batch, target_dict_list = self._prep_batch(batch)
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                losses = self._compute_loss(in_batch, target_dict_list, epoch=epoch)
            total = losses["total"].item()
            if not math.isfinite(total):
                skipped += 1
                if not logged_bad:
                    self._log_bad_val_batch(in_batch, target_dict_list)
                    logged_bad = True
                ts = getattr(in_batch.metadata, "time", ("?",))[0]
                log.warning(f"[nan-guard] non-finite VAL loss at time={ts}; skipping.")
                continue
            val_loss += total
            n += 1

        # v3 FIX: all-skipped must surface as NaN, never as a fake 0.0
        # ("0.0 < inf" would have marked every broken epoch as new-best).
        if n == 0:
            log.error(f"Epoch {epoch:03d} | VALIDATION PRODUCED ZERO FINITE BATCHES "
                      f"({skipped} skipped)  - reporting NaN. Run "
                      f"tools/diagnose_val_nan.py against the val years.")
            mean_val = float("nan")
        else:
            mean_val = val_loss / n

        if dist.is_initialized():
            # NaN propagates through the AVG all-reduce, which is what we
            # want: any rank with zero finite batches poisons (flags) the
            # global number rather than silently diluting it.
            t = torch.tensor(mean_val, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            mean_val = t.item()
        if self.is_main:
            log.info(f"Epoch {epoch:03d} | VAL loss={mean_val:.4f} "
                     f"({n} ok, {skipped} skipped)")
            self.metrics.log({"split": "val", "epoch": epoch, "step": self.global_step,
                              "loss": mean_val, "n_ok": n, "n_skipped": skipped})
        return mean_val

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def fit(self) -> int:
        """0 = complete (writes DONE); 99 = clean timeout stop (chain resumes)."""
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
                # v3 FIX: NaN can never become "best" (NaN < x is False, but
                # we make the requirement explicit and auditable).
                is_best = math.isfinite(val_loss) and val_loss < self.best_val
                if is_best:
                    self.best_val = val_loss
                self._save(f"epoch_{epoch:03d}", epoch + 1, 0,
                           val_loss=val_loss, is_best=is_best, reason="epoch-end")
            else:
                self._save(f"epoch_{epoch:03d}", epoch + 1, 0, reason="epoch-end")

            if dist.is_initialized():
                dist.barrier()

        if self.is_main:
            log.info(f"Training complete. Best val loss: {self.best_val:.4f}")
            (self.save_dir / "DONE").write_text(f"best_val={self.best_val}\n")
        return 0
