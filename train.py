#!/usr/bin/env python3
"""
train.py Aurora MJO fine-tuning entry point (unified, resumable).

One entry point, four modes (config overlays in configs/unified.yaml):

  python train.py --config configs/unified.yaml --mode baseline
  python train.py --config configs/unified.yaml --mode physics_informed
  python train.py --config configs/unified.yaml --mode lora
  python train.py --config configs/unified.yaml --mode combined

Resume semantics:
  --resume auto      (default) resume from the latest checkpoint in the
                     mode's checkpointing.save_dir if one exists;
                     otherwise start fresh (warm-starting from
                     experiment.init_from if set for this mode).
  --resume <path>    resume from an explicit checkpoint.
  --resume none      always start fresh.

Smoke-test (no GPU, tiny synthetic data):
  python train.py --config configs/unified.yaml --mode baseline --smoke-test
"""

import argparse
import logging
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

# Workaround for NERSC Errno 524 filelock issue with huggingface_hub, and
# make the HF cache PERSISTENT: /tmp is node-local and wiped, which forced
# every job (x4 ranks) to re-download the pretrained weights.

if "NERSC_HOST" in os.environ:
    os.environ.setdefault("HF_HUB_DISABLE_FILE_LOCKS", "1")
    # Backup: monkey-patch filelock just in case the env var isn't enough
    try:
        import filelock
        class DummyLock:
            def __init__(self, *args, **kwargs): pass
            def acquire(self, *args, **kwargs): return self
            def release(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args, **kwargs): pass
        filelock.FileLock = DummyLock
    except ImportError:
        pass

    if "HF_HOME" not in os.environ:
        scratch = os.environ.get("PSCRATCH") or os.environ.get("SCRATCH")
        if scratch:
            os.environ["HF_HOME"] = f"{scratch}/hf_home"
        else:
            os.environ["HF_HOME"] = f"/tmp/hf_home_{os.environ.get('USER', 'default')}"

# Reduce CUDA allocator fragmentation - directly targets the
# CUBLAS_STATUS_EXECUTION_FAILED (OOM-in-disguise) seen in job 52464118.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge `overlay` into a copy of `base` (overlay wins)."""
    out = deepcopy(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path: str, mode: str | None) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    log.info(f"Loaded config: {path}")

    if "modes" in raw:
        modes = raw.pop("modes")
        if mode is None:
            raise SystemExit("--mode is required with a unified config "
                             f"(available: {list(modes)})")
        if mode not in modes:
            raise SystemExit(f"Unknown mode {mode!r}; available: {list(modes)}")
        cfg = _deep_merge(raw, modes[mode])
        cfg.setdefault("experiment", {})["mode"] = mode
        log.info(f"Applied mode overlay: {mode}")
        return cfg
    return raw  # legacy single-experiment config still works


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override (expected key=value): {override!r}")
        key_path, raw_value = override.split("=", 1)
        keys = key_path.strip().split(".")
        value = yaml.safe_load(raw_value)
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
        log.info(f"Override: {key_path} = {value!r}")
    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_scale_memory(cfg: dict, world_size: int):
    """Adapt memory-relevant settings to the visible GPU.

    Aurora at 0.25deg (720x1440) realistically caps per-GPU batch at 1, so
    "batch scaling" is done through gradient accumulation to reach
    training.target_effective_batch. On < 70 GB cards (or whenever rollout
    is enabled) gradient checkpointing is forced ON - enabling it mid-run on
    a live DDP graph, as the old code did at k>=3, is unsafe and was a
    contributor to the cuBLAS crash.
    """
    tcfg = cfg.setdefault("training", {})
    mcfg = cfg.setdefault("model", {})
    bs = cfg.get("data", {}).get("batch_size", 1)

    target_eff = int(tcfg.get("target_effective_batch", 0) or 0)
    if target_eff:
        accum = max(1, target_eff // max(1, world_size * bs))
        if accum != tcfg.get("grad_accum_steps", 1):
            log.info(f"[auto] grad_accum_steps={accum} "
                     f"(target_eff={target_eff}, world={world_size}, bs={bs})")
        tcfg["grad_accum_steps"] = accum

    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"[auto] GPU memory: {total_gb:.0f} GB")

    # v3: checkpointing is only REQUIRED for full-BPTT rollout (the whole
    # k-step chain lives in one graph).  Detached rollout backwards each
    # step immediately, so its activation memory equals single-step training
    # and checkpointing stays OFF - which also sidesteps the unresolved
    # checkpointing illegal-memory-access (handoff §2).
    rcfg = tcfg.get("rollout", {})
    backprop = str(rcfg.get("backprop", "full")).lower()
    if rcfg.get("enabled", False) and rcfg.get("max_steps", 1) >= 2 \
            and backprop == "full" and not mcfg.get("gradient_checkpointing", False):
        log.warning("[auto] full-BPTT rollout - forcing gradient_checkpointing=true "
                    "(must be set BEFORE DDP wrap, never mid-run). NOTE: full-BPTT "
                    "+ checkpointing currently crashes with an illegal memory "
                    "access on Perlmutter; use rollout.backprop=detached unless "
                    "tools/repro_ima_matrix.py has found a working combination.")
        mcfg["gradient_checkpointing"] = True

    # Optional SDPA backend pin (IMA workaround knob): training.sdpa_backend:
    # "math" disables the flash and memory-efficient attention kernels, whose
    # backward is the prime suspect in the checkpointing crash.
    sdpa = str(tcfg.get("sdpa_backend", "default")).lower()
    if sdpa == "math" and torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        log.warning("[auto] SDPA pinned to MATH backend (flash/mem-efficient disabled).")
    return cfg


# ---------------------------------------------------------------------------
# Smoke-test patching (unchanged behaviour)
# ---------------------------------------------------------------------------

_AURORA_NATIVE_SURF_VARS = ("2t", "10u", "10v", "msl")


def _patch_config_for_smoke_test(cfg: dict) -> dict:
    log.warning("SMOKE-TEST MODE: overriding config for a single synthetic step.")
    cfg["training"]["epochs"] = 1
    cfg["training"]["grad_accum_steps"] = 1
    cfg["training"]["time_limit_hours"] = 0
    cfg["data"]["use_dummy"] = True
    cfg["logging"]["log_every_n_steps"] = 1
    cfg["logging"]["val_every_n_epochs"] = 1
    cfg["checkpointing"]["save_every_n_steps"] = 10 ** 9
    return cfg


def _install_smoke_test_loader(cfg: dict, device: torch.device):
    from aurora import Batch, Metadata
    import datetime

    H, W = 64, 128
    LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    SURF_KEYS = tuple(cfg["model"]["surface_variables"])
    ATMOS_KEYS = ("z", "u", "v", "t", "q")

    def _make_batch():
        init_time = datetime.datetime(2015, 1, 1, 6, 0, 0)
        meta = Metadata(
            lat=torch.linspace(90, -90, H),
            lon=torch.linspace(0, 360, W + 1)[:-1],
            time=(init_time,),
            atmos_levels=LEVELS,
            rollout_step=0,
        )
        in_batch = Batch(
            surf_vars={k: torch.randn(1, 2, H, W) for k in SURF_KEYS},
            atmos_vars={k: torch.randn(1, 2, len(LEVELS), H, W) for k in ATMOS_KEYS},
            static_vars={k: torch.zeros(H, W) for k in ("z", "lsm", "slt")},
            metadata=meta,
        )
        target = {**{k: torch.randn(1, H, W) for k in SURF_KEYS},
                  **{k: torch.randn(1, len(LEVELS), H, W) for k in ATMOS_KEYS}}
        return in_batch, target

    class _SyntheticLoader:
        def __iter__(self):
            yield _make_batch()

        def __len__(self):
            return 1

    return _SyntheticLoader(), _SyntheticLoader()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Aurora MJO fine-tuning driver",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--mode", default=None,
                   choices=["baseline", "physics_informed", "lora", "combined"],
                   help="Mode overlay to apply (required for unified configs).")
    p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                   help="Dot-notation config override; repeatable.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Single synthetic step, no real data.")
    p.add_argument("--resume", default="auto", metavar="auto|none|PATH",
                   help="'auto' = latest ckpt in this mode's save_dir; "
                        "'none' = fresh start; or an explicit path.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # 1. DDP setup (no-op when launched without torchrun)
    # ------------------------------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_ddp = world_size > 1

    if use_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        log.info(f"DDP initialised: rank={dist.get_rank()}/{world_size}, "
                 f"local_rank={local_rank}")
    is_main = (not use_ddp) or (dist.get_rank() == 0)

    # A100 free performance: TF32 matmuls for any residual fp32 ops.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    # 2. Config
    # ------------------------------------------------------------------
    cfg = load_config(args.config, args.mode)
    if args.override:
        cfg = apply_overrides(cfg, args.override)
    if args.smoke_test:
        cfg = _patch_config_for_smoke_test(cfg)
    cfg = auto_scale_memory(cfg, world_size)

    seed_everything(cfg.get("experiment", {}).get("seed", 42) + local_rank)

    # ------------------------------------------------------------------
    # 3. Device
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 4. Model - rank 0 downloads the pretrained checkpoint first so the
    #    other 3 ranks hit the (persistent) HF cache instead of racing the
    #    network. Classic double-barrier pattern.
    # ------------------------------------------------------------------
    from src.model import load_model

    model_cfg = cfg.get("model", {})
    norm_stats = model_cfg.get("norm_stats") or None

    if use_ddp and local_rank != 0:
        dist.barrier()                       # wait for rank 0's download
    model = load_model(model_cfg, norm_stats=norm_stats)
    if use_ddp and local_rank == 0:
        dist.barrier()                       # release the other ranks
    if use_ddp:
        dist.barrier()                       # everyone constructed

    # ------------------------------------------------------------------
    # 5. Resolve resume/warm-start BEFORE building trainer state.
    #    Priority: explicit path > auto-latest in save_dir > init_from.
    # ------------------------------------------------------------------
    from src.checkpoint import CheckpointManager

    resume_path, warm_start_path = None, None
    save_dir = cfg.get("checkpointing", {}).get("save_dir", "checkpoints/run")

    if args.resume not in ("none",):
        if args.resume == "auto":
            resume_path = CheckpointManager.find_latest(save_dir)
        else:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                log.error(f"Checkpoint not found: {resume_path}")
                sys.exit(1)

    if resume_path is None:
        # No checkpoint for THIS phase yet -> warm-start from a previous
        # phase's best weights if the config asks for it, e.g.
        #   experiment.init_from: "latest:checkpoints/physics_informed"
        init_from = cfg.get("experiment", {}).get("init_from")
        if init_from:
            if str(init_from).startswith("latest:"):
                warm_start_path = CheckpointManager.find_latest(str(init_from)[7:])
            else:
                warm_start_path = Path(init_from)
            if warm_start_path is None or not Path(warm_start_path).exists():
                log.warning(f"init_from={init_from!r} not found - training from "
                            "pretrained Aurora weights only.")
                warm_start_path = None

    # ------------------------------------------------------------------
    # 6. DDP wrap. find_unused_parameters only when the MJO head is on
    #    (its output may be dropped from the loss).
    # ------------------------------------------------------------------
    model = model.to(device)
    if use_ddp:
        has_mjo_head = model_cfg.get("mjo_head", {}).get("enabled", False)
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=has_mjo_head,
                    gradient_as_bucket_view=True)
        log.info("Model wrapped in DistributedDataParallel")

    # ------------------------------------------------------------------
    # 7. Trainer + state restore
    # ------------------------------------------------------------------
    from src.trainer import Trainer

    if args.smoke_test:
        train_loader, val_loader = _install_smoke_test_loader(cfg, device)
    else:
        train_loader, val_loader = None, None

    trainer = Trainer(model=model, cfg=cfg, device=device,
                      train_loader=train_loader, val_loader=val_loader,
                      is_main=is_main)

    if resume_path is not None:
        log.info(f"Resuming full training state from: {resume_path}")
        trainer.load_checkpoint(resume_path, weights_only=False)
    elif warm_start_path is not None:
        log.info(f"Warm-starting model weights from: {warm_start_path}")
        trainer.load_checkpoint(warm_start_path, weights_only=True)

    # ------------------------------------------------------------------
    # 8. Train
    # ------------------------------------------------------------------
    log.info(
        f"Starting training: experiment={cfg.get('experiment', {}).get('name', '?')} "
        f"mode={cfg.get('experiment', {}).get('mode', '?')} "
        f"epochs={trainer.epochs} device={device} world_size={world_size} "
        f"start_epoch={trainer.start_epoch} global_step={trainer.global_step}"
    )
    exit_code = trainer.fit()

    if use_ddp:
        dist.destroy_process_group()
    log.info(f"Done (exit code {exit_code}).")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
