#!/usr/bin/env python3
"""
train.py — Aurora MJO fine-tuning entry point.

Usage:
  python train.py --config configs/phase1_baseline.yaml [--override key=value ...]

Smoke-test (no GPU, tiny dummy data):
  python train.py --config configs/phase1_baseline.yaml --smoke-test

Override examples:
  python train.py --config configs/phase1_baseline.yaml \
    --override training.epochs=2 \
    --override data.use_dummy=true
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a YAML config file and return a plain dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    log.info(f"Loaded config: {path}")
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Apply dot-notation key=value overrides to the config dict in-place.

    Example:
        apply_overrides(cfg, ["training.epochs=5", "data.use_dummy=false"])
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override (expected key=value): {override!r}")
        key_path, raw_value = override.split("=", 1)
        keys = key_path.strip().split(".")

        # Try to parse the value as YAML (handles int, float, bool, list, null)
        value = yaml.safe_load(raw_value)

        # Walk into the nested dict
        node = cfg
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
        log.info(f"Override: {key_path} = {value!r}")

    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Smoke-test patching
# ---------------------------------------------------------------------------

# Aurora's pretrained checkpoint only knows these four native surface variables.
# ttr and tcwv extend the embedding layer and require norm_stats -- both are
# intentionally excluded from the smoke-test to keep it self-contained.
_AURORA_NATIVE_SURF_VARS = ("2t", "10u", "10v", "msl")

def _patch_config_for_smoke_test(cfg: dict) -> dict:
    """
    Override config values for a fast smoke-test (no real data needed).
    Uses random tensors to mimic one batch.

    Key restriction: ttr and tcwv are stripped from model.surface_variables
    so that Aurora is loaded in its unmodified pretrained form (no embedding
    extension, no norm_stats required).
    """
    log.warning("SMOKE-TEST MODE: overriding config for a single synthetic step.")
    cfg["training"]["epochs"] = 1
    cfg["training"]["grad_accum_steps"] = 1
    cfg["data"]["use_dummy"] = True          # still set; loader is injected synthetically
    cfg["logging"]["log_every_n_steps"] = 1
    cfg["logging"]["val_every_n_epochs"] = 1
    cfg["checkpointing"]["save_every_n_epochs"] = 999  # don't save during smoke

    # Strip non-native vars so Aurora loads without embedding extension.
    original = cfg.get("model", {}).get("surface_variables", list(_AURORA_NATIVE_SURF_VARS))
    filtered = [v for v in original if v in _AURORA_NATIVE_SURF_VARS]
    cfg["model"]["surface_variables"] = filtered
    cfg["model"]["norm_stats"] = {}          # no norm injection needed
    log.warning(
        f"Smoke-test: surface_variables restricted to native Aurora vars: {filtered}"
    )
    return cfg


def _install_smoke_test_loader(cfg: dict, device: torch.device):
    """
    Replace train/val DataLoader with a tiny synthetic loader that produces
    one batch of random tensors shaped like dummy_dataset output.
    Avoids any file I/O.

    Surface variables are read from the (already-patched) config so they
    exactly match what the model was built with -- in particular ttr/tcwv
    are absent during smoke-tests.
    """
    from aurora import Batch, Metadata
    import datetime

    # Tiny spatial resolution for speed
    H, W = 8, 16
    LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    # Read surface vars from config (already stripped of ttr/tcwv by _patch_config)
    SURF_KEYS  = tuple(cfg["model"]["surface_variables"])
    ATMOS_KEYS = ("z", "u", "v", "t", "q")

    def _rand_surf():
        return {k: torch.randn(1, 1, H, W) for k in SURF_KEYS}

    def _rand_atmos():
        return {k: torch.randn(1, 1, len(LEVELS), H, W) for k in ATMOS_KEYS}

    def _rand_static():
        return {k: torch.zeros(H, W) for k in ("z", "lsm", "slt")}

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
            surf_vars=_rand_surf(),
            atmos_vars=_rand_atmos(),
            static_vars=_rand_static(),
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
    parser = argparse.ArgumentParser(
        description="Aurora MJO fine-tuning driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML experiment config file.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a config key using dot-notation. "
            "Can be specified multiple times, e.g. --override training.epochs=5"
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a single synthetic step to verify the pipeline compiles "
            "and runs end-to-end.  No real data is loaded."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help="Path to a checkpoint .pt file to resume training from.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Config
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)
    if args.smoke_test:
        cfg = _patch_config_for_smoke_test(cfg)

    seed_everything(cfg.get("experiment", {}).get("seed", 42))

    # ------------------------------------------------------------------
    # 2. Device
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    from src.model import load_model

    model_cfg = cfg.get("model", {})
    norm_stats = model_cfg.get("norm_stats") or None
    model = load_model(model_cfg, norm_stats=norm_stats)

    # ------------------------------------------------------------------
    # 4. Resume from checkpoint (optional)
    # ------------------------------------------------------------------
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            log.error(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(f"Resumed model weights from: {ckpt_path}")

    # ------------------------------------------------------------------
    # 5. Trainer
    # ------------------------------------------------------------------
    from src.trainer import Trainer

    # For smoke-test: build synthetic loaders BEFORE Trainer so that
    # __init__ never calls build_dataloader() and file-path checks are
    # never reached.
    if args.smoke_test:
        train_loader, val_loader = _install_smoke_test_loader(cfg, device)
    else:
        train_loader, val_loader = None, None

    trainer = Trainer(
        model=model,
        cfg=cfg,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Optional: resume optimizer state
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        if "optimizer_state_dict" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            log.info("Resumed optimizer state.")

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    log.info(
        f"Starting training: experiment={cfg.get('experiment',{}).get('name','?')} "
        f"epochs={trainer.epochs} device={device}"
    )
    trainer.fit()
    log.info("Done.")


if __name__ == "__main__":
    main()
