#!/usr/bin/env python3
"""Smoke test: validate 2-step autoregressive rollout without real data.

This test:
1. Builds a minimal configuration with rollout enabled and k=2.
2. Constructs a fake Aurora Batch with random tensors.
3. Instantiates a stub model that echoes the input surface fields.
4. Verifies that _compute_loss runs 2 forward passes and returns a finite
   scalar loss.

Run with:
    python scripts/smoke_test_rollout.py

No GPU, no data files, and no Aurora checkpoint required.
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Make src importable from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from src.trainer import Trainer, _advance_batch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimal config
# ---------------------------------------------------------------------------

CFG = {
    "experiment": {"name": "smoke_rollout", "seed": 0},
    "data": {"use_dummy": True, "batch_size": 1, "num_workers": 0, "pin_memory": False,
             "dummy": {"surface_files": [], "pressure_files": [], "static_file": ""}},
    "loss": {
        "grid":    {"enabled": True,  "tropics_bbox": [-20, 20],
                    "tropics_weight": 1.0, "extratropics_weight": 0.1},
        "spectral":     {"enabled": False, "weight": 0.0},
        "mjo_head":     {"enabled": False, "weight": 0.0},
        "moisture_budget": {"enabled": False, "weight": 0.0},
    },
    "training": {
        "epochs": 1,
        "grad_accum_steps": 1,
        "max_grad_norm": 1.0,
        "optimizer": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-5, "betas": [0.9, 0.999]},
        "scheduler": {"name": "none"},
        "rollout": {
            "enabled": True,
            "start_steps": 2,       # force k=2 from epoch 1
            "max_steps": 2,
            "step_increase_every_n_epochs": 1,
            "step_loss_weighting": "uniform",
        },
    },
    "checkpointing": {"save_dir": "/tmp/smoke_rollout_ckpt", "save_every_n_epochs": 1, "keep_last_n": 1},
    "logging": {"log_every_n_steps": 1, "val_every_n_epochs": 1},
}

# ---------------------------------------------------------------------------
# Stub Aurora Batch (no real aurora dependency needed for this test)
# ---------------------------------------------------------------------------

class FakeMetadata:
    def __init__(self):
        self.lat = torch.linspace(90, -90, 720)
        self.lon = torch.linspace(0, 359.75, 1440)
        self.time = (datetime(2000, 1, 1, 0), )   # batch size = 1
        self.atmos_levels = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
        self.rollout_step = 0

class FakeBatch:
    """Minimal stand-in for aurora.batch.Batch."""
    def __init__(self, surf_vars, static_vars, atmos_vars, metadata):
        self.surf_vars   = surf_vars
        self.static_vars = static_vars
        self.atmos_vars  = atmos_vars
        self.metadata    = metadata

def make_fake_batch(B=1, H=720, W=1440, t=2, C=13):
    """Create a random FakeBatch with Aurora-like shapes."""
    meta = FakeMetadata()
    surf = {
        "2t":   torch.randn(B, t, H, W),
        "10u":  torch.randn(B, t, H, W),
        "10v":  torch.randn(B, t, H, W),
        "msl":  torch.randn(B, t, H, W),
        "ttr":  torch.randn(B, t, H, W),
        "tcwv": torch.randn(B, t, H, W),
    }
    static = {"z": torch.randn(H, W)}
    atmos  = {
        "t":  torch.randn(B, t, C, H, W),
        "u":  torch.randn(B, t, C, H, W),
        "v":  torch.randn(B, t, C, H, W),
        "q":  torch.randn(B, t, C, H, W),
        "z":  torch.randn(B, t, C, H, W),
    }
    return FakeBatch(surf_vars=surf, static_vars=static, atmos_vars=atmos, metadata=meta)

# ---------------------------------------------------------------------------
# Stub model – returns a FakeBatch mirroring the input (no grad path)
# ---------------------------------------------------------------------------

class StubModel(nn.Module):
    """Returns a FakeBatch whose surf/atmos vars are derived from input (via linear)."""

    def __init__(self):
        super().__init__()
        # A trivial learnable parameter so the optimizer has something to step.
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        # Return a FakeBatch with shape=(B, 1, ...) as Aurora does (1-step pred).
        pred_surf = {k: v[:, -1:, ...] + 0.0 * self.dummy_param
                     for k, v in batch.surf_vars.items()}
        pred_atmos = {k: v[:, -1:, ...] + 0.0 * self.dummy_param
                      for k, v in batch.atmos_vars.items()}
        meta = FakeMetadata()
        meta.time = batch.metadata.time
        meta.rollout_step = batch.metadata.rollout_step
        return FakeBatch(surf_vars=pred_surf, static_vars=batch.static_vars,
                         atmos_vars=pred_atmos, metadata=meta)

# ---------------------------------------------------------------------------
# Monkey-patch _advance_batch to accept FakeBatch (avoids aurora import)
# ---------------------------------------------------------------------------

import src.trainer as _trainer_module

def _fake_advance_batch(in_batch, pred_batch, step_index):
    """FakeBatch-compatible version of _advance_batch."""
    dt = timedelta(hours=6)

    new_surf = {}
    for k in in_batch.surf_vars:
        history = in_batch.surf_vars[k]
        pred_k  = pred_batch.surf_vars.get(k)
        if pred_k is not None:
            new_surf[k] = torch.cat([history[:, 1:, ...], pred_k], dim=1)
        else:
            new_surf[k] = torch.cat([history[:, 1:, ...], history[:, -1:, ...]], dim=1)

    new_atmos = {}
    for k in in_batch.atmos_vars:
        history = in_batch.atmos_vars[k]
        pred_k  = pred_batch.atmos_vars.get(k)
        if pred_k is not None:
            new_atmos[k] = torch.cat([history[:, 1:, ...], pred_k], dim=1)
        else:
            new_atmos[k] = torch.cat([history[:, 1:, ...], history[:, -1:, ...]], dim=1)

    new_time = tuple(t + dt for t in in_batch.metadata.time)
    new_meta = FakeMetadata()
    new_meta.lat   = in_batch.metadata.lat
    new_meta.lon   = in_batch.metadata.lon
    new_meta.time  = new_time
    new_meta.atmos_levels = in_batch.metadata.atmos_levels
    new_meta.rollout_step = step_index + 1

    return FakeBatch(
        surf_vars=new_surf,
        static_vars=in_batch.static_vars,
        atmos_vars=new_atmos,
        metadata=new_meta,
    )

_trainer_module._advance_batch = _fake_advance_batch

# ---------------------------------------------------------------------------
# Build a minimal target dict (same keys as surf_vars, shape (B, 1, H, W))
# ---------------------------------------------------------------------------

def make_target_dict(surf_vars, B=1, H=720, W=1440):
    return {k: torch.randn(B, 1, H, W) for k in surf_vars}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    log.info("=== Rollout smoke test ===")

    device = torch.device("cpu")
    model  = StubModel()

    # Use synthetic loaders (empty DataLoader wrappers) so Trainer doesn't
    # need file I/O.  We call _compute_loss directly.
    from torch.utils.data import DataLoader, TensorDataset
    dummy_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

    trainer = Trainer(
        model=model,
        cfg=CFG,
        device=device,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
    )

    # Verify curriculum
    assert trainer._current_rollout_steps(1) == 2, "Expected k=2 at epoch 1"
    assert trainer._current_rollout_steps(10) == 2, "Expected k capped at 2"
    log.info("Curriculum check passed: k=2 at epoch 1 and 10.")

    # Verify step weights
    w2 = trainer._step_weights(2)
    assert abs(sum(w2) - 1.0) < 1e-6, f"Weights don't sum to 1: {w2}"
    log.info(f"Step weights (k=2, uniform): {w2}")

    # Build synthetic batch and targets
    log.info("Building synthetic batch (B=1, H=720, W=1440, t=2) …")
    in_batch    = make_fake_batch()
    target_dict = make_target_dict(in_batch.surf_vars)

    # Run _compute_loss with k=2
    log.info("Running _compute_loss with rollout k=2 …")
    losses = trainer._compute_loss(in_batch, target_dict, epoch=1)

    assert "total" in losses
    assert torch.isfinite(losses["total"]), f"Non-finite total loss: {losses['total']}"
    log.info(
        f"Loss breakdown: total={losses['total'].item():.6f}  "
        f"grid={losses['grid'].item():.6f}  "
        f"spectral={losses['spectral'].item():.6f}  "
        f"mjo_head={losses['mjo_head'].item():.6f}"
    )

    # Backward pass
    log.info("Verifying backward pass …")
    losses["total"].backward()
    assert model.dummy_param.grad is not None, "Gradient not flowing."
    log.info(f"Gradient on dummy_param: {model.dummy_param.grad.item():.6f}")

    log.info("=== Smoke test PASSED ===")

if __name__ == "__main__":
    main()
