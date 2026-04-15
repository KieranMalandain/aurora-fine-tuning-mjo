#!/usr/bin/env python
"""Smoke test for AuroraMJO dual-head wrapper.

Run from repository root with:

    python scripts/smoke_test_mjo_head.py

The test uses AuroraSmallPretrained so no GPU is required, but you *will*
need network access to download the small pretrained checkpoint (~200 MB)
from HuggingFace on first run.  Subsequent runs use the HF cache.

Exit code 0 = pass.
"""

import sys
from datetime import datetime

import torch
from aurora import Batch, Metadata

# Make sure the local src/ is importable when running from repo root.
sys.path.insert(0, ".")
from src.model import load_model


# ---------------------------------------------------------------------------
# Tiny synthetic batch – matches AuroraSmallPretrained expectations.
# ---------------------------------------------------------------------------
# We use a very small spatial grid so the test is fast on CPU.
# Aurora's patch_size=4, so H and W must be multiples of 4.
H, W = 32, 64   # degrees: a stub, not real ERA5 resolution
ATMOS_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
SURF_VARS = ("2t", "10u", "10v", "msl")  # default Aurora surf_vars; ttr/tcwv injection is a training concern

# lat decreasing (90 ... -90 style), lon increasing (0...360)
lat = torch.linspace(90, -90, H)
lon = torch.linspace(0, 360 - 360 / W, W)

B, T = 1, 2   # batch=1, history=2

def _rand_surf():
    return {v: torch.randn(B, T, H, W) for v in SURF_VARS}  # (B, T, H, W)

def _rand_atmos():
    return {v: torch.randn(B, T, len(ATMOS_LEVELS), H, W)
            for v in ("z", "u", "v", "t", "q")}

def _rand_static():
    return {v: torch.randn(H, W) for v in ("lsm", "z", "slt")}

batch = Batch(
    surf_vars=_rand_surf(),
    static_vars=_rand_static(),
    atmos_vars=_rand_atmos(),
    metadata=Metadata(
        lat=lat,
        lon=lon,
        time=(datetime(2020, 1, 1, 0, 0),),
        atmos_levels=ATMOS_LEVELS,
    ),
)

# ---------------------------------------------------------------------------
# Shared config skeleton
# ---------------------------------------------------------------------------
BASE_CONFIG = dict(
    model_type="small",   # AuroraSmallPretrained
    surface_variables=list(SURF_VARS),
    use_lora=False,
    gradient_checkpointing=False,
)

# ---------------------------------------------------------------------------
# Test 1 – MJO head DISABLED: forward returns a plain Batch
# ---------------------------------------------------------------------------
print("=" * 60)
print("Test 1: head disabled")
cfg_no_head = {**BASE_CONFIG, "mjo_head": {"enabled": False}}
model_no_head = load_model(cfg_no_head)
model_no_head.eval()

with torch.no_grad():
    out = model_no_head(batch)

assert isinstance(out, Batch), f"Expected Batch, got {type(out)}"
print(f"  [PASS] returned Batch  (surf keys: {list(out.surf_vars.keys())})")

# ---------------------------------------------------------------------------
# Test 2 – MJO head ENABLED: forward returns (Batch, Tensor[B,3])
# ---------------------------------------------------------------------------
print("=" * 60)
print("Test 2: head enabled")
cfg_with_head = {
    **BASE_CONFIG,
    "mjo_head": {
        "enabled": True,
        "hidden_dim": 128,
        "dropout": 0.0,
        "lat_south": -15.0,
        "lat_north": 15.0,
    },
}
model_with_head = load_model(cfg_with_head)
model_with_head.eval()

with torch.no_grad():
    out_state, out_mjo = model_with_head(batch)

assert isinstance(out_state, Batch), f"Expected Batch, got {type(out_state)}"
assert isinstance(out_mjo, torch.Tensor), f"Expected Tensor, got {type(out_mjo)}"
assert out_mjo.shape == (B, 3), f"Expected ({B}, 3), got {out_mjo.shape}"
print(f"  [PASS] returned (Batch, Tensor{list(out_mjo.shape)})")
print(f"  MJO prediction: RMM1={out_mjo[0, 0]:.4f}  "
      f"RMM2={out_mjo[0, 1]:.4f}  Amp={out_mjo[0, 2]:.4f}")

# ---------------------------------------------------------------------------
# Test 3 – New MJO head params absent from backbone checkpoint → strict=False OK
# ---------------------------------------------------------------------------
print("=" * 60)
print("Test 3: MJO head params are NOT in the backbone state_dict")
backbone_sd_keys = set(model_with_head.backbone.state_dict().keys())
head_sd_keys = {f"mjo_head.{k}" for k in model_with_head.mjo_head.state_dict().keys()}
overlap = backbone_sd_keys & head_sd_keys
assert len(overlap) == 0, f"Unexpected key overlap: {overlap}"
print(f"  [PASS] zero key overlap ({len(head_sd_keys)} head params are new)")

print("=" * 60)
print("All smoke tests passed.")
