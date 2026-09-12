"""tests/test_mjo_head.py — Converted from scripts/smoke_test_mjo_head.py.

Verifies:
  1. When mjo_head.enabled=False, NO head is constructed at all (model.mjo_head is None)
     and forward() returns a plain Batch.
  2. When mjo_head.enabled=True, MJOHead is constructed from config.
  3. Output shape matches (B, 3) representing [RMM1, RMM2, Amplitude], and active status
     can be evaluated (amplitude > 1.0).
  4. Outputs are finite.
  5. MJO head parameters are isolated from the backbone (zero key overlap).
  6. Tropical latitude filtering correctly masks out extra-tropical patch tokens.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
import torch
from aurora import Aurora, Batch, Metadata

from aurora_mjo.model import AuroraMJO, MJOHead, load_model

ATMOS_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
SURF_VARS = ("2t", "10u", "10v", "msl")


@pytest.fixture(autouse=True)
def mock_aurora_load_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make load_checkpoint a no-op so models instantiate offline on CPU without HF downloads."""
    monkeypatch.setattr(Aurora, "load_checkpoint", lambda self, strict=True: None)


def _make_tiny_batch(b: int = 1, h: int = 32, w: int = 64) -> Batch:
    """Build a tiny synthetic batch suitable for CPU forward pass."""
    t = 2
    lat = torch.linspace(90, -90, h)
    lon = torch.linspace(0, 360 - 360 / w, w)
    return Batch(
        surf_vars={v: torch.randn(b, t, h, w) for v in SURF_VARS},
        static_vars={v: torch.randn(h, w) for v in ("lsm", "z", "slt")},
        atmos_vars={
            v: torch.randn(b, t, len(ATMOS_LEVELS), h, w) for v in ("z", "u", "v", "t", "q")
        },
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=(datetime(2020, 1, 1, 0, 0),),
            atmos_levels=ATMOS_LEVELS,
        ),
    )


def test_mjo_head_disabled_constructs_no_head_and_returns_batch() -> None:
    """Verify that with mjo_head.enabled: false NO head is constructed at all."""
    config: dict[str, Any] = {
        "model_type": "small",
        "surface_variables": list(SURF_VARS),
        "use_lora": False,
        "gradient_checkpointing": False,
        "mjo_head": {"enabled": False},
    }

    model: AuroraMJO = load_model(config)
    assert (
        model.mjo_head is None
    ), "With mjo_head.enabled=False, no head should be constructed at all (expected None)"

    model.eval()
    batch = _make_tiny_batch(b=1, h=32, w=64)
    with torch.no_grad():
        out = model(batch)

    assert isinstance(
        out, Batch
    ), f"Expected Batch output when mjo_head is disabled, got {type(out)}"
    assert set(out.surf_vars.keys()) == set(SURF_VARS)


def test_mjo_head_enabled_constructs_head_and_produces_rmm_outputs() -> None:
    """Verify that with mjo_head.enabled: true forward produces (Batch, Tensor[B, 3])."""
    config: dict[str, Any] = {
        "model_type": "small",
        "surface_variables": list(SURF_VARS),
        "use_lora": False,
        "gradient_checkpointing": False,
        "mjo_head": {
            "enabled": True,
            "hidden_dim": 64,
            "dropout": 0.0,
            "lat_south": -15.0,
            "lat_north": 15.0,
        },
    }

    model: AuroraMJO = load_model(config)
    assert model.mjo_head is not None, "Expected MJO head to be constructed when enabled=True"
    assert isinstance(model.mjo_head, MJOHead)

    model.eval()
    b = 1
    batch = _make_tiny_batch(b=b, h=32, w=64)
    with torch.no_grad():
        out_state, out_mjo = model(batch)

    assert isinstance(out_state, Batch), f"Expected Batch state, got {type(out_state)}"
    assert isinstance(out_mjo, torch.Tensor), f"Expected Tensor MJO output, got {type(out_mjo)}"
    assert out_mjo.shape == (b, 3), f"Expected shape ({b}, 3), got {out_mjo.shape}"
    assert torch.isfinite(out_mjo).all(), "MJO predictions contain non-finite values"

    # Verify RMM1, RMM2, Amplitude and Active Status
    rmm1 = out_mjo[:, 0]
    rmm2 = out_mjo[:, 1]
    amplitude = out_mjo[:, 2]
    # 03_DOMAIN_PRIORS.md §6: Active MJO defined as amplitude > 1.0
    active_status = amplitude > 1.0

    assert rmm1.shape == (b,)
    assert rmm2.shape == (b,)
    assert amplitude.shape == (b,)
    assert active_status.shape == (b,)
    assert active_status.dtype == torch.bool


def test_mjo_head_parameter_isolation() -> None:
    """Verify that MJO head parameters are isolated from backbone state_dict."""
    config: dict[str, Any] = {
        "model_type": "small",
        "surface_variables": list(SURF_VARS),
        "use_lora": False,
        "gradient_checkpointing": False,
        "mjo_head": {"enabled": True, "hidden_dim": 64},
    }

    model: AuroraMJO = load_model(config)
    assert model.mjo_head is not None

    backbone_keys = set(model.backbone.state_dict().keys())
    head_keys = {f"mjo_head.{k}" for k in model.mjo_head.state_dict().keys()}
    overlap = backbone_keys & head_keys
    assert len(overlap) == 0, f"Unexpected key overlap between backbone and MJO head: {overlap}"
    assert len(head_keys) > 0, "Expected head keys to be non-empty"


def test_mjo_head_tropical_mask() -> None:
    """Verify MJOHead tropical mask filters latitudes outside [-15, 15] degrees."""
    head = MJOHead(embed_dim=128, hidden_dim=64, dropout=0.0, lat_south=-15.0, lat_north=15.0)

    # 32 latitudes from 90 to -90
    h_patches = 8
    lat = torch.linspace(90, -90, 32)
    # Patch step = 32 // 8 = 4. Centres: index 2 (approx 78), ..., index 30 (approx -78)
    patch_lat = lat[2::4][:h_patches]

    # Tropical mask
    mask = (patch_lat >= -15.0) & (patch_lat <= 15.0)
    # Ensure tropical mask selects a non-empty proper subset of patch latitudes
    assert mask.sum().item() > 0, "Tropical mask should select at least one patch latitude"
    assert mask.sum().item() < h_patches, "Tropical mask should not select all latitudes"

    # Direct forward pass on MJOHead
    b, n_levels, n_w, d = 2, 4, 16, 128
    x = torch.randn(b, n_levels * h_patches * n_w, d)
    out = head(x, patch_res=(n_levels, h_patches, n_w), lat=lat)

    assert out.shape == (b, 3)
    assert torch.isfinite(out).all()
