"""Tests for the autoregressive rollout inference engine (aurora_mjo.inference).

Task J3 (tasks/J3_skill_harness.md):
- 120-step autoregressive rollout with history-window advance.
- Verification of output shapes and finiteness.
- Surface physical clamping during multi-step stepping.
- Extraction of targeted pressure levels (u850, u200).
- Data leakage assertion: no timestamp in a 120-day window later than t0.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
import torch
from aurora import Batch, Metadata

from aurora_mjo.inference import (
    advance_batch,
    extract_level_field,
    run_rollout_daily,
)


class DummyAuroraModel(torch.nn.Module):
    """Synthetic model mimicking Aurora forward pass on native 1° grid (180x360)."""

    def __init__(
        self,
        surf_vars: tuple[str, ...],
        atmos_vars: tuple[str, ...],
        levels: tuple[int | float, ...],
    ):
        super().__init__()
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.levels = levels

    def forward(self, batch: Batch) -> Batch:
        # Returns a predicted single-step batch (B, 1, ...)

        pred_surf = {}
        for k in self.surf_vars:
            # Predict slightly modified state (persistence + tiny variation)
            pred_surf[k] = batch.surf_vars[k][:, -1:] * 0.999 + 0.001

        pred_atmos = {}
        for k in self.atmos_vars:
            pred_atmos[k] = batch.atmos_vars[k][:, -1:] * 0.999 + 0.001

        dt = timedelta(hours=6)
        pred_meta = Metadata(
            lat=batch.metadata.lat,
            lon=batch.metadata.lon,
            time=tuple(t + dt for t in batch.metadata.time),
            atmos_levels=batch.metadata.atmos_levels,
            rollout_step=batch.metadata.rollout_step + 1,
        )
        return Batch(
            surf_vars=pred_surf,
            static_vars=batch.static_vars,
            atmos_vars=pred_atmos,
            metadata=pred_meta,
        )


def _make_dummy_batch(
    init_time: datetime = datetime(2016, 1, 1, 0, 0),
    H: int = 180,
    W: int = 360,
    levels: tuple[int, ...] = (50, 100, 200, 500, 850, 1000),
) -> Batch:
    lat = torch.linspace(89.5, -89.5, H)
    lon = torch.linspace(0.5, 359.5, W)

    surf_vars = {
        "2t": torch.full((1, 2, H, W), 280.0, dtype=torch.float32),
        "10u": torch.full((1, 2, H, W), 5.0, dtype=torch.float32),
        "10v": torch.full((1, 2, H, W), -3.0, dtype=torch.float32),
        "msl": torch.full((1, 2, H, W), 98000.0, dtype=torch.float32),
        "ttr": torch.full((1, 2, H, W), -230.0, dtype=torch.float32),
        "tcwv": torch.full((1, 2, H, W), 25.0, dtype=torch.float32),
    }
    atmos_vars = {
        "u": torch.full((1, 2, len(levels), H, W), 10.0, dtype=torch.float32),
        "v": torch.full((1, 2, len(levels), H, W), 2.0, dtype=torch.float32),
        "t": torch.full((1, 2, len(levels), H, W), 250.0, dtype=torch.float32),
        "q": torch.full((1, 2, len(levels), H, W), 0.005, dtype=torch.float32),
        "z": torch.full((1, 2, len(levels), H, W), 50000.0, dtype=torch.float32),
    }
    static_vars = {
        "lsm": torch.zeros((H, W), dtype=torch.float32),
        "z": torch.zeros((H, W), dtype=torch.float32),
        "slt": torch.zeros((H, W), dtype=torch.float32),
        "sst": torch.full((H, W), 295.0, dtype=torch.float32),
    }
    metadata = Metadata(
        lat=lat,
        lon=lon,
        time=(init_time,),
        atmos_levels=levels,
        rollout_step=0,
    )
    return Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=metadata,
    )


def test_advance_batch_rolls_history_and_clamps():
    """Verify advance_batch rolls [t-6h, t] -> [t, t+6h] and applies physical clamping."""
    batch = _make_dummy_batch()
    init_time = batch.metadata.time[0]

    # Create dummy prediction with an extreme out-of-bounds MSL value (1.5e5 Pa)
    pred_surf = {k: v[:, -1:].clone() for k, v in batch.surf_vars.items()}
    pred_surf["msl"] = torch.full_like(pred_surf["msl"], 1.5e5)
    pred_atmos = {k: v[:, -1:].clone() for k, v in batch.atmos_vars.items()}
    pred_meta = Metadata(
        lat=batch.metadata.lat,
        lon=batch.metadata.lon,
        time=(init_time + timedelta(hours=6),),
        atmos_levels=batch.metadata.atmos_levels,
        rollout_step=1,
    )
    pred_batch = Batch(
        surf_vars=pred_surf,
        static_vars=batch.static_vars,
        atmos_vars=pred_atmos,
        metadata=pred_meta,
    )

    next_batch = advance_batch(batch, pred_batch, step_index=1)

    # Time advanced by 6h
    assert next_batch.metadata.time[0] == init_time + timedelta(hours=6)
    assert next_batch.metadata.rollout_step == 2

    # History dimension is 2
    assert next_batch.surf_vars["2t"].shape == (1, 2, 180, 360)
    assert next_batch.atmos_vars["u"].shape == (1, 2, 6, 180, 360)

    # Clamping enforced: MSL capped at 1.1e5 Pa
    assert float(next_batch.surf_vars["msl"][:, -1].max()) <= 1.1e5 + 1e-4


def test_extract_level_field():
    """Verify pressure level extraction for 850 and 200 hPa."""
    levels = (50, 100, 200, 500, 850, 1000)
    atmos_tensor = torch.zeros((1, 1, len(levels), 180, 360), dtype=torch.float32)
    # Set unique values at 850 and 200 hPa
    idx_850 = levels.index(850)
    idx_200 = levels.index(200)
    atmos_tensor[0, 0, idx_850] = 850.5
    atmos_tensor[0, 0, idx_200] = 200.5

    f850 = extract_level_field(atmos_tensor, levels, 850)
    f200 = extract_level_field(atmos_tensor, levels, 200)

    assert f850.shape == (180, 360)
    assert f200.shape == (180, 360)
    assert np.allclose(f850, 850.5)
    assert np.allclose(f200, 200.5)


def test_rollout_120_steps_completes_finite():
    """Headline test: rollout 120 steps completes with exact shapes and all finite values."""
    batch = _make_dummy_batch()
    levels = batch.metadata.atmos_levels
    model = DummyAuroraModel(
        surf_vars=tuple(batch.surf_vars.keys()),
        atmos_vars=tuple(batch.atmos_vars.keys()),
        levels=levels,
    )

    # Run full 120-step daily aggregated rollout (30 days)
    res = run_rollout_daily(model, batch, max_lead_days=30)

    assert len(res["lead_days"]) == 30
    assert res["lead_days"] == list(range(1, 31))
    assert len(res["valid_dates"]) == 30

    # Shapes: (30, 180, 360)
    for var_name in ("ttr", "tcwv", "2t", "msl"):
        arr = res["surf_daily"][var_name]
        assert arr.shape == (30, 180, 360), f"Shape mismatch for {var_name}: {arr.shape}"
        assert np.isfinite(arr).all(), f"Non-finite values found in {var_name}"

    assert res["u850_daily"].shape == (30, 180, 360)
    assert np.isfinite(res["u850_daily"]).all()

    assert res["u200_daily"].shape == (30, 180, 360)
    assert np.isfinite(res["u200_daily"]).all()


def test_leakage_assertion_fires_on_future_timestamp():
    """Suematsu trap regression test: 120-day mean must contain NO timestamp after t0.

    02_SCIENTIFIC_CONTRACT.md §6.2: Using any data after the forecast valid time
    or after t0 in the 120-day mean calculation is data leakage.
    Assert that the validation check rejects any window containing future timestamps.
    """
    from aurora_mjo.rmm.evaluate import assert_no_leakage_120d_window

    t0 = datetime(2016, 6, 1, 0, 0)

    # 1. Strictly causal window [t0 - 120d, t0]: MUST PASS
    causal_times = [t0 - timedelta(days=d) for d in range(120, -1, -1)]
    assert_no_leakage_120d_window(causal_times, t0)  # should not raise

    # 2. Leaked window containing a timestamp t0 + 1 day: MUST RAISE AssertionError
    leaked_times = causal_times + [t0 + timedelta(days=1)]
    with pytest.raises(AssertionError, match="Data leakage detected"):
        assert_no_leakage_120d_window(leaked_times, t0)
