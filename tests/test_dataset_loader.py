"""tests/test_dataset_loader.py — Converted from scripts/verify_dataset_loader.py.

Verifies:
  1. Offline construction of LANLMJODataset for year 1980 yields exactly 1,462 samples.
  2. Static variables ("z", "lsm", "slt") are 2D tensors of shape (18, 36) and finite.
  3. Sample 0 returns a (Batch, dict, dict) triple.
  4. metadata.atmos_levels is the exact 13-tuple (50, 100, ..., 1000) hPa.
  5. Surface input variables carry a 2-timestep axis while targets do not.
  6. Atmospheric input variables carry a 2-timestep axis and 13 levels while targets do not.
  7. Real-data check against CFS archive (marked needs_data).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from aurora import Batch

from aurora_mjo.dataset import LANLMJODataset

EXPECTED_ATMOS_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
EXPECTED_SURF_VARS = {"2t", "10u", "10v", "msl", "ttr", "tcwv"}
EXPECTED_SURF_TARGET_VARS = EXPECTED_SURF_VARS | {"tp6h", "mslhf"}
EXPECTED_ATMOS_VARS = {"z", "q", "t", "u", "v"}
EXPECTED_STATIC_VARS = {"z", "lsm", "slt"}

CFS_ROOT = Path(
    "/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results"
)
DEFAULT_SLT_PATH = Path("data/static/slt_1deg.nc")


def test_dataset_offline_sample_count_and_length(synthetic_dataset: LANLMJODataset) -> None:
    """Verify offline LANLMJODataset reports exactly 1,462 samples for 1980 (1,464 - 2)."""
    # 03_DOMAIN_PRIORS.md §1: 1980 is a leap year (366 days * 4 = 1,464 steps).
    # With max_rollout_steps=1 (k=1), samples = timesteps - 2 = 1,462.
    assert len(synthetic_dataset) == 1462


def test_dataset_static_variables(synthetic_dataset: LANLMJODataset) -> None:
    """Verify presence, 2D rank, shape (18, 36), and finiteness of static variables."""
    static = synthetic_dataset.static_vars
    assert set(static.keys()) == EXPECTED_STATIC_VARS

    for name, tensor in static.items():
        assert isinstance(tensor, torch.Tensor), f"Static var '{name}' is not a Tensor"
        assert tensor.ndim == 2, f"Static var '{name}' ndim is {tensor.ndim}, expected 2"
        # 03_DOMAIN_PRIORS.md §2, §3: All static vars load at native resolution (18, 36)
        # without interpolation or truncation.
        assert tensor.shape == (
            18,
            36,
        ), f"Static var '{name}' shape is {tensor.shape}, expected (18, 36)"
        assert torch.isfinite(tensor).all(), f"Static var '{name}' contains non-finite values"

    # Check plausible numerical ranges
    assert static["z"].mean().item() > 0.0, "Surface geopotential mean should be positive"
    assert 0.0 <= static["lsm"].mean().item() <= 1.0, "Land-sea mask mean should be in [0, 1]"
    assert 0.0 <= static["slt"].mean().item() <= 7.0, "Soil type mean should be in [0, 7]"


def test_dataset_sample_zero_triple_and_metadata(synthetic_dataset: LANLMJODataset) -> None:
    """Verify sample 0 returns (Batch, list[dict], list[dict]) and metadata.atmos_levels 13-tuple.

    Note: LANLMJODataset __getitem__ returns targets as lists of dicts (one dict per rollout
    step), supporting multi-step autoregressive rollout. For max_rollout_steps=1, each list
    contains exactly one dictionary of target tensors.
    """
    in_batch, surf_targets_list, atmos_targets_list = synthetic_dataset[0]

    assert isinstance(in_batch, Batch), f"Expected in_batch to be Batch, got {type(in_batch)}"
    assert isinstance(surf_targets_list, list), "surf_targets_list must be list"
    assert isinstance(atmos_targets_list, list), "atmos_targets_list must be list"
    assert len(surf_targets_list) == 1, "Expected 1 target step for rollout_steps=1"
    assert len(atmos_targets_list) == 1, "Expected 1 target step for rollout_steps=1"

    surf_out = surf_targets_list[0]
    atmos_out = atmos_targets_list[0]
    assert isinstance(surf_out, dict), "surf_out step 0 must be dict"
    assert isinstance(atmos_out, dict), "atmos_out step 0 must be dict"

    # 03_DOMAIN_PRIORS.md §2: Exact 13 Aurora pressure levels probed independently of grid
    assert in_batch.metadata.atmos_levels == EXPECTED_ATMOS_LEVELS

    # First sample initialization time in 1980 (index 0 needs t-6h, so init is 06:00:00)
    init_time_str = str(in_batch.metadata.time[0])
    assert "1980-01-01 06:00:00" in init_time_str

    assert torch.isfinite(in_batch.metadata.lat).all()
    assert torch.isfinite(in_batch.metadata.lon).all()
    assert len(in_batch.metadata.lat) == 18
    assert len(in_batch.metadata.lon) == 36
    # 03_DOMAIN_PRIORS.md §2.1: Latitudes must be strictly decreasing for Aurora
    assert (in_batch.metadata.lat[1:] - in_batch.metadata.lat[:-1] < 0).all()
    assert (in_batch.metadata.lon[1:] - in_batch.metadata.lon[:-1] > 0).all()


def test_dataset_input_and_target_timestep_axes(synthetic_dataset: LANLMJODataset) -> None:
    """Verify surface inputs carry a 2-timestep axis and targets do not."""
    in_batch, surf_targets_list, atmos_targets_list = synthetic_dataset[0]
    surf_out = surf_targets_list[0]
    atmos_out = atmos_targets_list[0]

    assert set(in_batch.surf_vars.keys()) == EXPECTED_SURF_VARS
    assert set(surf_out.keys()) == EXPECTED_SURF_TARGET_VARS

    for k, v in in_batch.surf_vars.items():
        assert v.ndim == 4, f"Surface input '{k}' expected 4D (B, T, H, W), got {v.ndim}D"
        assert v.shape[0] == 1, f"Surface input '{k}' batch dimension should be 1"
        assert v.shape[1] == 2, f"Surface input '{k}' must have 2-timestep history axis"
        assert torch.isfinite(v).all(), f"Surface input '{k}' contains non-finite values"

    for k, v in surf_out.items():
        assert v.ndim == 3, f"Surface target '{k}' expected 3D (B, H, W) without time axis"
        assert v.shape[0] == 1, f"Surface target '{k}' batch dimension should be 1"
        assert torch.isfinite(v).all(), f"Surface target '{k}' contains non-finite values"

    assert set(in_batch.atmos_vars.keys()) == EXPECTED_ATMOS_VARS
    assert set(atmos_out.keys()) == EXPECTED_ATMOS_VARS

    for k, v in in_batch.atmos_vars.items():
        assert v.ndim == 5, f"Atmos input '{k}' expected 5D (B, T, C, H, W), got {v.ndim}D"
        assert v.shape[0] == 1, f"Atmos input '{k}' batch dimension should be 1"
        assert v.shape[1] == 2, f"Atmos input '{k}' must have 2-timestep history axis"
        assert v.shape[2] == 13, f"Atmos input '{k}' must have 13 pressure levels"
        assert torch.isfinite(v).all(), f"Atmos input '{k}' contains non-finite values"

    for k, v in atmos_out.items():
        assert v.ndim == 4, f"Atmos target '{k}' expected 4D (B, C, H, W) without time axis"
        assert v.shape[0] == 1, f"Atmos target '{k}' batch dimension should be 1"
        assert v.shape[1] == 13, f"Atmos target '{k}' must have 13 pressure levels"
        assert torch.isfinite(v).all(), f"Atmos target '{k}' contains non-finite values"


@pytest.mark.needs_data
def test_real_dataset_loader_priors() -> None:
    """Verify real CFS dataset against 03_DOMAIN_PRIORS.md when CFS is readable."""
    if not CFS_ROOT.exists():
        pytest.skip(f"CFS archive not mounted at {CFS_ROOT}")

    slt_path = DEFAULT_SLT_PATH if DEFAULT_SLT_PATH.exists() else None
    ds = LANLMJODataset(
        start_year=1980,
        end_year=1980,
        root_dir=CFS_ROOT,
        slt_path=slt_path,
        max_rollout_steps=1,
    )

    # 03_DOMAIN_PRIORS.md §1: MEASURED in docs/verify_output.txt
    assert len(ds) == 1462

    # Verify native static shapes
    assert ds.static_vars["z"].shape == (180, 360)
    assert ds.static_vars["lsm"].shape == (180, 360)
    if slt_path is not None:
        assert ds.static_vars["slt"].shape == (180, 360)

    # 03_DOMAIN_PRIORS.md §3: MEASURED static means
    z_mean = ds.static_vars["z"].mean().item()
    lsm_mean = ds.static_vars["lsm"].mean().item()
    assert z_mean == pytest.approx(3709.2466, rel=1e-3)
    assert lsm_mean == pytest.approx(0.3357, rel=1e-2)

    if slt_path is not None:
        slt_mean = ds.static_vars["slt"].mean().item()
        assert slt_mean == pytest.approx(0.6715, rel=1e-2)

    # Coordinate checks on real sample 0
    sample_0, _, _ = ds[0]
    assert len(sample_0.metadata.lat) == 180
    assert len(sample_0.metadata.lon) == 360
    assert (sample_0.metadata.lat[1:] - sample_0.metadata.lat[:-1] < 0).all()
    assert (sample_0.metadata.lon[1:] - sample_0.metadata.lon[:-1] > 0).all()
    assert sample_0.metadata.lat[0].item() == pytest.approx(89.5)
    assert sample_0.metadata.lat[-1].item() == pytest.approx(-89.5)
    assert sample_0.metadata.lon[0].item() == pytest.approx(0.5)
    assert sample_0.metadata.lon[-1].item() == pytest.approx(359.5)
