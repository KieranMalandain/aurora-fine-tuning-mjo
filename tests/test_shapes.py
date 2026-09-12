"""tests/test_shapes.py — Converted from scripts/verify_shapes.py.

Verifies:
  1. Static variable tensors are strictly 2D (H, W) as required by Aurora.
  2. Input and target tensor ranks and axis ordering across all surface and atmospheric fields.
  3. Static variables are upsampled to (720, 1440) even from reduced synthetic grids.
  4. Real-data absolute shape assertions against CFS archive (marked needs_data).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aurora_mjo.dataset import LANLMJODataset

EXPECTED_SURF_VARS = {"2t", "10u", "10v", "msl", "ttr", "tcwv"}
EXPECTED_ATMOS_VARS = {"z", "q", "t", "u", "v"}
EXPECTED_STATIC_VARS = {"z", "lsm", "slt"}

CFS_ROOT = Path(
    "/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results"
)
DEFAULT_SLT_PATH = Path("/pscratch/sd/k/kam352/Aurora/slt/slt_data.nc")


def test_static_vars_are_2d_and_upsampled_to_aurora_grid(
    synthetic_dataset: LANLMJODataset,
) -> None:
    """Verify that all static_vars are strictly 2D (H, W) with shape (720, 1440)."""
    for name, tensor in synthetic_dataset.static_vars.items():
        assert tensor.ndim == 2, f"Static var '{name}' has ndim={tensor.ndim}, expected 2"
        # Invariant z and lsm are upsampled to 0.25deg (720, 1440), slt truncated to (720, 1440).
        assert tensor.shape == (
            720,
            1440,
        ), f"Static var '{name}' has shape={tensor.shape}, expected (720, 1440)"


def test_input_and_target_tensor_ranks_and_axis_order(
    synthetic_dataset: LANLMJODataset,
) -> None:
    """Verify ranks, axis ordering, and batch dimensions on synthetic dataset sample 0."""
    in_batch, surf_targets_list, atmos_targets_list = synthetic_dataset[0]
    surf_out = surf_targets_list[0]
    atmos_out = atmos_targets_list[0]

    # Check input static variables in batch
    for name, tensor in in_batch.static_vars.items():
        assert (
            tensor.ndim == 2
        ), f"in_batch.static_vars['{name}'] has ndim={tensor.ndim}, expected 2"
        assert tensor.shape == (
            720,
            1440,
        ), f"in_batch.static_vars['{name}'] has shape={tensor.shape}, expected (720, 1440)"

    # Check input surface variables: rank 4 (B, T, H, W)
    for name, tensor in in_batch.surf_vars.items():
        assert (
            tensor.ndim == 4
        ), f"Surface input '{name}' has rank {tensor.ndim}, expected 4 (B, T, H, W)"
        assert tensor.shape[0] == 1, "Batch dimension must be 1"
        assert tensor.shape[1] == 2, "History timestep dimension must be 2"

    # Check input atmospheric variables: rank 5 (B, T, C, H, W)
    for name, tensor in in_batch.atmos_vars.items():
        assert (
            tensor.ndim == 5
        ), f"Atmos input '{name}' has rank {tensor.ndim}, expected 5 (B, T, C, H, W)"
        assert tensor.shape[0] == 1, "Batch dimension must be 1"
        assert tensor.shape[1] == 2, "History timestep dimension must be 2"
        assert tensor.shape[2] == 13, "Atmospheric pressure level dimension must be 13"

    # Check target surface variables: rank 3 (B, H, W) — no time axis
    for name, tensor in surf_out.items():
        assert (
            tensor.ndim == 3
        ), f"Surface target '{name}' has rank {tensor.ndim}, expected 3 (B, H, W)"
        assert tensor.shape[0] == 1, "Batch dimension must be 1"

    # Check target atmospheric variables: rank 4 (B, C, H, W) — no time axis
    for name, tensor in atmos_out.items():
        assert (
            tensor.ndim == 4
        ), f"Atmos target '{name}' has rank {tensor.ndim}, expected 4 (B, C, H, W)"
        assert tensor.shape[0] == 1, "Batch dimension must be 1"
        assert tensor.shape[1] == 13, "Atmospheric pressure level dimension must be 13"

    # Check metadata coordinates
    assert in_batch.metadata.lat.shape == torch.Size([720])
    assert in_batch.metadata.lon.shape == torch.Size([1440])
    assert len(in_batch.metadata.atmos_levels) == 13


@pytest.mark.needs_data
def test_real_cfs_data_absolute_shapes() -> None:
    """Verify absolute native grid (180, 360) and upsampled shapes against CFS data."""
    if not CFS_ROOT.exists():
        pytest.skip(f"CFS archive not mounted at {CFS_ROOT}")

    slt_path = DEFAULT_SLT_PATH if DEFAULT_SLT_PATH.exists() else None
    ds = LANLMJODataset(
        start_year=2016,
        end_year=2016,
        root_dir=CFS_ROOT,
        slt_path=slt_path,
        max_rollout_steps=1,
    )

    in_batch, surf_targets_list, atmos_targets_list = ds[0]
    surf_out = surf_targets_list[0]
    atmos_out = atmos_targets_list[0]

    # Native ERA5 grid in raw files is (180, 360)
    for name, tensor in in_batch.surf_vars.items():
        assert tensor.shape == (1, 2, 180, 360), f"Surface input '{name}' shape {tensor.shape}"

    for name, tensor in in_batch.atmos_vars.items():
        assert tensor.shape == (1, 2, 13, 180, 360), f"Atmos input '{name}' shape {tensor.shape}"

    for name, tensor in surf_out.items():
        assert tensor.shape == (1, 180, 360), f"Surface target '{name}' shape {tensor.shape}"

    for name, tensor in atmos_out.items():
        assert tensor.shape == (1, 13, 180, 360), f"Atmos target '{name}' shape {tensor.shape}"

    # Statics are upsampled to (720, 1440)
    for name, tensor in ds.static_vars.items():
        assert tensor.shape == (720, 1440), f"Static var '{name}' shape {tensor.shape}"

    assert in_batch.metadata.lat.shape == torch.Size([720])
    assert in_batch.metadata.lon.shape == torch.Size([1440])
