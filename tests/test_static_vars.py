"""tests/test_static_vars.py — Regression tests for Lessons 3 & 4: Invariants and HDF5 locking.

Reference:
    00_CONTEXT.md §3 (Lessons 3 & 4)
    02_UPSTREAM_CONTRACT.md §4.3, §4.5
    03_DOMAIN_PRIORS.md §3

Background:
    Lesson 3: Reading NetCDF files on CFS without `HDF5_USE_FILE_LOCKING=FALSE` triggers
    intermittent `OSError: NetCDF: HDF error` due to Lustre/GPFS advisory file locking.
    Lesson 4: In `dataset.py::_load_static_vars`, invariant loading was wrapped in
    `except Exception` and substituted zero tensors, emitting only a `UserWarning`.
    If locking failed during earlier training runs, the model may have trained on a
    planet with zero topography and zero land-sea contrast without raising an error.

    Task E1 will make missing/unreadable static variables a hard failure. This test module
    documents and locks the CURRENT behaviour, including the zero-substitution fallback
    that E1 will invert, ensuring the test suite captures the transition explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import xarray as xr

from aurora_mjo.dataset import LANLMJODataset

REAL_CFS_ROOT = Path(
    "/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results"
)


def test_hdf5_file_locking_env_var_set() -> None:
    """Verify HDF5_USE_FILE_LOCKING is set to FALSE in the runtime environment.

    Lesson 3: HDF5 advisory locking against CFS parallel filesystems triggers
    OSError: NetCDF: HDF error. Setting HDF5_USE_FILE_LOCKING=FALSE is mandatory
    for all NetCDF reads. Currently set in conftest.py; E1 will enforce it at
    package import time in aurora_mjo.env.
    """
    val = os.environ.get("HDF5_USE_FILE_LOCKING")
    assert val == "FALSE", (
        f"HDF5_USE_FILE_LOCKING is '{val}', expected 'FALSE'. "
        "Parallel filesystem NetCDF reads will fail without this setting."
    )


def test_static_vars_load_correct_shapes_and_finite(
    synthetic_dataset: LANLMJODataset,
) -> None:
    """Verify static variables load with shape (720, 1440) and finite values."""
    statics = synthetic_dataset.static_vars
    expected_shape = (720, 1440)

    for name in ("z", "lsm", "slt"):
        assert name in statics, f"Missing static variable '{name}'"
        tensor = statics[name]
        assert tensor.shape == expected_shape, f"Static '{name}' shape != {expected_shape}"
        assert torch.isfinite(tensor).all(), f"Static '{name}' contains non-finite values"


def test_slt_truncate_not_upsample_asymmetry(synthetic_root: Path, synthetic_slt: Path) -> None:
    """Assert the structural asymmetry between z/lsm (upsampled) and slt (truncated).

    03_DOMAIN_PRIORS.md §3 notes:
      - `z` and `lsm` arrive from CFS at native 1° (18x36 in synthetic fixture,
        180x360 in real archive) and are bilinearly upsampled to (720, 1440) by
        `_upsample_to_aurora`.
      - `slt` arrives from an external non-archive file already at 0.25° (720, 1440)
        and is truncated with `[:720, :]` rather than interpolated.
    This test guards against well-meaning refactors that attempt to 'unify' the pipeline.
    """
    inv_dir = synthetic_root / "Step00/ERA5.invariant"
    z_file = next(inv_dir.glob("*_z.*.nc"))
    lsm_file = next(inv_dir.glob("*_lsm.*.nc"))

    # 1. Verify source shapes on disk
    with xr.open_dataset(str(z_file), engine="netcdf4") as ds_z:
        assert ds_z["Z"].shape[-2:] == (18, 36)
    with xr.open_dataset(str(lsm_file), engine="netcdf4") as ds_lsm:
        assert ds_lsm["LSM"].shape[-2:] == (18, 36)
    with xr.open_dataset(str(synthetic_slt), engine="netcdf4") as ds_slt:
        assert ds_slt["slt"].shape[-2:] == (720, 1440)

    # 2. Verify dataset instance loads both into identical (720, 1440) Aurora grids
    ds = LANLMJODataset(
        start_year=1980,
        end_year=1980,
        root_dir=synthetic_root,
        slt_path=synthetic_slt,
        max_rollout_steps=1,
    )
    assert ds.static_vars["z"].shape == (720, 1440)
    assert ds.static_vars["lsm"].shape == (720, 1440)
    assert ds.static_vars["slt"].shape == (720, 1440)


def test_clean_zeroes_nan_and_inf_across_all_three_statics(
    synthetic_dataset: LANLMJODataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify _clean sanitizes NaN and Inf to 0.0 across all three statics (v3 behavior).

    In v2 only `slt` was sanitized. v3 wraps all three statics in `_clean`.
    """
    real_open = xr.open_dataset

    def mock_open_with_nans(path: Any, *args: Any, **kwargs: Any) -> Any:
        ds = real_open(path, *args, **kwargs)
        path_str = str(path)
        if "_z." in path_str:
            arr = ds["Z"].values.copy()
            arr[0, 0] = np.nan
            arr[0, 1] = np.inf
            ds = ds.assign({"Z": (ds["Z"].dims, arr)})
        elif "_lsm." in path_str:
            arr = ds["LSM"].values.copy()
            arr[0, 0] = np.nan
            arr[0, 1] = -np.inf
            ds = ds.assign({"LSM": (ds["LSM"].dims, arr)})
        elif "slt" in path_str:
            arr = ds["slt"].values.copy()
            arr[0, 0] = np.nan
            arr[0, 1] = np.inf
            ds = ds.assign({"slt": (ds["slt"].dims, arr)})
        return ds

    monkeypatch.setattr(xr, "open_dataset", mock_open_with_nans)

    cleaned_statics = synthetic_dataset._load_static_vars()

    for var_name in ("z", "lsm", "slt"):
        tensor = cleaned_statics[var_name]
        assert not torch.isnan(tensor).any(), f"Static '{var_name}' retained NaNs after _clean!"
        assert not torch.isinf(tensor).any(), f"Static '{var_name}' retained Infs after _clean!"


def test_current_zero_fallback_on_unreadable_invariant(
    synthetic_dataset: LANLMJODataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert current fallback substitutes zero tensor and warns with 'Using zeros'.

    CURRENT BEHAVIOUR (WRONG / TEMPORARY):
    Lesson 4 warns that swallowing load exceptions and substituting zeros hides
    underlying I/O and file locking failures, causing silent physics degradation.
    Task E1 will turn this into a hard exception (raising FileNotFoundError or OSError).
    WHEN TASK E1 IMPLEMENTS THE FIX, THIS TEST MUST BE INVERTED TO ASSERT THAT AN
    EXCEPTION IS RAISED INSTEAD OF RETURNING ZEROS.
    """
    real_open = xr.open_dataset

    def mock_open_failing_z(path: Any, *args: Any, **kwargs: Any) -> Any:
        if "_z." in str(path):
            raise OSError("NetCDF: HDF error: advisory locking failed")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(xr, "open_dataset", mock_open_failing_z)

    with pytest.warns(UserWarning, match=r"Using zeros"):
        statics = synthetic_dataset._load_static_vars()

    # Current behaviour: z is substituted with all-zeros tensor
    z_tensor = statics["z"]
    assert z_tensor.shape == (720, 1440)
    assert (z_tensor == 0.0).all(), "Current fallback must substitute all zeros"


def test_synthetic_z_mean_far_from_zero(synthetic_dataset: LANLMJODataset) -> None:
    """Verify synthetic Z mean elevation is far from zero, distinguishing it from zero fallback."""
    mean_z = float(synthetic_dataset.static_vars["z"].mean())
    assert mean_z > 3000.0, (
        f"Synthetic Z mean is {mean_z:.2f} m^2/s^2; expected > 3000 m^2/s^2. "
        "A mean near 0 indicates the zero-substitution fallback was triggered!"
    )


@pytest.mark.needs_data
def test_real_cfs_z_mean_matches_prior() -> None:
    """Assert real CFS invariant Z mean matches ~3709.25 m^2/s^2 (03_DOMAIN_PRIORS.md §3).

    Requires CFS archive access on NERSC Perlmutter.
    If the July runs trained on zero invariants due to HDF5 locking failures,
    the mean would have been exactly 0.0. Real topography geopotential mean is ~3709 m^2/s^2.
    """
    inv_dir = REAL_CFS_ROOT / "Step00/ERA5.invariant"
    z_files = list(inv_dir.glob("*_z.*.nc"))
    if not z_files:
        pytest.fail(f"Real invariant Z file not found under {inv_dir}")

    with xr.open_dataset(str(z_files[0]), engine="netcdf4") as ds:
        z_mean = float(ds["Z"].mean())

    # Prior from 03_DOMAIN_PRIORS.md §3 is 3709.2466 m^2/s^2
    assert abs(z_mean - 3709.2466) < 50.0, f"Real Z mean {z_mean:.4f} != prior 3709.2466"
    assert z_mean > 3000.0, "Real Z mean must be far from zero"
