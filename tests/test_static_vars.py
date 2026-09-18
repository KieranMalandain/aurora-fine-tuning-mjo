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

    Task E1 makes missing/unreadable static variables a hard failure (StaticVarLoadError).
    Earlier versions caught Exception and substituted all-zeros with a UserWarning, risking
    silent physics degradation (a planet with no topography and no continents, emitting only
    two warnings in an 11-hour log). docs/findings/2026-09-zeroed-statics.md (task B2)
    records whether that fallback ever fired in production.
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
from aurora_mjo.env import StaticVarLoadError

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
    """Verify static variables load with shape (18, 36) and finite values."""
    statics = synthetic_dataset.static_vars
    expected_shape = (18, 36)

    for name in ("z", "lsm", "slt"):
        assert name in statics, f"Missing static variable '{name}'"
        tensor = statics[name]
        assert tensor.shape == expected_shape, f"Static '{name}' shape != {expected_shape}"
        assert torch.isfinite(tensor).all(), f"Static '{name}' contains non-finite values"


def test_static_vars_native_resolution_symmetry(synthetic_root: Path, synthetic_slt: Path) -> None:
    """Assert all three static variables share native resolution without upsampling or truncation.

    Under native resolution ingestion (G1):
      - `z` and `lsm` arrive at native resolution (18, 36 in synthetic fixture)
        and are loaded directly without interpolation.
      - `slt` arrives pre-regridded to native resolution (18, 36 in synthetic fixture)
        and is loaded directly without truncation.
      - All three static variables match in spatial dimensions (H, W).
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
        assert ds_slt["slt"].shape[-2:] == (18, 36)

    # 2. Verify dataset instance loads all three into identical native grids
    ds = LANLMJODataset(
        start_year=1980,
        end_year=1980,
        root_dir=synthetic_root,
        slt_path=synthetic_slt,
        max_rollout_steps=1,
    )
    assert ds.static_vars["z"].shape == (18, 36)
    assert ds.static_vars["lsm"].shape == (18, 36)
    assert ds.static_vars["slt"].shape == (18, 36)


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


def test_hard_fail_on_unreadable_invariant_z(
    synthetic_dataset: LANLMJODataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert unreadable invariant Z raises StaticVarLoadError rather than substituting zeros.

    History (Lesson 4):
    Earlier versions caught Exception and substituted an all-zeros tensor with a UserWarning.
    If locking failed during earlier training runs, the model may have trained on a planet
    with zero topography while emitting only two warnings in an 11-hour log.
    Task E1 inverts this fallback into a hard failure chaining the underlying error and
    recommending HDF5_USE_FILE_LOCKING=FALSE.
    """
    real_open = xr.open_dataset

    def mock_open_failing_z(path: Any, *args: Any, **kwargs: Any) -> Any:
        if "_z." in str(path):
            raise OSError("[Errno -101] NetCDF: HDF error: advisory locking failed")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(xr, "open_dataset", mock_open_failing_z)

    with pytest.raises(StaticVarLoadError) as exc_info:
        synthetic_dataset._load_static_vars()

    err_msg = str(exc_info.value)
    assert "invariant variable 'z'" in err_msg
    assert "_z." in err_msg
    assert "HDF5_USE_FILE_LOCKING=FALSE" in err_msg
    assert exc_info.value.__cause__ is not None
    assert "NetCDF: HDF error" in str(exc_info.value.__cause__)


def test_hard_fail_on_unreadable_invariant_lsm(
    synthetic_dataset: LANLMJODataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert unreadable invariant LSM raises StaticVarLoadError rather than substituting zeros."""
    real_open = xr.open_dataset

    def mock_open_failing_lsm(path: Any, *args: Any, **kwargs: Any) -> Any:
        if "_lsm." in str(path):
            raise OSError("[Errno -101] NetCDF: HDF error: advisory locking failed")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(xr, "open_dataset", mock_open_failing_lsm)

    with pytest.raises(StaticVarLoadError) as exc_info:
        synthetic_dataset._load_static_vars()

    err_msg = str(exc_info.value)
    assert "invariant variable 'lsm'" in err_msg
    assert "_lsm." in err_msg
    assert "HDF5_USE_FILE_LOCKING=FALSE" in err_msg
    assert exc_info.value.__cause__ is not None
    assert "NetCDF: HDF error" in str(exc_info.value.__cause__)


def test_hard_fail_on_unreadable_static_slt(
    synthetic_dataset: LANLMJODataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert unreadable static SLT raises StaticVarLoadError with contextual details."""
    real_open = xr.open_dataset

    def mock_open_failing_slt(path: Any, *args: Any, **kwargs: Any) -> Any:
        if "slt" in str(path):
            raise OSError("I/O failure reading soil type")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(xr, "open_dataset", mock_open_failing_slt)

    with pytest.raises(StaticVarLoadError) as exc_info:
        synthetic_dataset._load_static_vars()

    err_msg = str(exc_info.value)
    assert "static soil type 'slt'" in err_msg
    assert "slt" in err_msg
    assert "/pscratch" in err_msg
    assert "download_slt.py" in err_msg
    assert "HDF5_USE_FILE_LOCKING=FALSE" in err_msg
    assert exc_info.value.__cause__ is not None


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


@pytest.mark.needs_data
def test_real_cfs_static_vars_load_correctly() -> None:
    """Assert all three static variables load correctly from real CFS archive without error."""
    if not REAL_CFS_ROOT.exists():
        pytest.skip(f"Real CFS archive not mounted at {REAL_CFS_ROOT}")
    slt_path = Path("data/static/slt_1deg.nc")
    if not slt_path.exists():
        pytest.skip(f"Native SLT file not found at {slt_path}")
    ds = object.__new__(LANLMJODataset)
    ds.root_dir = REAL_CFS_ROOT
    ds.slt_path = slt_path
    statics = ds._load_static_vars()
    assert statics["z"].shape == (180, 360)
    assert statics["lsm"].shape == (180, 360)
    assert statics["slt"].shape == (180, 360)
    assert torch.isfinite(statics["z"]).all()
    assert torch.isfinite(statics["lsm"]).all()
    assert torch.isfinite(statics["slt"]).all()


def test_sst_loaded_into_in_batch_static_vars(
    synthetic_dataset: LANLMJODataset,
) -> None:
    """Verify that SST is loaded into in_batch.static_vars per sample at t0."""
    in_batch, _, _ = synthetic_dataset[0]
    statics = in_batch.static_vars
    assert "sst" in statics, "Missing 'sst' in in_batch.static_vars"
    sst_tensor = statics["sst"]
    assert isinstance(sst_tensor, torch.Tensor)
    assert sst_tensor.ndim == 2, f"Expected 2D SST tensor, got shape {sst_tensor.shape}"
    assert sst_tensor.shape == (18, 36), f"Expected (18, 36), got {sst_tensor.shape}"
    assert torch.isfinite(sst_tensor).all(), "SST tensor contains non-finite values"
    # Physical range check: [271, 310] K per 03_DOMAIN_PRIORS.md §3
    assert 271.0 <= sst_tensor.min().item(), f"SST min {sst_tensor.min().item()} < 271 K"
    assert sst_tensor.max().item() <= 310.0, f"SST max {sst_tensor.max().item()} > 310 K"


def test_sst_freeze_backbone_trainable() -> None:
    """Verify freeze_backbone keeps newly injected static SST embedding trainable.

    Verification method:
    Aurora's Perceiver3DEncoder concatenates static_vars with surf_vars and
    passes them through surf_token_embeds (LevelPatchEmbed). Newly added
    static variables (not in _AURORA_DEFAULT_STATIC_VARS) are randomly initialized
    and must have requires_grad=True, while default statics (lsm, z, slt) remain frozen.
    Static variables have no decoder heads (decoder.surf_heads only contains surf_vars).
    """
    from aurora import AuroraSmallPretrained

    from aurora_mjo.model import freeze_backbone

    m = AuroraSmallPretrained(
        surf_vars=("2t", "10u", "10v", "msl", "ttr", "tcwv"),
        static_vars=("lsm", "z", "slt", "sst"),
    )
    freeze_backbone(
        backbone=m,
        new_surf_vars=("2t", "10u", "10v", "msl", "ttr", "tcwv"),
        use_lora=False,
        static_vars=("lsm", "z", "slt", "sst"),
    )

    surf_embed = m.encoder.surf_token_embeds
    assert "sst" in surf_embed.weights
    assert (
        surf_embed.weights["sst"].requires_grad is True
    ), "SST embedding weights must have requires_grad=True!"

    # Built-in statics must be frozen
    for var in ("lsm", "z", "slt"):
        assert (
            surf_embed.weights[var].requires_grad is False
        ), f"Built-in static '{var}' should be frozen!"

    # Built-in default surf vars must be frozen
    for var in ("2t", "10u", "10v"):
        assert surf_embed.weights[var].requires_grad is False

    # Injected surf vars must be unfrozen
    for var in ("ttr", "tcwv"):
        assert surf_embed.weights[var].requires_grad is True

    # SST has NO decoder head
    assert "sst" not in m.decoder.surf_heads


def test_sst_parameter_delta_exact() -> None:
    """Verify that adding SST increases model parameters by exactly the embedding size."""
    from aurora import AuroraSmallPretrained

    m_base = AuroraSmallPretrained(
        surf_vars=("2t", "10u", "10v", "msl", "ttr", "tcwv"),
        static_vars=("lsm", "z", "slt"),
    )
    m_sst = AuroraSmallPretrained(
        surf_vars=("2t", "10u", "10v", "msl", "ttr", "tcwv"),
        static_vars=("lsm", "z", "slt", "sst"),
    )

    params_base = sum(p.numel() for p in m_base.parameters())
    params_sst = sum(p.numel() for p in m_sst.parameters())
    delta = params_sst - params_base

    sst_weight = m_sst.encoder.surf_token_embeds.weights["sst"]
    expected_delta = sst_weight.numel()
    # For small: embed_dim=256, 1*2*4*4 = 32 -> 8,192
    assert expected_delta == 8192
    assert delta == expected_delta, f"Parameter delta {delta} != expected {expected_delta}"


def test_sst_rollout_persistence(synthetic_dataset: LANLMJODataset) -> None:
    """Verify SST persistence: in_batch.static_vars['sst'] is bitwise identical across 120 steps.

    _advance_batch passes in_batch.static_vars forward unchanged across all rollout steps.
    This test executes 119 state advances and asserts that static_vars['sst'] at step 119
    is bitwise identical (torch.equal) to step 0.
    """
    from aurora_mjo.trainer import _advance_batch

    in_batch, surf_targets, atmos_targets = synthetic_dataset[0]
    initial_sst = in_batch.static_vars["sst"].clone()

    current_batch = in_batch
    # Mock pred_batch with matching shapes
    mock_pred = in_batch

    for step in range(119):
        current_batch = _advance_batch(
            in_batch=current_batch,
            pred_batch=mock_pred,
            step_index=step,
            detach=True,
        )

    assert current_batch.metadata.rollout_step == 119
    step_119_sst = current_batch.static_vars["sst"]
    assert torch.equal(
        initial_sst, step_119_sst
    ), "SST at rollout step 119 is not bitwise identical to rollout step 0!"


@pytest.mark.needs_data
def test_real_cfs_sst_grid_equality_and_physical_range() -> None:
    """Verify real CFS dataset returns SST matching grid coordinates and physical priors."""
    if not REAL_CFS_ROOT.exists():
        pytest.skip(f"Real CFS archive not mounted at {REAL_CFS_ROOT}")
    slt_path = Path("data/static/slt_1deg.nc")
    sst_dir = Path("data/static/sst")
    if not (slt_path.exists() and sst_dir.exists()):
        pytest.skip("Static files slt or sst not found")

    ds = LANLMJODataset(
        start_year=1980,
        end_year=1980,
        root_dir=REAL_CFS_ROOT,
        slt_path=slt_path,
        sst_dir=sst_dir,
        max_rollout_steps=1,
    )
    in_batch, _, _ = ds[0]
    statics = in_batch.static_vars
    assert "sst" in statics
    sst = statics["sst"]
    assert sst.shape == (180, 360)
    assert torch.isfinite(sst).all()

    # Verify grid equality with invariants
    z = statics["z"]
    lsm = statics["lsm"]
    slt = statics["slt"]
    assert sst.shape == z.shape == lsm.shape == slt.shape == (180, 360)

    # Ocean physical range check: [271, 310] K
    ocean_mask = lsm < 0.5
    ocean_sst = sst[ocean_mask]
    assert ocean_sst.min().item() >= 270.0, f"Ocean SST min {ocean_sst.min().item()} < 270 K"
    assert ocean_sst.max().item() <= 310.0, f"Ocean SST max {ocean_sst.max().item()} > 310 K"
