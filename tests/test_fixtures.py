"""tests/test_fixtures.py — Comprehensive validation tests for synthetic NetCDF fixtures.

Tests the synthetic archives created by scripts/make_test_fixtures.py to ensure:
  1. Structural completeness matching upstream CFS layout.
  2. Native variable name correctness across surface, atmospheric, and static variables.
  3. Strict 6-hourly temporal contiguity with no duplicates.
  4. Leap-year arithmetic (1,464 steps in 1980, 1,460 steps in 1981).
  5. Per-variable chunking asymmetry (t2 1 file/year vs q 2 files/year).
  6. Gapped archive missing exactly 56 timesteps of tcwv in 1980.
  7. Physical plausibility of field means per 03_DOMAIN_PRIORS.md §5.
  8. Total archive size on disk under 20 MB ceiling (and standard archive under 5 MB).
  9. Offline construction of LANLMJODataset reporting exactly 1,462 samples for 1980.
  10. Auto-skip of needs_gpu tests when running in CPU-only mode.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import xarray as xr

from aurora_mjo.dataset import LANLMJODataset


def _get_directory_size(path: Path) -> int:
    """Return total size of directory or file in bytes."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# 1. Structural completeness
def test_expected_directories_and_files_exist(
    synthetic_root: Path,
    synthetic_root_gapped: Path,
    synthetic_slt: Path,
) -> None:
    """Verify that all required subdirectories and files exist in both archives."""
    expected_dirs = [
        "Step00/ERA5.invariant",
        "Step01/ERA5.remap_180x360MODIS_6hrInst/gopt",
        "Step01/ERA5.remap_180x360MODIS_6hrInst/sphu",
        "Step01/ERA5.remap_180x360MODIS_6hrInst/tprt",
        "Step01/ERA5.remap_180x360MODIS_6hrInst/uWnd",
        "Step01/ERA5.remap_180x360MODIS_6hrInst/vWnd",
        "Step02/ERA5.remap_180x360MODIS_6hrInst/T2",
        "Step02/ERA5.remap_180x360MODIS_6hrInst/U10",
        "Step02/ERA5.remap_180x360MODIS_6hrInst/V10",
        "Step02/ERA5.remap_180x360MODIS_6hrInst/PS",
        "Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv",
        "Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX",
        "Step06/ERA5.remap_180x360MODIS_6hrAccu/TP6H",
        "Step03/ERA5.remap_180x360MODIS_6hrInst/meanSLHFLX",
    ]

    for archive_root in (synthetic_root, synthetic_root_gapped):
        for sub in expected_dirs:
            d = archive_root / sub
            assert d.exists() and d.is_dir(), f"Missing directory: {d}"
            nc_files = list(d.glob("*.nc"))
            assert len(nc_files) > 0, f"No NetCDF files in: {d}"

    # Verify invariants in Step00
    for archive_root in (synthetic_root, synthetic_root_gapped):
        inv_dir = archive_root / "Step00/ERA5.invariant"
        assert len(list(inv_dir.glob("*_z.*.nc"))) >= 1, "Missing invariant Z file"
        assert len(list(inv_dir.glob("*_lsm.*.nc"))) >= 1, "Missing invariant LSM file"

    # Verify synthetic slt
    assert synthetic_slt.exists() and synthetic_slt.is_file(), f"Missing SLT file: {synthetic_slt}"


# 2. Native variable names
def test_files_contain_native_variable_names(
    synthetic_root: Path,
    synthetic_slt: Path,
) -> None:
    """Verify that each variable file contains its expected native variable name."""
    var_specs = [
        ("Step02/ERA5.remap_180x360MODIS_6hrInst/T2", "t2"),
        ("Step02/ERA5.remap_180x360MODIS_6hrInst/U10", "u10"),
        ("Step02/ERA5.remap_180x360MODIS_6hrInst/V10", "v10"),
        ("Step02/ERA5.remap_180x360MODIS_6hrInst/PS", "ps"),
        ("Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX", "mtnlwrf"),
        ("Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv", "tcwv"),
        ("Step06/ERA5.remap_180x360MODIS_6hrAccu/TP6H", "tp6h"),
        ("Step03/ERA5.remap_180x360MODIS_6hrInst/meanSLHFLX", "mslhf"),
        ("Step01/ERA5.remap_180x360MODIS_6hrInst/gopt", "z"),
        ("Step01/ERA5.remap_180x360MODIS_6hrInst/sphu", "q"),
        ("Step01/ERA5.remap_180x360MODIS_6hrInst/tprt", "t"),
        ("Step01/ERA5.remap_180x360MODIS_6hrInst/uWnd", "u"),
        ("Step01/ERA5.remap_180x360MODIS_6hrInst/vWnd", "v"),
    ]

    for subpath, native_name in var_specs:
        var_dir = synthetic_root / subpath
        files = list(var_dir.glob("*.nc"))
        assert len(files) > 0
        for f in files:
            with xr.open_dataset(str(f), engine="netcdf4") as ds:
                msg = f"Variable {native_name} not found in {f}. Found: {list(ds.variables)}"
                assert native_name in ds.variables, msg

    # Invariants in Step00
    inv_dir = synthetic_root / "Step00/ERA5.invariant"
    z_file = next(inv_dir.glob("*_z.*.nc"))
    with xr.open_dataset(str(z_file), engine="netcdf4") as ds:
        assert "Z" in ds.variables, f"Invariant 'Z' not found in {z_file}"

    lsm_file = next(inv_dir.glob("*_lsm.*.nc"))
    with xr.open_dataset(str(lsm_file), engine="netcdf4") as ds:
        assert "LSM" in ds.variables, f"Invariant 'LSM' not found in {lsm_file}"

    # SLT
    with xr.open_dataset(str(synthetic_slt), engine="netcdf4") as ds:
        assert "slt" in ds.variables, f"Variable 'slt' not found in {synthetic_slt}"


# 3. Exact 6-hour spacing and uniqueness
def test_time_coordinate_spacing_and_uniqueness(synthetic_root: Path) -> None:
    """Verify time coordinates have exact 6-hour delta and no duplicate timestamps."""
    six_hours_ns = int(6 * 3600 * 1_000_000_000)

    # Check surface t2 (single file per year)
    t2_dir = synthetic_root / "Step02/ERA5.remap_180x360MODIS_6hrInst/T2"
    for f in t2_dir.glob("*.nc"):
        with xr.open_dataset(str(f), engine="netcdf4") as ds:
            times = ds.time.values.astype("datetime64[ns]").astype(np.int64)
            diffs = np.diff(times)
            assert np.all(diffs == six_hours_ns), f"Non 6-hour steps found in {f}"
            assert len(times) == len(set(times)), f"Duplicate timestamps found in {f}"

    # Check q (half-year chunks): concatenated timestamps must form continuous 6-hour timeline
    q_dir = synthetic_root / "Step01/ERA5.remap_180x360MODIS_6hrInst/sphu"
    for year in (1980, 1981):
        h1_file = q_dir / f"e5.oper.an.pl.128_133_q.{year}_h1.nc"
        h2_file = q_dir / f"e5.oper.an.pl.128_133_q.{year}_h2.nc"
        assert h1_file.exists() and h2_file.exists()

        with xr.open_dataset(str(h1_file), engine="netcdf4") as ds1:
            t1 = ds1.time.values.astype("datetime64[ns]").astype(np.int64)
        with xr.open_dataset(str(h2_file), engine="netcdf4") as ds2:
            t2 = ds2.time.values.astype("datetime64[ns]").astype(np.int64)

        combined = np.concatenate([t1, t2])
        assert len(combined) == len(set(combined)), f"Duplicates in combined {year} q files"
        diffs = np.diff(combined)
        assert np.all(diffs == six_hours_ns), f"Gap between h1 and h2 in {year} q files"


# 4. Leap year vs non-leap year timestep counts
def test_timestep_counts_across_years(synthetic_root: Path) -> None:
    """Verify leap year 1980 has 1,464 timesteps and non-leap 1981 has 1,460."""
    t2_dir = synthetic_root / "Step02/ERA5.remap_180x360MODIS_6hrInst/T2"

    f_1980 = t2_dir / "e5.oper.an.sfc.128_t2.1980.nc"
    f_1981 = t2_dir / "e5.oper.an.sfc.128_t2.1981.nc"

    with xr.open_dataset(str(f_1980), engine="netcdf4") as ds80:
        n_1980 = len(ds80.time)
    with xr.open_dataset(str(f_1981), engine="netcdf4") as ds81:
        n_1981 = len(ds81.time)

    assert n_1980 == 1464, f"Expected 1,464 steps in leap year 1980, got {n_1980}"
    assert n_1981 == 1460, f"Expected 1,460 steps in 1981, got {n_1981}"
    assert n_1980 + n_1981 == 2924


# 5. Chunking asymmetry between t2 and q
def test_chunking_asymmetry_between_t2_and_q(synthetic_root: Path) -> None:
    """Verify variable file counts differ between t2 (1/yr) and q (2/yr)."""
    t2_files = list((synthetic_root / "Step02/ERA5.remap_180x360MODIS_6hrInst/T2").glob("*.nc"))
    q_files = list((synthetic_root / "Step01/ERA5.remap_180x360MODIS_6hrInst/sphu").glob("*.nc"))

    assert len(t2_files) == 2, f"Expected 2 t2 files, got {len(t2_files)}"
    assert len(q_files) == 4, f"Expected 4 q files, got {len(q_files)}"
    assert len(t2_files) != len(q_files)


# 6. Gapped variant tcwv missing steps
def test_gapped_variant_missing_tcwv_fortnight(
    synthetic_root: Path,
    synthetic_root_gapped: Path,
) -> None:
    """Verify that the gapped archive is missing exactly 56 timesteps of tcwv in 1980."""
    std_file = (
        synthetic_root
        / "Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv/e5.oper.an.sfc.128_tcwv.1980.nc"
    )
    gap_file = (
        synthetic_root_gapped
        / "Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv/e5.oper.an.sfc.128_tcwv.1980.nc"
    )

    with xr.open_dataset(str(std_file), engine="netcdf4") as ds_std:
        std_len = len(ds_std.time)
    with xr.open_dataset(str(gap_file), engine="netcdf4") as ds_gap:
        gap_len = len(ds_gap.time)

    assert std_len == 1464
    assert gap_len == 1408
    delta = std_len - gap_len
    assert delta == 56, f"Expected exactly 56 missing steps (14 days), got {delta}"


# 7. Field means match domain priors (03_DOMAIN_PRIORS.md §5 and §3)
def test_field_means_and_ranges_match_domain_priors(
    synthetic_root: Path,
    synthetic_slt: Path,
) -> None:
    """Verify field values and means match 03_DOMAIN_PRIORS.md §5 geophysical ranges."""
    # 2t: K, unweighted mean ~278-282 K
    with xr.open_dataset(
        str(
            synthetic_root
            / "Step02/ERA5.remap_180x360MODIS_6hrInst/T2/e5.oper.an.sfc.128_t2.1980.nc"
        )
    ) as ds:
        m = float(ds["t2"].mean())
        assert 275.0 <= m <= 285.0, f"2t mean {m} outside prior [275, 285]"
        assert float(ds["t2"].min()) > 200.0 and float(ds["t2"].max()) < 330.0

    # ps: Pa, mean ~96,000-99,000 Pa
    with xr.open_dataset(
        str(
            synthetic_root
            / "Step02/ERA5.remap_180x360MODIS_6hrInst/PS/e5.oper.an.sfc.128_ps.1980.nc"
        )
    ) as ds:
        m = float(ds["ps"].mean())
        assert 96000.0 <= m <= 99000.0, f"ps mean {m} outside prior [96000, 99000]"

    # mtnlwrf: W m^-2, negative, ~ -225 to -240
    mtnlwrf_path = (
        synthetic_root
        / "Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX"
        / "e5.oper.an.sfc.128_mtnlwrf.1980.nc"
    )
    with xr.open_dataset(str(mtnlwrf_path)) as ds:
        m = float(ds["mtnlwrf"].mean())
        assert -240.0 <= m <= -220.0, f"mtnlwrf mean {m} outside prior [-240, -220]"
        assert float(ds["mtnlwrf"].max()) < 0.0, "mtnlwrf must be strictly negative"

    # tcwv: kg m^-2, strictly non-negative, ~18-25
    with xr.open_dataset(
        str(
            synthetic_root
            / "Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv/e5.oper.an.sfc.128_tcwv.1980.nc"
        )
    ) as ds:
        m = float(ds["tcwv"].mean())
        assert 15.0 <= m <= 25.0, f"tcwv mean {m} outside prior [15, 25]"
        assert float(ds["tcwv"].min()) >= 0.0, "tcwv must be strictly non-negative"

    # Atmos t: 500 hPa ~253 K, 200 hPa ~218 K
    with xr.open_dataset(
        str(
            synthetic_root
            / "Step01/ERA5.remap_180x360MODIS_6hrInst/tprt/e5.oper.an.pl.128_t.1980.nc"
        )
    ) as ds:
        t_500 = float(ds["t"].sel(lev=500.0).mean())
        t_200 = float(ds["t"].sel(lev=200.0).mean())
        assert 250.0 <= t_500 <= 256.0, f"t@500hPa {t_500} outside prior ~253 K"
        assert 215.0 <= t_200 <= 222.0, f"t@200hPa {t_200} outside prior ~218 K"

    # Atmos z: 500 hPa ~54,000 m^2/s^2, 1000 hPa ~1,000 m^2/s^2
    with xr.open_dataset(
        str(
            synthetic_root
            / "Step01/ERA5.remap_180x360MODIS_6hrInst/gopt/e5.oper.an.pl.128_z.1980.nc"
        )
    ) as ds:
        z_500 = float(ds["z"].sel(lev=500.0).mean())
        z_1000 = float(ds["z"].sel(lev=1000.0).mean())
        assert 52000.0 <= z_500 <= 56000.0, f"z@500hPa {z_500} outside prior ~54,000"
        assert 800.0 <= z_1000 <= 1200.0, f"z@1000hPa {z_1000} outside prior ~1,000"

    # Atmos q: strictly positive, decreasing with height
    with xr.open_dataset(
        str(
            synthetic_root
            / "Step01/ERA5.remap_180x360MODIS_6hrInst/sphu/e5.oper.an.pl.128_133_q.1980_h1.nc"
        )
    ) as ds:
        q_1000 = float(ds["q"].sel(lev=1000.0).mean())
        q_200 = float(ds["q"].sel(lev=200.0).mean())
        assert 0.008 <= q_1000 <= 0.020, f"q@1000hPa {q_1000} outside prior ~0.014"
        assert 1e-6 <= q_200 <= 5e-5, f"q@200hPa {q_200} outside prior ~1e-5"
        assert float(ds["q"].min()) > 0.0

    # Invariants: Z ~3,709 m^2/s^2, LSM ~0.3357
    inv_dir = synthetic_root / "Step00/ERA5.invariant"
    with xr.open_dataset(str(next(inv_dir.glob("*_z.*.nc")))) as ds:
        m_z = float(ds["Z"].mean())
        assert 3500.0 <= m_z <= 4000.0, f"Z mean {m_z} outside prior ~3,709"
    with xr.open_dataset(str(next(inv_dir.glob("*_lsm.*.nc")))) as ds:
        m_lsm = float(ds["LSM"].mean())
        assert 0.25 <= m_lsm <= 0.40, f"LSM mean {m_lsm} outside prior ~0.33"

    # SLT: categorical 0..7, mean ~0.6708, std ~1.1682
    with xr.open_dataset(str(synthetic_slt)) as ds:
        m_slt = float(ds["slt"].mean())
        s_slt = float(ds["slt"].std())
        assert 0.50 <= m_slt <= 0.85, f"slt mean {m_slt} outside prior ~0.67"
        assert 1.0 <= s_slt <= 1.5, f"slt std {s_slt} outside prior ~1.17"


# 8. Fixture disk size under ceiling
def test_fixture_disk_size_under_hard_ceiling(
    synthetic_root: Path,
    synthetic_root_gapped: Path,
    synthetic_slt: Path,
) -> None:
    """Verify total committed fixture size is under 20 MB ceiling and ~5 MB target."""
    std_size = _get_directory_size(synthetic_root)
    gap_size = _get_directory_size(synthetic_root_gapped)
    slt_size = _get_directory_size(synthetic_slt)
    total_size = std_size + gap_size + slt_size

    # Target: standard archive under 5 MB
    msg_tgt = f"Standard archive ({std_size / 1e6:.2f} MB) exceeds 5 MB target"
    assert std_size < 5 * 1024 * 1024, msg_tgt
    # Hard ceiling: combined size under 20 MB
    msg_ceil = f"Total size ({total_size / 1e6:.2f} MB) exceeds 20 MB ceiling"
    assert total_size < 20 * 1024 * 1024, msg_ceil


# 9. Offline LANLMJODataset construction and sample count
def test_dataset_construction_offline_sample_count(synthetic_dataset: LANLMJODataset) -> None:
    """Verify LANLMJODataset constructs offline and reports exactly 1,462 samples for 1980."""
    assert len(synthetic_dataset) == 1462, f"Expected 1,462 samples, got {len(synthetic_dataset)}"

    batch, surf_tgt, atmos_tgt = synthetic_dataset[0]

    # Timestamp verification
    assert batch.metadata.time[0] == datetime(1980, 1, 1, 6, 0), "Expected first sample at 06:00:00"

    # Static variable shapes (native 18x36 in synthetic fixture)
    assert batch.static_vars["z"].shape == (18, 36)
    assert batch.static_vars["lsm"].shape == (18, 36)
    assert batch.static_vars["slt"].shape == (18, 36)

    # Dynamic variables at reduced grid (18, 36)
    assert batch.surf_vars["2t"].shape == (1, 2, 18, 36)
    assert batch.atmos_vars["t"].shape == (1, 2, 13, 18, 36)

    # Multi-step targets
    assert len(surf_tgt) == 1
    assert len(atmos_tgt) == 1
    assert surf_tgt[0]["2t"].shape == (1, 18, 36)
    assert atmos_tgt[0]["t"].shape == (1, 13, 18, 36)


# 10. Conftest session fixtures verification
def test_conftest_fixtures_load(
    unified_config: dict[str, Any],
    baseline_fingerprint: dict[str, Any],
) -> None:
    """Verify unified_config and baseline_fingerprint session fixtures load correctly."""
    assert "experiment" in unified_config
    assert unified_config["experiment"]["mode"] == "baseline"

    expected_baseline_keys = [
        "config_baseline",
        "config_baseline_override",
        "config_combined",
        "config_lora",
        "config_physics_informed",
        "dataset_1980",
        "model_parameters",
        "smoke_baseline",
    ]
    for key in expected_baseline_keys:
        assert key in baseline_fingerprint, f"Missing key '{key}' in baseline_fingerprint fixture"


# 11. needs_gpu auto-skip verification
@pytest.mark.needs_gpu
def test_needs_gpu_auto_skip_hook() -> None:
    """Verify that tests marked with needs_gpu execute when GPU is available."""
    skip_msg = "This test should have been auto-skipped if CUDA is unavailable."
    assert torch.cuda.is_available(), skip_msg
