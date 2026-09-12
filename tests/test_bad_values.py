"""tests/test_bad_values.py — Converted from scripts/scan_for_bad_values.py.

Verifies:
  1. Offline scan over synthetic NetCDF archive has zero NaNs, zero Infs, and zero extreme
     values exceeding physical sanity bounds (finite fraction is strictly 1.0).
  2. The bad-value detector fires deterministically when NaNs, Infs, or extreme values
     are injected into data files.
  3. Real CFS data scan when mounted (marked needs_data).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

SURFACE_VAR_MAP = {
    "2t": ("Step02/ERA5.remap_180x360MODIS_6hrInst/T2", "*", "t2"),
    "10u": ("Step02/ERA5.remap_180x360MODIS_6hrInst/U10", "*", "u10"),
    "10v": ("Step02/ERA5.remap_180x360MODIS_6hrInst/V10", "*", "v10"),
    "msl": ("Step02/ERA5.remap_180x360MODIS_6hrInst/PS", "*", "ps"),
    "ttr": ("Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX", "*", "mtnlwrf"),
    "tcwv": ("Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv", "*", "tcwv"),
}

ATMOS_VAR_MAP = {
    "z": ("Step01/ERA5.remap_180x360MODIS_6hrInst/gopt", "*", "z"),
    "q": ("Step01/ERA5.remap_180x360MODIS_6hrInst/sphu", "*", "q"),
    "t": ("Step01/ERA5.remap_180x360MODIS_6hrInst/tprt", "*", "t"),
    "u": ("Step01/ERA5.remap_180x360MODIS_6hrInst/uWnd", "*", "u"),
    "v": ("Step01/ERA5.remap_180x360MODIS_6hrInst/vWnd", "*", "v"),
}

SANE_ABS_MAX = {
    "2t": 400.0,
    "10u": 200.0,
    "10v": 200.0,
    "msl": 200000.0,
    "ttr": 2000.0,
    "tcwv": 500.0,
    "z": 1e6,
    "q": 1.0,
    "t": 400.0,
    "u": 300.0,
    "v": 300.0,
}

CFS_ROOT = Path(
    "/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results"
)


@dataclass
class ScanResult:
    name: str
    total: int
    n_nan: int
    n_inf: int
    n_extreme: int
    vmin: float
    vmax: float

    @property
    def finite_fraction(self) -> float:
        if self.total == 0:
            return 1.0
        return (self.total - self.n_nan - self.n_inf) / self.total


def scan_variable(
    root: Path,
    name: str,
    step_subdir: str,
    glob_pattern: str,
    native_name: str,
    start_year: int = 1980,
    end_year: int = 1980,
    max_files: int = 5,
) -> ScanResult:
    """Scan files for a variable and return counts of NaNs, Infs, and extreme values."""
    var_dir = root / step_subdir
    files: list[Path] = []
    for year in range(start_year, end_year + 1):
        year_dir = var_dir / str(year)
        matched = sorted(year_dir.glob(glob_pattern + ".nc"))
        if not matched:
            matched = sorted(var_dir.glob(f"*{year}*.nc"))
        files.extend(matched)
        if len(files) >= max_files:
            break

    if not files:
        raise FileNotFoundError(f"No files found for {name} under {var_dir}")

    files = files[:max_files]
    total = 0
    n_nan = 0
    n_inf = 0
    n_extreme = 0
    vmin = float("inf")
    vmax = float("-inf")
    bound = SANE_ABS_MAX.get(name, 1e30)

    for f in files:
        with xr.open_dataset(str(f), engine="netcdf4") as ds:
            if native_name not in ds:
                raise KeyError(f"Variable '{native_name}' not in {f}")
            arr = ds[native_name].values.astype(np.float64)

        total += arr.size
        n_nan += int(np.isnan(arr).sum())
        n_inf += int(np.isinf(arr).sum())
        finite = arr[np.isfinite(arr)]
        if finite.size:
            n_extreme += int((np.abs(finite) > bound).sum())
            vmin = min(vmin, float(finite.min()))
            vmax = max(vmax, float(finite.max()))

    return ScanResult(
        name=name,
        total=total,
        n_nan=n_nan,
        n_inf=n_inf,
        n_extreme=n_extreme,
        vmin=vmin,
        vmax=vmax,
    )


def test_scan_synthetic_archive_has_no_bad_values(synthetic_root: Path) -> None:
    """Verify that all variables in synthetic archive have finite_fraction=1.0 and no bad values."""
    all_vars = {**SURFACE_VAR_MAP, **ATMOS_VAR_MAP}

    for name, (subdir, glob_pat, native) in all_vars.items():
        res = scan_variable(
            root=synthetic_root,
            name=name,
            step_subdir=subdir,
            glob_pattern=glob_pat,
            native_name=native,
            start_year=1980,
            end_year=1980,
            max_files=1,
        )

        bound = SANE_ABS_MAX[name]
        assert res.total > 0, f"Variable '{name}' had 0 scanned values"
        assert res.n_nan == 0, f"Variable '{name}' contains {res.n_nan} NaNs"
        assert res.n_inf == 0, f"Variable '{name}' contains {res.n_inf} Infs"
        assert (
            res.n_extreme == 0
        ), f"Variable '{name}' contains {res.n_extreme} values exceeding bound {bound}"
        assert (
            res.finite_fraction == 1.0
        ), f"Variable '{name}' finite fraction is {res.finite_fraction}"
        assert np.isfinite(res.vmin), f"Variable '{name}' min is non-finite"
        assert np.isfinite(res.vmax), f"Variable '{name}' max is non-finite"


def test_bad_value_detector_fires_on_injected_nan_inf_and_extreme(
    synthetic_root: Path,
    tmp_path: Path,
) -> None:
    """Verify that the scanner detects injected NaNs, Infs, and extreme values."""
    # Choose 2t (T2) file
    subdir, _, native = SURFACE_VAR_MAP["2t"]
    src_file = synthetic_root / subdir / "e5.oper.an.sfc.128_t2.1980.nc"
    assert src_file.exists(), f"Source fixture missing at {src_file}"

    # Setup isolated test directory structure
    dst_dir = tmp_path / subdir
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / "e5.oper.an.sfc.128_t2.1980.nc"

    # Copy and inject anomalies into the array
    with xr.open_dataset(str(src_file), engine="netcdf4") as ds:
        arr = ds[native].values.copy()
        # Inject 1 NaN, 1 Inf, and 1 Extreme value (> 400 K for 2t)
        arr[0, 0, 0] = np.nan
        arr[0, 0, 1] = np.inf
        arr[0, 0, 2] = 999999.0

        ds_corrupted = ds.copy(deep=True)
        ds_corrupted[native].values = arr
        ds_corrupted.to_netcdf(str(dst_file), engine="netcdf4")

    # Run scan on corrupted directory
    res = scan_variable(
        root=tmp_path,
        name="2t",
        step_subdir=subdir,
        glob_pattern="*",
        native_name=native,
        start_year=1980,
        end_year=1980,
        max_files=1,
    )

    # Confirm the detector fires on each injected anomaly
    assert res.n_nan == 1, f"Expected 1 NaN detected, found {res.n_nan}"
    assert res.n_inf == 1, f"Expected 1 Inf detected, found {res.n_inf}"
    assert res.n_extreme == 1, f"Expected 1 extreme value detected, found {res.n_extreme}"
    assert res.finite_fraction < 1.0, "Finite fraction should decrease when NaNs/Infs are present"
    assert res.vmax >= 999999.0, f"Max value should reflect injected extreme value, got {res.vmax}"


@pytest.mark.needs_data
def test_real_cfs_data_bad_values_scan() -> None:
    """Verify that real CFS files have zero NaNs/Infs and finite_fraction=1.0."""
    if not CFS_ROOT.exists():
        pytest.skip(f"CFS archive not mounted at {CFS_ROOT}")

    for name in ("2t", "msl", "t"):
        var_map = SURFACE_VAR_MAP if name in SURFACE_VAR_MAP else ATMOS_VAR_MAP
        subdir, glob_pat, native = var_map[name]
        res = scan_variable(
            root=CFS_ROOT,
            name=name,
            step_subdir=subdir,
            glob_pattern=glob_pat,
            native_name=native,
            start_year=1980,
            end_year=1980,
            max_files=1,
        )
        assert res.total > 0
        assert res.n_nan == 0, f"CFS {name} has NaNs"
        assert res.n_inf == 0, f"CFS {name} has Infs"
        assert res.finite_fraction == 1.0
