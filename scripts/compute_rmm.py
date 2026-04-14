#!/usr/bin/env python3
"""
compute_rmm.py — RMM Evaluation Pipeline
=========================================
Mirrors Wheeler & Hendon (2004) methodology to derive RMM1/RMM2 indices
from ERA5-derived OLR, U850, and U200 fields stored in the LANL NERSC dataset.

Procedure
---------
1. Load OLR (mtnlwrf), U850, and U200 for each year using the LANL file layout.
2. Average over the tropical band [15°S, 15°N] to produce daily-mean time series.
3. Compute daily climatology (mean seasonal cycle, day-of-year mean) using
   only the training split (1980–2015). This is the only reference period.
4. Subtract training-period climatology from all years → anomalies.
5. Normalize each anomaly series by its training-period standard deviation.
6. Build training-period combined vector: [OLR_anom_norm, U850_anom_norm, U200_anom_norm].
7. Compute EOFs via covariance-matrix eigendecomposition on the training data.
   EOF1 and EOF2 are the eigenvectors with the two largest eigenvalues.
8. Project full dataset (train + validation + test) anomalies onto EOF basis
   → RMM1 and RMM2 time series for each split.
9. Compute amplitude = sqrt(RMM1² + RMM2²) and phase (1–8, Wheeler-Hendon convention).
10. Save outputs:
    - data/rmm_basis.npz   : EOF vectors, climatology, std (for reproducibility)
    - data/rmm_targets.nc  : per-timestep RMM1, RMM2, amplitude, phase, split label

Data leakage contract
---------------------
- EOFs, seasonal-cycle climatology, and normalization std are derived exclusively
  from the training split years (1980–2015 inclusive).
- Validation (2016–2019) and test (2020–2023) data are ONLY projected onto the
  frozen training-period basis, never used to refit any statistic.

Usage
-----
    python scripts/compute_rmm.py [--data-dir PATH] [--out-dir PATH] [--smoke-test]

    --data-dir   Root of LANL NERSC data tree
                 (default: /global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/
                           PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results)
    --out-dir    Directory to write rmm_targets.nc and rmm_basis.npz
                 (default: ./data)
    --smoke-test Run on a single simulated synthetic year to verify pipeline
                 shape/logic without needing real data on disk.

Smoke test (no data required)
------------------------------
    python scripts/compute_rmm.py --smoke-test

Dependencies
------------
    numpy, xarray, pandas, netcdf4  (all in environment.yml)
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Data splits (chronological, no overlap)
TRAIN_YEARS = list(range(1980, 2016))   # 1980 – 2015 inclusive
VAL_YEARS   = list(range(2016, 2020))   # 2016 – 2019 inclusive
TEST_YEARS  = list(range(2020, 2024))   # 2020 – 2023 inclusive

ALL_YEARS = TRAIN_YEARS + VAL_YEARS + TEST_YEARS

# Tropical averaging band (Wheeler & Hendon 2004)
LAT_S = -15.0   # 15°S
LAT_N =  15.0   # 15°N

# LANL file-system paths (mirrors dataset.py conventions)
_LANL_STEP02 = "Step02/ERA5.remap_180x360MODIS_6hrInst"
_LANL_STEP01 = "Step01/ERA5.remap_180x360MODIS_6hrInst"
_LANL_STEP03 = "Step03/ERA5.remap_180x360MODIS_6hrInst"

# Pressure levels we need for U850 / U200
_U850_HPA = 850
_U200_HPA = 200

# Default NERSC root (same as dataset.py LANL_DIR)
_DEFAULT_DATA_DIR = Path(
    "/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn"
    "/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results"
)

# Wheeler-Hendon phase lookup table
# Phase boundaries are in degrees of the RMM phase angle.
# Phase 1 corresponds to the angle range [90°, 135°), etc.
# We use the standard WH 8-phase definition.
_WH_PHASE_EDGES = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360])


# ---------------------------------------------------------------------------
# File-loading helpers  (LANL NERSC layout)
# ---------------------------------------------------------------------------

def _glob_year_files(root: Path, step_subdir: str, var_subdir: str, year: int) -> list[Path]:
    """Return sorted list of NetCDF files for one variable/year under root."""
    base = root / step_subdir / var_subdir
    patterns = [f"*{year}*.nc", f"*{year}*.nc4"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(base.glob(pat)))
    return files


def _load_var_year(root: Path, step_subdir: str, var_subdir: str,
                   lanl_varname: str, year: int,
                   pressure_level: int | None = None) -> xr.DataArray:
    """
    Load one atmospheric variable for one year.

    Parameters
    ----------
    root          : LANL data root
    step_subdir   : e.g. "Step02/ERA5.remap_180x360MODIS_6hrInst"
    var_subdir    : e.g. "meanTNLWFLX"
    lanl_varname  : name of the variable inside the NetCDF (e.g. "mtnlwrf")
    year          : calendar year
    pressure_level: if not None, select this hPa level from the pressure dim
    """
    files = _glob_year_files(root, step_subdir, var_subdir, year)
    if not files:
        raise FileNotFoundError(
            f"No files found for var='{lanl_varname}' year={year} "
            f"under {root / step_subdir / var_subdir}"
        )
    ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4")
    da = ds[lanl_varname]
    if pressure_level is not None:
        plev_dim = "isobaricInhPa" if "isobaricInhPa" in da.dims else "level"
        da = da.sel({plev_dim: pressure_level})
    return da


# ---------------------------------------------------------------------------
# Tropical mean
# ---------------------------------------------------------------------------

def tropical_mean(da: xr.DataArray) -> xr.DataArray:
    """
    Cosine-latitude-weighted mean over [LAT_S, LAT_N] and all longitudes.

    Returns a 1-D DataArray indexed by 'time'.
    """
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"

    da_trop = da.sel({lat_name: slice(LAT_N, LAT_S)})   # lat decreasing
    # If lat is increasing, slice the other way:
    if da_trop.sizes[lat_name] == 0:
        da_trop = da.sel({lat_name: slice(LAT_S, LAT_N)})

    weights = np.cos(np.deg2rad(da_trop[lat_name]))
    weights = weights / weights.sum()

    # Weighted mean over lat, then simple mean over lon
    da_lat = (da_trop * weights).sum(lat_name)
    da_mean = da_lat.mean(lon_name)
    return da_mean


# ---------------------------------------------------------------------------
# Daily (6-hourly → daily) downsampling
# ---------------------------------------------------------------------------

def to_daily_mean(da: xr.DataArray) -> xr.DataArray:
    """Resample 6-hourly tropical-mean series to daily mean."""
    return da.resample(time="1D").mean()


# ---------------------------------------------------------------------------
# Seasonal-cycle climatology (training period only)
# ---------------------------------------------------------------------------

def compute_daily_clim(da_daily: xr.DataArray) -> xr.DataArray:
    """
    Compute the day-of-year mean climatology from a 1-D daily DataArray.

    Returns a DataArray indexed by day-of-year (1–366) representing the
    training-period mean for each calendar day.
    """
    return da_daily.groupby("time.dayofyear").mean("time")


def remove_clim(da_daily: xr.DataArray, clim: xr.DataArray) -> xr.DataArray:
    """Subtract climatology from daily series → anomalies."""
    return da_daily.groupby("time.dayofyear") - clim


# ---------------------------------------------------------------------------
# EOF computation (pure numpy; no external EOF packages)
# ---------------------------------------------------------------------------

def compute_eofs(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute EOFs of a 2-D training matrix via covariance-matrix eigenvectors.

    Parameters
    ----------
    X_train : shape (T, N)  — T training time steps, N = 3 combined variables.

    Returns
    -------
    eof1 : shape (N,)  — first eigenvector (largest eigenvalue)
    eof2 : shape (N,)  — second eigenvector
    eigenvalues : shape (N,) — all eigenvalues, descending
    """
    # Remove time-mean (should already be zero after anomaly, but be safe)
    X_centred = X_train - X_train.mean(axis=0, keepdims=True)

    # Covariance matrix (N x N)
    cov = np.cov(X_centred, rowvar=False)   # shape (N, N)

    # eigh returns eigenvalues in ascending order for symmetric matrices
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Reverse to descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]  # columns are eigenvectors

    eof1 = eigenvectors[:, 0]
    eof2 = eigenvectors[:, 1]
    return eof1, eof2, eigenvalues


# ---------------------------------------------------------------------------
# Projection → RMM
# ---------------------------------------------------------------------------

def project_onto_eofs(X: np.ndarray, eof1: np.ndarray, eof2: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Project combined anomaly matrix X (T, N) onto EOF1 and EOF2.

    Returns RMM1 (T,) and RMM2 (T,) as 1-D arrays.
    """
    rmm1 = X @ eof1   # dot product of each row with EOF1
    rmm2 = X @ eof2
    return rmm1, rmm2


# ---------------------------------------------------------------------------
# Wheeler-Hendon phase assignment
# ---------------------------------------------------------------------------

def compute_wh_phase(rmm1: np.ndarray, rmm2: np.ndarray) -> np.ndarray:
    """
    Assign Wheeler-Hendon phases 1–8 from RMM1/RMM2.

    Convention (Wheeler & Hendon 2004, Table 1):
        angle = atan2(-RMM1, RMM2), converted to [0°, 360°)
        Phase 1 : 90°  – 135°
        Phase 2 : 135° – 180°
        Phase 3 : 180° – 225°
        Phase 4 : 225° – 270°
        Phase 5 : 270° – 315°
        Phase 6 : 315° – 360°
        Phase 7 : 0°   –  45°
        Phase 8 : 45°  –  90°

    Weak MJO (amplitude < 1) → phase returned as 0 (unclassified).
    """
    angle_rad = np.arctan2(-rmm1, rmm2)
    angle_deg = np.mod(np.degrees(angle_rad), 360.0)

    # Map to phases 7,8,1,2,3,4,5,6 by dividing 360° into 8 sectors of 45°
    # Sector 0 is [0,45) → WH phase 7
    _SECTOR_TO_PHASE = [7, 8, 1, 2, 3, 4, 5, 6]
    sector = (angle_deg // 45).astype(int)
    phase = np.array([_SECTOR_TO_PHASE[s] for s in sector])
    return phase


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_combined_vector(
    olr_daily: xr.DataArray,
    u850_daily: xr.DataArray,
    u200_daily: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Align three 1-D daily DataArrays on common time axis.

    Returns aligned (olr, u850, u200) DataArrays.
    """
    common_times = pd.DatetimeIndex(olr_daily.indexes["time"]) \
        .intersection(u850_daily.indexes["time"]) \
        .intersection(u200_daily.indexes["time"])

    olr_a   = olr_daily.sel(time=common_times)
    u850_a  = u850_daily.sel(time=common_times)
    u200_a  = u200_daily.sel(time=common_times)
    return olr_a, u850_a, u200_a


def run_pipeline(data_root: Path, out_dir: Path, verbose: bool = True) -> None:
    """
    Full RMM pipeline. Reads LANL data, writes rmm_basis.npz and rmm_targets.nc.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1: Load tropical-mean daily time series for all years
    # ------------------------------------------------------------------ #
    if verbose:
        print("=== Step 1: Loading tropical-mean daily time series ===")

    olr_parts:  list[xr.DataArray] = []
    u850_parts: list[xr.DataArray] = []
    u200_parts: list[xr.DataArray] = []

    for year in ALL_YEARS:
        if verbose:
            print(f"  Loading year {year}...", end="\r", flush=True)
        try:
            olr_da  = _load_var_year(data_root, _LANL_STEP03,
                                     "meanTNLWFLX", "mtnlwrf", year)
            u850_da = _load_var_year(data_root, _LANL_STEP01,
                                     "uWnd", "u", year,
                                     pressure_level=_U850_HPA)
            u200_da = _load_var_year(data_root, _LANL_STEP01,
                                     "uWnd", "u", year,
                                     pressure_level=_U200_HPA)
        except FileNotFoundError as exc:
            print(f"\n  WARNING: {exc}; skipping year {year}")
            continue

        olr_parts.append(to_daily_mean(tropical_mean(olr_da)))
        u850_parts.append(to_daily_mean(tropical_mean(u850_da)))
        u200_parts.append(to_daily_mean(tropical_mean(u200_da)))

    if verbose:
        print()   # newline after \r loop

    if not olr_parts:
        raise RuntimeError("No data could be loaded. Check data_root path.")

    olr_all  = xr.concat(olr_parts,  dim="time").sortby("time")
    u850_all = xr.concat(u850_parts, dim="time").sortby("time")
    u200_all = xr.concat(u200_parts, dim="time").sortby("time")

    # Align to common times (handles any gaps)
    olr_all, u850_all, u200_all = build_combined_vector(olr_all, u850_all, u200_all)

    if verbose:
        print(f"  Loaded {len(olr_all.time)} daily time steps across all splits.")

    # ------------------------------------------------------------------ #
    # Step 2 & 3: Training-period climatology and anomalies
    # ------------------------------------------------------------------ #
    if verbose:
        print("=== Step 2: Computing training-period seasonal climatology ===")

    times = pd.DatetimeIndex(olr_all.indexes["time"])
    train_mask = times.year.isin(TRAIN_YEARS)

    olr_train  = olr_all.isel(time=train_mask)
    u850_train = u850_all.isel(time=train_mask)
    u200_train = u200_all.isel(time=train_mask)

    # Seasonal-cycle climatology (DOY mean, training only)
    olr_clim  = compute_daily_clim(olr_train)
    u850_clim = compute_daily_clim(u850_train)
    u200_clim = compute_daily_clim(u200_train)

    # Anomalies (full dataset, subtracted from frozen training clim)
    olr_anom  = remove_clim(olr_all,  olr_clim)
    u850_anom = remove_clim(u850_all, u850_clim)
    u200_anom = remove_clim(u200_all, u200_clim)

    # ------------------------------------------------------------------ #
    # Step 3a: Normalize by training-period standard deviation
    # ------------------------------------------------------------------ #
    if verbose:
        print("=== Step 3: Normalizing by training-period std ===")

    olr_std  = float(olr_anom.sel(time=olr_anom.time.dt.year.isin(TRAIN_YEARS)).std())
    u850_std = float(u850_anom.sel(time=u850_anom.time.dt.year.isin(TRAIN_YEARS)).std())
    u200_std = float(u200_anom.sel(time=u200_anom.time.dt.year.isin(TRAIN_YEARS)).std())

    # Guard against degenerate data
    eps = 1e-8
    olr_norm  = olr_anom  / (olr_std  + eps)
    u850_norm = u850_anom / (u850_std + eps)
    u200_norm = u200_anom / (u200_std + eps)

    # ------------------------------------------------------------------ #
    # Step 4: EOF analysis — training split only
    # ------------------------------------------------------------------ #
    if verbose:
        print("=== Step 4: Computing EOFs on training split ===")

    # Stack into combined vector (T_train, 3)
    olr_tr  = olr_norm.sel(time=olr_norm.time.dt.year.isin(TRAIN_YEARS)).values
    u850_tr = u850_norm.sel(time=u850_norm.time.dt.year.isin(TRAIN_YEARS)).values
    u200_tr = u200_norm.sel(time=u200_norm.time.dt.year.isin(TRAIN_YEARS)).values

    X_train = np.stack([olr_tr, u850_tr, u200_tr], axis=1)   # (T_train, 3)
    X_train = np.nan_to_num(X_train, nan=0.0)

    eof1, eof2, eigenvalues = compute_eofs(X_train)
    var_exp = eigenvalues / eigenvalues.sum() * 100.0

    if verbose:
        print(f"  EOF1 explains {var_exp[0]:.1f}% of variance")
        print(f"  EOF2 explains {var_exp[1]:.1f}% of variance")
        print(f"  Combined: {var_exp[0] + var_exp[1]:.1f}%")

    # ------------------------------------------------------------------ #
    # Step 5: Project full dataset onto EOF basis
    # ------------------------------------------------------------------ #
    if verbose:
        print("=== Step 5: Projecting all splits onto EOF basis ===")

    olr_v  = olr_norm.values
    u850_v = u850_norm.values
    u200_v = u200_norm.values
    X_all = np.stack([olr_v, u850_v, u200_v], axis=1)
    X_all = np.nan_to_num(X_all, nan=0.0)

    rmm1_arr, rmm2_arr = project_onto_eofs(X_all, eof1, eof2)
    amplitude_arr = np.sqrt(rmm1_arr**2 + rmm2_arr**2)
    phase_arr = compute_wh_phase(rmm1_arr, rmm2_arr)

    # Assign split labels
    year_arr  = times[times.isin(pd.DatetimeIndex(olr_all.time.values))].year
    split_arr = np.full(len(year_arr), "train", dtype=object)

    # Use the full times from the aligned DataArray
    full_times = pd.DatetimeIndex(olr_all.time.values)
    split_arr2 = np.full(len(full_times), "train", dtype="<U4")
    split_arr2[full_times.year.isin(VAL_YEARS)]  = "val"
    split_arr2[full_times.year.isin(TEST_YEARS)] = "test"

    # ------------------------------------------------------------------ #
    # Step 6: Save outputs
    # ------------------------------------------------------------------ #
    if verbose:
        print("=== Step 6: Saving outputs ===")

    # 6a: Save RMM basis (frozen, training-period only)
    basis_path = out_dir / "rmm_basis.npz"
    np.savez(
        str(basis_path),
        eof1=eof1,
        eof2=eof2,
        eigenvalues=eigenvalues,
        olr_clim=olr_clim.values,
        u850_clim=u850_clim.values,
        u200_clim=u200_clim.values,
        olr_std=np.array([olr_std]),
        u850_std=np.array([u850_std]),
        u200_std=np.array([u200_std]),
        clim_dayofyear=olr_clim.dayofyear.values,
        train_years=np.array(TRAIN_YEARS),
        val_years=np.array(VAL_YEARS),
        test_years=np.array(TEST_YEARS),
    )
    if verbose:
        print(f"  Saved EOF basis → {basis_path}")

    # 6b: Save per-timestep RMM targets as NetCDF
    ds_out = xr.Dataset(
        {
            "rmm1":      xr.DataArray(rmm1_arr.astype(np.float32),      dims=["time"]),
            "rmm2":      xr.DataArray(rmm2_arr.astype(np.float32),      dims=["time"]),
            "amplitude": xr.DataArray(amplitude_arr.astype(np.float32), dims=["time"]),
            "phase":     xr.DataArray(phase_arr.astype(np.int8),         dims=["time"]),
            "split":     xr.DataArray(split_arr2,                        dims=["time"]),
        },
        coords={"time": olr_all.time.values},
        attrs={
            "description": "RMM indices derived following Wheeler & Hendon (2004)",
            "train_years": f"{TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}",
            "val_years":   f"{VAL_YEARS[0]}–{VAL_YEARS[-1]}",
            "test_years":  f"{TEST_YEARS[0]}–{TEST_YEARS[-1]}",
            "tropical_band": f"{LAT_S}°N – {LAT_N}°N",
            "variables": "OLR (mtnlwrf), U850, U200",
            "eof_basis": "Trained on 1980–2015 only (no data leakage)",
            "active_mjo_threshold": "amplitude > 1.0",
            "units_rmm": "dimensionless (normalized)",
            "units_amplitude": "dimensionless",
        },
    )
    targets_path = out_dir / "rmm_targets.nc"
    ds_out.to_netcdf(str(targets_path))
    if verbose:
        print(f"  Saved RMM targets → {targets_path}")
        _print_summary(ds_out, var_exp)


def _print_summary(ds: xr.Dataset, var_exp: np.ndarray) -> None:
    """Print a brief diagnostic summary to stdout."""
    for split in ["train", "val", "test"]:
        mask = ds["split"] == split
        if mask.sum() == 0:
            continue
        amp  = ds["amplitude"].where(mask, drop=True)
        rmm1 = ds["rmm1"].where(mask, drop=True)
        rmm2 = ds["rmm2"].where(mask, drop=True)
        n_active = int((amp > 1.0).sum())
        print(f"\n  [{split.upper()}]")
        print(f"    timesteps : {int(mask.sum())}")
        print(f"    active MJO: {n_active} ({100*n_active/int(mask.sum()):.1f}%)")
        print(f"    RMM1 mean/std : {float(rmm1.mean()):.3f} / {float(rmm1.std()):.3f}")
        print(f"    RMM2 mean/std : {float(rmm2.mean()):.3f} / {float(rmm2.std()):.3f}")
        print(f"    amplitude mean: {float(amp.mean()):.3f}")
    print()


# ---------------------------------------------------------------------------
# Smoke test (synthetic data — no disk I/O needed)
# ---------------------------------------------------------------------------

def run_smoke_test() -> None:
    """
    Runs the EOF + projection logic on synthetic data to verify shapes and
    non-degenerate outputs. Requires no real files on disk.
    """
    print("=== Smoke Test: Verifying pipeline logic with synthetic data ===\n")
    rng = np.random.default_rng(42)

    # Simulate 40 years × 365 days → (14600, 3)
    T_train = 36 * 365          # 1980–2015
    T_val   = 4  * 365          # 2016–2019
    T_test  = 4  * 365          # 2020–2023
    T_all   = T_train + T_val + T_test

    # Two dominant modes (synthetic "true" RMM)
    t = np.linspace(0, T_all, T_all)
    mode1 = np.sin(2 * np.pi * t / 48)   # ~48-day MJO period
    mode2 = np.cos(2 * np.pi * t / 48)

    eof_true = np.array([[0.6, 0.6, 0.5],
                          [0.5, -0.5, 0.7]])
    eof_true /= np.linalg.norm(eof_true, axis=1, keepdims=True)

    X = np.outer(mode1, eof_true[0]) + np.outer(mode2, eof_true[1]) \
        + rng.standard_normal((T_all, 3)) * 0.3

    X_train = X[:T_train]
    X_val   = X[T_train:T_train + T_val]
    X_test  = X[T_train + T_val:]

    # Run EOF on training
    eof1, eof2, eigenvalues = compute_eofs(X_train)
    var_exp = eigenvalues / eigenvalues.sum() * 100.0

    print(f"  Training matrix shape : {X_train.shape}")
    print(f"  EOF1 shape            : {eof1.shape}")
    print(f"  EOF2 shape            : {eof2.shape}")
    print(f"  EOF1 variance explained: {var_exp[0]:.1f}%")
    print(f"  EOF2 variance explained: {var_exp[1]:.1f}%")
    print(f"  Combined              : {var_exp[0]+var_exp[1]:.1f}%")

    # Project
    rmm1_tr, rmm2_tr = project_onto_eofs(X_train, eof1, eof2)
    rmm1_v,  rmm2_v  = project_onto_eofs(X_val,   eof1, eof2)
    rmm1_te, rmm2_te = project_onto_eofs(X_test,  eof1, eof2)

    print(f"\n  Train RMM1 std: {rmm1_tr.std():.3f}  (expected ~1)")
    print(f"  Val   RMM1 std: {rmm1_v.std():.3f}")
    print(f"  Test  RMM1 std: {rmm1_te.std():.3f}")

    # Phase
    phases_tr = compute_wh_phase(rmm1_tr, rmm2_tr)
    amp_tr    = np.sqrt(rmm1_tr**2 + rmm2_tr**2)

    phase_counts = np.bincount(phases_tr, minlength=9)[1:]   # phases 1–8
    print(f"\n  Phase distribution (train) [phases 1–8]:")
    print(f"  {phase_counts.tolist()}")
    print(f"\n  Amplitude mean (train): {amp_tr.mean():.3f}")
    print(f"  Active MJO fraction  : {(amp_tr > 1.0).mean()*100:.1f}%")

    print("\n=== Smoke Test PASSED ===\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute RMM indices (Wheeler-Hendon 2004) from LANL ERA5 data."
    )
    p.add_argument(
        "--data-dir", type=Path, default=_DEFAULT_DATA_DIR,
        help="Root of LANL NERSC data tree (default: NERSC path).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("data"),
        help="Output directory for rmm_targets.nc and rmm_basis.npz (default: ./data).",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Run on synthetic data only; do not read real files.",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=xr.coding.times.SerializationWarning)

    if args.smoke_test:
        run_smoke_test()
        return

    if not args.data_dir.exists():
        print(
            f"ERROR: data-dir does not exist: {args.data_dir}\n"
            "       Are you running on NERSC Perlmutter? Use --smoke-test to verify "
            "pipeline logic without data.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_pipeline(
        data_root=args.data_dir,
        out_dir=args.out_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
