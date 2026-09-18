#!/usr/bin/env python3
"""compute_rmm.py — RMM Evaluation Pipeline.

=========================================
Implements Wheeler & Hendon (2004) methodology per 02_SCIENTIFIC_CONTRACT.md §6
to derive RMM1/RMM2 indices from ERA5 OLR (-mtnlwrf), U850, and U200 fields stored
in the LANL NERSC dataset.

Procedure (WH 2004, 02_SCIENTIFIC_CONTRACT.md §6):
---------------------------------------------------
1. Load OLR (OLR = -mtnlwrf), U850, and U200 for 1980–2019. Test years (2020–2023)
   are strictly quarantined and never read.
2. Unweighted meridional mean over [15°S, 15°N] retaining zonal structure (360 longitudes).
   Resample to daily mean.
3. Fit annual cycle seasonal climatology as mean plus first three harmonics per longitude
   on the training split only (1980–2015).
4. Subtract training-period seasonal harmonics from all years → daily anomalies.
5. Subtract rolling 120-day mean at each longitude (Convention A, removing interannual/ENSO).
6. Normalise each field by its zonally averaged temporal standard deviation on the
   training period (one scalar per field).
7. Concatenate into (T, 1080) matrix.
8. Compute EOFs via SVD of the training matrix (no 1080x1080 covariance matrix formed).
9. PC normalisation: divide each PC by its training-period standard deviation so RMM1
   and RMM2 have unit variance on the training period.
10. Sign/order transform matrix: stored in rmm_basis.npz, defaults to identity.
11. Project 1980–2019 anomalies onto frozen basis → RMM1, RMM2, amplitude, phase.
12. Save:
    - data/rmm_basis.npz   : frozen EOF basis, harmonics, stds, sign transform
    - data/rmm_targets.nc  : per-day RMM1, RMM2, amplitude, phase, split label (1980–2019 only)

Data leakage contract:
----------------------
- Harmonic climatology, zonal standard deviations, EOF basis, and PC normalisation
  constants are derived strictly from 1980–2015.
- Validation (2016–2019) is only projected onto the frozen basis.
- Test years (2020–2023) are NOT touched by this campaign.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Ensure HDF5 file locking disabled for CFS
import aurora_mjo.env  # noqa: F401
from aurora_mjo.rmm.compute import (
    LAT_N,
    LAT_S,
    compute_eofs,
    compute_wh_phase,
    compute_zonal_std,
    fit_seasonal_harmonics,
    project_onto_eofs,
    remove_120d_mean,
    remove_clim,
    to_daily_mean,
    tropical_mean,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Data splits (chronological, no overlap)
# TEST_YEARS (2020–2023) are QUARANTINED per 04_AGENT_PROTOCOL.md §4.
TRAIN_YEARS = list(range(1980, 2016))  # 1980 – 2015 inclusive (36 years, 13,149 days)
VAL_YEARS = list(range(2016, 2020))  # 2016 – 2019 inclusive (4 years, 1,461 days)
ACTIVE_YEARS = TRAIN_YEARS + VAL_YEARS  # 1980 – 2019 inclusive (40 years, 14,610 days)

# LANL file-system paths
_LANL_STEP01 = "Step01/ERA5.remap_180x360MODIS_6hrInst"
_LANL_STEP03 = "Step03/ERA5.remap_180x360MODIS_6hrInst"

_U850_HPA = 850
_U200_HPA = 200

_DEFAULT_DATA_DIR = Path(
    "/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn"
    "/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results"
)


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------


def _glob_year_files(base_dir: Path, year: int) -> list[Path]:
    """Find files strictly for given year in base_dir using precise regex (Lesson 5)."""
    # Match .YYYYMM or _YYYYMM where MM is 01..12
    pattern = re.compile(rf"[\._]{year}(0[1-9]|1[0-2])")
    files = sorted(
        [
            f
            for f in base_dir.iterdir()
            if pattern.search(f.name) and f.suffix in (".nc", ".nc4")
        ]
    )
    if not files:
        year_dir = base_dir / str(year)
        if year_dir.is_dir():
            files = sorted(
                [
                    f
                    for f in year_dir.iterdir()
                    if pattern.search(f.name) and f.suffix in (".nc", ".nc4")
                ]
            )
    return files


def load_year_daily_tropical(
    data_root: Path, year: int
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Load OLR, U850, and U200 daily tropical-mean fields (shape (days, 360)) for one year.

    Slices [15°S, 15°N] per-file for memory and I/O efficiency.
    OLR sign convention: OLR = -mtnlwrf (Wheeler & Hendon 2004, 02_SCIENTIFIC_CONTRACT.md §6.1).
    """
    u_dir = data_root / _LANL_STEP01 / "uWnd"
    olr_dir = data_root / _LANL_STEP03 / "meanTNLWFLX"

    u_files = _glob_year_files(u_dir, year)
    olr_files = _glob_year_files(olr_dir, year)

    if len(u_files) != 12:
        raise FileNotFoundError(
            f"Expected 12 uWnd monthly files for year {year}, found {len(u_files)} under {u_dir}"
        )
    if len(olr_files) != 12:
        msg = f"Expected 12 meanTNLWFLX files for {year}, got {len(olr_files)} under {olr_dir}"
        raise FileNotFoundError(msg)

    # Process uWnd (both 850 and 200 hPa extracted in same file read)
    u850_parts: list[xr.DataArray] = []
    u200_parts: list[xr.DataArray] = []

    for f in u_files:
        with xr.open_dataset(f) as ds:
            u_var = ds["u"]
            # Identify vertical dimension ('lev', 'level', 'isobaricInhPa')
            vdim = None
            for d in ("lev", "level", "isobaricInhPa", "plev"):
                if d in u_var.dims:
                    vdim = d
                    break
            if vdim is None:
                raise KeyError(f"No pressure level dimension found in {f}: dims={u_var.dims}")

            u850_slice = u_var.sel({vdim: _U850_HPA})
            u200_slice = u_var.sel({vdim: _U200_HPA})

            u850_trop = tropical_mean(u850_slice).load()
            u200_trop = tropical_mean(u200_slice).load()

            u850_parts.append(u850_trop)
            u200_parts.append(u200_trop)

    u850_year = xr.concat(u850_parts, dim="time").sortby("time")
    u200_year = xr.concat(u200_parts, dim="time").sortby("time")

    u850_daily = to_daily_mean(u850_year)
    u200_daily = to_daily_mean(u200_year)

    # Process OLR (mtnlwrf, with sign flip OLR = -mtnlwrf)
    olr_parts: list[xr.DataArray] = []
    for f in olr_files:
        with xr.open_dataset(f) as ds:
            olr_var = ds["mtnlwrf"]
            olr_trop = tropical_mean(olr_var).load()
            olr_parts.append(olr_trop)

    olr_year = xr.concat(olr_parts, dim="time").sortby("time")
    # OLR is outgoing radiation; net downward flux mtnlwrf is negative
    olr_daily = to_daily_mean(-olr_year)

    return olr_daily, u850_daily, u200_daily


# ---------------------------------------------------------------------------
# Diagnostics & Summary
# ---------------------------------------------------------------------------


def compute_rmm_diagnostics(
    rmm1_tr: np.ndarray,
    rmm2_tr: np.ndarray,
    eigenvalues: np.ndarray,
) -> dict[str, float]:
    """Compute checkable properties matching 03_DOMAIN_PRIORS.md §7.

    1. Variance explained EOF1, EOF2, combined (~25% combined, roughly equal split)
    2. RMM1, RMM2 variance on training (1.0 each by construction)
    3. Correlation RMM1 vs RMM2 (≈ 0)
    4. Lag between RMM1 and RMM2 (~10–12 days in quadrature)
    5. Fraction of days with A > 1 (~50%)
    6. Mean period of an 8-phase cycle (30–60 days)
    """
    total_var = float(np.sum(eigenvalues))
    var_exp_1 = float(eigenvalues[0] / total_var * 100.0)
    var_exp_2 = float(eigenvalues[1] / total_var * 100.0)
    var_exp_comb = var_exp_1 + var_exp_2

    var_rmm1 = float(np.var(rmm1_tr))
    var_rmm2 = float(np.var(rmm2_tr))
    corr_12 = float(np.corrcoef(rmm1_tr, rmm2_tr)[0, 1])

    amp_tr = np.sqrt(rmm1_tr**2 + rmm2_tr**2)
    frac_active = float(np.mean(amp_tr > 1.0) * 100.0)

    # Quadrature lag: cross-correlation between RMM1 and RMM2
    lags = np.arange(-30, 31)
    cc = [
        float(
            np.corrcoef(
                rmm1_tr[:-lag] if lag > 0 else (rmm1_tr[-lag:] if lag < 0 else rmm1_tr),
                rmm2_tr[lag:] if lag > 0 else (rmm2_tr[:lag] if lag < 0 else rmm2_tr),
            )[0, 1]
        )
        for lag in lags
    ]
    peak_lag = int(abs(lags[np.argmax(np.abs(cc))]))

    # Mean period: estimate from phase angle advancement
    angle = np.mod(np.degrees(np.arctan2(-rmm1_tr, rmm2_tr)), 360.0)
    # Unwrapped phase advance
    d_angle = np.diff(np.unwrap(np.deg2rad(angle)))
    # Positive advance rate for active days
    active_mask = (amp_tr[:-1] > 1.0) & (d_angle > 0)
    if np.sum(active_mask) > 100:
        mean_advance_deg_day = float(np.degrees(np.median(d_angle[active_mask])))
        mean_period_days = float(360.0 / max(1e-3, mean_advance_deg_day))
    else:
        mean_period_days = float("nan")

    return {
        "var_exp_eof1": var_exp_1,
        "var_exp_eof2": var_exp_2,
        "var_exp_comb": var_exp_comb,
        "var_rmm1": var_rmm1,
        "var_rmm2": var_rmm2,
        "corr_rmm1_rmm2": corr_12,
        "peak_lag_days": float(peak_lag),
        "fraction_active_pct": frac_active,
        "mean_period_days": mean_period_days,
    }


def _print_diagnostic_table(diag: dict[str, float]) -> None:
    """Print 03_DOMAIN_PRIORS.md §7 diagnostic comparison table."""
    print("\n" + "=" * 80)
    print("  03_DOMAIN_PRIORS.md §7 RMM Sanity Check Table (Training Split 1980–2015)")
    print("=" * 80)
    print(
        f"  1. Variance explained (EOF1, EOF2, Combined): {diag['var_exp_eof1']:.2f}%, "
        f"{diag['var_exp_eof2']:.2f}%, {diag['var_exp_comb']:.2f}%  (Prior: ~25% combined)"
    )
    print(
        f"  2. Training PC variance (RMM1, RMM2):         {diag['var_rmm1']:.4f}, "
        f"{diag['var_rmm2']:.4f}  (Prior: 1.0000 each)"
    )
    print(
        f"  3. Orthogonality corr(RMM1, RMM2):            {diag['corr_rmm1_rmm2']:+.4f}  "
        "(Prior: ≈ 0)"
    )
    print(
        f"  4. Quadrature lag between RMM1 & RMM2:        {diag['peak_lag_days']:.0f} days  "
        "(Prior: ~10–12 days)"
    )
    print(
        f"  5. Active MJO fraction (A > 1.0):             {diag['fraction_active_pct']:.1f}%  "
        "(Prior: ~50%)"
    )
    print(
        f"  6. Estimated MJO period:                      {diag['mean_period_days']:.1f} days  "
        "(Prior: 30–60 days)"
    )
    print("  7. BoM Correlation (2016–2019):               DEFERRED TO TASK J2 (Primary Gate)")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(data_root: Path, out_dir: Path, verbose: bool = True) -> None:
    """Full RMM pipeline over 1980–2019. Saves rmm_basis.npz and rmm_targets.nc."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=== Step 1: Loading tropical daily fields (1980–2019) ===")
        print(f"  Training years:   {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]} ({len(TRAIN_YEARS)} years)")
        print(f"  Validation years: {VAL_YEARS[0]}–{VAL_YEARS[-1]} ({len(VAL_YEARS)} years)")
        print("  Test years:       QUARANTINED (not read)")

    olr_list: list[xr.DataArray] = []
    u850_list: list[xr.DataArray] = []
    u200_list: list[xr.DataArray] = []

    for year in ACTIVE_YEARS:
        if verbose:
            print(f"  Loading year {year}...", flush=True)
        olr_y, u850_y, u200_y = load_year_daily_tropical(data_root, year)
        olr_list.append(olr_y)
        u850_list.append(u850_y)
        u200_list.append(u200_y)

    olr_all = xr.concat(olr_list, dim="time").sortby("time")
    u850_all = xr.concat(u850_list, dim="time").sortby("time")
    u200_all = xr.concat(u200_list, dim="time").sortby("time")

    total_days = len(olr_all.time)
    assert total_days == 14610, f"Expected 14,610 days over 1980–2019, got {total_days}"

    if verbose:
        print(f"\n  Loaded {total_days} days across 1980–2019 (14,610 expected).")

    # Step 2: Fit seasonal cycle harmonics on training period only (1980–2015)
    if verbose:
        print("=== Step 2: Fitting seasonal cycle (mean + 3 harmonics) on 1980–2015 ===")

    train_mask = olr_all.time.dt.year.isin(TRAIN_YEARS).values
    train_indices = np.where(train_mask)[0]
    assert len(train_indices) == 13149, f"Expected 13,149 training days, got {len(train_indices)}"

    olr_train = olr_all.isel(time=train_indices)
    u850_train = u850_all.isel(time=train_indices)
    u200_train = u200_all.isel(time=train_indices)

    olr_clim, _ = fit_seasonal_harmonics(olr_train, n_harmonics=3)
    u850_clim, _ = fit_seasonal_harmonics(u850_train, n_harmonics=3)
    u200_clim, _ = fit_seasonal_harmonics(u200_train, n_harmonics=3)

    # Step 3: Remove seasonal cycle from all years (1980–2019)
    if verbose:
        print("=== Step 3: Subtracting seasonal harmonics → anomalies ===")

    olr_anom = remove_clim(olr_all, olr_clim)
    u850_anom = remove_clim(u850_all, u850_clim)
    u200_anom = remove_clim(u200_all, u200_clim)

    # Step 4: Subtract rolling 120-day mean (Convention A)
    if verbose:
        print("=== Step 4: Subtracting 120-day mean (Convention A) ===")

    olr_filt = remove_120d_mean(olr_anom, convention="A")
    u850_filt = remove_120d_mean(u850_anom, convention="A")
    u200_filt = remove_120d_mean(u200_anom, convention="A")

    # Step 5: Normalise each field by its zonally averaged temporal std (training period only)
    if verbose:
        print("=== Step 5: Computing zonal temporal std on 1980–2015 ===")

    olr_filt_tr = olr_filt.isel(time=train_indices)
    u850_filt_tr = u850_filt.isel(time=train_indices)
    u200_filt_tr = u200_filt.isel(time=train_indices)

    olr_std = compute_zonal_std(olr_filt_tr)
    u850_std = compute_zonal_std(u850_filt_tr)
    u200_std = compute_zonal_std(u200_filt_tr)

    if verbose:
        print(f"  olr_std:  {olr_std:.4f} W m⁻²")
        print(f"  u850_std: {u850_std:.4f} m s⁻¹")
        print(f"  u200_std: {u200_std:.4f} m s⁻¹")

    olr_norm = olr_filt / olr_std
    u850_norm = u850_filt / u850_std
    u200_norm = u200_filt / u200_std

    # Step 6: Construct combined (T, 1080) matrix
    if verbose:
        print("=== Step 6: Constructing (T, 1080) matrix ===")

    olr_vals = olr_norm.values  # (T, 360)
    u850_vals = u850_norm.values
    u200_vals = u200_norm.values

    X_all = np.hstack([olr_vals, u850_vals, u200_vals])  # (T, 1080)
    X_train = X_all[train_indices]  # (T_train, 1080)

    # Step 7: EOF analysis via SVD
    if verbose:
        print("=== Step 7: SVD on training matrix (T_train, 1080) ===")

    eof1, eof2, eigenvalues, pc1_std, pc2_std = compute_eofs(X_train)

    # Step 8: Frozen sign/order transform matrix (identity default for J1)
    transform_matrix = np.eye(2, dtype=np.float32)

    # Step 9: Project full dataset (1980–2019)
    if verbose:
        print("=== Step 8: Projecting 1980–2019 data onto basis ===")

    rmm1_all, rmm2_all = project_onto_eofs(
        X_all, eof1, eof2, pc1_std, pc2_std, transform_matrix=transform_matrix
    )
    amp_all = np.sqrt(rmm1_all**2 + rmm2_all**2)
    phase_all = compute_wh_phase(rmm1_all, rmm2_all)

    # Diagnostics on training period
    rmm1_tr = rmm1_all[train_indices]
    rmm2_tr = rmm2_all[train_indices]
    diag = compute_rmm_diagnostics(rmm1_tr, rmm2_tr, eigenvalues)
    if verbose:
        _print_diagnostic_table(diag)

    # Step 10: Save basis and targets
    if verbose:
        print("=== Step 9: Saving rmm_basis.npz and rmm_targets.nc ===")

    basis_path = out_dir / "rmm_basis.npz"
    np.savez_compressed(
        str(basis_path),
        eof1=eof1.astype(np.float32),
        eof2=eof2.astype(np.float32),
        eigenvalues=eigenvalues.astype(np.float32),
        olr_clim=olr_clim.astype(np.float32),
        u850_clim=u850_clim.astype(np.float32),
        u200_clim=u200_clim.astype(np.float32),
        olr_std=np.float32(olr_std),
        u850_std=np.float32(u850_std),
        u200_std=np.float32(u200_std),
        pc1_std=np.float32(pc1_std),
        pc2_std=np.float32(pc2_std),
        transform_matrix=transform_matrix,
        train_years=np.array(TRAIN_YEARS, dtype=np.int32),
        val_years=np.array(VAL_YEARS, dtype=np.int32),
        convention="A",
    )
    if verbose:
        print(f"  Saved basis → {basis_path}")

    # Build split coordinate labels
    times = pd.DatetimeIndex(olr_all.time.values)
    split_arr = np.full(len(times), "train", dtype="<U5")
    val_mask = olr_all.time.dt.year.isin(VAL_YEARS).values
    split_arr[val_mask] = "val"

    ds_out = xr.Dataset(
        {
            "rmm1": xr.DataArray(rmm1_all.astype(np.float32), dims=["time"]),
            "rmm2": xr.DataArray(rmm2_all.astype(np.float32), dims=["time"]),
            "amplitude": xr.DataArray(amp_all.astype(np.float32), dims=["time"]),
            "phase": xr.DataArray(phase_all.astype(np.int8), dims=["time"]),
            "split": xr.DataArray(split_arr, dims=["time"]),
        },
        coords={"time": times},
        attrs={
            "description": (
                "RMM indices derived following Wheeler & Hendon (2004) "
                "per 02_SCIENTIFIC_CONTRACT.md §6"
            ),
            "train_years": f"{TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}",
            "val_years": f"{VAL_YEARS[0]}–{VAL_YEARS[-1]}",
            "test_years": "QUARANTINED (not processed)",
            "tropical_band": f"{LAT_S}° – {LAT_N}° (unweighted meridional mean)",
            "zonal_resolution": "1.0 degree (360 longitudes retained)",
            "variables": "OLR (-mtnlwrf), U850, U200",
            "seasonal_removal": "Mean + 3 annual harmonics fitted per longitude on 1980–2015 only",
            "interannual_removal": "Previous 120-day mean removal under Convention (A)",
            "eof_algorithm": "SVD on (T_train, 1080) matrix (no covariance formed)",
            "pc_normalisation": "Normalized by training-period std to unit variance",
            "active_mjo_threshold": "amplitude > 1.0",
            "convention_120d": "A",
        },
    )

    targets_path = out_dir / "rmm_targets.nc"
    ds_out.to_netcdf(str(targets_path))
    if verbose:
        print(f"  Saved targets → {targets_path}")
        t_start = times[0].strftime("%Y-%m-%d")
        t_end = times[-1].strftime("%Y-%m-%d")
        print(f"  Target record: {t_start} to {t_end} ({len(times)} days)")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def run_smoke_test() -> None:
    """Verify pipeline logic with synthetic propagating wave data (no disk I/O)."""
    print("=== Smoke Test: Verifying pipeline logic with synthetic data ===\n")
    rng = np.random.default_rng(42)

    # 40 years: 36 train (1980–2015), 4 val (2016–2019)
    T_train = 36 * 365
    T_val = 4 * 365
    T_all = T_train + T_val
    times = pd.date_range("1980-01-01", periods=T_all, freq="1D")
    lon = np.linspace(0.5, 359.5, 360)
    lon_rad = np.deg2rad(lon)

    # Synthetic propagating wave (48-day period) + annual cycle
    doy = times.dayofyear.values.astype(float)
    omega_ann = 2.0 * np.pi / 365.25
    annual_cycle = 10.0 + 5.0 * np.cos(omega_ann * doy)[:, None]

    omega_mjo = 2.0 * np.pi / 48.0
    t_idx = np.arange(T_all)[:, None]
    mjo_wave = np.cos(lon_rad[None, :] - omega_mjo * t_idx)

    # Synthesize fields
    olr_data = annual_cycle - 20.0 * mjo_wave + rng.standard_normal((T_all, 360)) * 5.0
    u850_wave = 5.0 * np.sin(lon_rad[None, :] - omega_mjo * t_idx)
    u850_data = annual_cycle * 0.2 + u850_wave + rng.standard_normal((T_all, 360))
    u200_wave = -8.0 * np.sin(lon_rad[None, :] - omega_mjo * t_idx)
    u200_data = annual_cycle * 0.3 + u200_wave + rng.standard_normal((T_all, 360))

    olr_da = xr.DataArray(olr_data, coords={"time": times, "lon": lon}, dims=["time", "lon"])
    u850_da = xr.DataArray(u850_data, coords={"time": times, "lon": lon}, dims=["time", "lon"])
    u200_da = xr.DataArray(u200_data, coords={"time": times, "lon": lon}, dims=["time", "lon"])

    # 1. Fit harmonics on train
    olr_tr = olr_da.isel(time=slice(0, T_train))
    olr_clim, _ = fit_seasonal_harmonics(olr_tr)
    olr_anom = remove_clim(olr_da, olr_clim)

    u850_tr = u850_da.isel(time=slice(0, T_train))
    u850_clim, _ = fit_seasonal_harmonics(u850_tr)
    u850_anom = remove_clim(u850_da, u850_clim)

    u200_tr = u200_da.isel(time=slice(0, T_train))
    u200_clim, _ = fit_seasonal_harmonics(u200_tr)
    u200_anom = remove_clim(u200_da, u200_clim)

    # 2. 120-day mean removal
    olr_filt = remove_120d_mean(olr_anom, convention="A")
    u850_filt = remove_120d_mean(u850_anom, convention="A")
    u200_filt = remove_120d_mean(u200_anom, convention="A")

    # 3. Zonal std
    olr_std = compute_zonal_std(olr_filt.isel(time=slice(0, T_train)))
    u850_std = compute_zonal_std(u850_filt.isel(time=slice(0, T_train)))
    u200_std = compute_zonal_std(u200_filt.isel(time=slice(0, T_train)))

    X_all = np.hstack([
        olr_filt.values / olr_std,
        u850_filt.values / u850_std,
        u200_filt.values / u200_std,
    ])
    X_train = X_all[:T_train]

    eof1, eof2, eigenvalues, pc1_std, pc2_std = compute_eofs(X_train)
    rmm1, rmm2 = project_onto_eofs(X_all, eof1, eof2, pc1_std, pc2_std)

    diag = compute_rmm_diagnostics(rmm1[:T_train], rmm2[:T_train], eigenvalues)
    _print_diagnostic_table(diag)

    assert abs(diag["var_rmm1"] - 1.0) < 1e-3
    assert abs(diag["var_rmm2"] - 1.0) < 1e-3
    assert abs(diag["corr_rmm1_rmm2"]) < 0.05
    print("=== Smoke Test PASSED ===\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute RMM indices (Wheeler & Hendon 2004) from LANL ERA5 data."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Root of LANL NERSC data tree.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for rmm_targets.nc and rmm_basis.npz.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on synthetic data only; do not read CFS files.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
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
            "       Use --smoke-test to verify pipeline without CFS data.",
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
