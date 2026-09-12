"""Pure RMM computation functions (Wheeler & Hendon 2004).

Extracted from scripts/compute_rmm.py per task C3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# Tropical averaging band (Wheeler & Hendon 2004)
LAT_S = -15.0  # 15°S
LAT_N = 15.0  # 15°N

# Wheeler-Hendon phase lookup table
# Phase boundaries are in degrees of the RMM phase angle.
# Phase 1 corresponds to the angle range [90°, 135°), etc.
# We use the standard WH 8-phase definition.
_WH_PHASE_EDGES = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360])


def tropical_mean(da: xr.DataArray) -> xr.DataArray:
    """Cosine-latitude-weighted mean over [LAT_S, LAT_N] and all longitudes.

    Returns a 1-D DataArray indexed by 'time'.
    """
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"

    da_trop = da.sel({lat_name: slice(LAT_N, LAT_S)})  # lat decreasing
    # If lat is increasing, slice the other way:
    if da_trop.sizes[lat_name] == 0:
        da_trop = da.sel({lat_name: slice(LAT_S, LAT_N)})

    weights = np.cos(np.deg2rad(da_trop[lat_name]))
    weights = weights / weights.sum()

    # Weighted mean over lat, then simple mean over lon
    da_lat = (da_trop * weights).sum(lat_name)
    da_mean = da_lat.mean(lon_name)
    return da_mean


def to_daily_mean(da: xr.DataArray) -> xr.DataArray:
    """Resample 6-hourly tropical-mean series to daily mean."""
    return da.resample(time="1D").mean()


def compute_daily_clim(da_daily: xr.DataArray) -> xr.DataArray:
    """Compute the day-of-year mean climatology from a 1-D daily DataArray.

    Returns a DataArray indexed by day-of-year (1–366) representing the
    training-period mean for each calendar day.
    """
    return da_daily.groupby("time.dayofyear").mean("time")


def remove_clim(da_daily: xr.DataArray, clim: xr.DataArray) -> xr.DataArray:
    """Subtract climatology from daily series → anomalies."""
    return da_daily.groupby("time.dayofyear") - clim


def compute_eofs(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute EOFs of a 2-D training matrix via covariance-matrix eigenvectors.

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
    cov = np.cov(X_centred, rowvar=False)  # shape (N, N)

    # eigh returns eigenvalues in ascending order for symmetric matrices
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Reverse to descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]  # columns are eigenvectors

    eof1 = eigenvectors[:, 0]
    eof2 = eigenvectors[:, 1]
    return eof1, eof2, eigenvalues


def project_onto_eofs(
    X: np.ndarray, eof1: np.ndarray, eof2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project combined anomaly matrix X (T, N) onto EOF1 and EOF2.

    Returns RMM1 (T,) and RMM2 (T,) as 1-D arrays.
    """
    rmm1 = X @ eof1  # dot product of each row with EOF1
    rmm2 = X @ eof2
    return rmm1, rmm2


def compute_wh_phase(rmm1: np.ndarray, rmm2: np.ndarray) -> np.ndarray:
    """Assign Wheeler-Hendon phases 1–8 from RMM1/RMM2.

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


def build_combined_vector(
    olr_daily: xr.DataArray,
    u850_daily: xr.DataArray,
    u200_daily: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Align three 1-D daily DataArrays on common time axis.

    Returns aligned (olr, u850, u200) DataArrays.
    """
    common_times = (
        pd.DatetimeIndex(olr_daily.indexes["time"])
        .intersection(u850_daily.indexes["time"])
        .intersection(u200_daily.indexes["time"])
    )

    olr_a = olr_daily.sel(time=common_times)
    u850_a = u850_daily.sel(time=common_times)
    u200_a = u200_daily.sel(time=common_times)
    return olr_a, u850_a, u200_a
