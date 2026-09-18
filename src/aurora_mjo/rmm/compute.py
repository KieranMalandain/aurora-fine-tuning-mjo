"""Pure RMM computation functions (Wheeler & Hendon 2004).

Rebuilt in task J1 to conform to 02_SCIENTIFIC_CONTRACT.md §6:
- a(t) ∈ ℝ^(3·N_λ) with N_λ = 360 at 1° regular resolution (retaining zonal structure).
- Unweighted meridional mean over 15°S–15°N (simple mean matching WH).
- Seasonal cycle as mean plus first three harmonics of the annual cycle per longitude.
- 120-day mean removal under Convention (A) (observations up to t₀, held fixed).
- Zonal-mean-σ normalisation (one scalar per field).
- SVD on (T, 1080) matrix (no 1080x1080 covariance matrix formed).
- PC normalisation to unit variance on training period.
- Frozen sign/order transform matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# Tropical averaging band (Wheeler & Hendon 2004)
LAT_S = -15.0  # 15°S
LAT_N = 15.0  # 15°N

# Number of longitude points at 1° regular resolution
N_LON = 360

# Wheeler-Hendon phase lookup table
_WH_PHASE_EDGES = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360])


def tropical_mean(da: xr.DataArray) -> xr.DataArray:
    """Unweighted meridional mean over [LAT_S, LAT_N] retaining zonal structure.

    Wheeler & Hendon (2004) step 3:
    Meridional average over 15°S–15°N as a simple unweighted mean (no cos-latitude
    weighting). Retains longitude dimension so output has shape (..., N_λ), where
    N_λ = 360 at 1° resolution.

    Parameters
    ----------
    da : xr.DataArray with latitude and longitude coordinates.

    Returns
    -------
    xr.DataArray : DataArray averaged across latitudes in [LAT_S, LAT_N],
                   retaining the longitude dimension (360 points).
    """
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"

    # Select latitude band [-15, 15] in an order-invariant manner
    mask = (da[lat_name] >= LAT_S) & (da[lat_name] <= LAT_N)
    da_trop = da.isel({lat_name: np.where(mask.values)[0]})

    # Wheeler & Hendon (2004): simple unweighted mean over latitude
    da_meridional = da_trop.mean(dim=lat_name)

    # Standardize coordinate name to 'lon' if needed
    if lon_name != "lon" and "lon" not in da_meridional.coords:
        da_meridional = da_meridional.rename({lon_name: "lon"})

    return da_meridional


def to_daily_mean(da: xr.DataArray) -> xr.DataArray:
    """Resample 6-hourly tropical-mean series to daily mean."""
    return da.resample(time="1D").mean()


def fit_seasonal_harmonics(
    da_daily: xr.DataArray, n_harmonics: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Fit annual cycle as mean plus first three harmonics per longitude.

    Wheeler & Hendon (2004) step 4:
    Mean plus the first three harmonics of the annual cycle, fitted per longitude
    on the training period only (1980–2015).

    Parameters
    ----------
    da_daily : shape (T_train, N_lon) daily DataArray on training period.
    n_harmonics : number of annual cycle harmonics (default: 3).

    Returns
    -------
    clim_table : shape (366, N_lon) climatological mean for each day of year (1–366).
    coeffs : shape (1 + 2 * n_harmonics, N_lon) regression coefficients.
    """
    doy = da_daily.time.dt.dayofyear.values.astype(np.float64)  # (T,)
    y = da_daily.values  # (T, N_lon)
    n_times, n_lon = y.shape

    # Annual period in days
    t0 = 365.25

    # Design matrix: [1, cos(1*w*d), sin(1*w*d), ..., cos(k*w*d), sin(k*w*d)]
    cols: list[np.ndarray] = [np.ones((n_times, 1), dtype=np.float64)]
    for k in range(1, n_harmonics + 1):
        omega_k = 2.0 * np.pi * k / t0
        cols.append(np.cos(omega_k * doy)[:, None])
        cols.append(np.sin(omega_k * doy)[:, None])
    A = np.hstack(cols)  # (T, 1 + 2*n_harmonics)

    # Solve least-squares for each longitude
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)  # (1 + 2*n_harmonics, N_lon)

    # Evaluate on DOY 1..366 to build fast lookup table
    doy_eval = np.arange(1, 367, dtype=np.float64)
    eval_cols: list[np.ndarray] = [np.ones((366, 1), dtype=np.float64)]
    for k in range(1, n_harmonics + 1):
        omega_k = 2.0 * np.pi * k / t0
        eval_cols.append(np.cos(omega_k * doy_eval)[:, None])
        eval_cols.append(np.sin(omega_k * doy_eval)[:, None])
    A_eval = np.hstack(eval_cols)
    clim_table = A_eval @ coeffs  # (366, N_lon)

    return clim_table, coeffs


def compute_daily_clim(da_daily: xr.DataArray) -> xr.DataArray:
    """Compute mean + 3 harmonics annual climatology on training daily series."""
    clim_table, _ = fit_seasonal_harmonics(da_daily, n_harmonics=3)
    lon_name = (
        "lon"
        if "lon" in da_daily.dims
        else ("longitude" if "longitude" in da_daily.dims else str(da_daily.dims[-1]))
    )
    return xr.DataArray(
        clim_table,
        coords={"dayofyear": np.arange(1, 367), lon_name: da_daily[lon_name]},
        dims=["dayofyear", lon_name],
    )


def remove_clim(da_daily: xr.DataArray, clim: np.ndarray | xr.DataArray) -> xr.DataArray:
    """Subtract daily seasonal cycle climatology from daily series → anomalies."""
    doy = da_daily.time.dt.dayofyear.values  # 1-indexed (1..366)
    if isinstance(clim, xr.DataArray):
        clim_vals = clim.sel(dayofyear=doy).values
    else:
        clim_vals = clim[doy - 1]
    anom_vals = da_daily.values - clim_vals
    return xr.DataArray(anom_vals, coords=da_daily.coords, dims=da_daily.dims)


def remove_120d_mean(
    da_anom: xr.DataArray,
    convention: str = "A",
    obs_history_ending_at_t0: xr.DataArray | None = None,
) -> xr.DataArray:
    """Subtract the 120-day mean to remove interannual (ENSO) variability (Wheeler & Hendon 2004).

    Convention (A) (02_SCIENTIFIC_CONTRACT.md §6.2):
    - For continuous historical observations: rolling 120-day mean of anomalies
      ending at the valid time t (min_periods=1 to handle record start without NaNs).
      No future data is ever read.
    - For a forecast initialised at t₀: the 120-day mean is computed strictly from
      observed data ending at t₀ (obs_history in [t₀ - 120d, t₀]), containing no data
      after t₀, and held fixed across all forecast leads τ.
    """
    if obs_history_ending_at_t0 is not None:
        # Convention (A) forecast path: mean of up to 120 days of observations up to t₀
        # held fixed across the forecast
        n_days = min(120, len(obs_history_ending_at_t0.time))
        prior_mean = obs_history_ending_at_t0.isel(time=slice(-n_days, None)).mean(dim="time")
        return da_anom - prior_mean
    else:
        # Continuous historical observation path: rolling 120-day mean ending at valid time
        # min_periods=1 ensures no NaNs at record onset (1980)
        rolling_mean = da_anom.rolling(time=120, min_periods=1).mean()
        return da_anom - rolling_mean


def compute_zonal_std(da_filtered: xr.DataArray) -> float:
    """Compute zonally averaged temporal standard deviation (one scalar per field).

    Wheeler & Hendon (2004) step 6:
    Divide each field by its zonally averaged temporal standard deviation from the
    training period.
    """
    lon_name = str(
        "lon"
        if "lon" in da_filtered.dims
        else ("longitude" if "longitude" in da_filtered.dims else da_filtered.dims[-1])
    )
    std_per_lon = da_filtered.std(dim="time")
    zonal_std = float(std_per_lon.mean(dim=lon_name))
    return zonal_std


def compute_eofs(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Compute EOFs of a (T_train, 1080) matrix via SVD without forming covariance.

    Wheeler & Hendon (2004) steps 8 & 9:
    - SVD of the (T_train, 1080) training matrix.
    - No 1080x1080 covariance matrix is formed.
    - Computes PC training standard deviations for unit-variance normalisation.

    Parameters
    ----------
    X_train : shape (T, N) where N = 3 * N_lon = 1080.

    Returns
    -------
    eof1 : shape (1080,) — first EOF
    eof2 : shape (1080,) — second EOF
    eigenvalues : shape (1080,) — descending eigenvalues
    pc1_std : float — training standard deviation of PC1
    pc2_std : float — training standard deviation of PC2
    """
    # Remove time-mean
    X_centred = X_train - X_train.mean(axis=0, keepdims=True)

    # SVD: X = U @ diag(S) @ Vt
    # Vt has shape (1080, 1080); rows of Vt are right singular vectors (EOFs)
    U, S, Vt = np.linalg.svd(X_centred, full_matrices=False)

    eof1 = Vt[0]  # shape (1080,)
    eof2 = Vt[1]  # shape (1080,)

    n_samples = len(X_train)
    eigenvalues = (S**2) / max(1, n_samples - 1)

    # Step 9: PC normalisation constants to ensure unit variance on training set
    pc1_raw = X_centred @ eof1
    pc2_raw = X_centred @ eof2
    pc1_std = float(np.std(pc1_raw))
    pc2_std = float(np.std(pc2_raw))

    return eof1, eof2, eigenvalues, pc1_std, pc2_std


def project_onto_eofs(
    X: np.ndarray,
    eof1: np.ndarray,
    eof2: np.ndarray,
    pc1_std: float = 1.0,
    pc2_std: float = 1.0,
    transform_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project combined anomaly matrix X (T, 1080) onto EOF1 and EOF2.

    Applies PC unit-variance normalisation and the frozen sign/order transform.

    Parameters
    ----------
    X : shape (T, 1080)
    eof1, eof2 : shape (1080,)
    pc1_std, pc2_std : training standard deviations
    transform_matrix : shape (2, 2), frozen sign/order transform (default: I)

    Returns
    -------
    rmm1, rmm2 : shape (T,)
    """
    pc1 = (X @ eof1) / (pc1_std + 1e-12)
    pc2 = (X @ eof2) / (pc2_std + 1e-12)

    if transform_matrix is None:
        transform_matrix = np.eye(2, dtype=np.float32)

    pcs = np.column_stack([pc1, pc2])  # (T, 2)
    rmm = pcs @ transform_matrix.T
    rmm1 = rmm[:, 0]
    rmm2 = rmm[:, 1]
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
    _SECTOR_TO_PHASE = [7, 8, 1, 2, 3, 4, 5, 6]
    sector = (angle_deg // 45).astype(int)
    phase = np.array([_SECTOR_TO_PHASE[s] for s in sector])

    # Unclassified phase 0 if amplitude < 1.0
    amp = np.sqrt(rmm1**2 + rmm2**2)
    phase[amp < 1.0] = 0
    return phase


def build_combined_vector(
    olr_daily: xr.DataArray,
    u850_daily: xr.DataArray,
    u200_daily: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Align three 2-D daily DataArrays (time, lon) on common time axis."""
    common_times = (
        pd.DatetimeIndex(olr_daily.indexes["time"])
        .intersection(u850_daily.indexes["time"])
        .intersection(u200_daily.indexes["time"])
    )

    olr_a = olr_daily.sel(time=common_times)
    u850_a = u850_daily.sel(time=common_times)
    u200_a = u200_daily.sel(time=common_times)
    return olr_a, u850_a, u200_a
