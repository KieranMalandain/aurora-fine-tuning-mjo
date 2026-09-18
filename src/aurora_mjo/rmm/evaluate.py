"""Pure RMM evaluation functions and metrics (Wheeler & Hendon 2004).

Rebuilt in task J1 to eliminate scalar collapse and conform to 02_SCIENTIFIC_CONTRACT.md §6:
- Tropical averaging preserves the (longitude,) vector (360 points at 1°).
- Projection uses the full (1080,) combined vector with unit-variance PC normalisation.
- Supports Convention (A) fixed 120-day mean removal.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import torch

# Tropical averaging band (Wheeler & Hendon 2004)
LAT_S = -15.0  # 15°S
LAT_N = 15.0  # 15°N
STEP_HRS = 6  # 6-hourly time steps


def project_fields_to_rmm(
    olr_trop: np.ndarray,
    u850_trop: np.ndarray,
    u200_trop: np.ndarray,
    doy: int,
    basis: dict[str, Any],
    obs_120d_mean: np.ndarray | dict[str, np.ndarray] | None = None,
) -> tuple[float, float]:
    """Project daily tropical-mean profiles (shape 360) onto the frozen EOF basis.

    Parameters
    ----------
    olr_trop, u850_trop, u200_trop : np.ndarray of shape (360,)
        Tropical-mean field profiles along longitude for that day.
    doy : int
        Day of year (1–366).
    basis : dict[str, Any]
        Loaded from rmm_basis.npz.
    obs_120d_mean : np.ndarray | dict[str, np.ndarray] | None
        Under Convention (A), 120-day mean profile computed from observations
        up to t₀ and held fixed across the forecast.

    Returns
    -------
    rmm1, rmm2 : float
    """
    # Climatology lookup (day of year 1..366)
    clim_len = len(basis["olr_clim"])
    idx = int(np.clip(doy - 1, 0, clim_len - 1))

    # 1. Seasonal anomaly
    olr_anom = olr_trop - basis["olr_clim"][idx]
    u850_anom = u850_trop - basis["u850_clim"][idx]
    u200_anom = u200_trop - basis["u200_clim"][idx]

    # 2. Convention (A) 120-day mean removal
    if obs_120d_mean is not None:
        if isinstance(obs_120d_mean, dict):
            if "olr" in obs_120d_mean:
                olr_anom = olr_anom - obs_120d_mean["olr"]
            if "u850" in obs_120d_mean:
                u850_anom = u850_anom - obs_120d_mean["u850"]
            if "u200" in obs_120d_mean:
                u200_anom = u200_anom - obs_120d_mean["u200"]
        elif isinstance(obs_120d_mean, np.ndarray) and len(obs_120d_mean) == 1080:
            olr_anom = olr_anom - obs_120d_mean[0:360]
            u850_anom = u850_anom - obs_120d_mean[360:720]
            u200_anom = u200_anom - obs_120d_mean[720:1080]

    # 3. Zonal-mean-std normalisation (scalar per field)
    olr_raw = basis["olr_std"]
    olr_std = float(olr_raw.item() if hasattr(olr_raw, "item") else olr_raw)
    u850_raw = basis["u850_std"]
    u850_std = float(u850_raw.item() if hasattr(u850_raw, "item") else u850_raw)
    u200_raw = basis["u200_std"]
    u200_std = float(u200_raw.item() if hasattr(u200_raw, "item") else u200_raw)

    olr_norm = olr_anom / (olr_std + 1e-8)
    u850_norm = u850_anom / (u850_std + 1e-8)
    u200_norm = u200_anom / (u200_std + 1e-8)

    # 4. Concatenate into 1080-vector
    x = np.concatenate([olr_norm, u850_norm, u200_norm])

    # 5. Project onto EOFs and normalise to unit variance
    pc1_std = float(basis.get("pc1_std", 1.0))
    pc2_std = float(basis.get("pc2_std", 1.0))

    pc1 = float(x @ basis["eof1"]) / (pc1_std + 1e-12)
    pc2 = float(x @ basis["eof2"]) / (pc2_std + 1e-12)

    # 6. Apply frozen sign/order transform
    transform_matrix = basis.get("transform_matrix", np.eye(2, dtype=np.float32))
    rmm = np.array([pc1, pc2]) @ transform_matrix.T

    rmm1 = float(rmm[0])
    rmm2 = float(rmm[1])
    return rmm1, rmm2


def bivariate_acc(
    rmm1_fc: np.ndarray, rmm2_fc: np.ndarray, rmm1_ob: np.ndarray, rmm2_ob: np.ndarray
) -> float:
    """Bivariate Anomaly Correlation Coefficient (Wheeler & Hendon 2004).

    ACC = Σ(rmm1_f·rmm1_o + rmm2_f·rmm2_o)
          / sqrt[ Σ(rmm1_f²+rmm2_f²) · Σ(rmm1_o²+rmm2_o²) ]
    """
    n = len(rmm1_fc)
    if n < 2:
        return np.nan
    cov = np.sum(rmm1_fc * rmm1_ob + rmm2_fc * rmm2_ob)
    var_fc = np.sum(rmm1_fc**2 + rmm2_fc**2)
    var_ob = np.sum(rmm1_ob**2 + rmm2_ob**2)
    denom = np.sqrt(var_fc * var_ob)
    if denom < 1e-12:
        return np.nan
    return float(cov / denom)


def rmse_pair(fc: np.ndarray, ob: np.ndarray) -> float:
    """Root mean squared error between two equal-length arrays."""
    if len(fc) < 1:
        return np.nan
    return float(np.sqrt(np.mean((fc - ob) ** 2)))


def amplitude_error(amp_fc: np.ndarray, amp_ob: np.ndarray) -> float:
    """Mean signed amplitude error (bias): E[amp_fc - amp_ob]."""
    if len(amp_fc) < 1:
        return np.nan
    return float(np.mean(amp_fc - amp_ob))


def phase_error_deg(
    rmm1_fc: np.ndarray, rmm2_fc: np.ndarray, rmm1_ob: np.ndarray, rmm2_ob: np.ndarray
) -> float:
    """Mean absolute phase error in degrees."""
    if len(rmm1_fc) < 1:
        return np.nan
    angle_fc = np.degrees(np.arctan2(rmm2_fc, rmm1_fc))
    angle_ob = np.degrees(np.arctan2(rmm2_ob, rmm1_ob))
    diff = angle_fc - angle_ob
    # Wrap to [-180, 180]
    diff = (diff + 180) % 360 - 180
    return float(np.mean(np.abs(diff)))


def extract_rmm_from_mjo_head(
    mjo_pred: torch.Tensor,  # (B, 3)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract RMM1, RMM2, Amplitude from MJO head output tensor."""
    arr = mjo_pred.detach().cpu().numpy()  # (B, 3)
    rmm1 = arr[:, 0]
    rmm2 = arr[:, 1]
    amp = np.sqrt(rmm1**2 + rmm2**2)
    return rmm1, rmm2, amp


def extract_rmm_from_fields(
    pred_batch: Any,
    valid_time: datetime,
    basis: dict[str, Any],
    lat_name: str = "lat",
    obs_120d_mean: np.ndarray | dict[str, np.ndarray] | None = None,
) -> tuple[float, float, float] | tuple[None, None, None]:
    """Fallback: project needed OLR / U850 / U200 fields onto frozen EOF basis.

    Returns (rmm1, rmm2, amplitude) for a single sample or (None, None, None)
    if the required fields are missing.
    """
    surf = pred_batch.surf_vars
    atmos = pred_batch.atmos_vars
    meta = pred_batch.metadata

    # --- OLR fallback variable names (mtnlwrf or ttr) ---
    olr_key = None
    for k in ("mtnlwrf", "ttr"):
        if k in surf:
            olr_key = k
            break

    has_u = "u" in atmos
    if olr_key is None or not has_u:
        return None, None, None

    lat = meta.lat.cpu().numpy()  # (H,)
    levels = np.array(meta.atmos_levels)

    # Tropical mask [15°S, 15°N]
    trop = (lat >= LAT_S) & (lat <= LAT_N)
    if not trop.any():
        return None, None, None

    def _trop_mean_surf(var_key: str) -> np.ndarray:
        """Return tropical-mean 1D array along longitude of the last time step of a surf var."""
        t = surf[var_key][0, -1].cpu().numpy()  # (H, W)
        t_trop = t[trop, :]  # (n_trop, W)
        # Wheeler & Hendon (2004) unweighted meridional mean over 15°S–15°N retaining longitude
        return t_trop.mean(axis=0)

    def _trop_mean_pressure(var_key: str, plevel_hpa: int) -> np.ndarray | None:
        """Return tropical-mean 1D array along longitude at a given pressure level, last step."""
        t = atmos[var_key][0, -1]  # (C, H, W)
        if plevel_hpa not in levels:
            idx = int(np.argmin(np.abs(levels - plevel_hpa)))
        else:
            idx = int(np.where(levels == plevel_hpa)[0][0])
        t_lev = t[idx].cpu().numpy()  # (H, W)
        t_trop = t_lev[trop, :]
        return t_trop.mean(axis=0)

    olr_val = _trop_mean_surf(olr_key)
    # OLR sign convention: OLR = -mtnlwrf (mtnlwrf is negative net downward longwave flux)
    if olr_key in ("mtnlwrf", "ttr"):
        olr_val = -olr_val

    u850_val = _trop_mean_pressure("u", 850)
    u200_val = _trop_mean_pressure("u", 200)

    if u850_val is None or u200_val is None:
        return None, None, None

    doy = valid_time.timetuple().tm_yday
    rmm1, rmm2 = project_fields_to_rmm(
        olr_val, u850_val, u200_val, doy, basis, obs_120d_mean=obs_120d_mean
    )
    amp = float(np.sqrt(rmm1**2 + rmm2**2))
    return rmm1, rmm2, amp


def _advance_time(t: datetime, n_steps: int, step_hrs: int = STEP_HRS) -> datetime:
    return t + timedelta(hours=n_steps * step_hrs)
