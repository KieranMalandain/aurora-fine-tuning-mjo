"""Pure RMM evaluation functions and metrics (Wheeler & Hendon 2004).

Extracted from scripts/evaluate_mjo.py per task C3.
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
    olr_trop: float,
    u850_trop: float,
    u200_trop: float,
    doy: int,
    basis: dict[str, Any],
) -> tuple[float, float]:
    """Project a single set of daily tropical-mean anomalies onto the frozen EOF basis.

    Parameters
    ----------
    olr_trop, u850_trop, u200_trop : tropical-mean values for that day
    doy   : day of year (1–366)
    basis : loaded from load_basis()

    Returns
    -------
    rmm1, rmm2 : float
    """
    # Find climatology index (DOY array may not start at 1)
    doy_arr = basis["doy"]
    idx = np.searchsorted(doy_arr, doy)
    idx = np.clip(idx, 0, len(doy_arr) - 1)

    olr_anom = (olr_trop - basis["olr_clim"][idx]) / (basis["olr_std"] + 1e-8)
    u850_anom = (u850_trop - basis["u850_clim"][idx]) / (basis["u850_std"] + 1e-8)
    u200_anom = (u200_trop - basis["u200_clim"][idx]) / (basis["u200_std"] + 1e-8)

    x = np.array([olr_anom, u850_anom, u200_anom])
    rmm1 = float(x @ basis["eof1"])
    rmm2 = float(x @ basis["eof2"])
    return rmm1, rmm2


def bivariate_acc(
    rmm1_fc: np.ndarray, rmm2_fc: np.ndarray, rmm1_ob: np.ndarray, rmm2_ob: np.ndarray
) -> float:
    """Bivariate Anomaly Correlation Coefficient (Wheeler & Hendon 2004).

    ACC = Σ(rmm1_f·rmm1_o + rmm2_f·rmm2_o)
          / sqrt[ Σ(rmm1_f²+rmm2_f²) · Σ(rmm1_o²+rmm2_o²) ]

    Both arrays must have the same length N (paired forecast/observation).
    Returns NaN if fewer than 2 valid pairs.
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
    """Mean absolute phase error in degrees.

    Phase angle = atan2(RMM2, RMM1).
    Wraps difference to [-180, 180].
    """
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

    # Tropical mask
    trop = (lat >= LAT_S) & (lat <= LAT_N)
    if not trop.any():
        return None, None, None

    # Cosine-latitude weights for tropical band
    lat_trop = lat[trop]
    weights = np.cos(np.deg2rad(lat_trop))
    weights /= weights.sum()

    def _trop_mean_surf(var_key: str) -> float:
        """Return scalar tropical mean of the last time step of a surf var."""
        t = surf[var_key][0, -1].cpu().numpy()  # (H, W)
        t_trop = t[trop, :]  # (n_trop, W)
        return float((t_trop * weights[:, None]).sum(axis=0).mean())

    def _trop_mean_pressure(var_key: str, plevel_hpa: int) -> float | None:
        """Return scalar tropical mean at a given pressure level, last step."""
        t = atmos[var_key][0, -1]  # (C, H, W)
        if plevel_hpa not in levels:
            idx = np.argmin(np.abs(levels - plevel_hpa))
        else:
            idx = int(np.where(levels == plevel_hpa)[0][0])
        t_lev = t[idx].cpu().numpy()  # (H, W)
        t_trop = t_lev[trop, :]
        return float((t_trop * weights[:, None]).sum(axis=0).mean())

    olr_val = _trop_mean_surf(olr_key)
    u850_val = _trop_mean_pressure("u", 850)
    u200_val = _trop_mean_pressure("u", 200)

    if u850_val is None or u200_val is None:
        return None, None, None

    doy = valid_time.timetuple().tm_yday
    rmm1, rmm2 = project_fields_to_rmm(olr_val, u850_val, u200_val, doy, basis)
    amp = float(np.sqrt(rmm1**2 + rmm2**2))
    return rmm1, rmm2, amp


def _advance_time(t: datetime, n_steps: int, step_hrs: int = STEP_HRS) -> datetime:
    return t + timedelta(hours=n_steps * step_hrs)
