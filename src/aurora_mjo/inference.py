"""Autoregressive rollout and inference engine for Aurora MJO forecasting.

Task J3 (02_SCIENTIFIC_CONTRACT.md §5, §6; tasks/J3_skill_harness.md):
- 120-step (30 days at 6-hourly resolution) autoregressive forecast rollout.
- Advances the 2-step history window forward with model predictions.
- Enforces physical clamping on fed-back surface predictions to prevent drift.
- Aggregates 6-hourly predictions to daily resolution for Wheeler & Hendon (2004) RMM projection.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import torch
from aurora import Batch, Metadata

# Physical range bounds for fed-back surface variables in multi-step rollouts.
# Preserves numerical stability over 120-step rollouts without distorting physical signals.
_ROLLOUT_CLAMP: dict[str, tuple[float, float]] = {
    "msl": (3.0e4, 1.1e5),
    "2t": (150.0, 350.0),
    "10u": (-120.0, 120.0),
    "10v": (-120.0, 120.0),
    "ttr": (-600.0, 50.0),
}


def clamp_fed(name: str, t: torch.Tensor) -> torch.Tensor:
    """Clamp fed-back prediction tensor to physical sanity bounds, if configured."""
    b = _ROLLOUT_CLAMP.get(name)
    return t.clamp(*b) if (b is not None and isinstance(t, torch.Tensor)) else t


def advance_batch(in_batch: Batch, pred_batch: Batch, step_index: int) -> Batch:
    """Roll the 2-step history window forward with the model prediction.

    Under Aurora's 2-step input convention, [t - 6h, t] becomes [t, t + 6h].
    The fed-back prediction is clamped to physical bounds for surface variables.
    Static variables are preserved unchanged.
    """
    dt = timedelta(hours=6)

    new_surf: dict[str, torch.Tensor] = {}
    for k, hist in in_batch.surf_vars.items():
        pred_k = pred_batch.surf_vars.get(k)
        if pred_k is not None:
            # pred_k is (B, 1, H, W) or (B, H, W)
            if pred_k.ndim == 3:
                pred_k = pred_k.unsqueeze(1)
            fed = clamp_fed(k, pred_k)
            new_surf[k] = torch.cat([hist[:, 1:], fed], dim=1)
        else:
            new_surf[k] = torch.cat([hist[:, 1:], hist[:, -1:]], dim=1)

    new_atmos: dict[str, torch.Tensor] = {}
    for k, hist in in_batch.atmos_vars.items():
        pred_k = pred_batch.atmos_vars.get(k)
        if pred_k is not None:
            if pred_k.ndim == 4:
                pred_k = pred_k.unsqueeze(1)
            new_atmos[k] = torch.cat([hist[:, 1:], pred_k], dim=1)
        else:
            new_atmos[k] = torch.cat([hist[:, 1:], hist[:, -1:]], dim=1)

    new_time = tuple(t + dt for t in in_batch.metadata.time)
    new_metadata = Metadata(
        lat=in_batch.metadata.lat,
        lon=in_batch.metadata.lon,
        time=new_time,
        atmos_levels=in_batch.metadata.atmos_levels,
        rollout_step=step_index + 1,
    )
    return Batch(
        surf_vars=new_surf,
        static_vars=in_batch.static_vars,
        atmos_vars=new_atmos,
        metadata=new_metadata,
    )


def extract_level_field(
    atmos_tensor: torch.Tensor,
    levels: tuple[int | float, ...] | list[int | float] | np.ndarray,
    target_plevel_hpa: int,
) -> np.ndarray:
    """Extract a 2D (H, W) field at a specified pressure level from an atmospheric tensor.

    Parameters
    ----------
    atmos_tensor : torch.Tensor of shape (B, 1, C, H, W) or (B, C, H, W) or (C, H, W)
    levels : sequence of pressure levels in hPa corresponding to axis C
    target_plevel_hpa : target level (e.g. 850 or 200)

    Returns
    -------
    np.ndarray of shape (H, W) for batch element 0.
    """
    levels_arr = np.asarray(levels)
    if target_plevel_hpa in levels_arr:
        idx = int(np.where(levels_arr == target_plevel_hpa)[0][0])
    else:
        idx = int(np.argmin(np.abs(levels_arr - target_plevel_hpa)))

    t = atmos_tensor.detach().cpu()
    # Normalize dimensions: find the channel axis
    if t.ndim == 5:  # (B, 1, C, H, W)
        field = t[0, -1, idx]
    elif t.ndim == 4:  # (B, C, H, W)
        field = t[0, idx]
    elif t.ndim == 3:  # (C, H, W)
        field = t[idx]
    else:
        raise ValueError(f"Unexpected atmos tensor shape: {t.shape}")
    return field.numpy().astype(np.float32)


def rollout_forecast(
    model: torch.nn.Module,
    in_batch: Batch,
    steps: int = 120,
    device: torch.device | None = None,
) -> Generator[tuple[int, datetime, Batch, Any], None, None]:
    """Execute an autoregressive rollout of N steps from an initial condition.

    Parameters
    ----------
    model : Aurora or AuroraMJO model instance.
    in_batch : initial Batch at t0 (with 2-step history [t0-6h, t0]).
    steps : total number of 6-hour rollout steps (default: 120 for 30 days).
    device : target device (CPU or CUDA).

    Yields
    ------
    (step_idx, valid_time, pred_batch, mjo_pred) for step_idx in 1..steps.
    """
    if device is not None:
        in_batch = in_batch.to(device)

    model.eval()
    current_batch = in_batch
    init_time = in_batch.metadata.time[-1]

    with torch.no_grad():
        for step in range(1, steps + 1):
            out = model(current_batch)
            if isinstance(out, tuple):
                pred_batch, mjo_pred = out
            else:
                pred_batch, mjo_pred = out, None

            valid_time = init_time + timedelta(hours=6 * step)
            yield (step, valid_time, pred_batch, mjo_pred)

            current_batch = advance_batch(current_batch, pred_batch, step_index=step)


def run_rollout_daily(
    model: torch.nn.Module,
    in_batch: Batch,
    max_lead_days: int = 30,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Execute 120-step rollout and aggregate predictions to daily resolution.

    Wheeler & Hendon (2004) operates on daily fields (daily means of 4 6-hourly steps).
    This function groups steps 1..120 into days 1..30 and computes the daily mean for
    surface variables and targeted pressure levels (u850, u200).

    Returns
    -------
    dict with:
        "lead_days": list[int] 1..max_lead_days
        "valid_dates": list[datetime] valid dates for each lead day
        "surf_daily": dict[var_name, np.ndarray of shape (max_lead_days, H, W)]
        "u850_daily": np.ndarray of shape (max_lead_days, H, W)
        "u200_daily": np.ndarray of shape (max_lead_days, H, W)
        "surf_step_24h": instantaneous snapshot at 24h intervals (shape (max_lead_days, H, W))
        "u850_step_24h": instantaneous snapshot at 24h intervals (shape (max_lead_days, H, W))
        "u200_step_24h": instantaneous snapshot at 24h intervals (shape (max_lead_days, H, W))
        "mjo_head_daily": np.ndarray of shape (max_lead_days, 3) if present, else None
    """
    steps = max_lead_days * 4
    levels = in_batch.metadata.atmos_levels
    init_time = in_batch.metadata.time[-1]

    # Temporary storage for all 6-hourly steps
    step_surf: dict[str, list[np.ndarray]] = {k: [] for k in in_batch.surf_vars}
    step_u850: list[np.ndarray] = []
    step_u200: list[np.ndarray] = []
    step_mjo_pred: list[np.ndarray] = []
    has_mjo_pred = False

    for _step, _vtime, pred_batch, mjo_pred in rollout_forecast(
        model, in_batch, steps=steps, device=device
    ):
        for k in in_batch.surf_vars:
            t = pred_batch.surf_vars[k].detach().cpu()
            # Extract 2D field (H, W) for batch element 0, last step
            arr = (
                t[0, -1].numpy().astype(np.float32)
                if t.ndim >= 3
                else t.numpy().astype(np.float32)
            )
            step_surf[k].append(arr)

        if "u" in pred_batch.atmos_vars:
            u_arr = pred_batch.atmos_vars["u"]
            step_u850.append(extract_level_field(u_arr, levels, 850))
            step_u200.append(extract_level_field(u_arr, levels, 200))

        if mjo_pred is not None:
            has_mjo_pred = True
            step_mjo_pred.append(mjo_pred.detach().cpu().numpy()[0])

    # Aggregate 4 steps per day into daily means
    daily_surf: dict[str, np.ndarray] = {}
    snap_surf: dict[str, np.ndarray] = {}
    for k, steps_list in step_surf.items():
        daily_means = []
        snapshots = []
        for d in range(max_lead_days):
            day_steps = steps_list[d * 4 : (d + 1) * 4]
            daily_means.append(np.mean(np.stack(day_steps, axis=0), axis=0))
            snapshots.append(day_steps[-1])  # 24h instantaneous step
        daily_surf[k] = np.stack(daily_means, axis=0)
        snap_surf[k] = np.stack(snapshots, axis=0)

    daily_u850_list, snap_u850_list = [], []
    daily_u200_list, snap_u200_list = [], []
    for d in range(max_lead_days):
        day_u850 = step_u850[d * 4 : (d + 1) * 4]
        day_u200 = step_u200[d * 4 : (d + 1) * 4]
        daily_u850_list.append(np.mean(np.stack(day_u850, axis=0), axis=0))
        daily_u200_list.append(np.mean(np.stack(day_u200, axis=0), axis=0))
        snap_u850_list.append(day_u850[-1])
        snap_u200_list.append(day_u200[-1])

    u850_daily = np.stack(daily_u850_list, axis=0)
    u200_daily = np.stack(daily_u200_list, axis=0)
    u850_snap = np.stack(snap_u850_list, axis=0)
    u200_snap = np.stack(snap_u200_list, axis=0)

    mjo_head_daily = None
    if has_mjo_pred and len(step_mjo_pred) == steps:
        mjo_days = [
            np.mean(np.stack(step_mjo_pred[d * 4 : (d + 1) * 4], axis=0), axis=0)
            for d in range(max_lead_days)
        ]
        mjo_head_daily = np.stack(mjo_days, axis=0)

    valid_dates = [init_time + timedelta(days=d) for d in range(1, max_lead_days + 1)]

    return {
        "lead_days": list(range(1, max_lead_days + 1)),
        "valid_dates": valid_dates,
        "surf_daily": daily_surf,
        "u850_daily": u850_daily,
        "u200_daily": u200_daily,
        "surf_step_24h": snap_surf,
        "u850_step_24h": u850_snap,
        "u200_step_24h": u200_snap,
        "mjo_head_daily": mjo_head_daily,
    }
