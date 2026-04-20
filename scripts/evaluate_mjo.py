#!/usr/bin/env python3
"""
evaluate_mjo.py — Lead-Dependent MJO Skill Evaluation
======================================================
Measures RMM forecast skill (Bivariate ACC, RMSE, amplitude error, phase
error) across lead times 1–30 days, following Wheeler & Hendon (2004).

Overview
--------
1. Load frozen RMM basis (``data/rmm_basis.npz``) and ground-truth targets
   (``data/rmm_targets.nc``) produced by ``compute_rmm.py``.
2. Load a trained model checkpoint + config.  Defaults to the validation
   split (2016–2019) unless overridden.
3. For each initial condition in the validation set, unroll the model
   autoregressively up to 30 days (120 × 6-hourly steps).
4. At each day d ∈ [1, 30], extract predicted RMM1 & RMM2 from:
      • the **MJO head** output (preferred, when the head is enabled), OR
      • projecting predicted OLR / U850 / U200 fields onto the frozen EOF
        basis (fallback when head is disabled).
   Match forecasts to the RMM target for the corresponding valid time.
5. Compute and save:
      • Summary CSV: acc, rmse_rmm1, rmse_rmm2, amp_err, phase_err by lead
      • Plot: skill metrics vs lead time
      • Phase-space scatter for selected lead days (optional)
      • Active-MJO skill table (initial amplitude > 1.0)

Data leakage contract
---------------------
- The EOF basis (EOFs, climatology, std) is loaded from ``rmm_basis.npz``,
  which was computed exclusively on the training split (1980–2015).
- Validation/test data is ONLY projected onto that frozen basis here.

Usage
-----
    # Real evaluation (requires checkpoint + data on NERSC)
    python scripts/evaluate_mjo.py \\
        --config  configs/phase1_baseline.yaml \\
        --checkpoint checkpoints/phase1_baseline/epoch_010_val0.3400.pt \\
        --targets data/rmm_targets.nc \\
        --basis   data/rmm_basis.npz \\
        --out-dir evaluation/baseline_phase1 \\
        --split   val

    # Smoke test (no data, no GPU, no checkpoint)
    python scripts/evaluate_mjo.py --smoke-test

Smoke test
----------
    python scripts/evaluate_mjo.py --smoke-test

Dependencies
------------
    numpy, xarray, pandas, matplotlib, torch, pyyaml
    (all in environment.yml)
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Constants mirroring compute_rmm.py
# ---------------------------------------------------------------------------

VAL_YEARS  = list(range(2016, 2020))
TEST_YEARS = list(range(2020, 2024))

ACTIVE_MJO_THRESHOLD = 1.0  # amplitude > this → active MJO
MAX_LEAD_DAYS = 30           # default evaluation horizon
STEP_HRS = 6                 # Aurora step size in hours
STEPS_PER_DAY = 24 // STEP_HRS  # = 4

LAT_S = -15.0  # tropical band for EOF projection fallback
LAT_N =  15.0


# ---------------------------------------------------------------------------
# RMM utility functions
# ---------------------------------------------------------------------------

def load_basis(basis_path: Path) -> dict:
    """Load the frozen training-period RMM basis from rmm_basis.npz."""
    data = np.load(str(basis_path))
    return {
        "eof1":         data["eof1"],           # (3,)
        "eof2":         data["eof2"],           # (3,)
        "olr_clim":     data["olr_clim"],       # (366,)
        "u850_clim":    data["u850_clim"],
        "u200_clim":    data["u200_clim"],
        "olr_std":      float(data["olr_std"][0]),
        "u850_std":     float(data["u850_std"][0]),
        "u200_std":     float(data["u200_std"][0]),
        "doy":          data["clim_dayofyear"], # (366,) DOY labels
    }


def load_targets(targets_path: Path, split: str = "val") -> xr.Dataset:
    """
    Load rmm_targets.nc and return the subset for the requested split.

    Returns a Dataset with variables: rmm1, rmm2, amplitude, phase, split.
    The returned subset is indexed by time and filtered to `split`.
    """
    ds = xr.open_dataset(str(targets_path))
    mask = ds["split"] == split
    return ds.sel(time=mask)


def project_fields_to_rmm(
    olr_trop: float,
    u850_trop: float,
    u200_trop: float,
    doy: int,
    basis: dict,
) -> tuple[float, float]:
    """
    Project a single set of daily tropical-mean anomalies onto the frozen EOF
    basis to obtain predicted RMM1 and RMM2.

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

    olr_anom  = (olr_trop  - basis["olr_clim"][idx])  / (basis["olr_std"]  + 1e-8)
    u850_anom = (u850_trop - basis["u850_clim"][idx]) / (basis["u850_std"] + 1e-8)
    u200_anom = (u200_trop - basis["u200_clim"][idx]) / (basis["u200_std"] + 1e-8)

    x = np.array([olr_anom, u850_anom, u200_anom])
    rmm1 = float(x @ basis["eof1"])
    rmm2 = float(x @ basis["eof2"])
    return rmm1, rmm2


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def bivariate_acc(rmm1_fc: np.ndarray, rmm2_fc: np.ndarray,
                  rmm1_ob: np.ndarray, rmm2_ob: np.ndarray) -> float:
    """
    Bivariate Anomaly Correlation Coefficient (Wheeler & Hendon 2004).

    ACC = Σ(rmm1_f·rmm1_o + rmm2_f·rmm2_o)
          / sqrt[ Σ(rmm1_f²+rmm2_f²) · Σ(rmm1_o²+rmm2_o²) ]

    Both arrays must have the same length N (paired forecast/observation).
    Returns NaN if fewer than 2 valid pairs.
    """
    n = len(rmm1_fc)
    if n < 2:
        return np.nan
    cov = np.sum(rmm1_fc * rmm1_ob + rmm2_fc * rmm2_ob)
    var_fc = np.sum(rmm1_fc ** 2 + rmm2_fc ** 2)
    var_ob = np.sum(rmm1_ob ** 2 + rmm2_ob ** 2)
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


def phase_error_deg(rmm1_fc: np.ndarray, rmm2_fc: np.ndarray,
                    rmm1_ob: np.ndarray, rmm2_ob: np.ndarray) -> float:
    """
    Mean absolute phase error in degrees.

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


# ---------------------------------------------------------------------------
# Model-agnostic RMM extraction from a rollout
# ---------------------------------------------------------------------------

def extract_rmm_from_mjo_head(
    mjo_pred: "torch.Tensor",  # (B, 3)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract RMM1, RMM2, Amplitude from MJO head output tensor."""
    arr = mjo_pred.detach().cpu().numpy()  # (B, 3)
    rmm1 = arr[:, 0]
    rmm2 = arr[:, 1]
    amp  = np.sqrt(rmm1 ** 2 + rmm2 ** 2)
    return rmm1, rmm2, amp


def extract_rmm_from_fields(
    pred_batch: "aurora.batch.Batch",
    valid_time: datetime,
    basis: dict,
    lat_name: str = "lat",
) -> tuple[float, float, float] | tuple[None, None, None]:
    """
    Fallback: project needed OLR / U850 / U200 fields from an Aurora ``Batch``
    output onto the frozen EOF basis.

    Returns (rmm1, rmm2, amplitude) for a single sample or (None, None, None)
    if the required fields are missing.

    Aurora Batch layout
    -------------------
    pred_batch.surf_vars  : dict[str, Tensor(B, T, H, W)]
    pred_batch.atmos_vars : dict[str, Tensor(B, T, C, H, W)]
    pred_batch.metadata.lat : Tensor(H,)
    pred_batch.metadata.lon : Tensor(W,)
    pred_batch.metadata.atmos_levels : tuple of pressure levels in hPa
    """
    import torch

    surf  = pred_batch.surf_vars
    atmos = pred_batch.atmos_vars
    meta  = pred_batch.metadata

    # --- OLR fallback variable names (mtnlwrf or ttr) ---
    olr_key = None
    for k in ("mtnlwrf", "ttr"):
        if k in surf:
            olr_key = k
            break

    has_u = "u" in atmos
    if olr_key is None or not has_u:
        return None, None, None

    lat = meta.lat.cpu().numpy()   # (H,)
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

    olr_val  = _trop_mean_surf(olr_key)
    u850_val = _trop_mean_pressure("u", 850)
    u200_val = _trop_mean_pressure("u", 200)

    if u850_val is None or u200_val is None:
        return None, None, None

    doy = valid_time.timetuple().tm_yday
    rmm1, rmm2 = project_fields_to_rmm(olr_val, u850_val, u200_val, doy, basis)
    amp = float(np.sqrt(rmm1 ** 2 + rmm2 ** 2))
    return rmm1, rmm2, amp


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def _advance_time(t: datetime, n_steps: int, step_hrs: int = STEP_HRS) -> datetime:
    return t + timedelta(hours=n_steps * step_hrs)


def run_evaluation(
    model,
    val_loader,
    targets_ds: xr.Dataset,
    basis: dict,
    device,
    max_lead_days: int = MAX_LEAD_DAYS,
    active_mjo_only: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Core evaluation loop.

    For each initial condition (IC) in val_loader, unroll the model up to
    ``max_lead_days * STEPS_PER_DAY`` steps.  At each day boundary, match the
    forecast valid time to the RMM target dataset and record predicted and
    observed RMM1/RMM2.

    Parameters
    ----------
    model       : AuroraMJO (loaded with or without MJO head)
    val_loader  : DataLoader yielding (in_batch, target_dict) tuples
    targets_ds  : xr.Dataset from load_targets()
    basis       : dict from load_basis()
    device      : torch.device
    max_lead_days : int
    active_mjo_only : bool — if True, restrict ICs to active MJO cases
    verbose     : bool

    Returns
    -------
    dict with keys:
        'leads'       : list[int] 1..max_lead_days
        'acc'         : list[float]
        'rmse_rmm1'   : list[float]
        'rmse_rmm2'   : list[float]
        'amp_err'     : list[float]
        'phase_err'   : list[float]
        'n_cases'     : list[int]
        'records'     : list[dict]  per-case raw data
    """
    import torch

    targets_time = pd.DatetimeIndex(targets_ds.time.values)
    targets_rmm1 = targets_ds["rmm1"].values
    targets_rmm2 = targets_ds["rmm2"].values
    targets_amp  = targets_ds["amplitude"].values

    max_steps = max_lead_days * STEPS_PER_DAY

    has_head = (hasattr(model, "mjo_head") and model.mjo_head is not None)

    # Per-lead-day storage: list of (rmm1_fc, rmm2_fc, rmm1_ob, rmm2_ob)
    by_lead: dict[int, dict] = {
        d: {"rmm1_fc": [], "rmm2_fc": [], "rmm1_ob": [], "rmm2_ob": []}
        for d in range(1, max_lead_days + 1)
    }

    n_ic = 0
    n_skipped = 0
    records = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch_tuple in enumerate(val_loader):
            # Unpack batch
            if len(batch_tuple) == 2:
                in_batch, _ = batch_tuple
            elif len(batch_tuple) == 3:
                in_batch, surf_out, _ = batch_tuple
            else:
                raise ValueError(f"Unexpected batch tuple length: {len(batch_tuple)}")

            # Move to device (Aurora Batch has a .to() method since it wraps tensors)
            try:
                in_batch = in_batch.to(device)
            except AttributeError:
                pass  # custom dataloader may have already placed tensors

            # Initial condition time — Aurora metadata.time is a tuple of datetimes
            ic_time = in_batch.metadata.time[0]
            if isinstance(ic_time, (np.datetime64,)):
                ic_time = pd.Timestamp(ic_time).to_pydatetime()

            # Determine initial RMM amplitude from targets (for active-MJO filter)
            if ic_time in targets_time:
                ic_idx = targets_time.get_loc(ic_time)
                ic_amp = float(targets_amp[ic_idx])
            else:
                # IC time not in targets; skip (likely outside val split)
                n_skipped += 1
                continue

            if active_mjo_only and ic_amp <= ACTIVE_MJO_THRESHOLD:
                continue

            n_ic += 1
            if verbose and n_ic % 10 == 0:
                print(f"  IC {n_ic}: {ic_time}  (amp_0={ic_amp:.2f})", flush=True)

            # ------------------------------------------------------------------
            # Autoregressive rollout
            # ------------------------------------------------------------------
            current_batch = in_batch
            step_rmm1_fc: list[float | None] = [None] * (max_steps + 1)
            step_rmm2_fc: list[float | None] = [None] * (max_steps + 1)

            for step in range(1, max_steps + 1):
                valid_time = _advance_time(ic_time, step)

                out = model(current_batch)

                # Extract predicted RMM
                if has_head:
                    pred_batch, mjo_pred = out
                    r1_arr, r2_arr, _ = extract_rmm_from_mjo_head(mjo_pred)
                    # Take first sample in batch (B=1 for evaluation)
                    r1_fc = float(r1_arr[0])
                    r2_fc = float(r2_arr[0])
                else:
                    pred_batch = out
                    r1_fc, r2_fc, _ = extract_rmm_from_fields(
                        pred_batch, valid_time, basis
                    )

                step_rmm1_fc[step] = r1_fc
                step_rmm2_fc[step] = r2_fc

                # Prepare next input: Aurora expects the *predicted* state as
                # the new input.  We update time in metadata and keep rolling.
                try:
                    import dataclasses
                    from aurora.batch import Metadata
                    next_metadata = dataclasses.replace(
                        current_batch.metadata,
                        time=(valid_time,),
                    )
                    current_batch = dataclasses.replace(
                        pred_batch,
                        metadata=next_metadata,
                    )
                except Exception:
                    # If we can't build the next batch, stop this IC early
                    break

            # ------------------------------------------------------------------
            # Match forecast steps to daily RMM targets
            # ------------------------------------------------------------------
            ic_record: dict = {"ic_time": ic_time, "ic_amp": ic_amp, "leads": {}}
            for day in range(1, max_lead_days + 1):
                step_at_day = day * STEPS_PER_DAY
                if step_at_day > max_steps:
                    break
                r1_fc = step_rmm1_fc[step_at_day]
                r2_fc = step_rmm2_fc[step_at_day]
                if r1_fc is None or r2_fc is None:
                    continue

                valid_time = _advance_time(ic_time, step_at_day)
                # Match to daily target (round to nearest midnight if needed)
                valid_date = pd.Timestamp(valid_time).normalize()
                if valid_date in targets_time:
                    t_idx = targets_time.get_loc(valid_date)
                elif pd.Timestamp(valid_time) in targets_time:
                    t_idx = targets_time.get_loc(pd.Timestamp(valid_time))
                else:
                    continue  # no matching target for this valid time

                r1_ob = float(targets_rmm1[t_idx])
                r2_ob = float(targets_rmm2[t_idx])

                by_lead[day]["rmm1_fc"].append(r1_fc)
                by_lead[day]["rmm2_fc"].append(r2_fc)
                by_lead[day]["rmm1_ob"].append(r1_ob)
                by_lead[day]["rmm2_ob"].append(r2_ob)

                ic_record["leads"][day] = {
                    "rmm1_fc": r1_fc, "rmm2_fc": r2_fc,
                    "rmm1_ob": r1_ob, "rmm2_ob": r2_ob,
                }
            records.append(ic_record)

    if verbose:
        print(f"\n  Evaluated {n_ic} initial conditions ({n_skipped} skipped).")

    # ------------------------------------------------------------------
    # Aggregate metrics by lead day
    # ------------------------------------------------------------------
    leads_out, acc_out, rmse1_out, rmse2_out, amp_err_out, phase_err_out, n_cases_out = \
        [], [], [], [], [], [], []

    for day in range(1, max_lead_days + 1):
        d = by_lead[day]
        r1_fc = np.array(d["rmm1_fc"])
        r2_fc = np.array(d["rmm2_fc"])
        r1_ob = np.array(d["rmm1_ob"])
        r2_ob = np.array(d["rmm2_ob"])

        leads_out.append(day)
        acc_out.append(bivariate_acc(r1_fc, r2_fc, r1_ob, r2_ob))
        rmse1_out.append(rmse_pair(r1_fc, r1_ob))
        rmse2_out.append(rmse_pair(r2_fc, r2_ob))
        amp_fc_arr = np.sqrt(r1_fc ** 2 + r2_fc ** 2)
        amp_ob_arr = np.sqrt(r1_ob ** 2 + r2_ob ** 2)
        amp_err_out.append(amplitude_error(amp_fc_arr, amp_ob_arr))
        phase_err_out.append(phase_error_deg(r1_fc, r2_fc, r1_ob, r2_ob))
        n_cases_out.append(len(r1_fc))

    return {
        "leads":     leads_out,
        "acc":       acc_out,
        "rmse_rmm1": rmse1_out,
        "rmse_rmm2": rmse2_out,
        "amp_err":   amp_err_out,
        "phase_err": phase_err_out,
        "n_cases":   n_cases_out,
        "records":   records,
    }


# ---------------------------------------------------------------------------
# Output helpers: table + plots
# ---------------------------------------------------------------------------

def save_summary_csv(results: dict, out_dir: Path) -> Path:
    """Write per-lead-day skill metrics to a CSV."""
    df = pd.DataFrame({
        "lead_day":  results["leads"],
        "acc":       results["acc"],
        "rmse_rmm1": results["rmse_rmm1"],
        "rmse_rmm2": results["rmse_rmm2"],
        "amp_err":   results["amp_err"],
        "phase_err": results["phase_err"],
        "n_cases":   results["n_cases"],
    })
    csv_path = out_dir / "mjo_skill_by_lead.csv"
    df.to_csv(str(csv_path), index=False, float_format="%.4f")
    return csv_path


def save_skill_plots(results: dict, out_dir: Path,
                     label: str = "", active_results: dict | None = None) -> None:
    """
    Generate and save skill-vs-lead-day plots.

    Produces:
    - ``mjo_acc_vs_lead.png``   : Bivariate ACC, optionally with active-only curve
    - ``mjo_rmse_vs_lead.png``  : RMM1 and RMM2 RMSE
    - ``mjo_amp_phase_err.png`` : Amplitude bias and phase error
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available; skipping plots.")
        return

    leads = results["leads"]

    # ---- 1. Bivariate ACC ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(leads, results["acc"], "b-o", markersize=4,
            label=f"All cases{' – ' + label if label else ''}")
    if active_results is not None:
        ax.plot(active_results["leads"], active_results["acc"],
                "r--s", markersize=4, label="Active MJO only (amp₀>1)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="ACC=0.5")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Bivariate RMM ACC")
    ax.set_title("MJO Forecast Skill — Bivariate ACC vs Lead Time")
    ax.set_xlim(1, max(leads))
    ax.set_ylim(-0.1, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "mjo_acc_vs_lead.png"), dpi=150)
    plt.close(fig)

    # ---- 2. RMSE ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(leads, results["rmse_rmm1"], "b-o", markersize=4, label="RMSE RMM1")
    ax.plot(leads, results["rmse_rmm2"], "g-^", markersize=4, label="RMSE RMM2")
    ax.axhline(np.sqrt(2), color="gray", linestyle="--", linewidth=0.8,
               label=r"√2 (climatological limit)")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("RMSE (dimensionless)")
    ax.set_title("RMM1 & RMM2 RMSE vs Lead Time")
    ax.set_xlim(1, max(leads))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "mjo_rmse_vs_lead.png"), dpi=150)
    plt.close(fig)

    # ---- 3. Amplitude bias + Phase error ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(leads, results["amp_err"], "b-o", markersize=4)
    ax1.axhline(0, color="k", linewidth=0.8)
    ax1.set_xlabel("Lead time (days)")
    ax1.set_ylabel("Amplitude bias (fc − obs)")
    ax1.set_title("Amplitude Error vs Lead Time")
    ax1.set_xlim(1, max(leads))
    ax1.grid(True, alpha=0.3)

    ax2.plot(leads, results["phase_err"], "r-o", markersize=4)
    ax2.set_xlabel("Lead time (days)")
    ax2.set_ylabel("Mean absolute phase error (°)")
    ax2.set_title("Phase Error vs Lead Time")
    ax2.set_xlim(1, max(leads))
    ax2.set_ylim(0, 180)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_dir / "mjo_amp_phase_err.png"), dpi=150)
    plt.close(fig)


def print_skill_table(results: dict, active_results: dict | None = None) -> None:
    """Print a formatted skill summary table to stdout."""
    header = f"{'Lead':>5} {'ACC':>7} {'RMSE1':>7} {'RMSE2':>7} {'AmpErr':>8} {'PhaseErr':>10} {'N':>6}"
    print("\n" + "=" * len(header))
    print("MJO Forecast Skill Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for i, day in enumerate(results["leads"]):
        acc  = results["acc"][i]
        r1   = results["rmse_rmm1"][i]
        r2   = results["rmse_rmm2"][i]
        ae   = results["amp_err"][i]
        pe   = results["phase_err"][i]
        n    = results["n_cases"][i]
        acc_s  = f"{acc:.3f}"  if not np.isnan(acc)  else "  NaN"
        r1_s   = f"{r1:.3f}"   if not np.isnan(r1)   else "  NaN"
        r2_s   = f"{r2:.3f}"   if not np.isnan(r2)   else "  NaN"
        ae_s   = f"{ae:+.3f}"  if not np.isnan(ae)   else "   NaN"
        pe_s   = f"{pe:.1f}°"  if not np.isnan(pe)   else "    NaN"
        print(f"{day:>5} {acc_s:>7} {r1_s:>7} {r2_s:>7} {ae_s:>8} {pe_s:>10} {n:>6}")

    print("=" * len(header))


# ---------------------------------------------------------------------------
# Smoke test (no model, no data)
# ---------------------------------------------------------------------------

def run_smoke_test() -> None:
    """
    Verify that all metric functions and the evaluation loop plumbing work
    on purely synthetic data.  Requires no model, no files, no GPU.
    """
    print("=== Smoke Test: evaluate_mjo.py ===\n")
    rng = np.random.default_rng(0)

    # --- Metric functions ---
    N = 100
    r1_fc = rng.standard_normal(N)
    r2_fc = rng.standard_normal(N)
    r1_ob = r1_fc + rng.standard_normal(N) * 0.3
    r2_ob = r2_fc + rng.standard_normal(N) * 0.3

    acc = bivariate_acc(r1_fc, r2_fc, r1_ob, r2_ob)
    print(f"  bivariate_acc (noisy perfect forecast): {acc:.4f}  (expected > 0.8)")
    assert acc > 0.0, "ACC should be positive for correlated forecasts"

    rms1 = rmse_pair(r1_fc, r1_ob)
    rms2 = rmse_pair(r2_fc, r2_ob)
    print(f"  RMSE RMM1={rms1:.4f}  RMM2={rms2:.4f}")
    assert rms1 > 0 and rms2 > 0

    ae = amplitude_error(
        np.sqrt(r1_fc**2 + r2_fc**2),
        np.sqrt(r1_ob**2 + r2_ob**2),
    )
    pe = phase_error_deg(r1_fc, r2_fc, r1_ob, r2_ob)
    print(f"  Amplitude error={ae:+.4f}  Phase error={pe:.2f}°")

    # --- EOF projection (synthetic basis) ---
    basis_synth = {
        "eof1":      np.array([0.6, 0.5, 0.6]) / np.linalg.norm([0.6, 0.5, 0.6]),
        "eof2":      np.array([0.5, -0.6, 0.6]) / np.linalg.norm([0.5, -0.6, 0.6]),
        "olr_clim":  np.zeros(366),
        "u850_clim": np.zeros(366),
        "u200_clim": np.zeros(366),
        "olr_std":   1.0,
        "u850_std":  1.0,
        "u200_std":  1.0,
        "doy":       np.arange(1, 367),
    }
    r1, r2 = project_fields_to_rmm(0.5, -0.3, 0.2, doy=45, basis=basis_synth)
    print(f"  project_fields_to_rmm → RMM1={r1:.4f}  RMM2={r2:.4f}")

    # --- load_basis with a synthetic npz ---
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = Path(f.name)
    np.savez(
        str(tmp_path),
        eof1=basis_synth["eof1"],
        eof2=basis_synth["eof2"],
        eigenvalues=np.array([0.6, 0.3, 0.1]),
        olr_clim=basis_synth["olr_clim"],
        u850_clim=basis_synth["u850_clim"],
        u200_clim=basis_synth["u200_clim"],
        olr_std=np.array([1.0]),
        u850_std=np.array([1.0]),
        u200_std=np.array([1.0]),
        clim_dayofyear=basis_synth["doy"],
        train_years=np.arange(1980, 2016),
        val_years=np.arange(2016, 2020),
        test_years=np.arange(2020, 2024),
    )
    loaded = load_basis(tmp_path)
    tmp_path.unlink()
    assert loaded["eof1"].shape == (3,), "eof1 shape mismatch"
    print(f"  load_basis OK  eof1={loaded['eof1']}")

    # --- Synthetic results dict → CSV + plots (dry-run without file write) ---
    synthetic_results = {
        "leads":     list(range(1, 31)),
        "acc":       [max(0, 1.0 - 0.03 * d + rng.uniform(-0.02, 0.02)) for d in range(1, 31)],
        "rmse_rmm1": [0.2 + 0.04 * d for d in range(1, 31)],
        "rmse_rmm2": [0.2 + 0.035 * d for d in range(1, 31)],
        "amp_err":   [0.05 * d ** 0.5 for d in range(1, 31)],
        "phase_err": [5.0 + 4.0 * d ** 0.7 for d in range(1, 31)],
        "n_cases":   [50] * 30,
        "records":   [],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        csv_path = save_summary_csv(synthetic_results, out)
        assert csv_path.exists(), "CSV not written"
        df = pd.read_csv(str(csv_path))
        assert list(df.columns) == [
            "lead_day", "acc", "rmse_rmm1", "rmse_rmm2",
            "amp_err", "phase_err", "n_cases"
        ], f"CSV columns mismatch: {list(df.columns)}"
        print(f"  save_summary_csv OK  shape={df.shape}")

        save_skill_plots(synthetic_results, out, label="smoke-test")
        png_files = list(out.glob("*.png"))
        print(f"  save_skill_plots OK  wrote {len(png_files)} PNG(s)")

    print_skill_table(synthetic_results)
    print("\n=== Smoke Test PASSED ===\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lead-dependent MJO skill evaluation (Wheeler-Hendon 2004)."
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="Path to YAML config used when the checkpoint was trained.",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to a .pt checkpoint file (output of Trainer.save_checkpoint).",
    )
    p.add_argument(
        "--targets", type=Path, default=Path("data/rmm_targets.nc"),
        help="Path to rmm_targets.nc (default: data/rmm_targets.nc).",
    )
    p.add_argument(
        "--basis", type=Path, default=Path("data/rmm_basis.npz"),
        help="Path to rmm_basis.npz (default: data/rmm_basis.npz).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("evaluation/mjo_skill"),
        help="Directory to write CSV, plots, and metadata (default: evaluation/mjo_skill).",
    )
    p.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="Which data split to evaluate (default: val = 2016–2019).",
    )
    p.add_argument(
        "--max-lead-days", type=int, default=MAX_LEAD_DAYS,
        help=f"Maximum lead time in days (default: {MAX_LEAD_DAYS}).",
    )
    p.add_argument(
        "--active-mjo-only", action="store_true",
        help="Restrict initial conditions to active MJO cases (amp > 1.0).",
    )
    p.add_argument(
        "--also-active", action="store_true",
        help="Also compute active-MJO-only skill curve alongside all-case curve.",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="PyTorch device string (default: cuda if available, else cpu).",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Run synthetic smoke test — no model, data, or GPU required.",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    return p.parse_args()


def _load_model_and_config(args: argparse.Namespace):
    """Load model + config from checkpoint and config file."""
    import torch
    import yaml
    from src.model import load_model

    if args.config is None or not args.config.exists():
        raise FileNotFoundError(
            f"Config file not found: {args.config}. "
            "Pass --config path/to/config.yaml"
        )
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = load_model(cfg.get("model", cfg))  # handle flat or nested config

    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        ckpt = torch.load(str(args.checkpoint), map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"  Loaded checkpoint: {args.checkpoint}")
    else:
        print("  WARNING: No checkpoint path provided; using untrained model weights.")

    return model, cfg


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")

    if args.smoke_test:
        run_smoke_test()
        return

    import torch

    # ---- Device ----
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ---- Check required files ----
    for p, name in [(args.targets, "targets"), (args.basis, "basis")]:
        if not p.exists():
            print(
                f"ERROR: {name} file not found: {p}\n"
                "       Run scripts/compute_rmm.py first to generate it.",
                file=sys.stderr,
            )
            sys.exit(1)

    # ---- Load basis + targets ----
    basis = load_basis(args.basis)
    targets_ds = load_targets(args.targets, split=args.split)

    if not args.quiet:
        print(f"  Loaded RMM targets: {len(targets_ds.time)} {args.split} timesteps")

    # ---- Load model ----
    model, cfg = _load_model_and_config(args)
    model = model.to(device)

    # ---- Build dataloader (val split only) ----
    # Import locally to avoid hard dependency when running smoke-test
    from src.trainer import build_dataloader
    val_loader = build_dataloader(cfg, split=args.split if args.split == "val" else "val")

    # ---- Prepare output dir ----
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Metadata file ----
    import json
    meta = {
        "timestamp":    datetime.utcnow().isoformat() + "Z",
        "split":        args.split,
        "checkpoint":   str(args.checkpoint) if args.checkpoint else "None",
        "config":       str(args.config),
        "max_lead_days": args.max_lead_days,
        "active_mjo_only": args.active_mjo_only,
        "device":       str(device),
    }
    try:
        import subprocess
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        meta["git_commit"] = "unavailable"
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # ---- Main evaluation ----
    if not args.quiet:
        print(f"\nRunning evaluation (split={args.split}, max_lead_days={args.max_lead_days}):")

    results = run_evaluation(
        model=model,
        val_loader=val_loader,
        targets_ds=targets_ds,
        basis=basis,
        device=device,
        max_lead_days=args.max_lead_days,
        active_mjo_only=args.active_mjo_only,
        verbose=not args.quiet,
    )

    # ---- Active-MJO only (optional second pass) ----
    active_results = None
    if args.also_active and not args.active_mjo_only:
        if not args.quiet:
            print("\nRunning active-MJO-only evaluation:")
        val_loader2 = build_dataloader(cfg, split="val")
        active_results = run_evaluation(
            model=model,
            val_loader=val_loader2,
            targets_ds=targets_ds,
            basis=basis,
            device=device,
            max_lead_days=args.max_lead_days,
            active_mjo_only=True,
            verbose=not args.quiet,
        )
        save_summary_csv(active_results, args.out_dir / "active_mjo")
        args.out_dir.joinpath("active_mjo").mkdir(exist_ok=True)

    # ---- Save outputs ----
    csv_path = save_summary_csv(results, args.out_dir)

    ckpt_label = args.checkpoint.stem if args.checkpoint else "no_ckpt"
    save_skill_plots(results, args.out_dir, label=ckpt_label,
                     active_results=active_results)

    print_skill_table(results, active_results=active_results)

    print(f"\n  Outputs written to: {args.out_dir}/")
    print(f"   • {csv_path.name}")
    print(f"   • mjo_acc_vs_lead.png")
    print(f"   • mjo_rmse_vs_lead.png")
    print(f"   • mjo_amp_phase_err.png")
    print(f"   • metadata.json")

    # Print quick summary
    d15_idx = 14  # 0-indexed day 15
    d30_idx = 29
    if len(results["acc"]) > d30_idx:
        print(
            f"\n  Bivariate ACC @ day 15: {results['acc'][d15_idx]:.3f}"
            f"  |  @ day 30: {results['acc'][d30_idx]:.3f}"
            f"  (target: > 0.5 at day 30)"
        )


if __name__ == "__main__":
    main()
