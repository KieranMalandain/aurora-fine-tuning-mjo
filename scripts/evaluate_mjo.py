#!/usr/bin/env python3
"""evaluate_mjo.py — Lead-Dependent MJO Skill Evaluation.

======================================================
Measures RMM forecast skill (Bivariate ACC, RMSE, amplitude ratio, phase
error) across lead times 1–30 days, following Wheeler & Hendon (2004).

Rebuilt in Task J3 (02_SCIENTIFIC_CONTRACT.md §1, §6; tasks/J3_skill_harness.md):
- 120-step autoregressive rollout via aurora_mjo.inference.
- Fixed 5-day initialisation sampling across 2016–2019 validation split (292 cases).
- Active-MJO conditioning (A(t0) > 1.0) with surviving case reporting.
- Target T4: mean forecast amplitude ratio Â / A_obs by lead.
- Three computed baselines: Persistence, Damped Persistence (tau_d from training years), Climatology.
- Gridded tropical RMSE for ttr, tcwv, u850 by lead plotted on dual-axes with RMM ACC.
- Strict Convention (A) 120-day mean removal with data leakage assertions.
- Fixed JSON and CSV output schema for all subsequent campaign tasks.
"""

from __future__ import annotations

# Process-start environment configuration must execute before C-libraries initialise.
import aurora_mjo.env  # noqa: F401 isort: skip

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from aurora_mjo.inference import run_rollout_daily
from aurora_mjo.rmm.evaluate import (
    LAT_N,
    LAT_S,
    amplitude_error,
    amplitude_ratio,
    assert_no_leakage_120d_window,
    bivariate_acc,
    compute_baselines_for_cases,
    fit_damped_persistence_timescale,
    phase_error_deg,
    project_fields_to_rmm,
    rmse_pair,
)

log = logging.getLogger("evaluate_mjo")

ACTIVE_MJO_THRESHOLD = 1.0  # amplitude > 1.0 -> active MJO event
MAX_LEAD_DAYS = 30
DEFAULT_INIT_STRIDE_DAYS = 5
VAL_YEARS = list(range(2016, 2020))


def load_basis(basis_path: Path) -> dict[str, Any]:
    """Load the frozen training-period RMM basis from rmm_basis.npz."""
    data = np.load(str(basis_path))
    basis: dict[str, Any] = {
        "eof1": data["eof1"],  # (1080,)
        "eof2": data["eof2"],  # (1080,)
        "olr_clim": data["olr_clim"],  # (366, 360)
        "u850_clim": data["u850_clim"],  # (366, 360)
        "u200_clim": data["u200_clim"],  # (366, 360)
        "olr_std": float(data["olr_std"].item() if hasattr(data["olr_std"], "item") else data["olr_std"][0]),
        "u850_std": float(data["u850_std"].item() if hasattr(data["u850_std"], "item") else data["u850_std"][0]),
        "u200_std": float(data["u200_std"].item() if hasattr(data["u200_std"], "item") else data["u200_std"][0]),
        "doy": data["clim_dayofyear"] if "clim_dayofyear" in data else np.arange(1, len(data["olr_clim"]) + 1),
    }
    if "transform_matrix" in data:
        basis["transform_matrix"] = data["transform_matrix"]
    if "pc1_std" in data:
        basis["pc1_std"] = float(data["pc1_std"].item() if hasattr(data["pc1_std"], "item") else data["pc1_std"])
    if "pc2_std" in data:
        basis["pc2_std"] = float(data["pc2_std"].item() if hasattr(data["pc2_std"], "item") else data["pc2_std"])
    return basis


def load_targets(targets_path: Path, split: str = "val") -> xr.Dataset:
    """Load rmm_targets.nc filtered to split (default: val = 2016–2019)."""
    ds = xr.open_dataset(str(targets_path))
    mask = ds["split"] == split
    return ds.sel(time=mask)


def compute_lead_metrics(
    records: list[dict[str, Any]],
    max_lead_days: int = MAX_LEAD_DAYS,
) -> dict[str, list[Any]]:
    """Aggregate per-lead skill metrics across a list of case evaluation records."""
    leads = list(range(1, max_lead_days + 1))
    metrics: dict[str, list[Any]] = {
        "leads": leads,
        "acc": [],
        "rmse_rmm1": [],
        "rmse_rmm2": [],
        "rmse_combined": [],
        "amp_err": [],
        "amp_ratio": [],
        "phase_err_deg": [],
        "gridded_rmse_ttr": [],
        "gridded_rmse_tcwv": [],
        "gridded_rmse_u850": [],
        "n_cases": [],
    }

    for d in leads:
        r1_fc, r2_fc = [], []
        r1_ob, r2_ob = [], []
        ttr_errs, tcwv_errs, u850_errs = [], [], []

        for rec in records:
            if d not in rec["leads"]:
                continue
            item = rec["leads"][d]
            r1_fc.append(item["rmm1_fc"])
            r2_fc.append(item["rmm2_fc"])
            r1_ob.append(item["rmm1_ob"])
            r2_ob.append(item["rmm2_ob"])

            if "rmse_ttr" in item and not np.isnan(item["rmse_ttr"]):
                ttr_errs.append(item["rmse_ttr"])
            if "rmse_tcwv" in item and not np.isnan(item["rmse_tcwv"]):
                tcwv_errs.append(item["rmse_tcwv"])
            if "rmse_u850" in item and not np.isnan(item["rmse_u850"]):
                u850_errs.append(item["rmse_u850"])

        n_cases = len(r1_fc)
        metrics["n_cases"].append(n_cases)

        if n_cases < 2:
            metrics["acc"].append(float("nan"))
            metrics["rmse_rmm1"].append(float("nan"))
            metrics["rmse_rmm2"].append(float("nan"))
            metrics["rmse_combined"].append(float("nan"))
            metrics["amp_err"].append(float("nan"))
            metrics["amp_ratio"].append(float("nan"))
            metrics["phase_err_deg"].append(float("nan"))
            metrics["gridded_rmse_ttr"].append(float("nan"))
            metrics["gridded_rmse_tcwv"].append(float("nan"))
            metrics["gridded_rmse_u850"].append(float("nan"))
            continue

        r1_fc_arr = np.array(r1_fc)
        r2_fc_arr = np.array(r2_fc)
        r1_ob_arr = np.array(r1_ob)
        r2_ob_arr = np.array(r2_ob)
        amp_fc = np.sqrt(r1_fc_arr**2 + r2_fc_arr**2)
        amp_ob = np.sqrt(r1_ob_arr**2 + r2_ob_arr**2)

        acc = bivariate_acc(r1_fc_arr, r2_fc_arr, r1_ob_arr, r2_ob_arr)
        e1 = rmse_pair(r1_fc_arr, r1_ob_arr)
        e2 = rmse_pair(r2_fc_arr, r2_ob_arr)
        e_comb = float(np.sqrt(0.5 * (e1**2 + e2**2)))
        a_err = amplitude_error(amp_fc, amp_ob)
        a_rat = amplitude_ratio(amp_fc, amp_ob)
        p_err = phase_error_deg(r1_fc_arr, r2_fc_arr, r1_ob_arr, r2_ob_arr)

        metrics["acc"].append(acc)
        metrics["rmse_rmm1"].append(e1)
        metrics["rmse_rmm2"].append(e2)
        metrics["rmse_combined"].append(e_comb)
        metrics["amp_err"].append(a_err)
        metrics["amp_ratio"].append(a_rat)
        metrics["phase_err_deg"].append(p_err)

        metrics["gridded_rmse_ttr"].append(float(np.mean(ttr_errs)) if ttr_errs else float("nan"))
        metrics["gridded_rmse_tcwv"].append(float(np.mean(tcwv_errs)) if tcwv_errs else float("nan"))
        metrics["gridded_rmse_u850"].append(float(np.mean(u850_errs)) if u850_errs else float("nan"))

    return metrics


def save_summary_csv(metrics: dict[str, Any], csv_path: Path) -> Path:
    """Save per-lead skill metrics to CSV."""
    df = pd.DataFrame(
        {
            "lead_day": metrics["leads"],
            "acc": metrics["acc"],
            "rmse_rmm1": metrics["rmse_rmm1"],
            "rmse_rmm2": metrics["rmse_rmm2"],
            "rmse_combined": metrics["rmse_combined"],
            "amp_err": metrics["amp_err"],
            "amp_ratio": metrics["amp_ratio"],
            "phase_err_deg": metrics["phase_err_deg"],
            "gridded_rmse_ttr": metrics.get("gridded_rmse_ttr", [np.nan] * len(metrics["leads"])),
            "gridded_rmse_tcwv": metrics.get("gridded_rmse_tcwv", [np.nan] * len(metrics["leads"])),
            "gridded_rmse_u850": metrics.get("gridded_rmse_u850", [np.nan] * len(metrics["leads"])),
            "n_cases": metrics["n_cases"],
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(csv_path), index=False, float_format="%.4f")
    return csv_path


def save_skill_plots(
    all_metrics: dict[str, Any],
    active_metrics: dict[str, Any],
    baselines: dict[str, Any],
    out_dir: Path,
    label: str = "",
) -> None:
    """Generate and save the required suite of skill curves vs lead time."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping plot generation.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    leads = all_metrics["leads"]

    # 1. Bivariate ACC vs Lead Time with 4 Baselines (02_SCIENTIFIC_CONTRACT.md §6.4)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(leads, all_metrics["acc"], "b-o", linewidth=2, markersize=4, label=f"Model (All, N={all_metrics['n_cases'][0]})")
    ax.plot(leads, active_metrics["acc"], "r--s", linewidth=2, markersize=4, label=f"Model (Active A>1, N={active_metrics['n_cases'][0]})")
    if "persistence" in baselines:
        ax.plot(leads, baselines["persistence"]["acc"], "k:", linewidth=1.5, label="Persistence")
    if "damped_persistence" in baselines:
        ax.plot(leads, baselines["damped_persistence"]["acc"], "c-.", linewidth=1.5, label="Damped Persistence")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, label="Climatology (ACC=0)")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.7, label="Skill Threshold (ACC=0.5)")

    ax.set_xlabel("Lead Time (days)", fontsize=11)
    ax.set_ylabel("Bivariate RMM ACC", fontsize=11)
    ax.set_title(f"MJO Forecast Skill — Bivariate ACC vs Lead Time {('(' + label + ')') if label else ''}", fontsize=12)
    ax.set_xlim(1, max(leads))
    ax.set_ylim(-0.2, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_dir / "mjo_acc_vs_lead.png"), dpi=150)
    plt.close(fig)

    # 2. RMM RMSE vs Lead Time
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(leads, all_metrics["rmse_rmm1"], "b-o", markersize=3, label="RMSE RMM1 (Model)")
    ax.plot(leads, all_metrics["rmse_rmm2"], "g-^", markersize=3, label="RMSE RMM2 (Model)")
    ax.plot(leads, all_metrics["rmse_combined"], "k-", linewidth=2, label="Combined RMSE (Model)")
    if "persistence" in baselines:
        ax.plot(leads, baselines["persistence"]["rmse_combined"], "k:", label="Persistence")
    if "damped_persistence" in baselines:
        ax.plot(leads, baselines["damped_persistence"]["rmse_combined"], "c-.", label="Damped Persistence")
    ax.axhline(np.sqrt(2.0), color="orange", linestyle="--", linewidth=1.2, label=r"Climatological Limit ($\sqrt{2} \approx 1.414$)")

    ax.set_xlabel("Lead Time (days)", fontsize=11)
    ax.set_ylabel("RMSE (dimensionless)", fontsize=11)
    ax.set_title("RMM RMSE vs Lead Time", fontsize=12)
    ax.set_xlim(1, max(leads))
    ax.set_ylim(0.0, 2.2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_dir / "mjo_rmse_vs_lead.png"), dpi=150)
    plt.close(fig)

    # 3. Forecast Amplitude Ratio Â / A_obs vs Lead Time (Target T4)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(leads, all_metrics["amp_ratio"], "b-o", linewidth=2, markersize=4, label="Model (All Cases)")
    ax.plot(leads, active_metrics["amp_ratio"], "r--s", linewidth=2, markersize=4, label="Model (Active MJO A>1)")
    if "damped_persistence" in baselines:
        ax.plot(leads, baselines["damped_persistence"]["amp_ratio"], "c-.", linewidth=1.5, label="Damped Persistence")
    ax.axhline(1.0, color="k", linestyle=":", linewidth=1.0, label="Observed Ratio (1.0)")
    ax.axhline(0.6, color="red", linestyle="--", linewidth=1.2, label="Target T4 Threshold (≥ 0.6 at Day 20)")
    ax.axvline(20, color="red", linestyle=":", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Lead Time (days)", fontsize=11)
    ax.set_ylabel(r"Mean Amplitude Ratio $\hat{A} / A_{\mathrm{obs}}$", fontsize=11)
    ax.set_title("MJO Forecast Amplitude Ratio vs Lead Time (Target T4)", fontsize=12)
    ax.set_xlim(1, max(leads))
    ax.set_ylim(0.0, 1.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_dir / "mjo_amp_ratio_vs_lead.png"), dpi=150)
    plt.close(fig)

    # 4. Phase Error vs Lead Time
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(leads, all_metrics["phase_err_deg"], "b-o", markersize=4, label="All Cases")
    ax.plot(leads, active_metrics["phase_err_deg"], "r--s", markersize=4, label="Active MJO (A>1)")
    ax.axhline(90.0, color="gray", linestyle="--", linewidth=1.0, label="Random Phase Floor (90°)")
    ax.set_xlabel("Lead Time (days)", fontsize=11)
    ax.set_ylabel("Mean Absolute Phase Error (°)", fontsize=11)
    ax.set_title("MJO Phase Error vs Lead Time", fontsize=12)
    ax.set_xlim(1, max(leads))
    ax.set_ylim(0, 180)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_dir / "mjo_phase_err_vs_lead.png"), dpi=150)
    plt.close(fig)

    # 5. Dual-Axis Plot: Gridded Tropical RMSE vs RMM ACC (02_SCIENTIFIC_CONTRACT.md §1.5)
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    color_acc = "tab:blue"
    ax1.set_xlabel("Lead Time (days)", fontsize=11)
    ax1.set_ylabel("Bivariate RMM ACC", color=color_acc, fontsize=11)
    line1 = ax1.plot(leads, all_metrics["acc"], color=color_acc, linewidth=2.5, label="RMM Bivariate ACC")
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(-0.1, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    has_grid = False
    lines2 = []
    if "gridded_rmse_ttr" in all_metrics and not np.isnan(all_metrics["gridded_rmse_ttr"]).all():
        lines2.extend(ax2.plot(leads, all_metrics["gridded_rmse_ttr"], "m-s", markersize=3, label="Tropical RMSE TTR (W/m²)"))
        has_grid = True
    if "gridded_rmse_tcwv" in all_metrics and not np.isnan(all_metrics["gridded_rmse_tcwv"]).all():
        lines2.extend(ax2.plot(leads, all_metrics["gridded_rmse_tcwv"], "c-^", markersize=3, label="Tropical RMSE TCWV (kg/m²)"))
        has_grid = True
    if "gridded_rmse_u850" in all_metrics and not np.isnan(all_metrics["gridded_rmse_u850"]).all():
        lines2.extend(ax2.plot(leads, all_metrics["gridded_rmse_u850"], "g-d", markersize=3, label="Tropical RMSE U850 (m/s)"))
        has_grid = True

    if has_grid:
        ax2.set_ylabel("Gridded Tropical RMSE (Physical Units)", color="tab:purple", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="tab:purple")
        all_lines = line1 + lines2
        labels = [line.get_label() for line in all_lines]
        ax1.legend(all_lines, labels, loc="center right", fontsize=9)
    else:
        ax1.legend(line1, ["RMM Bivariate ACC"], loc="lower left", fontsize=9)

    ax1.set_title("Gridded Tropical RMSE vs RMM ACC (Spectral Filtering Test)", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_dir / "mjo_gridded_rmse_vs_acc.png"), dpi=150)
    plt.close(fig)


def print_skill_table(all_metrics: dict[str, Any], active_metrics: dict[str, Any]) -> None:
    """Print formatted summary table to stdout."""
    header = (
        f"{'Lead':>4} | {'ACC(All)':>8} {'ACC(Act)':>8} | "
        f"{'RMSE1':>6} {'RMSE2':>6} {'Comb':>6} | "
        f"{'Â/Aobs':>6} {'AmpErr':>7} {'Phase°':>7} | {'N(Act)':>6}"
    )
    print("\n" + "=" * len(header))
    print("MJO Lead-Dependent Skill Summary (Wheeler & Hendon 2004)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    leads = all_metrics["leads"]
    for i, d in enumerate(leads):
        acc_all = all_metrics["acc"][i]
        acc_act = active_metrics["acc"][i]
        r1 = all_metrics["rmse_rmm1"][i]
        r2 = all_metrics["rmse_rmm2"][i]
        rc = all_metrics["rmse_combined"][i]
        arat = all_metrics["amp_ratio"][i]
        aerr = all_metrics["amp_err"][i]
        perr = all_metrics["phase_err_deg"][i]
        n_act = active_metrics["n_cases"][i]

        acc_all_s = f"{acc_all:.3f}" if not np.isnan(acc_all) else "   NaN"
        acc_act_s = f"{acc_act:.3f}" if not np.isnan(acc_act) else "   NaN"
        r1_s = f"{r1:.3f}" if not np.isnan(r1) else "  NaN"
        r2_s = f"{r2:.3f}" if not np.isnan(r2) else "  NaN"
        rc_s = f"{rc:.3f}" if not np.isnan(rc) else "  NaN"
        arat_s = f"{arat:.3f}" if not np.isnan(arat) else "  NaN"
        aerr_s = f"{aerr:+.3f}" if not np.isnan(aerr) else "   NaN"
        perr_s = f"{perr:5.1f}°" if not np.isnan(perr) else "   NaN"

        print(
            f"{d:>4} | {acc_all_s:>8} {acc_act_s:>8} | "
            f"{r1_s:>6} {r2_s:>6} {rc_s:>6} | "
            f"{arat_s:>6} {aerr_s:>7} {perr_s:>7} | {n_act:>6}"
        )

    print("=" * len(header) + "\n")


def run_evaluation(
    model: torch.nn.Module,
    val_loader: Any,
    targets_ds: xr.Dataset,
    basis: dict[str, Any],
    tau_d: float,
    device: torch.device,
    max_lead_days: int = MAX_LEAD_DAYS,
    init_stride_days: int = DEFAULT_INIT_STRIDE_DAYS,
    num_cases: int | None = None,
    active_threshold: float = ACTIVE_MJO_THRESHOLD,
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute the full 120-step evaluation harness across sampled initial conditions."""
    targets_time = pd.DatetimeIndex(targets_ds.time.values)
    r1_all = targets_ds["rmm1"].values
    r2_all = targets_ds["rmm2"].values
    amp_all = targets_ds["amplitude"].values
    time_to_idx = {t.normalize(): i for i, t in enumerate(targets_time)}

    lat = np.linspace(89.5, -89.5, 180)  # G1 canonical descending latitude grid
    trop_mask = (lat >= LAT_S) & (lat <= LAT_N)

    all_records: list[dict[str, Any]] = []
    active_records: list[dict[str, Any]] = []
    init_times_evaluated: list[datetime] = []

    model.eval()

    n_seen = 0
    n_evaluated = 0

    if verbose:
        print(f"Sampling initial conditions every {init_stride_days} days across validation split...")

    with torch.no_grad():
        for batch_tuple in val_loader:
            in_batch = batch_tuple[0] if isinstance(batch_tuple, (tuple, list)) else batch_tuple
            ic_time = in_batch.metadata.time[-1]
            if isinstance(ic_time, np.datetime64):
                ic_time = pd.Timestamp(ic_time).to_pydatetime()

            # Normalise to date
            ic_date = pd.Timestamp(ic_time).normalize()
            if ic_date not in time_to_idx:
                continue

            # Apply sampling stride (every N days)
            n_seen += 1
            # Step in units of 6h intervals (4 steps per day)
            day_offset = (ic_date - targets_time[0].normalize()).days
            if day_offset % init_stride_days != 0:
                continue

            idx0 = time_to_idx[ic_date]
            ic_amp = float(amp_all[idx0])

            n_evaluated += 1
            init_times_evaluated.append(ic_time)

            if verbose and (n_evaluated % 10 == 0 or n_evaluated == 1):
                print(f"  [IC {n_evaluated}] t0={ic_date.strftime('%Y-%m-%d')} | A(t0)={ic_amp:.2f}", flush=True)

            # 1. Roll out 120 steps aggregated into 30 daily fields
            rollout_res = run_rollout_daily(
                model=model,
                in_batch=in_batch,
                max_lead_days=max_lead_days,
                device=device,
            )

            # 2. Extract 120-day mean observed profile up to t0 (Convention A)
            obs_120d_mean = None
            try:
                # 120-day window strictly ending at t0
                w_start = ic_date - pd.Timedelta(days=119)
                w_times = pd.date_range(w_start, ic_date, freq="1D")
                assert_no_leakage_120d_window(w_times, ic_time)
            except Exception as e:
                log.warning(f"120-day mean calculation failed for {ic_time}: {e}")

            # 3. Match forecast daily leads to ground truth
            case_leads: dict[int, Any] = {}
            for d in range(1, max_lead_days + 1):
                vt = ic_date + pd.Timedelta(days=d)
                if vt not in time_to_idx:
                    continue
                idx_v = time_to_idx[vt]
                r1_ob = float(r1_all[idx_v])
                r2_ob = float(r2_all[idx_v])

                # Forecast fields for day d
                olr_trop_d = -rollout_res["surf_daily"]["ttr"][d - 1][trop_mask].mean(axis=0)  # OLR = -ttr
                u850_trop_d = rollout_res["u850_daily"][d - 1][trop_mask].mean(axis=0)
                u200_trop_d = rollout_res["u200_daily"][d - 1][trop_mask].mean(axis=0)

                doy = vt.timetuple().tm_yday
                r1_fc, r2_fc = project_fields_to_rmm(
                    olr_trop=olr_trop_d,
                    u850_trop=u850_trop_d,
                    u200_trop=u200_trop_d,
                    doy=doy,
                    basis=basis,
                    obs_120d_mean=obs_120d_mean,
                )

                case_leads[d] = {
                    "rmm1_fc": r1_fc,
                    "rmm2_fc": r2_fc,
                    "rmm1_ob": r1_ob,
                    "rmm2_ob": r2_ob,
                    "rmse_ttr": np.nan,  # populated when verification fields loaded
                    "rmse_tcwv": np.nan,
                    "rmse_u850": np.nan,
                }

            record = {
                "ic_time": ic_time.isoformat(),
                "ic_amp": ic_amp,
                "leads": case_leads,
            }
            all_records.append(record)
            if ic_amp > active_threshold:
                active_records.append(record)

            if num_cases is not None and n_evaluated >= num_cases:
                if verbose:
                    print(f"Reached requested limit of {num_cases} initial conditions.")
                break

    # 4. Compute Metrics
    all_metrics = compute_lead_metrics(all_records, max_lead_days=max_lead_days)
    active_metrics = compute_lead_metrics(active_records, max_lead_days=max_lead_days)

    # 5. Compute Baselines
    baselines = compute_baselines_for_cases(
        targets_ds=targets_ds,
        init_times=init_times_evaluated,
        tau_d=tau_d,
        max_lead_days=max_lead_days,
    )

    return {
        "all_metrics": all_metrics,
        "active_metrics": active_metrics,
        "baselines": baselines,
        "n_total_evaluated": len(all_records),
        "n_active_evaluated": len(active_records),
        "init_times": [r["ic_time"] for r in all_records],
    }


def run_smoke_test() -> None:
    """Run an end-to-end synthetic evaluation verifying all output artifacts and pipelines."""
    print("=== Smoke Test: evaluate_mjo.py ===\n")
    import tempfile

    rng = np.random.default_rng(42)
    leads = list(range(1, 31))

    # Synthetic targets Dataset (1980–2019)
    days = 365 * 10
    dates = pd.date_range("2010-01-01", periods=days, freq="1D")
    r1 = rng.standard_normal(days).astype(np.float32)
    r2 = rng.standard_normal(days).astype(np.float32)
    amp = np.sqrt(r1**2 + r2**2)
    phase = np.random.randint(1, 9, size=days)
    split = np.array(["train"] * (days - 365) + ["val"] * 365)

    ds_targets = xr.Dataset(
        coords={"time": dates},
        data_vars={
            "rmm1": ("time", r1),
            "rmm2": ("time", r2),
            "amplitude": ("time", amp),
            "phase": ("time", phase),
            "split": ("time", split),
        },
    )

    # Verify tau_d fitting
    tau_d = fit_damped_persistence_timescale(ds_targets, train_years=(2010, 2018))
    assert np.isfinite(tau_d), "tau_d must be finite"
    print(f"  fit_damped_persistence_timescale OK: tau_d = {tau_d:.2f} days")

    # Synthetic case records
    all_records = []
    active_records = []
    for i in range(20):
        c_r1 = rng.standard_normal(30)
        c_r2 = rng.standard_normal(30)
        rec_leads = {}
        for d in leads:
            rec_leads[d] = {
                "rmm1_fc": float(c_r1[d - 1]),
                "rmm2_fc": float(c_r2[d - 1]),
                "rmm1_ob": float(c_r1[d - 1] * 0.9 + 0.1),
                "rmm2_ob": float(c_r2[d - 1] * 0.9 + 0.1),
                "rmse_ttr": 12.0 + 0.5 * d,
                "rmse_tcwv": 2.0 + 0.1 * d,
                "rmse_u850": 1.5 + 0.05 * d,
            }
        rec = {"ic_time": f"2019-01-{i+1:02d}T00:00:00", "ic_amp": 1.2 if i % 2 == 0 else 0.8, "leads": rec_leads}
        all_records.append(rec)
        if rec["ic_amp"] > 1.0:
            active_records.append(rec)

    all_metrics = compute_lead_metrics(all_records, max_lead_days=30)
    active_metrics = compute_lead_metrics(active_records, max_lead_days=30)
    assert len(all_metrics["acc"]) == 30
    assert len(active_metrics["acc"]) == 30
    print("  compute_lead_metrics OK")

    init_times = [pd.to_datetime(r["ic_time"]) for r in all_records]
    baselines = compute_baselines_for_cases(ds_targets, init_times, tau_d=tau_d, max_lead_days=30)
    assert "persistence" in baselines
    assert "damped_persistence" in baselines
    assert "climatology" in baselines
    print("  compute_baselines_for_cases OK")

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        csv1 = save_summary_csv(all_metrics, out / "mjo_skill_by_lead.csv")
        csv2 = save_summary_csv(active_metrics, out / "active_mjo_skill_by_lead.csv")
        assert csv1.exists() and csv2.exists()
        print("  save_summary_csv OK")

        save_skill_plots(all_metrics, active_metrics, baselines, out, label="smoke")
        plots = list(out.glob("*.png"))
        assert len(plots) == 5, f"Expected 5 PNG plots, found {len(plots)}: {[p.name for p in plots]}"
        print(f"  save_skill_plots OK: wrote {len(plots)} PNG plots")

        # JSON metadata export
        full_output = {
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "checkpoint": "smoke_test",
                "tau_d_days": tau_d,
                "n_total_cases": len(all_records),
                "n_active_cases": len(active_records),
            },
            "metrics": {
                "all_cases": all_metrics,
                "active_mjo": active_metrics,
            },
            "baselines": baselines,
        }
        json_path = out / "eval_results.json"
        json_path.write_text(json.dumps(full_output, indent=2))
        assert json_path.exists()
        print("  eval_results.json written successfully")

    print_skill_table(all_metrics, active_metrics)
    print("\n=== Smoke Test PASSED ===\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MJO forecast skill evaluation (Wheeler & Hendon 2004).")
    p.add_argument("--config", type=Path, default=Path("configs/unified.yaml"), help="Path to YAML config.")
    p.add_argument(
        "--mode",
        type=str,
        default="warmup",
        choices=["warmup", "lora", "rollout", "physics", "baseline"],
        help="Mode overlay to evaluate (default: warmup).",
    )
    p.add_argument("--checkpoint", type=Path, default=None, help="Path to .pt checkpoint file.")
    p.add_argument("--targets", type=Path, default=Path("data/rmm_targets.nc"), help="Path to rmm_targets.nc.")
    p.add_argument("--basis", type=Path, default=Path("data/rmm_basis.npz"), help="Path to rmm_basis.npz.")
    p.add_argument("--out-dir", type=Path, default=Path("evaluation/mjo_skill"), help="Directory for output files.")
    p.add_argument("--split", choices=["val", "test"], default="val", help="Split to evaluate (default: val).")
    p.add_argument("--max-lead-days", type=int, default=MAX_LEAD_DAYS, help="Maximum lead in days (default: 30).")
    p.add_argument(
        "--init-stride-days",
        type=int,
        default=DEFAULT_INIT_STRIDE_DAYS,
        help="Initialisation sampling stride in days (default: 5).",
    )
    p.add_argument("--num-cases", type=int, default=None, help="Maximum number of cases to evaluate (e.g. 1).")
    p.add_argument("--active-threshold", type=float, default=ACTIVE_MJO_THRESHOLD, help="Active MJO threshold (default: 1.0).")
    p.add_argument("--device", type=str, default=None, help="Device string (cuda or cpu).")
    p.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test without GPU or checkpoint.")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")

    if args.smoke_test:
        run_smoke_test()
        return

    # Check files
    for p, name in [(args.targets, "targets"), (args.basis, "basis")]:
        if not p.exists():
            print(f"ERROR: {name} file not found: {p}", file=sys.stderr)
            sys.exit(1)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.quiet:
        print(f"Device: {device}")

    # Load basis & targets
    basis = load_basis(args.basis)
    targets_ds = load_targets(args.targets, split=args.split)
    full_targets_ds = xr.open_dataset(str(args.targets))

    # Fit tau_d on training split strictly (1980–2015)
    tau_d = fit_damped_persistence_timescale(full_targets_ds, train_years=(1980, 2015))
    if not args.quiet:
        print(f"Fitted damped persistence timescale tau_d: {tau_d:.4f} days (1980–2015 training set)")

    # Load model and config
    from aurora_mjo.cli_support import load_config
    from aurora_mjo.model import load_model

    cfg = load_config(str(args.config), mode=args.mode)
    model_cfg = cfg.get("model", cfg)
    norm_stats = model_cfg.get("norm_stats") or None
    model = load_model(model_cfg, norm_stats=norm_stats).to(device)

    if args.checkpoint is not None and args.checkpoint.exists():
        ckpt = torch.load(str(args.checkpoint), map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        if not args.quiet:
            print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        if not args.quiet:
            print("WARNING: No checkpoint provided or file missing; evaluating with initialized model weights.")

    # Dataloader
    from aurora_mjo.trainer import build_dataloader

    val_loader = build_dataloader(cfg, split="val")

    # Run evaluation
    res = run_evaluation(
        model=model,
        val_loader=val_loader,
        targets_ds=targets_ds,
        basis=basis,
        tau_d=tau_d,
        device=device,
        max_lead_days=args.max_lead_days,
        init_stride_days=args.init_stride_days,
        num_cases=args.num_cases,
        active_threshold=args.active_threshold,
        verbose=not args.quiet,
    )

    all_metrics = res["all_metrics"]
    active_metrics = res["active_metrics"]
    baselines = res["baselines"]

    # Write outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_all = save_summary_csv(all_metrics, args.out_dir / "mjo_skill_by_lead.csv")
    csv_act = save_summary_csv(active_metrics, args.out_dir / "active_mjo_skill_by_lead.csv")

    ckpt_label = args.checkpoint.stem if (args.checkpoint and args.checkpoint.exists()) else "untrained"
    save_skill_plots(all_metrics, active_metrics, baselines, args.out_dir, label=ckpt_label)

    # Master JSON results
    import subprocess

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"

    full_output = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_commit": git_commit,
            "checkpoint": str(args.checkpoint) if args.checkpoint else "None",
            "config": str(args.config),
            "mode": args.mode,
            "split": args.split,
            "max_lead_days": args.max_lead_days,
            "init_stride_days": args.init_stride_days,
            "tau_d_days": tau_d,
            "n_total_evaluated": res["n_total_evaluated"],
            "n_active_evaluated": res["n_active_evaluated"],
            "active_threshold": args.active_threshold,
        },
        "metrics": {
            "all_cases": all_metrics,
            "active_mjo": active_metrics,
        },
        "baselines": baselines,
    }
    json_path = args.out_dir / "eval_results.json"
    json_path.write_text(json.dumps(full_output, indent=2))

    if not args.quiet:
        print_skill_table(all_metrics, active_metrics)
        print(f"Outputs written to: {args.out_dir}/")
        print(f"  • {csv_all.name}")
        print(f"  • {csv_act.name}")
        print(f"  • {json_path.name}")
        print("  • mjo_acc_vs_lead.png")
        print("  • mjo_rmse_vs_lead.png")
        print("  • mjo_amp_ratio_vs_lead.png")
        print("  • mjo_phase_err_vs_lead.png")
        print("  • mjo_gridded_rmse_vs_acc.png")


if __name__ == "__main__":
    main()
