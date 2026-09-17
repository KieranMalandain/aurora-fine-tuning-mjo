#!/usr/bin/env python3
# scripts/calc_norm_stats.py
"""Compute mean/std normalization statistics over the training years (1980–2015).

Thin CLI delegating computation to `aurora_mjo.stats.compute_norm_stats`.
Computes true normalisation statistics at native 1° resolution for:
  - Surface variables: 2t, 10u, 10v, msl (ps proxy), ttr (mtnlwrf), tcwv
  - Atmospheric variables: z, q, t, u, v across all 13 Aurora pressure levels

Usage:
    python scripts/calc_norm_stats.py
    python scripts/calc_norm_stats.py --years 1980 2015 --workers 32
    python scripts/calc_norm_stats.py --variables msl ttr tcwv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

import aurora_mjo.env  # noqa: F401 - set HDF5_USE_FILE_LOCKING=FALSE
from aurora_mjo.stats import compute_norm_stats

DEFAULT_ROOT = (
    "/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/"
    "prcs.ERA5/prcs.ERA5.Remap/Results"
)


def _get_git_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        default="configs/unified.yaml",
        help="Config to read data.root / data.real.train_years from.",
    )
    ap.add_argument("--root", default=None, help="Override the data root.")
    ap.add_argument(
        "--years",
        nargs=2,
        type=int,
        default=None,
        help="Override train years, e.g. --years 1980 2015.",
    )
    ap.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Subset of variables to compute, e.g. --variables msl ttr tcwv.",
    )
    ap.add_argument(
        "--out",
        default="configs/norm_stats_1980_2015.yaml",
        help="Output YAML path (default: configs/norm_stats_1980_2015.yaml).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of worker processes for parallel file processing (default: 16).",
    )
    args = ap.parse_args()

    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}

    root = Path(args.root or cfg.get("data", {}).get("root", DEFAULT_ROOT))
    if args.years:
        y0, y1 = args.years
    else:
        train_years = (
            cfg.get("data", {}).get("real", {}).get("train_years", [1980, 2015])
        )
        y0, y1 = train_years[0], train_years[1]

    print("======================================================================")
    print(f"Computing 1° Normalisation Statistics: Years {y0}–{y1}")
    print(f"Archive Root: {root}")
    print(f"Workers:      {args.workers}")
    if args.variables:
        print(f"Variables:    {args.variables}")
    else:
        print("Variables:    ALL (6 surface variables + 5 atmos variables x 13 levels)")
    print("======================================================================\n")

    results = compute_norm_stats(
        root=root,
        start_year=y0,
        end_year=y1,
        variables=args.variables,
        num_workers=args.workers,
    )

    git_sha = _get_git_sha()
    iso_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    provenance_metadata = {
        "generated_date": iso_date,
        "git_commit": git_sha,
        "year_range": [y0, y1],
        "total_files_read": results["total_files_read"],
        "archive_root": str(root),
        "grid_resolution": "1.0 degree (180x360)",
        "method": "Welford parallel reduction (Chan 1979) via aurora_mjo.stats",
    }

    output_payload = {
        "metadata": provenance_metadata,
        "surface": results["surface"],
        "atmos": results["atmos"],
        "norm_stats": results["norm_stats"],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_comment = (
        f"# Normalisation Statistics (1980–2015 Native 1° ERA5)\n"
        f"# Generated:   {iso_date}\n"
        f"# Commit:      {git_sha}\n"
        f"# Year Range:  [{y0}, {y1}]\n"
        f"# Files Read:  {results['total_files_read']}\n"
        f"# Archive:     {root}\n"
        f"# Description: Population mean and standard deviation (ddof=0) computed\n"
        f"#              via streaming Welford accumulation across all 1980–2015 6-hourly\n"
        f"#              ERA5 timesteps on the native 1° grid (180x360).\n\n"
    )

    with open(out_path, "w") as f:
        f.write(header_comment)
        yaml.safe_dump(output_payload, f, sort_keys=False)

    print("\n" + "=" * 70)
    print(f"SUMMARY OF COMPUTED STATISTICS (written to {out_path})")
    print("=" * 70)
    print("SURFACE VARIABLES:")
    for k, v in results["surface"].items():
        print(f"  {k:6s}: mean={v['mean']:12.4f}, std={v['std']:10.4f} (n={v['n']:,})")

    print("\nATMOSPHERIC VARIABLES (SAMPLE LEVELS 50, 500, 1000 hPa):")
    for var, levels in results["atmos"].items():
        for p in (50, 500, 1000):
            if p in levels:
                v = levels[p]
                m_str = f"{v['mean']:12.4f}" if abs(v['mean']) >= 0.01 else f"{v['mean']:12.6e}"
                s_str = f"{v['std']:10.4f}" if abs(v['std']) >= 0.01 else f"{v['std']:10.6e}"
                print(f"  {var}@{p:<4d}: mean={m_str}, std={s_str} (n={v['n']:,})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
