#!/usr/bin/env python3
# scripts/calc_norm_stats.py
"""
Compute mean/std normalization statistics, over the TRAINING years only,
for the surface variables that need them injected into Aurora's global
normalisation tables (`aurora.normalisation.locations` / `.scales`):

  - ttr, tcwv: injected variables with no pretrained Aurora entry at all.
  - msl:       AURORA_MJO_GAMEPLAN FIX 2. dataset.py maps 'msl' to LANL's
               surface-pressure ('ps') field as a proxy (true MSL is not in
               the LANL archive), but Aurora's *built-in* 'msl' stats are
               MSL-calibrated (mean~100958, scale~1332). Using them on ps
               values drives every high-terrain input to ~-36 sigma -
               the confirmed trigger for the 100%-non-finite validation
               (see AURORA_MJO_GAMEPLAN.md §Finding 1). This script computes
               the correct ps-based mean/std so it can override that entry.

RECONSTRUCTION NOTE
====================
This file was not present in the uploaded working set handed to the worker
agent; only the AURORA_MJO_GAMEPLAN.md description of the diff (FIX 2a) was
available. This is therefore a from-scratch reconstruction, not a literal
patch of the real script. It reuses the exact same file-discovery pattern
already proven working in `src/dataset.py::_collect_var_files` and
`tools/diagnose_val_nan.py::_collect`, and the ttr/tcwv path/native-name
pairs match the values already trusted and in use in `configs/unified.yaml`
(mean=-226.0498/std=49.2158 for ttr, mean=18.2967/std=16.3265 for tcwv) as a
sanity check once it's run for real on NERSC. **Before trusting the msl
output for a real training run, diff this file against the original on
disk (if it still exists in the repo) and reconcile any differences.**

Usage (Task A2, from an allocation with NERSC filesystem access):

    python scripts/calc_norm_stats.py --config configs/unified.yaml

    # Override the years or a single variable for a quick check:
    python scripts/calc_norm_stats.py --config configs/unified.yaml \
        --variables msl --years 1980 2015

Output: prints computed mean/std per variable AND writes them to
`scripts/computed_norm_stats.yaml` as a ready-to-paste snippet for
`configs/unified.yaml`'s `model.norm_stats` block.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

DEFAULT_ROOT = ("/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/"
                 "prcs.ERA5/prcs.ERA5.Remap/Results")

# {aurora_name: (step_subdir, native_var_name)} — must match src/dataset.py's
# SURFACE_VAR_MAP path/native-name pairs exactly, or the computed stats won't
# match what the dataloader actually feeds the model.
VARIABLES_TO_CALC = {
    'msl':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/PS',          'ps'),
    'ttr':  ('Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX', 'mtnlwrf'),
    'tcwv': ('Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv',        'tcwv'),
}


def _collect_files(root: Path, step_subdir: str, y0: int, y1: int) -> list[Path]:
    """Same over-match-then-filter discovery pattern as dataset.py /
    diagnose_val_nan.py: try the per-year subdirectory first, fall back to a
    root-level glob containing the year string."""
    var_dir = root / step_subdir
    files: list[Path] = []
    for year in range(y0, y1 + 1):
        year_dir = var_dir / str(year)
        matched = sorted(year_dir.glob("*.nc"))
        if not matched:
            matched = sorted(var_dir.glob(f"*{year}*.nc"))
        files.extend(matched)
    seen, uniq = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


class _Welford:
    """Numerically-stable single-pass mean/variance accumulator.

    Streams file-by-file so we never hold more than one file's array in
    memory at a time - required at this data volume (36 years x ~1460
    timesteps x 180x360 per variable).
    """
    __slots__ = ("n", "mean", "m2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update_batch(self, values: np.ndarray):
        values = values.astype(np.float64).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        batch_n = values.size
        batch_mean = float(values.mean())  # native float — numpy scalars poison
        batch_var = float(values.var())    # self.mean/m2 into np.float64 forever
        # otherwise, which yaml.safe_dump cannot serialize downstream.

        new_n = self.n + batch_n
        delta = batch_mean - self.mean
        self.mean += delta * batch_n / new_n
        self.m2 += batch_var * batch_n + delta**2 * self.n * batch_n / new_n
        self.n = new_n

    @property
    def std(self) -> float:
        return float(np.sqrt(self.m2 / max(self.n, 1)))


def compute_stats(root: Path, y0: int, y1: int, variables: dict) -> dict:
    import xarray as xr  # deferred: heavy import, only needed for real runs

    results = {}
    for aurora_name, (step_subdir, native_name) in variables.items():
        files = _collect_files(root, step_subdir, y0, y1)
        if not files:
            print(f"  [{aurora_name}] WARNING: no files found under "
                  f"{root / step_subdir} for {y0}-{y1}; skipping.")
            continue
        acc = _Welford()
        for i, f in enumerate(files):
            with xr.open_dataset(str(f), engine="netcdf4") as ds:
                if native_name not in ds:
                    print(f"  [{aurora_name}] WARNING: '{native_name}' not in "
                          f"{f.name}; skipping this file.")
                    continue
                arr = ds[native_name].values
            acc.update_batch(arr)
            if (i + 1) % 20 == 0 or (i + 1) == len(files):
                print(f"  [{aurora_name}] {i+1}/{len(files)} files | "
                      f"running mean={acc.mean:.4f} std={acc.std:.4f}")
        results[aurora_name] = {"mean": round(float(acc.mean), 4), "std": round(float(acc.std), 4)}
        print(f"  [{aurora_name}] FINAL: mean={acc.mean:.4f} std={acc.std:.4f} "
              f"(n={acc.n:,} finite values, native='{native_name}')")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/unified.yaml",
                     help="Config to read data.root / data.real.train_years from.")
    ap.add_argument("--root", default=None, help="Override the data root.")
    ap.add_argument("--years", nargs=2, type=int, default=None,
                     help="Override train years, e.g. --years 1980 2015.")
    ap.add_argument("--variables", nargs="+", default=None,
                     help="Subset of VARIABLES_TO_CALC to compute, e.g. --variables msl.")
    ap.add_argument("--out", default="scripts/computed_norm_stats.yaml")
    args = ap.parse_args()

    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        print(f"WARNING: config '{args.config}' not found; using CLI overrides / defaults only.")

    root = Path(args.root or cfg.get("data", {}).get("root", DEFAULT_ROOT))
    if args.years:
        y0, y1 = args.years
    else:
        train_years = cfg.get("data", {}).get("real", {}).get("train_years", [1980, 2015])
        y0, y1 = train_years[0], train_years[1]

    variables = VARIABLES_TO_CALC
    if args.variables:
        variables = {k: v for k, v in VARIABLES_TO_CALC.items() if k in args.variables}
        missing = set(args.variables) - set(variables)
        if missing:
            print(f"WARNING: unknown variable(s) requested and ignored: {missing}")

    print(f"Computing norm stats over TRAINING years {y0}-{y1} from root={root}")
    print(f"Variables: {list(variables)}\n")

    results = compute_stats(root, y0, y1, variables)

    print("\n" + "=" * 70)
    print("RESULTS - paste into configs/unified.yaml's model.norm_stats block:")
    print("=" * 70)
    for name, stats in results.items():
        print(f"    {name}:  {{ mean: {stats['mean']}, std: {stats['std']} }}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump({"norm_stats": results}, f, sort_keys=False)
    print(f"\nAlso written to {out_path}")

    if "msl" not in results:
        print("\nNOTE: msl stats were not computed. Training will keep using the "
              "FIX-2 fallback (Aurora sp stats: mean=96647.375, std=9586.6914) "
              "already placed in configs/unified.yaml until you run this "
              "script successfully and paste the real numbers in.")


if __name__ == "__main__":
    sys.exit(main())