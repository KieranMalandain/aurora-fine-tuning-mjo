#!/usr/bin/env python3
"""
scan_data_for_bad_values.py — sample a handful of files per variable and
report NaN / Inf / extreme-value counts, to find which variable(s) are
feeding the trainer non-finite or absurdly large values.

Usage (run from the repo root, with the aurora_mjo conda env active):
    python scan_data_for_bad_values.py --root /global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results \
        --years 1980 2015 --n-files-per-var 5

This does NOT import your training code — it re-implements just enough of
the file-discovery logic (mirroring src/dataset.py's SURFACE_VAR_MAP /
ATMOS_VAR_MAP) to sample real files directly, so it's safe to run standalone
on a login node without touching GPUs or the training pipeline.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

SURFACE_VAR_MAP = {
    '2t':   ('Step02/ERA5.remap_180x360MODIS_6hrInst/T2',          '*', 't2'),
    '10u':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/U10',         '*', 'u10'),
    '10v':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/V10',         '*', 'v10'),
    'msl':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/PS',          '*', 'ps'),
    'ttr':  ('Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX', '*', 'mtnlwrf'),
    'tcwv': ('Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv',        '*', 'tcwv'),
}
ATMOS_VAR_MAP = {
    'z': ('Step01/ERA5.remap_180x360MODIS_6hrInst/gopt', '*', 'z'),
    'q': ('Step01/ERA5.remap_180x360MODIS_6hrInst/sphu', '*', 'q'),
    't': ('Step01/ERA5.remap_180x360MODIS_6hrInst/tprt', '*', 't'),
    'u': ('Step01/ERA5.remap_180x360MODIS_6hrInst/uWnd', '*', 'u'),
    'v': ('Step01/ERA5.remap_180x360MODIS_6hrInst/vWnd', '*', 'v'),
}

# A generous physical sanity bound per variable, in native units, well
# outside any real value but well inside float32/bf16 overflow territory.
# Anything beyond this is almost certainly a fill/sentinel value, not physics.
SANE_ABS_MAX = {
    '2t': 400, '10u': 200, '10v': 200, 'msl': 200000, 'ttr': 2000, 'tcwv': 500,
    'z': 1e6, 'q': 1.0, 't': 400, 'u': 300, 'v': 300,
}


def scan_var(root: Path, name: str, step_subdir: str, glob_pattern: str,
             native_name: str, start_year: int, end_year: int, n_files: int):
    var_dir = root / step_subdir
    files = []
    for year in range(start_year, end_year + 1):
        year_dir = var_dir / str(year)
        matched = sorted(year_dir.glob(glob_pattern + ".nc"))
        if not matched:
            matched = sorted(var_dir.glob(f"*{year}*.nc"))
        files.extend(matched)
        if len(files) >= n_files:
            break

    if not files:
        print(f"  {name:6s}: NO FILES FOUND under {var_dir} — check root/path mapping")
        return

    files = files[:n_files]
    n_nan = n_inf = n_extreme = 0
    total = 0
    vmin, vmax = np.inf, -np.inf
    bound = SANE_ABS_MAX.get(name, 1e30)

    for f in files:
        try:
            with xr.open_dataset(str(f), engine="netcdf4") as ds:
                if native_name not in ds:
                    print(f"  {name:6s}: variable '{native_name}' not found in {f.name}, "
                          f"available: {list(ds.data_vars)[:5]}")
                    continue
                arr = ds[native_name].values.astype(np.float64)
        except Exception as e:
            print(f"  {name:6s}: FAILED to open {f.name}: {e}")
            continue

        total += arr.size
        n_nan += int(np.isnan(arr).sum())
        n_inf += int(np.isinf(arr).sum())
        finite = arr[np.isfinite(arr)]
        if finite.size:
            n_extreme += int((np.abs(finite) > bound).sum())
            vmin = min(vmin, finite.min())
            vmax = max(vmax, finite.max())

    flag = " <<< LOOK HERE" if (n_nan or n_inf or n_extreme) else ""
    print(f"  {name:6s}: n={total:>10} | nan={n_nan:>6} | inf={n_inf:>6} | "
          f"|x|>{bound:g}: {n_extreme:>6} | range=[{vmin:.4g}, {vmax:.4g}]{flag}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--years", nargs=2, type=int, default=[1980, 2015])
    p.add_argument("--n-files-per-var", type=int, default=5)
    args = p.parse_args()

    root = Path(args.root)
    start_year, end_year = args.years

    print(f"Scanning {args.n_files_per_var} file(s)/variable, years {start_year}-{end_year}")
    print(f"Root: {root}\n")

    print("Surface variables:")
    for name, (subdir, glob_pat, native) in SURFACE_VAR_MAP.items():
        scan_var(root, name, subdir, glob_pat, native, start_year, end_year, args.n_files_per_var)

    print("\nAtmospheric variables:")
    for name, (subdir, glob_pat, native) in ATMOS_VAR_MAP.items():
        scan_var(root, name, subdir, glob_pat, native, start_year, end_year, args.n_files_per_var)


if __name__ == "__main__":
    sys.exit(main())