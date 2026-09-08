#!/usr/bin/env python3
# tools/diagnose_val_nan.py
"""
One-shot diagnostic for the 100%-non-finite validation losses (handoff §3b).

Settles hypothesis A (bad 2016–2019 data) vs hypothesis B (cross-variable
index misalignment) on the REAL data, plus quantifies the train-range
leakage, in a single login-node/interactive run.  No GPU needed except for
the optional --forward stage.

Stages (each prints PASS/FAIL/FINDINGS):

  1. FILE & TIMESTAMP AUDIT (per variable, per file):
       n_files, per-file timestep counts, first/last timestamps,
       out-of-year-range timesteps (the leakage mechanism), duplicates,
       and cross-variable agreement of the full timestamp sets.
       -> Hypothesis B is CONFIRMED if any variable's timestamp set differs.
       -> Leakage is CONFIRMED if any timestamps fall outside the range.

  2. RAW VALUE SCAN over the requested years (default: the val years, which
     have NEVER been scanned): NaN / Inf / |x|>threshold counts and
     min/max/mean per variable, sampled across files.
       -> Hypothesis A is CONFIRMED if val-year files contain NaN/Inf or
          absurd magnitudes that the train years lack.

  3. DATASET-PATH SAMPLE CHECK: instantiates the (v3, timestamp-aligned)
     LANLMJODataset over the same years, loads N samples end-to-end through
     the exact production code path, prints per-variable min/max of inputs
     and targets.
       -> If stages 1–2 are clean but this stage shows garbage, the bug is
          in the dataset code, not the data.

  4. (--forward, needs GPU + checkpoint) Loads the model, runs ONE val
     sample forward, prints the per-variable grid-loss contribution.
       -> Pinpoints which variable's LOSS goes non-finite when data and
          indexing are both clean (e.g. normalization-stats issues).

Usage on Perlmutter (from the worktree root, aurora_mjo env active):

  # The two decisive checks (fast, login node OK, ~5–10 min):
  python tools/diagnose_val_nan.py --years 2016 2019
  # Compare against a clean train slice:
  python tools/diagnose_val_nan.py --years 1984 1985 --skip-dataset
  # Optional model-side stage on a GPU node:
  python tools/diagnose_val_nan.py --years 2016 2019 --forward \
      --checkpoint checkpoints/baseline/best.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_ROOT = ("/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/"
                "prcs.ERA5/prcs.ERA5.Remap/Results")
DEFAULT_SLT = "/pscratch/sd/k/kam352/Aurora/slt/slt_data.nc"

# Loose physical sanity bounds per native variable (values OUTSIDE these are
# counted as "extreme" - generous on purpose; we only want to catch garbage
# like fill values, wrong units, or wrong scale/offset decoding).
SANITY = {
    't2':      (150.0, 350.0),      # K
    'u10':     (-120.0, 120.0),     # m/s
    'v10':     (-120.0, 120.0),
    'ps':      (3.0e4, 1.15e5),     # Pa
    'mtnlwrf': (-600.0, 100.0),     # W/m2 (OLR, negative-up convention)
    'tcwv':    (0.0, 120.0),        # kg/m2
    'z':       (-6.0e3, 5.0e5),     # m2/s2 geopotential
    'q':       (-1e-3, 0.06),       # kg/kg
    't':       (150.0, 350.0),      # K
    'u':       (-250.0, 250.0),     # m/s
    'v':       (-250.0, 250.0),
}


def _collect(var_map, root: Path, y0: int, y1: int):
    out = {}
    for name, (sub, pat, native) in var_map.items():
        d = root / sub
        files = []
        for year in range(y0, y1 + 1):
            m = sorted((d / str(year)).glob(pat + ".nc"))
            if not m:
                m = sorted(d.glob(f"*{year}*.nc"))
            files.extend(m)
        seen, uniq = set(), []
        for f in files:
            if f not in seen:
                seen.add(f)
                uniq.append(f)
        out[name] = (uniq, native)
    return out


def stage1_audit(file_maps: dict, y0: int, y1: int):
    import xarray as xr
    print("\n" + "=" * 78)
    print("STAGE 1 - FILE & TIMESTAMP AUDIT")
    print("=" * 78)
    t_min = np.datetime64(f"{y0}-01-01T00:00:00", "s").astype("int64")
    t_max = np.datetime64(f"{y1}-12-31T23:59:59", "s").astype("int64")

    ts_sets, problems = {}, []
    for name, (files, native) in file_maps.items():
        if not files:
            problems.append(f"{name}: NO FILES FOUND")
            print(f"  {name:>5}: NO FILES FOUND  <-- variable would be silently skipped!")
            continue
        all_ts, oob, dupes = set(), 0, 0
        per_file = []
        for f in files:
            with xr.open_dataset(str(f), engine="netcdf4") as ds:
                t = np.asarray(ds.time.values)
            secs = t.astype("datetime64[s]").astype("int64")
            per_file.append((f.name, len(secs),
                             str(t.min())[:16], str(t.max())[:16]))
            for s in secs.tolist():
                if s < t_min or s > t_max:
                    oob += 1
                elif s in all_ts:
                    dupes += 1
                else:
                    all_ts.add(s)
        ts_sets[name] = all_ts
        flag = ""
        if oob:
            flag += f"  <-- {oob} TIMESTEPS OUTSIDE {y0}-{y1} (LEAKAGE MECHANISM)"
            problems.append(f"{name}: {oob} out-of-range timesteps")
        if dupes:
            flag += f"  <-- {dupes} DUPLICATE timestamps"
            problems.append(f"{name}: {dupes} duplicates")
        print(f"  {name:>5}: files={len(files):<4} in-range-steps={len(all_ts):<7}{flag}")
        for fn, n, lo, hi in per_file[:3] + ([("...", "", "", "")] if len(per_file) > 6 else []) + per_file[-3:] if len(per_file) > 6 else per_file:
            if fn == "...":
                print("           ...")
            else:
                print(f"           {fn}: {n} steps [{lo} .. {hi}]")

    # Cross-variable agreement
    if ts_sets:
        counts = {n: len(s) for n, s in ts_sets.items()}
        ref_name = max(counts, key=counts.get)
        ref = ts_sets[ref_name]
        inter = set.intersection(*ts_sets.values())
        print(f"\n  Cross-variable: intersection={len(inter)} steps")
        aligned = True
        for n, s in ts_sets.items():
            missing = len(ref - s)
            extra = len(s - ref)
            if missing or extra:
                aligned = False
                print(f"    {n:>5}: MISSING {missing} / EXTRA {extra} vs {ref_name}"
                      f"  <-- HYPOTHESIS B (misalignment) CONFIRMED for this variable")
                problems.append(f"{n}: timestamp set differs from {ref_name}")
        if aligned:
            print("    All variables share an identical timestamp set -> "
                  "hypothesis B (cross-variable misalignment) REFUTED for this range.")
    return problems


def stage2_scan(file_maps: dict, files_per_var: int, problems: list):
    import xarray as xr
    print("\n" + "=" * 78)
    print(f"STAGE 2 - RAW VALUE SCAN (up to {files_per_var} files/var, evenly spaced)")
    print("=" * 78)
    for name, (files, native) in file_maps.items():
        if not files:
            continue
        idx = np.unique(np.linspace(0, len(files) - 1,
                                    min(files_per_var, len(files))).round().astype(int))
        n_nan = n_inf = n_ext = n_tot = 0
        vmin, vmax, vsum = np.inf, -np.inf, 0.0
        lo, hi = SANITY.get(native, (-np.inf, np.inf))
        worst_file = None
        for i in idx:
            with xr.open_dataset(str(files[i]), engine="netcdf4") as ds:
                a = ds[native].values
            a = np.asarray(a, dtype=np.float64)
            nn = int(np.isnan(a).sum())
            ni = int(np.isinf(a).sum())
            fin = a[np.isfinite(a)]
            ne = int(((fin < lo) | (fin > hi)).sum()) if fin.size else 0
            if (nn or ni or ne) and worst_file is None:
                worst_file = files[i].name
            n_nan += nn; n_inf += ni; n_ext += ne; n_tot += a.size
            if fin.size:
                vmin = min(vmin, float(fin.min()))
                vmax = max(vmax, float(fin.max()))
                vsum += float(fin.mean())
        mean = vsum / max(len(idx), 1)
        flag = ""
        if n_nan or n_inf:
            flag = f"  <-- NON-FINITE VALUES (first seen: {worst_file}) - HYPOTHESIS A EVIDENCE"
            problems.append(f"{name}: {n_nan} NaN / {n_inf} Inf in scanned files")
        elif n_ext:
            flag = f"  <-- {n_ext} values outside sane range ({worst_file})"
            problems.append(f"{name}: {n_ext} extreme values")
        print(f"  {name:>5} ({native:>8}): scanned {len(idx)} files, {n_tot/1e6:.0f}M vals | "
              f"NaN={n_nan} Inf={n_inf} extreme={n_ext} | "
              f"range=[{vmin:.4g}, {vmax:.4g}] mean~{mean:.4g}{flag}")
    return problems


def stage3_dataset(args, problems: list):
    print("\n" + "=" * 78)
    print(f"STAGE 3 - DATASET-PATH SAMPLE CHECK ({args.n_samples} samples via LANLMJODataset)")
    print("=" * 78)
    import torch
    from src.dataset import LANLMJODataset
    ds = LANLMJODataset(start_year=args.years[0], end_year=args.years[1],
                        root_dir=args.root, slt_path=args.slt)
    idx = np.unique(np.linspace(0, len(ds) - 1,
                                min(args.n_samples, len(ds))).round().astype(int))
    stats = defaultdict(lambda: [np.inf, -np.inf, 0])
    bad = 0
    for i in idx:
        in_batch, surf_t, atmos_t = ds[int(i)]
        groups = [("in.surf", in_batch.surf_vars), ("in.atmos", in_batch.atmos_vars),
                  ("tgt.surf", surf_t[0]), ("tgt.atmos", atmos_t[0])]
        for prefix, g in groups:
            for name, t in g.items():
                key = f"{prefix}.{name}"
                fin = torch.isfinite(t)
                if not fin.all():
                    stats[key][2] += int((~fin).sum())
                    bad += 1
                tt = t[fin]
                if tt.numel():
                    stats[key][0] = min(stats[key][0], float(tt.min()))
                    stats[key][1] = max(stats[key][1], float(tt.max()))
    for key in sorted(stats):
        lo, hi, nf = stats[key]
        flag = f"  <-- {nf} NON-FINITE (should be impossible post-process_var!)" if nf else ""
        print(f"  {key:>14}: range=[{lo:.4g}, {hi:.4g}]{flag}")
    if bad:
        problems.append(f"dataset path emitted {bad} tensors with non-finite values")
    else:
        print(f"  All {len(idx)} samples finite through the production read path.")
    return problems


def stage4_forward(args, problems: list):
    print("\n" + "=" * 78)
    print("STAGE 4 - SINGLE-SAMPLE FORWARD + PER-VARIABLE LOSS BREAKDOWN")
    print("=" * 78)
    import torch
    from src.dataset import LANLMJODataset
    from src.model import load_model
    from src.trainer import Trainer, build_dataloader  # noqa
    import yaml
    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.get("model", {}), device)
    if args.checkpoint:
        from src.checkpoint import CheckpointManager
        CheckpointManager.load(args.checkpoint, model)
    model.eval()
    ds = LANLMJODataset(start_year=args.years[0], end_year=args.years[1],
                        root_dir=args.root, slt_path=args.slt)
    from src.trainer import _upsample_batch_gpu, _align_shapes
    in_b, s_t, a_t = ds[0]
    in_b, tgts = _upsample_batch_gpu(in_b, s_t, a_t, device)
    for g in (in_b.surf_vars, in_b.atmos_vars, in_b.static_vars):
        for k in g:
            g[k] = g[k].to(device).float().contiguous()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        out = model(in_b)
    pred = out[0] if isinstance(out, tuple) else out
    tgt = tgts[0]
    print(f"  init_time={in_b.metadata.time}")
    for attr in ("surf_vars", "atmos_vars"):
        for name, p in getattr(pred, attr).items():
            if name not in tgt:
                continue
            t = tgt[name].to(device).float()
            p2, t2 = _align_shapes(p.float(), t)
            l1 = (p2 - t2).abs().mean().item()
            pf = torch.isfinite(p2).all().item()
            flag = "" if (np.isfinite(l1) and pf) else "  <-- NON-FINITE HERE"
            print(f"  {attr[:-5]:>5}.{name:>5}: pred_range=[{p2.min():.4g},{p2.max():.4g}] "
                  f"pred_finite={pf} L1={l1:.4g}{flag}")
            if not (np.isfinite(l1) and pf):
                problems.append(f"forward: {name} produced non-finite output/loss")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs=2, type=int, default=[2016, 2019])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--slt", default=DEFAULT_SLT)
    ap.add_argument("--files-per-var", type=int, default=6)
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--skip-dataset", action="store_true")
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--config", default="configs/unified.yaml")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    from src.dataset import SURFACE_VAR_MAP, ATMOS_VAR_MAP
    root = Path(args.root)
    fmap = {**_collect(SURFACE_VAR_MAP, root, *args.years),
            **_collect(ATMOS_VAR_MAP, root, *args.years)}

    problems: list[str] = []
    problems = stage1_audit(fmap, *args.years)
    problems = stage2_scan(fmap, args.files_per_var, problems)
    if not args.skip_dataset:
        problems = stage3_dataset(args, problems)
    if args.forward:
        problems = stage4_forward(args, problems)

    print("\n" + "=" * 78)
    if problems:
        print(f"VERDICT: {len(problems)} FINDING(S):")
        for p in problems:
            print(f"  * {p}")
        print("\nInterpretation guide:")
        print("  - out-of-range timesteps        -> leakage / glob over-match (fixed by v3 dataset)")
        print("  - timestamp set differs         -> hypothesis B confirmed (fixed by v3 dataset)")
        print("  - NaN/Inf/extremes in raw scan  -> hypothesis A confirmed (v3 zeroes them; consider")
        print("                                     excluding affected files/years from val)")
        print("  - non-finite only at stage 4    -> model/normalization-side, not data")
        sys.exit(1)
    print("VERDICT: CLEAN - no leakage, no misalignment, no bad values in the scanned range.")
    sys.exit(0)


if __name__ == "__main__":
    main()
