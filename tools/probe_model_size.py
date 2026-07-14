#!/usr/bin/env python3
# tools/probe_model_size.py
"""
Task A3 / FIX 5 — model-size probe (AURORA_MJO_GAMEPLAN §0.3, §PART 2 FIX 5).

Runs a throwaway N-step (default 30) forward+backward+optimizer-step loop at
Aurora's REAL 0.25deg resolution (720x1440, 13 pressure levels) for either
`model_type: small` (AuroraSmallPretrained) or `model_type: huge` (full
Aurora), and reports peak GPU memory + mean seconds/step. This is the
evidence the gameplan's decision procedure needs:

    "Pick full iff it fits in memory with headroom and its per-step time
    lets baseline finish in <=~2 sessions. Otherwise small."

Why this can't just reuse `train.py --smoke-test`
===================================================
`train.py`'s smoke-test loader is deliberately TINY (H=64, W=128) - it only
exists to catch pipeline/shape bugs in ~seconds, not to measure real memory
or throughput. This probe instead builds a synthetic batch at PRODUCTION
resolution (720x1440), matching `tools/repro_ima_matrix.py`'s
`make_synth_batch` pattern, and - unlike that script - goes through the
REAL `src/model.py::load_model()` / `freeze_backbone()` path so the
measured trainable-parameter footprint, LoRA insertion, norm-stats
injection, and (once FIX 6 is in) the unfrozen `msl` head all match exactly
what Task A4's smoke test and the real baseline job will run.

No real data, no dataloader — this only needs GPU + the pretrained Aurora
checkpoint download (same as any other run on this cluster).

IMPORTANT — run each size as a SEPARATE PROCESS
=================================================
A CUDA OOM (or worse, an illegal-memory-access) can poison the whole CUDA
context for the rest of the process, same reasoning as
`tools/repro_ima_matrix.py`. Don't loop over both sizes in one Python
process; run this script twice.

Usage (inside your salloc allocation, single GPU is enough for a per-rank
memory reading — Aurora is data-parallel only, so one rank's peak memory at
batch_size=1 generalizes directly to every rank in the real 4-GPU job):

    CUDA_VISIBLE_DEVICES=0 python tools/probe_model_size.py --size small \
        --config configs/unified.yaml --mode baseline --steps 30
    CUDA_VISIBLE_DEVICES=0 python tools/probe_model_size.py --size huge  \
        --config configs/unified.yaml --mode baseline --steps 30

Then combine both results into the ~2-session decision:

    python tools/probe_model_size.py --decide \
        --config configs/unified.yaml --mode baseline

Each `--size` run writes `tools/probe_results/<size>.json`; `--decide` reads
both (if present) and prints the recommendation + the numbers behind it, so
you (Part 4 item 1: this is a human decision to ratify or override) can
write the final call into the run log.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Match production exactly (train.py sets this before any CUDA init too).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Workaround for NERSC Errno 524 filelock issue with huggingface_hub
if "NERSC_HOST" in os.environ:
    os.environ.setdefault("HF_HUB_DISABLE_FILE_LOCKS", "1")
    try:
        import filelock
        class DummyLock:
            def __init__(self, *args, **kwargs): pass
            def acquire(self, *args, **kwargs): return self
            def release(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args, **kwargs): pass
        filelock.FileLock = DummyLock
    except ImportError:
        pass

    if "HF_HOME" not in os.environ:
        scratch = os.environ.get("PSCRATCH") or os.environ.get("SCRATCH")
        if scratch:
            os.environ["HF_HOME"] = f"{scratch}/hf_home"
        else:
            os.environ["HF_HOME"] = f"/tmp/hf_home_{os.environ.get('USER', 'default')}"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).resolve().parent / "probe_results"


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Same mode-overlay merge as train.py::_deep_merge, duplicated here on
    purpose: --decide is pure config arithmetic and must not require
    importing train.py (which imports torch unconditionally at module
    level) just to read `training.epochs` / `max_steps_per_epoch`."""
    from copy import deepcopy
    out = deepcopy(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path: str, mode: str) -> dict:
    """Same semantics as train.py::load_config (mode overlay applied)."""
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    if "modes" in raw:
        modes = raw.pop("modes")
        if mode not in modes:
            raise SystemExit(f"Unknown mode {mode!r}; available: {list(modes)}")
        cfg = _deep_merge(raw, modes[mode])
        cfg.setdefault("experiment", {})["mode"] = mode
        return cfg
    return raw


def make_synth_batch(surf_vars, atmos_vars, device, H=720, W=1440, L=13):
    """Real-resolution synthetic Batch. Values are ~N(0,1) — a reasonable
    proxy for *normalized* inputs; this probe is about memory/throughput,
    not numerical correctness (that's what Task A4's smoke test verifies)."""
    import torch
    from aurora import Batch, Metadata
    from datetime import datetime

    surf = {k: torch.randn(1, 2, H, W, device=device) for k in surf_vars}
    atmos = {k: torch.randn(1, 2, L, H, W, device=device) for k in atmos_vars}
    static = {k: torch.randn(H, W, device=device) for k in ("z", "lsm", "slt")}
    meta = Metadata(
        lat=torch.linspace(90, -90, H),
        lon=torch.linspace(0, 360, W + 1)[:-1],
        time=(datetime(2016, 1, 1, 6),),
        atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
        rollout_step=0,
    )
    return Batch(surf_vars=surf, atmos_vars=atmos, static_vars=static, metadata=meta)


def run_probe(size: str, cfg: dict, steps: int, warmup: int) -> dict:
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        print("No CUDA available — this probe must run on a GPU node.")
        sys.exit(2)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    print(f"[env] torch={torch.__version__} cuda={torch.version.cuda} "
          f"dev={torch.cuda.get_device_name(0)} "
          f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")

    model_cfg = dict(cfg.get("model", {}))
    model_cfg["model_type"] = size  # "huge" -> full Aurora, else -> small
    if model_cfg.get("gradient_checkpointing", False):
        # Gameplan explicitly says not to enable this casually (known IMA,
        # handoff §2 / repro_ima_matrix.py). Force it off for the probe
        # regardless of what's in the config, and say so loudly.
        print("[probe] WARNING: gradient_checkpointing=true in config — "
              "forcing OFF for this probe (known IMA risk; see FIX 5 / "
              "tools/repro_ima_matrix.py). Re-enable only after that "
              "matrix finds a crash-free configuration.")
        model_cfg["gradient_checkpointing"] = False

    from src.model import load_model
    from src.loss import TropicalWeightedL1Loss

    norm_stats = model_cfg.get("norm_stats") or None
    print(f"\n=== Loading model_type={size!r} via the real load_model() path ===")
    model = load_model(model_cfg, norm_stats=norm_stats).to(device)

    surf_vars = tuple(model_cfg["surface_variables"])
    atmos_vars = ("z", "q", "t", "u", "v")

    train_cfg = cfg.get("training", {})
    opt_cfg = train_cfg.get("optimizer", {})
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    if not params:
        print("[probe] WARNING: zero trainable parameters — optimizer.step() "
              "will be a no-op. Check freeze_backbone/use_lora config.")
    optimizer = torch.optim.AdamW(
        params,
        lr=float(opt_cfg.get("lr", 1e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-5)),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    ) if params else None

    use_amp = bool(train_cfg.get("use_amp", True))
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

    loss_cfg = cfg.get("loss", {}).get("grid", {})
    lat_coords = torch.linspace(90, -90, 720)
    grid_loss_fn = TropicalWeightedL1Loss(
        lat_coords=lat_coords,
        tropics_bbox=loss_cfg.get("tropics_bbox", [-20, 20]),
        tropics_weight=loss_cfg.get("tropics_weight", 1.0),
        extratropics_weight=loss_cfg.get("extratropics_weight", 0.1),
    ).to(device)

    batch = make_synth_batch(surf_vars, atmos_vars, device)
    target = {**{k: torch.randn(1, 720, 1440, device=device) for k in surf_vars},
              **{k: torch.randn(1, 13, 720, 1440, device=device) for k in atmos_vars}}

    print(f"[probe] trainable params: {n_trainable:,} | use_amp={use_amp} "
          f"dtype={amp_dtype} | steps={steps} (warmup={warmup})")

    step_times = []
    nonfinite_grad_skips = 0
    torch.cuda.synchronize()
    t_all0 = time.time()
    try:
        for i in range(steps):
            t0 = time.time()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                out = model(batch)
                pred = out[0] if isinstance(out, tuple) else out
                losses = []
                for name, p in pred.surf_vars.items():
                    if name in target:
                        losses.append(grid_loss_fn(p.float(), target[name]))
                for name, p in pred.atmos_vars.items():
                    if name in target:
                        losses.append(grid_loss_fn(p.float(), target[name]))
                loss = sum(losses) / max(len(losses), 1)
            loss.backward()

            if optimizer is not None:
                # Same FIX-1 gradient-finiteness guard as production —
                # keeps the probe representative and self-protecting.
                grads_finite = all(
                    torch.isfinite(p.grad).all() for p in params if p.grad is not None
                )
                if grads_finite:
                    if max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(params, max_grad_norm)
                    optimizer.step()
                else:
                    nonfinite_grad_skips += 1

            torch.cuda.synchronize()
            dt = time.time() - t0
            if i >= warmup:
                step_times.append(dt)
            peak_gib = torch.cuda.max_memory_allocated() / 2**30
            print(f"  [step {i+1}/{steps}] loss={loss.item():.4f} "
                  f"dt={dt*1000:.0f}ms peak_mem={peak_gib:.2f} GiB"
                  + ("  (warmup)" if i < warmup else ""))
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        msg = str(e)
        if not ("out of memory" in msg.lower() or "cuda" in msg.lower()):
            raise  # not a memory-related failure — don't swallow real bugs
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        print(f"\n[probe] OOM/CUDA failure at step {len(step_times) + warmup + 1}: "
              f"{msg[:200]}\n[probe] peak memory before failure: {peak_gib:.2f} GiB")
        result = {
            "size": size, "status": "OOM", "peak_mem_gib": round(peak_gib, 2),
            "steps_completed": len(step_times) + warmup, "n_trainable_params": n_trainable,
        }
        _save_result(result)
        return result

    total_wall_s = time.time() - t_all0
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    mean_step_s = sum(step_times) / max(len(step_times), 1)

    result = {
        "size": size,
        "status": "OK",
        "peak_mem_gib": round(peak_gib, 3),
        "mean_step_s": round(mean_step_s, 4),
        "steps_measured": len(step_times),
        "steps_completed": steps,
        "n_trainable_params": n_trainable,
        "nonfinite_grad_skips": nonfinite_grad_skips,
        "total_gpu_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
    }
    print(f"\n=== RESULT size={size!r}: peak_mem={peak_gib:.2f} GiB "
          f"({result['total_gpu_gib']:.0f} GiB card) | "
          f"mean_step={mean_step_s*1000:.0f}ms | "
          f"trainable_params={n_trainable:,} | "
          f"nonfinite_grad_skips={nonfinite_grad_skips} "
          f"(elapsed {total_wall_s:.1f}s for {steps} steps) ===\n")
    _save_result(result)
    return result


def _save_result(result: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{result['size']}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[probe] wrote {path}")


def decide(cfg: dict, session_hours: float = 3.5):
    """Combine both sizes' saved results into the gameplan's decision rule:
    'Pick full iff it fits in memory with headroom and its per-step time
    lets baseline finish in <=~2 sessions. Otherwise small.'

    'Fits with headroom' here means peak_mem_gib <= 90% of the card's total
    memory — tune if you want a different margin. Total baseline work is
    read from the MERGED config (mode overlay applied): epochs *
    min(len(loader), max_steps_per_epoch) micro-steps, each ~= one probe
    step (this probe's per-step time already includes the optimizer step;
    grad-accumulation doesn't add activation memory, only more micro-steps
    per optimizer step, so this estimate is conservative-correct for wall
    time either way).
    """
    small_path = RESULTS_DIR / "small.json"
    huge_path = RESULTS_DIR / "huge.json"
    if not small_path.exists() or not huge_path.exists():
        missing = [p.name for p in (small_path, huge_path) if not p.exists()]
        print(f"[decide] Missing result file(s): {missing}. Run both sizes first:")
        print("  python tools/probe_model_size.py --size small --config ... --mode ...")
        print("  python tools/probe_model_size.py --size huge  --config ... --mode ...")
        sys.exit(1)

    small = json.load(open(small_path))
    huge = json.load(open(huge_path))

    epochs = int(cfg.get("training", {}).get("epochs", 3))
    max_spe = cfg.get("training", {}).get("max_steps_per_epoch")
    # We don't have the real dataloader length here (no data access needed
    # for this probe); max_steps_per_epoch is the effective cap in
    # unified.yaml for every mode, so use it directly.
    steps_per_epoch = int(max_spe) if max_spe else 2500
    total_micro_steps = epochs * steps_per_epoch
    session_budget_s = session_hours * 3600

    print(f"[decide] baseline plan: epochs={epochs} x steps/epoch={steps_per_epoch} "
          f"= {total_micro_steps:,} micro-steps | session budget ~{session_hours}h "
          f"({session_budget_s:.0f}s, leaves ~30min/4h margin for checkpointing/setup)\n")

    for r in (small, huge):
        if r.get("status") != "OK":
            print(f"  {r['size']:>5}: status={r.get('status')} "
                  f"(peak_mem={r.get('peak_mem_gib','?')} GiB before failure) "
                  f"— DISQUALIFIED, does not fit.")
            continue
        total_s = total_micro_steps * r["mean_step_s"]
        sessions = -(-total_s // session_budget_s)  # ceil
        headroom = 1.0 - r["peak_mem_gib"] / r["total_gpu_gib"]
        fits = headroom >= 0.10
        print(f"  {r['size']:>5}: peak_mem={r['peak_mem_gib']:.1f}/{r['total_gpu_gib']:.0f} GiB "
              f"({headroom*100:.0f}% headroom, {'FITS' if fits else 'TOO TIGHT'}) | "
              f"mean_step={r['mean_step_s']*1000:.0f}ms -> "
              f"~{total_s/3600:.1f}h total -> ~{int(sessions)} session(s) | "
              f"trainable_params={r['n_trainable_params']:,}")

    print("\n[decide] Recommendation (ratify or override — Part 4 item 1 is a "
          "human decision):")
    if huge.get("status") == "OK":
        headroom = 1.0 - huge["peak_mem_gib"] / huge["total_gpu_gib"]
        total_s = total_micro_steps * huge["mean_step_s"]
        sessions = -(-total_s // session_budget_s)
        if headroom >= 0.10 and sessions <= 2:
            print("  -> FULL ('huge'). Fits with headroom and finishes baseline "
                  "in <=2 sessions. Full also has stabilise_level_agg=True by "
                  "default (more NaN-resistant) — prefer it per §0.3.")
            return "huge"
        print(f"  -> full does not clear the bar (headroom={headroom*100:.0f}%, "
              f"~{int(sessions)} sessions needed) — falling back to SMALL, "
              "the guaranteed-deliverable option.")
    else:
        print("  -> full OOM'd or wasn't run — falling back to SMALL.")
    return "small"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=["small", "huge"], default=None,
                     help="Which model size to probe. Omit only when using --decide.")
    ap.add_argument("--config", default="configs/unified.yaml")
    ap.add_argument("--mode", default="baseline",
                     choices=["baseline", "physics_informed", "lora", "combined"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3,
                     help="Steps excluded from the mean-step-time average "
                          "(cuDNN/attention-kernel autotuning), but still "
                          "counted toward peak memory.")
    ap.add_argument("--decide", action="store_true",
                     help="Skip probing; combine tools/probe_results/{small,huge}.json "
                          "into a recommendation.")
    ap.add_argument("--session-hours", type=float, default=3.5,
                     help="Effective training time per interactive session "
                          "(4h alloc minus setup/checkpoint margin).")
    args = ap.parse_args()

    cfg = load_config(args.config, args.mode)

    if args.decide:
        decide(cfg, session_hours=args.session_hours)
        return
    if args.size is None:
        ap.error("--size is required unless --decide is given")
    run_probe(args.size, cfg, steps=args.steps, warmup=args.warmup)


if __name__ == "__main__":
    main()