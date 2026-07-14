#!/usr/bin/env python3
# tools/repro_ima_matrix.py
"""
Minimal reproducer + experiment matrix for the activation-checkpointing
"CUDA illegal memory access" (handoff §2).

Uses a SYNTHETIC batch (no NetCDF, no dataloader - sidesteps the HDF5
friction that broke the compute-sanitizer attempt) at the real 720x1440
resolution, real AuroraSmallPretrained, real freeze config, bf16 autocast,
one forward+backward.  The baseline experiment (E0) should crash exactly
like production; each subsequent experiment flips ONE variable.

Why these five experiments (ordered by prior x cost):

  E0  ckpt=on, everything as production        -> must CRASH (validates repro)
  E1  ckpt=on, SDPA pinned to MATH backend     -> Aurora's Swin3D attention
      calls F.scaled_dot_product_attention with an explicit float mask
      built by .repeat().reshape(); float masks dispatch to the
      MEMORY-EFFICIENT backend, whose backward is a stride-sensitive CUDA
      kernel with a documented history of IMAs, and checkpointing re-runs
      that forward inside the backward pass on freshly re-allocated
      tensors.  If E1 is clean: set training.sdpa_backend: "math" in the
      YAML and full-BPTT + checkpointing is usable again (~1.3–2x slower
      attention).
  E2  ckpt=on, expandable_segments REMOVED     -> PYTORCH_CUDA_ALLOC_CONF=
      expandable_segments:True (introduced in the v2 drop's SLURM script)
      changes allocation layout under recompute and has its own history of
      rare IMA interactions.  If E2 is clean: delete the env var from
      train_auto.slurm.
  E3  ckpt=on, fp32 (no autocast)              -> autocast-replay-under-
      recompute interplay.  Diagnostic more than a fix (fp32 memory won't
      fit k>=2 anyway).
  E4  ckpt=on, REENTRANT wrapper               -> swaps Aurora's
      NO_REENTRANT checkpoint impl for the legacy reentrant one (different
      recompute machinery entirely).  If clean: use our wrapper instead of
      model.configure_activation_checkpointing().
  E5  ckpt=on, backbone UNFROZEN               -> isolates the unusual
      "checkpointed region contains zero trainable params but its input
      requires grad" configuration.  Diagnostic only.

Run each experiment as a separate process (a CUDA IMA poisons the context):

  for E in 0 1 2 3 4 5; do
    python tools/repro_ima_matrix.py --exp $E ; echo "E$E exit=$?"
  done
  # add --sanitizer to wrap the crashing E0 in compute-sanitizer:
  compute-sanitizer --tool memcheck python tools/repro_ima_matrix.py --exp 0

Exit code 0 = ran clean, 1 = crashed (caught), other = hard crash.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_synth_batch(device, H=720, W=1440, L=13):
    import torch
    from aurora import Batch, Metadata
    surf = {k: torch.randn(1, 2, H, W, device=device)
            for k in ("2t", "10u", "10v", "msl", "ttr", "tcwv")}
    atmos = {k: torch.randn(1, 2, L, H, W, device=device)
             for k in ("z", "q", "t", "u", "v")}
    static = {k: torch.randn(H, W, device=device) for k in ("z", "lsm", "slt")}
    meta = Metadata(
        lat=torch.linspace(90, -90, H),
        lon=torch.linspace(0, 360, W + 1)[:-1],
        time=(datetime(2016, 1, 1, 6),),
        atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
        rollout_step=0,
    )
    return Batch(surf_vars=surf, atmos_vars=atmos, static_vars=static, metadata=meta)


def apply_reentrant_checkpointing(model):
    """Wrap the same module set Aurora targets, but with CheckpointImpl.REENTRANT."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl)
    import functools
    names = {"Basic3DDecoderLayer", "Basic3DEncoderLayer", "LinearPatchReconstruction",
             "Perceiver3DDecoder", "Perceiver3DEncoder", "Swin3DTransformerBackbone",
             "Swin3DTransformerBlock"}
    wrapper = functools.partial(checkpoint_wrapper,
                                checkpoint_impl=CheckpointImpl.REENTRANT)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=wrapper,
        check_fn=lambda m: m.__class__.__name__ in names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, required=True, choices=range(6))
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--w", type=int, default=1440)
    ap.add_argument("--steps", type=int, default=3, help="fwd+bwd iterations")
    args = ap.parse_args()

    # E2 must be decided BEFORE torch initializes CUDA.
    if args.exp == 2:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        print("[E2] PYTORCH_CUDA_ALLOC_CONF removed from env")
    else:
        # Match production (train_auto.slurm sets this).
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from aurora import AuroraSmallPretrained

    if not torch.cuda.is_available():
        print("No CUDA - this reproducer must run on a GPU node.")
        sys.exit(2)
    device = torch.device("cuda")
    print(f"[env] torch={torch.__version__} cuda={torch.version.cuda} "
          f"dev={torch.cuda.get_device_name(0)} "
          f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")

    if args.exp == 1:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[E1] SDPA pinned to MATH backend")

    use_amp = args.exp != 3
    if args.exp == 3:
        print("[E3] autocast DISABLED (fp32)")

    model = AuroraSmallPretrained()
    model.load_checkpoint(strict=False)
    model = model.to(device)

    # Freeze exactly like production baseline: everything except a couple of
    # small leaf modules, so the checkpointed backbone holds zero trainable
    # params while its input still requires grad.
    if args.exp != 5:
        for p in model.parameters():
            p.requires_grad = False
        n_unfrozen = 0
        for name, p in model.named_parameters():
            if ("surf_heads" in name and (".ttr" in name or ".tcwv" in name)) \
                    or ("surf_token_embeds" in name and (".ttr" in name or ".tcwv" in name)):
                p.requires_grad = True
                n_unfrozen += p.numel()
        print(f"[freeze] production-style: {n_unfrozen} trainable params")
    else:
        print("[E5] backbone UNFROZEN (all params trainable)")

    if args.exp == 4:
        apply_reentrant_checkpointing(model)
        print("[E4] REENTRANT checkpoint wrapper applied")
    else:
        model.configure_activation_checkpointing()
        print("[ckpt] Aurora default (NO_REENTRANT) checkpointing applied")

    torch.manual_seed(0)
    batch = make_synth_batch(device, H=args.h, W=args.w)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4) if params else None

    try:
        for i in range(args.steps):
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                pred = model(batch)
                loss = sum(v.float().abs().mean() for v in pred.surf_vars.values()) + \
                       sum(v.float().abs().mean() for v in pred.atmos_vars.values())
            loss.backward()
            if opt:
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(f"[step {i+1}/{args.steps}] loss={loss.item():.4f} "
                  f"peak_mem={mem:.1f} GiB  OK")
        print(f"\nEXPERIMENT E{args.exp}: CLEAN - no crash in {args.steps} fwd+bwd steps.")
        sys.exit(0)
    except RuntimeError as e:
        print(f"\nEXPERIMENT E{args.exp}: CRASHED - {str(e)[:400]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
