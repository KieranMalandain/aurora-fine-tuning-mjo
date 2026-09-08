#!/usr/bin/env python3
"""Quick shape-verification script for the LANL dataset.

Instantiates the dataset, fetches one sample, and prints all tensor
shapes to confirm static_vars are 2D (H, W) as required by Aurora.

Run:
  python scripts/verify_shapes.py
"""

import sys
import os

# Same HF_HOME workaround used in train.py
if "NERSC_HOST" in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = f"/tmp/hf_home_{os.environ.get('USER', 'default')}"

import yaml
import torch

def main():
    # Load config
    cfg_path = "configs/phase1_baseline.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    print("=" * 60)
    print(" Shape Verification: LANL MJO Dataset")
    print("=" * 60)

    # Instantiate dataset with a small year range (just 1 year for speed)
    from src.dataset import LANLMJODataset
    ds = LANLMJODataset(
        start_year=2016,
        end_year=2016,
        root_dir=data_cfg.get("root"),
        slt_path=data_cfg.get("slt_path"),
    )

    # Check static vars
    print("\n--- Static Variables ---")
    all_ok = True
    for name, tensor in ds.static_vars.items():
        shape = tensor.shape
        ndim = tensor.ndim
        status = "✓" if ndim == 2 else "✗ WRONG (expected 2D)"
        if ndim != 2:
            all_ok = False
        print(f"  {name:>4s}: shape={str(shape):<20s} ndim={ndim}  {status}")

    # Fetch one sample
    print("\n--- Sample 0 Shapes ---")
    in_batch, surf_out, atmos_out = ds[0]

    print("\n  Input Batch surf_vars:")
    for k, v in in_batch.surf_vars.items():
        print(f"    {k:>5s}: {v.shape}")

    print("\n  Input Batch atmos_vars:")
    for k, v in in_batch.atmos_vars.items():
        print(f"    {k:>5s}: {v.shape}")

    print("\n  Input Batch static_vars:")
    for k, v in in_batch.static_vars.items():
        shape = v.shape
        ndim = v.ndim
        status = "✓" if ndim == 2 else "✗ WRONG"
        if ndim != 2:
            all_ok = False
        print(f"    {k:>5s}: {str(shape):<20s} {status}")

    print("\n  Metadata:")
    print(f"    lat:          {in_batch.metadata.lat.shape}")
    print(f"    lon:          {in_batch.metadata.lon.shape}")
    print(f"    time:         {in_batch.metadata.time}")
    print(f"    atmos_levels: {in_batch.metadata.atmos_levels}")

    print("\n  Target surf_out:")
    for k, v in surf_out.items():
        print(f"    {k:>5s}: {v.shape}")

    print("\n  Target atmos_out:")
    for k, v in atmos_out.items():
        print(f"    {k:>5s}: {v.shape}")

    print("\n" + "=" * 60)
    if all_ok:
        print(" ALL CHECKS PASSED ✓")
    else:
        print(" SOME CHECKS FAILED ✗")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
