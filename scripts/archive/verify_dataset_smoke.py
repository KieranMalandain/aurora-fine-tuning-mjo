#!/usr/bin/env python3
"""Archived verification smoke test for LANLMJODataset.

HISTORICAL CONTEXT:
This script was originally `test_dataset.py` at the repository root.
It verified basic initialization of LANLMJODataset for year 2000 against CFS,
checking static variable shapes ('z', 'lsm') and sample count.

SUPERSEDED BY:
Superseded by `scripts/verify_dataset_loader.py`, which performs strictly more
comprehensive verification across dimensions, statistics, and batch formatting.
Renamed and moved to `scripts/archive/verify_dataset_smoke.py` because `test_*.py`
at the repo root or scripts/ would be collected by pytest and fail (calling sys.exit(1)
and requiring CFS data without pytest markers).

Task D2 converts the real check into a proper offline test fixture in `tests/`.
"""

import sys

try:
    from aurora_mjo.dataset import LANLMJODataset

    # Provide root_dir explicitly to avoid the default warning
    # and restrict to a single year to speed up the lazy loading test.
    dataset = LANLMJODataset(
        start_year=2000,
        end_year=2000,
        root_dir="/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results",
    )
    print("Dataset initialized successfully.")
    print("Static Vars Z shape:", dataset.static_vars["z"].shape)
    print("Static Vars LSM shape:", dataset.static_vars["lsm"].shape)
    print("Number of samples:", len(dataset))
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
