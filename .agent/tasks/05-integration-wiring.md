# Task: Config Integration & Wiring

## Objective
Merge the NERSC pathing configuration from `agent/env-portability` seamlessly into the richer `agent/training-loop` configuration schema, and update the dataset wiring to pass this path correctly.

## Working rules
- Retain the `agent/training-loop` schema as the base source of truth.
- Update `configs/phase1_baseline.yaml` to include the expected data root.
- Make the smallest possible patch to `src/trainer.py` to route this parameter.

## Read
- `configs/phase1_baseline.yaml`
- `src/dataset.py` (specifically `LANLMJODataset.__init__`)
- `src/trainer.py` (specifically where data loaders are built)

## Modify
- `configs/phase1_baseline.yaml`
- `src/trainer.py`

## Expected output
- `configs/phase1_baseline.yaml` contains `data.root: /global/cfs/...` alongside existing variables.
- `src/trainer.py` passes `root_dir=cfg["data"].get("root")` into `LANLMJODataset`.
- Verified merged `main` state without YAML conflicts.

## Steps
1. **Config Additions**: Inside `configs/phase1_baseline.yaml`, locate the `data` block and manually insert the NERSC pathing root:
   ```yaml
   data:
     use_dummy: true
     root: /global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results
     # ... [retain dummy and real split configs from training-loop branch]
   ```
2. **Trainer Wiring**: Inside `src/trainer.py`, trace where `LANLMJODataset` is instantiated. Add `root_dir=cfg["data"].get("root")` so it binds tightly to the config and no longer defaults to local environment paths silently.
3. **Commit**: Save as a consolidated commit on `integration/antigravity` resolving the cross-branch differences.
