# Task: Comprehensive Smoke Tests

## Objective

Create and execute a comprehensive suite of smoke tests covering all major pipeline entry points in the repository. The goal is to verify that there are no remaining compilation, dimension, or architectural errors before queuing Phase 1 training on Perlmutter. Results must be meticulously documented.

## Working rules
- Follow standard execution protocols: activate the `aurora_mjo` conda environment before running.
- Do not attempt to fix the codebase during this task; if a smoke test fails, record the failure and move to the next.
- Make the smallest possible coherent patch if you absolutely must fix a script to get a valid test result, but prefer to just document the failure.
- Ensure conda/mamba environment `aurora_mjo` is activated before running Python code (directory `/pscratch/sd/k/kam352/conda/envs/aurora_mjo/bin/python`).

## Read

- `train.py`
- `scripts/smoke_test_mjo_head.py`
- `scripts/smoke_test_freeze.py`
- `scripts/smoke_test_rollout.py`
- `scripts/verify_dataset_loader.py`
- `scripts/compute_rmm.py`
- `scripts/evaluate_mjo.py`

## Modify

- `docs/smoke_results.md` (create if it does not exist)

## Expected output

- `docs/smoke_results.md` is populated with a meticulous log of all smoke test executions, capturing the command run, timestamp, pass/fail status, and standard output/error traces (especially for failures).

## Steps

1. Run the main training pipeline smoke test:
   `python train.py --config configs/phase1_baseline.yaml --smoke-test`
2. Run the MJO head architecture smoke test:
   `python scripts/smoke_test_mjo_head.py`
3. Run the parameter freezing logic smoke test:
   `python scripts/smoke_test_freeze.py`
4. Run the autoregressive rollout logic smoke test:
   `python scripts/smoke_test_rollout.py`
5. Run the NERSC dataset loader verification:
   `python scripts/verify_dataset_loader.py`
6. Run the RMM basis computation smoke test:
   `python scripts/compute_rmm.py --smoke-test`
7. Run the MJO evaluation metrics smoke test:
   `python scripts/evaluate_mjo.py --smoke-test`
8. For each of the above, append the results to `docs/smoke_results.md`. Use clear markdown sections for each test, detailing the command, the result (Pass/Fail), and the relevant output snippet or stack trace.
