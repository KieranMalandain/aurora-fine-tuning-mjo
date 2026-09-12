STATUS: GREEN
PROCEED: YES
BLOCKED-ON: NONE
SUMMARY: Re-asserted B1 behavioral fingerprint across config, model params, dataset, and smoke test with 0 delta.

<!--
The four lines above are MACHINE-READ by the next agent. Keep them as the first
four lines of the file, one per line, in this order, with these exact key names.
Everything below is for humans.
-->

# C4 — Re-assert the B1 fingerprint

| | |
| --- | --- |
| **Branch** | `epic/refactor-C4-fingerprint-reassert` |
| **Agent / date** | gemini-3.8-flash, 2026-09-11 |
| **Wall clock** | 20 min (budget: 90 min) |
| **Commits** | 2, listed below |

---

## 1. What was done

Re-asserted every element of the B1 behavioural fingerprint against the post-refactor codebase (after C1 package move, C2 thin CLI, and C3 scripts consolidation). Generated canonical post-refactor fixtures under `tests/fixtures/postrefactor/` covering config resolutions for all four modes plus override, CPU model parameter audits, 1980 dataset index and static variable properties, and single-step GPU smoke-test loss trajectories. Confirmed exact equivalence (Δ = 0 across all integers and bitwise identical config JSONs) and verified zero loss deviation against the 1e-4 tolerance.

---

## 2. Definition of Done

- [x] Provenance for this run **and** B1's, with every difference stated
```text
B1 Provenance:
  Base commit SHA: 3a6ee1fc5acdb34a29484a5cf3af5479f17b083d
  Hostname:        nid008469 (NERSC Perlmutter compute node)
  Python:          Python 3.10.20 (/pscratch/sd/k/kam352/conda_envs/aurora_mjo/bin/python)
  Torch:           2.5.1+cu121 (CUDA available: True, device count: 4)
  GPU Topology:    4 x NVIDIA A100-SXM4-80GB

C4 Provenance:
  Base commit SHA: be10d03130c56955c69dcfc985f4213e13878a0f
  Hostname:        nid008272 (NERSC Perlmutter compute node)
  Python:          Python 3.10.21 (/pscratch/sd/k/kam352/Aurora/aurora-fine-tuning-mjo/.venv/bin/python3)
  Torch:           2.5.1+cu121 (CUDA available: True, device count: 4)
  GPU Topology:    4 x NVIDIA A100-SXM4-80GB

Differences stated:
  1. Hostname differs across Perlmutter compute nodes (nid008272 vs nid008469); hardware is identical (4x A100 80GB).
  2. Python interpreter is managed under uv virtual environment (3.10.21) rather than conda (3.10.20).
  3. Base commit includes refactor tasks C1, C2, and C3.
```

- [x] Five config diffs run and pasted, all empty
```bash
$ uv run python run.py show-config --mode baseline > tests/fixtures/postrefactor/config_baseline.json
$ diff -u tests/fixtures/baseline/config_baseline.json tests/fixtures/postrefactor/config_baseline.json
(empty, exit 0)

$ uv run python run.py show-config --mode physics_informed > tests/fixtures/postrefactor/config_physics_informed.json
$ diff -u tests/fixtures/baseline/config_physics_informed.json tests/fixtures/postrefactor/config_physics_informed.json
(empty, exit 0)

$ uv run python run.py show-config --mode lora > tests/fixtures/postrefactor/config_lora.json
$ diff -u tests/fixtures/baseline/config_lora.json tests/fixtures/postrefactor/config_lora.json
(empty, exit 0)

$ uv run python run.py show-config --mode combined > tests/fixtures/postrefactor/config_combined.json
$ diff -u tests/fixtures/baseline/config_combined.json tests/fixtures/postrefactor/config_combined.json
(empty, exit 0)

$ uv run python run.py show-config --mode baseline --override training.optimizer.lr=1e-5 > tests/fixtures/postrefactor/config_baseline_override.json
$ diff -u tests/fixtures/baseline/config_baseline_override.json tests/fixtures/postrefactor/config_baseline_override.json
(empty, exit 0)
```

- [x] Parameter-count table for four modes with a Δ column; all Δ = 0
```text
Executed on: CPU (matching B1)
```
| Mode | Metric | B1 | Now | Δ |
| --- | --- | --- | --- | --- |
| `baseline` | Total params | 112,830,384 | 112,830,384 | 0 |
| `baseline` | Trainable params | 41,008 | 41,008 | 0 |
| `baseline` | Frozen params | 112,789,376 | 112,789,376 | 0 |
| `baseline` | MJO Head constructed | False | False | 0 |
| `baseline` | LoRA constructed | False | False | 0 |
| `physics_informed` | Total params | 112,830,384 | 112,830,384 | 0 |
| `physics_informed` | Trainable params | 41,008 | 41,008 | 0 |
| `physics_informed` | Frozen params | 112,789,376 | 112,789,376 | 0 |
| `physics_informed` | MJO Head constructed | False | False | 0 |
| `physics_informed` | LoRA constructed | False | False | 0 |
| `lora` | Total params | 113,371,056 | 113,371,056 | 0 |
| `lora` | Trainable params | 581,680 | 581,680 | 0 |
| `lora` | Frozen params | 112,789,376 | 112,789,376 | 0 |
| `lora` | MJO Head constructed | False | False | 0 |
| `lora` | LoRA constructed | True | True | 0 |
| `combined` | Total params | 113,371,056 | 113,371,056 | 0 |
| `combined` | Trainable params | 581,680 | 581,680 | 0 |
| `combined` | Frozen params | 112,789,376 | 112,789,376 | 0 |
| `combined` | MJO Head constructed | False | False | 0 |
| `combined` | LoRA constructed | True | True | 0 |

- [x] `len(dataset)` == 1462, stated
```text
Measured len(dataset): 1462
Expected prior (03_DOMAIN_PRIORS.md §1): 1462 (1464 timesteps - 2)
Delta: 0 (Exact match)
```

- [x] Static-var mean/std match B1 to recorded precision — table pasted
| Field | B1 Mean | Now Mean | B1 Std | Now Std | Finite Fraction | Match |
| --- | --- | --- | --- | --- | --- | --- |
| `z` | 3709.2466 | 3709.2466 | 8216.3809 | 8216.3809 | 1.0 (100%) | EXACT |
| `lsm` | 0.3357 | 0.3357 | 0.4484 | 0.4484 | 1.0 (100%) | EXACT |
| `slt` | 0.6708 | 0.6708 | 1.1682 | 1.1682 | 1.0 (100%) | EXACT |

- [x] Alignment report diffed; every differing line explained individually
```bash
$ diff -u tests/fixtures/baseline/dataset_1980_alignment.txt tests/fixtures/postrefactor/dataset_1980_alignment.txt
(empty, exit 0)
```
There are zero differing lines (0 diffs).

- [x] `Using zeros` presence matches B1, stated either way
```text
Captured warnings during LANLMJODataset init:
  Total warnings captured: 1
  Warning message: RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility. Expected 16 from C header, got 96 from PyObject
  'Using zeros' warnings found: 0 (NONE)
Matches B1 exactly: zero 'Using zeros' warnings emitted.
```

- [x] Smoke test run through **both** `run.py` and the `train.py` shim
```text
run.py execution:
  $ HF_HOME=/pscratch/sd/k/kam352/hf_home uv run python run.py train --mode baseline --smoke-test --resume none 2>&1 | tee tests/fixtures/postrefactor/smoke_baseline_runpy.log
  TRAIN loss: 7472.0244140625
  VAL loss:   7421.9248046875

train.py shim execution:
  $ HF_HOME=/pscratch/sd/k/kam352/hf_home uv run python train.py --config configs/unified.yaml --mode baseline --smoke-test --resume none 2>&1 | tee tests/fixtures/postrefactor/smoke_baseline_shim.log
  DEPRECATED: train.py is deprecated; use 'run.py train' instead.
  TRAIN loss: 7472.0244140625
  VAL loss:   7421.9248046875
```

- [x] Max absolute and max relative loss deviation reported as numbers, against the tolerance B1 stated
```text
B1 baseline:
  train_loss: 7472.0244140625
  val_loss:   7421.9248046875
Now:
  train_loss: 7472.0244140625
  val_loss:   7421.9248046875

Max absolute deviation: 0.0
Max relative deviation: 0.0
Stated tolerance from B1: relative 1e-4 (0.0001)
Verdict: Well within tolerance (bitwise identical).
```

- [x] Non-finite-grad skip count compared **exactly** to B1
```text
B1 non-finite-grad skips: 0 (0 / 1 steps, ratio 0.0)
Now non-finite-grad skips: 0 (0 / 1 steps, ratio 0.0)
Comparison: EXACT MATCH (Delta: 0)
```

- [x] `torchrun` invocation of the shim verified
```bash
$ HF_HOME=/pscratch/sd/k/kam352/hf_home uv run torchrun --nproc_per_node=1 train.py --config configs/unified.yaml --mode baseline --smoke-test --resume none
DEPRECATED: train.py is deprecated; use 'run.py train' instead.
...
Epoch 001 | batch 00001/1 | step 1 | rollout_k=1 | loss=7472.0244 (grid=7472.0244, spec=0.0000, mjo=0.0000, phys=0.0000) | lr=1.00e-06 | t=0.00h | nan_skipped=0 | grad_skipped=0
Epoch 001 | TRAIN loss=7472.0244
Epoch 001 | VAL loss=7421.9248 (1 ok, 0 skipped)
Done (exit code 0).
```

- [x] `uv run python scripts/check.py` green — summary table pasted
```text
=== Gate: lockfile (uv lock --check) ===
Resolved 100 packages in 1ms

=== Gate: ruff lint (uv run ruff check .) ===
All checks passed!

=== Gate: ruff format (uv run ruff format --check .) ===
2 files already formatted

=== Gate: types (uv run pyrefly check) ===
 WARN PYTHONPATH environment variable is set to `/opt/nersc/pymon`. Checks in other environments may not include these paths.
 INFO Checking project configured at `/pscratch/sd/k/kam352/Aurora/aurora-fine-tuning-mjo/pyproject.toml`
 INFO 0 errors (1 suppressed, 1 warning not shown)                                                                                                                                                                                                                                                                                  

=== Gate: pytest (uv run pytest -q -m not live and not needs_data and not needs_gpu) ===
....                                                                                                                                                                                                                                                                                                                         [100%]
4 passed in 4.09s

===============================================================================
Gate            Status   Time     Why
-------------------------------------------------------------------------------
lockfile        PASS      0.02s   uv.lock matches pyproject.toml
ruff lint       PASS      0.10s   lint
ruff format     PASS      0.10s   formatting is canonical
types           PASS      0.57s   static types, ratcheted scope
pytest          PASS      5.32s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
```

- [x] Verdict is exactly one of GREEN / AMBER / RED with the required evidence
```text
VERDICT: GREEN
Evidence:
- All 5 config diffs are empty (exact match).
- Parameter counts for all 4 modes have Delta = 0 (exact match).
- Dataset length is 1462 (exact match).
- Static variable means and stds match to the fourth decimal place (exact match).
- Alignment report diff is empty (exact match).
- Zero 'Using zeros' warnings emitted (exact match).
- Single-step smoke test losses have 0.0 relative deviation against 1e-4 tolerance (bitwise identical).
- Non-finite grad skips is 0 on both runs (exact match).
- Both run.py and train.py shim pass, including under torchrun.
- Check script is 100% green.
```

- [x] **`tests/fixtures/baseline/` is byte-identical to before** — prove it with `git status --porcelain tests/fixtures/baseline/` returning nothing
```bash
$ git status --porcelain tests/fixtures/baseline/
(empty)
```

- [x] Result file written from `results/_TEMPLATE.md`

---

## 3. The gate

```text
===============================================================================
Gate            Status   Time     Why
-------------------------------------------------------------------------------
lockfile        PASS      0.02s   uv.lock matches pyproject.toml
ruff lint       PASS      0.10s   lint
ruff format     PASS      0.10s   formatting is canonical
types           PASS      0.57s   static types, ratcheted scope
pytest          PASS      5.32s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
```

---

## 4. Measurements

### 4.1 Config Fingerprints
All 5 configuration overlay resolutions produced byte-identical JSON outputs to the B1 baseline fixtures:
```bash
$ diff -r -x "*.log" tests/fixtures/baseline/ tests/fixtures/postrefactor/
(empty, exit 0)
```

### 4.2 Parameter Counts
All parameter numbers across all modes matched the B1 baseline exactly:
- `baseline`: 112,830,384 total, 41,008 trainable (0.04%), 112,789,376 frozen
- `physics_informed`: 112,830,384 total, 41,008 trainable (0.04%), 112,789,376 frozen
- `lora`: 113,371,056 total, 581,680 trainable (0.51%), 112,789,376 frozen
- `combined`: 113,371,056 total, 581,680 trainable (0.51%), 112,789,376 frozen

### 4.3 1980 Dataset Fingerprint
- Sample count: `1,462`
- Pressure levels probed: `[0, 2, 4, 6, 8, 9, 11, 13, 15, 17, 22, 25, 28]`
- Invariants:
  - `z`: mean `3709.24658203125`, std `8216.380859375`, finite fraction `1.0`
  - `lsm`: mean `0.3356685936450958`, std `0.4484005272388458`, finite fraction `1.0`
  - `slt`: mean `0.6707561612129211`, std `1.168187141418457`, finite fraction `1.0`

### 4.4 Smoke Test Metrics
- Train loss: `7472.0244140625` (B1: `7472.0244140625`, delta = `0.0`)
- Validation loss: `7421.9248046875` (B1: `7421.9248046875`, delta = `0.0`)
- Non-finite gradient skips: `0` (B1: `0`)
- Max relative deviation: `0.0` (tolerance: `1e-4`)

---

## 5. What was ruled out, and by what evidence

- **Behavioral drift from Phase C refactor**: Ruled out. The package move (`aurora_mjo`), the CLI replacement (`run.py`), and the consolidation of tools into `scripts/` introduced zero changes to configuration resolution, parameter initialization, dataset alignment, or training loss trajectory.
- **`train.py` shim incompatibility under `torchrun`**: Ruled out. Executing `torchrun --nproc_per_node=1 train.py` forwarded all command-line arguments to `run.py train` and completed the synthetic epoch cleanly.

---

## 6. Caveats

NONE (Status: GREEN).

---

## 7. Observations

1. **`HF_HOME` environment configuration on Perlmutter**: When running Hugging Face checkpoint downloads (e.g. during `load_model`), `huggingface_hub` defaults to `$HOME/.cache/huggingface/hub` where file locking (`fcntl.flock`) triggers `OSError: [Errno 524] Unknown error 524` on Perlmutter's GPFS/Lustre home directories. Setting `HF_HOME=/pscratch/sd/k/kam352/hf_home` (as done in `slurm_scripts/train_auto.slurm`) completely avoids this locking issue.
2. **Deterministic execution of single-step smoke tests**: Passing `--resume none` is necessary when re-running smoke tests if `checkpoints/baseline` already contains an `epoch_001.pt` checkpoint, preventing auto-resume from skipping epoch 1.

---

## 8. Questions raised

NONE.

---

## 9. Commits

```text
e197d88  C4: generate postrefactor behavioral fixtures matching B1 baseline
6aa689a  C4: record task results in docs/campaigns/refactor/results/C4_result.md
```

## 10. Files changed

```text
 docs/campaigns/refactor/results/C4_result.md              | 334 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/fixtures/postrefactor/config_baseline.json          | 135 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/fixtures/postrefactor/config_baseline_override.json | 135 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/fixtures/postrefactor/config_combined.json          | 141 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/fixtures/postrefactor/config_lora.json              | 136 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/fixtures/postrefactor/config_physics_informed.json  | 141 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/fixtures/postrefactor/dataset_1980.json             | 374 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/fixtures/postrefactor/dataset_1980_alignment.txt    |  13 +++++++++
 tests/fixtures/postrefactor/model_parameters.json         |  46 ++++++++++++++++++++++++++++++++
 tests/fixtures/postrefactor/smoke_baseline.json           |  32 ++++++++++++++++++++++
 10 files changed, 1487 insertions(+)
```
