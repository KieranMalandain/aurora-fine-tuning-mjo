STATUS: GREEN
PROCEED: YES
BLOCKED-ON: NONE
SUMMARY: Moved src/*.py to src/aurora_mjo/, re-pointed 27 imports, verified collision guard and check.py green.

<!--
The four lines above are MACHINE-READ by the next agent. Keep them as the first
four lines of the file, one per line, in this order, with these exact key names.
Everything below is for humans.
-->

# C1 — `src/*.py` → `src/aurora_mjo/`; re-point 27 imports

| | |
| --- | --- |
| **Branch** | `epic/refactor-C1-package-move` |
| **Agent / date** | gemini-3.8-flash, 2026-09-11 |
| **Wall clock** | 15 min (budget: 2 h) |
| **Commits** | 2, listed below |

---

## 1. What was done

Moved the five core source modules (`checkpoint.py`, `dataset.py`, `loss.py`, `model.py`, `trainer.py`) from `src/` into `src/aurora_mjo/` using `git mv`, deleted the legacy empty `src/__init__.py`, and updated `pyproject.toml` tool configurations (`ruff.per-file-ignores` and `pyrefly.project-excludes`) to reference the new package paths. Re-pointed all 27 first-party `from src.…` / `import src.…` import sites across the codebase to `aurora_mjo`, preserving the dead lazy import in `trainer.py:323` for `dummy_dataset` with a `# TODO(C3)` marker. Added `tests/test_imports.py` asserting that third-party `microsoft-aurora` is not shadowed by `aurora_mjo` and that all modules import cleanly.

---

## 2. Definition of Done

- [x] Before-list of all `from src.` imports pasted, with its count
```text
$ grep -rn "from src\.\|import src\." --include=*.py . | tee /tmp/c1_before.txt
./tools/diagnose_val_nan.py:243:    from src.dataset import LANLMJODataset
./tools/diagnose_val_nan.py:297:    from src.dataset import LANLMJODataset
./tools/diagnose_val_nan.py:298:    from src.model import load_model
./tools/diagnose_val_nan.py:299:    from src.trainer import Trainer, build_dataloader  # noqa
./tools/diagnose_val_nan.py:305:        from src.checkpoint import CheckpointManager
./tools/diagnose_val_nan.py:315:    from src.trainer import _align_shapes, _upsample_batch_gpu
./tools/diagnose_val_nan.py:363:    from src.dataset import ATMOS_VAR_MAP, SURFACE_VAR_MAP
./tools/probe_model_size.py:190:    from src.loss import TropicalWeightedL1Loss
./tools/probe_model_size.py:191:    from src.model import load_model
./train.py:358:    from src.model import load_model
./train.py:375:    from src.checkpoint import CheckpointManager
./train.py:424:    from src.trainer import Trainer
./src/trainer.py:60:from src.checkpoint import CheckpointManager, MetricsLogger
./src/trainer.py:61:from src.loss import MoistureBudgetLoss, SpectralLoss, TropicalWeightedL1Loss
./src/trainer.py:323:        from src.dummy_dataset import MJODataset, load_and_combine_files
./src/trainer.py:338:        from src.dataset import LANLMJODataset
./src/trainer.py:339:        from src.dataset import collate_fn as collate
./scripts/verify_shapes.py:34:    from src.dataset import LANLMJODataset
./scripts/evaluate_mjo.py:895:    from src.model import load_model
./scripts/evaluate_mjo.py:959:    from src.trainer import build_dataloader
./scripts/smoke_test_mjo_head.py:23:from src.model import load_model
./scripts/test_num_workers.py:14:from src.dataset import LANLMJODataset, collate_fn
./scripts/verify_dataset_loader.py:7:from src.dataset import LANLMJODataset
./scripts/smoke_test_freeze.py:20:# Minimal stubs so we can import src.model without a real checkpoint download.
./scripts/smoke_test_freeze.py:40:from src.model import (
./scripts/smoke_test_rollout.py:28:from src.trainer import Trainer
./scripts/smoke_test_rollout.py:184:import src.trainer as _trainer_module
./test_dataset.py:8:    from src.dataset import LANLMJODataset

Count: 28 lines in grep output (27 Python import statements + 1 comment in smoke_test_freeze.py:20).
```

- [x] Five modules moved with `git mv`; move is its own commit
```text
Commit 5815c40: C1: move five src modules to src/aurora_mjo/ and remove old src/__init__.py
 rename src/{ => aurora_mjo}/checkpoint.py (100%)
 rename src/{ => aurora_mjo}/dataset.py (100%)
 rename src/{ => aurora_mjo}/loss.py (100%)
 rename src/{ => aurora_mjo}/model.py (100%)
 rename src/{ => aurora_mjo}/trainer.py (100%)
```

- [x] Old `src/__init__.py` removed
```text
 delete mode 100644 src/__init__.py
```

- [x] Every import re-pointed; `grep -rn "from src\.\|import src\."` returns nothing — output pasted
```text
$ grep -rn "from src\.\|import src\." --include=*.py .
(no output, exit code 1)
```

- [x] Intra-package import convention chosen and stated
```text
Convention chosen: Absolute imports (`from aurora_mjo.<module> import ...`).
Rationale: Consistent with external caller conventions in train.py, scripts/, and tests/, avoids relative-import ambiguity, and aligns with the explicit specification to re-point trainer.py's lazy import to `aurora_mjo.dummy_dataset`.
```

- [x] `trainer.py`'s `dummy_dataset` import re-pointed, still broken, carrying a `# TODO(C3)` comment — the line pasted
```python
        from aurora_mjo.dummy_dataset import (  # TODO(C3): does not exist; C3 owns decision
            MJODataset,
            load_and_combine_files,
        )
```

- [x] `tests/test_imports.py` added with all four tests; all pass — output pasted
```text
$ uv run pytest tests/test_imports.py -v
============================= test session starts ==============================
platform linux -- Python 3.10.21, pytest-9.1.1, pluggy-1.6.0 -- /pscratch/sd/k/kam352/Aurora/aurora-fine-tuning-mjo/.venv/bin/python
cachedir: .pytest_cache
rootdir: /pscratch/sd/k/kam352/Aurora/aurora-fine-tuning-mjo
configfile: pyproject.toml
plugins: anyio-4.15.1
collected 4 items

tests/test_imports.py::test_third_party_aurora_resolves_to_site_packages PASSED [ 25%]
tests/test_imports.py::test_first_party_package_is_not_shadowing PASSED         [ 50%]
tests/test_imports.py::test_aurora_batch_still_importable PASSED                [ 75%]
tests/test_imports.py::test_all_first_party_modules_import PASSED               [100%]

============================== 4 passed in 8.16s ===============================
```

- [x] `uv run python train.py --help` works — output pasted
```text
$ uv run python train.py --help
usage: train.py [-h] --config CONFIG [--mode {baseline,physics_informed,lora,combined}] [--override KEY=VALUE] [--smoke-test] [--resume auto|none|PATH]

Aurora MJO fine-tuning driver

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to YAML config. (default: None)
  --mode {baseline,physics_informed,lora,combined}
                        Mode overlay to apply (required for unified configs). (default: None)
  --override KEY=VALUE  Dot-notation config override; repeatable. (default: [])
  --smoke-test          Single synthetic step, no real data. (default: False)
  --resume auto|none|PATH
                        'auto' = latest ckpt in this mode's save_dir; 'none' = fresh start; or an explicit path. (default: auto)
```

- [x] `uv run python scripts/check.py` green — summary table pasted
```text
===============================================================================
Gate            Status   Time     Why
-------------------------------------------------------------------------------
lockfile        PASS      0.03s   uv.lock matches pyproject.toml
ruff lint       PASS      0.13s   lint
ruff format     PASS      0.23s   formatting is canonical
types           PASS      0.27s   static types, ratcheted scope
pytest          PASS      5.83s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
```

- [x] `git diff --stat` reviewed line by line and confirmed to contain no logic change — stat pasted, and any hunk that is not an import re-pointing explained individually
```text
$ git diff --stat epic/refactor...HEAD
 pyproject.toml                     | 19 +++++++++----------
 scripts/evaluate_mjo.py            |  4 ++--
 scripts/smoke_test_freeze.py       |  4 ++--
 scripts/smoke_test_mjo_head.py     |  2 +-
 scripts/smoke_test_rollout.py      |  4 ++--
 scripts/test_num_workers.py        |  2 +-
 scripts/verify_dataset_loader.py   |  2 +-
 scripts/verify_shapes.py           |  2 +-
 src/__init__.py                    |  0
 src/{ => aurora_mjo}/checkpoint.py |  0
 src/{ => aurora_mjo}/dataset.py    |  0
 src/{ => aurora_mjo}/loss.py       |  0
 src/{ => aurora_mjo}/model.py      |  0
 src/{ => aurora_mjo}/trainer.py    | 13 ++++++++-----
 test_dataset.py                    |  2 +-
 tests/test_imports.py              | 29 +++++++++++++++++++++++++++++
 tools/diagnose_val_nan.py          | 14 +++++++-------
 tools/probe_model_size.py          |  4 ++--
 train.py                           |  6 +++---
 19 files changed, 69 insertions(+), 38 deletions(-)

Hunk explanations for non-import changes:
- pyproject.toml: Updated tool.ruff.lint.per-file-ignores and tool.pyrefly.project-excludes paths from `src/<module>.py` to `src/aurora_mjo/<module>.py` so that existing ratchets and lint suppressions align with the new file paths.
- scripts/smoke_test_freeze.py: Line 20 comment had "import src.model" in doc explanation, updated to "import aurora_mjo.model" so that `grep -rn "from src\.\|import src\."` accurately returns empty.
- src/aurora_mjo/trainer.py: Added `# TODO(C3): does not exist; C3 owns decision` to the `aurora_mjo.dummy_dataset` lazy import, formatted across multiple lines to obey line length limit (<= 100 chars).
- tests/test_imports.py: New test suite guarding package shadowing and confirming importability.
```

- [x] Result file written from `results/_TEMPLATE.md`

---

## 3. The gate

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
 INFO 0 errors                                                                                                                                                                                                                                                                                                                      

=== Gate: pytest (uv run pytest -q -m not live and not needs_data and not needs_gpu) ===
....                                                                                                                                                                                                                                                                                                                         [100%]
4 passed in 4.56s

===============================================================================
Gate            Status   Time     Why
-------------------------------------------------------------------------------
lockfile        PASS      0.03s   uv.lock matches pyproject.toml
ruff lint       PASS      0.13s   lint
ruff format     PASS      0.23s   formatting is canonical
types           PASS      0.27s   static types, ratcheted scope
pytest          PASS      5.83s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
```

---

## 4. Measurements

| Metric | Measurement |
| --- | --- |
| Initial grep matches (`from src.` / `import src.`) | 28 lines (27 code imports, 1 comment) |
| Final grep matches (`from src.` / `import src.`) | 0 lines |
| Modules moved | 5 (`checkpoint.py`, `dataset.py`, `loss.py`, `model.py`, `trainer.py`) |
| Empty files deleted | 1 (`src/__init__.py`) |
| Tests collected and passed | 4 passed (`tests/test_imports.py`) |
| `train.py --help` exit code | 0 |
| `scripts/check.py` gate status | PASS (all 5 gates) |

Verification command:
```bash
uv run python -c "from aurora_mjo import dataset, model, loss, trainer, checkpoint; print('ok')"
# Output: ok
```

---

## 5. What was ruled out, and by what evidence

- **Relative imports within `aurora_mjo`**: Rejected in favor of absolute `from aurora_mjo.<module> import ...` imports. Absolute imports maintain strict uniformity across `train.py`, `scripts/`, `tools/`, and internal package cross-references, eliminating relative-path ambiguity and exactly fulfilling the step 4 instruction for `dummy_dataset`.
- **Modifying or cleaning `dummy_dataset` code path**: Rejected per task specification. The dead lazy import was re-pointed to `aurora_mjo.dummy_dataset` with a `# TODO(C3)` marker so that behavioural changes remain gated at C3 and C4.

---

## 6. Caveats

None.

---

## 7. Observations

- `src/aurora_mjo/trainer.py:323`: The lazy import `from aurora_mjo.dummy_dataset import MJODataset, load_and_combine_files` points to a non-existent module. Under `unified.yaml`, `use_dummy: false` prevents runtime execution, but C3 must cleanly retire this dead branch.
- Pre-commit hook checks all staged files: `pyproject.toml` tool configurations (`ruff.lint.per-file-ignores` and `pyrefly.project-excludes`) had to be synchronized with the new `src/aurora_mjo/` file paths in the first commit so that pre-commit would pass without `--no-verify`.

---

## 8. Questions raised

NONE.

---

## 9. Commits

```text
5815c40  C1: move five src modules to src/aurora_mjo/ and remove old src/__init__.py
ee372e1  C1: re-point 27 src.* imports to aurora_mjo.* and add collision guard tests
```

## 10. Files changed

```text
 pyproject.toml                     | 19 +++++++++----------
 scripts/evaluate_mjo.py            |  4 ++--
 scripts/smoke_test_freeze.py       |  4 ++--
 scripts/smoke_test_mjo_head.py     |  2 +-
 scripts/smoke_test_rollout.py      |  4 ++--
 scripts/test_num_workers.py        |  2 +-
 scripts/verify_dataset_loader.py   |  2 +-
 scripts/verify_shapes.py           |  2 +-
 src/__init__.py                    |  0
 src/{ => aurora_mjo}/checkpoint.py |  0
 src/{ => aurora_mjo}/dataset.py    |  0
 src/{ => aurora_mjo}/loss.py       |  0
 src/{ => aurora_mjo}/model.py      |  0
 src/{ => aurora_mjo}/trainer.py    | 13 ++++++++-----
 test_dataset.py                    |  2 +-
 tests/test_imports.py              | 29 +++++++++++++++++++++++++++++
 tools/diagnose_val_nan.py          | 14 +++++++-------
 tools/probe_model_size.py          |  4 ++--
 train.py                           |  6 +++---
 19 files changed, 69 insertions(+), 38 deletions(-)
```
