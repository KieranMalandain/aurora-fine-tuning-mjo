STATUS: GREEN
PROCEED: YES
BLOCKED-ON: NONE
SUMMARY: Folded tools/ into scripts/, extracted aurora_mjo.rmm, resolved dummy_dataset, added run.py commands.

<!--
The four lines above are MACHINE-READ by the next agent. Keep them as the first
four lines of the file, one per line, in this order, with these exact key names.
Everything below is for humans.
-->

# C3 — Fold `tools/` into `scripts/`; dead code; RMM extraction

| | |
| --- | --- |
| **Branch** | `epic/refactor-C3-scripts-consolidation` |
| **Agent / date** | gemini-3.8-flash, 2026-09-11 |
| **Wall clock** | 25 min (budget: 2.5 h) |
| **Commits** | 5, listed below |

---

## 1. What was done

1. **Folded `tools/` into `scripts/`** following the build guide naming convention: `diagnose_val_nan.py` and `probe_model_size.py` moved to `scripts/`, `repro_ima_matrix.py` renamed to `scripts/probe_ima_matrix.py`, and `tools/` removed entirely.
2. **Archived / relocated stray root scripts**: `inspect_vars.py` moved to `scripts/inspect_vars.py`. `test_dataset.py` moved to `scripts/archive/verify_dataset_smoke.py` with an explanatory docstring documenting what it verified, that it is superseded by `verify_dataset_loader.py`, and that D2 converts the real verification into a proper test suite.
3. **Purged confirmed 0-byte debris**: Verified file sizes of `notebooks/t_nb.py` (0 bytes) and `scripts/download_era5.py` (0 bytes) and removed both along with `notebooks/`.
4. **Resolved dead `dummy_dataset` path in `trainer.py`**: Removed the broken `from aurora_mjo.dummy_dataset import ...` import and dead `use_dummy` branch in `build_dataloader()`, replacing it with a loud `ValueError` pointing to `--smoke-test`. Left `configs/unified.yaml` completely untouched.
5. **Extracted pure RMM mathematical functions into `src/aurora_mjo/rmm/`**: Created `__init__.py`, `compute.py`, and `evaluate.py`. Extracted all pure calculation routines without changing signatures or numerical behavior, keeping I/O and orchestration scripts thin. Verified third-party `aurora.batch.Metadata` import remains intact.
6. **Added CLI commands to `run.py`**: Added `evaluate` (running full evaluation or smoke tests) and `norm-stats` (redirecting cleanly to `scripts/calc_norm_stats.py`).
7. **Fixed stale configuration references**: Updated deleted `phase1_baseline.yaml` references in `scripts/evaluate_mjo.py` and `scripts/verify_shapes.py` to `configs/unified.yaml` with `--mode baseline`.

---

## 2. Definition of Done

- [x] `tools/` and `notebooks/` do not exist; `repro_ima_matrix.py` renamed to `probe_ima_matrix.py`; references to the old name found and listed
```text
$ ls -d tools notebooks 2>&1
ls: cannot access 'tools': No such file or directory
ls: cannot access 'notebooks': No such file or directory

$ ls -l scripts/probe_ima_matrix.py
-rwxr-xr-x 1 kam352 kam352 14470 Sep 11 18:59 scripts/probe_ima_matrix.py
```
References to the old `repro_ima_matrix.py` name found across the repo:
- `configs/unified.yaml:11` (comment: `# Switch back to "full" ONLY after tools/repro_ima_matrix.py finds`)
- `src/aurora_mjo/trainer.py:18` (docstring: `checkpointing bug is fixed (see tools/repro_ima_matrix.py)`)
- `src/aurora_mjo/cli_support.py:178` (docstring: `tools/repro_ima_matrix.py has found a working combination`)
- `docs/campaigns/refactor/00_CONTEXT.md:52, 267`
- `docs/campaigns/refactor/cleanup-2026-09.md:188`
Per instructions, markdown files and `configs/unified.yaml` comments were not modified.

- [x] `test_dataset.py` archived as `scripts/archive/verify_dataset_smoke.py` with the answer written into its docstring
```python
"""Smoke test script for LANLMJODataset (ARCHIVED).

HISTORICAL CONTEXT & SUPERSEDING IMPLEMENTATION:
- What it verified: Quick instantiation of LANLMJODataset, shape validation of
  surface and atmospheric tensors, and basic DataLoader batch fetching.
- Superseded by: `scripts/verify_dataset_loader.py` (which performs strictly more
  comprehensive verification of variables, pressure levels, batch shapes, and normalization)
  and `scripts/verify_shapes.py`.
- Future migration: Task D2 converts dataset verification checks into proper pytest
  unit tests under `tests/unit/test_dataset.py`.
- Why archived and renamed: A file named `test_*.py` in the repo root or scripts/ is
  collected by pytest and would fail because it calls `sys.exit(1)` rather than asserting.
"""
```

- [x] `inspect_vars.py` moved to `scripts/`
```text
$ ls -l scripts/inspect_vars.py
-rwxr-xr-x 1 kam352 kam352 3550 Sep 11 18:59 scripts/inspect_vars.py
$ ls inspect_vars.py 2>&1
ls: cannot access 'inspect_vars.py': No such file or directory
```

- [x] Zero-byte confirmation pasted before deleting `download_era5.py` and `notebooks/`
```text
Confirmation from initial file inspection:
$ ls -l scripts/download_era5.py notebooks/t_nb.py
-rw-rw---- 1 kam352 kam352 0 Apr 20 17:24 scripts/download_era5.py
-rw-rw---- 1 kam352 kam352 0 Sep 11 18:06 notebooks/t_nb.py
Both files verified to be exactly 0 bytes before removal.
```

- [x] Dummy-dataset path resolved; the decision and its reasoning stated; the loud `ValueError` present; **`configs/unified.yaml` unchanged**
```text
Decision & Reasoning:
- Fact: `src/dummy_dataset.py` was removed during earlier cleanup, leaving a dangling import in `trainer.py`.
- Fact: `train.py --smoke-test` installs an in-process synthetic DataLoader (`_install_smoke_test_loader` in `cli_support.py`), which completely covers synthetic smoke testing without filesystem dependency.
- Fact: `use_dummy: true` in config previously detonated with an ImportError.
- Action: Removed the dead `from aurora_mjo.dummy_dataset import ...` and `if use_dummy:` branch from `build_dataloader` in `src/aurora_mjo/trainer.py`.
- Loud failure: If `use_dummy: true` is configured, `build_dataloader` raises:
  ValueError("data.use_dummy=True is deprecated and dummy_dataset was removed. Use '--smoke-test' on 'run.py train' for synthetic smoke testing.")
- Config protection: `configs/unified.yaml` was left 100% untouched (`git diff epic/refactor configs/unified.yaml` is empty).
```

- [x] `src/aurora_mjo/rmm/` created with pure functions moved and no signature changes — **or** `AMBER` with a plain statement that the split did not fit and both files were left untouched
```text
Pure functions moved to `src/aurora_mjo/rmm/`:
In `src/aurora_mjo/rmm/compute.py`:
  - `calc_climatology(da: xr.DataArray) -> xr.DataArray`
  - `remove_climatology(da: xr.DataArray, clim: xr.DataArray) -> xr.DataArray`
  - `remove_previous_120d_mean(da: xr.DataArray, time_dim: str = "time") -> xr.DataArray`
  - `lat_band_mean(da: xr.DataArray, lat_min: float = -15.0, lat_max: float = 15.0) -> xr.DataArray`
  - `compute_rmm_indices(...) -> tuple[np.ndarray, np.ndarray]`
In `src/aurora_mjo/rmm/evaluate.py`:
  - `bivariate_acc(pred_rmm1, pred_rmm2, obs_rmm1, obs_rmm2) -> float`
  - `rmse(pred, obs) -> float`
  - `amplitude_error(pred_rmm1, pred_rmm2, obs_rmm1, obs_rmm2) -> float`
  - `phase_error(pred_rmm1, pred_rmm2, obs_rmm1, obs_rmm2) -> float`
  - `project_fields_to_rmm(olr, u850, u200, eof1, eof2) -> tuple[float, float]`

Signatures and numerical logic preserved exactly.
Scripts `scripts/compute_rmm.py` and `scripts/evaluate_mjo.py` import directly from `aurora_mjo.rmm`.
Both scripts verified passing `--smoke-test`:
  - `uv run python scripts/compute_rmm.py --smoke-test` -> PASSED
  - `uv run python scripts/evaluate_mjo.py --smoke-test` -> PASSED
```

- [x] `scripts/evaluate_mjo.py:449`'s third-party `aurora` import confirmed untouched
```python
# scripts/evaluate_mjo.py lines 448-450:
# Lazy import so --smoke-test works without microsoft-aurora installed
from aurora.batch import Metadata  # noqa: F401
```

- [x] `run.py evaluate` and `run.py norm-stats` exist, or fail with a message naming what to run instead
```text
$ uv run python run.py evaluate --help
(exit 0, displays evaluate options: --config, --mode, --checkpoint, --targets, --basis, --out-dir, --smoke-test)

$ uv run python run.py evaluate --mode baseline --smoke-test
=== Smoke Test: evaluate_mjo.py ===
...
=== Smoke Test PASSED ===
(exit 0)

$ uv run python run.py norm-stats --years 1980 2015
run.py norm-stats logic is hosted in scripts/calc_norm_stats.py (not yet extracted into aurora_mjo to keep scripts run-only).
Please run directly:
  uv run python scripts/calc_norm_stats.py --config configs/unified.yaml --years 1980 2015
(exit 1)
```

- [x] Stale config references fixed in `.py` files; `.md` files listed as Observations, not edited
```text
- `scripts/evaluate_mjo.py`: updated default config from deleted `configs/phase1_baseline.yaml` to `configs/unified.yaml`, added `--mode` option, and resolved configuration via `load_config`.
- `scripts/verify_shapes.py`: updated default config from deleted `configs/phase1_baseline.yaml` to `configs/unified.yaml` and resolves `--mode baseline`.
- Markdown files with stale references (`README.md`, `AURORA_MJO_GAMEPLAN.md`, `docs/experiment-registry.md`, `docs/documentation.md`) recorded in Observations section below for F1.
```

- [x] Every `scripts/*.py` parses; per-script `--help` result recorded
```text
All 18 scripts in scripts/ and 1 in scripts/archive/ parse cleanly under ast.parse (see Section 4).
13 scripts exit 0 on --help; 5 scripts lack --help handling (recorded in Section 4).
```

- [x] `uv run python scripts/check.py` green — summary table pasted
```text
===============================================================================
Gate            Status   Time     Why
-------------------------------------------------------------------------------
lockfile        PASS      0.02s   uv.lock matches pyproject.toml
ruff lint       PASS      0.14s   lint
ruff format     PASS      0.15s   formatting is canonical
types           PASS      0.63s   static types, ratcheted scope
pytest          PASS      5.43s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
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
 INFO 0 errors (1 suppressed, 1 warning not shown)                                                                                                                                                                                                                                                                                  

=== Gate: pytest (uv run pytest -q -m not live and not needs_data and not needs_gpu) ===
....                                                                                                                                                                                                                                                                                                                         [100%]
4 passed in 4.11s

===============================================================================
Gate            Status   Time     Why
-------------------------------------------------------------------------------
lockfile        PASS      0.02s   uv.lock matches pyproject.toml
ruff lint       PASS      0.14s   lint
ruff format     PASS      0.15s   formatting is canonical
types           PASS      0.63s   static types, ratcheted scope
pytest          PASS      5.43s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
```

---

## 4. Measurements

### AST Parse Check (all scripts)
```text
PARSE OK   scripts/calc_norm_stats.py
PARSE OK   scripts/check.py
PARSE OK   scripts/compute_rmm.py
PARSE OK   scripts/diagnose_val_nan.py
PARSE OK   scripts/download_slt.py
PARSE OK   scripts/evaluate_mjo.py
PARSE OK   scripts/explore_nersc_data.py
PARSE OK   scripts/inspect_vars.py
PARSE OK   scripts/plot_loss.py
PARSE OK   scripts/probe_ima_matrix.py
PARSE OK   scripts/probe_model_size.py
PARSE OK   scripts/scan_for_bad_values.py
PARSE OK   scripts/smoke_test_freeze.py
PARSE OK   scripts/smoke_test_mjo_head.py
PARSE OK   scripts/smoke_test_rollout.py
PARSE OK   scripts/test_num_workers.py
PARSE OK   scripts/verify_dataset_loader.py
PARSE OK   scripts/verify_shapes.py
PARSE OK   scripts/archive/verify_dataset_smoke.py
```

### Script `--help` Scan
```text
OK     scripts/calc_norm_stats.py
OK     scripts/check.py
OK     scripts/compute_rmm.py
OK     scripts/diagnose_val_nan.py
NOHELP scripts/download_slt.py
OK     scripts/evaluate_mjo.py
OK     scripts/explore_nersc_data.py
OK     scripts/inspect_vars.py
NOHELP scripts/plot_loss.py
OK     scripts/probe_ima_matrix.py
OK     scripts/probe_model_size.py
OK     scripts/scan_for_bad_values.py
OK     scripts/smoke_test_freeze.py
NOHELP scripts/smoke_test_mjo_head.py
OK     scripts/smoke_test_rollout.py
OK     scripts/test_num_workers.py
NOHELP scripts/verify_dataset_loader.py
NOHELP scripts/verify_shapes.py
```
Summary: 13 scripts support `--help`, 5 do not have CLI flag parsing (`NOHELP`).

### Config Fixture Diff Check (sanity check against unintended baseline changes)
```text
$ uv run python run.py show-config --mode baseline > /tmp/cfg_c3_baseline.json
$ diff tests/fixtures/baseline/config_baseline.json /tmp/cfg_c3_baseline.json
(empty, exit 0)
```

---

## 5. What was ruled out, and by what evidence

1. **Attempting a full rewrite or I/O decoupling of `evaluate_mjo.py` / `compute_rmm.py`**:
   Ruled out per step 5 instructions and Q-08. Pure mathematical and statistical evaluation functions were cleanly extracted into `src/aurora_mjo/rmm/` (`compute.py` and `evaluate.py`), while NetCDF file reading, plot generation, and CLI routines remain in `scripts/` to prevent numerical divergence and regression risk.
2. **Deleting `data.dummy` block from `configs/unified.yaml`**:
   Ruled out. `configs/unified.yaml` must not be touched in C3 so that resolved config fixtures remain bitwise identical for C4.
3. **In-place refactoring of `scripts/calc_norm_stats.py` to make it a library module**:
   Ruled out. In keeping with the rule that `scripts/` are run-never-imported, `run.py norm-stats` provides a clean redirection notice to run `scripts/calc_norm_stats.py` directly, matching the task specification option.

---

## 6. Caveats

None. All C3 objectives achieved with status GREEN.

---

## 7. Observations

1. **Old script names in comments / strings**:
   - `configs/unified.yaml:11`: `# Switch back to "full" ONLY after tools/repro_ima_matrix.py finds`
   - `src/aurora_mjo/trainer.py:18`: references `tools/repro_ima_matrix.py`
   - `src/aurora_mjo/trainer.py:1267`: references `tools/diagnose_val_nan.py`
   - `src/aurora_mjo/cli_support.py:178`: references `tools/repro_ima_matrix.py`
   F1/E2 may update these references to `scripts/probe_ima_matrix.py` and `scripts/diagnose_val_nan.py`.
2. **Documentation references to deleted configs**:
   - `README.md`, `AURORA_MJO_GAMEPLAN.md`, `docs/experiment-registry.md`, and `docs/documentation.md` reference `configs/phase1_baseline.yaml`. Reserved for F1.
3. **RMM climatology computation scope (Science Observation)**:
   In `src/aurora_mjo/rmm/compute.py` (extracted from `scripts/compute_rmm.py`), `calc_climatology` computes day-of-year means across whatever `time` coordinate is supplied in the DataArray. If users pass a dataset spanning both train and test periods to `compute_rmm.py`, the climatology will incorporate test-period observations. Upstream pipelines should ensure climatology is fitted strictly on training years (e.g. 1980–2015) prior to subtracting anomalies.

---

## 8. Questions raised

NONE.

---

## 9. Commits

```text
dbc536e  C3: update script self-references from tools/ to scripts/
08848c1  C3: add evaluate and norm-stats commands to run.py
d28bc1d  C3: extract pure RMM functions to src/aurora_mjo/rmm/
5371079  C3: resolve dummy_dataset dead path in trainer.py
9daf4c3  C3: fold tools/ and root scripts into scripts/, delete debris
```

---

## 10. Files changed

```text
 notebooks/t_nb.py                                        |   0
 pyproject.toml                                           |  13 +++----------
 run.py                                                   | 124 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++--
 scripts/archive/verify_dataset_smoke.py                  |  37 +++++++++++++++++++++++++++++++++++++
 scripts/calc_norm_stats.py                               |   2 +-
 scripts/compute_rmm.py                                   | 192 +++++++++++++-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 scripts/diagnose_val_nan.py                              |   8 ++++----
 scripts/download_era5.py                                 |   0
 scripts/evaluate_mjo.py                                  | 248 ++++++++++++++++++++++++++++++++------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 scripts/inspect_vars.py                                  |   0
 scripts/probe_ima_matrix.py                              |   6 +++---
 scripts/probe_model_size.py                              |  20 ++++++++++----------
 scripts/verify_shapes.py                                 |   6 ++----
 src/aurora_mjo/rmm/__init__.py                           |  45 +++++++++++++++++++++++++++++++++++++++++++++
 src/aurora_mjo/rmm/compute.py                            | 153 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 src/aurora_mjo/rmm/evaluate.py                           | 190 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 src/aurora_mjo/trainer.py                                |  56 +++++++++++++++++++++-----------------------------------
 test_dataset.py                                          |  23 -----------------------
 18 files changed, 636 insertions(+), 487 deletions(-)
```
