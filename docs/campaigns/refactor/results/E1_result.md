STATUS: GREEN
PROCEED: YES
BLOCKED-ON: NONE
SUMMARY: HDF5_USE_FILE_LOCKING=FALSE configured at process start; static loading inverts zero fallback to StaticVarLoadError.

<!--
The four lines above are MACHINE-READ by the next agent. Keep them as the first
four lines of the file, one per line, in this order, with these exact key names.
Everything below is for humans.
-->

# E1 — `HDF5_USE_FILE_LOCKING` at process start; statics fail loudly

| | |
| --- | --- |
| **Branch** | `epic/refactor-E1-env-guards-hard-fail` |
| **Agent / date** | Gemini 3.8 Flash, 2026-09-12 |
| **Wall clock** | 30 min (budget: 2 h) |
| **Commits** | 3, listed below |

---

## 1. What was done

1. Created `src/aurora_mjo/env.py` defining `configure_environment()` and `StaticVarLoadError(RuntimeError)`. The module docstring records the full incident history: the `OSError: [Errno -101] NetCDF: HDF error` failure on readable CFS files due to Lustre/GPFS advisory file locking, the 8 September 2026 verification confirming `HDF5_USE_FILE_LOCKING=FALSE` resolves it, and why guards living only in shell scripts are inadequate.
2. Wired `configure_environment()` to execute at process startup in `src/aurora_mjo/__init__.py` (before any submodules or C-libraries initialise) and explicitly at the top of `run.py` (with `# isort: skip` and `# noqa: E402` to protect ordering against automated import linter passes).
3. Inverted the silent zero-tensor fallbacks in `src/aurora_mjo/dataset.py::_load_static_vars`. Load failures for `z`, `lsm`, and `slt` now raise `StaticVarLoadError` chained to the original exception (`raise ... from e`). Each error message names the variable, the file path, and provides the `HDF5_USE_FILE_LOCKING=FALSE` fix hint; `slt` notes its non-archive `/pscratch` location and provenance in `scripts/download_slt.py`.
4. Inverted D3's fallback test in `tests/test_static_vars.py` to assert that unreadable invariants hard-fail with `StaticVarLoadError`, checking message contents, path reporting, and exception chaining. Added a `needs_data` test verifying all statics load correctly from the real CFS archive on Perlmutter.
5. Created `tests/test_env.py` covering all environment guard scenarios on the default CI path: setting when unset, preserving pre-existing values with warnings, setting on package import, and setting on direct submodule import.

---

## 2. Definition of Done

- [x] `src/aurora_mjo/env.py` created; docstring carries the full incident including the `Errno -101` text and the 8 September verification
- [x] `configure_environment()` sets the variable only if unset; warns rather than overwriting a different pre-existing value
- [x] Called from `src/aurora_mjo/__init__.py` **before** submodule imports, with a comment saying why the position matters
- [x] The lint suppression used is named and explained (`# isort: skip` and `# noqa: E402` in `run.py` prevent import sorters from moving imports ahead of the environment initialization call)
- [x] Ordering verified before **and** after `netCDF4` import — output pasted
```text
$ env -u HDF5_USE_FILE_LOCKING uv run python -c "
import os
assert 'HDF5_USE_FILE_LOCKING' not in os.environ, 'Must be unset before import'
import aurora_mjo
print('after package import:', os.environ.get('HDF5_USE_FILE_LOCKING'))
import netCDF4
print('netCDF4 imported after:', os.environ.get('HDF5_USE_FILE_LOCKING'))
"
after package import: FALSE
netCDF4 imported after: FALSE
```
- [x] Ordering verified for direct submodule import — output pasted
```text
$ env -u HDF5_USE_FILE_LOCKING uv run python -c "
import os
assert 'HDF5_USE_FILE_LOCKING' not in os.environ, 'Must be unset before import'
from aurora_mjo.dataset import LANLMJODataset
print('after direct submodule import:', os.environ.get('HDF5_USE_FILE_LOCKING'))
import netCDF4
print('netCDF4 imported after direct submodule:', os.environ.get('HDF5_USE_FILE_LOCKING'))
"
after direct submodule import: FALSE
netCDF4 imported after direct submodule: FALSE
```
- [x] `run.py` calls it explicitly
- [x] Both `z` and `lsm` zero-fallbacks replaced with `StaticVarLoadError`
- [x] Every message names the variable, the path, chains the original exception, and names the `HDF5_USE_FILE_LOCKING` fix — one message pasted verbatim
```text
Failed to load invariant variable 'z' from tests/fixtures/synthetic_archive/Step00/ERA5.invariant/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100_remap_180x360MODIS.nc: [Errno -101] NetCDF: HDF error: advisory locking failed. If this error mentions 'NetCDF: HDF error' or 'Errno -101', ensure HDF5_USE_FILE_LOCKING=FALSE is set in your environment to disable advisory file locking on parallel filesystems.
```
- [x] `slt` wrapped with its own contextual error mentioning `/pscratch` and `download_slt.py`
```text
Failed to load static soil type 'slt' from /nonexistent/path/slt_data.nc: [Errno 2] No such file or directory: '/nonexistent/path/slt_data.nc'. Note that 'slt_data.nc' is NOT in the LANL archive; it lives on purgeable /pscratch (default /pscratch/sd/k/kam352/Aurora/slt/slt_data.nc). scripts/download_slt.py is the only record of how it was produced. If this error mentions 'NetCDF: HDF error' or 'Errno -101', ensure HDF5_USE_FILE_LOCKING=FALSE is set in your environment to disable advisory file locking on parallel filesystems.
```
- [x] Existing v3 docstring preserved and extended, not replaced
- [x] D3's fallback test inverted; docstring updated to history; message content asserted
- [x] `tests/test_env.py` covers all four cases; on the default CI path
- [x] `uv run python scripts/check.py` green — summary table pasted below
- [x] A `needs_data` run against real CFS confirms statics still load correctly
```text
$ uv run pytest -m needs_data tests/test_static_vars.py
collected 10 items / 8 deselected / 2 selected

tests/test_static_vars.py ..                                             [100%]
2 passed, 8 deselected, 1 warning in 1.42s
```
- [x] Result file written from `results/_TEMPLATE.md`

---

## 3. The gate

```text
===============================================================================
Gate            Status   Time     Why
-------------------------------------------------------------------------------
lockfile        PASS      0.02s   uv.lock matches pyproject.toml
ruff lint       PASS      0.11s   lint
ruff format     PASS      0.10s   formatting is canonical
types           PASS      0.50s   static types, ratcheted scope
pytest          PASS     28.04s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
```

---

## 4. Measurements

- **Test suite expansion**: Total passing tests on default CI path increased from 61 to 69 (+6 new env tests in `tests/test_env.py`, +2 new hard-fail invariant tests in `tests/test_static_vars.py`, 0 regressions).
- **`needs_data` verification**: 2 tests passed against `/global/cfs/cdirs/m4946/...` in 1.42s.
  - `test_real_cfs_z_mean_matches_prior`: Mean geopotential verified against prior 3709.2466 m² s⁻² (measured ~3709 m² s⁻², far from zero).
  - `test_real_cfs_static_vars_load_correctly`: Verified real `_load_static_vars` on CFS returns tensors of shape `(720, 1440)` with 100% finite values for `z`, `lsm`, and `slt`.

---

## 5. What was ruled out, and by what evidence

- **Creating a separate `errors.py` module**: Ruled out. Defining `StaticVarLoadError` directly in `src/aurora_mjo/env.py` keeps new code strictly confined to the Touches list without introducing an extra single-class file.
- **Overwriting existing `HDF5_USE_FILE_LOCKING` values**: Ruled out. If a user or diagnostic script sets `HDF5_USE_FILE_LOCKING=TRUE` deliberately (e.g. debugging concurrency), `configure_environment()` logs a warning and preserves the setting rather than stomping over user intent.
- **Altering `_clean`**: Ruled out as prohibited by task instructions. Sanitization of NaNs and Infs across statics is legitimate dataset cleanup, distinct from swallowing I/O load failures.

---

## 6. Caveats

NONE (`STATUS: GREEN`).

---

## 7. Observations

- `main` was updated with the D3 merge commit directly (`578ca73`), while local `epic/refactor` had remained at `2bb86a8`. Merged `578ca73` via `--ff-only` into `epic/refactor` prior to branching `epic/refactor-E1-env-guards-hard-fail` so D3's tests and result files were properly inherited.
- Pyrefly type checking automatically checked `src/aurora_mjo/env.py` and `tests/test_env.py` because they are outside `project-excludes` in `pyproject.toml`, reporting 0 errors.

---

## 8. Questions raised

NONE.

---

## 9. Commits

```text
8e622c7  docs(campaign): record task E1 completion results in E1_result.md
99169ed  feat(dataset): fail loudly with StaticVarLoadError on invariant load error
8b76c81  feat(env): configure HDF5_USE_FILE_LOCKING at process start
```

## 10. Files changed

```text
 docs/campaigns/refactor/results/E1_result.md | 161 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 run.py                                       |  19 ++++++++++++-------
 src/aurora_mjo/__init__.py                   |  11 +++++++++++
 src/aurora_mjo/dataset.py                    |  58 ++++++++++++++++++++++++++++++++++++++++++++--------------
 src/aurora_mjo/env.py                        |  78 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/test_env.py                            | 128 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/test_static_vars.py                    | 109 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-------------------
 7 files changed, 524 insertions(+), 40 deletions(-)
```
