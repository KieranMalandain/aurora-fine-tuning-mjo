# E1 — `HDF5_USE_FILE_LOCKING` at process start; statics fail loudly

| | |
| --- | --- |
| **Phase** | E |
| **Depends on** | D3 (`PROCEED: YES`) |
| **Base branch** | `epic/refactor` |
| **Budget** | 2 h |
| **Touches** | `src/aurora_mjo/env.py` (new), `src/aurora_mjo/__init__.py`, `src/aurora_mjo/dataset.py` (`_load_static_vars` only), `run.py` (one import), `tests/test_static_vars.py`, `tests/test_env.py` (new) |
| **Must not touch** | any other function in `dataset.py`. `_clean` — it is legitimate and different. `slurm_scripts/` (E3). `configs/unified.yaml`. |

## Objective

At the end, `HDF5_USE_FILE_LOCKING=FALSE` is set at process start in Python
rather than depending on a shell script, and a static variable that cannot be
loaded is a hard failure with a message that names the file and the fix — not a
silent zero tensor.

## Why this is a separate task

These are the two fixes that make every other failure legible. They are also
behaviour changes on the data path, so they land **after** D3 has a test around
current behaviour, so the change is visible as a deliberately inverted test
rather than as a mysterious diff.

## What you may assume

- D3 wrote `tests/test_static_vars.py` asserting the **current** zero-fallback
  behaviour, with a docstring saying **E1 inverts this test**. Inverting it is
  part of your job, not a conflict. Read that docstring first.
- `02_UPSTREAM_CONTRACT.md` §4.3–4.5 has the full reasoning.
- `HDF5_USE_FILE_LOCKING=FALSE` is currently set **only** in
  `slurm_scripts/train_auto.slurm`. Not in `eval.slurm`, `test_train.slurm`,
  `submit_chain.sh`, or any interactive session.
- Verified 8 September 2026: with the variable set, a previously failing CFS
  file opens cleanly (180×360×124, variable `t2`).
- B2 may have determined whether the July runs were affected. **E1 proceeds
  regardless of B2's answer** — the code should hard-fail either way; B2 only
  determines whether past results mean anything.
- `_clean` (`torch.nan_to_num` across all three statics) is a **different**
  mechanism and is legitimate. Do not conflate it with the fallback. `02_…`
  §4.3 says so explicitly.

## Steps

1. Write `src/aurora_mjo/env.py`. One function, `configure_environment()`, and a
   module docstring carrying the full incident: the `Errno -101` text, that the
   files exist and are readable, that the cause is HDF5 advisory locking against
   a parallel filesystem, the 8 September verification, and that a guard living
   only in a shell script is not a guard.

   Behaviour:

   - Set `HDF5_USE_FILE_LOCKING=FALSE` **only if not already set.** If it is set
     to something else, log at warning level rather than overwriting — a human
     who set it deliberately outranks a default.
   - Return what it changed, so a caller can log it.
   - Be importable and callable with **no side effects beyond `os.environ`**. No
     torch import, no logging config, nothing heavy.

   **The ordering constraint that makes this work:** the variable must be set
   **before HDF5 is initialised**, which happens on first `netcdf4`/`h5py`
   import. Call `configure_environment()` from `src/aurora_mjo/__init__.py`, at
   the top, before any submodule import. Add a comment saying so — this is
   exactly the kind of ordering that a later import-sorting pass would silently
   break, and `ruff`'s `I` rule will want to move it.

   You will likely need an `# isort: skip` / `# noqa: E402` with a comment
   explaining why. State which you used.

2. Verify the ordering actually holds. Do not trust it:

   ```bash
   uv run python -c "
   import os
   assert 'HDF5_USE_FILE_LOCKING' not in os.environ
   import aurora_mjo
   print('after package import:', os.environ.get('HDF5_USE_FILE_LOCKING'))
   import netCDF4
   print('netCDF4 imported after:', os.environ.get('HDF5_USE_FILE_LOCKING'))
   "
   ```

   Also confirm it holds when a submodule is imported directly
   (`from aurora_mjo.dataset import LANLMJODataset`) without importing the
   package root first — that is the common case in `scripts/`.

3. Add an explicit call in `run.py` before anything else, even though the
   package `__init__` covers it. Belt and braces is correct here: the cost is
   one line and the failure mode is an eleven-hour job dying at hour zero.

4. **Invert the fallback in `_load_static_vars`.** Replace both
   `except Exception → zeros + warn` blocks with a hard failure. Requirements
   for the exception message — all four, because the whole point is that the
   next person does not have to rediscover this:

   - names the variable (`z` or `lsm`)
   - names the file path that failed
   - includes the original exception, chained with `raise … from e`
   - **names the fix**: says to set `HDF5_USE_FILE_LOCKING=FALSE` if the error
     mentions `NetCDF: HDF error` or `Errno -101`

   Use a narrow custom exception, e.g. `StaticVarLoadError(RuntimeError)`, in
   `env.py` or a small `errors.py`. Say which and why.

   Keep the comment history: the existing docstring records the v3 change
   ("NaN/Inf are zeroed for ALL three statics"). Preserve it and **add** to it —
   say what the fallback used to do, why it was dangerous (a planet with no
   topography and no continents, two warnings in an 11-hour log), and that
   `docs/findings/2026-09-zeroed-statics.md` (B2) records whether it ever fired
   in production.

5. **Do the same for `slt`.** It currently has no handler at all, so it already
   raises — but it raises a bare `KeyError`/`OSError` with no context. Wrap it in
   the same `StaticVarLoadError` naming `slt_data.nc`, that it is **not** in the
   LANL archive, that it lives on purgeable `/pscratch`, and that
   `scripts/download_slt.py` is the only record of how it was produced.
   (`02_…` §4.4, Q-07)

   Do **not** relocate the file. Q-07 defers that to a human.

6. **Invert D3's test.** In `tests/test_static_vars.py`, change the
   current-behaviour test to assert `StaticVarLoadError` is raised, and update
   the docstring: remove the "E1 inverts this" note and replace it with the
   history — that this used to return zeros with a warning, and why that was
   worse. Assert the message contains the file path and the
   `HDF5_USE_FILE_LOCKING` hint, because an error message is a feature here and
   features get tested.

7. Write `tests/test_env.py`: `configure_environment()` sets the variable when
   unset; does **not** overwrite a pre-existing different value; importing
   `aurora_mjo` sets it; importing `aurora_mjo.dataset` directly also sets it.
   These run on the default CI path.

## Definition of Done

- [ ] `src/aurora_mjo/env.py` created; docstring carries the full incident
      including the `Errno -101` text and the 8 September verification
- [ ] `configure_environment()` sets the variable only if unset; warns rather
      than overwriting a different pre-existing value
- [ ] Called from `src/aurora_mjo/__init__.py` **before** submodule imports,
      with a comment saying why the position matters
- [ ] The lint suppression used is named and explained
- [ ] Ordering verified before **and** after `netCDF4` import — output pasted
- [ ] Ordering verified for direct submodule import — output pasted
- [ ] `run.py` calls it explicitly
- [ ] Both `z` and `lsm` zero-fallbacks replaced with `StaticVarLoadError`
- [ ] Every message names the variable, the path, chains the original exception,
      and names the `HDF5_USE_FILE_LOCKING` fix — one message pasted verbatim
- [ ] `slt` wrapped with its own contextual error mentioning `/pscratch` and
      `download_slt.py`
- [ ] Existing v3 docstring preserved and extended, not replaced
- [ ] D3's fallback test inverted; docstring updated to history; message content
      asserted
- [ ] `tests/test_env.py` covers all four cases; on the default CI path
- [ ] `uv run python scripts/check.py` green — summary table pasted
- [ ] A `needs_data` run against real CFS confirms statics still load correctly
      **or** it is stated that this was not run and why
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- Editing any SLURM script (E3). E3 adds the variable to the remaining three
  and re-points them at `run.py`.
- Relocating `slt_data.nc` (Q-07).
- Any other function in `dataset.py`. In particular, do not touch
  `_build_aligned_index`, `_collect_var_files`, `_read_var_at_times` or
  `_probe_pressure_level_indices`.
- Changing `_clean`. It is legitimate.
- Widening any other `except`. If you find another over-broad handler, that is
  an Observation.
- Acting on B2's finding. Recording it in the docstring is in scope; drawing
  scientific conclusions is not.
