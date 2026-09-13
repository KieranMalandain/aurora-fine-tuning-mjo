# A1 — conda → `uv`: `pyproject.toml`, lockfile, retire `environment.yml`

| | |
| --- | --- |
| **Phase** | A |
| **Depends on** | none |
| **Base branch** | `epic/refactor` |
| **Budget** | 90 min |
| **Touches** | `pyproject.toml` (new), `.python-version` (new), `uv.lock` (new), `environment.yml` (delete), `docs/campaigns/refactor/QUESTIONS.md` |
| **Must not touch** | anything under `src/`, `scripts/`, `slurm_scripts/`, `configs/`. The conda env on Perlmutter — **do not delete or modify it.** |

## Objective

At the end, the repository declares its dependencies in a `pyproject.toml` with
a committed `uv.lock`, installs the first-party source as an editable package,
and no longer contains `environment.yml`. The existing conda environment is
untouched and still works.

## Why this is a separate task

Everything else in Phase C and later runs under `uv`. If the dependency
resolution is wrong, every subsequent task inherits a broken environment and the
failures will look like code failures. This task also carries the campaign's
largest single unknown (Q-03), so isolating it means a failure here costs one
task rather than five.

## What you may assume

- Target Python is **3.10**, torch **2.5.1+cu121**, `microsoft-aurora` from pip.
  (`00_CONTEXT.md` §2.2)
- `environment.yml` is **not** the source of truth. It is unpinned and it lies —
  it lists conda `pytorch` while the real environment has `microsoft-aurora`
  from pip. (`00_CONTEXT.md` §2.2)
- The dependency ground truth is `$HOME/aurora-backup/aurora_mjo-explicit-*.txt`
  (a `conda list --explicit` export), with `-full.yml` and `-list.txt` alongside.
  (`cleanup-2026-09.md` §8)
- The first-party package will be `aurora_mjo` at `src/aurora_mjo/`, created in
  C1. (`01_TARGET_STATE.md` D1, Q-01)

## Steps

1. If you have access to `$HOME/aurora-backup/`, read all three env exports and
   record the exact pinned versions of: `python`, `torch`, `numpy`, `xarray`,
   `dask`, `netcdf4`, `h5netcdf`, `pandas`, `matplotlib`, `microsoft-aurora`.
   If you do **not** have access, say so plainly in your result file and derive
   from `environment.yml` plus the four versions given in `00_CONTEXT.md` §2.2 —
   then raise a question so the human can supply the export.

2. Write `.python-version` containing exactly `3.10`.

3. Write `pyproject.toml`. Required shape:

   ```toml
   [project]
   name = "aurora-mjo"
   version = "0.1.0"
   description = "Fine-tuning Microsoft Aurora for MJO sub-seasonal prediction."
   requires-python = ">=3.10,<3.11"   # NARROW: microsoft-aurora's constraint

   dependencies = [
       # Runtime only — everything imported by src/ or scripts/ at runtime,
       # even if it currently arrives via another package's dependencies.
       "microsoft-aurora",
       "torch==2.5.1",
       "numpy",
       "xarray",
       "netcdf4",       # dataset.py passes engine="netcdf4" explicitly
       "dask",
       "pandas",
       "pyyaml",        # config loading
       "typer",         # run.py, added in C2
       "matplotlib",    # scripts/plot_loss.py
   ]

   [dependency-groups]
   dev = ["ruff>=0.6", "pytest>=8.3", "pre-commit>=3.8"]

   [[tool.uv.index]]
   name = "pytorch-cu121"
   url = "https://download.pytorch.org/whl/cu121"
   explicit = true

   [tool.uv.sources]
   torch = { index = "pytorch-cu121" }

   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [tool.hatch.build.targets.wheel]
   packages = ["src/aurora_mjo"]
   ```

   Pin the versions you actually measured in step 1 rather than leaving bare
   names. **Add a comment next to any version you could not verify**, saying so.

   The type checker, ruff and pytest configuration blocks are **A3's** job, not
   yours. Do not add them.

4. `packages = ["src/aurora_mjo"]` points at a directory C1 has not created yet.
   Create `src/aurora_mjo/__init__.py` containing only a module docstring and
   `__version__ = "0.1.0"`, so the build backend has a target. **Do not move any
   other file** — that is C1.

5. Run `uv lock`. Commit `uv.lock`.

6. Run `uv sync --all-groups`, then verify the environment:

   ```bash
   uv run python -c "import torch; print(torch.__version__)"
   uv run python -c "import aurora; print('microsoft-aurora at', aurora.__file__)"
   uv run python -c "import aurora_mjo; print(aurora_mjo.__version__)"
   uv run python -c "import xarray, numpy; print(xarray.__version__, numpy.__version__)"
   ```

7. Delete `environment.yml`. Its content is preserved in git history and in the
   `$HOME/aurora-backup/` exports; say both of those things in the commit
   message.

8. If any of steps 5–6 failed, or if you could not test on Perlmutter, append a
   question to `QUESTIONS.md` recording exactly what you ran, what happened, and
   what you recommend. Then report **AMBER** with specific caveats. See Q-03 for
   the expected fallback position.

## Definition of Done

Copy verbatim into your result file and tick each box. Paste actual output for
every item that has a command.

- [ ] `.python-version` contains `3.10`
- [ ] `pyproject.toml` exists; `requires-python` is `>=3.10,<3.11`; every
      dependency either carries a measured pin or a comment saying it is
      unverified
- [ ] `src/aurora_mjo/__init__.py` exists with `__version__`; **no other file
      moved**
- [ ] `uv lock` succeeded and `uv.lock` is committed **in the same commit as
      `pyproject.toml`**
- [ ] `uv lock --check` exits 0
- [ ] `uv sync --all-groups` succeeded — output pasted
- [ ] `uv run python -c "import torch; print(torch.__version__)"` prints a
      2.5.1 build — output pasted
- [ ] `import aurora` resolves to site-packages and `import aurora_mjo` resolves
      into `src/` — both paths pasted
- [ ] `environment.yml` deleted
- [ ] Where the environment was tested (laptop / login node / GPU node) is
      stated explicitly
- [ ] Result file written from `results/_TEMPLATE.md`

There is no `scripts/check.py` yet — **A3 builds it**. Do not attempt to run it.

## Out of scope

- Creating or moving any source file other than `src/aurora_mjo/__init__.py`.
- `ruff`, type checker or `pytest` configuration (A3).
- `.pre-commit-config.yaml` (A2).
- Deleting, modifying or migrating the conda environment on Perlmutter. It stays
  until F2 passes, and this campaign never deletes it.
- Upgrading any package. The pins are constraints inherited from a working
  training run, not preferences. If you believe something should be upgraded,
  that is an Observation.
