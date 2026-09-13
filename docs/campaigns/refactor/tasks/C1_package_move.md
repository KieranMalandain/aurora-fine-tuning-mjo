# C1 — `src/*.py` → `src/aurora_mjo/`; re-point 27 imports

| | |
| --- | --- |
| **Phase** | C |
| **Depends on** | A1, A3, **B1** — all with `PROCEED: YES` |
| **Base branch** | `epic/refactor` |
| **Budget** | 2 h |
| **Touches** | `src/` (all files move), every file containing a `from src.` import, `tests/test_imports.py` (new) |
| **Must not touch** | `configs/unified.yaml`. `slurm_scripts/` (E3 handles those). Any logic — **this is a move, not a refactor.** |

## Objective

At the end, the five source modules live in `src/aurora_mjo/`, are imported as
`aurora_mjo.<module>`, and every one of the 27 first-party import statements
points at the new location. **No logic changes.**

## Why this is a separate task

It is a large mechanical edit across many files with one specific catastrophic
failure mode — shadowing `microsoft-aurora` — and one specific silent failure
mode: an agent "tidying up" while moving. Separating the move from the CLI change
(C2) and the scripts consolidation (C3) means C4's gate can attribute a mismatch
to one of three tasks instead of one big one.

## What you may assume

- **B1 has captured the behavioural baseline.** If it has not, stop and report
  `BLOCKED`. (`README.md` Phase B note)
- `src/aurora_mjo/__init__.py` already exists from A1, containing a docstring
  and `__version__`.
- `pyproject.toml` already declares
  `[tool.hatch.build.targets.wheel] packages = ["src/aurora_mjo"]`.
- **The collision is real and is the whole reason for the name.**
  `microsoft-aurora` owns the top-level `aurora` import name and is imported at
  nine sites across six first-party files. The exact list is in
  `00_CONTEXT.md` §4 — read it before you start.
- There are **27** `from src.…` / `import src.…` statements in the repo. Verify
  the count yourself before and after; if it is not 27, say so.

## Steps

1. Count and list, before touching anything:

   ```bash
   grep -rn "from src\.\|import src\." --include=*.py . | tee /tmp/c1_before.txt
   wc -l /tmp/c1_before.txt
   ```

   Paste this into your result file. It is your checklist.

2. Move the five modules with `git mv` so history follows:

   ```bash
   git mv src/dataset.py    src/aurora_mjo/dataset.py
   git mv src/model.py      src/aurora_mjo/model.py
   git mv src/loss.py       src/aurora_mjo/loss.py
   git mv src/trainer.py    src/aurora_mjo/trainer.py
   git mv src/checkpoint.py src/aurora_mjo/checkpoint.py
   git rm src/__init__.py   # the old empty one; aurora_mjo/__init__.py replaces it
   ```

   Commit this as its own commit, before re-pointing imports, so the move and
   the edits are separable in review.

3. Re-point every import. `from src.dataset import X` → `from aurora_mjo.dataset
   import X`, and so on. Do it file by file against your step-1 list, not with a
   blind `sed` across the tree — `sed` will also rewrite the string `"src."`
   inside docstrings and comments, which changes documentation silently.

   **Intra-package imports:** modules inside `aurora_mjo` importing each other
   should use explicit relative imports (`from .dataset import …`) or absolute
   `aurora_mjo.` imports. Pick one and be consistent; say which you picked.

4. **The one place that needs care.** `src/trainer.py:307` contains

   ```python
   from src.dummy_dataset import MJODataset, load_and_combine_files
   ```

   and `src/dummy_dataset.py` **does not exist** — it was deleted during
   consolidation. It is a lazy import inside the `if use_dummy:` branch, so
   nothing hits it today (`unified.yaml` sets `use_dummy: false`).

   **In this task, re-point it to `aurora_mjo.dummy_dataset` and leave it
   broken.** Do not delete it, do not fix it, do not remove the `use_dummy`
   branch. Add a `# TODO(C3)` comment on the line saying the module does not
   exist and C3 owns the decision. Rationale: deleting the dummy path is a
   behaviour change, C4 must gate it, and C3 is where dead-code decisions live.
   Note it as an Observation as well.

5. Add `tests/test_imports.py`. It must assert **the collision does not
   happen**, which is the single test that would have caught the failure this
   whole naming decision exists to avoid:

   ```python
   """Guards the microsoft-aurora / aurora_mjo name collision.

   microsoft-aurora owns the top-level `aurora` import name. A first-party
   package called `aurora` would shadow it, surfacing as an ImportError on
   `Batch`. See docs/campaigns/refactor/00_CONTEXT.md section 4.
   """
   from __future__ import annotations

   import aurora
   import aurora_mjo


   def test_third_party_aurora_resolves_to_site_packages() -> None:
       assert "site-packages" in aurora.__file__ or "dist-packages" in aurora.__file__


   def test_first_party_package_is_not_shadowing() -> None:
       assert aurora_mjo.__file__ != aurora.__file__
       assert "aurora_mjo" in aurora_mjo.__file__


   def test_aurora_batch_still_importable() -> None:
       from aurora import Batch, Metadata  # noqa: F401


   def test_all_first_party_modules_import() -> None:
       from aurora_mjo import checkpoint, dataset, loss, model, trainer  # noqa: F401
   ```

   `tests/` may not exist yet — D1 builds the full scaffold. Create just the
   directory and this one file; a bare `tests/test_imports.py` collects fine
   without a `conftest.py`.

   Note the last test imports `trainer`, which contains the broken lazy import
   from step 4 — that is fine, it is lazy and inside a branch, so module import
   succeeds. If it does **not** succeed, say so loudly; it means the import is
   not as lazy as believed.

6. Verify:

   ```bash
   grep -rn "from src\.\|import src\." --include=*.py .   # expect NOTHING
   uv run pytest tests/test_imports.py -v
   uv run python -c "from aurora_mjo import dataset, model, loss, trainer, checkpoint; print('ok')"
   uv run python scripts/check.py
   ```

   Also confirm `train.py` still runs its argument parser:
   `uv run python train.py --help`.

## Definition of Done

- [ ] Before-list of all `from src.` imports pasted, with its count
- [ ] Five modules moved with `git mv`; move is its own commit
- [ ] Old `src/__init__.py` removed
- [ ] Every import re-pointed; `grep -rn "from src\.\|import src\."` returns
      nothing — output pasted
- [ ] Intra-package import convention chosen and stated
- [ ] `trainer.py`'s `dummy_dataset` import re-pointed, still broken, carrying a
      `# TODO(C3)` comment — the line pasted
- [ ] `tests/test_imports.py` added with all four tests; all pass — output pasted
- [ ] `uv run python train.py --help` works — output pasted
- [ ] `uv run python scripts/check.py` green — summary table pasted
- [ ] **`git diff --stat` reviewed line by line and confirmed to contain no
      logic change** — stat pasted, and any hunk that is not an import
      re-pointing explained individually
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Any logic change at all.** No renaming a variable, no extracting a helper,
  no fixing a lint finding beyond what the import move itself creates, no
  reordering functions, no adding type hints to existing code. If `ruff` or the
  type checker complains about pre-existing code, add the narrowest
  `per-file-ignores` with a `# TODO` naming the task that removes it, and list
  every one you added.
- Creating `run.py` or editing `train.py` beyond the import re-point (C2).
- Deleting the dummy-dataset path (C3).
- Moving `tools/` or anything in `scripts/` (C3).
- Building the rest of `tests/` (D1).
- Touching `configs/unified.yaml` — the config references no module paths, so it
  should need no change. **If you find that it does, stop and report**: that
  would mean config and code are coupled in a way nobody has documented.
