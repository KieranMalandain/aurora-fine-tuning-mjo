# A3 — `scripts/check.py`, ruff, type checker, CI

| | |
| --- | --- |
| **Phase** | A |
| **Depends on** | A1 (`PROCEED: YES` or `YES-WITH-CAVEATS`) |
| **Base branch** | `epic/refactor` |
| **Budget** | 90 min |
| **Touches** | `scripts/check.py` (new), `pyproject.toml` (tool config sections only), `.github/workflows/ci.yml` (new), `.pre-commit-config.yaml` (type hook only) |
| **Must not touch** | `[project]` or `[dependency-groups]` in `pyproject.toml` beyond adding the type checker to `dev`. Any `src/`, `scripts/*` (other than the new `check.py`) or `configs/` file. |

## Objective

At the end, `uv run python scripts/check.py` is the single command that defines
"done" for every subsequent task, and CI runs exactly that command.

## Why this is a separate task

Every task from C1 onward reports the output of this command. Building it once,
properly, means sixteen result files are comparable. Building it inside another
task means it gets bolted on and the first agent to hit a red gate deletes a
step.

## What you may assume

- `pyproject.toml`, `.python-version` (3.10) and `uv.lock` exist from A1.
- Line length is **100**. (`01_TARGET_STATE.md`)
- The type gate is a **ratchet**: it covers `src/aurora_mjo/` and `tests/` and
  excludes `scripts/` and `slurm_scripts/` initially. ~6,950 lines of untyped
  scientific Python cannot be made type-clean in one campaign, and pretending
  otherwise produces either `Any` everywhere or a red gate everyone ignores.
  (`01_TARGET_STATE.md` D5)
- Four pytest markers: `needs_data`, `needs_gpu`, `slow`, `live`. CI excludes
  the first, second and fourth. (`01_TARGET_STATE.md` D6)
- There is no `tests/` directory yet — D1 creates it. Your pytest step must
  exit 0 on zero collected tests, not fail.

## Steps

1. **Choose the type checker by measurement, not by preference.** Try `pyrefly`
   first, since that is the template's tool. Add it to the `dev` group, sync,
   and run it. If it does not work on Python 3.10 in this environment, use
   `mypy` with the same scope. Record in your result file: which you tried,
   the exact command, the exact output, and which you chose. See Q-05.

   Either way the step is **named `types`** in `check.py` so the tool can be
   swapped later without changing the contract.

2. Add the tool config sections to `pyproject.toml`:

   ```toml
   [tool.ruff]
   line-length = 100
   target-version = "py310"

   [tool.ruff.lint]
   select = ["E", "F", "I", "UP", "B"]

   [tool.ruff.lint.per-file-ignores]
   "scripts/**" = ["E402"]   # scripts may adjust sys.path before importing
   "run.py" = ["B008"]       # typer's API is function calls in argument defaults

   [tool.pytest.ini_options]
   testpaths = ["tests"]
   addopts = "--strict-markers"
   markers = [
       "needs_data: needs the real CFS ERA5 archive. Excluded from CI.",
       "needs_gpu: needs a CUDA device. Excluded from CI.",
       "slow: takes more than a few seconds.",
       "live: needs a live external system (CDS API, HF hub). Excluded from CI.",
   ]
   ```

   Plus the type-checker section, whose exact key depends on step 1. It **must**
   carry this comment verbatim, because the ratchet only works if the next agent
   understands the direction of travel:

   ```text
   # RATCHET, not a permanent exclusion. Entries come OFF this list, never on.
   # ~6950 lines of untyped scientific Python; trainer.py alone is 1138.
   # See docs/campaigns/refactor/01_TARGET_STATE.md D5.
   ```

3. Write `scripts/check.py`. **Stdlib only** — it must be able to run before
   dependencies are correctly installed, since one of the things it checks is
   whether they are. Base it on the template's version
   (`research-repo-template/scripts/check.py`); it is close to right already.

   Steps in order, each with a `name` and a one-line `why`:

   | Name | Command | Why |
   | --- | --- | --- |
   | `lockfile` | `uv lock --check` | `uv.lock` matches `pyproject.toml` |
   | `ruff lint` | `uv run ruff check .` | lint |
   | `ruff format` | `uv run ruff format --check .` | formatting is canonical |
   | `types` | (from step 1) | static types, ratcheted scope |
   | `pytest` | `uv run pytest -q -m "<marks>"` | tests, excluding what needs data/GPU/network |

   Marker expression: `not live and not needs_data and not needs_gpu`, plus
   `and not slow` when `--fast` is passed.

   Required behaviours, all inherited from the template and all load-bearing:

   - Flags `--fast` (skip `slow`) and `--fix` (run ruff's autofix and formatter
     before gating).
   - **Run every step even after one fails**, then print a summary table. Seeing
     all failures in one run beats fixing them one at a time six CI runs deep.
   - Exit 0 only if every step passed; otherwise exit 1 and name the failing
     gates.
   - If `uv` is not on `PATH`, print a message saying this repo does not support
     pip, venv, poetry or conda, and exit 127.

   One addition specific to this repo: **`pytest` exiting 5 (no tests
   collected) must be treated as a pass** until D1 lands, with a comment saying
   so and saying to remove the special case once `tests/` exists. Without this,
   the gate is red for every task in Phases A–C and everyone learns to ignore it.

4. Add the type-checker hook to `.pre-commit-config.yaml` as a `local` hook
   running from the project venv (`entry: uv run <tool> check`,
   `language: system`, `pass_filenames: false`) so the hook and `check.py` can
   never disagree about which version ran.

5. Write `.github/workflows/ci.yml`: on `pull_request` and on `push` to `main`,
   `runs-on: ubuntu-latest`, concurrency group cancelling in-progress runs,
   `astral-sh/setup-uv` with caching, `uv sync --all-groups`, then
   `uv run python scripts/check.py`.

   **`ubuntu-latest`, not the template's `windows-latest`** — this project is
   Linux-only on Perlmutter and always will be, so a Windows runner would test
   an environment nobody uses. Say that in a comment.

   Do **not** adopt the template's `standards-drift` job (there is no org
   standards URL for this repo) or its `runtime-deps` job (it would install
   torch on every run for little gain here). Note both omissions in a comment.

   Add a comment stating plainly that CI has **no GPU and no CFS mount**, that
   this is why the markers exist, and that a green CI therefore does not mean
   the training path works — F2 is what means that.

6. Verify the gate in **both** directions. A gate that has never been seen red
   has not been tested:

   ```bash
   uv run python scripts/check.py            # expect exit 0
   uv run python scripts/check.py --fast     # expect exit 0
   ```

   Then introduce a deliberate lint error in a scratch file, confirm the gate
   goes red and names `ruff lint` in the summary, and remove it. Paste both
   outputs.

## Definition of Done

- [ ] Type checker chosen by measurement; what was tried, the output, and the
      choice are all recorded
- [ ] `pyproject.toml` has ruff, pytest and type-checker config; the ratchet
      comment is present verbatim
- [ ] `scripts/check.py` exists, is stdlib-only, runs all five steps, continues
      past failures, prints a summary table, and handles missing `uv`
- [ ] `pytest` exit code 5 is treated as a pass, with the comment saying to
      remove that once `tests/` exists
- [ ] `uv run python scripts/check.py` exits 0 — summary table pasted
- [ ] `uv run python scripts/check.py --fast` exits 0 — summary table pasted
- [ ] **Gate proven red**: deliberate lint error → exit 1 with `ruff lint`
      named — output pasted, and confirmation the scratch file was removed
- [ ] `.pre-commit-config.yaml` has the type hook as a local hook
- [ ] `.github/workflows/ci.yml` targets `ubuntu-latest`, runs `check.py`, and
      carries the no-GPU/no-CFS comment
- [ ] `git status --porcelain` is empty
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- Writing any test (D1–D3).
- Fixing any lint or type finding in existing code. If `ruff check` reports
  findings in `src/` or `scripts/`, **do not fix them in this task** — that is
  C1 and C3. If they would make the gate red before C1 can run, add the
  narrowest possible `per-file-ignores` entry with a `# TODO(C1)` or
  `# TODO(C3)` comment naming the task that removes it, and list every ignore
  you added in your result file.
- Configuring branch protection or required status checks — that is a human
  repo-settings action. Record it as a follow-up for F1.
- Adding `nbstripout` (no notebooks).
