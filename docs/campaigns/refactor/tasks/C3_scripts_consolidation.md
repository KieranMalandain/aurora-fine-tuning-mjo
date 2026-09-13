# C3 — Fold `tools/` into `scripts/`; dead code; RMM extraction

| | |
| --- | --- |
| **Phase** | C |
| **Depends on** | C1 (`PROCEED: YES`) |
| **Base branch** | `epic/refactor` |
| **Budget** | 2.5 h |
| **Touches** | `scripts/`, `tools/` (delete), `notebooks/` (delete), `test_dataset.py`, `inspect_vars.py`, `src/aurora_mjo/rmm/` (new), `src/aurora_mjo/trainer.py` (**one** decision only, see step 4) |
| **Must not touch** | `src/aurora_mjo/{dataset,model,loss,checkpoint}.py`. `run.py` structure beyond adding two commands. `configs/unified.yaml`. |

## Objective

At the end there is one `scripts/` directory of run-never-imported entry points,
`tools/` and `notebooks/` are gone, the two stray root-level scripts have moved,
the dead dummy-dataset path is resolved, and the importable core of the RMM code
lives in the package.

## Why this is a separate task

It touches many files but each change is small and independent, so it parallelises
with C2 (which touches a disjoint set). It also carries the campaign's one
genuinely open scoping question — Q-08, the 1,637-line RMM split — and isolating
it means an `AMBER` there does not contaminate anything else.

## What you may assume

- Modules import as `aurora_mjo.<module>` (C1).
- `scripts/download_era5.py` and `notebooks/t_nb.py` are **zero bytes**.
  (`00_CONTEXT.md` §2.5)
- `tools/probe_results/*.json` were untracked by A2, and their numbers are
  already transcribed into `03_DOMAIN_PRIORS.md` §7, so nothing is lost by
  removing the files.
- `scripts/evaluate_mjo.py` (984 lines) and `scripts/compute_rmm.py` (653
  lines) together are 1,637 lines — nearly a quarter of the repo.
  **Read Q-08 before starting step 5.** The default is a *mechanical* move
  only, and reporting `AMBER` with the files left in place is an explicitly
  acceptable outcome.
- `src/aurora_mjo/trainer.py` line ~307 has a re-pointed but still broken
  `from aurora_mjo.dummy_dataset import …` carrying a `# TODO(C3)` from C1.

## Steps

1. **Move `tools/` into `scripts/`** using the build guide's naming convention
   (`probe_*`, `diagnose_*`):

   ```bash
   git mv tools/diagnose_val_nan.py    scripts/diagnose_val_nan.py
   git mv tools/probe_model_size.py    scripts/probe_model_size.py
   git mv tools/repro_ima_matrix.py    scripts/probe_ima_matrix.py
   git rm -r tools/
   ```

   `repro_ima_matrix.py` → `probe_ima_matrix.py` because it searches a
   configuration space for a crash-free setting; that is a probe. Update any
   reference to the old name — check `configs/unified.yaml`'s comments and
   `docs/`, and report anything you find rather than editing `configs/`.

2. **Move the two stray root scripts.**

   ```bash
   git mv test_dataset.py   scripts/archive/verify_dataset_smoke.py
   git mv inspect_vars.py   scripts/inspect_vars.py
   ```

   `test_dataset.py` goes to `scripts/archive/` **and gets renamed**, for two
   reasons: it is superseded by `scripts/verify_dataset_loader.py` (which does
   strictly more), and a file named `test_*.py` at the repo root will be
   collected by pytest and fail, because it calls `sys.exit(1)` on error rather
   than asserting. Per the build guide, write the answer into the archived
   file's docstring: what it verified, what superseded it, and that D2 converts
   the real check into a proper test.

3. **Delete the debris.**

   ```bash
   git rm scripts/download_era5.py    # zero bytes
   git rm -r notebooks/               # contains only a zero-byte t_nb.py
   ```

   Confirm both are actually zero bytes before deleting, and paste the
   confirmation. If either is not, stop — someone's work is in there.

4. **Resolve the dummy-dataset dead path.** This is a decision, and the
   reasoning must be in your result file.

   Facts: `src/dummy_dataset.py` was deleted during consolidation.
   `configs/unified.yaml` sets `use_dummy: false` and retains an empty
   `data.dummy` block. `slurm_scripts/eval.slurm` passes
   `--override data.use_dummy=true`, which would detonate it — and that script
   is already non-functional for three other reasons.
   `train.py --smoke-test` provides a synthetic in-process loader, which covers
   the actual need the dummy dataset served.

   **Default: remove the `use_dummy` branch entirely** from
   `build_dataloader`, along with the `data.dummy` block's use. Raise a
   `ValueError` naming `--smoke-test` if `use_dummy: true` is ever set, so the
   failure is loud rather than an `ImportError` on a missing module.

   **Do not delete the `data.dummy` block from `configs/unified.yaml`** — that
   would change the resolved config and break C4's diff against B1. Leave it,
   and note it for E2 to reject at validation time.

   This is a real behaviour change on a dead path, so: state it prominently,
   and expect C4 to confirm nothing else moved.

5. **RMM extraction — read Q-08 first.**

   Create `src/aurora_mjo/rmm/__init__.py`, `compute.py`, `evaluate.py`. Move
   the **pure** functions out of `scripts/compute_rmm.py` and
   `scripts/evaluate_mjo.py` with **no signature changes and no numerical
   changes**. Leave the two scripts as thin `argparse` wrappers that import from
   the package.

   Hard constraints:

   - No renaming. No reordering. No "while I'm here" improvements.
   - If a function mixes computation and file I/O, **leave it in `scripts/`**.
     Splitting it is a numerical risk this task does not have budget for.
   - `scripts/evaluate_mjo.py:449` has a lazy `from aurora.batch import
     Metadata` — that is the **third-party** package. Do not rewrite it to
     `aurora_mjo`. (`00_CONTEXT.md` §4)
   - RMM computation is where a leakage bug would hide
     (`03_DOMAIN_PRIORS.md` §6). If you notice anything computing statistics
     over the full record rather than the training period, **do not fix it** —
     record it as an Observation. It is a science finding, not a refactor.

   **If this does not fit the budget, report `AMBER`, leave both files in
   `scripts/` untouched, and say so plainly.** Q-08 pre-authorises that. A
   half-split 1,637-line numerical module is much worse than an unsplit one.

6. Add the two remaining `run.py` commands, thin wrappers only:

   ```text
   uv run python run.py evaluate   --mode baseline --checkpoint latest
   uv run python run.py norm-stats --years 1980 2015
   ```

   `norm-stats` calls into `scripts/calc_norm_stats.py`'s logic. If that logic
   is not importable without a large move, wire the command to fail with a
   message naming the script to run directly, and note it. A command that
   honestly says "not wired yet, run X" is better than one that half-works.

7. Fix stale references **in Python files only** — `scripts/evaluate_mjo.py` and
   `scripts/verify_shapes.py` both reference deleted configs like
   `phase1_baseline.yaml`. Point them at `configs/unified.yaml --mode <m>`.
   Markdown files are **F1's**; list what you saw as Observations.

8. Verify:

   ```bash
   uv run python scripts/check.py
   for f in scripts/*.py; do uv run python -c "import ast,sys; ast.parse(open('$f').read())" || echo "PARSE FAIL $f"; done
   for f in scripts/*.py; do timeout 60 uv run python "$f" --help >/dev/null 2>&1 && echo "OK   $f" || echo "NOHELP $f"; done
   uv run python run.py --help
   ```

   `NOHELP` is not necessarily a failure — several scripts predate having a
   `--help`. Record which, so F1 knows what `docs/CLI.md` cannot document.

## Definition of Done

- [ ] `tools/` and `notebooks/` do not exist; `repro_ima_matrix.py` renamed to
      `probe_ima_matrix.py`; references to the old name found and listed
- [ ] `test_dataset.py` archived as `scripts/archive/verify_dataset_smoke.py`
      with the answer written into its docstring
- [ ] `inspect_vars.py` moved to `scripts/`
- [ ] Zero-byte confirmation pasted before deleting `download_era5.py` and
      `notebooks/`
- [ ] Dummy-dataset path resolved; the decision and its reasoning stated; the
      loud `ValueError` present; **`configs/unified.yaml` unchanged**
- [ ] `src/aurora_mjo/rmm/` created with pure functions moved and no signature
      changes — **or** `AMBER` with a plain statement that the split did not fit
      and both files were left untouched
- [ ] `scripts/evaluate_mjo.py:449`'s third-party `aurora` import confirmed
      untouched
- [ ] `run.py evaluate` and `run.py norm-stats` exist, or fail with a message
      naming what to run instead
- [ ] Stale config references fixed in `.py` files; `.md` files listed as
      Observations, not edited
- [ ] Every `scripts/*.py` parses; per-script `--help` result recorded
- [ ] `uv run python scripts/check.py` green — summary table pasted
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- Any edit to `dataset.py`, `model.py`, `loss.py`, `checkpoint.py`.
- Any edit to `trainer.py` other than the step-4 dummy-dataset decision.
- Any numerical change in the RMM code. Moving a function is in scope;
  changing what it computes is not, **even to fix a bug you are confident
  about**. Record it.
- Editing any `.md` file (F1) or any SLURM script (E3).
- Editing `configs/unified.yaml`, including deleting the now-unused `data.dummy`
  block. C4 diffs the resolved config against B1.
- Writing tests (D1–D3).
