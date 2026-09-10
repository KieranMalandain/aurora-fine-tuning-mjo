# A2 — `.gitignore`, untrack artifacts, archive the July metrics

| | |
| --- | --- |
| **Phase** | A |
| **Depends on** | none |
| **Base branch** | `epic/refactor` |
| **Budget** | 45 min |
| **Touches** | `.gitignore`, `.gitattributes` (new), `.pre-commit-config.yaml` (new), `docs/archive/metrics/` (new), and removal-from-tracking of the six files listed below |
| **Must not touch** | any `.py`, `.yaml` or `.slurm` file. Any git ref in `cleanup-2026-09.md` §8. |

## Objective

At the end, `git status` is clean, no build artifact or run output is tracked,
and the two `metrics.jsonl` files that are currently tracked *on purpose* have
been moved somewhere durable rather than left inside an ignored directory.

## Why this is a separate task

It is pure hygiene with no behavioural risk, so it can run concurrently with A1
and A4. It is also the one task where a careless `git rm` destroys the only
surviving evidence about the July runs — which is why the exception in step 3 is
spelled out rather than left to judgement.

## What you may assume

- Six paths are tracked and should not be (`00_CONTEXT.md` §2.4 item 5):

  ```text
  checkpoints/baseline/metrics.jsonl      ~21 KB   ← EXCEPTION, see step 3
  checkpoints/lora/metrics.jsonl          ~368 B   ← EXCEPTION, see step 3
  test_outputs/diagnose_val_nan_outputs.txt
  tools/probe_results/huge.json
  tools/probe_results/small.json
  docs/verify_output.txt                  ← keep tracked; F1 archives it
  docs/verify_output_slt.txt              ← keep tracked; it is current evidence
  ```

- The current `.gitignore` contains **deliberate `!` negations** to keep the two
  `metrics.jsonl` files tracked despite ignoring `checkpoints/`. That was
  intentional. (`01_TARGET_STATE.md` D7)
- The two `docs/verify_output*.txt` files are the *only* committed record of
  dataset shapes and static-variable statistics, and `03_DOMAIN_PRIORS.md` §2–3
  is built from them. **They stay tracked.** F1 archives the stale one with a
  header; that is not your job.
- `tools/` is deleted by C3, but its two JSON files are cited in
  `03_DOMAIN_PRIORS.md` §7 — the numbers are already transcribed there, so
  untracking the files loses nothing.

## Steps

1. Rewrite `.gitignore`. Cover: `__pycache__/`, `*.py[cod]`, `.venv/`,
   `checkpoints/`, `*.pt`, `*.nc`, `*.grib`, `*.idx`, `slurm_logs/`, `*.log`,
   `experiment_outputs/`, `test_outputs/`, `tools/probe_results/`,
   `.pytest_cache/`, `.ruff_cache/`, `.ipynb_checkpoints/`, `.DS_Store`,
   `dist/`, `build/`, `*.egg-info/`, `.env` and `.env.*` with
   `!.env.example`.

   Two comments are required, because both rules are non-obvious:
   - `# uv.lock IS committed — it is the reproducibility guarantee.`
   - `# tests/fixtures/*.nc ARE committed — they are small and synthetic. See D1.`
     with the matching `!tests/fixtures/**` negation.

   Remove the old `!checkpoints/*/metrics.jsonl` negations — step 3 makes them
   unnecessary. Drop the stale `investigation_phase1/` entry (that directory
   only ever existed on local storage).

2. Write `.gitattributes` normalising line endings: `* text=auto eol=lf`, plus
   `*.nc binary`, `*.pt binary`, `*.pdf binary`, `*.png binary`, and
   `uv.lock -diff` so lockfile diffs do not swamp a review.

3. **The exception — do this before any `git rm`.** Create
   `docs/archive/metrics/` and move both metrics files there:

   ```bash
   mkdir -p docs/archive/metrics
   git mv checkpoints/baseline/metrics.jsonl docs/archive/metrics/baseline-2026-07.jsonl
   git mv checkpoints/lora/metrics.jsonl     docs/archive/metrics/lora-2026-07.jsonl
   ```

   Add `docs/archive/metrics/README.md`, about ten lines, saying: these are the
   loss/metric traces from the July 2026 Perlmutter runs; the runs are declared
   fully non-finite by `AURORA_MJO_GAMEPLAN.md` so the checkpoints themselves
   were deliberately discarded; these traces are retained as evidence of *how*
   a run went non-finite; they were previously tracked inside an otherwise
   gitignored `checkpoints/` directory via `.gitignore` negations, which is a
   configuration that survives exactly until someone simplifies the
   `.gitignore`.

   **Record the byte count of each file before and after the move** and put both
   numbers in your result file. They must be identical.

4. Untrack the rest without deleting the working copies:

   ```bash
   git rm --cached test_outputs/diagnose_val_nan_outputs.txt
   git rm --cached tools/probe_results/huge.json tools/probe_results/small.json
   ```

   Note `--cached`: the files stay on disk. C3 deals with `tools/`.

5. Write `.pre-commit-config.yaml` with `ruff` (`--fix`) and `ruff-format` from
   `astral-sh/ruff-pre-commit`, plus `gitleaks`. Add a comment that these hooks
   are a fast subset of `scripts/check.py` and not a substitute for it, because
   pre-commit sees only staged files.

   **Do not** add the template's `nbstripout` hook — there are no notebooks
   (`notebooks/t_nb.py` is a zero-byte file that C3 deletes). **Do not** add the
   type-checker hook; A3 chooses the tool and owns that config.

   Install and run it once over everything:

   ```bash
   uv run pre-commit install
   uv run pre-commit run --all-files
   ```

   This will produce formatting changes across the existing source. **That is
   acceptable and expected here**, because it happens *before* B1's fingerprint
   and C1's move. Commit it as its own commit titled as a pure formatting pass so
   that a reviewer can skip it. Formatting must not change behaviour — if ruff's
   autofix wants to change anything that is not whitespace, quotes or import
   order, stop and report it instead of accepting it.

6. Confirm `git status --porcelain` is empty.

## Definition of Done

- [ ] `.gitignore` rewritten; both required comments present; old
      `checkpoints/*/metrics.jsonl` negations removed
- [ ] `.gitattributes` created with `* text=auto eol=lf`
- [ ] Both metrics files are at `docs/archive/metrics/<mode>-2026-07.jsonl`, with
      before/after byte counts pasted and identical
- [ ] `docs/archive/metrics/README.md` explains what they are and why they moved
- [ ] `git ls-files | grep -E '(checkpoints|test_outputs|probe_results)/'`
      returns nothing — output pasted
- [ ] `git ls-files | grep verify_output` still returns **both** files — output
      pasted
- [ ] `.pre-commit-config.yaml` created; `pre-commit run --all-files` passes —
      output pasted
- [ ] The formatting pass is a separate commit from every other change
- [ ] `git status --porcelain` is empty — output pasted
- [ ] Result file written from `results/_TEMPLATE.md`

There is no `scripts/check.py` yet — A3 builds it.

## Out of scope

- Deleting `tools/`, `notebooks/` or the two empty `.py` files (C3).
- Archiving `docs/verify_output.txt` (F1).
- Type-checker configuration (A3).
- Any `.py` edit whatsoever, including the ones ruff wants to make beyond
  formatting.
- Deleting any tag or branch. `cleanup-2026-09.md` §8 lists refs that must
  survive, including `origin/archive/pre-antigravity-baseline`.
