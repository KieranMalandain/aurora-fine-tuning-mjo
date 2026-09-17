# Agent protocol

How to run a task in this campaign. **Binding.**

Most of this is carried forward unchanged from `../refactor/04_AGENT_PROTOCOL.md`,
which worked. §4, §5 and §10 are new and exist because this campaign does things
the last one did not: it changes scientific behaviour on purpose, it touches the
validation split, and it submits real training jobs.

---

## 1. The loop

1. Read `AGENTS.md`, then `00_CONTEXT.md` → `04_AGENT_PROTOCOL.md` in this
   directory, then **every file already in `results/`**.
2. Read **your** task file. Only yours.
3. Branch (§9). Work. Commit at internal milestones.
4. Run the gate: `uv run python scripts/check.py`.
5. Write `results/<TASK_ID>_result.md` from `results/_TEMPLATE.md`.
6. Push the branch. **Stop.** Do not merge.

`02_SCIENTIFIC_CONTRACT.md` is not optional reading. It is where every settled
scientific decision lives, and a task that re-opens one has failed regardless of
whether the code is good.

---

## 2. Scope discipline

Do exactly your task. **Touches** and **Must not touch** are binding. Anything
you notice outside your scope goes under **Observations** in your result file.

**Code that carries an explanatory comment is load-bearing.** The comments in
this repo record failures that cost real time on a machine with ~50-hour queue
latency. Read the comment before changing the code.

Things that look like cruft and are not:

- `gradient_checkpointing: false` — Lesson 6. **Only G2 may change it**, and only
  with the IMA matrix probe behind it.
- The per-variable index maps in `dataset.py` — Lesson 5. G1 changes what is
  read, **not** `_build_aligned_index`.
- The DDP-collective grad guard in `trainer.py` — Lesson 2.
- `StaticVarLoadError` instead of a zero fallback — Lesson 4.
- The `USR1` trap and `-c 64` in `train_auto.slurm`.
- `_ROLLOUT_CLAMP` in `trainer.py` — it is Finding 3's fix and it already works.
  H4 may **replace** it with Aurora's native `positive_*_vars`; nothing else
  touches it.

Things that look load-bearing and are **not**, because this campaign is
deliberately changing them — do not defend them:

- The `msl` norm-stat override. It is a documented PLACEHOLDER. **G3 replaces
  it.** (`../refactor/04_AGENT_PROTOCOL.md` §7 forbade removing it; that applied
  to that campaign.)
- `_upsample_to_aurora` and `_upsample_batch_gpu`. **G1 deletes both.**
- `mean(|R|)` in `MoistureBudgetLoss`. **H3 replaces it.**
- The 3-element combined vector in `rmm/`. **J1 replaces it.**

---

## 3. Code standards

Unchanged from `../refactor/04_AGENT_PROTOCOL.md` §3. Summarised:

- Python 3.10, `from __future__ import annotations` in new modules, type hints on
  new code, `ruff` lint and format at line length 100.
- **`src/aurora_mjo/` is imported and tested, never run. `scripts/` is run, never
  imported.** If two scripts need the same logic, it belongs in the package.
- Docstrings and comments carry the **why**, including refuted hypotheses. The
  `dataset.py` header is the house style.
- **Tests are not optional** for new logic in `src/aurora_mjo/`. Nothing on the
  default path may require CFS, a GPU or a network — use synthetic fixtures or
  mark `needs_data` / `needs_gpu` / `slow` / `live`.
- No secrets, ever.

### The physical-safety rules

- **Never write to `/global/cfs/cdirs/m4946/`.** Another group's allocation,
  attached read-only. `../refactor/02_UPSTREAM_CONTRACT.md` §1.
- Never delete or move anything under `checkpoints/`.
- Never kill a running training process.

---

## 4. The test-year quarantine — new, and absolute

**2020–2023 is not read by anything in this campaign.** Not by a diagnostic, not
by a sanity plot, not for one sample, not "just to check the loader works".

This is stricter than a normal train/test discipline and it is deliberate. The
project's own target metric is a correlation on a heavily projected index over a
small number of MJO events; the opportunity to fool yourself is large, the
literature has a retraction in it for exactly this class of error, and this
campaign's whole output is a baseline that a later campaign will compare against.
A baseline contaminated by a glance at the test set is worth less than no
baseline.

Practical consequences:

- Every number in every result file comes from **2016–2019** (validation) or
  **1980–2015** (training).
- `J1` writes `data/rmm_targets.nc` covering **1980–2019 only**.
- Any config with `test_years` in an active code path is a defect. The key stays
  in `configs/unified.yaml` for the later campaign; nothing reads it.
- If a task genuinely cannot proceed without test data, that is a `Q-nn` and a
  `BLOCKED` status, not a judgement call.

The 120-day-mean convention in `02_SCIENTIFIC_CONTRACT.md` §6.2 is part of this
rule. Using **any** data after a forecast valid time invalidates the run.

---

## 5. Recording a discontinuity — new

The `refactor` campaign's job was to change nothing. This campaign's job is to
change specific things on purpose. Several measurements will move, and a moved
number that is not explained looks exactly like a regression.

**If your task changes a number that a previous campaign measured, your result
file must contain a section with three things:** the old value and its source,
the new value, and the one-sentence reason the change is intended.

The numbers known in advance to move:

| Quantity | Old | Source | Moves in |
| --- | --- | --- | --- |
| Train loss, step 1, smoke | 7472.024414 | `B1`, `C4`, `F2` | G1, H1 |
| Val loss, step 1, smoke | 7421.924805 | as above | G1, H1 |
| Total parameters, production | 112,830,384 | `B1` | G2 |
| Grid fed to Aurora | 720 × 1440 | `../refactor/03_DOMAIN_PRIORS.md` §2 | G1 |
| `msl` norm stats | 96,667.98 / 9,504.64 | `configs/unified.yaml` | G3 |
| Peak memory, step time | 39.71 GiB / 0.694 s | `../refactor/03_DOMAIN_PRIORS.md` §7 | G1, G2 |

Sample counts (`03_DOMAIN_PRIORS.md` §1) are **not** on this list. They must come
out identical after G1. If they move, that is a regression in Lesson 5.

`tests/fixtures/baseline/` and `tests/fixtures/postrefactor/` stay byte-identical.
G4 writes a new set alongside them.

---

## 6. The gate

```bash
uv run python scripts/check.py
```

A task is not done with a red gate. **Paste the summary table into your result
file.**

A pre-existing failure unrelated to your change must be stated explicitly **with
the failure output**. Do not delete the test, do not `@pytest.mark.skip` it, and
do not widen an `except` to make it pass.

Two tests are expected to change state in this campaign, and only in the task
named:

- `tests/test_norm_stats.py::test_placeholder_stats_warning_or_failure` inverts
  from `xfail` to pass in **G3**.
- Any test asserting 720 × 1440 shapes is updated in **G1**.

Changing either outside those tasks is a defect.

---

## 7. Result files

One per task at `results/<TASK_ID>_result.md`, from `results/_TEMPLATE.md`. **The
first four lines are machine-read. Get them exactly right.**

```markdown
STATUS: GREEN | AMBER | RED | BLOCKED
PROCEED: YES | YES-WITH-CAVEATS | NO
BLOCKED-ON: <question ID, or NONE>
SUMMARY: <one sentence, under 120 characters>
```

| Status | Meaning |
| --- | --- |
| `GREEN` | Every Definition of Done item is literally true. |
| `AMBER` | Main objective achieved; something incomplete or uncertain. **Caveats must be specific**: what is unknown, what it affects, what would resolve it. |
| `RED` | Objective not achieved. State root cause, what was ruled out, and the concrete next step. |
| `BLOCKED` | Could not start. Precondition unmet or question outstanding. |

**Write results as you go.** A task that dies at minute 50 of a 60-minute budget
with nothing written has destroyed all of its own value.

- **Numbers, not adjectives.**
- **What was ruled out, and by what evidence.**
- **Admit when an earlier hypothesis was wrong**, plainly.
- **Distinguish observation from inference.** Use MEASURED / INFERRED /
  RECOMMENDED as `../refactor/` did.
- **Paste actual command output** for every Definition-of-Done item with one.
- **Include the discontinuity section (§5) if your task moves a known number.**

---

## 8. Asking the human

Append to `QUESTIONS.md` as the next free `Q-nn`. State what it blocks and
**propose a default** so work is not blocked while you wait. Then carry on with
anything that does not depend on the answer.

Q-12 through Q-18 are pre-seeded with defaults. **If your task depends on one
still unanswered, proceed on the stated default and say so in your result file.**
Do not wait; do not silently pick something else.

**The exception: a question that would change a scientific result.** If
proceeding on a default would produce a number that goes in the paper, stop and
mark `BLOCKED`. Q-13 (checkpoint choice) and Q-15 (BoM reference provenance) are
both of this kind.

---

## 9. Git hygiene

```bash
git fetch origin
git checkout epic/science-baseline
git pull --ff-only origin epic/science-baseline
git checkout -b epic/science-<TASK_ID>-<short-slug>
```

- **One branch per task**, cut from an up-to-date `epic/science-baseline`.
- Branch naming is `epic/science-<ID>-<slug>` — flat, not nested. Git ref storage
  rejects `epic/science-baseline/<id>` because `epic/science-baseline` exists as
  a file (`../refactor/handoff-2026-09-refactor.md` §7.2).
- **Nothing in this campaign touches `main`.**
- **No git worktrees.**
- **Commit at each internal milestone**, with a message that says **why**.
- **`uv.lock` is committed.** Change `pyproject.toml` → run `uv sync` → commit
  the lockfile in the same commit.
- **Pre-commit must pass.** No `--no-verify`.
- **Do not merge.** Push and stop.
- **If the base moved under you, rebase — do not merge** — then re-run the gate.
- **Never `git add -f`** around a gitignore rule.
- **Never commit** a `.pt`, a `.nc` over ~1 MB, a credential, or anything from a
  SLURM log directory.

Do not delete, move or overwrite any ref in the recovery table at
`../refactor/handoff-2026-09-refactor.md` §11.

---

## 10. Perlmutter and compute discipline — new, and the part most likely to burn a week

This campaign submits real jobs. `refactor` submitted one.

- **Confirm which machine you are on before anything.** `hostname`.
- **Show the full command and get human approval before any `sbatch`.** No
  exceptions, including for a job you are re-submitting after a failure.
- **Every task states its compute budget in GPU-hours** in its header table. If
  you are about to exceed it, stop and report — do not spend into the next task's
  budget.
- **Smoke first, always.** No multi-hour job is submitted before the same config
  has run under `--smoke-test` on `model_type: small` and produced a finite loss.
  This is how the 24-hour submit-wait-diagnose loop is avoided, and it is the
  single biggest cost sink the previous campaign identified
  (`../refactor/handoff-2026-09-refactor.md` §7.1).
- **One variable per job.** A job that changes the loss *and* the resolution
  cannot be diagnosed. The phase ordering exists for this reason.
- **Log to the repo, not to scratch.** `/pscratch` is purgeable and the previous
  campaign lost its July logs to a purge. Every job redirects stdout and stderr
  to a path that gets summarised into `results/`.
- `export HDF5_USE_FILE_LOCKING=FALSE` is automatic since E1, but check it is in
  the environment of anything running outside `run.py`.
- Export `UV_CACHE_DIR`, `PRE_COMMIT_HOME` and `HF_HOME` to `/pscratch` — Lustre
  `$HOME` rejects `flock` with error 524.

---

## 11. Never negotiable

Report the truth about the gate. Everything else is recoverable; a false green is
not, because it removes the human's ability to trust any other report you make.

Also never acceptable:

- **Reading the test years** (§4).
- **Using data after a forecast valid time** in an RMM computation
  (`02_SCIENTIFIC_CONTRACT.md` §6.2).
- **Tuning `w_v`, λ_p, or any hyperparameter against validation skill and then
  reporting that validation skill** (`02_SCIENTIFIC_CONTRACT.md` §6.3).
- **Reporting a result from `model_type: small`** as anything other than a smoke
  test (`02_SCIENTIFIC_CONTRACT.md` §2.4).
- **Citing a literature skill number** without recording ensemble-vs-deterministic,
  member count, verification period, and amplitude conditioning
  (`02_SCIENTIFIC_CONTRACT.md` §1.3).
- Deleting or skipping a failing test to get green.
- Widening an `except` to swallow an error you do not understand.
- Writing to the CFS archive.
- Reporting a Definition-of-Done item as met when it is not literally true.
- Merging your own branch.
- `sbatch` without approval.
