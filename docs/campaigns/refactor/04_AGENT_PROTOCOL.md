# Agent protocol

How to run a task in this campaign. **Binding.**

---

## 1. The loop

1. Read `AGENTS.md`, then `00_CONTEXT.md` → `03_DOMAIN_PRIORS.md` in this
   directory, then **every file already in `results/`**.
2. Read **your** task file. Only yours.
3. Branch (§8). Work. Commit at internal milestones.
4. Run the gate: `uv run python scripts/check.py`.
5. Write `results/<TASK_ID>_result.md` from `results/_TEMPLATE.md`.
6. Push the branch. **Stop.** Do not merge.

Tasks `A1`–`A3` are the exception to step 4: they are the tasks that *build* the
gate. Each says explicitly what to run instead.

---

## 2. Scope discipline

Do exactly your task. Your task file has a **Touches** and a **Must not touch**
list; they are binding. If you notice something wrong outside your scope, write
it under **Observations** in your result file and move on.

A task that fixes four unrelated things cannot be reviewed and cannot be
reverted without losing the one thing that mattered. That is not a hypothetical
here: this repo has already spent nine tasks unwinding a state where changes had
been hand-carried between branches until nobody could tell what was authoritative.

**Code that carries an explanatory comment is load-bearing.** This repo is
unusually well commented, and the comments record failures that cost real time on
a machine with ~50-hour queue latency. Before changing commented code, read the
comment. If you still believe it should change and it is outside your task, that
is an Observation, not an edit.

Specific things that look like cruft and are not:

- `gradient_checkpointing: false` everywhere — Lesson 6.
- The per-variable index maps in `dataset.py` — Lesson 5, `02_…` §4.2.
- The `msl` norm-stat override — Lesson 1, `02_…` §4.1.
- `-c 64` and the `USR1` trap in `train_auto.slurm` — `00_CONTEXT.md` §2.3.
- The DDP-collective grad guard in `trainer.py` — Lesson 2.
- `slt` being truncated rather than upsampled — `03_…` §3.

---

## 3. Code standards

Match what is already there.

- Python 3.10 as pinned in `.python-version`. `from __future__ import
  annotations` in new modules.
- Type hints on **new and moved** code in `src/aurora_mjo/`. The type gate is a
  ratchet, not a cliff (`01_TARGET_STATE.md` D5) — do not add `Any` to silence
  it, and do not "fix" `scripts/` types unless your task says to.
- `ruff` for lint and format, line length 100.
- **`src/aurora_mjo/` is imported and tested, never run. `scripts/` is run,
  never imported.** If two scripts need the same logic, it belongs in the
  package.
- Docstrings and comments carry the **why**, including the incident behind a
  non-obvious choice and any hypothesis that was refuted. The existing
  `dataset.py` header is the house style — imitate it.
- **Tests are not optional** for new logic in `src/aurora_mjo/`. No test may
  require CFS data, a GPU, or a network on the default path — build synthetic
  fixtures, or mark it `needs_data` / `needs_gpu` / `live`. If your new code
  cannot be tested without real ERA5 data, the logic is in the wrong layer.
- No secrets, ever.

### The one physical-safety rule

**Never write to `/global/cfs/cdirs/m4946/`.** It is another group's allocation
and this repo attaches it read-only, conceptually and in practice. Also: never
delete or move anything under `checkpoints/`, and never kill a running training
process. See `02_UPSTREAM_CONTRACT.md` §1.

---

## 4. The gate

```bash
uv run python scripts/check.py
```

A task is not done with a red gate. **Paste the summary table into your result
file.**

If a pre-existing failure is unrelated to your change, say so explicitly **with
the failure output**. Do not delete the test, do not `@pytest.mark.skip` it, and
do not widen an `except` to make it pass.

---

## 5. Result files

One per task at `results/<TASK_ID>_result.md`, built from `results/_TEMPLATE.md`.

**The first four lines are machine-read by the next agent. Get them exactly
right.**

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

**Write results as you go, not at the end.** A task that dies at minute 50 of a
60-minute budget with nothing written has destroyed all of its own value.

What makes a good result file:

- **Numbers, not adjectives.** "1,462 samples for 1980; formula gives 1,464 − 2 =
  1,462" beats "sample count looks right".
- **What was ruled out, and by what evidence.** A hypothesis eliminated is worth
  as much as one confirmed.
- **Admit when an earlier hypothesis was wrong**, plainly.
- **Distinguish observation from inference.** Label guesses as guesses.
- **Paste actual command output** for every Definition-of-Done item that has one.

---

## 6. Asking the human

Append to `QUESTIONS.md` as the next free `Q-nn`. State what it blocks and
**propose a default so work is not blocked while you wait.** Then carry on with
anything that does not depend on the answer.

Do not guess past an unknown. Do not stall on one either.

Q-01 through Q-08 are pre-seeded with proposed defaults. **If your task depends
on one that is still unanswered, proceed on the stated default and say so in your
result file** — do not wait, and do not silently pick something else.

---

## 7. Never negotiable

Report the truth about the gate. Everything else in this campaign is
recoverable; a false green is not, because it removes the human's ability to
trust any other report you make.

Also never acceptable:

- Deleting or skipping a failing test to get green.
- Widening an `except` to swallow an error you do not understand. (This repo
  already has one of those and fixing it is task E1.)
- Committing a `.pt`, a `.nc` over ~1 MB, a credential, or anything from a
  SLURM log directory.
- Writing to the CFS archive.
- Reporting a Definition-of-Done item as met when it is not literally true.
- Merging your own branch.
- Flipping `gradient_checkpointing` to `true`.
- Removing the `msl` norm-stat override.

---

## 8. Git hygiene

```bash
git fetch origin
git checkout epic/refactor
git pull --ff-only origin epic/refactor
git checkout -b epic/refactor/<TASK_ID>-<short-slug>
```

- **One branch per task**, always cut from an up-to-date `epic/refactor`.
- **Nothing in this campaign touches `main`.**
- **No git worktrees.** The worktree-per-agent model in `docs/git-policy.md` is
  what produced the state the September consolidation had to unwind. One clone,
  one branch per task, sequential.
- **Commit at each internal milestone, not once at the end.** The message says
  **why**:

  ```text
  C1: src/ becomes src/aurora_mjo/ so it cannot shadow microsoft-aurora

  microsoft-aurora owns the top-level `aurora` import name and is imported in
  six first-party files. A first-party package called `aurora` would shadow it
  and surface as ImportError on Batch. 27 `from src.…` imports re-pointed.
  ```

- **`uv.lock` is committed.** If you change `pyproject.toml`, run `uv sync` and
  commit the lockfile **in the same commit**.
- **Pre-commit must pass.** Do not use `--no-verify`.
- **Do not merge.** Push and stop. The human reviews the result file and merges
  into `epic/refactor`. `epic/refactor` merges to `main` once, at the end, by
  the human.
- **If the base moved under you, rebase — do not merge** — then re-run the gate.
- **Never `git add -f`** around a gitignore rule.

### Recovery points you must not touch

`cleanup-2026-09.md` §8 lists backup tags, `wip/snapshot/*` tags, and
`origin/archive/pre-antigravity-baseline` (remote-only, never merged, **marked
do not delete**). Do not delete, move, or overwrite any ref in that table.

---

## 9. If you are working on Perlmutter

Applies to tasks marked as needing the cluster (B1, B2, F2, and any
`needs_data` test run).

- **Confirm which machine you are on before doing anything.** `hostname`.
- **Show the full command and get approval before submitting any SLURM job.**
  GPU time on this allocation is scarce and the batch queue has been observed at
  ~50-hour latency.
- **Never `sbatch` a training job as part of a task in this campaign** unless
  the task file says so explicitly. F2 is the only one that does.
- `export HDF5_USE_FILE_LOCKING=FALSE` before touching any NetCDF file, until
  E1 lands and makes that automatic.
- Nothing durable lives on `/pscratch` — it is purgeable. Evidence goes in
  `results/`, in the repo.
- Do not create git worktrees on `/pscratch`. See §8.