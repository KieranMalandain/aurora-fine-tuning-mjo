# E3 — Rewrite `eval.slurm`; make every SLURM script consistent

| | |
| --- | --- |
| **Phase** | E |
| **Depends on** | E1, C2 (both `PROCEED: YES`) |
| **Base branch** | `epic/refactor` |
| **Budget** | 2 h |
| **Touches** | `slurm_scripts/eval.slurm`, `slurm_scripts/test_train.slurm`, `slurm_scripts/submit_chain.sh`, `slurm_scripts/train_auto.slurm` (**environment block only**), `slurm_scripts/README.md` (new) |
| **Must not touch** | `train_auto.slurm`'s SBATCH directives, its `USR1` trap, its `DONE`/`STOP_TRAINING` logic, or its exit-code contract. Anything under `src/`. `configs/unified.yaml`. |

## Objective

At the end, all four SLURM scripts run under `uv` instead of a hardcoded conda
path, set the same environment, invoke `run.py`, and reference configs that
exist. `eval.slurm` becomes functional for the first time.

## Why this is a separate task

It needs E1 (so the environment variable story is settled) and C2 (so `run.py`
exists), and it is the only task that touches the cluster's entry surface. It is
also the task with the highest blast radius per line: a broken SLURM script
costs an 11-hour slot plus a queue wait to discover.

## What you may assume

- `run.py train --mode <m>` and `run.py show-config --mode <m>` exist (C2).
  `run.py evaluate` exists or fails with a message naming what to run instead
  (C3) — **check which**, and if it is not wired, invoke the script directly and
  say so.
- `train.py` is a working deprecation shim (C2), so `train_auto.slurm` does not
  strictly need re-pointing. **Re-point it anyway** — the shim exists for jobs
  already in the queue, not as a permanent path.
- `aurora_mjo` sets `HDF5_USE_FILE_LOCKING` at process start (E1). Keep it in
  the shell scripts too: a job that fails before Python starts should still be
  correct, and the SLURM env is also inherited by any tool the script runs
  before Python.
- **`train_auto.slurm` is careful, deliberate work.** Read
  `00_CONTEXT.md` §2.3 before touching it. The `-t 11:30:00` against an
  in-process 11.0 h limit, `--signal=B:USR1@1800`, the `scancel --signal=USR1
  "${SLURM_JOB_ID}.0"` forwarding, the `DONE` marker, the `STOP_TRAINING` kill
  switch and `-c 64` all encode observed failures. You are changing its
  **environment block and its python invocation only.**

`eval.slurm` is broken four ways (`00_CONTEXT.md` §2.4 item 2):

1. references deleted `configs/phase1_baseline.yaml`
2. passes `--override data.use_dummy=true`, which C3 made a loud `ValueError`
   and E2 made a config-validation error
3. hardcodes `/pscratch/sd/k/kam352/conda/envs/aurora_mjo/bin/python` — wrong
   path; the real one is `conda_envs`, not `conda/envs`
4. it is 25 lines and is the only copy anywhere, retained as a draft

## Steps

1. Read all four scripts and `submit_chain.sh` end to end first. Produce a table
   in your result file: script, what it does, what is wrong with it, what you
   changed. `submit_chain.sh` chains dependent jobs — understand the dependency
   flags before editing.

2. **Extract the shared environment into one sourced file**,
   `slurm_scripts/env.sh`, holding exactly what `train_auto.slurm` currently
   sets and nothing new:

   ```bash
   export HDF5_USE_FILE_LOCKING=FALSE
   export HF_HOME=${PSCRATCH}/hf_home        # PERSISTENT; was /tmp → re-download every job
   export HF_HUB_ETAG_TIMEOUT=60
   export PYTHONUNBUFFERED=1
   export OMP_NUM_THREADS=8
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # anti-fragmentation
   ```

   **Carry the existing comments across verbatim.** Each records a real failure.
   Do not add anything to this file that is not already set somewhere — this is
   consolidation, not tuning.

   `MASTER_ADDR` and `MASTER_PORT` stay in `train_auto.slurm`: the port
   derivation `29000 + SLURM_JOB_ID % 1000` avoids collisions between chained
   jobs and is training-specific.

3. **Replace conda activation with `uv`** in all four scripts. Depending on
   Q-03's outcome that is either:

   ```bash
   cd "${SLURM_SUBMIT_DIR}"
   uv run --frozen python run.py ...
   ```

   or, if A1 reported that `uv` cannot work on Perlmutter, the documented
   fallback from Q-03. **Read A1's result file before writing this.** If A1 was
   `AMBER` on installability, keep conda activation but add a comment naming
   Q-03 and the intended end state, and report `AMBER` yourself. Do not guess.

   `--frozen` matters: it fails rather than silently re-resolving the lockfile
   inside a batch job, which is exactly where you do not want dependency
   resolution happening.

4. **Rewrite `eval.slurm` against `unified.yaml`.** Requirements:

   - `MODE` environment variable defaulting to `baseline`, matching
     `train_auto.slurm`'s convention
   - `configs/unified.yaml --mode "${MODE}"`, never a deleted per-phase config
   - **no `--override data.use_dummy=true`** — that path is gone and now fails
     loudly by design
   - source `env.sh`
   - SBATCH directives appropriate to evaluation, not training: shorter
     walltime, and fewer GPUs if evaluation is single-device. **State your
     reasoning** for whatever you choose; if you cannot determine the right
     resources without running it, use conservative values, say they are
     unverified, and note it for F2.
   - output/error paths under `slurm_logs/`, consistent with the other scripts

5. Fix `test_train.slurm` the same way. It also references deleted configs.
   Its purpose is a short smoke run, so point it at
   `run.py train --mode baseline --smoke-test` and give it a short walltime.
   That makes it the cheapest possible cluster check and F2 will use it.

6. Update `submit_chain.sh` for the new invocations. **Preserve the dependency
   semantics exactly** — if it uses `--dependency=afterok:` or similar, the
   chain's correctness depends on it, and the `DONE`-marker no-op behaviour
   means an over-provisioned chain is harmless only if the exit-code contract
   holds.

7. Update `train_auto.slurm`: source `env.sh`, swap conda for `uv`, change
   `train.py` to `run.py train`. **Nothing else.** Paste a `git diff` of this
   file in your result file so a reviewer can confirm the trap logic and SBATCH
   block are untouched.

8. Write `slurm_scripts/README.md`: one section per script — what it is for, how
   to submit it, what environment variables it honours, and the exit-code
   contract (`0` = phase complete and `DONE` written; `99` = clean timeout save,
   resume via `--resume auto`; anything else = crash, chain retry is safe). Also
   document `STOP_TRAINING` as the chain kill switch. This contract currently
   exists only as a comment inside one script.

9. Verify what you can **without submitting a job**:

   ```bash
   bash -n slurm_scripts/*.slurm slurm_scripts/*.sh   # syntax only
   grep -rn "phase1_baseline\|phase2_physics\|phase3_longrun\|use_dummy\|conda" slurm_scripts/
   grep -rn "conda/envs" slurm_scripts/
   ```

   The greps must return nothing (except intentional mentions in `README.md`
   explaining the migration). Paste them.

   **Do not `sbatch` anything.** F2 owns the first real submission, and
   `04_AGENT_PROTOCOL.md` §9 requires approval before any job submission.

## Definition of Done

- [ ] Table of all four scripts plus `submit_chain.sh`: purpose, what was
      wrong, what changed
- [ ] `slurm_scripts/env.sh` created with the six variables and their original
      comments carried across verbatim; nothing new added
- [ ] `MASTER_ADDR`/`MASTER_PORT` left in `train_auto.slurm`
- [ ] All four scripts run Python via `uv run --frozen` — **or** A1's `AMBER`
      fallback is used, with Q-03 referenced and the task reported `AMBER`
- [ ] `eval.slurm` rewritten: `MODE` variable, `unified.yaml --mode`, no
      `use_dummy`, sources `env.sh`, correct log paths; resource choices
      justified or flagged unverified
- [ ] `test_train.slurm` points at `run.py train --mode baseline --smoke-test`
      with a short walltime
- [ ] `submit_chain.sh` updated with dependency semantics preserved and stated
- [ ] `train_auto.slurm` diff pasted, showing **only** env-block and invocation
      changes; SBATCH directives, `USR1` trap, `DONE`/`STOP_TRAINING` and exit
      codes untouched
- [ ] `slurm_scripts/README.md` documents each script and the full exit-code
      contract including `STOP_TRAINING`
- [ ] `bash -n` clean on all five files — output pasted
- [ ] Both greps return nothing outside `README.md` — output pasted
- [ ] **No job submitted** — stated explicitly
- [ ] `uv run python scripts/check.py` green — summary table pasted
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Submitting any SLURM job.** F2 only, with approval.
- Changing `train_auto.slurm`'s walltime, signal handling, CPU/GPU request,
  `DONE` logic or exit codes. Every one encodes an observed failure.
- Tuning `OMP_NUM_THREADS`, `num_workers` or `PYTORCH_CUDA_ALLOC_CONF`. Consolidate
  the existing values; do not optimise them. `-c 64` exists because an earlier
  version requested 8 CPUs for 16 workers and thrashed.
- Deleting `train.py` — the shim stays this campaign (Q-04).
- Any edit under `src/` or to `configs/unified.yaml`.
- Writing new SLURM scripts for modes that do not have one. If `physics_informed`
  or `combined` lack coverage, that is an Observation.
