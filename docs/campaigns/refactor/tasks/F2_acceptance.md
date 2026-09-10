# F2 — Acceptance on Perlmutter: smoke test end-to-end under the new CLI

| | |
| --- | --- |
| **Phase** | F |
| **Depends on** | F1 (`PROCEED: YES`), and every prior task |
| **Base branch** | `epic/refactor` |
| **Budget** | 3 h, dominated by queue wait |
| **Touches** | `docs/PROJECT_STATE.md` (status table only), `results/F2_result.md` |
| **Must not touch** | anything else. **This task is acceptance, not repair.** |
| **Needs** | Perlmutter, one GPU node, CFS read access, **human approval before any `sbatch`** |

## Objective

Prove the refactored repository works on the machine it exists to run on:
clone → `uv sync` → gate green → smoke test on a GPU node → real-data dataset
construction → a short SLURM submission that completes.

## Why this is a separate task

Every gate so far has been offline or single-purpose. This is the only test of
the whole chain in the environment that matters. A campaign that ends without it
has produced a repository that is tidy and unproven.

## What you may assume

- Everything in Phases A–F1 has landed on `epic/refactor` with `PROCEED: YES`.
- **`04_AGENT_PROTOCOL.md` §9 applies in full**: confirm the machine, show the
  full command and get approval before submitting any job, never kill a running
  process, GPU time is scarce and the batch queue has been observed at ~50-hour
  latency.
- `03_DOMAIN_PRIORS.md` holds the expected values. Every measurement below has a
  target: 1,462 samples for 1980; `z` 3709.2466, `lsm` 0.3357, `slt` 0.6708;
  41,008 trainable for `small`; ~0.69 s/step; ~39.7 GiB peak.
- **B1's fingerprint is the reference for behaviour.** If a measurement here
  differs from B1, that is the finding — the refactor was supposed to change
  nothing.
- A1 and E3 may have reported `AMBER` on `uv` on Perlmutter (Q-03). **Read both
  result files first.** If `uv` does not work there, this task documents what
  actually works and reports `AMBER` — that is a legitimate outcome, not a
  failure of the campaign.

## Steps

1. **Fresh clone.** Not the working copy — a clone into a new directory on
   `/pscratch`, checked out to `epic/refactor`. The point is to test what a new
   contributor gets. Record the commit SHA.

2. **Install.** `uv sync --all-groups`. Record wall-clock time and any warning.
   If it fails, capture the full output, record it, and report `AMBER` or `RED`
   per Q-03's fallback rather than improvising a fix.

3. **The gate, on the cluster.**

   ```bash
   uv run python scripts/check.py
   ```

   Paste the summary table. This is the first time it has run on Perlmutter
   rather than a dev machine; a path or encoding assumption would surface here.

4. **Data-dependent tests.** These have never run — CI excludes them by design.

   ```bash
   uv run pytest -q -m "needs_data" -v
   ```

   Report the count and every failure in full. This is where D2's and D3's
   real-data assertions meet real data for the first time, so **a failure here
   is valuable information, not a setback** — it means a synthetic fixture
   diverges from reality, and that is precisely what this run exists to find.

5. **Real dataset construction, one year.**

   ```bash
   uv run python scripts/verify_dataset_loader.py
   ```

   Compare against B1's `dataset_1980.json` and against
   `03_DOMAIN_PRIORS.md` §§1–3:

   - `len(dataset)` == **1462**
   - `z` mean **3709.2466**, `lsm` mean **0.3357**, `slt` mean **0.6708**
   - the per-variable alignment report matches B1 modulo paths and timestamps

   **Confirm E1's hard failure did not fire** — statics loaded, no
   `StaticVarLoadError`. If it did fire, E1's message should tell you exactly
   what to do; record whether it did, because the quality of that message is
   itself an acceptance criterion.

6. **GPU-dependent tests**, on a GPU node via `salloc`:

   ```bash
   uv run pytest -q -m "needs_gpu" -v
   ```

   Report counts and failures.

7. **Smoke test, both entry points**, on the GPU node:

   ```bash
   uv run python run.py train --mode baseline --smoke-test
   uv run torchrun --nproc_per_node=1 train.py \
     --config configs/unified.yaml --mode baseline --smoke-test
   ```

   Compare the loss trajectory against B1's `smoke_baseline.json` at B1's stated
   tolerance. Report max absolute and max relative deviation as numbers.

   **Report the non-finite-gradient skip ratio.** Per
   `03_DOMAIN_PRIORS.md` §7, a previous probe recorded 30 skips in 30 steps —
   every step skipped, the run learning nothing. If that reproduces, **it is the
   correct outcome for this campaign** (the refactor preserved behaviour) and
   simultaneously **the single most important thing in your report**, because it
   means the underlying forward pass is still broken and the next campaign's
   first job is the `msl` normalisation statistics. State both facts plainly and
   put the ratio in your `SUMMARY`.

8. **SLURM submission.** `test_train.slurm` is now the cheapest cluster check
   (E3 pointed it at the smoke test with a short walltime).

   **Show the full command and get human approval before submitting.** Then:

   ```bash
   sbatch slurm_scripts/test_train.slurm
   ```

   Record: job ID, queue wait, run time, exit code, and the full log. Verify
   `env.sh` was sourced (the environment banner), that `uv run --frozen` did not
   re-resolve, and that the exit-code contract held.

   **Do not submit `train_auto.slurm`.** An 11.5-hour training job is not an
   acceptance test and this campaign does not produce scientific results.

9. **Config resolution on the cluster**, since E2 changed the boundary:

   ```bash
   for m in baseline physics_informed lora combined; do
     uv run python run.py show-config --mode $m > /tmp/f2_$m.json
     diff tests/fixtures/baseline/config_$m.json /tmp/f2_$m.json && echo "$m OK"
   done
   ```

   Four empty diffs, unless E2 documented a specific reason byte-identity is
   impossible — in which case use E2's semantic comparison and say so.

10. Update **only** the status table in `docs/PROJECT_STATE.md` with what you
    measured, and set **Last updated**. Do not rewrite F1's prose. If your
    findings change what the **Next action** should be, say so in your result
    file and let the human decide — do not overwrite it.

## Definition of Done

- [ ] Fresh clone on `/pscratch` at `epic/refactor`; commit SHA recorded
- [ ] `uv sync --all-groups` result and wall-clock recorded — or the Q-03
      fallback documented with full output
- [ ] `scripts/check.py` run **on Perlmutter**; summary table pasted
- [ ] `needs_data` tests run; count and every failure reported in full
- [ ] `len(dataset)` == 1462 confirmed against B1 and the priors
- [ ] Static means match: `z` 3709.2466, `lsm` 0.3357, `slt` 0.6708 — table
      pasted
- [ ] Alignment report compared against B1; differences explained
- [ ] E1's `StaticVarLoadError` did not fire — **or** it did and the message
      quality is assessed
- [ ] `needs_gpu` tests run; counts and failures reported
- [ ] Smoke test run through **both** entry points on a GPU node
- [ ] Loss deviation vs B1 reported as max absolute and max relative numbers
- [ ] **Non-finite-grad skip ratio reported and in `SUMMARY`**, with its
      implication for the next campaign stated
- [ ] Human approval obtained before `sbatch` — stated explicitly
- [ ] `test_train.slurm` submitted; job ID, queue wait, run time, exit code and
      log recorded; `env.sh` sourcing and `--frozen` behaviour confirmed
- [ ] `train_auto.slurm` **not** submitted — stated explicitly
- [ ] Four config diffs against B1 pasted
- [ ] `docs/PROJECT_STATE.md` status table and date updated; prose untouched
- [ ] A **campaign verdict**: is `epic/refactor` ready to merge to `main`? If
      not, exactly what blocks it
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Fixing anything.** If a measurement fails, record it and report. This is the
  acceptance gate; a failure here is information for the human, and repairing it
  inside the acceptance run destroys the evidence.
- Submitting `train_auto.slurm` or any multi-hour job.
- Any training run whose purpose is a scientific result.
- Computing real `ps` normalisation statistics — even though this run will very
  likely demonstrate exactly why they are needed. Record it as the next
  campaign's first task.
- Merging `epic/refactor` to `main`. The human does that, once, after reading
  your verdict.
- Rewriting `docs/PROJECT_STATE.md` beyond the status table.
