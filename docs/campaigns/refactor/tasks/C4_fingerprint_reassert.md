# C4 — **THE GATE.** Re-assert the B1 fingerprint

| | |
| --- | --- |
| **Phase** | C |
| **Depends on** | C1, C2, C3 — all with `PROCEED: YES` |
| **Base branch** | `epic/refactor` |
| **Budget** | 90 min |
| **Touches** | `tests/fixtures/postrefactor/*.json` (new), `results/C4_result.md` |
| **Must not touch** | **anything else.** In particular: not `tests/fixtures/baseline/` — the B1 fixtures are the reference and are read-only. Not `src/`, not `run.py`, not `configs/`. |
| **Needs** | the same environment B1 used, plus a GPU node for step 4 |

## Objective

Prove that C1, C2 and C3 changed nothing observable. Produce the same
measurements B1 produced, diff them, and report the result honestly.

## Why this is a separate task — and why it is the gate

C1 moved every module and re-pointed 27 imports. C2 rebuilt the entry point. C3
removed a dead code path and possibly moved 1,637 lines of numerics. Any one of
those could have changed behaviour silently, and `trainer.py` has no test
coverage to catch it.

**This task has no repair mandate.** If a measurement differs, you report `RED`
and stop. You do not fix the code, and you absolutely do not adjust the B1
fixture to match. Nothing in Phase D or later is trustworthy on a failed C4, and
the correct response is a human deciding which of C1/C2/C3 to revisit.

## What you may assume

- `tests/fixtures/baseline/` holds B1's artifacts: `config_baseline.json`,
  `config_physics_informed.json`, `config_lora.json`, `config_combined.json`,
  `config_baseline_override.json`, `dataset_1980.json`,
  `dataset_1980_alignment.txt`, `smoke_baseline.{log,json}`.
- B1's result file states the **float comparison tolerance** (default: relative
  1e-4) and whether parameter counts were taken on CPU or GPU. **Read B1's
  result file before starting** — it may have recorded caveats that change what
  a fair comparison looks like.
- C3 removed the `use_dummy` branch from `build_dataloader`. That is a
  behaviour change on a **dead** path: `unified.yaml` sets `use_dummy: false`,
  so no measurement in B1 exercised it. It must not show up in any diff. If it
  does, something else changed too.
- C3 may have reported `AMBER` on the RMM split (Q-08). That does not affect
  this gate — nothing in B1 measured RMM code.
- `run.py show-config --mode <m>` is the new way to produce the config JSON
  (C2). It must emit the same canonical form B1 used: sorted keys, indent 2.

## Steps

1. Record provenance exactly as B1 did: commit SHA, hostname, `python -V`,
   torch version, GPU model if applicable. **Also record B1's provenance
   alongside it** and state every difference. If B1 ran under conda and you run
   under `uv`, that is a second variable and you must say so — it does not
   invalidate the gate but it changes how a float mismatch should be read.

2. **Config resolution — exact equality required.**

   ```bash
   mkdir -p tests/fixtures/postrefactor
   for m in baseline physics_informed lora combined; do
     uv run python run.py show-config --mode $m > tests/fixtures/postrefactor/config_$m.json
     diff -u tests/fixtures/baseline/config_$m.json tests/fixtures/postrefactor/config_$m.json
   done
   uv run python run.py show-config --mode baseline \
     --override training.optimizer.lr=1e-5 \
     > tests/fixtures/postrefactor/config_baseline_override.json
   diff -u tests/fixtures/baseline/config_baseline_override.json \
           tests/fixtures/postrefactor/config_baseline_override.json
   ```

   **Five empty diffs, or `RED`.** Paste all five commands and their output even
   when empty — an empty diff that was never run looks identical in a result
   file to one that passed.

   C2 should already have checked these. If C2 passed them and you do not,
   something landed between the two tasks; say which commits are in your range
   that were not in C2's.

3. **Parameter counts — exact integer equality required.** Rebuild each of the
   four modes' models via `aurora_mjo.model.load_model` and record total,
   trainable and frozen counts plus MJO-head and LoRA presence. Use whichever
   device B1 used.

   Present as a table with a delta column:

   | Mode | Metric | B1 | Now | Δ |
   | --- | --- | --- | --- | --- |

   **Every Δ must be exactly 0.** A non-zero delta means freezing, LoRA
   insertion or head construction changed — report `RED` and name which mode
   and which metric.

4. **Dataset index — exact equality required.** With
   `HDF5_USE_FILE_LOCKING=FALSE` set, rebuild
   `LANLMJODataset(start_year=1980, end_year=1980, …)` and reproduce every field
   B1 recorded: `len(dataset)`, the per-variable alignment report, probed
   pressure-level indices, static-var shape/mean/std/finite-fraction, and the
   sample-0 and sample-(len−1) tensor summaries.

   `len(dataset)` must equal **1462**. Static means must equal B1's to the
   precision B1 recorded them (`z` 3709.2466, `lsm` 0.3357, `slt` 0.6708).
   The alignment report should be textually identical modulo timestamps and
   paths — diff it and explain every differing line individually.

   **Also re-check the `Using zeros` warning** with
   `warnings.catch_warnings(record=True)`. Its presence or absence must match
   B1. If B1 saw it and you do not — or vice versa — that is a real signal about
   the environment, not a pass; report it prominently.

5. **Smoke-test trajectory — tolerance-based comparison.** Run the smoke test
   through **both** entry points, because C2 introduced a second one and both
   must work:

   ```bash
   uv run python run.py train --mode baseline --smoke-test 2>&1 \
     | tee tests/fixtures/postrefactor/smoke_baseline_runpy.log
   uv run python train.py --config configs/unified.yaml --mode baseline --smoke-test 2>&1 \
     | tee tests/fixtures/postrefactor/smoke_baseline_shim.log
   ```

   Compare per-step losses against B1 at the tolerance B1 stated. Report
   max absolute and max relative deviation as numbers, not adjectives.

   **Compare the non-finite-grad skip count exactly.** If B1 recorded
   skips == steps (see `03_DOMAIN_PRIORS.md` §7), you must reproduce that — it
   is the fingerprint of the current broken forward pass, and reproducing it is
   the correct outcome for a refactor. A run that suddenly *stops* skipping is
   not good news here; it means something changed.

6. Verify the shim path specifically works under `torchrun`, since that is how
   SLURM invokes it and C2's check may have been the only one:

   ```bash
   uv run torchrun --nproc_per_node=1 train.py \
     --config configs/unified.yaml --mode baseline --smoke-test
   ```

7. Write the verdict. One of exactly three:

   - **GREEN** — every exact comparison is exact and every float comparison is
     within tolerance. Phase D may start.
   - **AMBER** — everything exact passed; a float comparison exceeded tolerance
     but the deviation is characterised and attributed to a stated environment
     difference (e.g. conda→uv, different GPU). **Requires the numbers and the
     attribution.** Not a place for optimism.
   - **RED** — any exact comparison differs. Name the mode, the metric, both
     values, and which of C1/C2/C3 most plausibly caused it. **Stop.**

## Definition of Done

- [ ] Provenance for this run **and** B1's, with every difference stated
- [ ] Five config diffs run and pasted, all empty
- [ ] Parameter-count table for four modes with a Δ column; all Δ = 0
- [ ] `len(dataset)` == 1462, stated
- [ ] Static-var mean/std match B1 to recorded precision — table pasted
- [ ] Alignment report diffed; every differing line explained individually
- [ ] `Using zeros` presence matches B1, stated either way
- [ ] Smoke test run through **both** `run.py` and the `train.py` shim
- [ ] Max absolute and max relative loss deviation reported as numbers, against
      the tolerance B1 stated
- [ ] Non-finite-grad skip count compared **exactly** to B1
- [ ] `torchrun` invocation of the shim verified
- [ ] `uv run python scripts/check.py` green — summary table pasted
- [ ] Verdict is exactly one of GREEN / AMBER / RED with the required evidence
- [ ] **`tests/fixtures/baseline/` is byte-identical to before** — prove it with
      `git status --porcelain tests/fixtures/baseline/` returning nothing
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Fixing anything.** No exceptions. If a comparison fails, that is the
  deliverable.
- **Adjusting, regenerating or "correcting" any B1 fixture.** If you believe a
  B1 fixture is wrong, say so in Observations and report `RED`; the human
  decides.
- Adding tests (D1–D3). The comparisons here are one-off measurements, not a
  suite.
- Multi-year dataset construction, or any run longer than the smoke test.
