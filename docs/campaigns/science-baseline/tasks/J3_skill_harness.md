# J3 — 120-step rollout harness; skill vs lead; four baselines

| | |
| --- | --- |
| **Phase** | J |
| **Depends on** | J2 (`PROCEED: YES` — **the gate must be green**), H4 (`PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 6 h |
| **Compute** | **4 GPU-hours** for the harness shakedown only. `sbatch` with approval. |
| **Touches** | `src/aurora_mjo/rmm/evaluate.py`, `src/aurora_mjo/inference.py` (new), `scripts/evaluate_mjo.py`, `run.py` (the `evaluate` subcommand), `slurm_scripts/eval.slurm`, `tests/test_rmm.py`, `tests/test_inference.py` (new), `docs/evaluation-spec.md`, `docs/PROJECT_STATE.md` |
| **Must not touch** | `src/aurora_mjo/rmm/compute.py` and `data/rmm_basis.npz` — **frozen by J2**. `src/aurora_mjo/loss.py`, `dataset.py`, `model.py`. Any `results/*.md`. |

## Objective

A checkpoint can be rolled out 120 steps from a set of 2016–2019 initialisations,
scored through the frozen J2 basis, and reported as bivariate ACC, RMM1/RMM2
RMSE, amplitude ratio and phase error versus lead — alongside three computed
baselines. The output format is fixed here and every later result reuses it.

## Why this is a separate task

Phase K produces four checkpoints and each needs the identical treatment. Building
the harness once, before any of them exists, is what makes the four comparable.
Building it alongside a training run is how a subtle change between stage 2 and
stage 3 evaluation becomes indistinguishable from a change in the model.

It is also where the campaign's compute risk concentrates: 120 autoregressive
steps × N initialisations × 4 checkpoints. Getting the initialisation-sampling
strategy right here saves GPU-hours four times over.

## What you may assume

- J2 is green: the basis reproduces BoM to `r > 0.95` and the sign/order transform
  is frozen (`results/J2_result.md`). **The basis is read-only from here on.**
- H4 added `torch.no_grad()` to the validation path and replaced the clamp table
  with Aurora's native positivity arguments (`results/H4_result.md`). Inference
  uses the same mechanism.
- `02_SCIENTIFIC_CONTRACT.md` §1.2 defines targets T1–T4; §1.4 promotes amplitude
  ratio to a required output; §6.4 lists the baselines.
- `02_SCIENTIFIC_CONTRACT.md` §6.2: the 120-day mean uses **convention (A)** for
  both forecast and observation. Getting this wrong is the one error that makes
  a bad model look excellent.
- `04_AGENT_PROTOCOL.md` §4: all initialisations come from 2016–2019.

## Steps

1. Build `src/aurora_mjo/inference.py`: given a checkpoint and an initialisation
   time, roll out 120 steps and return the predicted surface and `u850` / `u200`
   fields at daily resolution. Reuse Aurora's own `rollout` semantics where
   possible rather than re-deriving the history-window update.
2. **Decide and justify the initialisation sampling.** The trade-off is coverage
   against GPU-hours: 2016–2019 has ~1,460 days, and 120 steps each is not
   affordable. Options include every 5th day, every day with `A(t₀) > 1`, or a
   stratified sample across all eight phases and all four seasons. **Recommend
   one, state the GPU-hour implication at G2's measured step time, and record why
   the others were rejected.** This choice constrains every later result, so it is
   made once and written down.
3. Condition metrics on `A(t₀) > 1`, following Suematsu et al. [9], so skill is
   measured on active events and not on noise. Report the number of cases
   surviving the condition — a skill curve computed on 30 events is a different
   object from one computed on 300, and the paper must say which.
4. Implement the metrics: bivariate ACC by lead, RMM1 and RMM2 RMSE by lead,
   **mean amplitude ratio `Â/A_obs` by lead** (T4), and mean absolute phase error
   by lead. `bivariate_acc`, `rmse_pair`, `amplitude_error` and `phase_error_deg`
   already exist in `evaluate.py` and were reviewed as correct — reuse, do not
   rewrite.
5. Implement the three computed baselines from `02_SCIENTIFIC_CONTRACT.md` §6.4:
   persistence, damped persistence with `τ_d` fitted on **training years only**,
   and climatology. The fourth baseline is J4.
6. **Also report gridded tropical RMSE for `ttr`, `tcwv` and `u850` by lead**, on
   the same axes as RMM ACC. `02_SCIENTIFIC_CONTRACT.md` §1.5 predicts that
   gridded skill and RMM skill diverge, because the RMM projection filters the
   small scales where a deterministic model blurs first. That prediction is
   testable and this is the task that tests it.
7. Fix the output format: one JSON per evaluated checkpoint with the full
   per-lead table, plus a plot writer. Every later result reuses both. Write the
   schema into `docs/evaluation-spec.md`.
8. **Add the leakage assertion as a test, not a comment.** Assert that no
   timestamp used in a forecast's 120-day window is later than `t₀`. This is the
   Suematsu trap and it deserves an assertion that fires.
9. Shake the harness down on **one** initialisation with an untrained model.
   Confirm 120 steps complete, output is finite, and the metrics are computable.
   Do not interpret the numbers — that is J4.
10. Rewrite `slurm_scripts/eval.slurm` against the new entry point. It was
    non-functional before E3 and E3 rewrote it against the old CLI.

## Definition of Done

- [ ] `src/aurora_mjo/inference.py` rolls out 120 steps; output shapes and
      finiteness pasted
- [ ] **Initialisation sampling decided, justified, GPU-hour cost stated at G2's
      measured step time, rejected alternatives named**
- [ ] Metrics conditioned on `A(t₀) > 1`; surviving case count reported
- [ ] ACC, RMM1/RMM2 RMSE, amplitude ratio and phase error all implemented by lead
- [ ] Persistence, damped persistence (`τ_d` from training years) and climatology
      baselines computed; `τ_d` value pasted
- [ ] Gridded tropical RMSE for `ttr`, `tcwv`, `u850` by lead, on the same axes
- [ ] Output JSON schema fixed and documented in `docs/evaluation-spec.md`
- [ ] **Leakage assertion implemented as a test**: no 120-day-window timestamp
      later than `t₀`; test output pasted
- [ ] Single-initialisation shakedown completes 120 steps with finite output
- [ ] `slurm_scripts/eval.slurm` rewritten and exits 0
- [ ] `data/rmm_basis.npz` unchanged; `git diff --stat` pasted showing no change
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] `docs/PROJECT_STATE.md` updated
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Interpreting any skill number.** J3 builds the instrument. J4 takes the first
  reading.
- Touching `rmm/compute.py` or the frozen basis. If you believe the basis is
  wrong, that is an Observation and a `Q-nn`, not an edit — J2 froze it for a
  reason.
- Any fine-tuned checkpoint. None exists yet.
- Literature comparison numbers. **Q-12**, and `02_SCIENTIFIC_CONTRACT.md` §1.3:
  only the four self-computed baselines go on the axes.
- Reading test years.
