# J4 — **GATE.** The zero-shot Aurora control

| | |
| --- | --- |
| **Phase** | J |
| **Depends on** | J3, G2 (both `PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 4 h plus job wall-clock |
| **Compute** | **Up to 12 GPU-hours**, governed by J3's sampling decision. `sbatch` with approval. |
| **Touches** | `docs/findings/2026-1x-zeroshot-control.md` (new), `data/results/zeroshot_control.json` (new), `docs/PROJECT_STATE.md`, `03_DOMAIN_PRIORS.md` (§9 only) |
| **Must not touch** | Anything under `src/`. `data/rmm_basis.npz`. Any `results/*.md`. **If the harness is broken, you report it; you do not fix it.** |

## Objective

The MJO forecast skill of **un-fine-tuned** `AuroraPretrained` — rolled out 120
steps, scored through the frozen J2 basis, on 2016–2019 active events — is
measured and written down. Target **T1** is defined relative to this number.

## Why this is a gate, and why it is the most important number in the campaign

This project has never had a control. Every claim it has made is of the form "our
fine-tuned model does X", with nothing to compare X against. The May research
script's central hypothesis — that Aurora's pre-trained physics transfers to the
MJO setting given the injected variables (§IV-E, "First") — has never been tested,
because testing it requires exactly this measurement and nobody made it.

It is structurally the same move as `refactor`'s B1: capture the before-state
before touching anything, because afterwards you cannot.

Three things it settles:

1. **T1 becomes measurable.** "Fine-tuning beats zero-shot at every lead" is the
   campaign's definition of success and is meaningless without this number.
2. **It bounds the value of the whole project.** If zero-shot Aurora already has
   MJO skill at 15 days, the fine-tuning has to beat a strong baseline. If it has
   none, the paper's contribution is larger and the risk is higher.
3. **It validates the harness end to end** on a model whose behaviour is at least
   partly known from Aurora's own published evaluation.

> **A surprisingly high result is not good news.** Before reporting anything above
> ~0.6 ACC at day 20, work the leakage list in `03_DOMAIN_PRIORS.md` §9: 120-day
> mean leakage across the initialisation, test-year contamination, and whether
> the "forecast" RMM is being computed from observed rather than predicted fields.
> A control that outperforms the literature has a bug in it.

## What you may assume

- J3 built and shook down the harness and fixed the initialisation sampling
  (`results/J3_result.md`). **Use its sampling exactly** — a control measured on
  a different sample is not a control.
- G2 confirmed `AuroraPretrained` as the ERA5 base and measured its step time at
  1° (`results/G2_result.md`).
- J2's basis is frozen and reproduces BoM to `r > 0.95` (`results/J2_result.md`).
- The model has **randomly initialised** `ttr` and `tcwv` patch embeddings and
  decoder heads, because no training has happened. See Step 2 — this is the one
  genuinely awkward part of the measurement.

## Steps

1. Load `AuroraPretrained` with the six surface variables, LoRA off, no
   checkpoint beyond the pretrained weights, and G3's normalisation statistics.
2. **Decide how to handle the untrained injected channels, and justify it.** The
   options are not equivalent and the choice changes the number:
   - **(a)** Roll out with the six-variable model as-is. The random `ttr` / `tcwv`
     heads will inject noise that compounds over 120 steps, and the control will
     partly measure that noise rather than Aurora's physics.
   - **(b)** Roll out Aurora's **native four-variable** configuration and take
     `u850` / `u200` from the atmospheric output, using **observed** OLR for the
     RMM projection. This isolates Aurora's dynamical skill but does not test the
     injection, and the RMM is then only two-thirds forecast.
   - **(c)** Run both and report both.
   **Recommendation: (c).** (b) is the honest measure of "does Aurora's
   pre-trained physics transfer", which is the hypothesis; (a) is the honest
   starting point for the fine-tuning comparison, which is T1. They answer
   different questions and the paper needs both. Whatever you choose, **state
   which of the two questions each number answers.**
3. Roll out 120 steps from every initialisation in J3's sample.
4. Score through the frozen basis. Produce the full per-lead table: bivariate
   ACC, RMM1/RMM2 RMSE, amplitude ratio, phase error, plus the three computed
   baselines on the same axes.
5. Report the **ACC = 0.5 crossing lead** for the control and for each baseline.
   This is the single number T1 and T3 are stated against.
6. Report the **amplitude ratio at day 10, 20 and 30**. A zero-shot deterministic
   model is expected to damp
   (`02_SCIENTIFIC_CONTRACT.md` §1.4); quantifying by how much sets the scale for
   what Phase K needs to improve.
7. Report gridded tropical RMSE alongside RMM ACC and **state whether the
   §1.5 prediction held** — that the RMM projection filters the small-scale
   blurring and the two curves diverge.
8. Run the leakage checks in the gate box above **before** writing any
   conclusion, and state each verdict explicitly in the result file.
9. Write `docs/findings/2026-1x-zeroshot-control.md`. This must outlive the
   campaign directory — it is the reference every future result on this project
   is measured against, including in the paper campaign and the v1.5 work.
10. Update `03_DOMAIN_PRIORS.md` §9 with the measured control row, relabelled
    **MEASURED**, and say whether the LITERATURE rows around it now look
    plausible.

## Definition of Done

- [ ] Configuration decision made between (a), (b), (c) and **justified**, with
      each reported number tied to the question it answers
- [ ] 120-step rollouts complete for J3's full initialisation sample; case count
      and any failures reported
- [ ] Full per-lead table pasted: ACC, RMSE, amplitude ratio, phase error, three
      baselines
- [ ] **ACC = 0.5 crossing lead** reported for control and every baseline
- [ ] Amplitude ratio at days 10, 20, 30 reported
- [ ] Gridded tropical RMSE reported; the §1.5 divergence prediction assessed
- [ ] **All three leakage checks run and each verdict stated explicitly**
- [ ] `data/results/zeroshot_control.json` committed in J3's schema
- [ ] `docs/findings/2026-1x-zeroshot-control.md` written
- [ ] `03_DOMAIN_PRIORS.md` §9 updated to MEASURED
- [ ] GPU-hours used, reported against the 12-hour budget
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] `docs/PROJECT_STATE.md` updated with the control number as the T1 reference
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Any training, of any kind.** J4 measures the model as Microsoft shipped it.
- **Fixing the harness.** If J3's code is wrong, report it and mark `RED`. An
  agent that fixes the instrument while taking the reading has destroyed the
  reading.
- Touching `rmm/` or the frozen basis.
- Literature comparison. **Q-12** and `02_SCIENTIFIC_CONTRACT.md` §1.3.
- Reading test years.
- Interpreting a low control number as bad news. A control with no skill makes
  Phase K's job clearer, not harder.
