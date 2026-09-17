# H1 — Normalised, weighted, area-correct grid loss

| | |
| --- | --- |
| **Phase** | H |
| **Depends on** | G3, G4 (both `PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 4 h |
| **Compute** | **None.** CPU, synthetic fixtures only. No `sbatch`. |
| **Touches** | `src/aurora_mjo/loss.py`, `src/aurora_mjo/trainer.py` (only `_single_step_losses` and loss construction), `configs/unified.yaml`, `tests/test_loss.py` (new), `docs/SPEC.md`, `docs/PROJECT_STATE.md` |
| **Must not touch** | `src/aurora_mjo/dataset.py`. `src/aurora_mjo/model.py`. `src/aurora_mjo/rmm/`. `MoistureBudgetLoss` (H3) or `SpectralLoss` (H2) beyond what the shared interface requires. Any `results/*.md`. |

## Objective

`L_grid` is computed on normalised values with per-variable weights `w_v`,
per-level weights `c_ℓ`, cos-latitude area weighting `a(φ)` and tropical emphasis
`m(φ)`, exactly as `02_SCIENTIFIC_CONTRACT.md` §4 specifies. Every training step
logs a per-variable loss component.

## Why this is a separate task

This is the single highest-impact change in the campaign. `00_CONTEXT.md` R1:
the current loss gives `msl` and `z` **99.03%** of the gradient and `q`
**0.000012%**, with an `msl`:`q` ratio of **5,808,226 : 1**. The two variables
injected because the MJO is a moisture-mode phenomenon receive 0.49% between
them.

It is isolated because it invalidates every historical loss number, and an
isolated commit is the only way that invalidation stays attributable.

## What you may assume

- G3 committed true 1980–2015 statistics (`results/G3_result.md`). The loss
  normalises with **the same constants the model normalises with** — not a
  second, independently computed set.
- Aurora denormalises its output, so both prediction and target must be
  re-normalised before the loss. This is not optional and it is the change.
- `03_DOMAIN_PRIORS.md` §5 gives the before-column (measured) and the
  after-column (derived from `w_v`). Your test asserts the after-column.
- The existing `TropicalWeightedL1Loss` has **no area weighting**. At 1° a cell
  at 89.5° counts the same as one at the equator despite covering ~1% of the
  area. This is a defect independent of R1 and is fixed here.

## Steps

1. Implement the loss to `02_SCIENTIFIC_CONTRACT.md` §4. Keep `a(φ)` and `m(φ)`
   **separate and multiplicative** — collapsing them means the tropical emphasis
   cannot be changed without breaking area weighting.
2. Default `c_ℓ = Δp_ℓ / Σ Δp`, which naturally emphasises the lower troposphere
   where the moisture is. `c_ℓ ≡ 1` for surface variables. Make it config-visible
   with a `uniform` alternative.
3. Put `w_v` in `configs/unified.yaml` with the §4.1 defaults and a comment saying
   they were set **a priori and are not tuned against validation skill** — doing
   so and then reporting that skill is leakage
   (`02_SCIENTIFIC_CONTRACT.md` §6.3).
4. In `_single_step_losses`, emit `loss/grid/<var>` for all eleven variables
   every logged step. The single scalar that hid this problem for six months is
   the same scalar that will hide the next one.
5. Make the grid-loss weight explicit. Currently `spectral`, `mjo_head` and
   `moisture_budget` are pre-multiplied by their weights inside
   `_single_step_losses` while `grid` is not, so its weight is implicitly 1.
   Apply all four the same way.
6. Write `tests/test_loss.py` to `01_TARGET_STATE.md` D8. The headline test:
   **per-variable gradient share is within tolerance of `w_v / Σw_v` when all
   normalised errors are equal.** Run it against the pre-H1 loss and confirm it
   **fails**; paste that failure. A regression test that passes on the broken
   code is not a regression test.
7. Also test: area weights integrate to 1 over the sphere; level weights sum to 1;
   a constant offset in one variable moves only that variable's component.
8. Run `--smoke-test` and record the new loss value. Compare against
   **7472.024414** from B1 and explain the difference in one sentence.
9. Update `docs/SPEC.md` with the loss specification — it is durable truth and
   should outlive this campaign directory.

## Definition of Done

- [ ] Loss computed on normalised values using the G3 constants; the
      denormalise-then-renormalise path stated explicitly in a code comment
- [ ] `a(φ)`, `m(φ)`, `c_ℓ`, `w_v` all present, separate, and config-visible
- [ ] `w_v` defaults match `02_SCIENTIFIC_CONTRACT.md` §4.1, with the
      no-tuning comment
- [ ] `loss/grid/<var>` logged for all eleven variables; sample log line pasted
- [ ] All four loss components weighted consistently
- [ ] `tests/test_loss.py` passes; **the gradient-share test fails on the pre-H1
      loss and that failure output is pasted**
- [ ] Measured per-variable gradient share within tolerance of
      `03_DOMAIN_PRIORS.md` §5 after-column; table pasted
- [ ] Area weights integrate to 1; level weights sum to 1
- [ ] Smoke-test loss recorded and compared to 7472.024414
- [ ] `docs/SPEC.md` updated with the loss specification
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] Discontinuity section — smoke losses, and a note that
      `docs/archive/metrics/*.jsonl` and `docs/papers/` Figures 2 and 6 are now
      incomparable
- [ ] `docs/PROJECT_STATE.md` updated
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- The spectral loss (H2) and the moisture term (H3), beyond the shared weighting
  interface.
- Tuning `w_v`. They are a prior. Changing them needs a result file behind it and
  this is not that task.
- Any training run. A `--smoke-test` is not a training run.
- Touching `docs/archive/metrics/` or the `docs/papers/` PDF. Both are historical
  record; the discontinuity is *recorded*, not retro-fixed.
