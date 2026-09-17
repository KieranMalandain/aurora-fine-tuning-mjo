# H2 — A real 2-D spatial spectral loss, per variable, default off

| | |
| --- | --- |
| **Phase** | H |
| **Depends on** | G4 (`PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 2 h |
| **Compute** | **None.** CPU, synthetic fixtures only. |
| **Touches** | `src/aurora_mjo/loss.py` (only `SpectralLoss`), `src/aurora_mjo/trainer.py` (only `_extract_batch_outputs` and the spectral call site), `configs/unified.yaml`, `tests/test_loss.py`, `docs/PROJECT_STATE.md` |
| **Must not touch** | `TropicalWeightedL1Loss` / the H1 grid loss. `MoistureBudgetLoss`. `src/aurora_mjo/dataset.py`, `model.py`, `rmm/`. Any `results/*.md`. |

## Objective

`SpectralLoss` computes a genuine 2-D spatial power-spectrum discrepancy, per
variable, on the `(lat, lon)` axes. `_extract_batch_outputs` no longer flattens
every variable into one vector for it. The term stays **disabled by default**.

## Why this is a separate task

`00_CONTEXT.md` R6: `_extract_batch_outputs` (`trainer.py:126-138`) flattens and
concatenates every variable into a `(B, N)` tensor, and `SpectralLoss`
(`loss.py:69-72`) then calls `rfft2` on it — transforming over
`(batch, concatenated-everything)`. At `B = 1` that is a 1-D FFT over a
row-major-flattened mixture of `z`, `q`, `t`, `u`, `v` across 13 levels and the
surface variables.

No training run was affected because the term is disabled. But
`docs/papers/aurora-mjo-fall-paper.pdf` §IV-B attributes the sharp convective
structures and the Day-10 grittiness to it at λ = 0.05, and that interpretation
is not supported by what the code computes. Fixing it is small, and leaving a
term in the repo that does not do what its docstring says is how the next person
gets misled.

## What you may assume

- G4 confirmed the loader produces `[B, 1, 180, 360]` surface and
  `[B, 1, 13, 180, 360]` atmospheric predictions (`results/G4_result.md`).
- The term is `enabled: false, weight: 0.0` in `configs/unified.yaml` and stays
  that way. `02_SCIENTIFIC_CONTRACT.md` reserves it for a controlled ablation in
  a later campaign.
- H1 may be in flight in parallel. Coordinate only through the shared weighting
  interface; do not edit the grid loss.

## Steps

1. Rewrite `SpectralLoss` to take a single variable's field, `(B, L, H, W)` or
   `(B, H, W)`, and take `rfft2` over the **last two axes only**, per level.
2. Normalise per variable before the transform, using the same G3 constants the
   grid loss uses. An unnormalised spectral loss inherits R1's scale problem
   wholesale — `z` and `msl` would dominate the spectrum term exactly as they
   dominated the grid term.
3. Apply the same `w_v` weights as H1, so a variable's influence is one number
   and not two.
4. Longitude is periodic and latitude is not. Applying a plain `rfft2` treats
   both as periodic, which introduces a spurious high-wavenumber signal from the
   pole-to-pole discontinuity. Either window in latitude or restrict the term to
   the tropical band. **State which you chose and why** — this is a real modelling
   choice, not a detail.
5. Consider whether the term should compare **amplitude spectra** rather than
   complex coefficients. `|FFT(x̂) − FFT(x)|` penalises phase error, which the
   grid loss already does; `||FFT(x̂)| − |FFT(x)||` penalises only the texture,
   which is the term's stated purpose. **Recommend one, implement it, and record
   the argument** — the docstring's justification ("match the texture and spatial
   variance rather than just the position") argues for amplitude-only.
6. Delete the spectral path through `_extract_batch_outputs`, or delete the
   function if nothing else uses it. Grep before deleting.
7. Add to `tests/test_loss.py`: a single sinusoid at known wavenumber `k`
   concentrates the loss in bin `k`; two identical fields give zero; a field and
   its spatial shift give a **small** amplitude-spectrum loss but a large
   complex-spectrum loss — this is the test that distinguishes the two
   formulations and it is worth having regardless of which you ship.
8. Update the docstring so it describes what the code does.

## Definition of Done

- [ ] `rfft2` taken over `(lat, lon)` only, per variable, per level
- [ ] Fields normalised with G3 constants before the transform
- [ ] `w_v` applied consistently with H1
- [ ] Latitude non-periodicity handled; the choice stated and justified in the
      docstring and the result file
- [ ] Amplitude-vs-complex recommendation made, implemented, and argued
- [ ] Spectral path removed from `_extract_batch_outputs`; the function deleted
      if unused (`grep` output pasted)
- [ ] Sinusoid test passes: loss concentrates in the correct wavenumber bin
- [ ] Identical-fields test gives exactly zero
- [ ] Shifted-field test distinguishes amplitude from complex formulation; both
      numbers pasted
- [ ] Term remains `enabled: false, weight: 0.0` in every mode
- [ ] Docstring rewritten to describe the implementation
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] Result file written from `results/_TEMPLATE.md`, including a plain
      statement that `docs/papers/` §IV-B's interpretation of the Day-10
      grittiness is not supported by the pre-H2 code

## Out of scope

- **Enabling the term.** It ships off. `02_SCIENTIFIC_CONTRACT.md` reserves it
  for a later ablation and this campaign has one run per stage, not a grid.
- Tuning λ_s.
- The grid loss (H1), the moisture term (H3).
- Editing `docs/papers/`. Record the discrepancy; the paper campaign fixes it.
