# H3 — ERA5-supervised `E − P`; `tp6h` and `mslhf` in the loader

| | |
| --- | --- |
| **Phase** | H |
| **Depends on** | G4, H1 (both `PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 5 h |
| **Compute** | Perlmutter login node for CFS reads and the §Step-1 diagnostic. **No `sbatch`, no GPU.** |
| **Touches** | `src/aurora_mjo/loss.py` (only `MoistureBudgetLoss`), `src/aurora_mjo/dataset.py` (only the variable maps and target assembly for the two new fields), `configs/unified.yaml`, `tests/test_loss.py`, `tests/fixtures/synthetic_archive*/`, `docs/PROJECT_STATE.md`, `03_DOMAIN_PRIORS.md` (§8 only) |
| **Must not touch** | `_build_aligned_index` (Lesson 5). The H1 grid loss. `SpectralLoss`. `src/aurora_mjo/model.py`, `rmm/`. Any `results/*.md`. |

## Objective

The moisture term penalises `|R − (E−P)_ERA5|` over the tropics instead of `|R|`,
with the column integral masked at surface pressure. `tp6h` and `mslhf` are read
from the archive as loss targets. The term ships at `weight: 0.0` and is logged
as a diagnostic.

## Why this is a separate task

`00_CONTEXT.md` R2. The current term drives `R = ∂⟨q⟩/∂t + ∇·⟨vq⟩` toward zero.
By the budget equation `R` **is** `E − P`, whose true value in an active MJO
envelope is **−15 to −35 mm/day** (`03_DOMAIN_PRIORS.md` §8). Driving it to zero
penalises the convective envelope the project exists to predict, and rewards
smoothing — the opposite of the docstring's stated intent.

The argument for this fix over retiring the term entirely, including the case
against, is `02_SCIENTIFIC_CONTRACT.md` §3. It was decided by the human on
2026-09-13. Do not re-open it; do run the Step-1 diagnostic, which can still
retire the term on evidence.

## What you may assume

- Archive locations, from `docs/nersc-dataset-information.md`:
  `tp6h` at `Step06/ERA5.remap_180x360MODIS_6hrAccu/TP6H`, accumulation over the
  previous 6 h, **metres**; `mslhf` at
  `Step03/ERA5.remap_180x360MODIS_6hrInst/meanSLHFLX`, mean surface latent heat
  flux over the previous 1 h, **W m⁻²**.
- **`tp1h` must not be used.** The archive is missing
  `e5.accumulated_tp_1h.202206.nc` and the unit changes from m to kg m⁻² s⁻¹ at
  `202204`. That is documented upstream and it is a trap.
- The v3 timestamp maps already handle variables with different file chunking, so
  adding two variables is a map entry each, not new machinery.
- The existing numerical scheme — spherical divergence with circular longitude
  padding, replicate at the poles, `cos φ` clamped at 1e-5, float32 outside
  autocast — was reviewed and is **correct**. Keep it.

## Steps

1. **Run the diagnostic before writing any loss code.** Using ERA5 fields only,
   compute `R = ∂⟨q⟩/∂t + ∇·⟨vq⟩` over a month of training data and compare it
   against ERA5's own `E − P` from `mslhf` and `tp6h`. Report the correlation and
   the ratio of magnitudes over the tropical band.
   - **Agreement within a factor of ~2 → the numerics are sound**, proceed.
   - **Residual dominated by discretisation noise → retire the term**, mark the
     task `AMBER`, and say so. That is a clean result and it saves a later
     ablation. `02_SCIENTIFIC_CONTRACT.md` §3.3 authorises this outcome.
2. Add `tp6h` and `mslhf` to `SURFACE_VAR_MAP`. They are **loss targets, not
   model inputs** — they must not appear in `model.surface_variables`, must not
   get patch embeddings, and must not enter the grid loss. Assemble them into the
   target dict only.
3. Convert to a common basis in mm/day:
   `E = mslhf / L_v` with `L_v = 2.501 × 10⁶ J kg⁻¹`, then × 86400;
   `P = (tp6h / Δt) × 1000` with `Δt = 21600 s`, then × 86400.
   **Check ERA5's downward-positive flux convention explicitly** and state the
   resulting sign of `E` in a comment. A sign error here flips the target and
   would be invisible in the loss value.
4. Mask the column integral at surface pressure: levels with `p > ps` are
   excluded, not extrapolated. The current integral runs 50–1000 hPa everywhere,
   which is wrong over land and specifically wrong over the Maritime Continent —
   the region where MJO propagation fails in most models.
5. Change the reduction to `mean(|R − (E−P)_ERA5|)` over the tropical band. Keep
   the existing clamp and the non-finite guard.
6. **Log the term as a diagnostic at `weight: 0.0`** in every mode. It is enabled
   with a non-zero weight only in K4.
7. Add to `tests/test_loss.py`: an analytic non-divergent flow with a prescribed
   `E − P` returns that `E − P` to tolerance; the surface-pressure mask excludes
   the right levels for a synthetic mountain; a field whose `E − P` matches ERA5
   exactly gives zero loss; a **smoothed** field gives a **larger** loss than a
   sharp one — this last is the direct regression test for R2 and it fails on the
   pre-H3 code.
8. Extend the synthetic fixtures with `tp6h` and `mslhf`.
9. Update `03_DOMAIN_PRIORS.md` §8 with the Step-1 measured correlation and ratio.

## Definition of Done

- [ ] **Step-1 diagnostic run and reported**: correlation and magnitude ratio
      between computed `R` and ERA5 `E − P` over one training month, tropical band
- [ ] Verdict stated plainly: proceed, or retire the term
- [ ] `tp6h` and `mslhf` read from the archive; `tp1h` **not** used anywhere
- [ ] Both are targets only — `grep` showing neither appears in
      `model.surface_variables` or the grid loss, output pasted
- [ ] Unit conversion to mm/day implemented; ERA5 sign convention checked and the
      resulting sign of `E` stated in a comment and the result file
- [ ] Column integral masked at `ps`; synthetic-mountain test passes
- [ ] Reduction is `mean(|R − (E−P)_ERA5|)`
- [ ] **Smoothed-field-gives-larger-loss test passes, and fails on the pre-H3
      code**; both outputs pasted
- [ ] Analytic non-divergent flow test passes to tolerance
- [ ] Term at `weight: 0.0` in every mode; diagnostic logging confirmed
- [ ] Synthetic fixtures extended
- [ ] `03_DOMAIN_PRIORS.md` §8 updated with measured values
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] `docs/PROJECT_STATE.md` updated
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Enabling the term.** K4 does that, after the diagnostic has run for a whole
  stage.
- Tuning λ_p.
- Adding precipitation as a **prognostic** variable. That changes the decoder's
  variable set and is a much larger decision than this task.
- Re-opening the retire-vs-supervise decision on preference. The Step-1
  diagnostic may retire it **on evidence**; nothing else may.
- Touching the divergence numerics, which were reviewed and are correct.
