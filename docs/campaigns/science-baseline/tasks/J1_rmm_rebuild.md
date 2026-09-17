# J1 — RMM rebuilt with zonal structure, harmonics and the 120-day mean

| | |
| --- | --- |
| **Phase** | J |
| **Depends on** | G1, G4 (both `PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 6 h |
| **Compute** | Perlmutter login node or one CPU SLURM job over 1980–2019. **No GPU.** |
| **Touches** | `src/aurora_mjo/rmm/compute.py`, `src/aurora_mjo/rmm/evaluate.py`, `src/aurora_mjo/rmm/__init__.py`, `scripts/compute_rmm.py`, `tests/test_rmm.py` (new), `data/rmm_basis.npz`, `data/rmm_targets.nc`, `docs/evaluation-spec.md`, `docs/PROJECT_STATE.md` |
| **Must not touch** | `src/aurora_mjo/loss.py`, `dataset.py`, `trainer.py`, `model.py`. Phase H is running in parallel — **reaching across serialises the campaign.** Any `results/*.md`. |

## Objective

`aurora_mjo.rmm` implements Wheeler & Hendon (2004) as specified in
`02_SCIENTIFIC_CONTRACT.md` §6: `a(t) ∈ ℝ^(3·N_λ)` with `N_λ = 360`, seasonal
cycle as mean plus three harmonics, 120-day mean removal under convention (A),
zonal-mean-σ normalisation, SVD, unit-variance PCs, and a sign/order transform
frozen into the basis file. `docs/evaluation-spec.md` is updated to match.

## Why this is a separate task

`00_CONTEXT.md` R3. `rmm/compute.py:42` averages over latitude **and longitude**,
returning one scalar per field per day, and `scripts/compute_rmm.py` then
eigendecomposes a 3×3 covariance. The zonal structure is what makes the MJO an
eastward-propagating phenomenon; without it the phase angle is meaningless and
the `A > 1` threshold is uncalibrated.

**`docs/evaluation-spec.md` specifies the 3-element vector.** The code is a
faithful implementation of a wrong spec, so fixing only the code leaves the
landmine in place. Both change in this task or neither does.

`src/aurora_mjo/rmm/` has **zero test coverage** — there is no `test_rmm.py`, and
`tests/test_mjo_head.py` matches only on the string "RMM". That is the mechanism
by which this survived a whole refactor campaign.

## What you may assume

- G1: data at native 1°, 360 longitudes, grid read from the archive
  (`results/G1_result.md`).
- The existing **leakage contract in `scripts/compute_rmm.py` is correct** and is
  the one part of the pipeline that does not need fixing: harmonics, σ and EOFs
  are fitted on 1980–2015 only, and everything else is projected onto a frozen
  basis. Preserve it and extend it.
- `OLR = −mtnlwrf`. The sign must be **confirmed in J2**, not asserted here; a
  flip silently renumbers every phase.
- `03_DOMAIN_PRIORS.md` §7 gives seven checkable properties of a correct RMM
  (variance explained, PC variance, RMM1/RMM2 orthogonality, the ~10–12 day
  quadrature lag, the ~50% active fraction, the 30–60 day cycle period).
- `04_AGENT_PROTOCOL.md` §4: `data/rmm_targets.nc` covers **1980–2019 only**.
  Test years are not read.

## Steps

1. Rewrite `tropical_mean` → a meridional average over 15°S–15°N that **retains
   longitude**, returning `(time, longitude)`. Use a **simple unweighted mean** to
   match WH; the existing cos-latitude weighting is a deviation and is removed.
2. Replace `compute_daily_clim`'s raw day-of-year mean with **mean plus the first
   three harmonics** of the annual cycle, fitted per longitude on 1980–2015.
3. Implement the **120-day mean removal**, which is currently absent entirely.
   Read `02_SCIENTIFIC_CONTRACT.md` §6.2 before writing a line of it — it is the
   Suematsu trap, and convention (A) (observations only, up to `t₀`, held fixed
   across the forecast) is mandated for both forecast and observation.
4. Normalise each field by its **zonally averaged temporal standard deviation**,
   one scalar per field, from 1980–2015.
5. Concatenate to `(T, 1080)` and take EOFs by **SVD**, not by forming a
   1080×1080 covariance matrix.
6. Normalise each PC by its training-period standard deviation so RMM1 and RMM2
   have unit variance and `A > 1` means what it means everywhere else.
7. Implement the sign/order transform as a **stored, applied, frozen** part of
   `rmm_basis.npz`. J2 determines its values; J1 provides the mechanism and
   defaults to identity.
8. Rewrite `extract_rmm_from_fields` in `evaluate.py`, which has the same scalar
   collapse (`_trop_mean_surf` returns a `float`). It must produce the
   `(longitude,)` vector and project it.
9. Write `tests/test_rmm.py` to `01_TARGET_STATE.md` D8. The headline test:
   **the tropical average has a longitude dimension of 360, not a scalar** —
   it fails on the pre-J1 code, and that failure must be pasted. Plus: a
   synthetic eastward wavenumber-1 signal produces RMM1/RMM2 in quadrature with a
   monotonically advancing phase through all eight phases; the 120-day window at
   a forecast valid time contains no timestamp after `t₀`; fitting on a train
   slice and projecting a val slice does not change the basis.
10. Run the full pipeline over 1980–2019 and check all seven properties in
    `03_DOMAIN_PRIORS.md` §7 except the BoM correlation, which is J2.
11. **Rewrite `docs/evaluation-spec.md`'s RMM sections** to
    `02_SCIENTIFIC_CONTRACT.md` §6, and say in the file that the previous
    3-element specification was wrong and when it was corrected. A spec that
    quietly changes teaches nobody.

## Definition of Done

- [ ] `tropical_mean` returns `(time, longitude)` with 360 longitudes; simple
      unweighted meridional mean
- [ ] Seasonal cycle is mean + three harmonics, fitted per longitude on 1980–2015
- [ ] 120-day mean removal implemented under **convention (A)**; the convention
      named in a code comment and the result file
- [ ] Zonal-mean-σ normalisation, one scalar per field, training period only
- [ ] EOFs by SVD of `(T, 1080)`; no 1080×1080 covariance formed
- [ ] PCs normalised to unit variance
- [ ] Sign/order transform mechanism present and stored in `rmm_basis.npz`,
      defaulting to identity
- [ ] `extract_rmm_from_fields` rewritten; no scalar collapse remains
      (`grep` for `float(` in the tropical-mean path, pasted)
- [ ] `tests/test_rmm.py` passes; **the longitude-dimension test fails on the
      pre-J1 code and that failure is pasted**
- [ ] Six of seven `03_DOMAIN_PRIORS.md` §7 properties checked and pasted
      (BoM correlation deferred to J2)
- [ ] `data/rmm_targets.nc` covers **1980–2019 only**; year range pasted
- [ ] `docs/evaluation-spec.md` rewritten, with a note recording the correction
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] `docs/PROJECT_STATE.md` updated
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **The BoM comparison.** That is J2 and it is the gate. J1 builds the pipeline;
  J2 decides whether it works.
- Skill metrics and the rollout harness (J3).
- The MJO head (J5).
- Anything in `src/aurora_mjo/loss.py` or `dataset.py`. Phase H is running in
  parallel.
- Reading test years, for any reason.
