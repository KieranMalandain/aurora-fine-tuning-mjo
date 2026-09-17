# J7 — ENSO, season and phase stratification; the regime inventory

| | |
| --- | --- |
| **Phase** | J |
| **Depends on** | J3 (`PROCEED: YES`). Interpretable only if G5 landed — see §Why. |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 4 h |
| **Compute** | **None** beyond reusing J3/J4 rollout output. No new `sbatch`. |
| **Touches** | `src/aurora_mjo/rmm/regimes.py` (new), `scripts/fetch_oni.py` (new), `data/reference/oni.csv` (new), `tests/test_regimes.py` (new), `docs/evaluation-spec.md`, `docs/findings/2026-1x-regime-inventory.md` (new), `docs/PROJECT_STATE.md`, `03_DOMAIN_PRIORS.md` (§11 only) |
| **Must not touch** | `src/aurora_mjo/rmm/compute.py` and `data/rmm_basis.npz` — **frozen by J2**. Anything else under `src/`. Any `results/*.md`. |

## Objective

Every day from 1980 to 2019 is classified by ENSO regime, season and MJO phase.
The regime composition of the train and validation splits is documented. Every
J3 skill metric is reported stratified by ENSO regime, by season, and by initial
MJO phase.

## Why this task exists, and what it can and cannot conclude

MJO behaviour is ENSO-modulated: the warm pool extends east under El Niño and
MJO convection propagates further into the central Pacific; under La Niña activity
is more confined to the Indian Ocean and Maritime Continent. A single skill curve
averaged over all regimes hides this, and `docs/papers/` §Future Work item 5
already names it as an open question.

**But be precise about what a regime-stratified result means here.**
`02_SCIENTIFIC_CONTRACT.md` §2.5: Aurora v1.0 is atmosphere-only. Without G5's
persisted SST the model has **no input channel carrying ENSO state**, so any
skill difference between regimes would reflect the atmospheric initial condition
and the observed predictability of each regime — not learned ENSO-conditional
behaviour.

- **If G5 landed:** the stratification tests whether the model uses the ocean
  boundary condition it was given. That is a real result.
- **If G5 did not land:** run the stratification anyway, but state in the first
  line of the result file that it measures *regime-dependent predictability of the
  initial condition*, not ENSO-conditional model behaviour. Mark the task `AMBER`.

Getting this distinction wrong would put an unsupportable causal claim in the
paper, and it is exactly the kind of claim a reviewer takes apart.

## What you may assume

- J3 saved per-case rollout output and metrics (`results/J3_result.md`).
- `02_SCIENTIFIC_CONTRACT.md` §6.1 step 5: the 120-day mean removal **is** the
  Wheeler–Hendon ENSO filter — it removes the interannual mean-state projection.
  It does **not** remove ENSO's modulation of MJO *behaviour*, which is what this
  task measures. Necessary, not sufficient; say so.
- **The splits are not ENSO-balanced and this matters.** 2020–2023 spans a
  triple-dip La Niña, one of three on record. 2016 is the decay of the very strong
  2015–16 El Niño. Validation (2016–2019) is reasonably mixed; the test period is
  not. **Document this now**, even though the test years are quarantined
  (`04_AGENT_PROTOCOL.md` §4) — it is a stated limitation of every future claim.
- Sample sizes are small. 36 training years hold roughly 10–12 El Niño events.
  Split by flavour (see step 3) and you have ~5–6 each. **This is underpowered for
  significance testing and adequate for compositing.** Do not report a
  significance claim you cannot support.

## Steps

1. Source the **Oceanic Niño Index** (ONI, 3-month running mean of Niño 3.4 SST
   anomaly) for 1980–2019. Commit it to `data/reference/oni.csv` with provenance
   alongside, exactly as J2 did for the BoM series. Write `scripts/fetch_oni.py`
   so the fetch is reproducible, and record a checksum.
2. Classify each month by the conventional ONI threshold: El Niño ≥ +0.5,
   La Niña ≤ −0.5, neutral in between, with the standard five-consecutive-overlapping-
   seasons requirement for an *event*. **Report both the per-month state and the
   event classification** — they differ, and which one a metric is conditioned on
   changes the answer.
3. Classify El Niño events by flavour — **Eastern Pacific (canonical) vs Central
   Pacific (Modoki)** — using a published index (Niño3 vs Niño4 relative
   magnitude, or the E/C indices). Record which method and cite it. **Treat this
   as a compositing stratification only**; state the event count per flavour and
   say plainly that it is underpowered for skill significance.
4. Build the **regime inventory**: for train (1980–2015) and validation
   (2016–2019), the fraction of days in each ENSO state, each season, and each MJO
   phase. Write it to `docs/findings/2026-1x-regime-inventory.md`.
   **Flag any regime that is under-represented in training** — that is the
   generalisation risk, and it is cheaper to know now than to discover in the
   paper.
5. Stratify every J3 metric — bivariate ACC, RMM1/RMM2 RMSE, amplitude ratio,
   phase error — by:
   - **ENSO regime**: El Niño / La Niña / neutral
   - **Season**: boreal winter (NDJFMA) vs boreal summer (MJJASO)
   - **Initial MJO phase**: all eight, with phases 2–3 called out because those are
     the Maritime Continent crossing cases (J6)
   Report the **case count in every cell**. A cell with fewer than ~20 cases gets a
   number and an explicit "underpowered" label, not a confidence interval.
6. Apply the same stratification to the **J4 zero-shot control**, so every Phase K
   comparison is like-for-like within regime.
7. **The boreal summer caveat.** RMM is an all-season index, but the boreal summer
   intraseasonal oscillation propagates northward over the Asian monsoon region and
   is poorly represented by RMM; dedicated BSISO indices exist for it. If summer
   skill is markedly worse than winter, **that may be an index artefact rather than
   a model failure.** State this explicitly rather than reporting a summer skill
   deficit as a model result. BSISO indices are **out of scope** and are a
   follow-on item.
8. Add the stratification to `docs/evaluation-spec.md` and to J3's output schema,
   so it runs automatically on every future evaluation.

## Definition of Done

- [ ] ONI sourced, committed to `data/reference/oni.csv` with provenance, licence,
      retrieval date and checksum; `scripts/fetch_oni.py` reproducible
- [ ] Monthly ENSO classification implemented; **per-month state and event
      classification both reported**, with the difference explained
- [ ] EP vs CP flavour classification implemented; method cited; **event count per
      flavour stated with an explicit underpowered label**
- [ ] Regime inventory for train and validation: ENSO state, season and MJO phase
      fractions; under-represented regimes flagged
- [ ] **Split imbalance documented**, including the 2020–2023 triple-dip La Niña
      and the 2015–16 El Niño decay in early validation
- [ ] All J3 metrics stratified by ENSO regime, season and initial phase; **case
      count in every cell**; cells under ~20 cases labelled underpowered
- [ ] Same stratification applied to the J4 control
- [ ] Boreal summer / BSISO index caveat stated explicitly
- [ ] **G5 dependency stated in the first line of the result file**: does this
      measure ENSO-conditional model behaviour, or regime-dependent initial-condition
      predictability?
- [ ] `docs/findings/2026-1x-regime-inventory.md` written
- [ ] Stratification added to `docs/evaluation-spec.md` and the J3 JSON schema
- [ ] `data/rmm_basis.npz` unchanged; `git diff --stat` pasted
- [ ] All citations checked or flagged unverified per **Q-12**
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] `docs/PROJECT_STATE.md` updated
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Training on an ENSO-stratified curriculum.** See **Q-21** for the argument.
  Short version: it halves the training set, invites catastrophic forgetting on a
  0.5% trainable surface, and without G5 it trains on a label the model cannot
  see. The neutral-only-training ablation is a **follow-on campaign** item and is
  recorded there.
- Reweighting the sampler by regime. Measure the imbalance; do not act on it.
- Implementing BSISO indices.
- Reading test years. The 2020–2023 imbalance is documented **from the ONI record**,
  not from model output or ERA5 fields.
- Any significance claim on a cell with fewer than ~20 cases.
