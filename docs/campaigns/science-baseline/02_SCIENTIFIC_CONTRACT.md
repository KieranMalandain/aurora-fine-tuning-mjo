# Scientific contract

**Binding.** The scientific decisions this campaign runs on, each with the
argument that produced it. This is the analogue of `refactor`'s
`02_UPSTREAM_CONTRACT.md`: a settled question here is settled, and re-opening one
costs a `Q-nn` and a human answer, not an agent's judgement.

`00_CONTEXT.md` §3 is the evidence. This file is the decisions.

Three of these were decided by the human on 2026-09-13 and are marked
**HUMAN-DECIDED**. Do not re-litigate those; implement them.

---

## 1. The target — revised, deterministic, single-member

### 1.1 What changed and why

**HUMAN-DECIDED (2026-09-13):** this campaign is **single deterministic**. No
ensembling of any kind — not multi-seed, not lagged-initialisation, not
perturbed-input. Ensemble work is deferred to follow-on research with Aurora
v1.5, which is ensemble-capable by design.

The inherited target was **bivariate ACC > 0.5 at 30-day lead**
(`../refactor/00_CONTEXT.md` §1, `docs/evaluation-spec.md`). That number was
taken from the operational and ML literature, where the systems achieving it are
**ensemble means**. An ensemble mean has a structural advantage on a correlation
metric: averaging suppresses the unpredictable component, so the mean correlates
with truth better than any member does. Carrying an ensemble-mean number over to
a single deterministic member sets a target that is not the same problem.

**The 30-day target is retired for this campaign.** It is not abandoned as a
research goal — it is the right target for the v1.5 ensemble work, and saying so
in the paper is stronger than quietly missing it.

### 1.2 The targets that replace it

The primary targets are **relative and self-owned**, because those are the ones
this campaign can definitively measure and defend.

| ID | Target | Measured by | Status |
| --- | --- | --- | --- |
| **T1** | Fine-tuned Aurora beats **zero-shot Aurora** in bivariate ACC at every lead from 5 to 30 days | J4 control vs K2–K4 | **PRIMARY.** This is the campaign's definition of success. |
| **T2** | Fine-tuned Aurora beats **damped persistence** at every lead ≥ 5 days | J3 baselines | **PRIMARY.** A floor. Failing it means something is wrong, not that the method is weak. |
| **T3** | Bivariate ACC > 0.5 at **≥ 20 days**, single member, conditioned on `A(t₀) > 1`, on 2016–2019 | J3 | **SECONDARY.** Stretch: ≥ 25 days. |
| **T4** | Mean forecast amplitude ratio `Â/A_obs` ≥ 0.6 at day 20 | J3 | **SECONDARY.** See §1.4. |
| **T5** | Eastward/westward power ratio in the MJO band ≥ 1.5 for the best checkpoint | J6 | **PRIMARY.** The model must *propagate*, not oscillate in place. See §1.6. |
| **T6** | T1 holds within each ENSO regime, not only in the all-regime average | J7 | **SECONDARY**, and interpretable only if SST is an input. See §2.5. |

**T1 is the one that matters.** "Does parameter-efficient fine-tuning of a
pretrained Earth-system foundation model buy MJO skill over the same model
unmodified?" is a controlled comparison in which this project owns both sides,
needs no external baseline to be reproducible, and is publishable whichever way
it comes out. It is also the question the May research script actually poses
(§II-D: domain adaptation vs emulation).

T3 is the number a reader will look for, so it is stated — but it is secondary
and it is a range, not a claim.

### 1.3 Literature comparison — **must be verified, not assumed**

The campaign will want to position T3 against published numbers. **No agent may
write a literature comparison number into a result file, a plot or a doc without
checking it against the paper itself.** The references in
`docs/papers/aurora-mjo-fall-paper.pdf` §REFERENCES are the starting point:
Shin et al. 2024 [5], Wang et al. 2025 [7], Li et al. 2026 [12], Cao et al. 2025
[14], and Suematsu et al. 2024 [9] for the methodology critique.

What must be recorded for every number cited: **ensemble or deterministic**, the
number of members, the verification period, and whether metrics were conditioned
on initial amplitude. A skill number without those four facts is not comparable
to ours and must not be plotted next to ours. This is precisely the failure mode
Suematsu et al. document.

Raised as **Q-12**.

### 1.4 Amplitude damping is a first-class metric, not a secondary one

A deterministic model trained on a pointwise reconstruction loss regresses toward
the conditional mean. Over a 120-step rollout that shows up as **RMM amplitude
decay**: the forecast stays in the right phase but shrinks toward the origin of
the phase diagram.

Bivariate ACC is a *normalised* correlation and is largely blind to this. A
forecast whose amplitude has collapsed to 0.2 can still score a high ACC. **A
campaign that reports ACC alone can report skill on a model that has stopped
forecasting the MJO.**

Therefore `Â/A_obs` versus lead is promoted from `docs/evaluation-spec.md`'s
secondary list to a required output of every evaluation run (task J3), and T4
exists to make it a pass/fail rather than a plot.

### 1.5 One upside worth stating

RMM is a projection onto the two leading EOFs of a meridionally averaged,
zonally structured field — i.e. onto planetary zonal wavenumbers 1–3. Small-scale
blurring, which is the standard failure mode of deterministic AI weather models
past ~10 days and which `docs/papers/` §IV-C observes at Day 10, is **filtered
out by the projection**. Gridded RMSE and RMM skill can therefore diverge, and a
model that looks visually degraded may retain RMM skill.

This is a reason to trust RMM over eyeballed OLR maps, and a reason not to over-
interpret the Day-10 "grittiness" figure. It is **INFERRED**, it is testable, and
J3 should test it: report gridded tropical RMSE and RMM ACC on the same axis.

### 1.6 The standing-wave problem — why ACC alone cannot establish an MJO forecast

RMM is a projection onto two EOFs. A **standing** oscillation — tropical
convection amplifying and decaying in place without moving east — excites those
two EOFs in quadrature and is, in the 2-D phase space, **indistinguishable from
propagation**. A model with no MJO can therefore post a respectable bivariate ACC.

This is not a hypothetical failure mode. It is the best-documented MJO deficiency
in GCMs: models produce MJO-band variance that stalls at the Maritime Continent
barrier rather than crossing into the West Pacific.

It is also the failure mode *this* model is most likely to exhibit. A
deterministic model trained on a pointwise reconstruction loss regresses toward
the conditional mean, and the conditional mean of an ensemble of propagating
events — averaged over propagation-speed uncertainty — is a **standing** pattern.
Amplitude damping (§1.4) and propagation loss are two expressions of the same
regression pressure, and the primary metric is blind to both.

**Target T5 therefore exists and is PRIMARY.** The measurement is Wheeler–Kiladis
space–time spectral decomposition of tropical OLR and `u850` into eastward- and
westward-propagating power, integrated over zonal wavenumbers 1–3 and periods
30–60 days. Observations run roughly 2–3× eastward-dominant
(`03_DOMAIN_PRIORS.md` §10). Below **1.5** for the best Phase K checkpoint, the
campaign has not produced an MJO forecast whatever the ACC says, and the result
file must say so in its first line. Task **J6**, and it is a gate.

Two supporting diagnostics, both in J6, because the spectral ratio is a
whole-record statistic and does not localise a failure: the **net RMM
phase-advance rate** (a propagating MJO advances monotonically at ~7.5° day⁻¹; a
standing oscillation has a near-zero net rate with sign reversals, and the
*fraction of cases with a sign reversal* is the cleaner signature), and the
**Maritime Continent crossing fraction** for cases initialised in phases 2–3.

---

## 2. Model and checkpoint

**HUMAN-DECIDED (2026-09-13):** production is the **1.3B model**; the small model
is retained for debugging.

### 2.1 The mapping, which is not the current mapping

| `model_type` | Class | Checkpoint | Use |
| --- | --- | --- | --- |
| `small` | `AuroraSmallPretrained` | `aurora-0.25-small-pretrained.ckpt` | Debug, smoke tests, CI, fixtures. **Never a reported result.** |
| `full` | `AuroraPretrained` | `aurora-0.25-pretrained.ckpt` | **Every production run.** |
| ~~`huge`~~ | ~~`Aurora`~~ | ~~`aurora-0.25-finetuned.ckpt`~~ | **Retired. Config must hard-fail on this string.** |

### 2.2 Why `"huge"` must hard-fail rather than be renamed

`model.py:424` currently maps `"huge"` → `Aurora`. `Aurora.default_checkpoint_name`
is `aurora-0.25-finetuned.ckpt` (**MEASURED**, `microsoft-aurora==1.8.0`), which
is the model fine-tuned on **IFS HRES analysis** for operational medium-range
forecasting — a different input distribution from ERA5 reanalysis.

Nobody has run `"huge"`, so no result is contaminated. But the string is sitting
in the code as a one-word config change away from silently loading the wrong
weights and producing a plausible, wrong, expensive result. A `ValueError` at
config-validation time costs nothing and removes the failure mode permanently.

`AuroraPretrained` is the ERA5-appropriate base and is what `full` selects.

**Unverified:** that `AuroraPretrained` is what Microsoft's own ERA5 example uses.
It is strongly implied by the class split and the checkpoint names, and the
review could not construct the 1.3B model to confirm (container OOM). **G2 must
confirm it** against `https://microsoft.github.io/aurora/example_era5.html`
before the first production run. Raised as **Q-13**.

### 2.3 Parameter counts

| Model | Total parameters | Provenance |
| --- | ---: | --- |
| `AuroraSmallPretrained`, 6 surf vars | **112,830,384** | **MEASURED** — review 2026-09-13, and B1/C4/F2 fingerprints |
| `AuroraPretrained`, 6 surf vars | ~1.3 × 10⁹ | **DERIVED** — `Aurora` docstring: *"Defaults to the 1.3 B parameter configuration."* **G2 must measure the exact integer.** |

### 2.5 Aurora v1.0 has no ocean, and that bounds every ENSO claim

Measured from `microsoft-aurora==1.8.0`:

```python
surf_vars   = ("2t", "10u", "10v", "msl")      # this project adds ttr, tcwv
static_vars = ("lsm", "z", "slt")
atmos_vars  = ("z", "u", "v", "t", "q")
```

**No SST, no skin temperature, no ocean state of any kind.** The LANL archive
documentation contains zero mentions of SST (`docs/nersc-dataset-information.md`,
grepped 2026-09-13), so the field is not merely unused — it is unavailable.

Over a 6-hour step this is immaterial. Over **120 steps SST is the dominant
boundary forcing on tropical convection**, and it is the only field that encodes
which ENSO state the forecast is in. The atmospheric initial condition
decorrelates in roughly 10–15 days; beyond that an ocean-free rollout relaxes
toward the model's own, ENSO-agnostic, climatology.

The consequence is sharp and it is easy to get wrong in a paper: **without an
ocean, a regime-stratified skill result measures regime-dependent predictability
of the initial condition, not learned ENSO-conditional model behaviour.** Those
are different claims and only one of them is interesting.

**Decision: SST enters as a persisted static** — initialised from observation at
`t₀` and held fixed through the forecast. Task **G5**. Three reasons it is a
static rather than a surface variable: it matches the S2S convention of
persisting SST anomalies at sub-seasonal lead (defensible to ~30 days, which is
where this campaign stops); it needs no randomly initialised decoder head, so it
adds no new drift source over 120 steps; and it cannot wander, which a prognostic
SST under an ocean-free model certainly would.

**Stated limitation, in `docs/SPEC.md` and in the paper:** SST is persisted, not
predicted; there is no ocean dynamics and no coupling; every claim is bounded at
30 days for this reason. Damped-persistence SST is the obvious refinement and is
**follow-on work** (§7).

### 2.4 The rule that stops the R4 failure recurring

- `--smoke-test` **forces** `model_type: small` regardless of config.
- Any run launched from `slurm_scripts/train_auto.slurm` (or any successor
  production script) that resolves to `model_type: small` **aborts at startup**.
- The resolved `model_type` and the resolved checkpoint filename are written into
  every checkpoint and every `metrics.jsonl` header.

Implemented in G2, tested in G2, enforced by config validation
(`aurora_mjo/config.py`, Pydantic, `extra="forbid"` — already in place from E2).

---

## 3. The moisture-budget loss — the argument, then the decision

**HUMAN-DECIDED (2026-09-13):** replace with ERA5-supervised `E − P`. Both cases
are recorded here because the ablation in a later campaign will need them, and
because a decision without its rejected alternative is not reviewable.

### 3.1 The case for retiring the term entirely

1. Aurora already encodes moisture transport implicitly from pre-training on a
   corpus that includes ERA5, CMIP6 and IFS. A hand-rolled 13-level column
   integral on a 1° grid is a weak, noisy addition to something the backbone
   already does better.
2. The residual is dominated by the flux-divergence term, computed with central
   differences. Truncation error at 1° is large relative to the convective-scale
   signal it is meant to constrain.
3. It introduces λ_p, a hyperparameter with no principled value, and a documented
   failure mode (over-smoothing) that points directly away from the project's
   goal.
4. **Every term added is an ablation that must be run.** With a 24-hour queue
   turnaround and a three-month horizon, each extra term costs a full cell of the
   ablation grid that a later campaign has to pay for.
5. The vertical integral runs 50–1000 hPa with no surface-pressure masking, so
   the budget is wrong over land — including the Maritime Continent, which is
   exactly where MJO propagation fails in most models and exactly where the term
   would need to be trustworthy to be interesting.
6. "Physics-informed" is cheap to claim and expensive to defend. A reviewer will
   ask for the ablation, and a term that does not help is worse than no term.

### 3.2 The case for ERA5-supervised `E − P`

1. **It changes the term's type.** Unsupervised regularisation toward `R = 0` is
   what breaks it (R2). Supervising `R` against a known, non-zero, physically
   correct target makes it an auxiliary prediction task. The target has the right
   magnitude, so it *cannot* reward smoothing.
2. **It is an information channel Aurora does not otherwise have.** Precipitation
   is the MJO's defining observable and Aurora predicts neither E nor P. Fitting
   the implied `E − P` gives the model access to precipitation information
   without adding P as a prognostic variable and without touching the decoder's
   variable set.
3. It constrains **moisture–convection feedback specifically**, which is what the
   moisture-mode account of the MJO says the phenomenon *is*. This is the rare
   case where a physics term is motivated by the target phenomenon rather than
   generically.
4. **The data already exists**, in the archive this project already reads:
   `tp6h` (`Step06/…6hrAccu/TP6H`, accumulation over the previous 6 h, metres)
   and `mslhf` (`Step03/…6hrInst/meanSLHFLX`, mean surface latent heat flux over
   the previous 1 h, W m⁻²). `dataset.py`'s per-variable timestamp maps already
   handle heterogeneous chunking, so the marginal loader cost is low.
5. **It pays even at λ_p = 0.** Plotting predicted implied `E − P` against ERA5
   `E − P` is a physically interpretable diagnostic of whether the model's
   convection is in the right place with the right magnitude. That is a figure
   worth having whether or not the term is ever trained on.
6. It makes the paper's "physics-informed" claim true rather than aspirational.

### 3.3 The decision, and the constraints it carries

**Replace `mean(|R|)` with `mean(|R − (E−P)_ERA5|)` over the tropical band.**
The following are part of the decision, not implementation details:

- **Unit discipline.** `E = mslhf / L_v` with `L_v = 2.501 × 10⁶ J kg⁻¹`, sign
  convention checked against ERA5's downward-positive flux convention.
  `P = tp6h / Δt` with `tp6h` in metres of water and `Δt = 21600 s`, giving
  kg m⁻² s⁻¹. Both converted to mm/day (× 86400) to match the residual.
  **`tp1h` must not be used** — `docs/nersc-dataset-information.md` records that
  the archive is missing `e5.accumulated_tp_1h.202206.nc` and that the unit
  changes from m to kg m⁻² s⁻¹ at `202204`. `tp6h` is the safe field.
- **Surface-pressure masking.** The column integral is masked at `ps`; sub-surface
  levels are excluded, not extrapolated.
- **Diagnostic first.** The term ships at `weight: 0.0` and is logged as a
  diagnostic for the whole of K2 before it is ever given a non-zero weight. If
  the diagnostic shows the residual is dominated by numerical noise rather than
  by the `E − P` signal, the term is retired and that is a result.
- **Stage 3 only.** It is enabled in K4, warm-started from K3, with λ_p tuned so
  the term is ≤ 10% of the total loss at the start of training.
- **Mandatory ablation.** K4 is run with and without. A later campaign runs the
  weight sweep; this campaign runs the on/off.

Task **H3**.

---

## 4. The loss — normalised, weighted, area-correct

Binding shape. `L_grid` is computed on **normalised** values, never on physical
units (R1).

```text
L_grid = Σ_v  w_v · ( 1 / (L_v · H · W) ) · Σ_ℓ Σ_φ Σ_λ  c_ℓ · a(φ) · m(φ) · | x̂_vℓφλ − x_vℓφλ |
```

where, for every variable `v`:

- `x̂`, `x` are normalised by the **same training-period constants** the model
  normalises with (G3). Aurora denormalises its output, so both prediction and
  target are re-normalised before the loss. This is the single change that fixes
  R1.
- `w_v` is a per-variable weight, **config-visible**, defaults in §4.1.
- `c_ℓ` is a per-level weight, default `Δp_ℓ / Σ Δp`, which naturally emphasises
  the lower troposphere where the moisture is. `c_ℓ ≡ 1` for surface variables.
- `a(φ) = cos(φ) / mean(cos(φ))` is **area weighting**. The current loss has
  none, so at 1° a grid cell at 89.5° counts the same as one at the equator
  despite covering ~1% of the area. This is a defect independent of R1.
- `m(φ)` is the tropical emphasis, default 1.0 inside ±20° and 0.1 outside —
  the existing `tropics_weight` / `extratropics_weight`, preserved.

`a(φ)` and `m(φ)` are **separate**, applied multiplicatively. Collapsing them
into one mask loses the ability to change the tropical emphasis without also
breaking area weighting.

### 4.1 Default variable weights

Starting point, to be recorded in `configs/unified.yaml` and changed only with a
result file behind it:

| Variable | `w_v` | Reason |
| --- | ---: | --- |
| `ttr`, `tcwv`, `q` | 2.0 | The MJO variables. The whole point of the injection. |
| `u`, `v`, `t` | 1.0 | `u850` and `u200` are two of the three RMM fields. |
| `z`, `msl`, `2t`, `10u`, `10v` | 0.5 | Needed for a coherent atmosphere; not the target. |

These are a **prior, not a result**. The honest statement in the paper is that
they were set once, a priori, on the reasoning above, and never tuned against
validation skill. Tuning them against validation skill and then reporting
validation skill is leakage. If they are ever changed, the change and its
justification go in a result file and the paper says so.

### 4.2 What this breaks, deliberately

Every loss value in the repo's history becomes incomparable: the B1/C4/F2
fingerprints (7472.024414 / 7421.924805), the archived July traces in
`docs/archive/metrics/`, and Figures 2 and 6 of
`docs/papers/aurora-mjo-fall-paper.pdf`. That is expected and correct — the old
numbers measured `z` and `msl`. `04_AGENT_PROTOCOL.md` §5 says how to record the
discontinuity.

Task **H1**.

---

## 5. The training objective across stages

### 5.1 "Detached rollout" is pushforward training, and that is the choice

`backprop: "detached"` with `_advance_batch(..., detach=True)` is **pushforward
training**: each step is a single-step supervised problem whose input is the
model's own previous output. It corrects exposure bias — the mismatch between
training on reanalysis inputs and inferring on self-generated inputs — and its
activation memory is O(1) in `k`.

It is **not** backpropagation through time, and it does not optimise the
multi-step objective in `docs/papers/` Eq. 14. There is no gradient path from
step `k` back to step 1, so the model cannot learn to make an early step easier
for a later one.

**Decision: keep detached as the default and describe it accurately.** Full BPTT
is retained as a config option (`backprop: "full"`) and is **not** exercised in
this campaign. The reasoning: memory headroom freed by 1° ingestion is being
spent on the 1.3B model, not on BPTT, and pushforward is the better first
intervention against the specific failure mode observed (`docs/papers/` §IV-C:
noise accumulation over self-generated rollouts).

**Papers must not describe this as multi-step rollout training.** Task **K3**
owns the wording in `docs/SPEC.md`.

### 5.2 The train/evaluate horizon gap is real and is not closed by this campaign

Training rollout is `k ≤ 4` (24 hours). Evaluation is 120 steps (30 days). That
is a 30× extrapolation beyond anything that receives a gradient.

This is **stated, measured and not fixed**. The curriculum in K3 extends `k` as
far as the memory and wall-clock budget allow and records where it stopped. J3's
skill-versus-lead curve is the measurement of what the extrapolation costs. If
the curve collapses at the training horizon, that is a headline finding and
belongs in the synthesis, not in a panicked scope expansion.

### 5.3 Stage structure

Four stages, each warm-started from the previous, each producing a checkpoint and
an evaluation. Mode names in `configs/unified.yaml` follow.

| Stage | Mode | Trainable | Objective | Task |
| --- | --- | --- | --- | --- |
| 0 | `warmup` | Injected `ttr`/`tcwv` patch embeddings + their decoder heads + `msl` head only | Single-step, `L_grid` | K1 |
| 1 | `lora` | Stage 0 surface + LoRA adapters | Single-step, `L_grid` | K2 |
| 2 | `rollout` | As Stage 1 | Pushforward, `k` curriculum, `L_grid` | K3 |
| 3 | `physics` | As Stage 1 | Pushforward + `L_phys` | K4 |

Stage 0 is new and exists because of a specific risk: the injected embeddings and
their decoder heads are randomly initialised, so at step 0 they emit noise into a
loss that (after H1) they now meaningfully contribute to. Letting LoRA adapt the
backbone to that noise is how catastrophic forgetting happens. Stage 0 gets the
new channels to a sane operating point while the backbone is entirely frozen.

The MJO head is **not** in this table. See §6.5.

---

## 6. Evaluation — the RMM specification is binding

This section replaces `docs/evaluation-spec.md` §"RMM Targets" and the
"Implemented in `scripts/compute_rmm.py`" block beneath it, both of which specify
the 3-element vector (R3). **J1 updates `docs/evaluation-spec.md` to match this
section**; until it does, this section wins.

### 6.1 The procedure

Wheeler & Hendon (2004), in order. Deviations are permitted only where this
document names them.

1. **Fields.** OLR, `u850`, `u200`, daily mean from the 6-hourly archive.
2. **OLR sign.** `OLR = −mtnlwrf`. `mtnlwrf` is net top long-wave flux in the
   downward-positive convention, so it is negative and outgoing radiation is its
   magnitude. **The sign must be confirmed against the BoM series in J2**, not
   assumed — getting it wrong flips EOF1 and silently renumbers every phase.
3. **Meridional average**, 15°S–15°N, **simple unweighted mean** to match WH.
   The existing cos-latitude weighting (`rmm/compute.py:36`) is a deviation; over
   a band where `cos φ` runs 1.000 to 0.966 it is small, but it is a deviation and
   it is removed. Retain zonal structure: the output is `(time, longitude)`, not
   a scalar. **This is the fix for R3.**
4. **Seasonal cycle removal.** Mean plus the **first three harmonics** of the
   annual cycle, fitted per longitude, on the **training period only**. Not a raw
   day-of-year mean (`rmm/compute.py:56`), which is noisy at 36 samples per day.
5. **Interannual removal.** Subtract the mean of the **previous 120 days** at each
   longitude. This is the step that removes ENSO and it is currently absent
   entirely. §6.2 governs how it is applied to forecasts.
6. **Normalisation.** Divide each field by its **zonally averaged temporal
   standard deviation** — one scalar per field, from the training period.
7. **Concatenate** → `a(t) ∈ ℝ^(3·N_λ)`, `N_λ = 360` at 1°.
8. **EOFs** by SVD of the `(T_train, 1080)` training matrix. Take the first two.
   Do not form the 1080×1080 covariance matrix.
9. **PC normalisation.** Divide each PC by its training-period standard deviation,
   so RMM1 and RMM2 have unit variance and `A > 1` means what it means everywhere
   else.
10. **Sign and order convention.** `eigh`/SVD fix neither. Correlate the resulting
    series against the BoM reference, then flip signs and/or swap EOF1↔EOF2 as
    needed. **The applied transform is written into `rmm_basis.npz` and is part
    of the frozen basis.**

WH used 2.5° (144 longitudes); this project has 1° (360). The EOF structure is
planetary-scale so this should not matter. If the J2 gate fails, **coarsening to
2.5° before step 7 is the first diagnostic to try**, not the twentieth.

### 6.2 The 120-day mean is the Suematsu trap — read this before implementing

Step 5 is where MJO skill evaluations go wrong, and it is the subject of
Suematsu et al. 2024 [9], which this project's own paper cites while making a
related error elsewhere.

For an **observed** series the 120-day window is unambiguous: the 120 days before
the valid time, all observed.

For a **forecast** at lead τ initialised at `t₀`, the window `[t − 120d, t]`
straddles the initialisation. Two conventions exist and they are not equivalent:

- **(A) Observations only, up to `t₀`.** The 120-day mean is computed from
  observed data ending at `t₀` and held fixed across the whole forecast. This is
  real-time-realisable and is the conservative choice.
- **(B) Mixed.** Observed data up to `t₀`, forecast data after. Closer to what an
  operational centre does; more moving parts.

**Decision: convention (A)**, for both forecast and observation, for every lead.
Applying (A) to the observation too keeps forecast and truth on the same footing.
It is written into the basis file and stated in the paper.

Using **any** data after the forecast valid time in either series is leakage and
invalidates the run. There is no exception for "just the climatology".

### 6.3 The leakage contract

Carried forward unchanged from `scripts/compute_rmm.py`'s existing contract,
which was already correct, and extended:

- Seasonal harmonics, zonal-mean standard deviations, EOF basis and PC
  normalisation constants are fitted on **1980–2015 only**.
- Validation (2016–2019) and test (2020–2023) are **projected onto the frozen
  basis** and never used to fit any statistic.
- **Test years are not touched by this campaign at all.** Not for diagnostics,
  not for a sanity plot, not for a single sample. Every number in this campaign
  comes from 2016–2019. `04_AGENT_PROTOCOL.md` §4.
- Variable weights `w_v` (§4.1) are set a priori and not tuned against validation
  skill.

### 6.4 Baselines that must be computed

A skill curve with no baseline on it is not interpretable. J3 produces all four
on the same axes:

| Baseline | Definition |
| --- | --- |
| **Persistence** | `RMM(t₀)` held constant |
| **Damped persistence** | `RMM(t₀) · exp(−τ/τ_d)`, `τ_d` fitted on training years |
| **Climatology** | `RMM ≡ 0` — the ACC floor |
| **Zero-shot Aurora** | Un-fine-tuned `AuroraPretrained`, same rollout, same pipeline (J4) |

### 6.7 One index is not enough, and RMM is the wind index

**RMM's variance is dominated by the circulation fields**, `u850` and `u200`; OLR
contributes comparatively little. That is a documented property of the index
(Straub 2013 — **unverified, Q-12**), and it is a direct threat to this project's
premise.

The whole thesis is that injecting OLR and TCWV buys MJO skill *through
convection*. If we optimise on a loss that weights `ttr` and `tcwv` heavily and
then evaluate on an index carried by winds, a positive result **cannot
distinguish** "the convective injection worked" from "the model got better at
tropical winds". That is a confound sitting at the centre of the campaign's
headline claim.

**Decision: evaluate on two indices.** RMM stays primary, because it is what the
literature reports and what the BoM gate anchors. **OMI** (Kiladis et al. 2014 —
**unverified, Q-12**), computed from OLR alone, is added as a second, convectively
weighted index. The disambiguation is clean:

| RMM | OMI | Reading |
| --- | --- | --- |
| improves | improves | Real MJO improvement. The claim holds. |
| improves | flat | Winds improved; the convective injection is not doing the work. |
| flat | improves | Convection improved but the circulation coupling did not follow. |

The marginal cost is near zero — both are projections onto fields J3 already
produces. **The one hard constraint: OMI conventionally uses a 20–96 day bandpass
filter, which is non-causal. Applied to a forecast that is leakage.** Use the
real-time variant (ROMI) or a strictly backward-looking filter, and state exactly
what was done. Raised as **Q-22**.

### 6.8 The basis must be validated by pattern, not only by correlation

The J2 gate correlates against BoM, which is the right acceptance test but tells
you nothing about *why* when it fails. Two additions, both required outputs:

1. **EOF spatial maps.** EOF1 and EOF2 must show the canonical structure —
   enhanced convection over the Maritime Continent and the Indian Ocean
   respectively — and the phase numbering must place phases 2–3 over the Indian
   Ocean, 4–5 over the Maritime Continent, 6–7 over the West Pacific. If the maps
   are wrong, the sign/order transform is wrong, and this is the diagnostic that
   says so. Required figure.
2. **Phase composites of the forecast.** At predicted phase *N*, does the model's
   OLR and `u850` anomaly field resemble the canonical phase-*N* composite? A model
   can get RMM1 and RMM2 right while producing a spatially wrong field, and the
   scalar index will not notice. This is the "does it look like the MJO" check and
   it is the figure a reviewer turns to first.

### 6.9 Model drift aliases into the anomaly — use a lead-dependent climatology

A problem created by convention (A) in §6.2 and not solved by it.

The 120-day mean is computed from observations up to `t₀` and **held fixed across
the forecast**. Any systematic drift the model develops after `t₀` — a warm bias,
a moist bias, a slow relaxation toward its own climatology — is therefore **not
removed** and projects directly onto the anomaly, manufacturing RMM signal that is
model bias rather than MJO.

Standard S2S practice verifies against a **lead-dependent model climatology**:
the mean forecast state at each lead, computed over training-period hindcasts,
subtracted before the anomaly is formed. That removes drift by construction.

**Decision:** J3 reports skill **both ways** — against the observed climatology
and against a lead-dependent model climatology built from training-year
initialisations — and reports the difference. If the two differ materially, drift
is contaminating the index and that is a finding, not a nuisance. Building the
model climatology costs training-year rollouts, so J3 states the GPU-hour
implication alongside its initialisation-sampling decision.

### 6.5 The MJO head is not enabled in this campaign

The head (`model.py:44`) is currently supervised against the R3 targets, i.e.
against a quantity that is not the RMM index. J1 fixes the targets, which makes
the head *trainable* — but enabling it adds a loss term, a hyperparameter and an
ablation to a campaign whose job is to establish a baseline.

**Decision: keep `mjo_head.enabled: false` through Phase K.** J5 re-points its
targets, verifies it trains on the corrected targets in a smoke test, and stops.
Enabling it is the first task of the follow-on campaign, where it has a baseline
to be measured against — which is exactly the comparison the May research script
§III-E was designed to make and could not.

---

## 7. Summary of what a later campaign inherits

Recorded here so the synthesis task does not have to reconstruct it:

- Ensembling and Aurora v1.5 (§1.1).
- The 30-day target (§1.1).
- MJO-head enablement and the post-hoc-projection vs latent-prediction comparison
  (§6.5).
- Full BPTT rollout (§5.1).
- λ_p weight sweep, LoRA rank and insertion-point sweeps, `w_v` sensitivity
  (§3.3, §4.1).
- Closing the train/evaluate horizon gap (§5.2).
- **Damped-persistence SST** (relaxing the anomaly toward climatology with lead),
  and any form of ocean coupling (§2.5).
- **The ENSO-stratified training ablation**: train on ENSO-neutral years only,
  evaluate on El Niño and La Niña, and measure the skill drop. That quantifies how
  regime-specific the learned behaviour is. It is a *diagnostic ablation*, not a
  training strategy — **Q-21** has the argument.
- **BSISO indices** for the boreal-summer northward-propagating mode, which RMM
  represents poorly (J7 step 7).
- EP vs CP El Niño skill decomposition at a sample size that can support it.
