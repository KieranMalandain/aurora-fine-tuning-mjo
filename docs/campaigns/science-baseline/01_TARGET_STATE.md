# Target state

The exact end state, decision by decision, each with its rationale — so no agent
has to guess at intent and no settled question gets re-opened.

`02_SCIENTIFIC_CONTRACT.md` holds the *scientific* decisions and their arguments.
This file holds what the **repository** looks like when the campaign is done.
Where the two touch, the contract wins and this file points at it.

---

## D1. Native 1° is the working resolution. Both upsamplers are deleted.

**Decision.** `LANLMJODataset` returns 1° fields and `Batch.metadata` carries the
1° grid. `_upsample_to_aurora` (`dataset.py:85`) and `_upsample_batch_gpu`
(`trainer.py:155`) are removed, not disabled. The `needs_upsample` branch in
`_prep_batch` (`trainer.py:898`) goes with them.

**Why.** Aurora ingests 180×360 and 181×360 directly — measured, live forward
pass, `00_CONTEXT.md` R5 — and the Aurora team recommended exactly this
(microsoft/aurora#184). The current pipeline fabricates 15 of every 16 grid
points and then spends model capacity on the interpolation artefacts. It also
puts a seam at the dateline, because `F.interpolate(..., mode="bilinear")` does
not wrap in longitude.

It additionally retires a whole class of bug for free: the `linspace(90, -90, 720)`
grid at `dataset.py:153` is 0.2503° spacing, not 0.25°, and `slt` is truncated
`[:720, :]` from a genuine 0.25° grid — so `slt` currently sits on a different
grid from `z` and `lsm` (`00_CONTEXT.md` R7.5, and `../refactor/` PROJECT_STATE
§5 "Unverified Assumptions" item 1, which is hereby resolved by deletion).

**The grid is read from the archive, not constructed.** `lat` and `lon` come from
the NetCDF coordinate variables of a reference file, are asserted descending in
latitude, and are asserted to match across all eleven variables. A hard-coded
`linspace` is what produced the 0.2503° defect; it is not replaced with a
better-tuned `linspace`.

**`slt` is regridded to 1°, not truncated.** It is categorical (integers 0–7), so
it is regridded by **nearest-neighbour**, never bilinear. Bilinear interpolation
of a soil-type code produces soil type 3.7, which does not exist.

**How it is verified.** `tests/test_dataset_loader.py` asserts returned shapes are
`(1, 2, 180, 360)` and `(1, 2, 13, 180, 360)`; a new test asserts
`metadata.lat[0] > metadata.lat[-1]` and that the lat/lon arrays equal the
archive's to float32 exactness. G4 additionally checks physical ranges against
`03_DOMAIN_PRIORS.md` §2.

---

## D2. `model_type` is `small | full`. `huge` raises.

**Decision.** `configs/unified.yaml` production modes set `model_type: full` →
`AuroraPretrained`. `small` → `AuroraSmallPretrained` and is confined to smoke
tests, CI and fixtures. The string `huge` raises a `ValueError` in
`aurora_mjo/config.py`.

**Why.** `02_SCIENTIFIC_CONTRACT.md` §2, in full. Short version: every claim in
`docs/papers/` is about a 1.3B model; every run was the debug model; and the
existing `huge` path points at the HRES-fine-tuned checkpoint, which is the wrong
base for ERA5 input.

**Two guard rails, both enforced at startup, both tested.**

1. `--smoke-test` forces `small` regardless of what the config says.
2. A production SLURM launch that resolves to `small` **aborts**. Detection is on
   an explicit `training.require_full_model: true` key set in the production
   modes, not on sniffing the SLURM environment.

**Provenance is written down.** Resolved `model_type`, the resolved checkpoint
filename, and the `microsoft-aurora` version go into every checkpoint payload and
the header line of every `metrics.jsonl`. A results file that cannot say which
model produced it is not a result.

---

## D3. The loss is computed on normalised values, weighted, and area-correct.

**Decision.** `TropicalWeightedL1Loss` is replaced by a loss implementing
`02_SCIENTIFIC_CONTRACT.md` §4: per-variable normalisation, per-variable weights
`w_v`, per-level weights `c_ℓ`, cos-latitude area weighting `a(φ)`, tropical
emphasis `m(φ)`.

**Why.** `00_CONTEXT.md` R1. The current loss gives `msl` and `z` 99.03% of the
gradient and `q` 0.000012%.

**Per-variable loss components are logged.** The single scalar that hid this
problem for six months is the same scalar that will hide the next one. Every
logged step emits `loss/grid/<var>` for all eleven variables. This is cheap and
it is the difference between noticing and not noticing.

**Consequence, accepted deliberately.** Every historical loss number becomes
incomparable — B1/C4/F2 fingerprints, `docs/archive/metrics/*.jsonl`, and
Figures 2 and 6 of `docs/papers/aurora-mjo-fall-paper.pdf`. §D9 and
`04_AGENT_PROTOCOL.md` §5 say how that is recorded rather than hidden.

---

## D4. The moisture-budget term is supervised against ERA5 `E − P`.

**Decision.** `MoistureBudgetLoss` penalises `|R − (E−P)_ERA5|`, not `|R|`. The
loader gains `tp6h` and `mslhf`. The column integral is masked at surface
pressure. It ships at `weight: 0.0` and is logged as a diagnostic before it is
ever trained on.

**Why.** `02_SCIENTIFIC_CONTRACT.md` §3, including the case for retiring it
entirely, which was considered and rejected.

---

## D5. RMM retains zonal structure, and reproducing the BoM index is a gate.

**Decision.** `aurora_mjo/rmm/` is rebuilt to
`02_SCIENTIFIC_CONTRACT.md` §6: `a(t) ∈ ℝ^(3·N_λ)`, three harmonics, 120-day mean
under convention (A), zonal-mean-σ normalisation, SVD, unit-variance PCs, and a
sign/order transform anchored to the BoM series and frozen into the basis file.

`docs/evaluation-spec.md` is updated in the same task — it currently *specifies*
the 3-element vector, so the code is a faithful implementation of a wrong spec
and fixing only the code leaves a landmine.

**Why.** `00_CONTEXT.md` R3.

**The gate.** J2 requires `r > 0.95` against the official BoM RMM series over
2016–2019 for both RMM1 and RMM2. Below that, **the campaign stops** and the
human is told. Everything downstream of J2 is a number produced by this pipeline;
if the pipeline cannot reproduce a published index from the same fields, no
downstream number means anything.

The BoM reference series is a **new external dependency**. It must be fetched
once, committed or placed on durable project storage with recorded provenance,
and never re-fetched silently. Raised as **Q-15**.

---

## D6. Four training modes, renamed to match the stages.

**Decision.** `configs/unified.yaml` keeps its one-file, `--mode` shape — that
works and E2's Pydantic validation is built on it. The mode *set* changes:

| Old mode | New mode | Change |
| --- | --- | --- |
| `baseline` | `warmup` | Stage 0. Backbone fully frozen; only injected embeddings, their decoder heads, and the `msl` head train. |
| `lora` | `lora` | Stage 1. Single-step, LoRA on. Rollout **off** (it was on at `k≤4`). |
| — | `rollout` | Stage 2. **New.** Pushforward curriculum. |
| `physics_informed` | `physics` | Stage 3. Warm-starts from `rollout`, not from `baseline`. |
| `combined` | — | **Retired.** It was `lora` + `physics_informed` simultaneously; Stage 3 now subsumes it and the chain is linear. |

**Why.** The old chain had `physics_informed` warm-starting from `baseline` in
parallel with `lora`, so the physics term was being evaluated on a model that had
never been through rollout. The stages are now strictly sequential, each
warm-starting from the last, which is what makes the stage-to-stage deltas
interpretable.

**`init_from` is explicit and checked.** Each mode's `init_from` names the
previous stage's save directory, and startup **fails** if the checkpoint is
absent rather than silently cold-starting. Silent cold-start is how a "Stage 3"
result turns out to be a Stage 0 result.

---

## D7. The placeholder normalisation statistics become impossible to use.

**Decision.** G3 computes true statistics for all eleven variables plus `ttr`,
`tcwv` and `ps`, over 1980–2015, at native 1°, by Welford accumulation. They are
written to `configs/norm_stats_1980_2015.yaml` with a provenance header (date,
commit, sample count, file count) and referenced from `configs/unified.yaml`.

`tests/test_norm_stats.py::test_placeholder_stats_warning_or_failure` inverts
from `xfail` to a passing test that asserts the placeholder values are **absent**
from the config.

**Why.** `../refactor/` Lesson 1 and the OPEN-01 checklist item. The handoff
called this the single most critical next action; `00_CONTEXT.md` §5 explains why
it is third rather than first. It is still mandatory.

**Welford moves into the package.** `scripts/calc_norm_stats.py` currently holds
the accumulation logic in a non-importable script, flagged by D3 of the previous
campaign. It moves to `aurora_mjo/stats.py` and gets a unit test against a known
distribution. The script becomes a thin CLI, matching the `scripts/` convention
in `../refactor/01_TARGET_STATE.md` D2.

---

## D8. The two zero-coverage modules get tests.

**Decision.** `tests/test_loss.py` and `tests/test_rmm.py` are created. They do
not exist today (**MEASURED** — review 2026-09-13; `tests/test_mjo_head.py`
matches only on the string "RMM").

**Why.** The two modules with no test coverage are the two modules where the
review found the worst scientific defects. That is not a coincidence — it is the
mechanism. R1, R2, R3 and R6 all live in `loss.py` or `rmm/`.

**Minimum content, each with a hand-checkable expected value:**

`tests/test_loss.py`
- Per-variable gradient share is within tolerance of `w_v` when all normalised
  errors are equal — this is the direct regression test for R1, and it fails
  loudly on the current code.
- Area weights integrate to 1 over the sphere.
- Level weights sum to 1.
- Moisture residual of an analytic non-divergent flow with a known `E − P` returns
  that `E − P`.
- Spectral loss on a single sinusoid concentrates in the correct wavenumber bin —
  the direct regression test for R6.

`tests/test_rmm.py`
- Output of the tropical average has a longitude dimension of 360, not a scalar —
  the direct regression test for R3.
- A synthetic eastward-propagating wavenumber-1 signal produces RMM1/RMM2 in
  quadrature and a phase that advances monotonically through all eight phases.
- The 120-day window at a forecast valid time contains no timestamp after `t₀`.
- Fitting the basis on a train slice and projecting a val slice does not change
  the basis.

---

## D11. The model gets an ocean, as a persisted static

**Decision.** SST enters `static_vars`, initialised from observation at `t₀` and
held fixed through each forecast. Task **G5**.

**Why.** `02_SCIENTIFIC_CONTRACT.md` §2.5. Aurora v1.0 carries no ocean variable
at all, so over a 120-step rollout there is nothing anchoring the forecast to the
ENSO state it started in. Without it, target **T6** and task **J7** measure
regime-dependent predictability of the initial condition rather than learned
ENSO-conditional behaviour, and stating otherwise in a paper would be
unsupportable.

Static rather than surface because it matches the S2S persisted-anomaly
convention at sub-seasonal lead, needs no randomly initialised decoder head, and
cannot drift.

**Structural consequence.** `_load_static_vars` currently loads three
time-invariant fields once at `__init__`. SST varies between samples and is fixed
within a forecast, so it is read per sample in `__getitem__` — the first static
that is not loaded once. Keep the read fork-safe.

**Stated limitation.** `docs/SPEC.md` records that SST is persisted, not
predicted, that there is no ocean dynamics, and that this bounds every claim at
~30 days.

---

## D12. Two indices, and a propagation test

**Decision.** RMM stays primary; **OMI** is added as a second, OLR-only index
(**Q-22**). **J6** adds Wheeler–Kiladis propagation diagnostics as a gate on
target **T5**. **J7** stratifies every metric by ENSO regime, season and initial
phase.

**Why.** `02_SCIENTIFIC_CONTRACT.md` §6.7 and §1.6. RMM's variance is carried by
`u850` and `u200`, not OLR — so a project whose premise is convective injection
cannot establish its claim on RMM alone. And a standing oscillation is
indistinguishable from propagation in RMM space, so ACC alone cannot establish
that a forecast is an MJO forecast at all.

**Required figures**, all new and all in `docs/findings/`: EOF1/EOF2 spatial
maps; phase composites of the forecast against the canonical composites;
Hovmöller diagrams for phase 2–3 initialisations; the eastward/westward power
spectrum.

**Skill is reported against two climatologies** — observed, and lead-dependent
model climatology from training-year hindcasts — because model drift after `t₀`
otherwise aliases straight into the anomaly (`02_SCIENTIFIC_CONTRACT.md` §6.9).

---

## D9. What does not change

Listed so nobody improves it.

- `uv` as the sole package manager, the committed `uv.lock`, the pinned
  `microsoft-aurora==1.8.0`, Python 3.10.
- `uv run python scripts/check.py` as **the** gate.
- `run.py` as the CLI; `train.py` as the shim.
- `src/aurora_mjo/` package layout; `scripts/` run-never-imported discipline.
- The v3 per-variable timestamp maps in `dataset.py` (Lesson 5). D1 changes what
  is read, **not** how the timeline is built. A G1 that touches
  `_build_aligned_index` has exceeded its scope.
- The DDP-collective grad guard (Lesson 2), `HDF5_USE_FILE_LOCKING` at process
  start (Lesson 3), `StaticVarLoadError` (Lesson 4), the `USR1` trap and `-c 64`
  in the SLURM scripts.
- `gradient_checkpointing: false` — Lesson 6. G2 may **measure** whether it is
  still needed at 1°; only G2, and only with the IMA matrix probe, may change it.
- Data splits: 1980–2015 / 2016–2019 / 2020–2023.

**The historical fingerprints are preserved, not updated.**
`tests/fixtures/baseline/` and `tests/fixtures/postrefactor/` stay byte-identical.
G4 writes a **third** fixture set, `tests/fixtures/science-baseline/`, and the
result file states plainly that the numbers differ from B1 because the input grid
and the loss both changed on purpose.

---

## D10. What exists at the end

| Artefact | Task | What it is |
| --- | --- | --- |
| `configs/norm_stats_1980_2015.yaml` | G3 | True training statistics with provenance |
| `tests/fixtures/science-baseline/` | G4 | Physical-sanity fingerprint at 1° |
| `data/rmm_basis.npz` | J1 | Frozen EOF basis, harmonics, σ, sign transform |
| `data/rmm_targets.nc` | J1 | Observed RMM 1980–2019 (**not** 2020–2023) |
| `results/J2_result.md` | J2 | BoM reproduction correlations. **The gate.** |
| `results/J4_result.md` | J4 | Zero-shot Aurora skill curve. **The control.** |
| `checkpoints/{warmup,lora,rollout,physics}/` | K1–K4 | One checkpoint per stage |
| `docs/results/skill-curves-2026-1x.md` | K4 | ACC, RMSE, amplitude ratio, phase error vs lead, four baselines on the same axes, one line per stage |
| `docs/campaigns/science-baseline/handoff-*.md` | L1 | The exit document |
| Updated `docs/SPEC.md`, `docs/evaluation-spec.md` | J1, K3, L1 | Durable truth promoted out of this campaign |

The paper is **not** in this table. `00_CONTEXT.md` §6.
