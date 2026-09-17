# Questions

Agents **append**. The human answers **in place** — do not move or renumber
existing entries.

Numbering continues from `../refactor/QUESTIONS.md`, which ended at Q-11. Do not
reuse Q-01–Q-11.

Format:

```markdown
## Q-nn — <one-line question>

**Raised by:** <TASK_ID>, <YYYY-MM-DD>
**Blocks:** <what cannot proceed, or "nothing — proceeding on the default">
**Proposed default:** <what the agent will do absent an answer>

**ANSWER (human, YYYY-MM-DD):** <answer>
```

Proposing a default is not optional. A question with no default blocks the
campaign on the human's inbox.

**Q-12 to Q-22 were raised when the campaign was written.** Most carry a default
the task files already assume. **Q-13 and Q-15 are marked HARD-BLOCKING** — they
would change a number that goes in the paper, so `04_AGENT_PROTOCOL.md` §8 says
to stop rather than proceed on the default.

---

## Q-12 — Who verifies the literature skill numbers, and against what?

**Raised by:** campaign authoring, 2026-09-13
**Blocks:** nothing until J3 plots a comparison; then it blocks the plot.

`03_DOMAIN_PRIORS.md` §9 gives approximate ACC = 0.5 crossings for operational
and ML systems. **Every one of those is from memory and none has been checked
against its source paper.** `02_SCIENTIFIC_CONTRACT.md` §1.3 forbids an agent
writing an unverified number into any output.

Four facts are needed per citation: ensemble or deterministic, member count,
verification period, and whether metrics were conditioned on initial amplitude.
Without all four the number is not comparable to ours, which is the exact failure
Suematsu et al. 2024 [9] documents.

**Proposed default:** J3 plots **only** the four baselines it computes itself
(persistence, damped persistence, climatology, zero-shot Aurora). Literature
numbers appear in **prose with the four facts stated**, never on the same axes,
until a human has checked them against the PDFs. The four-fact table is built
during the paper campaign, not this one.

**ANSWER (human, YYYY-MM-DD):**

---

## Q-13 — Is `AuroraPretrained` the correct base for ERA5 fine-tuning? **HARD-BLOCKING**

**Raised by:** campaign authoring, 2026-09-13
**Blocks:** G2, and therefore every production run in Phase K.

`microsoft-aurora==1.8.0` ships three relevant classes (**MEASURED**):

| Class | Default checkpoint |
| --- | --- |
| `Aurora` | `aurora-0.25-finetuned.ckpt` |
| `AuroraPretrained` | `aurora-0.25-pretrained.ckpt` |
| `AuroraSmallPretrained` | `aurora-0.25-small-pretrained.ckpt` |

`model.py:424` currently maps `"huge"` → `Aurora`, i.e. the **IFS HRES
analysis** fine-tune, which is a different input distribution from ERA5
reanalysis. `02_SCIENTIFIC_CONTRACT.md` §2 argues `AuroraPretrained` is the
correct ERA5 base, but the review could not confirm it against Microsoft's own
ERA5 example (no network; the 1.3B model also OOM'd on construction).

This is hard-blocking because loading the wrong base produces a plausible,
wrong, expensive result that nothing downstream would catch.

**Proposed default:** none — **stop and confirm.** G2 checks
`https://microsoft.github.io/aurora/example_era5.html` and the Aurora docs, and
records the confirming quote verbatim in its result file. If the docs are
ambiguous, open an issue on microsoft/aurora as was done for the 1° question
(#184) and mark G2 `BLOCKED`.

**ANSWER (G2, 2026-09-17):**
Confirmed against official Microsoft Aurora documentation (`https://microsoft.github.io/aurora/example_era5.html`).
The tutorial explicitly notes:
> *"The fine-tuned version of Aurora specifically only works with IFS HRES T0, so we use the non-fine-tuned version of Aurora in this example."*

In `microsoft-aurora==1.8.0`, `AuroraPretrained` is the non-fine-tuned foundation model (default checkpoint: `aurora-0.25-pretrained.ckpt`), whereas `Aurora` is the IFS HRES analysis fine-tune (default checkpoint: `aurora-0.25-finetuned.ckpt`). Therefore, `AuroraPretrained` is confirmed as the correct ERA5 base model. Model mapping updated in `model.py:424` for `model_type: full`.

---

## Q-14 — Confirm Aurora v1.0 and no ensembling for this campaign.

**Raised by:** campaign authoring, 2026-09-13
**Blocks:** nothing — proceeding on the default.

Microsoft released Aurora v1.5 recently. This campaign pins
`microsoft-aurora==1.8.0` and uses v1.0 weights, single-member deterministic
throughout (`02_SCIENTIFIC_CONTRACT.md` §1.1, §6). Recorded here so the decision
is greppable rather than only living in a conversation.

Note the consequence: the inherited "ACC > 0.5 at 30 days" target is retired for
this campaign and replaced by T1–T4 (`02_SCIENTIFIC_CONTRACT.md` §1.2). If that
retirement is not acceptable, the whole target section needs rewriting before
Phase J starts, not after.

**Proposed default:** v1.0 via `microsoft-aurora==1.8.0`, single member, no
ensembling of any kind, T1–T4 as stated. v1.5 and ensembles are follow-on work.

**ANSWER (human, YYYY-MM-DD):**

---

## Q-15 — Where does the BoM reference RMM series come from, and where does it live? **HARD-BLOCKING**

**Raised by:** campaign authoring, 2026-09-13
**Blocks:** J2, which is the campaign's primary gate. Everything downstream of J2
is a number this pipeline produced; without the reference there is nothing to
check it against.

J2 requires `r > 0.95` against the official Wheeler–Hendon RMM series over
2016–2019. That series is a **new external dependency** — the first this project
has had outside the CFS archive and the HuggingFace checkpoints. It needs:

1. A source and a retrieval date recorded.
2. A durable home. **Not `/pscratch`** — the previous campaign lost its July logs
   to a purge and `slt_data.nc` is still sitting there (Q-07).
3. A commitment decision. The series is small (~4 numbers/day × 40 years ≈ 60k
   values, well under 1 MB as CSV), so committing it to the repo is viable and
   makes J2 reproducible offline and in CI.

**Proposed default:** fetch once, commit to `data/reference/rmm_bom.csv` with a
`README.md` alongside recording URL, retrieval date, and the exact column
semantics. Never re-fetch silently. If licensing prevents committing it, store it
under `/global/cfs/cdirs/m4946/…`-adjacent **project-owned** storage (never
inside the read-only archive) and commit the retrieval script plus a checksum.

**ANSWER (human, YYYY-MM-DD):**

---

## Q-16 — What is the LANL 1° grid convention?

**Raised by:** campaign authoring, 2026-09-13
**Blocks:** nothing — G1 reads it from the file. But it determines every latitude
weight in the loss and the values fed to Aurora's position encoding, so it must
be **recorded**, not inferred.

The archive is `remap_180x360MODIS`. Whether latitudes are cell-centred
(89.5 … −89.5) or something else is not recorded anywhere in this repo, and the
review had no CFS access. Both work — Aurora accepts 180×360 directly and crops
181→180 (**MEASURED**, `00_CONTEXT.md` R5).

The current code sidesteps this by hard-coding `linspace(90, -90, 720)`, which is
how the 0.2503°-spacing defect happened (`00_CONTEXT.md` R7.5).

**Proposed default:** G1 reads `lat` and `lon` from the NetCDF coordinates of a
reference file, asserts latitude is descending, asserts all eleven variables
agree, pastes the first and last five values of each into its result file, and
adds the values to `03_DOMAIN_PRIORS.md` §2.1. **No `linspace` for grid
construction, ever.**

**ANSWER (G1, 2026-09-17):**
Measured from CFS archive reference files across all eleven variables and invariants:
- All 11 variables match to float32 exactness.
- Grid is cell-centred regular 1.0°: 180 latitudes and 360 longitudes.
- Archive file coordinates: `lat` runs -89.5 to 89.5 (ascending); `lon` runs 0.5 to 359.5 (ascending).
- Raw archive latitudes: First 5 are `[-89.5, -88.5, -87.5, -86.5, -85.5]`, last 5 are `[85.5, 86.5, 87.5, 88.5, 89.5]`.
- Raw archive longitudes: First 5 are `[0.5, 1.5, 2.5, 3.5, 4.5]`, last 5 are `[355.5, 356.5, 357.5, 358.5, 359.5]`.
- Aurora convention: `Metadata` strictly enforces decreasing latitude (`lat[1:] - lat[:-1] < 0`), and Aurora expects North at row 0. Therefore, `dataset.py` inverts the latitude coordinate to descending (`89.5 ... -89.5`) and flips data arrays along the latitude axis (`axis=-2`) to maintain physical spatial consistency.
- Descending latitudes: First 5 are `[89.5, 88.5, 87.5, 86.5, 85.5]`, last 5 are `[-85.5, -86.5, -87.5, -88.5, -89.5]`.

---

## Q-17 — `slt` at 1°: regrid from the 0.25° file, or source it fresh?

**Raised by:** campaign authoring, 2026-09-13
**Blocks:** nothing — proceeding on the default.

Q-07 is still open in substance: `slt_data.nc` lives at
`/pscratch/sd/k/kam352/Aurora/slt/slt_data.nc`, which is purgeable, and
`scripts/download_slt.py` is the only record of how it was produced.

G1 makes it more urgent. `slt` is currently truncated `[:720, :]` from a 0.25°
grid; at 1° it must be **regridded**. It is categorical (integers 0–7), so
nearest-neighbour only — bilinear would produce soil type 3.7.

**Proposed default:** G1 regrids the existing 0.25° file to 1° by nearest
neighbour, writes the result to `data/static/slt_1deg.nc`, and **commits it** —
a 180×360 int8 field is ~65 kB. That removes the scratch dependency entirely and
closes Q-07 as a side effect. `scripts/download_slt.py` is retained as
provenance and gets a header saying what the committed file was derived from.

**ANSWER (human, YYYY-MM-DD):**

---

## Q-18 — What is the GPU budget for Phase K?

**Raised by:** campaign authoring, 2026-09-13
**Blocks:** the Phase K schedule. Not G, H or J.

Phase K is four sequential training stages plus evaluation. Task headers carry
GPU-hour budgets, but they were written without knowing the allocation, and
`03_DOMAIN_PRIORS.md` §6 flags the two numbers they depend on (`full` step time
and peak memory at 1°) as **DERIVED and WEAK** until G2 measures them.

The previous campaign's evidence: 11.5 h walltime per `train_auto.slurm` job,
~24 h observed queue latency, 4 × A100 80GB per node.

**Proposed default:** budget Phase K at **4 × A100 × 11.5 h per stage, one
SLURM chain per stage, four stages**, and treat that as the ceiling. If G2's
measured step time implies a stage cannot complete an epoch in that envelope,
that is a finding for G2's result file and a replan, not a silent extension.
Phase K tasks re-state their budget after G2 lands.

**ANSWER (human, YYYY-MM-DD):**

---

## Q-19 — Does `combined` mode get deleted or kept as a deprecated alias?

**Raised by:** campaign authoring, 2026-09-13
**Blocks:** nothing — proceeding on the default.

`01_TARGET_STATE.md` D6 retires `combined` (which was `lora` + `physics_informed`
run together) because Stage 3 subsumes it and the new chain is strictly linear.
Deleting a mode breaks any saved command line that used it, and there are
checkpoints under `checkpoints/combined/` from July.

**Proposed default:** **delete the mode** from `configs/unified.yaml` and let
Pydantic's `extra="forbid"` reject it with a clear error naming `physics` as the
replacement. Do not keep a silent alias — a silent alias means a July command
line runs and produces something that is not what it produced in July.
`checkpoints/combined/` is left untouched on disk (`04_AGENT_PROTOCOL.md` §3).

**ANSWER (human, YYYY-MM-DD):**


---

## Q-20 — Where does SST come from, given it is not in the LANL archive? **HARD-BLOCKING**

**Raised by:** campaign authoring, 2026-09-13 (revised after scientific review,
2026-09-16)
**Blocks:** G5, and therefore the interpretability of J7 and target T6.

`02_SCIENTIFIC_CONTRACT.md` §2.5: Aurora v1.0 carries no ocean variable, and
`docs/nersc-dataset-information.md` contains **zero** mentions of SST. The field
is not merely unused — it is unavailable in the archive this project reads.

Options, in decreasing order of preference:

1. **ERA5 `sst` sourced directly** from CDS or from another NERSC allocation,
   regridded to the G1 1° grid. Cleanest, and it is the field everyone else uses.
2. **ERA5 `skt` (skin temperature) masked to ocean by `lsm`.** Available in more
   archives; over ocean it is close to SST.
3. **`2t` over ocean as a proxy.** 2-metre air temperature is tightly coupled to
   SST over water, and we already read it. But it is *prognostic* in our setup, so
   using it as a persisted static would mean persisting a field the model is also
   predicting — confusing, and it would drift out of agreement with itself.

This is hard-blocking because the choice changes what T6 can claim, and because
(3) is superficially free and scientifically the worst.

**Proposed default:** none — **stop and decide.** If (1) is available, take it.
If only (3) is available, G5 should be **deferred** rather than done badly, and
J7 runs with the `AMBER` caveat in its result file. A bad ocean is worse than a
documented absence of one.

**ANSWER (human, YYYY-MM-DD):**

---

## Q-21 — Should training be staged by ENSO regime (neutral first, then El Niño / La Niña)?

**Raised by:** scientific review, 2026-09-16, following external advice
**Blocks:** nothing — proceeding on the default. But it is a real strategic
question and the human should overrule if they disagree.

The proposal is to train on ENSO-neutral years first and then introduce El Niño
and La Niña, so the model learns a base state before the regimes are "activated".

**The question behind it is the right question**: has the model learned
ENSO-conditional MJO behaviour, or memorised a climatological MJO? That deserves
an answer.

**Three arguments against the sequential curriculum as the instrument:**

1. **Sample cost.** Neutral is roughly 40–50% of months, so neutral-only training
   halves an already-modest 36-year record.
2. **Catastrophic forgetting.** Sequential fine-tuning across regimes on a 1.3B
   model with a ~0.5% trainable surface is a textbook forgetting setup — and
   avoiding catastrophic forgetting is the explicit reason for the
   frozen-backbone / LoRA design (`docs/papers/` §II-D).
3. **No input channel.** Without G5 the model has no field carrying ENSO state
   (`02_SCIENTIFIC_CONTRACT.md` §2.5). The curriculum would be training against a
   label the model cannot see.

**Proposed default: do not stage training. Instead —**

- Train once on all 36 years.
- **Stratify evaluation** by ENSO regime, season and initial phase (task **J7**).
- Give the model an ocean so the conditioning is physically possible (task **G5**).
- Run **neutral-only training as a deliberate ablation in the follow-on
  campaign**: train on neutral, evaluate on El Niño and La Niña, measure the skill
  drop. That answers the original question directly, with a control, and without
  compromising the baseline this campaign exists to produce.

**ANSWER (human, YYYY-MM-DD):**

---

## Q-22 — Which MJO index is primary, and how is OMI made causal?

**Raised by:** scientific review, 2026-09-16
**Blocks:** nothing — proceeding on the default. It changes what J3 and J6 report,
so answer before J3 fixes its output schema.

`02_SCIENTIFIC_CONTRACT.md` §6.7: RMM's variance is dominated by `u850` and
`u200`, not OLR. Since this project's premise is that injecting OLR and TCWV buys
skill *through convection*, evaluating on RMM alone cannot separate "the
convective injection worked" from "the model got better at tropical winds".

Adding OMI — an OLR-only index — resolves the confound at near-zero marginal cost,
since both are projections onto fields J3 already produces.

The complication: **OMI conventionally uses a 20–96 day bandpass filter, which is
non-causal.** Applied to a forecast it uses information from after the valid time,
which is leakage and invalidates the skill number — the same class of error as the
120-day-mean trap (`02_SCIENTIFIC_CONTRACT.md` §6.2).

**Proposed default:** **RMM primary** (it is what the literature reports and what
the J2 gate anchors), **OMI secondary and reported alongside**, computed with the
real-time variant (**ROMI**) or a strictly backward-looking filter. J3 documents
precisely which filter was applied and in which direction, and adds the same
leakage assertion it applies to the 120-day window. If a causal OMI cannot be
implemented cleanly, report RMM only and state the confound in the paper rather
than reporting a leaked second index.

**ANSWER (human, YYYY-MM-DD):**

---

## Q-23 — Decoupling historical B1 baseline fixture round-trip tests from evolving configs/unified.yaml

**Raised by:** G1, 2026-09-17
**Blocks:** nothing — proceeding on the default.

In campaign 1 (refactor), `tests/test_config_validation.py::test_b1_baseline_fixture_exact_round_trip` and `tests/test_config_modes.py::test_apply_overrides_locked_against_b1_fingerprint` were written to verify that `configs/unified.yaml` was byte-identical to `tests/fixtures/baseline/config_{mode}.json`.
In campaign 2 (science-baseline), `configs/unified.yaml` is intentionally modified on purpose (G1 points `slt_path` to `data/static/slt_1deg.nc`, G2 updates `model_type: full`, G3 updates `norm_stats.msl`), while `tests/fixtures/baseline/` must remain byte-identical historical records (`01_TARGET_STATE.md` §D9).
Comparing `configs/unified.yaml` directly against historical baseline fixtures causes all five tests to fail upon any intentional configuration modification.

**Proposed default:** Update `test_b1_baseline_fixture_exact_round_trip` to validate that the B1 baseline fixtures round-trip through Pydantic `Config.model_validate(expected_dict).to_dict() == expected_dict` directly, and update `test_apply_overrides_locked_against_b1_fingerprint` to apply overrides to `baseline_fingerprint['config_baseline']`. This ensures `tests/fixtures/baseline/` remains an immutable historical benchmark while allowing `configs/unified.yaml` to evolve across the science campaign.

**ANSWER (human, YYYY-MM-DD):**
