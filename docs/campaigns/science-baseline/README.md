# Campaign: science-baseline — make the science right, then measure it

**Purpose.** Fix the four scientific defects the 2026-09-13 review found, build
an evaluation pipeline that can be trusted, and produce the first MJO skill
numbers this project has ever had that mean something.

The `refactor` campaign made the code correct-by-construction against its own
past behaviour. It did not, and could not, ask whether that behaviour was right.
`00_CONTEXT.md` §2 explains why its gates passed trivially. This campaign asks
the question it excluded.

**What it deliberately does not do.** No ablation studies. No ensembling. No
Aurora v1.5. No paper. The MJO head stays disabled. One run per training stage,
not a grid. `00_CONTEXT.md` §6 has the full list, and it is a list of things a
helpful agent must not implement.

**Audience.** Autonomous coding agents working one task at a time, plus the human
owner reviewing result files.

**Base branch.** `epic/science-baseline`, cut from `main` after the human merges
`epic/refactor`. Every task branches off it as `epic/science-<ID>-<slug>` and
merges back into it. **Nothing in this campaign touches `main`.**

---

## Read these in order, once, before your first task

| # | Doc | Why |
| --- | --- | --- |
| 1 | [`00_CONTEXT.md`](00_CONTEXT.md) | What is already true, what the review found, and why the previous campaign's priority order is superseded. |
| 2 | [`01_TARGET_STATE.md`](01_TARGET_STATE.md) | The exact end state of the repository, decision by decision. |
| 3 | [`02_SCIENTIFIC_CONTRACT.md`](02_SCIENTIFIC_CONTRACT.md) | The scientific decisions and the arguments behind them. **Binding.** |
| 4 | [`03_DOMAIN_PRIORS.md`](03_DOMAIN_PRIORS.md) | Numbers to sanity-check a measurement against. Priors, not truth. |
| 5 | [`04_AGENT_PROTOCOL.md`](04_AGENT_PROTOCOL.md) | How to run a task, the test-year quarantine, compute discipline. |

Also read [`QUESTIONS.md`](QUESTIONS.md) — Q-12 to Q-19 are pre-seeded. **Q-13
and Q-15 are hard-blocking**; the rest carry defaults the task files assume.

The predecessor campaign's docs are still live background, not archive material.
Read [`../refactor/handoff-2026-09-refactor.md`](../refactor/handoff-2026-09-refactor.md)
for the engineering history and [`../refactor/03_DOMAIN_PRIORS.md`](../refactor/03_DOMAIN_PRIORS.md)
§1, §3, §5–§8, which are carried forward unchanged.

Then read **the single task file you have been assigned**. Do not read ahead.

---

## The three gates

`refactor` had two gates and both were equivalence gates. These are correctness
gates. Each one stops the campaign if it fails.

> ### Gate 1 — **G4**: the physical-sanity fingerprint
>
> After G1 changes the input grid, equivalence with B1 is meaningless because the
> numbers are *supposed* to move. G4 replaces it: every field is checked against
> the physical ranges in `03_DOMAIN_PRIORS.md` §3, the sample counts are checked
> against §1 (which must **not** move), and a new fixture set is captured.
> **If G4 fails, Phase H and Phase J are both untrustworthy** — they would be
> built on a loader that is silently reading the wrong thing.

> ### Gate 2 — **J2**: the BoM reproduction gate
>
> `compute_rmm` must reproduce the official Wheeler–Hendon RMM series to
> `r > 0.95` on 2016–2019, both components. **If J2 fails, stop the campaign and
> report.** Everything after J2 is a number this pipeline produced; a pipeline
> that cannot reproduce a published index from the same fields cannot be used to
> claim anything. Do not fix forward past a red J2.

> ### Gate 3 — **J4**: the zero-shot control
>
> Un-fine-tuned `AuroraPretrained`, rolled out 120 steps, scored through the J1
> pipeline. This is the number every Phase K result is measured against and
> **nobody has it**. It is captured before any fine-tuning, exactly as B1 was
> captured before any refactoring. Target **T1** is defined relative to it.
>
> A J4 that produces surprisingly *high* skill is not good news — check the
> leakage list in `03_DOMAIN_PRIORS.md` §9 before celebrating.

> ### Gate 4 — **J6**: does the forecast propagate?
>
> A standing oscillation excites the two RMM EOFs in quadrature and is
> indistinguishable from propagation in the 2-D phase space, so **a model with no
> MJO can post a respectable ACC**. It is also the failure mode a deterministic
> model trained on a pointwise loss is most likely to have, because the
> conditional mean of an ensemble of propagating events is a standing pattern.
>
> J6 measures eastward versus westward power in the MJO band. **Below a ratio of
> 1.5 for the best checkpoint, target T5 fails and the campaign has not produced
> an MJO forecast** — whatever the skill curve says. That is a publishable result
> and it must not be buried.

---

## Task map

Phases are ordered. A task may start only when every task it depends on has a
result file in `results/` with `PROCEED: YES` (or `YES-WITH-CAVEATS` where that
task's preconditions say caveats are acceptable).

### Phase G — Substrate: resolution, scale, statistics

Everything downstream depends on the model eating the right data at the right
size. Nothing else can be diagnosed until this is settled.

| Task | Title | Depends on | Cluster? |
| --- | --- | --- | --- |
| [`G1`](tasks/G1_native_resolution.md) | Native 1° ingestion; delete both upsamplers; read the grid from the archive | — | yes |
| [`G2`](tasks/G2_model_scale.md) | `small`/`full`; retire `huge`; measure 1.3B memory and throughput at 1° | G1 | yes, GPU |
| [`G3`](tasks/G3_norm_stats.md) | True 1980–2015 normalisation statistics; Welford into the package | G1 | yes |
| [`G5`](tasks/G5_sst_boundary.md) | SST as a persisted static: give the model an ocean | G1, G2 | yes |
| [`G4`](tasks/G4_physical_fingerprint.md) | **GATE.** Physical-sanity fingerprint at 1° | G1, G2, G3, G5 | yes, GPU |

### Phase H — The objective

| Task | Title | Depends on | Cluster? |
| --- | --- | --- | --- |
| [`H1`](tasks/H1_weighted_loss.md) | Normalised, weighted, area-correct grid loss; per-variable logging | G3, G4 | no |
| [`H2`](tasks/H2_spectral_loss.md) | Real 2-D spatial spectral loss, per variable; default off | G4 | no |
| [`H3`](tasks/H3_moisture_supervised.md) | ERA5-supervised `E − P`; `tp6h` and `mslhf` in the loader | G4, H1 | yes |
| [`H4`](tasks/H4_trainable_surface.md) | Freezing audit; Aurora's native clamping and `surf_stats`; `no_grad` in validation | G4 | no |

### Phase J — The ruler

| Task | Title | Depends on | Cluster? |
| --- | --- | --- | --- |
| [`J1`](tasks/J1_rmm_rebuild.md) | RMM rebuilt with zonal structure, harmonics, 120-day mean | G1, G4 | yes |
| [`J2`](tasks/J2_bom_gate.md) | **GATE.** `r > 0.95` against the official BoM RMM series | J1 | no |
| [`J3`](tasks/J3_skill_harness.md) | 120-step rollout harness; ACC / RMSE / amplitude / phase vs lead; four baselines | J2, H4 | yes, GPU |
| [`J4`](tasks/J4_zeroshot_control.md) | **GATE.** Zero-shot `AuroraPretrained` skill curve | J3, G2 | yes, GPU |
| [`J5`](tasks/J5_mjo_head_retarget.md) | Re-point MJO-head targets at the corrected RMM; verify; leave disabled | J2 | no |
| [`J6`](tasks/J6_propagation_diagnostics.md) | **GATE.** Does the forecast propagate, or stand still? | J3 | no |
| [`J7`](tasks/J7_enso_stratification.md) | ENSO, season and phase stratification; the regime inventory | J3 | no |

### Phase K — Staged fine-tuning

One run per stage. Each warm-starts from the previous and is evaluated through
J3 before the next begins. **A stage is not started until the previous stage's
skill curve exists.**

| Task | Title | Depends on | Cluster? |
| --- | --- | --- | --- |
| [`K1`](tasks/K1_stage0_warmup.md) | Stage 0: injected embeddings and their heads only, backbone frozen | H1, H2, H4, J4, J6 | yes, GPU |
| [`K2`](tasks/K2_stage1_lora.md) | Stage 1: LoRA, single-step; `E − P` logged as a diagnostic | K1, H3 | yes, GPU |
| [`K3`](tasks/K3_stage2_rollout.md) | Stage 2: pushforward curriculum | K2 | yes, GPU |
| [`K4`](tasks/K4_stage3_physics.md) | Stage 3: physics term on, plus the on/off ablation | K3 | yes, GPU |

### Phase L — Synthesis

| Task | Title | Depends on |
| --- | --- | --- |
| [`L1`](tasks/L1_synthesis.md) | Synthesise the campaign; promote durable truth; recommend the paper campaign | K4, and every prior result file |

### Parallelism

**G1 is on the critical path and nothing else starts until it lands.** G2 and G3
then run concurrently; both touch disjoint files.

After G4, **Phase H and Phase J are fully independent and should run in
parallel** — H changes the objective, J builds the ruler, and they share no
files. That parallelism is the campaign's main schedule lever and is worth
protecting: a task that reaches across from H into `rmm/`, or from J into
`loss.py`, serialises the whole campaign.

Within Phase J, **J5, J6 and J7 are independent of each other** and all three
depend only on J2/J3. J6 and J7 reuse J3's saved rollout output and run no new
inference, so they cost agent time rather than GPU-hours.

Phase K is strictly sequential and cannot start until both H and J are complete,
because a training run needs a correct objective *and* a way to score it.

---

## How the human deploys an agent

```text
You are an agent working on the `aurora-fine-tuning-mjo` repo.

Read AGENTS.md first. Then read docs/campaigns/science-baseline/00_CONTEXT.md
through 04_AGENT_PROTOCOL.md, QUESTIONS.md, and the existing result files in
docs/campaigns/science-baseline/results/. Do NOT read the other task files in
tasks/ — we do not want look-ahead errors.

Your task is to implement <ID> in
docs/campaigns/science-baseline/tasks/<FILE>.md exactly as specified. Respect its
Touches / Must not touch lists. Follow the git hygiene in 04_AGENT_PROTOCOL.md:
branch off epic/science-baseline as epic/science-<ID>-<slug>, commit at internal
milestones, do not merge.

You may not run `sbatch` without showing me the command first.

Paste the summary table from `uv run python scripts/check.py` into your result
file. A task is not done with a red gate. If a Definition-of-Done item is not
literally true, the status is not GREEN.
```

---

## Finishing

When `epic/science-baseline` merges to `main`, move this directory to
`docs/archive/campaigns/science-baseline/`. The `results/` files are the permanent
record of what was measured and what was ruled out — they are why a later agent
does not re-run a dead end.

Anything **durably** true gets promoted into `docs/SPEC.md`,
`docs/evaluation-spec.md` or a document under `docs/findings/` rather than left
buried in a campaign result. In particular:

- The RMM specification (`02_SCIENTIFIC_CONTRACT.md` §6) belongs in
  `docs/evaluation-spec.md` — J1 puts it there, because the current spec is what
  made the code wrong.
- The J4 zero-shot control curve belongs in `docs/findings/`, because it is the
  reference every future result is measured against and it must outlive this
  campaign directory.
- The loss specification (`02_SCIENTIFIC_CONTRACT.md` §4) belongs in
  `docs/SPEC.md`.
- The propagation finding (J6) and the regime inventory (J7) belong in
  `docs/findings/` — the first is the evidence behind target T5, and the second
  documents a split imbalance that limits every future claim.
- The persisted-SST limitation (`02_SCIENTIFIC_CONTRACT.md` §2.5) belongs in
  `docs/SPEC.md`, because it bounds every result at 30 days and the reason is not
  obvious from the code.
