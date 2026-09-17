# Context

Everything an agent needs that is **already true**. Written so a fresh context
window needs nothing else. If this file and the code disagree, the code is right
and this file is stale — say so in your result file.

The document that preceded this campaign is
[`../refactor/handoff-2026-09-refactor.md`](../refactor/handoff-2026-09-refactor.md).
Read it for the engineering history. **Do not take its §10 priority ordering as
binding** — this campaign supersedes it, and §5 below says where and why.

---

## 1. What this repo is

`aurora-fine-tuning-mjo` fine-tunes **Microsoft Aurora** for sub-seasonal
prediction of the **Madden–Julian Oscillation**. The approach treats MJO
forecasting as a physics-consistent initial value problem — prognostic state
stepping, not statistical anomaly regression.

Training runs on **NERSC Perlmutter**, 1 node × 4 × A100 80GB, under SLURM.
Input data is the 1° NERSC/LANL ERA5 preprocessing output on CFS.

The `refactor` campaign (A1–F3, September 2026) took the repo from 73 untested
files to a verified, locked, packaged, test-covered state. That campaign is
**done and it succeeded**. This campaign is what comes after it, and its subject
is the thing `refactor` deliberately did not touch: whether the science is right.

---

## 2. What the refactor campaign could and could not establish

This section exists because a reader who sees eighteen GREEN task results will
reasonably assume the project is in good shape. It is in good *engineering*
shape. That is a different claim.

`refactor` was governed by two gates, both of which were **behavioural
equivalence** gates:

- **B1 / C4** captured config serialisations, parameter counts, dataset
  alignment and GPU smoke-test losses before the refactor, and re-asserted them
  after. All five config diffs empty, parameter delta 0, smoke loss relative
  deviation 0.0 (**MEASURED**, `../refactor/results/C4_result.md`).
- **F2** repeated the same assertions on a fresh Perlmutter clone.

Both passed. Neither could have failed for a scientific reason, because neither
asked a scientific question. The campaign's own §8 records this:

> **Did C4 or F2 catch regressions? NO, PASSED TRIVIALLY.**

The gates worked exactly as designed. The design excluded the science. The
consequence is that every scientific defect present before September 2026 is
still present, now with a test suite around it asserting that it does not
change.

**The rule that follows:** a green gate in this repo currently means "unchanged",
not "correct". This campaign adds gates that mean "correct" (§4 of
`README.md`).

---

## 3. What the review of 2026-09-13 found

Seven findings. Each was verified against the code at commit `85b0db5`, and
where possible against `microsoft-aurora==1.8.0` source and a live forward pass.
Confidence labels are as in `../refactor/`: **MEASURED**, **INFERRED**,
**RECOMMENDED**.

### R1 — The grid loss gives 99% of its gradient to two variables (MEASURED)

`trainer.py:732-743` computes L1 per variable in **raw physical units** and
averages across the eleven variables with equal weight. For an L1 loss the
gradient budget a variable receives is exactly proportional to its physical
standard deviation. Using Aurora's own `normalisation.py` constants and the
`msl` override in `configs/unified.yaml`:

| Variable | σ (physical) | Share of grid-loss gradient |
| --- | ---: | ---: |
| `msl` (= surface pressure) | 9,504.64 | **70.59%** |
| `z` (13-level mean) | 3,828.21 | **28.43%** |
| `ttr` | 49.22 | 0.37% |
| `2t` | 21.22 | 0.16% |
| `tcwv` | 16.33 | 0.12% |
| `u` (13-level mean) | 12.75 | 0.09% |
| `t` (13-level mean) | 12.38 | 0.09% |
| `v` (13-level mean) | 8.91 | 0.07% |
| `10u` | 5.55 | 0.04% |
| `10v` | 4.77 | 0.04% |
| `q` (13-level mean) | 0.00164 | **0.000012%** |

`msl` and `z` take 99.03% between them. The `msl`:`q` ratio is **5,808,226 : 1**.
`tcwv`:`q` is **9,977 : 1**.

The two variables injected specifically because the MJO is a convective,
moisture-mode phenomenon receive 0.49% of the optimisation pressure combined.
Specific humidity receives none.

This also explains the B1 fingerprint. A baseline train loss of **7472.024414**
is not mysterious; it is what `z` and `msl` alone produce.

Every comparable model — Aurora, GraphCast, Pangu — computes loss in
**normalised** space with explicit per-variable and per-level weights.
Task **H1**.

### R2 — The moisture-budget loss penalises the MJO (MEASURED)

`loss.py:258-268` computes `R = ∂⟨q⟩/∂t + ∇·⟨vq⟩`, which by the budget equation
**is** `E − P`, then penalises `mean(|R|)` toward zero.

`E − P` is not zero in the tropics. In an active MJO envelope P is 20–40 mm/day
against E of 4–5 mm/day, so `E − P ≈ −20 to −35 mm/day`. That *is* the signal.
A smoothed field has small gradients, small divergence, small tendency, and
therefore small `|R|`. **The term as written rewards smoothing**, which is the
opposite of its documented intent and would fight the spectral loss if both were
enabled.

Second, independent error: the justification in `docs/papers/` §III-C argues the
residual "should balance to zero over large temporal/spatial tropical averages",
which is a statement about `|mean(R)|`. The code computes `mean(|R|)`. Different
quantities; only the first is defensible.

Decided: replace with ERA5-supervised `E − P`. Argument on both sides and the
decision record is `02_SCIENTIFIC_CONTRACT.md` §3. Task **H3**.

### R3 — The RMM pipeline does not compute RMM (MEASURED)

`rmm/compute.py:42` (`tropical_mean`) averages over latitude **and longitude**,
returning one scalar per field per day. `scripts/compute_rmm.py` then builds a
`(T_train, 3)` matrix and eigendecomposes a 3×3 covariance.

Wheeler & Hendon requires `a(t) ∈ ℝ^(3·N_λ)`. The zonal structure is what makes
the MJO an eastward-propagating phenomenon; averaging it away leaves three global
scalars whose EOFs carry no propagation information. Phase angle is meaningless.
The `A > 1` threshold is uncalibrated. `rmm/evaluate.py:159-174`
(`_trop_mean_surf`) has the same collapse and returns a `float`.

Also absent: previous-120-day mean removal, seasonal cycle as mean + first three
harmonics, and any EOF sign/order convention anchoring.

Two aggravating factors. `docs/evaluation-spec.md` **specifies** the 3-element
vector, so the code faithfully implements a wrong spec. And
`src/aurora_mjo/rmm/` has **zero test coverage** — there is no `test_rmm.py`;
`tests/test_mjo_head.py` matches only on the string "RMM". Tasks **J1**, **J2**.

### R4 — Production runs used the debug model (MEASURED)

`configs/unified.yaml:47` sets `model_type: "small"`. B1 fingerprints
**112,830,384** parameters, reproduced exactly in the review. That is
`AuroraSmallPretrained`, whose docstring in `microsoft-aurora==1.8.0` reads:

> "Small pretrained version of Aurora. **Should only be used for debugging.**"

Every claim of a 1.3B foundation model in `docs/papers/aurora-mjo-fall-paper.pdf`
is a claim about a model that did not run. Task **G2**.

Related and not yet triggered: `model.py:424` maps `"huge"` → `Aurora`, whose
default checkpoint is `aurora-0.25-finetuned.ckpt` — the **IFS HRES analysis**
fine-tune, not the ERA5 pretrained model. See `02_SCIENTIFIC_CONTRACT.md` §2.

### R5 — Aurora accepts native 1° input (MEASURED, live forward pass)

Run against `microsoft-aurora==1.8.0`, `AuroraSmallPretrained`, the project's six
surface variables, random init, CPU:

```text
1 deg cell-centred    180x360 -> crop 180x360 -> patch grid (4, 45, 90) -> FORWARD OK, all finite
1 deg pole-inclusive  181x360 -> crop 180x360 -> patch grid (4, 45, 90) -> FORWARD OK, all finite
```

Both 180 and 360 divide by `patch_size=4`. The 45×90 patch grid is not a
multiple of the `(2,6,12)` window, but Aurora's Swin3D pads and crops internally
(`aurora/model/swin3d.py:350`). `Batch.crop` explicitly handles
`h % patch_size == 1`, which is ERA5's pole-inclusive convention (181→180,
721→720). Aurora was built to accept this.

This confirms the guidance from the Aurora team
([microsoft/aurora#184](https://github.com/microsoft/aurora/issues/184),
2026-05-13): *"Your best bet will likely to give the one-degree data directly to
Aurora and fine-tune the model."*

`_upsample_to_aurora` (`dataset.py:85`) and `_upsample_batch_gpu`
(`trainer.py:155`) are therefore both retired. Task **G1**.

### R6 — The spectral loss does not compute a spatial spectrum (MEASURED)

`_extract_batch_outputs` (`trainer.py:126-138`) flattens and concatenates every
variable into a `(B, N)` tensor. `SpectralLoss` (`loss.py:69-72`) then calls
`torch.fft.rfft2` on it, transforming over `(batch, concatenated-everything)`.
At `B=1` that is a 1-D FFT over a row-major-flattened mixture of `z`, `q`, `t`,
`u`, `v` across 13 levels and the surface variables.

The term is disabled in `configs/unified.yaml`, so no training run was affected.
But `docs/papers/` §IV-B attributes the sharp convective structures and the
Day-10 grittiness to it at λ = 0.05. That interpretation is not supported by what
the code computes. Task **H2**.

### R7 — Smaller items (mixed)

1. **"Rollout training" is detached.** `backprop: "detached"` everywhere;
   `_advance_batch(..., detach=True)` cuts the graph at every step boundary.
   That is pushforward / scheduled-sampling training, which is genuinely useful
   against exposure bias, but there is no gradient path from step *k* to step 1,
   so it does not optimise the multi-step objective the paper's Eq. 14 describes
   (**INFERRED**). Decision recorded in `02_SCIENTIFIC_CONTRACT.md` §5.
2. **Validation will OOM in `lora` / `combined`.** `validate()`
   (`trainer.py:1236`) calls `_compute_loss`, which runs `_advance_batch` with
   `detach=False` and **no `torch.no_grad()`**. At `k=4` that holds four full
   forward graphs at once. Baseline (`k=1`) never exercised this path
   (**INFERRED**; expected to fail on the first validation epoch of the first
   rollout run). Task **H4**.
3. **Finding 3 is already fixed.** The handoff §7.1 item 4 and §10.2 item 3 both
   list unclamped rollout feedback as outstanding. `_ROLLOUT_CLAMP` exists at
   `trainer.py:218` and `_clamp_fed` is applied in `_advance_batch`
   (**MEASURED**). Separately, Aurora 1.8.0 exposes native `positive_surf_vars`,
   `positive_atmos_vars` and `clamp_at_first_step` constructor arguments that do
   this properly. Task **H4**.
4. **Vendor extension points unused.** Aurora exposes `surf_stats` as a
   constructor argument; `model.py:438` instead mutates the module-global
   `locations` / `scales` dicts, which is process-global state that leaks across
   pytest runs. `batch_transform_hook` and `_pre_encoder_hook` exist for exactly
   the variable injection being done by hand. Task **H4**.
5. **Grid metadata is subtly wrong.** `dataset.py:153` sets
   `lat = torch.linspace(90, -90, 720)`, which is 0.2503° spacing, not 0.25°.
   `slt` is truncated `[:720, :]` from a real 0.25° grid, so `slt` sits on a
   *different* grid from `z` and `lsm`. Retired by G1.
6. **4-step training, 120-step evaluation.** `rollout.max_steps: 4` is 24 hours.
   Skilful RMM at 30 days is 120 autoregressive steps. Discussed in
   `02_SCIENTIFIC_CONTRACT.md` §5 and §6.

---

## 4. What is still true from the refactor campaign

Do not re-derive these. All seven `refactor` lessons still hold and all have
regression tests (`../refactor/results/D3_result.md`). In particular:

- **Lesson 1** — `msl` is surface pressure wearing an MSL costume. The override
  in `configs/unified.yaml` is still a **PLACEHOLDER**. Task **G3** replaces it.
- **Lesson 3** — `HDF5_USE_FILE_LOCKING=FALSE` is load-bearing on CFS and is now
  set at process start in `aurora_mjo/env.py`.
- **Lesson 5** — a glob is not a year filter. The v3 per-variable timestamp maps
  in `dataset.py` are the fix and the single highest-value regression test in
  the repo. **G1 must not weaken them.**
- **Lesson 6** — `gradient_checkpointing: false` is load-bearing; checkpointing
  deterministically triggers an illegal memory access on Perlmutter. G2
  re-tests this at 1°, where the memory pressure that motivated it is 16× lower.
- **Lesson 7** — plan against measured state. That includes this campaign's plan.

The engineering surface is unchanged and stays unchanged: `uv sync
--all-groups`, `uv run python scripts/check.py` as the gate, `run.py` as the CLI,
92 tests, `epic/`-prefixed task branches.

---

## 5. Where this campaign overrules the refactor handoff

`../refactor/handoff-2026-09-refactor.md` §10.1 names the true `ps` normalisation
statistics as **"the single most critical next action"**. That was a defensible
reading of the evidence available to F3, which had not audited `loss.py` or
`rmm/`.

It is superseded. Correct normalisation statistics feed a loss that gives
`q` 0.000012% of its gradient (R1) and are scored by a pipeline that does not
compute RMM (R3). Fixing them first improves the precision of a measurement
nobody can interpret.

The revised order is **G → {H, J} → K**: fix the substrate, then fix the ruler
and the objective in parallel, then train. The `ps` statistics are task **G3**
and are still mandatory — they have simply moved from first to third.

Two other handoff recommendations are affected:

- §10.2 item 3 (rollout input clipping) is **already done** — see R7.3.
- §10.2 item 5 (small vs full Aurora benchmark) is **promoted** from item 5 to
  task G2, because R5 changes its answer: at 1° the memory argument that forced
  `small` is 16× weaker.

---

## 6. What is out of scope for this campaign

Written down so that a helpful agent does not implement it.

- **The paper.** This campaign produces a measured baseline and the evidence
  behind it. Writing `docs/papers/` is a later campaign and needs the ablations
  this one deliberately defers.
- **Ablation studies.** LoRA rank sweeps, loss-term ablations, rollout-curriculum
  variants. Phase K produces **one** run per stage, not a grid. An ablation with
  a 24-hour queue turnaround and no validated ruler is wasted compute.
- **Aurora v1.5.** The ensemble-capable release is explicitly deferred to
  follow-on work. This campaign uses **v1.0 weights via
  `microsoft-aurora==1.8.0`** and is single-member deterministic throughout.
  See `02_SCIENTIFIC_CONTRACT.md` §6 and Q-14.
- **Ensembling of any kind**, including lagged-initialisation ensembles and
  multi-seed averaging. Same reason. If a Phase K result makes this look
  necessary, that is a finding for the synthesis task, not a scope expansion.
- **Changing the data splits.** 1980–2015 / 2016–2019 / 2020–2023 are fixed.
  **The test years are not touched by anything in this campaign**, including
  diagnostics. See `04_AGENT_PROTOCOL.md` §4.
- **Re-running the `refactor` gates.** B1/C4/F2 fingerprints are historical
  record. G1 deliberately changes the numbers they captured; that is not a
  regression and `04_AGENT_PROTOCOL.md` §5 says how to record it.
- **History rewriting**, and **deleting anything** in the recovery table at
  `../refactor/handoff-2026-09-refactor.md` §11.
