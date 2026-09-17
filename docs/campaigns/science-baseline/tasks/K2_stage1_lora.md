# K2 — Stage 1: LoRA, single-step; `E − P` logged as a diagnostic

| | |
| --- | --- |
| **Phase** | K |
| **Depends on** | K1, H3 (both `PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 2 h of agent time, plus one SLURM chain |
| **Compute** | **4 × A100 × 11.5 h**, one chain. `sbatch` with approval. See **Q-18**. |
| **Touches** | `configs/unified.yaml` (the `lora` mode), `docs/results/skill-curves-2026-1x.md`, `checkpoints/lora/`, `docs/PROJECT_STATE.md` |
| **Must not touch** | Anything under `src/`. `data/rmm_basis.npz`. Any `results/*.md`. |

## Objective

LoRA adapters are trained on single-step prediction against the H1 loss,
warm-started from Stage 0, with the H3 moisture term logged at zero weight
throughout. The checkpoint is evaluated through J3 and compared against both the
J4 control and Stage 0.

## Why this is a separate stage

This is the stage that tests the project's central hypothesis. Stage 0 only
adapted the new channels; Stage 1 is the first time the backbone's behaviour
changes at all. The Stage 1 − Stage 0 delta **is** the answer to "does
parameter-efficient fine-tuning of a pretrained Earth-system foundation model buy
MJO skill" — target **T1**.

It also runs the H3 diagnostic for a full stage at zero weight, which is what
`02_SCIENTIFIC_CONTRACT.md` §3.3 requires before the term is ever trained on.
K4 cannot start without it.

## What you may assume

- K1 produced a Stage 0 checkpoint and answered whether the warm-up surface was
  adequate (`results/K1_result.md`).
- H3 landed the supervised `E − P` term at `weight: 0.0` with diagnostic logging,
  and its Step-1 diagnostic said the numerics were sound (`results/H3_result.md`).
  **If H3 came back `AMBER` recommending retirement, K4 is cancelled and this
  task records that.**
- H4 recorded the LoRA configuration: rank 8, `lora_mode: "single"`, QKV and
  output projection in every Swin3D attention block (`results/H4_result.md`).
  **Do not change it** — K1–K4 must be mutually comparable.
- `init_from` points at `checkpoints/warmup` and **fails loudly if absent**
  (`01_TARGET_STATE.md` D6). A silent cold start turns a Stage 1 result into a
  Stage 0 result.

## Steps

1. Configure the `lora` mode: `model_type: full`, LoRA on, **rollout off**
   (it was previously on at `k ≤ 4`; Stage 2 owns rollout), moisture term at
   weight 0 with diagnostic logging, spectral off, MJO head off.
2. **Confirm the warm start actually loaded.** Paste the log line showing the
   Stage 0 checkpoint path and the loaded parameter count. This is the single
   easiest thing to get silently wrong in the whole chain.
3. Smoke first on `small`. Paste it.
4. Submit one chain. Monitor per-variable loss components and the `E − P`
   diagnostic.
5. **Report the `E − P` diagnostic over the whole stage**: correlation between
   the model's implied residual and ERA5's `E − P` over the tropical band, at the
   start and end of training. This is the evidence K4 needs, and it is a
   physically interpretable statement about whether the model's convection is in
   the right place — worth a figure regardless of what K4 decides.
6. Evaluate through J3. Full per-lead table.
7. Compare against **both** J4 and K1. Report the ACC = 0.5 crossing for all
   three, the amplitude ratio at days 10/20/30, and state whether **T1 holds so
   far** — fine-tuned beats zero-shot at every lead from 5 to 30 days.
8. Append to `docs/results/skill-curves-2026-1x.md`.
9. Report GPU-hours against budget.

## Definition of Done

- [ ] `lora` mode configured; rollout confirmed **off**; resolved config pasted
- [ ] Warm start from `checkpoints/warmup` confirmed; log line pasted
- [ ] Smoke test on `small` passes; output pasted
- [ ] SLURM command shown and approved before submission
- [ ] Chain completes or stops for a stated reason; steps and wall-clock reported
- [ ] Per-variable loss components pasted at start and end
- [ ] **`E − P` diagnostic reported**: correlation with ERA5 over the tropics at
      start and end of the stage
- [ ] J3 evaluation run; full per-lead table pasted
- [ ] Three-way comparison J4 / K1 / K2: ACC = 0.5 crossing, amplitude ratio at
      days 10/20/30
- [ ] **T1 verdict stated explicitly** for this stage
- [ ] **Eastward/westward MJO-band power ratio reported** via J6; T5 status
      stated for this stage
- [ ] Metrics stratified by ENSO regime, season and initial phase via J7, if J7
      is complete
- [ ] `docs/results/skill-curves-2026-1x.md` appended
- [ ] GPU-hours reported against the 11.5 h budget
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] `docs/PROJECT_STATE.md` updated
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Any change under `src/`.**
- Changing the LoRA rank, mode or insertion points. H4 recorded them and a later
  campaign ablates them; changing them here makes the stage comparison
  meaningless.
- Enabling rollout, the physics weight, the spectral term or the MJO head.
- Tuning against validation skill.
- Reading test years.
