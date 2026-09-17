# K3 — Stage 2: pushforward rollout curriculum

| | |
| --- | --- |
| **Phase** | K |
| **Depends on** | K2 (`PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 3 h of agent time, plus one SLURM chain |
| **Compute** | **4 × A100 × 11.5 h**, one chain. `sbatch` with approval. See **Q-18**. |
| **Touches** | `configs/unified.yaml` (the `rollout` mode), `docs/results/skill-curves-2026-1x.md`, `checkpoints/rollout/`, `docs/SPEC.md`, `docs/PROJECT_STATE.md` |
| **Must not touch** | Anything under `src/`. `data/rmm_basis.npz`. Any `results/*.md`. |

## Objective

The model is trained on its own generated inputs under a lengthening pushforward
curriculum, warm-started from Stage 1, and evaluated through J3. `docs/SPEC.md`
describes the training objective accurately.

## Why this is a separate stage

`docs/papers/` §IV-C observed that a single-step-trained model stays stable over
40 self-generated steps but accumulates high-frequency noise. Pushforward
training attacks exactly that: the exposure-bias gap between training on
reanalysis inputs and inferring on self-generated ones.

It is the stage most likely to fail operationally. `k` rises during the
curriculum, which means memory rises, sample count falls, and the validation path
runs `k` forward passes per batch — the path H4 had to add `no_grad()` to because
nobody had ever exercised it at `k > 1` (`00_CONTEXT.md` R7.2).

## What you may assume

- K2 produced a Stage 1 checkpoint and a T1 verdict (`results/K2_result.md`).
- H4 wrapped validation in `torch.no_grad()` (`results/H4_result.md`). Without it
  this stage OOMs on its first validation epoch.
- `02_SCIENTIFIC_CONTRACT.md` §5.1: **detached pushforward is the objective, and
  it is not BPTT.** `backprop: "full"` is not exercised in this campaign.
- `02_SCIENTIFIC_CONTRACT.md` §5.2: the train/evaluate horizon gap (`k ≤ 4` vs
  120 steps) is **stated and measured, not fixed**.
- `03_DOMAIN_PRIORS.md` §1: sample count falls by `k − 1` per curriculum
  increment. **A curriculum step that does not reduce the sample count has not
  taken effect** — check this, it is the cheapest possible verification.

## Steps

1. Configure the `rollout` mode: `model_type: full`, LoRA on, rollout on with
   `backprop: "detached"`, `start_steps: 1`, `step_increase_every_n_epochs: 1`,
   moisture term at weight 0 with diagnostic logging.
2. **Set `max_steps` from G2's measured memory, not from the old config's 4.**
   The 16× grid reduction may allow a longer curriculum than was ever possible at
   0.25°. State the value chosen, the memory headroom it leaves, and the
   reasoning. Longer is better here, up to the wall-clock budget — every extra
   step narrows the §5.2 extrapolation gap.
3. Smoke first on `small` **at the maximum `k` you intend to reach**, not at
   `k = 1`. The failure mode this stage carries is a `k`-dependent one, and a
   `k = 1` smoke test does not test for it.
4. Confirm the warm start from `checkpoints/lora` loaded; paste the log line.
5. Submit one chain. At every curriculum increment, record: `k`, sample count,
   peak memory, step time. **Confirm the sample count fell by `k − 1`.**
6. Watch for the specific failure this stage carries: non-finite loss appearing
   only after a curriculum increment. The grad guard (Lesson 2) will skip those
   steps and log `[grad-guard]`. **Report the skip ratio per `k`** — a rising
   skip ratio means the curriculum outran the model's stability and the stage
   should stop at the last stable `k` rather than push on.
7. Evaluate through J3. Full per-lead table.
8. Four-way comparison: J4, K1, K2, K3. **State whether the noise accumulation
   that `docs/papers/` §IV-C describes is reduced** — gridded tropical RMSE at
   days 5 and 10 is the direct measure, alongside RMM ACC.
9. **Report where the skill curve sits relative to the training horizon.** If ACC
   collapses at or near the trained `k`, that is a headline finding for the
   synthesis, not a reason to extend the campaign.
10. Update `docs/SPEC.md` to describe the objective as **pushforward / detached
    rollout**, not as multi-step rollout training
    (`02_SCIENTIFIC_CONTRACT.md` §5.1).

## Definition of Done

- [ ] `rollout` mode configured; `max_steps` chosen from G2's measured memory,
      with the reasoning and headroom stated
- [ ] Smoke test on `small` **at maximum `k`** passes; output pasted
- [ ] Warm start from `checkpoints/lora` confirmed; log line pasted
- [ ] SLURM command shown and approved before submission
- [ ] Per-increment table pasted: `k`, sample count, peak memory, step time
- [ ] **Sample count falls by `k − 1` at every increment**, confirmed
- [ ] Grad-guard skip ratio reported per `k`
- [ ] J3 evaluation run; full per-lead table pasted
- [ ] Four-way comparison J4 / K1 / K2 / K3
- [ ] Gridded tropical RMSE at days 5 and 10 compared against K2; noise-reduction
      verdict stated
- [ ] Skill curve position relative to the training horizon stated explicitly
- [ ] `docs/SPEC.md` describes the objective accurately
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

- **Any change under `src/`.** If the curriculum exposes a trainer bug, stop and
  report; a Phase H follow-up owns it.
- Enabling `backprop: "full"`. `02_SCIENTIFIC_CONTRACT.md` §5.1.
- Enabling the physics weight (K4), the spectral term or the MJO head.
- Extending the campaign to close the horizon gap. Measure it; report it.
- Reading test years.
