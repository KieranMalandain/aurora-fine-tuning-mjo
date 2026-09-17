# K1 — Stage 0: injected embeddings and their heads, backbone frozen

| | |
| --- | --- |
| **Phase** | K |
| **Depends on** | H1, H2, H4, J4, J6 (all `PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 2 h of agent time, plus one SLURM chain |
| **Compute** | **4 × A100 × 11.5 h**, one chain. `sbatch` with approval. See **Q-18**. |
| **Touches** | `configs/unified.yaml` (the `warmup` mode), `slurm_scripts/train_auto.slurm`, `docs/results/skill-curves-2026-1x.md` (new), `checkpoints/warmup/`, `docs/PROJECT_STATE.md` |
| **Must not touch** | Anything under `src/`. **If code needs to change, the run stops and a Phase H or J follow-up owns the fix.** `data/rmm_basis.npz`. Any `results/*.md`. |

## Objective

The randomly initialised `ttr` and `tcwv` patch embeddings, their decoder heads
and the `msl` head are trained to a sane operating point against the H1 loss,
with the Aurora backbone entirely frozen. The resulting checkpoint is evaluated
through J3 and compared against the J4 control.

## Why this is a separate stage

The injected embeddings and decoder heads are random at initialisation. After H1
they now carry a meaningful share of the loss — `ttr` and `tcwv` go from 0.49%
of the gradient to ~47% (`03_DOMAIN_PRIORS.md` §5). Letting LoRA adapt a 1.3B
backbone to random-noise channels is a textbook route to catastrophic forgetting,
which is the failure mode the May research script §II-D names as the central
risk.

Stage 0 gets the new channels into range first, while the backbone cannot move.
It is cheap, it is bounded, and it makes the Stage 1 delta attributable to LoRA
rather than to the embeddings finally converging.

## What you may assume

- H1: the loss is normalised and weighted (`results/H1_result.md`).
- H4: the `warmup` freezing mode exists and its trainable set is tested
  (`results/H4_result.md`). H4 flagged whether the trainable surface is large
  enough — **this task answers that question**.
- J4: the control curve exists (`results/J4_result.md`). Every number here is
  reported against it.
- J6: propagation diagnostics exist and were run on the control
  (`results/J6_result.md`). **Every stage reports the eastward/westward power
  ratio alongside its skill curve** — target T5 is pass/fail, not a plot.
- J7, if complete: every stage's metrics are also reported stratified by ENSO
  regime, season and initial phase (`results/J7_result.md`).
- G2: measured step time and memory for `full` at 1° (`results/G2_result.md`).
  The chain length is planned against those, not against the WEAK estimates.
- `04_AGENT_PROTOCOL.md` §10: smoke first, one variable per job, log to the repo.

## Steps

1. Configure the `warmup` mode: `model_type: full`, LoRA off, rollout off,
   moisture term at weight 0 (diagnostic only), spectral off, MJO head off.
2. **Smoke first.** Run `--smoke-test` on `model_type: small` and confirm a
   finite loss and non-zero gradients on exactly the expected parameters. Paste
   it. No multi-hour job is submitted without this.
3. Submit one chain. Monitor the per-variable loss components that H1 added —
   `loss/grid/ttr` and `loss/grid/tcwv` should fall substantially; everything
   else should be roughly flat, because nothing else can learn.
   **If a frozen variable's loss moves, the freezing is wrong**: stop the chain
   and report.
4. Record the trainable parameter count and what fraction of the model it is.
   Answer H4's open question: was the surface large enough for the injected
   channels to reach a sane operating point, or did it saturate?
5. Evaluate the checkpoint through J3. Produce the full per-lead table.
6. Compare against J4. **State plainly whether Stage 0 alone moves the ACC = 0.5
   crossing**, and by how much. A null result here is informative and expected —
   the backbone has not changed, so large movement would be surprising and worth
   investigating before Stage 1.
7. Start `docs/results/skill-curves-2026-1x.md` with the control and Stage 0 on
   the same axes. Every later stage appends to it. One document, four curves,
   four baselines.
8. Report GPU-hours used against budget.

## Definition of Done

- [ ] `warmup` mode configured as specified; resolved config pasted
- [ ] Smoke test on `small` passes with finite loss; output pasted
- [ ] SLURM command shown and approved before submission
- [ ] Chain completes or is stopped for a stated reason; steps completed and
      wall-clock reported
- [ ] Per-variable loss components pasted at start and end; **frozen variables
      confirmed flat**
- [ ] Trainable parameter count and fraction reported; **H4's open question
      answered explicitly**
- [ ] J3 evaluation run; full per-lead table pasted
- [ ] Comparison against the J4 control: ACC = 0.5 crossing, amplitude ratio at
      days 10/20/30, delta stated
- [ ] `docs/results/skill-curves-2026-1x.md` created with control + Stage 0
- [ ] **Eastward/westward MJO-band power ratio reported** via J6's module; T5
      status stated for this stage
- [ ] Checkpoint carries model provenance (G2's header)
- [ ] GPU-hours reported against the 11.5 h budget
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] `docs/PROJECT_STATE.md` updated
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Any change under `src/`.** A training stage that also changes code produces a
  result nobody can attribute. If the run reveals a bug, stop, report, and let a
  Phase H or J follow-up own it.
- Enabling LoRA, rollout, the physics term, the spectral term or the MJO head.
- Tuning the learning rate against validation skill. Set it once, from the config,
  and report it.
- Running a second chain to "get a better number".
- Reading test years.
