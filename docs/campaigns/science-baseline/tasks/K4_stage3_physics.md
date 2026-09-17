# K4 — Stage 3: physics term on, plus the on/off ablation

| | |
| --- | --- |
| **Phase** | K |
| **Depends on** | K3 (`PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 3 h of agent time, plus two SLURM chains |
| **Compute** | **2 × (4 × A100 × 11.5 h)** — this is the campaign's only two-chain task, and the second chain is the ablation control. `sbatch` with approval. See **Q-18**. |
| **Touches** | `configs/unified.yaml` (the `physics` mode), `docs/results/skill-curves-2026-1x.md`, `checkpoints/physics/`, `checkpoints/physics_ablation/`, `docs/PROJECT_STATE.md` |
| **Must not touch** | Anything under `src/`. `data/rmm_basis.npz`. Any `results/*.md`. |

## Objective

The ERA5-supervised moisture term is given a non-zero weight, warm-started from
Stage 2, and evaluated through J3 — **alongside an otherwise identical run with
the term at zero weight**, so the term's effect is isolated rather than confounded
with a further epoch of training.

## Why this is a separate stage, and why it is two chains

`02_SCIENTIFIC_CONTRACT.md` §3.3 makes the ablation part of the decision, not an
optional extra. A Stage 3 that only ran with the term on would be
indistinguishable from Stage 2 plus more training, and the "physics-informed"
claim would rest on nothing. A reviewer will ask for exactly this comparison and
it costs one extra chain.

The rest of the campaign is one run per stage by design (`00_CONTEXT.md` §6).
This is the single authorised exception and the reason is stated here so nobody
generalises it into a sweep.

## What you may assume

- K3 produced a Stage 2 checkpoint and a noise-reduction verdict
  (`results/K3_result.md`).
- H3's Step-1 diagnostic said the numerics were sound, and K2 ran the term at
  zero weight for a full stage and reported its correlation with ERA5 `E − P`
  (`results/H3_result.md`, `results/K2_result.md`).
  **If either said the residual is dominated by discretisation noise, this task
  does not run.** Mark it `BLOCKED`, record that the term was retired on
  evidence, and hand straight to L1 — that is a clean result and
  `02_SCIENTIFIC_CONTRACT.md` §3.3 authorises it.
- `02_SCIENTIFIC_CONTRACT.md` §3.3: λ_p is tuned so the term is **≤ 10% of total
  loss at the start of training**. That is a scale-matching choice made from the
  first few logged steps, not a search over validation skill.

## Steps

1. Configure the `physics` mode: as K3's `rollout` mode, plus the moisture term
   at a non-zero weight. Everything else identical.
2. **Set λ_p by scale matching, not by search.** Run a handful of steps at
   weight 1.0, read off the ratio of the moisture term to the grid term, and set
   λ_p so the ratio is ≤ 0.1. Paste both the measured ratio and the resulting
   λ_p. Tuning λ_p against validation skill and then reporting that skill is
   leakage (`02_SCIENTIFIC_CONTRACT.md` §6.3).
3. Smoke first on `small`. Paste it.
4. Confirm warm start from `checkpoints/rollout`; paste the log line.
5. Submit **chain A**: physics term on.
6. Submit **chain B**: byte-identical config except `moisture_budget.weight: 0.0`,
   same seed, same number of steps, saving to `checkpoints/physics_ablation/`.
   Paste the config diff between A and B and confirm it is **one line**.
7. Evaluate both through J3.
8. **Report the ablation as the headline**: the A − B difference in ACC = 0.5
   crossing, in amplitude ratio at days 10/20/30, and in gridded tropical RMSE.
   State plainly whether the physics term helped, hurt, or did nothing. All three
   are publishable; only a confounded result is not.
9. Report the `E − P` correlation with ERA5 for both chains. If the term is
   working as intended, chain A's should be higher — **that is the mechanistic
   check**, independent of whether RMM skill moved.
10. Five-way comparison on the skill-curve document: J4, K1, K2, K3, K4-A, plus
    K4-B as the ablation control.
11. **State the final verdict on T1 through T6** against the best checkpoint.
    T5 comes from J6's module and is pass/fail; T6 from J7's stratification and
    is interpretable only if G5 landed.
    This is the campaign's headline result and L1 synthesises from it.

## Definition of Done

- [ ] `physics` mode configured; λ_p set by scale matching; measured ratio and
      resulting λ_p pasted
- [ ] Smoke test on `small` passes; output pasted
- [ ] Warm start from `checkpoints/rollout` confirmed; log line pasted
- [ ] Both SLURM commands shown and approved before submission
- [ ] **Config diff between chains A and B is exactly one line**; diff pasted
- [ ] Both chains complete or stop for a stated reason; steps and wall-clock for
      each
- [ ] Both evaluated through J3; full per-lead tables pasted
- [ ] **Ablation reported**: A − B difference in ACC = 0.5 crossing, amplitude
      ratio at days 10/20/30, gridded tropical RMSE
- [ ] Physics-term verdict stated plainly: helped, hurt, or no effect
- [ ] `E − P` correlation with ERA5 for both chains; mechanistic check assessed
- [ ] Five-way skill-curve document complete with the ablation control
- [ ] **T1–T6 verdicts stated** against the best checkpoint. T5 (propagation) is
      pass/fail and goes in the first line of the result file if it failed.
- [ ] GPU-hours reported against the 23 h budget
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] `docs/PROJECT_STATE.md` updated
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Any change under `src/`.**
- A λ_p sweep. Two chains, on and off. The weight sweep is a later campaign
  (`02_SCIENTIFIC_CONTRACT.md` §7).
- A third chain to "get a better number".
- Enabling the spectral term or the MJO head.
- Reading test years. **Especially here** — this is the stage where the
  temptation to check the test set is strongest, and the quarantine
  (`04_AGENT_PROTOCOL.md` §4) is absolute.
