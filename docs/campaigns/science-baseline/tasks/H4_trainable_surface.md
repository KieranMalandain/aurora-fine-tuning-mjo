# H4 — Trainable-surface audit; Aurora's native clamping; `no_grad` in validation

| | |
| --- | --- |
| **Phase** | H |
| **Depends on** | G4 (`PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 4 h |
| **Compute** | **None** for the audit. 1 GPU-hour permitted to reproduce the validation OOM if it is cheap; `sbatch` with approval. |
| **Touches** | `src/aurora_mjo/model.py`, `src/aurora_mjo/trainer.py` (only `validate`, `_advance_batch`, `_ROLLOUT_CLAMP`), `configs/unified.yaml`, `tests/test_freeze.py`, `tests/test_rollout.py`, `docs/SPEC.md`, `docs/PROJECT_STATE.md` |
| **Must not touch** | `src/aurora_mjo/loss.py` (H1/H2/H3). `src/aurora_mjo/dataset.py`, `rmm/`. The DDP-collective grad guard (Lesson 2). Any `results/*.md`. |

## Objective

Exactly which parameters train in each mode is known, stated, and tested. Aurora's
native positivity clamping replaces the hand-rolled `_ROLLOUT_CLAMP` table.
Validation runs under `torch.no_grad()`. The stage-0 `warmup` freezing
configuration exists.

## Why this is a separate task

Four small, related defects around the parameter surface, all of which would
otherwise surface during Phase K as a wasted SLURM slot:

1. **Validation will OOM in rollout modes** (`00_CONTEXT.md` R7.2). `validate()`
   calls `_compute_loss`, which advances with `detach=False` and **no
   `torch.no_grad()`**. At `k = 4` that holds four full forward graphs at once.
   Baseline (`k = 1`) never exercised this, so nobody has hit it — and K3 will hit
   it on its first validation epoch.
2. **`_ROLLOUT_CLAMP` re-implements vendor functionality** (R7.3). Aurora 1.8.0
   exposes `positive_surf_vars`, `positive_atmos_vars` and `clamp_at_first_step`.
   A hard `clamp` also has zero gradient outside its range, which matters if
   `backprop: "full"` is ever used.
3. **The `warmup` stage needs a freezing mode that does not exist** — backbone
   entirely frozen, only injected embeddings, their decoder heads and the `msl`
   head trainable (`02_SCIENTIFIC_CONTRACT.md` §5.3).
4. **Nobody has written down what trains in each mode.** The parameter counts in
   `../refactor/handoff-2026-09-refactor.md` §5.2 are for `small` at the old mode
   set and will not survive G2 or D6.

## What you may assume

- G2 settled `model_type: full` → `AuroraPretrained` and measured its parameter
  count (`results/G2_result.md`).
- `01_TARGET_STATE.md` D6 defines the new mode set: `warmup`, `lora`, `rollout`,
  `physics`. `combined` is retired (**Q-19**).
- `freeze_backbone` already unfreezes injected patch embeddings, injected decoder
  heads, and the `msl` decoder head (`model.py:298-366`). Those are FIX 4 and
  FIX 6 and they are correct — do not remove them.
- `02_SCIENTIFIC_CONTRACT.md` §5.1: detached pushforward stays the default and
  full BPTT is **not** exercised in this campaign.

## Steps

1. Wrap the validation forward pass in `torch.no_grad()`. Confirm the loss value
   is unchanged at `k = 1` (it must be) and that peak memory at `k = 4` falls by
   roughly the expected factor.
2. Replace `_ROLLOUT_CLAMP` with Aurora's constructor arguments:
   `positive_surf_vars=("tcwv",)` and `positive_atmos_vars=("q",)` at minimum.
   Decide whether `clamp_at_first_step` should be `True` given that the injected
   heads are randomly initialised at the start of `warmup` — **state the
   reasoning**. If any bound in the existing table has no vendor equivalent, keep
   that entry and say which and why; do not delete a guard you cannot replace.
3. Add the `warmup` freezing mode: everything frozen except injected `ttr` /
   `tcwv` patch embeddings, their `decoder.surf_heads` entries, and the `msl`
   head. LoRA adapters **off**.
4. Produce the parameter table — for `model_type: full`, for all four modes:
   total, trainable, frozen, trainable %, and a per-module breakdown naming which
   modules contribute. This replaces
   `../refactor/handoff-2026-09-refactor.md` §5.2.
5. **Sanity-check the trainable count in `warmup`.** If it is a few tens of
   thousands of parameters against a 1.3B backbone, say so explicitly and note
   whether that is a large enough surface for the injected channels to reach a
   sane operating point. This is a real open question, it is the kind of thing
   F3 flagged ("whether the ~41k trainable parameters are the right training
   surface at all"), and K1's result will answer it. Do **not** change the LoRA
   rank here on a hunch.
6. Record the LoRA configuration as it stands: rank, `lora_mode`, which modules
   are adapted, and `lora_steps`. Note that `lora_mode: "single"` shares one
   adapter across rollout steps while Aurora supports a per-step mode up to
   `lora_steps=40` — **record it as an option for a later campaign, do not
   change it.**
7. Add tests: each mode's trainable set matches the expectation exactly, by name;
   validation runs under `no_grad`; positivity clamping is active on `q` and
   `tcwv`.
8. Update `docs/SPEC.md` with the parameter table and the freezing rules.

## Definition of Done

- [ ] Validation wrapped in `torch.no_grad()`; loss unchanged at `k = 1` (both
      values pasted); memory at `k = 4` measured or the reduction argued
- [ ] `_ROLLOUT_CLAMP` replaced by Aurora's `positive_surf_vars` /
      `positive_atmos_vars`; any retained entry named and justified
- [ ] `clamp_at_first_step` decision made and reasoned in the result file
- [ ] `warmup` freezing mode implemented and tested
- [ ] Parameter table for `model_type: full` × four modes: total, trainable,
      frozen, %, per-module breakdown — pasted
- [ ] `warmup` trainable count stated with an explicit comment on whether the
      surface is large enough, flagged for K1 to answer
- [ ] LoRA configuration recorded: rank, mode, adapted modules, `lora_steps`;
      the per-step-adapter option recorded as later-campaign material
- [ ] Per-mode trainable-set tests pass, matching by parameter name
- [ ] `docs/SPEC.md` updated
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] Discontinuity section — parameter counts vs
      `../refactor/handoff-2026-09-refactor.md` §5.2
- [ ] `docs/PROJECT_STATE.md` updated
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Changing the LoRA rank or insertion points.** Record the option; a later
  campaign ablates it. Changing it here makes K1–K4 uninterpretable against J4.
- Enabling full BPTT (`backprop: "full"`).
- The MJO head — J5 owns it.
- Any loss change.
- Touching the DDP-collective grad guard. Lesson 2, and it works.
