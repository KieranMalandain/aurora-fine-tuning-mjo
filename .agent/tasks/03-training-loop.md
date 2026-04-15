# Task: Training & Rollout Loop Setup

## Objective
Implement the missing core training logic and single-step to multi-step rollout curriculum.

## Working rules
- Read only the files listed above unless you discover a direct dependency.
- If you need to expand scope, stop and report why.
- Make the smallest possible coherent patch.
- Keep all changes confined to the modify list unless explicitly authorized.

## Read
- `docs/architecture.md`
- `docs/evaluation-spec.md`
- `src/model.py`
- `src/loss.py`
- `src/trainer.py`
- `train.py`
- `configs/phase1_baseline.yaml`

## Modify
- `train.py`
- `src/trainer.py`
- `configs/phase1_baseline.yaml`

## Do not touch
- dataset internals unless required for wiring
- evaluation code unless needed for trainer hooks
- model internals except for interface compatibility

## Expected output
- runnable baseline train script
- train/validate loop
- config-driven setup
- smoke-test command

## Steps
1. **Config setup**: Populate `configs/phase1_baseline.yaml` to include batch sizes, variable flags, and single-step baseline targets.
2. **Trainer Boilerplate**: Implement the `train_epoch` and `validate` loops in `src/trainer.py`, making sure to appropriately call `src.dataset` (with the dummy flag switch if needed).
3. **Loss Integration**: Import the existing functions from `src/loss.py` and hook them up for both grid loss and MJO head loss cleanly.
4. **Driver**: Flesh out `train.py` to parse arguments and initialize the trainer/model correctly.

## Risks & Reminders
- Keep the loop modular so that autoregressive *rollouts* and *LoRA integration* can easily be layered on top next.
- Avoid placing direct hardcoded constants inside `train.py`.
