# Task: Autoregressive Rollout Training

## Objective
Implement multi-step rollout logic in `src/trainer.py` to allow the model to predict recursively $k$ steps into the future, computing losses at each step (or heavily weighting the final steps) to enable subseasonal horizon training.

## Working rules
- Read only the files listed above unless you discover a direct dependency.
- If you need to expand scope, stop and report why.
- Make the smallest possible coherent patch.
- Keep all changes confined to the modify list unless explicitly authorized.

## Read
- `docs/architecture.md`
- `src/trainer.py`
- `src/model.py`
- `configs/phase2_rollout.yaml` (create this inherited config)

## Modify
- `src/trainer.py`
- `configs/` (create a new phase2 experiment config)

## Do not touch
- Loss definitions
- Model architecture (MJO head logic)
- Baseline configurations (keep phase1 configs intact)

## Expected output
- `_compute_loss` in the trainer updated to loop over $k$ rollout steps recursively.
- Rollout curriculum configured (e.g., start at $T=1$, grow to $T=4$ across epochs).
- Smoke-test command validating a 2-step rollout execution.

## Risks & Reminders
- Ensure that updated predictions properly replace the `atmos_vars` and `surf_vars` prior to the subsequent forward pass while advancing the time tags correctly.
