# Task: Integrate physics loss into rollout branch

## Objective
Update the temporary integration branch so that it incorporates the intended behavior from `agent/physics-loss` on top of the already-merged rollout training logic, without performing a direct git merge of `agent/physics-loss`.

## Read
- src/trainer.py
- src/loss.py
- src/model.py
- configs/
- .agent/tasks/<this-file>.md

## Reference branch
Use `agent/physics-loss` as a reference only. Inspect what it changed, but do not merge that branch directly.

## Modify
- src/trainer.py
- configs/ if strictly required

## Do not touch
- git history
- unrelated docs
- model freezing / LoRA code unless required for compatibility

## Required behavior
- Preserve the rollout-training structure already present in `src/trainer.py`
- Add the moisture-budget / physics-loss logic in the correct place inside the autoregressive rollout loop
- Keep all existing rollout metrics and loss accounting intact
- Ensure any returned loss dictionaries include the physics-loss term cleanly
- Keep config changes minimal and consistent with the current integration schema

## Working rules
- Treat `agent/physics-loss` as a design reference, not a merge target
- Make the smallest coherent patch
- If there is ambiguity, prefer preserving the rollout branch structure and porting over only the physics-loss logic

## Expected output
- Updated `src/trainer.py` on this temporary integration branch
- Minimal config adjustments if needed
- A short note summarizing exactly what was ported from `agent/physics-loss`
- A smoke-test command