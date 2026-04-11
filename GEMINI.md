# Project: Aurora fine-tuning for MJO forecasting

This repository develops a trainable MJO-focused forecasting system built on Microsoft Aurora.

## Primary objective
Optimize MJO-specific skill, not just generic weather reconstruction quality.

## Intended model direction
- Aurora backbone with injected OLR/TTR and TCWV inputs
- Explicit MJO head predicting RMM1, RMM2, amplitude, and active-MJO status
- Multi-step rollout training
- LoRA specialization for long-horizon optimization
- Optional moisture-budget auxiliary physics loss after the supervised baseline is stable

## What to optimize for
- Clean, reproducible experiment structure
- Correct RMM evaluation pipeline
- Time-split validation
- Minimal training/evaluation leakage
- Clear experiment configs and logs

## What to avoid
- Do not optimize only visual sharpness of OLR fields
- Do not introduce physics-informed losses before the baseline is stable
- Do not add speculative architecture changes without a measurable evaluation target
- Do not use random train/val splits
- Do not break remote training scripts casually

## Required outputs for substantive work
- Updated code
- Updated experiment config(s)
- Updated documentation in docs/
- Exact commands to run
- Risks / assumptions

## Git workflow
Follow `docs/git-policy.md`.

Never propose direct edits on `main`.
For parallel work, use isolated branches and isolated Git worktrees.
When suggesting merges, follow:
agent branch -> integration/antigravity -> main