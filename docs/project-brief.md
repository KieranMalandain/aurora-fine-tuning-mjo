# Project Brief

## Objective

This repository develops an MJO-focused subseasonal forecasting system built on Microsoft Aurora.

The intended research direction is:

- Fine-tune Aurora on ERA5-based global 6-hourly physical fields.
- Inject MJO-critical variables, especially OLR/TTR and TCWV.
- Move from generic state forecasting toward explicit MJO-specific skill optimization.
- Add an MJO prediction head that predicts RMM1, RMM2, amplitude, and optionally active-MJO probability.
- Train using multi-step rollout objectives.
- Use LoRA for efficient long-horizon specialization.
- Add a moisture-centered auxiliary physics-informed loss only after the supervised baseline is stable.

## Current Status / Completed

The repository already contains code for:
- ERA5 ingestion and preprocessing
- normalization/statistics workflows
- Aurora fine-tuning
- custom losses
- training loop infrastructure
- SLURM submission
- **Phase 1 Audit**: The repository has been audited for Phase 1 Slurm readiness (optimizer configs fixed, NERSC Slurm scripts populated) and execution tasks have been generated and applied.

The following major Phase 2 milestones are now fully completed and integrated:
- Build or validate a correct RMM evaluation pipeline
- Add an explicit MJO head to the model
- Introduce rollout training
- Implement LoRA-based long-horizon specialization
- Add optional moisture-budget auxiliary loss

Agents should treat the existing codebase as authoritative:
- existing implementation details are real and should be inspected first
- roadmap documents are authoritative for future direction
- if the two conflict, the conflict should be surfaced explicitly before implementation

## Primary Success Metric

The main scientific target is lead-dependent MJO skill in RMM space on held-out years.

Key metrics include:
- RMM1 / RMM2 forecast quality
- bivariate correlation skill vs lead time
- amplitude error
- phase error
- active-MJO event skill
- seasonal and phase-conditioned skill

## Current Priorities

1. Audit the existing codebase against the current roadmap.

## Non-goals for now

- Do not add speculative architectural complexity without a corresponding evaluation plan.
- Do not prioritize visually sharp OLR fields over MJO benchmark skill.
- Do not claim benchmark improvement without actual evaluation runs.

## Human Notes

- **Target Environment:** NERSC Perlmutter (1 node × 4 × A100 80GB). All training runs here under SLURM; input paths under `/global/cfs/cdirs/...` are read-only defaults.

## Scientific Direction & Priorities (from GEMINI.md)

*Imported from GEMINI.md during task A4 to preserve original scientific priorities and guidelines.*

### What to optimize for
- Clean, reproducible experiment structure
- Correct RMM evaluation pipeline
- Time-split validation
- Minimal training/evaluation leakage
- Clear experiment configs and logs

### What to avoid
- Do not optimize only visual sharpness of OLR fields
- Do not introduce physics-informed losses before the baseline is stable
- Do not add speculative architecture changes without a measurable evaluation target
- Do not use random train/val splits
- Do not break remote training scripts casually

### Required outputs for substantive work
- Updated code
- Updated experiment config(s)
- Updated documentation in `docs/`
- Exact commands to run
- Risks / assumptions