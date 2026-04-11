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

## Current Status

The repository already contains code for:
- ERA5 ingestion and preprocessing
- normalization/statistics workflows
- Aurora fine-tuning
- custom losses
- training loop infrastructure
- SLURM submission

Some of this code is out of date relative to the current intended roadmap. In particular, the old code may reflect:
- single-step fine-tuning assumptions
- older loss designs
- no explicit MJO head
- incomplete or earlier-stage evaluation
- older cluster/storage assumptions

Agents should treat the existing codebase as partially authoritative:
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
2. Build or validate a correct RMM evaluation pipeline.
3. Add an explicit MJO head to the model.
4. Introduce rollout training.
5. Implement LoRA-based long-horizon specialization.
6. Add optional moisture-budget auxiliary loss.

## Non-goals for now

- Do not add speculative architectural complexity without a corresponding evaluation plan.
- Do not prioritize visually sharp OLR fields over MJO benchmark skill.
- Do not claim benchmark improvement without actual evaluation runs.

## Human Notes

- **Target Environment:** We are migrating compute from Yale Bouchet to NERSC Perlmutter. Agents should assume NERSC paths (`/global/cfs/cdirs/...`) are the default for new code.
- **Aurora Wrapper:** The wrapper is currently runnable for single-step physical prediction, but *lacks* the MJO Head and the rollout loop logic.
- **`compute_rmm.py`:** Is completely placeholder. Building this to generate `(RMM1, RMM2)` targets for our dataset is a blocking priority before we can train the MJO head.