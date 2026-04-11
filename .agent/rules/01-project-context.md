---
trigger: always_on
---

# Project context

This project studies Madden-Julian Oscillation forecasting using Microsoft Aurora.

## Scientific direction
The repository is moving toward:
- Aurora + injected MJO variables
- MJO-specific prediction head
- rollout training
- LoRA adaptation
- physics-informed auxiliary loss only after baseline stabilization

## Current priorities
1. robust ERA5 data pipeline
  - Note that the data is stored off of Yale Bouchet and in LANL's NERSC cluster
  _ there are currently two dataloaders: `dummy_dataset.py`, which points to one month of ERA5 data that we have stored on Yale Bouchet, and `dataset.py`, which should be used as the 'real' file. We will use this when we are on the NERSC cluster. Within our training script and others, it will be useful to have a flag `DUMMY` and then select `if DUMMY: src.dummy_dataset \ else: src.dataset`
2. correct RMM target/evaluation pipeline
3. MJO head implementation
4. rollout curriculum
5. LoRA training
6. moisture-budget auxiliary loss

## Primary success metric
Lead-dependent MJO skill in RMM space on proper time-held-out validation and test years.

## Secondary metrics
- amplitude error
- phase error
- active-event skill
- stability under autoregressive rollouts