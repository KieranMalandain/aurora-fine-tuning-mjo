# Evaluation Specification

## Purpose

This document defines the authoritative evaluation protocol for MJO forecasting in this repository.

Agents must follow this document when implementing or modifying evaluation code.

## Core Principle

The repository's scientific target is MJO forecast skill, not just generic atmospheric reconstruction quality.

Therefore, evaluation must include explicit MJO phase-space metrics, not only gridpoint losses.

## Dataset Splits

Use chronological splits only.

Proposed default:
- Train: 1980–2015
- Validation: 2016–2019
- Test: 2020–2023

[EDIT ME]
Replace with actual intended years if different.

## Normalization Rules

- Compute normalization statistics using training years only.
- Do not use validation or test years to compute train-time statistics.
- If climatology/anomaly references are used for RMM calculation, define them using training-period references only unless the scientific protocol requires another standard.

## RMM Targets

The repository intends to evaluate forecasts in RMM space using fields such as:
- OLR
- U850
- U200

The evaluation pipeline should:
1. compute anomalies
2. project anomalies onto the chosen EOF basis
3. derive RMM1 and RMM2
4. compute amplitude and phase

[EDIT ME]
Specify exact anomaly/climatology/EOF details if already decided:
- Wheeler-Hendon standard implementation?
- custom ERA5-based EOF recomputation?
- fixed training-period EOF basis?
- seasonal-cycle removal method?

## Primary Metrics

- bivariate correlation coefficient vs lead time
- RMSE of RMM1 and RMM2 vs lead time
- amplitude error vs lead time
- phase error vs lead time

## Secondary Metrics

- skill for active MJO cases only
- skill by MJO phase
- skill by season
- skill across Maritime Continent crossing cases
- gridded losses for selected fields

## Active MJO Definition

Default:
- active if amplitude > 1.0

[EDIT ME]
If you use a different threshold, specify it.

## Forecast Horizons

Default evaluation should support:
- short-range lead checks
- medium rollout horizons
- subseasonal horizons up to 30 days or longer

## Required Validation Outputs

Each evaluation run should produce:
- summary table by lead time
- plots of skill vs lead time
- phase-space diagnostics if available
- metadata including checkpoint, config, years, variables, and commit hash if possible

## Pitfalls to Avoid

- incorrect anomaly calculation
- mixing train and test climatology
- comparing incompatible RMM definitions
- claiming skill based only on selected events
- relying only on qualitative OLR maps

## Evaluation Pipeline & RMM Protocol

- **`compute_rmm.py` Status:** Currently MISSING / PLACEHOLDER. Needs to be built immediately to generate training targets for the MJO Head.
- **RMM Implementation Rules:** 
  - Must compute Empirical Orthogonal Functions (EOFs) for OLR, U850, and U200.
  - **CRITICAL:** EOFs and climatological means must be computed using *only* the training split (e.g., 1980–2015) to prevent data leakage into the validation/test sets.
- **Benchmark Protocol:** Mirror the Wheeler & Hendon (2004) methodology. 
- **Target Metrics:** 
  - Bivariate Anomaly Correlation Coefficient (COR/ACC) for RMM1 and RMM2 at lead times 1 through 30 days. Target is ACC > 0.5 at day 30.
  - Phase error and Amplitude error.
  - Skill conditional on "Active MJO" events only (Initial Amplitude > 1.0).