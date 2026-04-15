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

These years are confirmed as the defaults used in `compute_rmm.py`.
Customize via the `TRAIN_YEARS`, `VAL_YEARS`, `TEST_YEARS` constants in that script.

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

**Implemented in `scripts/compute_rmm.py` (2026-04-13):**
- **Protocol:** Wheeler & Hendon (2004) methodology.
- **Variables:** OLR (`mtnlwrf`), U850, U200 — tropical mean over 15°S–15°N.
- **Seasonal-cycle removal:** Day-of-year (DOY) climatological mean computed on the training split (1980–2015); subtracted from all splits.
- **Normalization:** Each anomaly series is divided by its training-period standard deviation before EOF computation.
- **EOF basis:** Computed from the covariance matrix of the 3-element combined normalized anomaly vector, using only training-split data. The top two eigenvectors (EOF1, EOF2) define the RMM basis.
- **Projection:** Validation and test anomalies projected onto the frozen training-period EOF basis.
- **Outputs:** `data/rmm_basis.npz` (frozen basis), `data/rmm_targets.nc` (per-day RMM1, RMM2, amplitude, phase, split label).

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

The default threshold of 1.0 is used in `compute_rmm.py` (reported in summary output
and stored in `rmm_targets.nc` global attributes). No change from default.

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

- **`compute_rmm.py` Status:** **IMPLEMENTED** (2026-04-13). Generates `data/rmm_basis.npz` and `data/rmm_targets.nc`. Smoke-test available; full run requires NERSC access.
- **RMM Implementation Rules:** 
  - Must compute Empirical Orthogonal Functions (EOFs) for OLR, U850, and U200.
  - **CRITICAL:** EOFs and climatological means must be computed using *only* the training split (e.g., 1980–2015) to prevent data leakage into the validation/test sets.
- **Benchmark Protocol:** Mirror the Wheeler & Hendon (2004) methodology. 
- **Target Metrics:** 
  - Bivariate Anomaly Correlation Coefficient (COR/ACC) for RMM1 and RMM2 at lead times 1 through 30 days. Target is ACC > 0.5 at day 30.
  - Phase error and Amplitude error.
  - Skill conditional on "Active MJO" events only (Initial Amplitude > 1.0).