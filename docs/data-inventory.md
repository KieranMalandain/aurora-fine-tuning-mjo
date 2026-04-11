# Data Inventory

## Purpose

This document records where data currently lives, what subset is available where, and what assumptions the codebase currently makes.

## Current Storage Locations

### Yale Bouchet HPC
- Contains: local development sample / subset data
- Intended use: code development, smoke testing, small-scale validation
- Constraints: may not contain full historical span or all variables

### LANL / NERSC
- Contains: larger or full dataset
- Intended use: large-scale training and full evaluation
- Constraints: not yet the main active development environment

## Important Principle

Yale and LANL/NERSC should be treated as distinct data environments until parity is verified.

Agents must not assume:
- identical file paths
- identical preprocessing state
- identical variable availability
- identical normalization statistics
- identical chunking/layout
- identical climatology coverage

## Current Risks

- code written against Yale sample paths may not generalize to NERSC paths
- normalization statistics from Yale sample data may not be valid for full training
- RMM climatology built on partial data may differ from final evaluation protocol
- scripts may silently assume local file naming conventions

## Required Mitigations

- path roots must be configurable
- dataset manifests should be explicit
- train-time normalization stats should be recomputed on the full training dataset
- data validation scripts should compare variable coverage across environments

## Recommended Next Step

Create a machine-readable manifest for each environment:
- dataset root
- years available
- variables available
- pressure levels available
- temporal resolution
- format and chunking details

## Data Locations & Variables

### 1. NERSC / LANL (Primary Production Data)
- **Path:** `/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results`
- **Format:** 1-degree (180x360), 6-hourly, split into `StepXX/` subdirectories.
- **Surface Variables Present:** `t2*` (2t), `u10*` (10u), `v10*` (10v), `Ps` (used as proxy for `msl`), `mtnlwrf` (TTR/OLR), `tcwv`.
- **Atmos Variables Present:** `z*`, `q*`, `t*`, `u*`, `v*` (29 levels available, we slice 13).
- **Preprocessing:** No pre-processed outputs exist yet. All upsampling to 0.25-degree happens dynamically in `src/dataset.py`.

### 2. Yale Grace/Bouchet (Legacy / Investigation Data)
- **Path:** `/home/kam352/project_pi_ll2247/kam352/aurora-fine-tuning-mjo/era5_jan2015_daily`. _note that this was previously `/gpfs/gibbs/project/lu_lu/kam352/era5_jan2015_daily` before migrating to Yale Bouchet---do not use this second, deprecated path._
- **Format:** 0.25-degree (721x1440 - sliced to 720x1440), 6-hourly.
- **Status:** Used for Phase 1 micro-fine-tuning proof-of-concept. Do not use for production rollout training.