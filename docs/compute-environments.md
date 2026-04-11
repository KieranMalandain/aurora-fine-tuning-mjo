# Compute Environments

## Overview

This project is currently developed across multiple compute environments.

## Environment A: Yale Bouchet HPC

Role:
- primary code editing and iteration environment for now

Typical use:
- code inspection
- smoke tests
- short debug runs
- small-sample experiments

Risks:
- data subset may not represent final training distribution
- local assumptions may not hold on LANL/NERSC

## Environment B: LANL / NERSC

Role:
- eventual large-scale training and evaluation environment

Typical use:
- full-data runs
- long rollouts
- full climatology/statistics generation
- final experiments

Risks:
- project not yet fully migrated
- scripts may need path, scheduler, environment, and storage adaptation

## Engineering Rule

All code should be written so that:
- paths are configurable
- environment-specific values are externalized
- training/eval scripts do not hardcode Yale-specific assumptions

## Configuration Guidance

Prefer environment-specific configuration layers for:
- data roots
- checkpoint roots
- output roots
- temporary cache paths
- SLURM/account settings

## Human Notes

At this point, we are not setup on the LANL NERSC cluster. More information will be provided here, for example the points below, once the human has completed that step:

[EDIT ME]
Add:
- scheduler/account differences
- conda/module differences
- storage mount differences
- whether internet access differs