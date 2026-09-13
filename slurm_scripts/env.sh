#!/bin/bash
# slurm_scripts/env.sh — shared environment configuration for NERSC Perlmutter.
# Sourced by all batch scripts before launching python.

export HDF5_USE_FILE_LOCKING=FALSE
export HF_HOME=${PSCRATCH}/hf_home            # PERSISTENT cache (was /tmp → re-download every job)
export HF_HUB_ETAG_TIMEOUT=60
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # anti-fragmentation (cuBLAS/OOM fix)
