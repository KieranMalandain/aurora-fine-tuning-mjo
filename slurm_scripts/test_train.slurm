#!/bin/bash
#SBATCH -A m4946_g
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --mail-user=kieran.malandain@yale.edu
#SBATCH --job-name=aurora_mjo_test
#SBATCH --output=slurm_logs/test_train_%j.out
#SBATCH --error=slurm_logs/test_train_%j.err
#
# Fast smoke-test verification on NERSC Perlmutter.
#
# Usage:
#   sbatch slurm_scripts/test_train.slurm
#   MODE=lora sbatch slurm_scripts/test_train.slurm
#
# Runs one synthetic training step without requiring CFS data or multi-GPU DDP.

set -euo pipefail

# Ensure working directory is repo root
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

mkdir -p slurm_logs

# Sourced NERSC environment
source slurm_scripts/env.sh

MODE="${MODE:-baseline}"
CONFIG="${CONFIG:-configs/unified.yaml}"

echo "[test_train] Starting smoke test: MODE=${MODE} CONFIG=${CONFIG} JOB=${SLURM_JOB_ID:-manual} NODE=$(hostname)"

srun uv run --frozen python run.py train \
    --config "${CONFIG}" \
    --mode "${MODE}" \
    --smoke-test \
    "$@"
