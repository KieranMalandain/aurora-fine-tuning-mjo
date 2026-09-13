#!/bin/bash
#SBATCH -A m4946_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --mail-user=kieran.malandain@yale.edu
#SBATCH --job-name=aurora_mjo_eval
#SBATCH --output=slurm_logs/eval_%j.out
#SBATCH --error=slurm_logs/eval_%j.err
#
# MJO forecast skill evaluation on NERSC Perlmutter.
#
# Usage:
#   MODE=baseline sbatch slurm_scripts/eval.slurm
#   MODE=lora CHECKPOINT=checkpoints/lora/checkpoint_epoch_003.pt sbatch slurm_scripts/eval.slurm
#
# Resource justification:
#   --gpus-per-task=1 : evaluate_mjo.py evaluates on a single PyTorch device without DDP.
#   -c 32             : half a Perlmutter CPU socket (32 physical cores) for data loading/xarray.
#   -q regular        : 1 hour walltime in regular queue provides ample headroom for multi-day
#                       lead evaluation across validation years (unverified on Perlmutter; F2 benchmarks).

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
CHECKPOINT="${CHECKPOINT:-latest}"

echo "[eval] Starting evaluation: MODE=${MODE} CONFIG=${CONFIG} CHECKPOINT=${CHECKPOINT} JOB=${SLURM_JOB_ID:-manual} NODE=$(hostname)"

srun uv run --frozen python run.py evaluate \
    --config "${CONFIG}" \
    --mode "${MODE}" \
    --checkpoint "${CHECKPOINT}" \
    "$@"
