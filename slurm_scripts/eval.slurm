#!/bin/bash
#SBATCH -A m4946_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 03:00:00
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
#   MODE=warmup sbatch slurm_scripts/eval.slurm
#   MODE=lora CHECKPOINT=checkpoints/lora/checkpoint_epoch_003.pt sbatch slurm_scripts/eval.slurm
#   MODE=warmup NUM_CASES=1 sbatch slurm_scripts/eval.slurm  # single-case shakedown
#
# Resource justification:
#   --gpus-per-task=1 : evaluate_mjo.py evaluates on a single PyTorch device without DDP.
#   -c 32             : half a Perlmutter CPU socket (32 physical cores) for data loading/xarray.
#   -t 03:00:00       : 3 hours walltime accommodates full 2016-2019 validation split at 5-day stride
#                       (292 cases x 120 steps x 0.221s ~ 2.15 GPU-hours, G2 benchmark).

set -euo pipefail

# Ensure working directory is repo root
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

mkdir -p slurm_logs

# Sourced NERSC environment
source slurm_scripts/env.sh

MODE="${MODE:-warmup}"
CONFIG="${CONFIG:-configs/unified.yaml}"
CHECKPOINT="${CHECKPOINT:-latest}"
INIT_STRIDE_DAYS="${INIT_STRIDE_DAYS:-5}"
MAX_LEAD_DAYS="${MAX_LEAD_DAYS:-30}"
OUT_DIR="${OUT_DIR:-evaluation/mjo_skill}"

echo "[eval] Starting evaluation: MODE=${MODE} CONFIG=${CONFIG} CHECKPOINT=${CHECKPOINT} STRIDE=${INIT_STRIDE_DAYS} JOB=${SLURM_JOB_ID:-manual} NODE=$(hostname)"

CMD=(uv run --frozen python run.py evaluate \
    --config "${CONFIG}" \
    --mode "${MODE}" \
    --checkpoint "${CHECKPOINT}" \
    --init-stride-days "${INIT_STRIDE_DAYS}" \
    --max-lead-days "${MAX_LEAD_DAYS}" \
    --out-dir "${OUT_DIR}")

if [ -n "${NUM_CASES:-}" ]; then
    CMD+=(--num-cases "${NUM_CASES}")
fi

if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
    srun "${CMD[@]}" "$@"
else
    "${CMD[@]}" "$@"
fi
