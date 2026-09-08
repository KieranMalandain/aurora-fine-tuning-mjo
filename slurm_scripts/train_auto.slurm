#!/bin/bash
#SBATCH -A m4946_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 11:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 64
#SBATCH --signal=B:USR1@1800
#SBATCH --mail-user=kieran.malandain@yale.edu
#SBATCH --job-name=aurora_mjo
#SBATCH --output=slurm_logs/train_%j.out
#SBATCH --error=slurm_logs/train_%j.err
#
# Unified auto-resuming training job for NERSC Perlmutter.
#
#   MODE=baseline sbatch slurm_scripts/train_auto.slurm
#   MODE=combined sbatch slurm_scripts/train_auto.slurm
#
# Design (each line maps to an observed failure):
#   -t 11:30:00           : walltime; python's own time_limit_hours=11.0 stops
#                           training with 30 min to spare for the final save.
#   --signal=B:USR1@1800  : backup path — SLURM signals the batch shell 30 min
#                           before timeout; we forward USR1 to the trainer,
#                           which checkpoints and exits 99.
#   -c 64                 : the old script requested 8 CPUs for 4 GPUs and 16
#                           dataloader workers → thrashing. Perlmutter GPU
#                           nodes have 64 physical cores; take them all.
#   DONE marker           : chained jobs exit 0 immediately once a phase has
#                           finished, so over-provisioned chains are harmless.

set -uo pipefail

MODE="${MODE:-baseline}"
CONFIG="${CONFIG:-configs/unified.yaml}"
CKPT_DIR="checkpoints/${MODE}"
mkdir -p slurm_logs "${CKPT_DIR}"

# Phase already finished? (trainer writes DONE on completion) → no-op.
if [ -f "${CKPT_DIR}/DONE" ]; then
    echo "[chain] ${CKPT_DIR}/DONE exists — phase '${MODE}' complete. Exiting."
    exit 0
fi
# Manual kill-switch for the whole chain: touch STOP_TRAINING
if [ -f "STOP_TRAINING" ]; then
    echo "[chain] STOP_TRAINING present — exiting without training."
    exit 0
fi

# 1. Conda (bypassing NERSC global module conflicts)
source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
conda activate /pscratch/sd/k/kam352/conda_envs/aurora_mjo

# 2. NERSC environment
export HF_HOME=${PSCRATCH}/hf_home            # PERSISTENT cache (was /tmp → re-download every job)
export HF_HUB_ETAG_TIMEOUT=60
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # anti-fragmentation (cuBLAS/OOM fix)
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 1000 ))     # avoid port collisions between chained jobs

echo "[chain] MODE=${MODE}  CONFIG=${CONFIG}  JOB=${SLURM_JOB_ID}  NODE=$(hostname)"

# 3. Launch DDP under srun in the background so this shell can catch USR1
#    and forward it to every python rank.
srun --cpu-bind=cores torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    train.py --config "${CONFIG}" --mode "${MODE}" --resume auto &
SRUN_PID=$!

forward_usr1() {
    echo "[chain] USR1 from SLURM (T-30min) — forwarding to training processes"
    # Signal the whole step; torchrun propagates to workers, whose Python
    # handler checkpoints and exits 99.
    scancel --signal=USR1 "${SLURM_JOB_ID}.0" 2>/dev/null || kill -USR1 ${SRUN_PID} 2>/dev/null
}
trap forward_usr1 USR1

wait ${SRUN_PID}
RC=$?
echo "[chain] training exited with code ${RC}"

# Exit-code contract with the trainer:
#   0  → phase complete (DONE written). Chained jobs will no-op.
#  99  → clean timeout save. Chained job resumes via --resume auto.
# else → crash; an emergency checkpoint was written if it was OOM/cuBLAS,
#        so letting the chain retry is still safe and usually productive.
exit ${RC}
