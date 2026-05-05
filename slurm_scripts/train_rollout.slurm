#!/bin/bash
#SBATCH -A m4946_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -c 8
#SBATCH --mail-user=kieran.malandain@yale.edu
#SBATCH --job-name=aurora_mjo_rollout
#SBATCH --output=slurm_logs/train_rollout_%j.out
#SBATCH --error=slurm_logs/train_rollout_%j.err

# 1. Initialize Conda cleanly (bypassing NERSC global module conflicts)
source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh

# 2. Activate your GPU-enabled environment using the explicit scratch path
conda activate /pscratch/sd/k/kam352/conda_envs/aurora_mjo

# 3. Environment settings for NERSC
export HF_HOME=/tmp/hf_home_${USER}
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

echo "Starting Phase 2 Rollout+LoRA Training (4x GPU DDP)..."
echo "Node: $(hostname), GPUs: $SLURM_GPUS_PER_NODE"

# Check for latest checkpoint and resume if available
CKPT_DIR="checkpoints/phase2_rollout_lora"
RESUME_FLAG=""
if [ -d "$CKPT_DIR" ]; then
    LATEST_CKPT=$(ls -t ${CKPT_DIR}/*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "Resuming from checkpoint: $LATEST_CKPT"
        RESUME_FLAG="--resume $LATEST_CKPT"
    fi
fi

# Launch with torchrun for DDP across 4 GPUs on 1 node
srun torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    train.py --config configs/phase2_rollout_lora.yaml $RESUME_FLAG
