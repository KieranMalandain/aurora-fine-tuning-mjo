#!/bin/bash
#SBATCH -A m4946_g
#SBATCH -C gpu&hbm80g
#SBATCH -q interactive
#SBATCH -t 00:10:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --mail-user=kieran.malandain@yale.edu
#SBATCH --job-name=aurora_mjo_test
#SBATCH --output=slurm_logs/test_train_%j.out
#SBATCH --error=slurm_logs/test_train_%j.err

# 1. Initialize Conda cleanly (bypassing NERSC global module conflicts)
source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh

# 2. Activate your GPU-enabled environment using the explicit scratch path
conda activate /pscratch/sd/k/kam352/conda_envs/aurora_mjo

# 3. Point Hugging Face to the fast, local cache
export HF_HOME=/tmp/hf_home_${USER}
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

echo "Starting Phase 1 Baseline TEST Training (1 epoch)..."
srun python train.py --config configs/phase1_baseline.yaml \
    --override training.epochs=1
