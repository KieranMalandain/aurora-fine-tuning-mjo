#!/bin/bash
#SBATCH -A m4946_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --mail-user=kieran.malandain@yale.edu
#SBATCH --job-name=aurora_mjo_train
#SBATCH --output=slurm_logs/train_%j.out
#SBATCH --error=slurm_logs/train_%j.err

# 1. Initialize Conda cleanly (bypassing NERSC global module conflicts)
source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh

# 2. Activate your GPU-enabled environment using the explicit scratch path
conda activate /pscratch/sd/k/kam352/conda_envs/aurora_mjo

# echo "Staging Hugging Face cache to node-local NVMe..."
# 3. Create a job-specific temp directory on the node's local storage
# LOCAL_HF_CACHE=/tmp/hf_home_${SLURM_JOB_ID}
# mkdir -p $LOCAL_HF_CACHE

# 4. Copy the permanent cache from scratch to the local NVMe
# rsync -a $PSCRATCH/hf_cache_master/ $LOCAL_HF_CACHE/

# 5. Point Hugging Face to the fast, local cache
# export HF_HOME=$LOCAL_HF_CACHE
export HF_HOME=/tmp/hf_home_${USER}
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

# 6. Force Hugging Face offline so it doesn't waste time checking the internet
# export HF_HUB_OFFLINE=1 

echo "Starting Phase 1 Baseline Training..."
# Reminder: verify configs/phase1_baseline.yaml points to the 1.3B model!
srun python train.py --config configs/phase1_baseline.yaml

# 7. Sync any newly downloaded models back to the master scratch cache before the job ends
# echo "Syncing cache back to persistent storage..."
# rsync -a $LOCAL_HF_CACHE/ $PSCRATCH/hf_cache_master/