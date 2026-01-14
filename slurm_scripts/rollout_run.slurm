#!/bin/bash
#SBATCH --job-name=rollout-run
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/rollout_%j.log

# --- 1. SETUP ---
echo "======================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Confirm Update 1"
echo "======================================================"

mkdir -p slurm_logs

module load miniconda
module load CUDA/12.1.1 

eval "$(conda shell.bash hook)"
conda activate aurora_mjo

unset LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 2. EXECUTION ---
echo "\n--- Running the rollout script... ---\n"

python scripts/long_rollout.py

# --- 3. COMPLETION ---
echo "\n======================================================"
echo "Job finished at $(date)"
echo "======================================================"