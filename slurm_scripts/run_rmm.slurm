#!/bin/bash
#SBATCH --job-name=calc_rmm
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/rmm_%j.out
#SBATCH --error=slurm_logs/rmm_%j.err

# --- 1. SETUP ---
echo "======================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================================"

mkdir -p slurm_logs

module load miniconda
# module load CUDA/12.1.1 # Optional, usually not needed if installed in Conda

eval "$(conda shell.bash hook)"
conda activate aurora_mjo

unset LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 2. EXECUTION ---
echo "\n--- GPU Information ---"
nvidia-smi
echo "-----------------------\n"

echo "--- Running RMM Skill Calculation ---"

# -u for unbuffered output so you see print statements immediately
python -u scripts/vit_scripts/calc_rmm_skill.py

# --- 3. COMPLETION ---
echo "\n======================================================"
echo "Job finished at $(date)"
echo "======================================================"