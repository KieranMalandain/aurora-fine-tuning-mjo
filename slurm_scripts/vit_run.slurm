#!/bin/bash
#SBATCH --job-name=vit_mjo_fix
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/vit_fix_%j.out
#SBATCH --error=slurm_logs/vit_fix_%j.err

# --- 1. SETUP ---
echo "======================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================================"

mkdir -p slurm_logs

# Load modules (Keep it simple - usually Conda manages CUDA inside the env)
# If your conda env has torch+cuda installed, you often don't need the system CUDA module.
# Try commenting out the module load if you get version conflicts, 
# but for now, we will stick to what you had, just cleaner.
module load miniconda
# module load CUDA/12.1.1  <-- Optional: Try commenting this out if it fails again. 
                           # PyTorch usually ships its own CUDA runtime.

eval "$(conda shell.bash hook)"
conda activate aurora_mjo

# DEBUG: Check if the node actually lists the GPU
echo "\n--- GPU Information (nvidia-smi) ---"
nvidia-smi
echo "------------------------------------\n"

# Check if PyTorch can see it before running the main script
python -c "import torch; print(f'PyTorch detects: {torch.cuda.device_count()} GPUs. Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Unset library path to avoid conflicts between System CUDA and Conda Torch
unset LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 2. EXECUTION ---
echo "\n--- Running the ViT Training script... ---\n"

# python -u unbuffers output so you see logs in real-time in the .out file
python -u scripts/vit_scripts/train_mjo.py

# --- 3. COMPLETION ---
echo "\n======================================================"
echo "Job finished at $(date)"
echo "======================================================"