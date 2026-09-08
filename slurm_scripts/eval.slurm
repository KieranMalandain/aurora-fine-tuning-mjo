#!/bin/bash
#SBATCH -A m4946_g
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mail-user=kieran.malandain@yale.edu
#SBATCH --gpus-per-task=1
#SBATCH --job-name=aurora_mjo_eval
#SBATCH --output=slurm_logs/eval_%j.out
#SBATCH --error=slurm_logs/eval_%j.err

# Setup conda environment (implicitly via absolute path)
module load python

export HF_HOME=/tmp/hf_home_${USER}
export HDF5_USE_FILE_LOCKING=FALSE

echo "Starting Evaluation..."
# Note: train.py does not currently have a dedicated --eval-only flag. 
# Once implemented, the command should be updated to use it alongside --resume.
# Example: srun python train.py --config configs/phase1_baseline.yaml --eval-only --resume path/to/checkpoint.pt
srun /pscratch/sd/k/kam352/conda/envs/aurora_mjo/bin/python train.py --config configs/phase1_baseline.yaml --override training.epochs=1 --override data.use_dummy=true --smoke-test
