# Task: Populate NERSC Slurm Scripts

## Objective

Populate the currently empty `slurm_scripts/train.slurm` and `slurm_scripts/eval.slurm` files with the correct NERSC Slurm directives to allow Phase 1 training and evaluation on the Perlmutter cluster.

## Working rules
- Read only the files listed below unless you discover a direct dependency.
- If you need to expand scope, stop and report why.
- Make the smallest possible coherent patch.
- Keep all changes confined to the modify list unless explicitly authorized.
- Ensure conda/mamba environment `aurora_mjo` is activated before running Python code (directory `/pscratch/sd/k/kam352/conda/envs/aurora_mjo/bin/python`).

## Read

- `docs/compute-environments.md` (if available for NERSC details)
- `train.py`
- `docs/nersc-slurm-doc.md`

## Modify

- `slurm_scripts/train.slurm`
- `slurm_scripts/eval.slurm`

## Do not touch

- Python training logic in `train.py` or `src/`

## Expected output

- `slurm_scripts/train.slurm` contains standard NERSC directives, loads the `aurora_mjo` conda environment, and runs `train.py` using `configs/phase1_baseline.yaml`.
- `slurm_scripts/eval.slurm` is similarly configured but runs an evaluation-only command (if supported) or is structured to resume from a checkpoint for validation.

## Steps

1. Edit `slurm_scripts/train.slurm` to include `#SBATCH` directives for a standard GPU job on NERSC Perlmutter (e.g., `-C gpu`, `-q regular`, `-t 24:00:00`, `-N 1`).
2. Add the environment setup lines: `source /pscratch/sd/k/kam352/conda/etc/profile.d/conda.sh` and `conda activate aurora_mjo`.
3. Add the execution command: `python train.py --config configs/phase1_baseline.yaml`.
4. Edit `slurm_scripts/eval.slurm` to include similar directives, but perhaps targeting a debug or shorter queue, and executing the evaluation workflow if a dedicated script exists, or using `train.py` with evaluation flags if applicable.

## Risks & Reminders

- Always verify the NERSC host and specific GPU requirements (e.g., A100 vs. others).
- Do not hardcode paths to checkpoints unless strictly necessary; use environment variables or CLI arguments.
