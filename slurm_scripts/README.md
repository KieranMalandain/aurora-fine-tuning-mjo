# SLURM Batch Execution on NERSC Perlmutter

This directory contains the SLURM batch scripts and chaining utilities for training and evaluating Microsoft Aurora fine-tuned on the Madden–Julian Oscillation (MJO) on NERSC Perlmutter (1 node × 4 × NVIDIA A100 80GB SXM4).

---

## 1. Quick Reference

| Script | Purpose | Queue & Resources | Typical Command |
|---|---|---|---|
| `train_auto.slurm` | Auto-resuming 4-GPU DDP training | `regular`, 1 node, 4 GPUs, 64 CPUs, 11.5 h | `MODE=baseline sbatch slurm_scripts/train_auto.slurm` |
| `submit_chain.sh` | Submits dependent job chains for multi-day plans | N/A (client script) | `./slurm_scripts/submit_chain.sh baseline 1` |
| `eval.slurm` | Lead-dependent MJO forecast skill evaluation | `regular`, 1 node, 1 GPU, 32 CPUs, 1.0 h | `MODE=baseline sbatch slurm_scripts/eval.slurm` |
| `test_train.slurm` | 1-step synthetic smoke test | `debug`, 1 node, 1 GPU, 32 CPUs, 10 min | `sbatch slurm_scripts/test_train.slurm` |
| `env.sh` | Shared environment exports sourced by all scripts | Sourced shell script | `source slurm_scripts/env.sh` |

---

## 2. Shared Environment (`env.sh`)

All scripts source `slurm_scripts/env.sh` before executing Python. This ensures uniform cluster configuration across interactive, batch, and test jobs:

- `HDF5_USE_FILE_LOCKING=FALSE`: Disables advisory file locking on CFS/Lustre parallel filesystems to prevent NetCDF `[Errno -101] HDF error` read exceptions.
- `HF_HOME=${PSCRATCH}/hf_home`: Directs Hugging Face model and weights caches to persistent scratch storage (avoiding re-downloading 1.3B-parameter models on every job run from `/tmp`).
- `HF_HUB_ETAG_TIMEOUT=60`: Extended timeout for Hugging Face Hub metadata queries under cluster network latency.
- `PYTHONUNBUFFERED=1`: Ensures unbuffered standard output and standard error streams so log lines appear in `slurm_logs/` in real time.
- `OMP_NUM_THREADS=8`: Sets OpenMP CPU threading limits to avoid thread oversubscription.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Enables memory allocation expandable segments to prevent CUDA memory fragmentation.

---

## 3. Scripts

### 3.1 `train_auto.slurm`

Unified auto-resuming training script for Perlmutter.

- **Execution:** Runs PyTorch DDP across 4 GPUs using `srun --cpu-bind=cores uv run --frozen torchrun --nproc_per_node=4 run.py train --config ${CONFIG} --mode ${MODE} --resume auto`.
- **Environment variables honoured:**
  - `MODE`: Configuration mode overlay (`baseline`, `physics_informed`, `lora`, `combined`). Default: `baseline`.
  - `CONFIG`: Configuration file path. Default: `configs/unified.yaml`.
- **Walltime & Signal Trap:**
  - Allocated walltime: `11:30:00`.
  - Python's internal time limit is configured to 11.0 hours (`training.time_limit_hours=11.0`), saving a final checkpoint and exiting before the SLURM wall clock expires.
  - Backup signal handling: `#SBATCH --signal=B:USR1@1800` sends `SIGUSR1` to the batch script 30 minutes before allocation expiration. The script catches `USR1` and forwards it via `scancel --signal=USR1 "${SLURM_JOB_ID}.0"` to all rank processes, prompting the trainer to save an emergency checkpoint and exit with code `99`.
- **CPU Allocation:** Requests `-c 64` (all 64 physical cores on a Perlmutter GPU node) to provide adequate core capacity for dataloader workers without thread thrashing.

### 3.2 `submit_chain.sh`

Submits self-resuming job chains for unattended execution.

- **Usage:**
  ```bash
  ./slurm_scripts/submit_chain.sh <mode> <n_jobs> [after_job_id]
  ```
- **Examples (48-hour campaign plan submitted in one command sequence):**
  ```bash
  J1=$(./slurm_scripts/submit_chain.sh baseline         1)
  J2=$(./slurm_scripts/submit_chain.sh physics_informed 1 $J1)
  J3=$(./slurm_scripts/submit_chain.sh lora             1 $J2)
  J4=$(./slurm_scripts/submit_chain.sh combined         2 $J3)
  ```
- **Chaining Semantics:** Each subsequent job in a chain is submitted with `--dependency=afterany:<prev_job_id>`. This guarantees that the next job starts whether the predecessor exited with code `0` (complete), `99` (clean timeout), or crashed (retry). Because `--resume auto` finds the latest valid checkpoint and the `DONE` marker immediately exits `0`, chained executions are idempotent and safe against over-provisioning.
- **Output:** Prints only the final submitted job ID to stdout, allowing easy composition in shell pipelines.

### 3.3 `eval.slurm`

Evaluates trained checkpoints for lead-dependent MJO prediction skill using Wheeler-Hendon RMM metrics.

- **Usage:**
  ```bash
  MODE=baseline sbatch slurm_scripts/eval.slurm
  MODE=lora CHECKPOINT=checkpoints/lora/checkpoint_epoch_004.pt sbatch slurm_scripts/eval.slurm
  ```
- **Environment variables honoured:**
  - `MODE`: Evaluation mode overlay. Default: `baseline`.
  - `CONFIG`: Configuration file. Default: `configs/unified.yaml`.
  - `CHECKPOINT`: Path to checkpoint or `latest` (searches `checkpoints/${MODE}`). Default: `latest`.
- **Resource Selection:**
  - 1 GPU (`--gpus-per-task=1`, `-n 1`): Evaluation is single-device without DDP.
  - 32 CPUs (`-c 32`): Allocates half a CPU socket for dataloading and xarray computations.
  - Walltime: `01:00:00` in `regular` queue provides safe headroom for multi-lead rollout evaluations across held-out years.

### 3.4 `test_train.slurm`

Fast cluster smoke test verifying that the GPU environment, CUDA drivers, dependencies, and model forward/backward pass work.

- **Usage:**
  ```bash
  sbatch slurm_scripts/test_train.slurm
  MODE=lora sbatch slurm_scripts/test_train.slurm
  ```
- **Resource Selection:**
  - 1 GPU, 32 CPUs, 10-minute walltime on the `debug` queue (`-q debug`).
  - Runs `uv run --frozen python run.py train --config configs/unified.yaml --mode baseline --smoke-test`.
  - Executes a single synthetic step without requiring access to CFS ERA5 files.

---

## 4. Exit-Code & Auto-Resume Contract

The training scripts and chained executions adhere to a strict exit-code contract:

| Exit Code | Meaning | Action by Chained Job |
|---|---|---|
| `0` | Phase complete. `checkpoints/${MODE}/DONE` was written. | Subsequent chained jobs detect `DONE` at startup and exit `0` within seconds without running training. |
| `99` | Clean timeout save. Saved checkpoint before wall clock expiry. | Next chained job resumes training from the latest checkpoint via `--resume auto`. |
| Any other non-zero | Unplanned error or crash. | Emergency checkpoint saved if caused by OOM or numerical issues; chained retry resumes from the last valid checkpoint. |

### Emergency Kill Switch (`STOP_TRAINING`)

To stop an active chain without needing to cancel individual SLURM jobs:
```bash
touch STOP_TRAINING
```
When `STOP_TRAINING` exists in the repository root, `train_auto.slurm` exits immediately with code `0` before starting Python or initializing GPUs. When ready to resume, delete the file:
```bash
rm -f STOP_TRAINING
```

---

## 5. Environment & Migration Notes

- **Package Manager:** All scripts run via `uv run --frozen python ...` against the committed `uv.lock`. Legacy Conda environment activation is retired.
- **Unified Configuration:** Legacy individual YAML configuration files (`phase1_baseline.yaml`, `phase2_physics.yaml`, etc.) have been retired in favor of `configs/unified.yaml` with `--mode <mode>`.
- **Synthetic Testing:** The deprecated `data.use_dummy=true` parameter is retired; synthetic validation is performed using `--smoke-test`.
