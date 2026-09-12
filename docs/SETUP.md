# Setup & Environment Guide

This document describes how to set up, verify, and run the `aurora-fine-tuning-mjo` codebase for both local development and distributed execution on NERSC Perlmutter.

---

## 1. Quickstart: Clone to Green Gate

### Prerequisites
- **Git**
- **[`uv`](https://docs.astral.sh/uv/)** (version $\ge 0.4.0$): The single package and environment manager used in this repository. Conda, mamba, and bare pip are not supported.
- CPython 3.10 (managed and fetched automatically by `uv` via `.python-version`).

### Clone and Install
```bash
# 1. Clone repository
git clone git@github.com:KieranMalandain/aurora-fine-tuning-mjo.git
cd aurora-fine-tuning-mjo

# 2. Synchronize virtual environment with committed lockfile
uv sync --all-groups

# 3. Install git pre-commit hooks
uv run pre-commit install

# 4. Run the single verification gate
uv run python scripts/check.py
```

### Local Smoke Test
Verify the execution pipeline end-to-end using synthetic in-memory data on CPU (no GPU or CFS data required):
```bash
uv run python run.py train --mode baseline --smoke-test
```
This executes a full synthetic forward pass, loss calculation, backward pass, and parameter step in under 10 seconds.

---

## 2. NERSC Perlmutter Deployment

Training and evaluation execute on **NERSC Perlmutter** on 1 node × 4 × NVIDIA A100 80GB SXM4 GPUs under SLURM.

### 2.1 Installing `uv` on Perlmutter
If `uv` is not present in your path on Perlmutter login nodes, install it into `~/.local/bin`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Ensure `~/.local/bin` is in your `$PATH`.

### 2.2 Lustre / GPFS Cache Redirection (Critical)
> [!IMPORTANT]
> **Filesystem Locking Error 524:** Default user directories in `$HOME` reside on a filesystem that does not support POSIX `flock` operations. Running `uv` or `pre-commit` against default paths in `$HOME/.cache` triggers `OSError: [Errno 524] Unknown error 524`.

To resolve this, redirect all package and toolchain caches to `/pscratch` by adding the following exports to your `~/.bashrc`:

```bash
# Redirect uv cache and python installations to scratch
export UV_CACHE_DIR=/pscratch/sd/k/kam352/.cache/uv
export UV_PYTHON_INSTALL_DIR=/pscratch/sd/k/kam352/.local/share/uv/python
export UV_DATA_DIR=/pscratch/sd/k/kam352/.local/share/uv

# Redirect pre-commit and virtualenv caches
export PRE_COMMIT_HOME=/pscratch/sd/k/kam352/.cache/pre-commit
export VIRTUALENV_APP_DATA=/pscratch/sd/k/kam352/.cache/virtualenv
export GOCACHE=/pscratch/sd/k/kam352/.cache/go-build
export GOPATH=/pscratch/sd/k/kam352/go
```

### 2.3 Upstream Data Paths & Storage Locations
- **ERA5 Production Data (Read-Only CFS):**
  ```text
  /global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results
  ```
  This allocation is owned by another group. Do not write or touch permissions here.
- **Static Soil Type (`slt_data.nc`):**
  ```text
  /pscratch/sd/k/kam352/Aurora/slt/slt_data.nc
  ```
  Lives on purgeable scratch space. If missing, regenerate via `scripts/download_slt.py`.

### 2.4 Parallel Filesystem HDF5 Locking
Opening NetCDF4/HDF5 files on CFS parallel mounts without disabling advisory locking produces `OSError: [Errno -101] NetCDF: HDF error`.
- **Python-level Guard:** `src/aurora_mjo/env.py` automatically sets `HDF5_USE_FILE_LOCKING=FALSE` at Python process startup before C-libraries initialise.
- **Shell-level Guard:** `slurm_scripts/env.sh` exports `HDF5_USE_FILE_LOCKING=FALSE` for all batch and interactive SLURM jobs.

### 2.5 Running on GPU Compute Nodes

#### Submitting via Batch Queue
Submit the quick debug smoke test to verify node GPU allocations:
```bash
sbatch slurm_scripts/test_train.slurm
```

Submit full production training with automatic chained resumption:
```bash
sbatch slurm_scripts/train_auto.slurm
```

#### Running Interactively via `salloc`
To allocate an interactive 4×A100 GPU node for testing:
```bash
salloc -N 1 -C 'gpu&hbm80g' -q debug -t 00:30:00 --gpus-per-node=4 -c 64
```
Once the allocation opens on the compute node:
```bash
source slurm_scripts/env.sh
uv run --frozen python run.py train --mode baseline --smoke-test
```

### 2.6 Deployment Verification Status
Task A1 verified that `uv` cleanly resolves and executes PyTorch 2.5.1+cu121 on Perlmutter compute nodes, detecting all 4 NVIDIA A100 80GB GPUs. Task E3 updated all SLURM submission scripts to run through `uv run --frozen`. Both tasks reported `STATUS: GREEN`.

---

## 3. Routine Maintenance & Verification Commands

```bash
# Verify the entire verification gate (fast mode skips slow tests)
uv run python scripts/check.py --fast

# Automatically apply Ruff lint and formatting fixes
uv run python scripts/check.py --fix

# Run tests requiring the CFS data archive (only on Perlmutter)
uv run pytest -m needs_data

# Run tests requiring CUDA GPUs (only on GPU nodes)
uv run pytest -m needs_gpu

# Inspect resolved configuration for a mode without executing training
uv run python run.py show-config --mode baseline
```
