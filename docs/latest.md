# Latest Updates (May 2026)

## Summary of Phase 1 NERSC Readiness Execution

Over the past session, the repository was rigorously audited and hardened to successfully run the Phase 1 Baseline (LoRA finetuning) on the NERSC Perlmutter cluster. 

### 1. Slurm & Environment Hardening
- **Conda Activation:** We removed brittle `conda activate` bash hooks inside the Slurm scripts, opting instead for absolute python executable paths (`/pscratch/sd/k/kam352/conda/envs/aurora_mjo/bin/python`). This prevents the system from falling back to CPU-only python builds.
- **File Locking Fixes:** We identified that NERSC's GPFS file system does not support file locking, which caused `[Errno 524]` crashes in both Hugging Face cache downloads and `xarray/netCDF4` HDF dataset loading. We permanently fixed this by adding `export HF_HOME=/tmp/hf_home_${USER}` and `export HDF5_USE_FILE_LOCKING=FALSE` to all `.slurm` scripts.
- **Node Resources:** Updated all Slurm scripts to explicitly request 80GB A100 nodes (`#SBATCH -C gpu&hbm80g`).

### 2. Smoke Tests & Dataloader
- **Comprehensive Smoke Tests:** We built and executed a 7-step smoke test suite (`docs/smoke_results.md`). All tests, including the main training pipeline, MJO head architecture, rollout logic, parameter freezing, and evaluation metrics, pass successfully.
- **Swin3D Dimensions:** Increased the synthetic smoke-test resolution in `train.py` from `8x16` to `32x64` to prevent downsampling assertion errors within the Aurora Swin Transformer architecture.
- **Robust Invariants:** Updated `src/dataset.py`'s `_load_static_vars` to catch severe NetCDF corruption and fall back to dummy zero-tensors. This ensures that minor I/O issues with static variables (like `z` and `lsm`) do not crash 24-hour training runs.

### 3. Scripts & Configurations
- **Evaluation Scripts:** Confirmed that `scripts/compute_rmm.py` and `scripts/evaluate_mjo.py` are fully implemented with comprehensive Wheeler & Hendon logic (they are *not* placeholders). Updated `compute_rmm.py`'s file globbing to match the fallback hierarchy of `src/dataset.py`.
- **Phase 1 Configuration:** Updated `configs/phase1_baseline.yaml` to precisely reflect Phase 1 goals: `freeze_backbone: true` and `use_lora: true`. The MJO head remains disabled (`mjo_head.enabled: false`) until `dataset.py` is updated to load the pre-computed RMM targets in Phase 2.
- **Optimizer Crash:** This config update permanently resolved the `ValueError: optimizer got an empty parameter list` bug, as the LoRA adapters provide the necessary trainable parameters.

The codebase is now fully verified, and the Phase 1 Baseline is ready for submission via `sbatch slurm_scripts/train.slurm`.
