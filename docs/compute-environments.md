# Compute Environments

**Target Environment:** NERSC Perlmutter (National Energy Research Scientific Computing Center)  
**Hardware Configuration:** 1 node × 4 × NVIDIA A100 80GB SXM4 GPUs per job  
**Workload Manager:** SLURM  

---

## 1. Primary Production Environment: NERSC Perlmutter

All active development, verification, model fine-tuning, and sub-seasonal evaluation execute on NERSC Perlmutter.

### 1.1 Compute Nodes
- **Node Architecture:** HPE Cray EX liquid-cooled compute nodes.
- **CPUs:** AMD EPYC 7763 64-core processor (64 CPU cores per GPU node; `-c 64` requested for training jobs to ensure optimal dataloader worker throughput).
- **GPUs:** 4 × NVIDIA A100 80GB SXM4 with NVLink-3 interconnect (total 320GB HBM2e per node).
- **Interconnect:** HPE Slingshot 11 network (200 Gbps).

### 1.2 Storage Subsystems
- **Community File System (CFS):** Read-only project data storage at `/global/cfs/cdirs/m4946/...` hosting the 45-year 6-hourly ERA5 remapped NetCDF dataset.
  - *Constraint:* Access requires disabling parallel filesystem advisory locks: `HDF5_USE_FILE_LOCKING=FALSE` (handled in Python via `src/aurora_mjo/env.py` and sourced via `slurm_scripts/env.sh`).
- **Scratch Space (`/pscratch`):** High-speed Lustre scratch filesystem used for virtual environments, caches (`UV_CACHE_DIR`), checkpoints (`checkpoints/`), and static soil type data (`slt_data.nc`).
  - *Warning:* `/pscratch` is subject to periodic purge policies. Checkpoint and model artifacts must be synced to permanent project storage for long-term archival.

### 1.3 Environment & Package Management
- **`uv`:** Single package manager for dependency resolution, virtual environment management, and script execution.
- **Cache Redirection:** Because default `$HOME` directories lack POSIX file locking support (triggering `Errno 524`), all package, virtualenv, and pre-commit caches are redirected to `/pscratch`.

---

## 2. Historical & Retired Environments

- **Early Development Environments:** Early proof-of-concept testing was performed on a 1-month subset of ERA5 data (January 2015). That environment is retired. All production pipelines target NERSC Perlmutter exclusively.