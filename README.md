# 🌍 Aurora Fine-Tuning for MJO Prediction

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&style=flat-square)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-managed-blueviolet?style=flat-square)](https://docs.astral.sh/uv/)
[![Target: NERSC Perlmutter](https://img.shields.io/badge/NERSC-Perlmutter_4x_A100-006699?style=flat-square)](https://www.nersc.gov/systems/perlmutter/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

Fine-tuning Microsoft Aurora (1.3B-parameter 3D Swin Transformer) for sub-seasonal prediction of the Madden–Julian Oscillation (MJO). Rather than predicting statistical anomaly indices via empirical regression, the system treats MJO forecasting as a physics-consistent initial-value prognostic simulation problem on 6-hourly global atmospheric states. The research target is achieving an RMM bivariate correlation skill of $r > 0.5$ at a 30-day forecast lead time.

---

## 🚀 Quickstart

All dependency and environment management is handled strictly by [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies into virtual environment
uv sync --all-groups

# 2. Install pre-commit hooks
uv run pre-commit install

# 3. Verify the complete verification gate (lockfile, ruff lint/format, pyrefly types, pytest)
uv run python scripts/check.py

# 4. Run an end-to-end synthetic training smoke test (no GPU or CFS data required)
uv run python run.py train --mode baseline --smoke-test
```

---

## 🖥️ Compute Environment & Data Lineage

- **Primary Compute Platform:** **NERSC Perlmutter**, 1 node × 4 × NVIDIA A100 80GB SXM4 under SLURM.
- **Upstream Data Archive:** ERA5 reanalysis data remapped to 1.0° (180 × 360), 6-hourly instantaneous, dynamically upsampled on GPU to Aurora's native 0.25° (720 × 1440) grid. Located read-only on the NERSC Community File System (CFS) at:
  ```text
  /global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results
  ```
  *(Attached strictly read-only; see [`docs/SPEC.md`](docs/SPEC.md) §3 for the read-only contract).*
- **Static Invariants:** Surface geopotential (`z`) and land-sea mask (`lsm`) from CFS; soil type (`slt`) loaded from `/pscratch/sd/k/kam352/Aurora/slt/slt_data.nc`.

---

## 🧭 Documentation Map

| Document | Purpose |
| :--- | :--- |
| **[`AGENTS.md`](AGENTS.md)** | Non-negotiable repository rules, verification gates, and hardware prohibitions. |
| **[`docs/SPEC.md`](docs/SPEC.md)** | Authoritative technical specification: domain background, tensor shapes, sample-count arithmetic, split definitions, the seven paid-for lessons, and upstream traps. |
| **[`docs/SETUP.md`](docs/SETUP.md)** | Step-by-step setup guide: local development, NERSC Perlmutter deployment, scratch cache configuration, and GPU smoke testing. |
| **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** | System architecture: the core processing spine (`dataset.py` → `model.py` → `loss.py` → `trainer.py` → `checkpoint.py`), layout rules, and third-party import collision guards. |
| **[`docs/CLI.md`](docs/CLI.md)** | Command-line interface reference for `run.py`, flag definitions, and the Configuration Validity Matrix. |
| **[`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md)** | Living project status: active milestones, test counts, open items, unverified assumptions, and the single next action. |
| **[`docs/git-policy.md`](docs/git-policy.md)** | Git branch and PR governance under the epic branch model. |
| **[`slurm_scripts/README.md`](slurm_scripts/README.md)** | Operational guide for multi-GPU training jobs, chained resumptions, and signal traps on Perlmutter. |
| **[`docs/archive/README.md`](docs/archive/README.md)** | Catalog of preserved pre-refactor analyses, metric logs, and historical documents. |