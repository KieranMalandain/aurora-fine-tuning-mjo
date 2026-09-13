# Command-Line Interface (CLI) Specification

This document specifies the command-line interface for `aurora-fine-tuning-mjo`.
`run.py` serves as the unified, thin entry point for all training, evaluation, configuration inspection, and utility operations.

---

## 1. Top-Level Usage

```bash
uv run python run.py [COMMAND] [OPTIONS]
```

### Commands Overview

| Command | Description | Implementation Status |
| :--- | :--- | :--- |
| `train` | Train or fine-tune Aurora for MJO prediction | Full CLI & DDP support |
| `show-config` | Resolve mode overlays and overrides, printing canonical JSON | Full Pydantic v2 validation |
| `evaluate` | Evaluate forecast checkpoints against Wheeler–Hendon RMM targets | Wraps `scripts/evaluate_mjo.py` |
| `norm-stats` | Compute dataset normalization statistics over training years | Wraps `scripts/calc_norm_stats.py` |

---

## 2. Command Flags & Options

### 2.1 `run.py train`

Trains the model using Distributed Data Parallel (DDP) across available GPUs.

```bash
uv run python run.py train --mode baseline [OPTIONS]
```

| Option | Flag | Type | Default | Description / Validity |
| :--- | :--- | :--- | :--- | :--- |
| **Config File** | `--config` | `Path` | `configs/unified.yaml` | Path to unified YAML configuration file. Must exist and be readable. |
| **Operational Mode** | `--mode` | `str` | `None` *(Required)* | Mode overlay: `baseline`, `physics_informed`, `lora`, or `combined`. |
| **Override Key** | `--override` | `list[str]` | `[]` *(Repeatable)* | Dot-notation override (e.g. `--override training.optimizer.lr=1e-5`). Parsed via YAML. |
| **Smoke Test** | `--smoke-test` | `bool` | `False` | Executes a single synthetic step on CPU without requiring CFS data or GPUs. |
| **Resume Mode** | `--resume` | `str` | `auto` | Resumption strategy: `auto` (finds latest in `save_dir`), `none` (fresh), or explicit checkpoint path. |

#### Examples
```bash
# Standard baseline training on 4 GPUs
uv run python run.py train --mode baseline

# Resuming a LoRA run from the latest checkpoint
uv run python run.py train --mode lora --resume auto

# Quick single-step synthetic smoke test
uv run python run.py train --mode baseline --smoke-test

# Overriding learning rate and batch size on the command line
uv run python run.py train --mode combined --override training.optimizer.lr=5e-5 --override training.batch_size=2
```

---

### 2.2 `run.py show-config`

Resolves the configuration by loading `configs/unified.yaml`, applying the requested `--mode` overlay, applying all `--override` modifications, and validating through Pydantic v2. Outputs canonical JSON to standard output.

```bash
uv run python run.py show-config --mode [MODE] [OPTIONS]
```

| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Config File** | `--config` | `Path` | `configs/unified.yaml` | Path to unified configuration file. |
| **Operational Mode** | `--mode` | `str` | `None` *(Required)* | Mode overlay to resolve. |
| **Override Key** | `--override` | `list[str]` | `[]` *(Repeatable)* | Dot-notation configuration overrides. |

#### Example
```bash
uv run python run.py show-config --mode physics_informed
```

---

### 2.3 `run.py evaluate`

Evaluates model checkpoints across sub-seasonal forecast lead times against Wheeler–Hendon RMM index targets.

```bash
uv run python run.py evaluate --mode baseline [OPTIONS]
```

| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Config File** | `--config` | `Path` | `configs/unified.yaml` | Unified configuration file. |
| **Operational Mode** | `--mode` | `str` | *(Required)* | Mode overlay evaluated. |
| **Checkpoint** | `--checkpoint` | `str` | `latest` | `latest` (scans mode save dir) or explicit path to `.pt` checkpoint file. |
| **Target Dataset** | `--targets` | `Path` | `data/rmm_targets.nc` | NetCDF file containing ground-truth RMM targets. |
| **Basis File** | `--basis` | `Path` | `data/rmm_basis.npz` | Numpy archive holding precomputed EOF eigenvectors. |
| **Output Directory**| `--out-dir` | `Path` | `evaluation/mjo_skill` | Directory where evaluation metric curves and plots are stored. |
| **Smoke Test** | `--smoke-test` | `bool` | `False` | Executes synthetic evaluation harness without requiring model checkpoints or CFS data. |

#### Examples
```bash
# Evaluate latest checkpoint for baseline mode
uv run python run.py evaluate --mode baseline --checkpoint latest

# Run evaluation smoke test
uv run python run.py evaluate --mode baseline --smoke-test
```

---

### 2.4 `run.py norm-stats`

Helper entry point that routes users to `scripts/calc_norm_stats.py` to compute dataset statistics over training years.

```bash
uv run python run.py norm-stats --years 1980 2015
```

| Option | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Year Range** | `--years` | `tuple[int, int]` | `(1980, 2015)` | Start and end years (inclusive) for normalisation statistics. |
| **Config File** | `--config` | `Path` | `configs/unified.yaml` | Unified configuration file. |

*Note: Per design rules (`scripts/` is run-never-imported), `norm-stats` instructs the user to run `scripts/calc_norm_stats.py` directly for large batch calculations.*

---

## 3. Configuration Validity Matrix

Configuration is validated once, at the boundary, into a strongly-typed `Config` object using Pydantic v2. Invalid combinations fail immediately at load time with actionable remediation hints. All schemas enforce `extra="forbid"`, ensuring that mistyped override keys (e.g. `--override training.optimzer.lr=1e-5`) fail loudly rather than executing silently with defaults.

| Invalid Combination | Enforced Constraint & Remediation Message |
| :--- | :--- |
| `model.gradient_checkpointing: true` | **Strictly forbidden.** Deterministically triggers an illegal memory access crash on Perlmutter A100s. Full 1.3B model requires checkpointing to fit on 4×A100; use `model_type: small` instead. |
| `data.use_dummy: true` | **Deprecated.** The dummy dataset was removed during repository consolidation; use `--smoke-test` for CPU/synthetic verification. |
| `model.norm_stats.msl` absent | **Strictly required.** Removing MSL normalisation stats restores a −36 σ surface pressure proxy input and causes 100% non-finite validation loss (Lesson 1). |
| `training.rollout.enabled: true` with `backprop: "full"` | Full-BPTT rollout requires gradient checkpointing which crashes on Perlmutter; only `backprop: "detached"` has been verified crash-free (see `probe_ima_matrix.py`). |
| `training.rollout.start_steps > rollout.max_steps` | Nonsensical curriculum ordering. `start_steps` must be $\le$ `max_steps`. |
| `data.real.val_years` overlapping `train_years` | **Chronological splits only.** Prevents evaluation leakage (the 54,060 sample leakage class). |
| `data.real.test_years` overlapping `train_years` or `val_years` | **Chronological splits only.** Prevents evaluation leakage. |
| Two modes sharing a `checkpointing.save_dir` | Prevents silent checkpoint overwrite and metric file corruption between operational modes. |
| `model.mjo_head.enabled: false` with `loss.mjo_head.enabled: true` | Prevents scoring a loss term against an MJO projection head that does not exist. |
| Unrecognized key path (e.g. `training.optimzer.lr`) | Extra inputs are forbidden (`extra="forbid"`). Rejects typos at load time before job execution begins. |

---

## 4. Standalone Scripts & `--help` Status

Per the repository layout rules, scripts located in `scripts/` are run directly and never imported.

### Scripts Supporting `--help`
The following 13 scripts implement formal CLI argument parsing via `argparse` or `typer` and exit 0 upon `--help`:
- `scripts/check.py`: The single repo verification gate (`--fast`, `--fix`).
- `scripts/calc_norm_stats.py`: Computes channel mean and std for ERA5 variables.
- `scripts/compute_rmm.py`: Computes climatology, anomalies, and Wheeler–Hendon EOFs.
- `scripts/evaluate_mjo.py`: Evaluates MJO forecast skill across lead times.
- `scripts/diagnose_val_nan.py`: Diagnostic tool characterizing first-batch non-finite inputs.
- `scripts/explore_nersc_data.py`: Inspects NetCDF coordinates and attributes on CFS.
- `scripts/inspect_vars.py`: Prints shape and summary statistics for specified NetCDF variables.
- `scripts/probe_ima_matrix.py`: Searches configuration space for illegal memory access triggers.
- `scripts/probe_model_size.py`: Measures GPU VRAM consumption for small vs. full Aurora.
- `scripts/scan_for_bad_values.py`: Scans ERA5 NetCDF archives for NaN, Inf, or fill values.
- `scripts/smoke_test_freeze.py`: Validates model parameter freezing and LoRA parameter status.
- `scripts/smoke_test_rollout.py`: Validates autoregressive state advancement.
- `scripts/test_num_workers.py`: Measures DataLoader worker throughput scaling.

### Scripts Lacking CLI Options (Non-Documentable Flags)
The following 5 scripts are hardcoded execution utilities that lack formal `--help` flag handling (`NOHELP` in C3 audit):
- `scripts/download_slt.py`: Downloads static soil type file (`slt_data.nc`) from Hugging Face.
- `scripts/plot_loss.py`: Plots training and validation loss curves from metric JSON lines.
- `scripts/smoke_test_mjo_head.py`: Direct Python assertion script verifying MJO head initialization.
- `scripts/verify_dataset_loader.py`: Direct script printing dataset initialization reports.
- `scripts/verify_shapes.py`: Direct script asserting tensor shapes against domain priors.

---

## 5. Deprecated Entry Points

- `train.py`: Preserved as an 8-line backward compatibility shim for in-flight SLURM jobs. Delegates directly to `run.py train` while preserving all command-line arguments:
  ```bash
  python train.py --mode baseline [ARGS]  # Emits deprecation notice and execs run.py train
  ```
