# Command-Line Interface (CLI) Specification

This document specifies the command-line interface for `aurora-fine-tuning-mjo`.
`run.py` is the unified, thin entry point for all training, evaluation, and utility operations.

> [!NOTE]
> This is a stub authored in task C2. Tasks C3 (script consolidation) and E2 (config validation) will expand it, and F1 completes it.

---

## 1. Top-Level Usage

```bash
uv run python run.py [COMMAND] [OPTIONS]
```

### Commands

| Command | Description | Status |
| --- | --- | --- |
| `train` | Fine-tune Aurora for MJO prediction | Live (C2) |
| `show-config` | Resolve mode overlays/overrides and print canonical JSON | Live (C2) |
| `evaluate` | Evaluate model checkpoints on MJO metrics | Reserved (C3) |
| `norm-stats` | Compute dataset normalization statistics | Reserved (C3) |

---

## 2. Command Flags

### `run.py train`

| Option | Flag | Type | Default | Description / Validity |
| --- | --- | --- | --- | --- |
| Configuration file | `--config` | `Path` | `configs/unified.yaml` | Path to unified YAML configuration file. Must exist and be readable. |
| Operational mode | `--mode` | `str` | `None` (required for unified) | Mode overlay to apply: `baseline`, `physics_informed`, `lora`, or `combined`. |
| Override parameter | `--override` | `list[str]` | `[]` (repeatable) | Dot-notation config override (e.g. `--override training.optimizer.lr=1e-5`). Coerced via `yaml.safe_load`. |
| Smoke test | `--smoke-test` | `bool` | `False` | Executes a single synthetic step on CPU/stub data without requiring CFS or GPU. |
| Resume option | `--resume` | `str` | `auto` | Resume training: `auto` (latest ckpt in `save_dir`), `none` (fresh start), or explicit path to checkpoint. |

### `run.py show-config`

| Option | Flag | Type | Default | Description / Validity |
| --- | --- | --- | --- | --- |
| Configuration file | `--config` | `Path` | `configs/unified.yaml` | Path to unified YAML configuration file. |
| Operational mode | `--mode` | `str` | `None` (required for unified) | Mode overlay to apply: `baseline`, `physics_informed`, `lora`, or `combined`. |
| Override parameter | `--override` | `list[str]` | `[]` (repeatable) | Dot-notation config override. |

---

## 3. Deprecated Entry Points

- `train.py`: Preserved as a ≤10-line backward compatibility shim for in-flight SLURM jobs (delegates directly to `run.py train` while preserving `sys.argv`).

---

## 4. Configuration Validity Matrix

Configuration is validated once, at the boundary, into a strongly-typed `Config` object using Pydantic v2. Invalid combinations fail immediately at load time with actionable remediation hints. All schemas enforce `extra="forbid"`, ensuring that mistyped override keys (e.g. `--override training.optimzer.lr=1e-5`) fail loudly rather than executing silently with defaults.

| Invalid Combination | Rationale & Remediation Message |
| --- | --- |
| `model.gradient_checkpointing: true` | Deterministically triggers an illegal memory access crash on Perlmutter A100s. Full 1.3B model requires checkpointing to fit on 4×A100; use `model_type: small` instead. |
| `data.use_dummy: true` | The dummy dataset was removed during repository consolidation; use `--smoke-test` for CPU/synthetic verification. |
| `model.norm_stats.msl` absent | Removing MSL normalisation stats restores a −36 σ surface pressure proxy input and causes 100% non-finite validation loss. |
| `training.rollout.enabled: true` with `backprop: "full"` | Full-BPTT rollout requires gradient checkpointing which crashes on Perlmutter; only `backprop: "detached"` has been verified crash-free (see `probe_ima_matrix.py`). |
| `training.rollout.start_steps > rollout.max_steps` | Nonsensical curriculum ordering. `start_steps` must be ≤ `max_steps`. |
| `data.real.val_years` overlapping `train_years` | Chronological splits only; prevents evaluation leakage (the 54,060 sample leakage class). |
| `data.real.test_years` overlapping `train_years` or `val_years` | Chronological splits only; prevents evaluation leakage. |
| Two modes sharing a `checkpointing.save_dir` | Prevents silent checkpoint overwrite and metric file corruption between operational modes. |
| `model.mjo_head.enabled: false` with `loss.mjo_head.enabled: true` | Prevents scoring a loss term against an MJO projection head that does not exist. |
| Unrecognized key path (e.g. `training.optimzer.lr`) | Extra inputs are forbidden (`extra="forbid"`). Rejects typos at load time before job execution begins. |
