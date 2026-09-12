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
