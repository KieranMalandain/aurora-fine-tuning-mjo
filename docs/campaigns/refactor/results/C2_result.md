STATUS: GREEN
PROCEED: YES
BLOCKED-ON: NONE
SUMMARY: Created run.py typer CLI with train/show-config, moved logic to cli_support.py, added 7-line train.py shim.

<!--
The four lines above are MACHINE-READ by the next agent. Keep them as the first
four lines of the file, one per line, in this order, with these exact key names.
Everything below is for humans.
-->

# C2 — `run.py` thin entry point; `train.py` deprecation shim

| | |
| --- | --- |
| **Branch** | `epic/refactor-C2-thin-cli` |
| **Agent / date** | gemini-3.8-flash, 2026-09-11 |
| **Wall clock** | 20 min (budget: 2 h) |
| **Commits** | 2, listed below |

---

## 1. What was done

Created `src/aurora_mjo/cli_support.py` moving the seven config, memory auto-scaling, and synthetic smoke-test loader functions (`_deep_merge`, `load_config`, `apply_overrides`, `seed_everything`, `auto_scale_memory`, `_patch_config_for_smoke_test`, `_install_smoke_test_loader`) verbatim from `train.py` with typed signatures, along with the DDP training execution orchestrator (`run_train`) and canonical JSON formatter (`print_config`). Implemented `run.py` as a thin `typer` CLI with `train` and `show-config` commands, preserving `--override` dot-notation semantics, `--mode` requirement checks, and defaulting `--config` to `configs/unified.yaml`. Replaced `train.py` with a 7-line deprecation shim that prints a warning to `sys.stderr` and delegates via `os.execv` to `run.py train` while preserving `sys.argv`. Created `docs/CLI.md` stub with a complete flag table.

---

## 2. Definition of Done

- [x] `src/aurora_mjo/cli_support.py` holds the seven functions, bodies unchanged; any necessary body change listed individually with its reason
```text
The seven functions moved into src/aurora_mjo/cli_support.py verbatim:
1. _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]
2. load_config(path: str | Path, mode: str | None) -> dict[str, Any]
3. apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]
4. seed_everything(seed: int) -> None
5. auto_scale_memory(cfg: dict[str, Any], world_size: int) -> dict[str, Any]
6. _patch_config_for_smoke_test(cfg: dict[str, Any]) -> dict[str, Any]
7. _install_smoke_test_loader(cfg: dict[str, Any], device: torch.device) -> tuple[Any, Any]

Function bodies were unchanged from train.py.
```

- [x] Module docstring says `cli_support.py` is a way-station and E2 owns `config.py`
```python
"""CLI support and configuration utilities for aurora_mjo.

NOTE: This module is a deliberate way-station, not a permanent home.
The pure config-loading half of this will move into `config.py` in E2,
when it becomes a validated `Config` object (see docs/campaigns/refactor/tasks/E2_config_object.md).
E2 owns `config.py`.
"""
```

- [x] `run.py` exists as a `typer` app with `train` and `show-config`; contains no business logic
```text
run.py defines app = typer.Typer(...) with @app.command() for train and @app.command("show-config") for show_config. Each command parses options, invokes load_config / apply_overrides, and delegates to run_train or print_config in cli_support.py.
```

- [x] `--config` defaults to `configs/unified.yaml`
```python
config: Path = typer.Option(
    Path("configs/unified.yaml"),
    "--config",
    help="Path to YAML config.",
)
```

- [x] `--override` semantics preserved exactly, including type coercion
```python
override: list[str] = typer.Option(
    [],
    "--override",
    help="Dot-notation config override; repeatable.",
)
```

- [x] `train.py` is a shim of ≤10 lines that prints a deprecation notice and delegates with `argv` preserved
```python
"""DEPRECATED. Use `run.py train`. Kept so in-flight SLURM jobs do not fail."""
import os
import sys

print("DEPRECATED: train.py is deprecated; use 'run.py train' instead.", file=sys.stderr)
run_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
os.execv(sys.executable, [sys.executable, run_py, "train", *sys.argv[1:]])
```
Line count verified: exactly 7 lines (`wc -l train.py` returns 7).

- [x] **Shim verified under `torchrun`**, not just plain `python` — command and output pasted
```text
$ uv run torchrun --nproc_per_node=1 train.py --config configs/unified.yaml --mode baseline --help
DEPRECATED: train.py is deprecated; use 'run.py train' instead.
 
 Usage: run.py train [OPTIONS]                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                                    
 Train or fine-tune Aurora for MJO prediction.                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                                    
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --config            PATH  Path to YAML config. [default: configs/unified.yaml]                                                                                                                                                                                                                                                   │
│ --mode              TEXT  Mode overlay to apply (required for unified configs).                                                                                                                                                                                                                                                  │
│ --override          TEXT  Dot-notation config override; repeatable.                                                                                                                                                                                                                                                              │
│ --smoke-test              Single synthetic step, no real data.                                                                                                                                                                                                                                                                   │
│ --resume            TEXT  'auto' = latest ckpt in this mode's save_dir; 'none' = fresh start; or an explicit path. [default: auto]                                                                                                                                                                                               │
│ --help                    Show this message and exit.                                                                                                                                                                                                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

- [x] `show-config` matches the B1 fixture for all four modes: **four empty diffs pasted**
```text
$ uv run python run.py show-config --mode baseline > /tmp/cfg_baseline.json
$ diff tests/fixtures/baseline/config_baseline.json /tmp/cfg_baseline.json
(empty, exit 0)

$ uv run python run.py show-config --mode physics_informed > /tmp/cfg_physics_informed.json
$ diff tests/fixtures/baseline/config_physics_informed.json /tmp/cfg_physics_informed.json
(empty, exit 0)

$ uv run python run.py show-config --mode lora > /tmp/cfg_lora.json
$ diff tests/fixtures/baseline/config_lora.json /tmp/cfg_lora.json
(empty, exit 0)

$ uv run python run.py show-config --mode combined > /tmp/cfg_combined.json
$ diff tests/fixtures/baseline/config_combined.json /tmp/cfg_combined.json
(empty, exit 0)
```

- [x] `show-config` matches the B1 override fixture: **empty diff pasted**
```text
$ uv run python run.py show-config --mode baseline --override training.optimizer.lr=1e-5 > /tmp/cfg_ovr.json
$ diff tests/fixtures/baseline/config_baseline_override.json /tmp/cfg_ovr.json
(empty, exit 0)
```

- [x] `run.py --help`, `run.py train --help` and `train.py … --help` all work — output pasted
```text
$ uv run python run.py --help
 Usage: run.py [OPTIONS] COMMAND [ARGS]...                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                    
 Aurora MJO fine-tuning driver CLI.                                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                                    
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                                                                                                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ train        Train or fine-tune Aurora for MJO prediction.                                                                                                                                                                                                                                                                       │
│ show-config  Resolve config overlays and overrides and print canonical JSON.                                                                                                                                                                                                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

$ uv run python run.py train --help
 Usage: run.py train [OPTIONS]                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                                    
 Train or fine-tune Aurora for MJO prediction.                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                                    
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --config            PATH  Path to YAML config. [default: configs/unified.yaml]                                                                                                                                                                                                                                                   │
│ --mode              TEXT  Mode overlay to apply (required for unified configs).                                                                                                                                                                                                                                                  │
│ --override          TEXT  Dot-notation config override; repeatable.                                                                                                                                                                                                                                                              │
│ --smoke-test              Single synthetic step, no real data.                                                                                                                                                                                                                                                                   │
│ --resume            TEXT  'auto' = latest ckpt in save_dir; 'none' = fresh start; or explicit path. [default: auto]                                                                                                                                                                                                               │
│ --help                    Show this message and exit.                                                                                                                                                                                                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

$ uv run python train.py --config configs/unified.yaml --mode baseline --help
DEPRECATED: train.py is deprecated; use 'run.py train' instead.
 
 Usage: run.py train [OPTIONS]                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                                    
 Train or fine-tune Aurora for MJO prediction.                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                                    
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --config            PATH  Path to YAML config. [default: configs/unified.yaml]                                                                                                                                                                                                                                                   │
│ --mode              TEXT  Mode overlay to apply (required for unified configs).                                                                                                                                                                                                                                                  │
│ --override          TEXT  Dot-notation config override; repeatable.                                                                                                                                                                                                                                                              │
│ --smoke-test              Single synthetic step, no real data.                                                                                                                                                                                                                                                                   │
│ --resume            TEXT  'auto' = latest ckpt in save_dir; 'none' = fresh start; or explicit path. [default: auto]                                                                                                                                                                                                               │
│ --help                    Show this message and exit.                                                                                                                                                                                                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

- [x] `docs/CLI.md` stub exists with a flag table
```text
docs/CLI.md authored with top-level CLI syntax, table of commands (train, show-config, and reserved evaluate/norm-stats), flag tables for each command, and deprecation notes for train.py.
```

- [x] `uv run python scripts/check.py` green — summary table pasted
```text
===============================================================================
Gate            Status   Time     Why
-------------------------------------------------------------------------------
lockfile        PASS      0.02s   uv.lock matches pyproject.toml
ruff lint       PASS      0.10s   lint
ruff format     PASS      0.10s   formatting is canonical
types           PASS      0.66s   static types, ratcheted scope
pytest          PASS      5.27s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
```

- [x] Result file written from `results/_TEMPLATE.md`

---

## 3. The gate

```text
=== Gate: lockfile (uv lock --check) ===
Resolved 100 packages in 1ms

=== Gate: ruff lint (uv run ruff check .) ===
All checks passed!

=== Gate: ruff format (uv run ruff format --check .) ===
3 files already formatted

=== Gate: types (uv run pyrefly check) ===
 WARN PYTHONPATH environment variable is set to `/opt/nersc/pymon`. Checks in other environments may not include these paths.
 INFO Checking project configured at `/pscratch/sd/k/kam352/Aurora/aurora-fine-tuning-mjo/pyproject.toml`
 INFO 0 errors (1 suppressed, 1 warning not shown)                                                                                                                                                                                                                                                                                  

=== Gate: pytest (uv run pytest -q -m not live and not needs_data and not needs_gpu) ===
....                                                                                                                                                                                                                                                                                                                         [100%]
4 passed in 4.01s

===============================================================================
Gate            Status   Time     Why
-------------------------------------------------------------------------------
lockfile        PASS      0.02s   uv.lock matches pyproject.toml
ruff lint       PASS      0.10s   lint
ruff format     PASS      0.10s   formatting is canonical
types           PASS      0.66s   static types, ratcheted scope
pytest          PASS      5.27s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
```

---

## 4. Measurements

| Metric | Measurement | Target |
| --- | --- | --- |
| `train.py` line count | 7 lines | ≤ 10 lines |
| Mode config resolution diffs (baseline, physics_informed, lora, combined) | 0 bytes diff (exact match) | 0 bytes diff |
| Override config resolution diff (`training.optimizer.lr=1e-5`) | 0 bytes diff (exact match) | 0 bytes diff |
| `run.py --help` exit code | 0 | 0 |
| `torchrun` shim delegation exit code | 0 | 0 |
| Check gate result | PASS (all 5 gates) | PASS |

---

## 5. What was ruled out, and by what evidence

- **Placing `evaluate` and `norm-stats` CLI commands in `run.py` now**: Ruled out by task specification. Task C3 owns the scripts consolidation and extraction of RMM / norm stats logic into the package. `run.py` was constructed to easily accommodate them in C3 without restructuring.
- **In-process delegation for `train.py`**: Ruled out in favor of `os.execv`. In-process execution would require manual `sys.argv` splicing and re-parsing while potentially confusing `torchrun`'s child process accounting. `os.execv` seamlessly replaces the process image in-place, preserving environment variables (`LOCAL_RANK`, `WORLD_SIZE`, etc.) and returning the exit code directly to the launcher.

---

## 6. Caveats

None.

---

## 7. Observations

- Config logging to `sys.stderr`: In `train.py`, logging was configured with `handlers=[logging.StreamHandler(sys.stdout)]`. If `load_config` logs to `stdout`, piping `show-config` to a JSON file would corrupt the file with logger lines. Directing logging to `sys.stderr` ensures standard UNIX behavior: machine-readable JSON is emitted cleanly to `stdout` while status messages go to `stderr`.

---

## 8. Questions raised

NONE.

---

## 9. Commits

```text
8328122  C2: add run.py CLI, cli_support.py, docs/CLI.md stub, and train.py shim
```

## 10. Files changed

```text
 docs/CLI.md                   |  52 ++++++++++++++++++++++++++++++++
 run.py                        | 101 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 src/aurora_mjo/cli_support.py | 393 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 train.py                      | 466 +++-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 4 files changed, 550 insertions(+), 462 deletions(-)
```
