#!/usr/bin/env python3
"""Thin CLI entry point for Microsoft Aurora MJO fine-tuning.

Commands:
  train        - Fine-tune Aurora for MJO prediction
  show-config  - Resolve and print canonical JSON configuration
  evaluate     - Evaluate MJO forecast skill across lead times
  norm-stats   - Compute normalization statistics over training years
"""

from __future__ import annotations

# Process-start environment configuration must execute before C-libraries initialise.
from aurora_mjo.env import configure_environment  # isort: skip

configure_environment()

import logging  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import typer  # noqa: E402

from aurora_mjo.checkpoint import CheckpointManager  # noqa: E402
from aurora_mjo.cli_support import (  # noqa: E402
    apply_overrides,
    load_config,
    print_config,
    run_train,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

app = typer.Typer(
    help="Aurora MJO fine-tuning driver CLI.",
    add_completion=False,
)


@app.command()
def train(
    config: Path = typer.Option(
        Path("configs/unified.yaml"),
        "--config",
        help="Path to YAML config.",
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="Mode overlay to apply (required for unified configs).",
    ),
    override: list[str] = typer.Option(
        [],
        "--override",
        help="Dot-notation config override; repeatable.",
    ),
    smoke_test: bool = typer.Option(
        False,
        "--smoke-test",
        help="Single synthetic step, no real data.",
    ),
    resume: str = typer.Option(
        "auto",
        "--resume",
        help="'auto' = latest ckpt in save_dir; 'none' = fresh start; or explicit path.",
    ),
) -> None:
    """Train or fine-tune Aurora for MJO prediction."""
    cfg = load_config(str(config), mode)
    if override:
        cfg = apply_overrides(cfg, override)
    exit_code = run_train(cfg, resume=resume, smoke_test=smoke_test)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@app.command("show-config")
def show_config(
    config: Path = typer.Option(
        Path("configs/unified.yaml"),
        "--config",
        help="Path to YAML config.",
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="Mode overlay to apply (required for unified configs).",
    ),
    override: list[str] = typer.Option(
        [],
        "--override",
        help="Dot-notation config override; repeatable.",
    ),
) -> None:
    """Resolve config overlays and overrides and print canonical JSON."""
    cfg = load_config(str(config), mode)
    if override:
        cfg = apply_overrides(cfg, override)
    print_config(cfg)


@app.command()
def evaluate(
    config: Path = typer.Option(
        Path("configs/unified.yaml"),
        "--config",
        help="Path to YAML config.",
    ),
    mode: str = typer.Option(
        ...,
        "--mode",
        help="Mode overlay to evaluate (required for unified configs).",
    ),
    checkpoint: str = typer.Option(
        "latest",
        "--checkpoint",
        help="'latest' or explicit path to model checkpoint.",
    ),
    targets: Path = typer.Option(
        Path("data/rmm_targets.nc"),
        "--targets",
        help="Path to rmm_targets.nc.",
    ),
    basis: Path = typer.Option(
        Path("data/rmm_basis.npz"),
        "--basis",
        help="Path to rmm_basis.npz.",
    ),
    out_dir: Path = typer.Option(
        Path("evaluation/mjo_skill"),
        "--out-dir",
        help="Directory to write evaluation outputs.",
    ),
    smoke_test: bool = typer.Option(
        False,
        "--smoke-test",
        help="Run synthetic smoke test without checkpoint or data.",
    ),
) -> None:
    """Evaluate MJO prediction skill across lead times."""
    if smoke_test:
        cmd = [sys.executable, "scripts/evaluate_mjo.py", "--smoke-test"]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise typer.Exit(code=res.returncode)
        return

    ckpt_path: Path | None = None
    if checkpoint == "latest":
        cfg = load_config(str(config), mode)
        save_dir = cfg.get("checkpointing", {}).get("save_dir", f"checkpoints/{mode}")
        ckpt_path = CheckpointManager.find_latest(save_dir)
        if ckpt_path is None:
            typer.echo(
                f"No checkpoint found in save_dir '{save_dir}'. "
                f"Run scripts/evaluate_mjo.py directly with an explicit checkpoint:\n"
                f"  uv run python scripts/evaluate_mjo.py --config {config} --mode {mode} "
                f"--checkpoint <path/to/checkpoint.pt>",
                err=True,
            )
            raise typer.Exit(code=1)
    else:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            typer.echo(
                f"Checkpoint not found: {ckpt_path}. "
                f"Run scripts/evaluate_mjo.py directly with valid paths:\n"
                f"  uv run python scripts/evaluate_mjo.py --config {config} --mode {mode} "
                f"--checkpoint <path>",
                err=True,
            )
            raise typer.Exit(code=1)

    cmd = [
        sys.executable,
        "scripts/evaluate_mjo.py",
        "--config",
        str(config),
        "--mode",
        mode,
        "--checkpoint",
        str(ckpt_path),
        "--targets",
        str(targets),
        "--basis",
        str(basis),
        "--out-dir",
        str(out_dir),
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise typer.Exit(code=res.returncode)


@app.command("norm-stats")
def norm_stats(
    years: tuple[int, int] = typer.Option(
        (1980, 2015),
        "--years",
        help="Start and end years for training statistics.",
    ),
    config: Path = typer.Option(
        Path("configs/unified.yaml"),
        "--config",
        help="Path to YAML config.",
    ),
) -> None:
    """Compute normalization statistics over training years."""
    years_str = f"{years[0]} {years[1]}"
    typer.echo(
        "run.py norm-stats logic is hosted in scripts/calc_norm_stats.py "
        "(not yet extracted into aurora_mjo to keep scripts run-only).\n"
        f"Please run directly:\n"
        f"  uv run python scripts/calc_norm_stats.py --config {config} --years {years_str}",
        err=True,
    )
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
