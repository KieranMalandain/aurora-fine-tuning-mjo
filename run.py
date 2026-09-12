#!/usr/bin/env python3
"""Thin CLI entry point for Microsoft Aurora MJO fine-tuning.

Commands:
  train        - Fine-tune Aurora for MJO prediction
  show-config  - Resolve and print canonical JSON configuration

Note: `evaluate` and `norm-stats` commands are planned for task C3.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from aurora_mjo.cli_support import (
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


if __name__ == "__main__":
    app()
