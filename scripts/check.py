#!/usr/bin/env python3
"""scripts/check.py — Single verification gate for aurora-fine-tuning-mjo.

Runs all verification checks in order:
1. lockfile     (uv lock --check)
2. ruff lint    (uv run ruff check .)
3. ruff format  (uv run ruff format --check .)
4. types        (uv run pyrefly check)
5. pytest       (uv run pytest -q -m "<marks>")

Stdlib only. Continues past failures and reports a summary table.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time


def check_uv() -> None:
    """Verify uv is installed and available on PATH."""
    if shutil.which("uv") is None:
        print("error: 'uv' executable not found on PATH.", file=sys.stderr)
        print(
            "This repository exclusively supports uv for environment and package management.\n"
            "pip, venv, poetry, and conda are not supported.\n"
            "Please install uv (https://docs.astral.sh/uv/) and ensure it is in your PATH.\n"
            "See docs/campaigns/refactor/01_TARGET_STATE.md D3.",
            file=sys.stderr,
        )
        sys.exit(127)


def run_gate(
    name: str,
    cmd: list[str],
    why: str,
) -> tuple[str, bool, float, str]:
    """Run a single gate step, returning (name, passed, duration_s, why)."""
    print(f"\n=== Gate: {name} ({' '.join(cmd)}) ===")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd)
    dt = time.perf_counter() - t0

    passed = proc.returncode == 0

    # Pytest exit code 5 means no tests were collected.
    # Until task D1 creates the tests/ directory and adds initial tests,
    # zero collected tests must be treated as a pass.
    # TODO(D1): Remove special case for pytest exit code 5 once tests/ exists.
    if name == "pytest" and proc.returncode == 5:
        print(
            "Note: pytest exited 5 (no tests collected). Treated as PASS until D1 creates tests/."
        )
        passed = True

    return name, passed, dt, why


def main() -> None:
    check_uv()

    parser = argparse.ArgumentParser(description="Run all repository verification checks.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Exclude slow tests (adds 'and not slow' to pytest marker expression).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Run ruff autofix and formatter before gating.",
    )
    args = parser.parse_args()

    if args.fix:
        print("=== Autofix: Running ruff check --fix and ruff format ===")
        subprocess.run(["uv", "run", "ruff", "check", "--fix", "."])
        subprocess.run(["uv", "run", "ruff", "format", "."])

    marker_expr = "not live and not needs_data and not needs_gpu"
    if args.fast:
        marker_expr += " and not slow"

    steps: list[tuple[str, list[str], str]] = [
        ("lockfile", ["uv", "lock", "--check"], "uv.lock matches pyproject.toml"),
        ("ruff lint", ["uv", "run", "ruff", "check", "."], "lint"),
        (
            "ruff format",
            ["uv", "run", "ruff", "format", "--check", "."],
            "formatting is canonical",
        ),
        ("types", ["uv", "run", "pyrefly", "check"], "static types, ratcheted scope"),
        (
            "pytest",
            ["uv", "run", "pytest", "-q", "-m", marker_expr],
            "tests, excluding what needs data/GPU/network",
        ),
    ]

    results: list[tuple[str, bool, float, str]] = []
    for name, cmd, why in steps:
        results.append(run_gate(name, cmd, why))

    print("\n" + "=" * 79)
    print(f"{'Gate':<15} {'Status':<8} {'Time':<8} {'Why'}")
    print("-" * 79)
    failed_gates: list[str] = []
    for name, passed, dt, why in results:
        status_str = "PASS" if passed else "FAIL"
        if not passed:
            failed_gates.append(name)
        print(f"{name:<15} {status_str:<8} {dt:>5.2f}s   {why}")
    print("=" * 79)

    if failed_gates:
        print(f"FAILED GATES: {', '.join(failed_gates)}")
        sys.exit(1)
    else:
        print("ALL GATES PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
