"""tests/test_config_modes.py — Regression tests for Lesson 6 and configuration mechanics.

Reference:
    00_CONTEXT.md §3 (Lesson 6: gradient checkpointing crashes this machine)
    01_TARGET_STATE.md D8, D10, §9
    03_DOMAIN_PRIORS.md §7

Background:
    On NERSC Perlmutter A100s, enabling gradient checkpointing deterministically triggers
    an illegal memory access (IMA) in CUDA kernels during backward passes. The codebase
    sidesteps this by:
      1. Setting `gradient_checkpointing: false` across all four modes in `unified.yaml`.
      2. Using detached rollout (step-by-step backward with detached state advance),
         which keeps activation memory O(1) in k rollout steps (~40 GB footprint)
         without needing checkpointing.
    This test module ensures that:
      - All four modes resolve without error and preserve distinct save directories.
      - Gradient checkpointing remains strictly disabled everywhere (config invariant).
      - Warm-start `init_from` inheritance chain matches specification.
      - CLI `--override` dot-notation parsing, precedence, error handling, and type
        coercion remain strictly locked against B1's behavioural fingerprint.
      - Missing or invalid `--mode` flags raise `SystemExit` with actionable help.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aurora_mjo.cli_support import apply_overrides, load_config

MODES = ("baseline", "physics_informed", "lora", "combined")


@pytest.fixture
def config_path() -> Path:
    """Path to the canonical configs/unified.yaml."""
    path = Path(__file__).resolve().parent.parent / "configs" / "unified.yaml"
    if not path.exists():
        pytest.fail(f"Config file not found at {path}")
    return path


def test_all_four_modes_resolve_cleanly(config_path: Path) -> None:
    """Verify all four modes in configs/unified.yaml resolve without error."""
    for mode in MODES:
        cfg = load_config(config_path, mode=mode)
        assert isinstance(cfg, dict), f"Failed to load config for mode {mode}"
        assert cfg.get("experiment", {}).get("mode") == mode


def test_gradient_checkpointing_false_in_all_modes(config_path: Path) -> None:
    """Assert gradient_checkpointing is False in every resolved mode.

    CRITICAL REPO RULE (AGENTS.md §5, Lesson 6):
    Gradient checkpointing deterministically triggers an illegal memory access crash
    on Perlmutter A100s. Never flip gradient_checkpointing to true.
    This test guards against well-meaning 'enable checkpointing to save memory' edits.
    """
    for mode in MODES:
        cfg = load_config(config_path, mode=mode)
        ckpt_enabled = cfg.get("model", {}).get("gradient_checkpointing")
        assert ckpt_enabled is False, (
            f"Mode '{mode}' has gradient_checkpointing={ckpt_enabled}; "
            "it MUST remain False to prevent illegal memory access crashes on Perlmutter."
        )


def test_mode_chain_init_from_values(config_path: Path) -> None:
    """Verify the mode warm-start inheritance chain is configured correctly.

    Chain specification:
      - baseline: starts from scratch (no init_from)
      - physics_informed: warm-starts from latest baseline checkpoint
      - lora: warm-starts from latest baseline checkpoint
      - combined: warm-starts from latest lora checkpoint
    """
    cfg_baseline = load_config(config_path, mode="baseline")
    assert "init_from" not in cfg_baseline.get("experiment", {})

    cfg_physics = load_config(config_path, mode="physics_informed")
    assert cfg_physics.get("experiment", {}).get("init_from") == "latest:checkpoints/baseline"

    cfg_lora = load_config(config_path, mode="lora")
    assert cfg_lora.get("experiment", {}).get("init_from") == "latest:checkpoints/baseline"

    cfg_combined = load_config(config_path, mode="combined")
    assert cfg_combined.get("experiment", {}).get("init_from") == "latest:checkpoints/lora"


def test_distinct_save_dir_per_mode(config_path: Path) -> None:
    """Verify that every mode writes to a distinct checkpoint save directory.

    Two modes sharing a save_dir would overwrite checkpoints and corrupt metrics.
    """
    expected_save_dirs = {
        "baseline": "checkpoints/baseline",
        "physics_informed": "checkpoints/physics_informed",
        "lora": "checkpoints/lora",
        "combined": "checkpoints/combined",
    }
    save_dirs: dict[str, str] = {}
    for mode in MODES:
        cfg = load_config(config_path, mode=mode)
        save_dir = cfg.get("checkpointing", {}).get("save_dir")
        assert save_dir is not None, f"Mode '{mode}' missing checkpointing.save_dir"
        save_dirs[mode] = save_dir

    assert save_dirs == expected_save_dirs
    assert len(set(save_dirs.values())) == len(MODES), "Checkpoint save_dirs must be unique"


def test_apply_overrides_locked_against_b1_fingerprint(
    config_path: Path, baseline_fingerprint: dict[str, Any]
) -> None:
    """Lock --override dot-notation and type coercion against B1 baseline fingerprint."""
    raw_cfg = load_config(config_path, mode="baseline")
    overridden = apply_overrides(raw_cfg, ["training.optimizer.lr=1e-5"])

    expected_fp = baseline_fingerprint["config_baseline_override"]

    # In PyYAML 1.1 / safe_load, scientific notation without decimal (1e-5) coerces to str '1e-5'
    assert overridden["training"]["optimizer"]["lr"] == "1e-5"
    assert (
        overridden == expected_fp
    ), "Resolved config with override differs from B1 canonical fingerprint!"


def test_apply_overrides_semantics_and_type_coercion(config_path: Path) -> None:
    """Verify dot-notation, ordering precedence, error handling, and type coercion."""
    cfg = load_config(config_path, mode="baseline")

    # 1. Invalid override without '=' must raise ValueError
    with pytest.raises(ValueError, match="Invalid override"):
        apply_overrides(cfg, ["training.epochs"])

    # 2. Repeated overrides apply in sequential order (last wins)
    cfg_ordered = apply_overrides(cfg, ["training.epochs=5", "training.epochs=10"])
    assert cfg_ordered["training"]["epochs"] == 10

    # 3. Type coercions via yaml.safe_load
    coercions = [
        ("training.epochs=12", 12, int),
        ("training.use_amp=false", False, bool),
        ("model.norm_stats.msl.mean=96667.5", 96667.5, float),
        ("loss.grid.tropics_bbox=[-15, 15]", [-15, 15], list),
        ("experiment.name=custom_run", "custom_run", str),
    ]
    for override_str, expected_val, expected_type in coercions:
        cfg_coerced = apply_overrides(cfg, [override_str])
        key_path = override_str.split("=")[0].split(".")
        val = cfg_coerced
        for k in key_path:
            val = val[k]
        assert val == expected_val
        assert isinstance(val, expected_type)


def test_mode_flag_validation_system_exit(config_path: Path) -> None:
    """Verify missing or invalid mode raises SystemExit with helpful message."""
    # 1. Missing mode raises SystemExit
    with pytest.raises(SystemExit) as exc_missing:
        load_config(config_path, mode=None)
    assert "--mode is required" in str(exc_missing.value)

    # 2. Unknown mode raises SystemExit naming valid modes
    with pytest.raises(SystemExit) as exc_unknown:
        load_config(config_path, mode="invalid_mode_name")
    assert "Unknown mode 'invalid_mode_name'" in str(exc_unknown.value)
    for m in MODES:
        assert m in str(exc_unknown.value)
