"""tests/test_config_validation.py — Unit tests for Task E2 boundary config validation.

Tests verify:
  1. All 9 rows of validity matrix raise validation exceptions with actionable messages.
  2. Unknown keys and typo overrides are strictly rejected via extra="forbid".
  3. Round-trip proof against B1 baseline fixtures for all four operational modes.
All tests run on CPU on the default CI path (no GPU, no data, no network).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from aurora_mjo.config import (
    Config,
    apply_overrides,
    load_config,
)

MODES = ("baseline", "physics_informed", "lora", "combined")


@pytest.fixture
def config_path() -> Path:
    """Path to the canonical configs/unified.yaml."""
    path = Path(__file__).resolve().parent.parent / "configs" / "unified.yaml"
    if not path.exists():
        pytest.fail(f"Config file not found at {path}")
    return path


@pytest.fixture
def baseline_raw_dict(config_path: Path) -> dict[str, Any]:
    """Resolved baseline config dictionary before validation."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    from aurora_mjo.config import _deep_merge

    modes = raw.pop("modes")
    cfg = _deep_merge(raw, modes["baseline"])
    cfg.setdefault("experiment", {})["mode"] = "baseline"
    return cfg


# ---------------------------------------------------------------------------
# Validity Matrix Tests (Rows 1–9)
# ---------------------------------------------------------------------------


def test_row1_gradient_checkpointing_true_fails_with_actionable_hint(
    baseline_raw_dict: dict[str, Any],
) -> None:
    """Row 1: gradient_checkpointing: true must fail with Perlmutter IMA explanation."""
    baseline_raw_dict["model"]["gradient_checkpointing"] = True
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(baseline_raw_dict)
    err_msg = str(exc_info.value)
    assert (
        "it deterministically triggers an illegal memory access on Perlmutter; "
        "see the gameplan; the full model does not fit on 4×A100 without it, "
        "so use `model_type: small`" in err_msg
    )


def test_row2_data_use_dummy_true_fails_with_actionable_hint(
    baseline_raw_dict: dict[str, Any],
) -> None:
    """Row 2: data.use_dummy: true must fail directing user to --smoke-test."""
    baseline_raw_dict["data"]["use_dummy"] = True
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(baseline_raw_dict)
    err_msg = str(exc_info.value)
    assert "the dummy dataset was deleted; use `--smoke-test`" in err_msg


def test_row3_norm_stats_msl_absent_fails_with_actionable_hint(
    baseline_raw_dict: dict[str, Any],
) -> None:
    """Row 3: model.norm_stats.msl absent must fail with -36 sigma non-finite explanation."""
    del baseline_raw_dict["model"]["norm_stats"]["msl"]
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(baseline_raw_dict)
    err_msg = str(exc_info.value)
    assert "removing it restores a −36 σ input and 100% non-finite validation" in err_msg


def test_row4_rollout_full_backprop_fails_with_actionable_hint(
    baseline_raw_dict: dict[str, Any],
) -> None:
    """Row 4: rollout.enabled: true with backprop: 'full' must fail with probe hint."""
    baseline_raw_dict["training"]["rollout"]["enabled"] = True
    baseline_raw_dict["training"]["rollout"]["backprop"] = "full"
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(baseline_raw_dict)
    err_msg = str(exc_info.value)
    assert 'only "detached" has been verified crash-free; see probe_ima_matrix.py' in err_msg


def test_row5_rollout_start_steps_greater_than_max_fails(
    baseline_raw_dict: dict[str, Any],
) -> None:
    """Row 5: rollout.start_steps > rollout.max_steps must fail with nonsensical curriculum."""
    baseline_raw_dict["training"]["rollout"]["start_steps"] = 5
    baseline_raw_dict["training"]["rollout"]["max_steps"] = 4
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(baseline_raw_dict)
    err_msg = str(exc_info.value)
    assert "nonsensical curriculum" in err_msg


def test_row6_val_years_overlapping_train_years_fails(
    baseline_raw_dict: dict[str, Any],
) -> None:
    """Row 6: val_years overlapping train_years must fail with 54,060 leakage explanation."""
    baseline_raw_dict["data"]["real"]["train_years"] = [1980, 2016]
    baseline_raw_dict["data"]["real"]["val_years"] = [2016, 2019]
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(baseline_raw_dict)
    err_msg = str(exc_info.value)
    assert "chronological splits only; this is the 54,060 leakage class" in err_msg


def test_row7_test_years_overlapping_train_or_val_fails(
    baseline_raw_dict: dict[str, Any],
) -> None:
    """Row 7: test_years overlapping val_years must fail with 54,060 leakage explanation."""
    baseline_raw_dict["data"]["real"]["test_years"] = [2018, 2023]
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(baseline_raw_dict)
    err_msg = str(exc_info.value)
    assert "chronological splits only; this is the 54,060 leakage class" in err_msg


def test_row8_two_modes_sharing_save_dir_fails(tmp_path: Path, config_path: Path) -> None:
    """Row 8: Two modes sharing a checkpointing.save_dir must raise with silent overwrite."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    # Intentionally force lora to share save_dir with baseline
    raw["modes"]["lora"]["checkpointing"] = {"save_dir": "checkpoints/baseline"}

    bad_config = tmp_path / "bad_unified.yaml"
    with open(bad_config, "w") as f:
        yaml.safe_dump(raw, f)

    with pytest.raises(ValueError) as exc_info:
        load_config(bad_config, mode="baseline")
    assert "silent checkpoint overwrite" in str(exc_info.value)


def test_row9_mjo_head_disabled_with_loss_enabled_fails(
    baseline_raw_dict: dict[str, Any],
) -> None:
    """Row 9: mjo_head.enabled: false with loss.mjo_head.enabled: true must fail."""
    baseline_raw_dict["model"]["mjo_head"]["enabled"] = False
    baseline_raw_dict["loss"]["mjo_head"]["enabled"] = True
    baseline_raw_dict["loss"]["mjo_head"]["weight"] = 1.0
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(baseline_raw_dict)
    err_msg = str(exc_info.value)
    assert "a loss term scoring a head that does not exist" in err_msg


# ---------------------------------------------------------------------------
# Unknown-Key / Typo Rejection Tests (Step 5)
# ---------------------------------------------------------------------------


def test_unknown_key_rejection_catches_typo_override(config_path: Path) -> None:
    """--override training.optimzer.lr=1e-5 must fail loudly with extra inputs forbidden."""
    cfg = load_config(config_path, mode="baseline")
    with pytest.raises(ValidationError) as exc_info:
        apply_overrides(cfg, ["training.optimzer.lr=1e-5"])
    err_msg = str(exc_info.value)
    assert "training.optimzer" in err_msg
    assert "Extra inputs are not permitted" in err_msg


def test_unknown_top_level_key_rejected(config_path: Path) -> None:
    """Top-level unknown key override must fail with extra inputs forbidden."""
    cfg = load_config(config_path, mode="baseline")
    with pytest.raises(ValidationError) as exc_info:
        apply_overrides(cfg, ["nonexistent_section.key=123"])
    err_msg = str(exc_info.value)
    assert "Extra inputs are not permitted" in err_msg


# ---------------------------------------------------------------------------
# B1 Baseline Fixtures Round-Trip Proof (Step 3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_b1_baseline_fixture_exact_round_trip(mode: str) -> None:
    """Config.to_dict() must produce exact canonical JSON matching B1 fixtures."""
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "baseline" / f"config_{mode}.json"
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"

    with open(fixture_path) as f:
        expected_dict = json.load(f)

    cfg = Config.model_validate(expected_dict)
    resolved_dict = cfg.to_dict()

    # 1. Semantic equality
    assert resolved_dict == expected_dict, f"Semantic mismatch for mode {mode}"

    # 2. Canonical JSON string byte-identity
    canonical_json = json.dumps(resolved_dict, indent=2, sort_keys=True) + "\n"
    fixture_content = fixture_path.read_text()
    assert canonical_json == fixture_content, f"Byte diff detected for mode {mode}"
