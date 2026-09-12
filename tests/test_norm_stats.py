"""tests/test_norm_stats.py — Regression tests for Lesson 1: Normalisation statistics and MSL proxy.

Reference:
    00_CONTEXT.md §3 (Lesson 1)
    02_UPSTREAM_CONTRACT.md §4.1
    03_DOMAIN_PRIORS.md §4

Background:
    `dataset.py` maps Aurora's `msl` channel to LANL's `ps` (surface pressure) because
    the ERA5 archive has no true mean-sea-level pressure. Aurora natively normalises `msl`
    using built-in constants (location=100958, scale=1332). Surface pressure over high
    terrain (e.g. Tibetan Plateau ~52,000 Pa) therefore lands at roughly -36.8 sigma.
    This was the confirmed trigger for 100% non-finite validation loss in July 2026.
    `configs/unified.yaml` overrides `model.norm_stats.msl` with surface-pressure stats,
    pulling the worst-case high-terrain input to -4.7 sigma.

    This test module verifies:
      1. `model.norm_stats.msl` override is present in all four modes.
      2. `load_model` actively mutates `aurora.normalisation.locations` and `.scales`.
      3. A placeholder detector flags that current values are Aurora's built-in `sp` stats
         and requires real computation via `scripts/calc_norm_stats.py` before production.
      4. The sigma arithmetic (-36.8 sigma vs -4.7 sigma) holds mathematically.
      5. Normalisation training periods do not overlap with validation or test years (anti-leakage).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from aurora import Aurora
from aurora.normalisation import locations, scales

from aurora_mjo.cli_support import load_config
from aurora_mjo.model import load_model

MODES = ("baseline", "physics_informed", "lora", "combined")

# Known placeholder values currently placed in configs/unified.yaml
PLACEHOLDER_MSL_MEAN = 96667.9822
PLACEHOLDER_MSL_STD = 9504.6359


@pytest.fixture
def config_path() -> Path:
    """Path to canonical configs/unified.yaml."""
    return Path(__file__).resolve().parent.parent / "configs" / "unified.yaml"


def test_msl_override_present_in_all_modes(config_path: Path) -> None:
    """Verify model.norm_stats.msl override exists in all four modes.

    CRITICAL REPO RULE (AGENTS.md §5, Lesson 1):
    Never remove the model.norm_stats.msl override in configs/unified.yaml.
    Removing it forces surface pressure through MSL normalisation, restoring a
    -36 sigma input and 100% non-finite validation loss.
    """
    for mode in MODES:
        cfg = load_config(config_path, mode=mode)
        norm_stats = cfg.get("model", {}).get("norm_stats", {})
        assert "msl" in norm_stats, (
            f"Mode '{mode}' missing model.norm_stats.msl override! "
            "Its absence triggers fatal -36 sigma inputs on high terrain."
        )
        msl_cfg = norm_stats["msl"]
        assert "mean" in msl_cfg and "std" in msl_cfg, f"Mode '{mode}' missing mean/std"
        assert msl_cfg["std"] > 0, f"Mode '{mode}' msl std must be positive."


def test_msl_override_applied_to_aurora_locations_and_scales(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify load_model actively mutates aurora.normalisation.locations and scales.

    An override that is parsed into the config but not passed to Aurora's global
    normalisation tables would silently fail to protect the model from -36 sigma inputs.
    """
    monkeypatch.setattr(Aurora, "load_checkpoint", lambda self, strict=True: None)

    cfg = load_config(config_path, mode="baseline")
    model_cfg = cfg["model"]
    msl_mean = model_cfg["norm_stats"]["msl"]["mean"]
    msl_std = model_cfg["norm_stats"]["msl"]["std"]

    # Record and preserve original global values
    orig_loc = locations.get("msl")
    orig_scale = scales.get("msl")

    try:
        # Intentionally tamper global tables before load_model to verify they get overwritten
        locations["msl"] = 100958.0
        scales["msl"] = 1332.0

        load_model(model_cfg, norm_stats=model_cfg.get("norm_stats"))

        assert locations["msl"] == msl_mean, "locations['msl'] not updated to config mean"
        assert scales["msl"] == msl_std, "scales['msl'] not updated to config std"
    finally:
        # Restore global state so other tests are not polluted
        if orig_loc is not None:
            locations["msl"] = orig_loc
        if orig_scale is not None:
            scales["msl"] = orig_scale


def test_placeholder_norm_stats_guard(config_path: Path) -> None:
    """Detect if placeholder normalisation statistics are currently active.

    The current msl override values (mean=96667.9822, std=9504.6359) are Aurora's own
    built-in `sp` stats, used as a stopgap to unblock development. They are not stats
    computed over this dataset's training years.
    Per 00_CONTEXT.md §5 and D3 task spec:
      - This test must XFAIL (or warn), not hard-fail, because computing real stats
        is out of scope for this refactoring campaign.
      - It explicitly names `scripts/calc_norm_stats.py` as the required fix.
    """
    cfg = load_config(config_path, mode="baseline")
    msl_stats = cfg["model"]["norm_stats"]["msl"]

    is_placeholder = (
        abs(msl_stats["mean"] - PLACEHOLDER_MSL_MEAN) < 1e-3
        and abs(msl_stats["std"] - PLACEHOLDER_MSL_STD) < 1e-3
    )

    if is_placeholder:
        pytest.xfail(
            f"msl norm_stats in configs/unified.yaml are PLACEHOLDER values "
            f"({msl_stats['mean']}, {msl_stats['std']}). "
            f"Must be replaced by computing real statistics over 1980–2015 using "
            f"`scripts/calc_norm_stats.py` before executing production science runs."
        )


def test_sigma_arithmetic_tibetan_plateau() -> None:
    """Executable verification of Lesson 1 sigma arithmetic from 03_DOMAIN_PRIORS.md §4.

    Tibetan Plateau surface pressure at ~5,300 m elevation is roughly 52,000 Pa.
    1. Under Aurora's built-in MSL constants (mean=100958, scale=1332):
         (52000 - 100958) / 1332 = -36.76 sigma -> FATAL: drives validation loss to NaN.
    2. Under the surface pressure override (mean=96668, scale=9505):
         (52000 - 96668) / 9505 = -4.70 sigma -> REASONABLE: within physical dynamic range.
    """
    p_high_terrain = 52000.0  # Pa (~5.3 km elevation)

    # Built-in MSL parameters
    aurora_msl_loc = 100958.0
    aurora_msl_scale = 1332.0

    sigma_builtin = (p_high_terrain - aurora_msl_loc) / aurora_msl_scale
    assert sigma_builtin < -30.0, (
        f"Expected built-in MSL normalisation to exceed -30 sigma on Tibetan Plateau, "
        f"got {sigma_builtin:.2f} sigma."
    )

    # Config override parameters
    override_loc = PLACEHOLDER_MSL_MEAN
    override_scale = PLACEHOLDER_MSL_STD

    sigma_override = (p_high_terrain - override_loc) / override_scale
    assert -6.0 < sigma_override < -4.0, (
        f"Expected override normalisation to be manageable (-6 to -4 sigma), "
        f"got {sigma_override:.2f} sigma."
    )

    # Scale factor ratio: surface pressure varies ~7x more than MSL across global topography
    scale_ratio = override_scale / aurora_msl_scale
    assert scale_ratio > 7.0, f"Expected override scale to be >7x MSL scale, got {scale_ratio:.2f}x"


def test_train_period_only_normalisation_bounds(config_path: Path) -> None:
    """Assert normalisation derivation period is strictly disjoint from val and test periods.

    Per 01_TARGET_STATE.md §9 and 03_DOMAIN_PRIORS.md §6:
    Normalisation statistics and climatology must derive strictly from the training
    period to prevent evaluation leakage.
    """
    for mode in MODES:
        cfg: dict[str, Any] = load_config(config_path, mode=mode)
        real_data = cfg.get("data", {}).get("real", {})
        train_years = real_data.get("train_years")
        val_years = real_data.get("val_years")
        test_years = real_data.get("test_years")

        assert train_years is not None and len(train_years) == 2
        assert val_years is not None and len(val_years) == 2
        assert test_years is not None and len(test_years) == 2

        # Assert strictly chronological non-overlapping splits
        assert train_years[0] <= train_years[1], "train_years must be ordered"
        assert val_years[0] <= val_years[1], "val_years must be ordered"
        assert test_years[0] <= test_years[1], "test_years must be ordered"

        assert train_years[1] < val_years[0], "Train period end must precede val period start"
        assert val_years[1] < test_years[0], "Val period end must precede test period start"
