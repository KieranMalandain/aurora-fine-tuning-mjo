"""tests/test_loss.py - Verification suite for normalised, weighted, area-correct grid loss.

Covers Task H1 requirements from 01_TARGET_STATE.md §D8 and 02_SCIENTIFIC_CONTRACT.md §4:
1. Headline regression test: per-variable gradient share matches w_v / Σw_v when
   normalised errors are equal.
2. Direct regression proof: pre-H1 loss fails gradient share test (msl/z take >98%, q takes ~0%).
3. Area weights integrate to 1 over the sphere.
4. Level weights sum to 1 (both pressure_delta and uniform).
5. Multiplicative separation of a(φ) and m(φ).
6. Cross-variable perturbation isolation.
7. Configuration defaults and schema verification.
"""

from __future__ import annotations

import math

import pytest
import torch
import yaml
from aurora import Batch, Metadata

from aurora_mjo.config import load_config
from aurora_mjo.loss import (
    DEFAULT_AURORA_PLEVS,
    DEFAULT_VARIABLE_WEIGHTS,
    SpectralLoss,
    TropicalWeightedL1Loss,
)

SURFACE_VARS = ["2t", "10u", "10v", "msl", "ttr", "tcwv"]
ATMOS_VARS = ["z", "q", "t", "u", "v"]
ALL_VARS = SURFACE_VARS + ATMOS_VARS


def _make_synth_batch_and_target(
    H: int = 180,
    W: int = 360,
    device: str = "cpu",
):
    """Create a synthetic Batch and target dictionary with unit normalised errors."""
    lat = torch.linspace(89.5, -89.5, H)
    lon = torch.linspace(0.5, 359.5, W)

    from datetime import datetime

    metadata = Metadata(
        lat=lat,
        lon=lon,
        time=(datetime(2020, 1, 1, 0, 0),),
        atmos_levels=DEFAULT_AURORA_PLEVS,
    )

    surf_vars_dict = {}
    target_dict = {}
    for v in SURFACE_VARS:
        # Target at 0.0, pred requiring grad
        t = torch.zeros(1, H, W, device=device)
        p = torch.zeros(1, H, W, device=device, requires_grad=True)
        surf_vars_dict[v] = p
        target_dict[v] = t

    atmos_vars_dict = {}
    for v in ATMOS_VARS:
        t = torch.zeros(1, len(DEFAULT_AURORA_PLEVS), H, W, device=device)
        p = torch.zeros(1, len(DEFAULT_AURORA_PLEVS), H, W, device=device, requires_grad=True)
        atmos_vars_dict[v] = p
        target_dict[v] = t

    batch = Batch(
        surf_vars=surf_vars_dict,
        static_vars={},
        atmos_vars=atmos_vars_dict,
        metadata=metadata,
    )
    return batch, target_dict


# ===========================================================================
# 1. Headline Test: Per-Variable Gradient Share
# ===========================================================================


def test_headline_per_variable_gradient_share():
    """Headline test (01_TARGET_STATE.md D8, 03_DOMAIN_PRIORS.md §5):

    When all 11 variables have equal normalised errors (e.g. 1.0 σ), the share
    of the L_grid gradient budget received by each variable must be within tolerance
    of w_v / Σw_v.
    """
    H, W = 180, 360
    lat_coords = torch.linspace(89.5, -89.5, H)
    loss_fn = TropicalWeightedL1Loss(
        lat_coords=lat_coords,
        level_weighting="pressure_delta",
    )

    batch, target_dict = _make_synth_batch_and_target(H=H, W=W)

    # Set each variable's prediction so that normalized error equals 1.0 everywhere:
    # x_pred_raw = mu + 1.0 * sigma -> normalized error = |(mu + sigma - mu)/sigma - 0| = 1.0
    # For atmospheric variables, each level gets mu_l + 1.0 * sigma_l.
    for v in SURFACE_VARS:
        mean = getattr(loss_fn, f"surf_mean_{v}")
        std = getattr(loss_fn, f"surf_std_{v}")
        p = (mean + 1.0 * std).repeat(1, H, W).detach().requires_grad_(True)
        batch.surf_vars[v] = p
        target_dict[v] = torch.full((1, H, W), mean.item())

    for v in ATMOS_VARS:
        mean = getattr(loss_fn, f"atmos_mean_{v}").view(1, 13, 1, 1)
        std = getattr(loss_fn, f"atmos_std_{v}").view(1, 13, 1, 1)
        p = (mean + 1.0 * std).repeat(1, 1, H, W).detach().requires_grad_(True)
        batch.atmos_vars[v] = p
        target_dict[v] = mean.repeat(1, 1, H, W)

    total_loss, var_losses = loss_fn.compute_batch(batch, target_dict)
    total_loss.backward()

    # Calculate L1 gradient magnitude with respect to normalized inputs
    grad_shares = {}
    for v in SURFACE_VARS:
        p = batch.surf_vars[v]
        std = getattr(loss_fn, f"surf_std_{v}")
        # dp_norm = dp_raw * std, so dL/dp_norm = dL/dp_raw * std
        assert p.grad is not None
        grad_norm = (p.grad * std).abs().sum().item()
        grad_shares[v] = grad_norm

    for v in ATMOS_VARS:
        p = batch.atmos_vars[v]
        std = getattr(loss_fn, f"atmos_std_{v}").view(1, 13, 1, 1)
        assert p.grad is not None
        grad_norm = (p.grad * std).abs().sum().item()
        grad_shares[v] = grad_norm

    total_grad = sum(grad_shares.values())
    measured_shares = {k: v / total_grad for k, v in grad_shares.items()}

    # Compute expected shares: w_v / Σ w_v
    weights = loss_fn.variable_weights
    total_weight = sum(weights[k] for k in ALL_VARS)
    expected_shares = {k: weights[k] / total_weight for k in ALL_VARS}

    for k in ALL_VARS:
        # Assert each variable's gradient share matches w_v / Σ w_v within 1% relative tolerance
        err_msg = (
            f"Variable {k}: measured share {measured_shares[k]:.4f} "
            f"!= expected {expected_shares[k]:.4f}"
        )
        assert measured_shares[k] == pytest.approx(expected_shares[k], rel=0.01), err_msg

    # Verify key specific ratios:
    # msl : q must be 0.5 : 2.0 = 1 : 4 (0.25), resolving the 5,808,226 : 1 defect
    ratio_msl_to_q = measured_shares["msl"] / measured_shares["q"]
    assert ratio_msl_to_q == pytest.approx(0.5 / 2.0, rel=0.01)


# ===========================================================================
# 2. Pre-H1 Regression Proof Test
# ===========================================================================


def test_pre_h1_loss_fails_gradient_share():
    """Verify that the pre-H1 unnormalised physical loss fails the gradient share test.

    Direct regression test for R1 (00_CONTEXT.md R1):
    In pre-H1, physical L1 without normalisation gives msl + z > 98% of gradient
    and q < 0.0001%.
    """
    with open("configs/norm_stats_1980_2015.yaml") as f:
        stats_data = yaml.safe_load(f)

    sigmas = {}
    for v in SURFACE_VARS:
        sigmas[v] = float(stats_data["surface"][v]["std"])
    for v in ATMOS_VARS:
        sigmas[v] = sum(
            float(stats_data["atmos"][v][p]["std"]) for p in DEFAULT_AURORA_PLEVS
        ) / len(DEFAULT_AURORA_PLEVS)

    # Simulate pre-H1 TropicalWeightedL1Loss in raw physical units:
    # unweighted average across 11 variables in physical units
    grad_norms = {}
    for v in SURFACE_VARS:
        x_norm = torch.zeros(1, 180, 360, requires_grad=True)
        # Denormalisation: x_raw = x_norm * sigma
        x_raw = x_norm * sigmas[v]
        l1_phys = (x_raw + sigmas[v]).abs().mean()
        l1_phys.backward()
        assert x_norm.grad is not None
        grad_norms[v] = x_norm.grad.abs().sum().item()

    for v in ATMOS_VARS:
        x_norm = torch.zeros(1, 13, 180, 360, requires_grad=True)
        x_raw = x_norm * sigmas[v]
        l1_phys = (x_raw + sigmas[v]).abs().mean()
        l1_phys.backward()
        assert x_norm.grad is not None
        grad_norms[v] = x_norm.grad.abs().sum().item()

    total_grad = sum(grad_norms.values())
    pre_h1_shares = {k: v / total_grad for k, v in grad_norms.items()}

    # Check pre-H1 defect R1 values
    assert pre_h1_shares["msl"] > 0.65  # ~70.6%
    assert pre_h1_shares["z"] > 0.25  # ~28.4%
    assert pre_h1_shares["msl"] + pre_h1_shares["z"] > 0.98  # >99% between them
    assert pre_h1_shares["q"] < 1e-5  # ~0.000012%

    # Confirm that pre-H1 fails the H1 target share assertion
    with pytest.raises(AssertionError):
        # Expected share for q under H1 is ~17.4% (w=2.0 / 11.5)
        assert pre_h1_shares["q"] == pytest.approx(2.0 / 11.5, rel=0.10)


# ===========================================================================
# 3. Area Weights Integration
# ===========================================================================


def test_area_weights_integrate_to_one_over_sphere():
    """Area weights a(φ) = cos(φ) / mean(cos(φ)) must integrate to 1 over the sphere."""
    lat_coords = torch.linspace(89.5, -89.5, 180)
    loss = TropicalWeightedL1Loss(lat_coords=lat_coords)

    # 1. Discrete mean over latitude grid points is identically 1.0
    assert loss.area_weights.mean().item() == pytest.approx(1.0, abs=1e-6)

    # 2. Discrete 2D spherical average over (180, 360) is identically 1.0
    spherical_grid = loss.area_weights.unsqueeze(-1).repeat(1, 360)
    assert spherical_grid.mean().item() == pytest.approx(1.0, abs=1e-6)

    # 3. Cos-latitude properties: equatorial weight > polar weight
    equator_idx = 89  # latitude +0.5 / -0.5
    pole_idx = 0  # latitude 89.5
    assert loss.area_weights[equator_idx].item() > 50.0 * loss.area_weights[pole_idx].item()


# ===========================================================================
# 4. Vertical Level Weights Sum to 1
# ===========================================================================


def test_level_weights_sum_to_one():
    """Per-level weights c_ℓ must sum to 1.0 for both pressure_delta and uniform."""
    lat_coords = torch.linspace(89.5, -89.5, 180)

    # Default pressure_delta
    loss_p = TropicalWeightedL1Loss(lat_coords=lat_coords, level_weighting="pressure_delta")
    assert loss_p.c_l.sum().item() == pytest.approx(1.0, abs=1e-6)

    # Lower troposphere (e.g. 850 hPa) has higher layer thickness than upper levels (50 hPa)
    idx_50 = 0
    idx_850 = 10
    assert loss_p.c_l[idx_850].item() > loss_p.c_l[idx_50].item()

    # Uniform alternative
    loss_u = TropicalWeightedL1Loss(lat_coords=lat_coords, level_weighting="uniform")
    assert loss_u.c_l.sum().item() == pytest.approx(1.0, abs=1e-6)
    expected_uniform = 1.0 / len(DEFAULT_AURORA_PLEVS)
    for w in loss_u.c_l:
        assert w.item() == pytest.approx(expected_uniform, abs=1e-6)


# ===========================================================================
# 5. Multiplicative Separation of a(φ) and m(φ)
# ===========================================================================


def test_multiplicative_separation_area_and_tropical():
    """a(φ) and m(φ) are separate and multiplicative."""
    lat_coords = torch.linspace(89.5, -89.5, 180)
    loss1 = TropicalWeightedL1Loss(
        lat_coords=lat_coords, tropics_weight=1.0, extratropics_weight=0.1
    )
    loss2 = TropicalWeightedL1Loss(
        lat_coords=lat_coords, tropics_weight=2.0, extratropics_weight=0.5
    )

    # Changing tropics_weight and extratropics_weight must NOT change area_weights
    assert torch.equal(loss1.area_weights, loss2.area_weights)

    # spatial_weights is strictly equal to a(φ) * m(φ)
    assert torch.allclose(loss1.spatial_weights, loss1.area_weights * loss1.tropical_weights)
    assert torch.allclose(loss2.spatial_weights, loss2.area_weights * loss2.tropical_weights)


# ===========================================================================
# 6. Cross-Variable Perturbation Isolation
# ===========================================================================


def test_cross_variable_isolation():
    """A constant offset in one variable moves only that variable's component."""
    H, W = 180, 360
    lat_coords = torch.linspace(89.5, -89.5, H)
    loss_fn = TropicalWeightedL1Loss(lat_coords=lat_coords)

    batch, target_dict = _make_synth_batch_and_target(H=H, W=W)

    # Initial zero error
    _, initial_losses = loss_fn.compute_batch(batch, target_dict)

    # Perturb only 'msl'
    batch_perturbed, target_dict2 = _make_synth_batch_and_target(H=H, W=W)
    batch_perturbed.surf_vars["msl"] = batch_perturbed.surf_vars["msl"] + 1000.0  # +1000 Pa

    _, perturbed_losses = loss_fn.compute_batch(batch_perturbed, target_dict2)

    # 'msl' loss must change
    assert perturbed_losses["msl"].item() > initial_losses["msl"].item()

    # All other 10 variables must have identical zero loss
    for v in ALL_VARS:
        if v != "msl":
            assert math.isclose(
                perturbed_losses[v].item(), initial_losses[v].item(), abs_tol=1e-7
            ), f"Variable {v} was affected by msl perturbation!"


# ===========================================================================
# 7. Config Defaults & Integration
# ===========================================================================


def test_config_loss_grid_defaults():
    """Verify configs/unified.yaml defaults and schema validation."""
    cfg = load_config("configs/unified.yaml", mode="baseline")
    grid_cfg = cfg.loss.grid

    assert grid_cfg.enabled is True
    assert grid_cfg.weight == 1.0
    assert grid_cfg.level_weighting == "pressure_delta"
    assert grid_cfg.tropics_bbox == [-20, 20]
    assert grid_cfg.tropics_weight == 1.0
    assert grid_cfg.extratropics_weight == 0.1

    # Verify w_v matches 02_SCIENTIFIC_CONTRACT.md §4.1
    vw = grid_cfg.variable_weights
    assert vw is not None
    for k, v in DEFAULT_VARIABLE_WEIGHTS.items():
        assert vw[k] == v, f"Default variable weight mismatch for {k}: {vw[k]} != {v}"


# ===========================================================================
# 8. SpectralLoss H2 — per-variable, rfft2 over (lat, lon), amplitude spectrum
# ===========================================================================

_SPEC_H = 32  # small synthetic grid for spectral tests (fast; full 180x360 not needed)
_SPEC_W = 64


def _make_spectral_loss(H: int = _SPEC_H, W: int = _SPEC_W) -> SpectralLoss:
    """Return a SpectralLoss with a custom small-grid Hann window."""
    lat_coords = torch.linspace(89.5, -89.5, H)
    return SpectralLoss(lat_coords=lat_coords)


def test_spectral_sinusoid_concentrates_in_correct_wavenumber():
    """A pure sinusoid at wavenumber k_lon concentrates spectral energy in bin k_lon.

    The amplitude-spectrum loss between a pure sinusoid and a zero field
    should be entirely due to the single non-zero frequency bin at k_lon.
    Equivalently, the max spectral-difference bin index along the last FFT
    axis must equal k_lon.
    """
    H, W = _SPEC_H, _SPEC_W
    loss_fn = _make_spectral_loss(H, W)

    # Construct a 1-D sinusoid in longitude at wavenumber k_lon
    k_lon = 5
    lons = torch.arange(W, dtype=torch.float32)
    sinusoid = torch.sin(2.0 * torch.pi * k_lon * lons / W)  # (W,)

    # Shape (B=1, L=1, H, W)
    pred = sinusoid.view(1, 1, 1, W).expand(1, 1, H, W).clone()
    target = torch.zeros_like(pred)

    # Apply the Hann window manually to match what SpectralLoss does
    win = loss_fn.hann_window.view(H, 1)  # (H, 1)
    p_win = pred[0, 0] * win  # (H, W)
    t_win = target[0, 0] * win

    p_fft = torch.fft.rfft2(p_win, norm="ortho")
    t_fft = torch.fft.rfft2(t_win, norm="ortho")
    amp_diff = torch.abs(torch.abs(p_fft) - torch.abs(t_fft))  # (H, W//2+1)

    # Mean over latitude
    mean_per_lon_bin = amp_diff.mean(dim=0)  # (W//2+1,)
    dominant_bin = int(mean_per_lon_bin.argmax().item())

    assert dominant_bin == k_lon, (
        f"Expected dominant bin {k_lon}, got {dominant_bin}. "
        "SpectralLoss is not concentrating energy at the sinusoid's wavenumber."
    )

    # The scalar loss from forward() must also be positive
    scalar_loss = loss_fn.forward(pred, target)
    assert scalar_loss.item() > 0.0, "Sinusoid vs. zeros should give positive spectral loss."


def test_spectral_identical_fields_give_zero():
    """Two identical fields must yield exactly zero spectral loss.

    This also verifies that normalisation does not introduce a bias.
    """
    H, W = _SPEC_H, _SPEC_W
    loss_fn = _make_spectral_loss(H, W)

    field = torch.randn(1, 4, H, W)  # (B=1, L=4, H, W)
    loss = loss_fn.forward(field, field)
    assert (
        loss.item() == 0.0
    ), f"Identical fields should give exactly 0 spectral loss, got {loss.item()}"


def test_spectral_shift_distinguishes_amplitude_vs_complex():
    """A spatially-shifted field should give a small amplitude-spectrum loss
    but a large complex-coefficient loss.

    This is the distinguishing test between the two formulations.

    Physical interpretation: the amplitude spectrum measures *texture* — what
    spatial frequencies are present.  A field and its cyclic shift have the
    same texture (same amplitudes) but completely different phases, so:
      - amplitude loss ≈ 0 (small but not exactly zero due to Hann window)
      - complex loss >> 0

    The test pastes both numbers into the docstring for the result file.
    """
    H, W = _SPEC_H, _SPEC_W
    loss_fn = _make_spectral_loss(H, W)

    # Multi-scale random field in (B=1, H, W) — 3-D for forward()
    torch.manual_seed(42)
    field = torch.randn(1, H, W)
    shift = W // 4  # 90-degree longitude shift
    shifted = torch.roll(field, shifts=shift, dims=-1)

    # 1. Amplitude spectrum loss (H2 implementation)
    amp_loss = loss_fn.forward(field, shifted).item()

    # 2. Complex coefficient loss (pre-H2 / naive formulation)
    #    Apply the same Hann window for a fair comparison
    win = loss_fn.hann_window.view(1, H, 1)  # (1, H, 1)
    p_win = field.unsqueeze(1) * win  # (1, 1, H, W)
    s_win = shifted.unsqueeze(1) * win
    complex_loss = (
        torch.abs(
            torch.fft.rfft2(p_win.float(), norm="ortho")
            - torch.fft.rfft2(s_win.float(), norm="ortho")
        )
        .mean()
        .item()
    )

    # The amplitude loss must be much smaller than the complex loss.
    # A cyclic shift is the canonical example: perfect amplitude match,
    # large phase mismatch.
    assert amp_loss < complex_loss, (
        f"Expected amp_loss ({amp_loss:.6f}) < complex_loss ({complex_loss:.6f}). "
        "Spatial shift should produce near-zero amplitude loss but large complex loss."
    )

    # Quantitative ratio: complex_loss should be at least 5x amp_loss
    assert complex_loss > 5 * amp_loss, (
        f"Ratio complex/amplitude = {complex_loss / max(amp_loss, 1e-12):.1f}x — "
        "expected at least 5x. The two formulations are not sufficiently distinguished."
    )

    # Print both numbers for the result file
    print(
        f"\n[H2 shift test]  amplitude_loss={amp_loss:.6f}  "
        f"complex_loss={complex_loss:.6f}  "
        f"ratio={complex_loss / max(amp_loss, 1e-12):.1f}x",
    )
