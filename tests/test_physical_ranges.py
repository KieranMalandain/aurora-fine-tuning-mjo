"""tests/test_physical_ranges.py — Offline regression tests asserting physical bounds.

Task G4: Physical-sanity fingerprint at 1° native resolution.
Validates:
  1. All 6 surface variables adhere to 03_DOMAIN_PRIORS.md §3 physical ranges.
  2. All 5 atmospheric variables across 13 levels adhere to §3 physical ranges.
  3. The three named traps:
     - ttr mean is negative (downward-positive flux convention)
     - surface z mean / 9.80665 ≈ 378 m (geopotential, not geopotential height)
     - q spans 4 orders of magnitude between 1000 hPa and 50 hPa
  4. Static variables: lsm in [0, 1], slt categorical integers {0..7}.
  5. Normalized sigma limits under G3 statistics (configs/norm_stats_1980_2015.yaml).
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from aurora_mjo.dataset import LANLMJODataset

NORM_STATS_PATH = Path(__file__).resolve().parent.parent / "configs" / "norm_stats_1980_2015.yaml"


def test_surface_variables_physical_bounds(synthetic_dataset: LANLMJODataset) -> None:
    """Verify surface variables fall within 03_DOMAIN_PRIORS.md §3 literature bounds."""
    batch, _, _ = synthetic_dataset[0]
    surf = batch.surf_vars

    # 1. 2t: K, plausible mean 275-290 K, hard bounds 180-340 K
    t2 = surf["2t"]
    assert 180.0 <= t2.min().item(), f"2t min {t2.min().item()} < 180 K"
    assert t2.max().item() <= 340.0, f"2t max {t2.max().item()} > 340 K"
    assert 275.0 <= t2.mean().item() <= 290.0, f"2t mean {t2.mean().item()} outside [275, 290]"

    # 2. 10u, 10v: m/s, plausible mean -1 to 1 m/s, hard bounds -110 to 110 m/s
    u10 = surf["10u"]
    v10 = surf["10v"]
    assert -110.0 <= u10.min().item() and u10.max().item() <= 110.0
    assert -110.0 <= v10.min().item() and v10.max().item() <= 110.0
    assert -1.5 <= u10.mean().item() <= 1.5, f"10u mean {u10.mean().item()} outside [-1.5, 1.5]"
    assert -1.5 <= v10.mean().item() <= 1.5, f"10v mean {v10.mean().item()} outside [-1.5, 1.5]"

    # 3. msl (surface pressure): Pa, plausible mean 96,000-99,000 Pa, hard bounds 47,000-108,000 Pa
    msl = surf["msl"]
    assert 47000.0 <= msl.min().item(), f"msl min {msl.min().item()} < 47,000 Pa"
    assert msl.max().item() <= 108000.0, f"msl max {msl.max().item()} > 108,000 Pa"
    msl_mean = msl.mean().item()
    assert 96000.0 <= msl_mean <= 99000.0, f"msl mean {msl_mean} outside [96000, 99000]"

    # 4. ttr: W/m^2, strictly negative, plausible mean -250 to -215, hard bounds -400 to -50
    ttr = surf["ttr"]
    assert -400.0 <= ttr.min().item(), f"ttr min {ttr.min().item()} < -400"
    assert ttr.max().item() <= -50.0, f"ttr max {ttr.max().item()} > -50"
    ttr_mean = ttr.mean().item()
    assert -250.0 <= ttr_mean <= -215.0, f"ttr mean {ttr_mean} outside [-250, -215]"
    assert ttr.max().item() < 0.0, "ttr must be strictly negative"

    # 5. tcwv: kg/m^2, strictly non-negative, plausible mean 18-26, hard bounds 0-100
    tcwv = surf["tcwv"]
    assert 0.0 <= tcwv.min().item(), f"tcwv min {tcwv.min().item()} < 0"
    assert tcwv.max().item() <= 100.0, f"tcwv max {tcwv.max().item()} > 100"
    assert 15.0 <= tcwv.mean().item() <= 30.0, f"tcwv mean {tcwv.mean().item()} outside [15, 30]"


def test_atmospheric_variables_physical_bounds(synthetic_dataset: LANLMJODataset) -> None:
    """Verify atmospheric variables across 13 levels fall within §3 literature bounds."""
    batch, _, _ = synthetic_dataset[0]
    atmos = batch.atmos_vars
    plevs = list(synthetic_dataset.atmos_levels)

    # 1. Temperature t: K, hard bounds 170-330 K
    t = atmos["t"]
    assert 170.0 <= t.min().item() and t.max().item() <= 330.0
    idx_500 = plevs.index(500)
    t_500_mean = t[:, :, idx_500, :, :].mean().item()
    assert 245.0 <= t_500_mean <= 260.0, f"t@500hPa mean {t_500_mean} outside [245, 260]"

    # 2. Geopotential z: m^2/s^2, hard bounds -5000 to 250000 m^2/s^2
    z = atmos["z"]
    assert -5000.0 <= z.min().item() and z.max().item() <= 250000.0
    z_500_mean = z[:, :, idx_500, :, :].mean().item()
    assert 50000.0 <= z_500_mean <= 58000.0, f"z@500hPa mean {z_500_mean} outside [50000, 58000]"

    # 3. Specific humidity q: kg/kg, strictly positive, decreasing with height
    q = atmos["q"]
    assert q.min().item() > 0.0, f"q min {q.min().item()} must be strictly positive"
    assert q.max().item() <= 0.05, f"q max {q.max().item()} > 0.05"
    idx_1000 = plevs.index(1000)
    idx_50 = plevs.index(50)
    q_1000_mean = q[:, :, idx_1000, :, :].mean().item()
    q_50_mean = q[:, :, idx_50, :, :].mean().item()
    assert 0.005 <= q_1000_mean <= 0.030, f"q@1000hPa mean {q_1000_mean} outside [0.005, 0.030]"
    assert 1e-7 <= q_50_mean <= 1e-5, f"q@50hPa mean {q_50_mean} outside [1e-7, 1e-5]"

    # 4. Wind components u, v: m/s
    u = atmos["u"]
    v = atmos["v"]
    assert -150.0 <= u.min().item() and u.max().item() <= 150.0
    assert -150.0 <= v.min().item() and v.max().item() <= 150.0
    idx_200 = plevs.index(200)
    u_200_mean = u[:, :, idx_200, :, :].mean().item()
    assert -50.0 <= u_200_mean <= 50.0


def test_named_traps_verdicts(synthetic_dataset: LANLMJODataset) -> None:
    """Verify the three named traps from G4 step 3."""
    batch, _, _ = synthetic_dataset[0]

    # Trap 1: ttr mean is negative
    ttr_mean = batch.surf_vars["ttr"].mean().item()
    assert ttr_mean < 0.0, f"TRAP 1: ttr mean {ttr_mean} is non-negative!"

    # Trap 2: surface z mean / 9.80665 ≈ 378 m
    z_surf_mean = batch.static_vars["z"].mean().item()
    equiv_h = z_surf_mean / 9.80665
    diff_h = abs(equiv_h - 378.0)
    assert diff_h < 30.0, f"TRAP 2: Surface z mean / 9.80665 is {equiv_h} m"

    # Trap 3: q spans 4 orders of magnitude across 13 levels
    plevs = list(synthetic_dataset.atmos_levels)
    idx_1000 = plevs.index(1000)
    idx_50 = plevs.index(50)
    q_1000 = batch.atmos_vars["q"][:, :, idx_1000, :, :].mean().item()
    q_50 = batch.atmos_vars["q"][:, :, idx_50, :, :].mean().item()
    ratio = q_1000 / q_50
    assert ratio > 1000.0, f"TRAP 3: q vertical range ratio {ratio:.1f} < 1000x!"


def test_static_variables_constraints(synthetic_dataset: LANLMJODataset) -> None:
    """Verify static variable constraints (lsm in [0, 1], slt integer-valued 0-7)."""
    batch, _, _ = synthetic_dataset[0]
    statics = batch.static_vars

    # lsm in [0, 1]
    lsm = statics["lsm"]
    assert 0.0 <= lsm.min().item() and lsm.max().item() <= 1.0, "lsm outside [0, 1]"

    # slt categorical integers 0..7
    slt = statics["slt"]
    slt_unique = torch.unique(slt).cpu().numpy()
    is_valid_slt = all(float(x).is_integer() and 0 <= x <= 7 for x in slt_unique)
    assert is_valid_slt, f"slt contains invalid categorical values: {slt_unique}"


def test_normalization_sigmas_on_synthetic_data(synthetic_dataset: LANLMJODataset) -> None:
    """Verify that normalized synthetic fields do not exhibit pathological sigma values."""
    assert NORM_STATS_PATH.exists(), f"Missing norm stats at {NORM_STATS_PATH}"
    with open(NORM_STATS_PATH) as f:
        norm_stats = yaml.safe_load(f)

    batch, _, _ = synthetic_dataset[0]

    # Surface variables
    for var, tensor in batch.surf_vars.items():
        mean_val = norm_stats["surface"][var]["mean"]
        std_val = norm_stats["surface"][var]["std"]
        norm_t = (tensor.detach().cpu().to(torch.float64) - mean_val) / std_val
        worst_sigma = float(torch.abs(norm_t).max().item())
        assert worst_sigma < 10.0, f"Pathological sigma for {var}: {worst_sigma:.2f} >= 10 σ"

    # Atmospheric variables
    plevs = list(synthetic_dataset.atmos_levels)
    for var, tensor in batch.atmos_vars.items():
        t_cpu = tensor.detach().cpu().to(torch.float64)
        for idx, plev in enumerate(plevs):
            mean_val = norm_stats["atmos"][var][plev]["mean"]
            std_val = norm_stats["atmos"][var][plev]["std"]
            norm_t = (t_cpu[:, :, idx, :, :] - mean_val) / std_val
            worst_sigma = float(torch.abs(norm_t).max().item())
            assert worst_sigma < 10.0, f"Pathological sigma {var}@{plev}: {worst_sigma:.2f} >= 10 σ"
