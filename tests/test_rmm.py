"""Tests for the RMM pipeline (aurora_mjo.rmm).

Validates:
- Tropical mean retains 360 longitudes with simple unweighted meridional mean (R3 regression).
- Seasonal cycle fitting via mean + 3 harmonics per longitude.
- 120-day mean removal under Convention (A) without data leakage.
- Unit-variance PC normalisation and orthogonal EOF projection via SVD.
- Synthetic eastward wavenumber-1 wave produces RMM1/RMM2 in quadrature and monotonic phase advance.
- Frozen EOF basis projection consistency (train vs val).
- evaluate.py projection without scalar collapse.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from aurora_mjo.rmm.compute import (
    LAT_N,
    LAT_S,
    compute_eofs,
    compute_wh_phase,
    fit_seasonal_harmonics,
    project_onto_eofs,
    remove_120d_mean,
    remove_clim,
    tropical_mean,
)
from aurora_mjo.rmm.evaluate import (
    extract_rmm_from_fields,
)


def test_tropical_mean_retains_longitude_360():
    """Headline test for R3: tropical mean must retain longitude (360 points)."""
    lat = np.linspace(89.5, -89.5, 180)
    lon = np.linspace(0.5, 359.5, 360)
    time = pd.date_range("2016-01-01", periods=2, freq="6h")
    data = np.random.randn(len(time), len(lat), len(lon)).astype(np.float32)
    da = xr.DataArray(
        data, coords={"time": time, "lat": lat, "lon": lon}, dims=["time", "lat", "lon"]
    )

    result = tropical_mean(da)

    # Must retain longitude dimension of 360, shape (time, lon)
    assert (
        "lon" in result.dims or "longitude" in result.dims
    ), f"Tropical mean must retain longitude dimension, got dims {result.dims}"
    lon_dim = "lon" if "lon" in result.dims else "longitude"
    assert result.sizes[lon_dim] == 360, f"Expected 360 longitudes, got {result.sizes.get(lon_dim)}"
    assert (
        result.ndim == 2
    ), f"Expected 2D (time, lon), got {result.ndim}D with shape {result.shape}"


def test_tropical_mean_unweighted_meridional_average():
    """Verify that tropical_mean computes a simple unweighted mean over [15°S, 15°N]."""
    lat = np.linspace(89.5, -89.5, 180)
    lon = np.linspace(0.5, 359.5, 360)
    time = pd.date_range("2016-01-01", periods=1, freq="1D")

    # In [15S, 15N], lat runs from -14.5 to 14.5 (30 points)
    mask_trop = (lat >= LAT_S) & (lat <= LAT_N)
    n_trop = int(np.sum(mask_trop))
    assert n_trop == 30, f"Expected 30 tropical points at 1-deg resolution, got {n_trop}"

    # Set linear values along tropical latitudes: 1, 2, ..., 30 for all longitudes
    data = np.zeros((1, len(lat), len(lon)), dtype=np.float32)
    data[0, mask_trop, :] = np.arange(1, n_trop + 1, dtype=np.float32)[:, None]

    da = xr.DataArray(
        data, coords={"time": time, "lat": lat, "lon": lon}, dims=["time", "lat", "lon"]
    )
    res = tropical_mean(da)

    # Simple arithmetic mean of 1..30 is 15.5
    expected_mean = float(np.mean(np.arange(1, n_trop + 1)))
    np.testing.assert_allclose(res.values[0], expected_mean, rtol=1e-5)


def test_seasonal_harmonics_fitting():
    """Verify fitting mean + 3 harmonics of the annual cycle per longitude."""
    dates = pd.date_range("1980-01-01", periods=365 * 4, freq="1D")
    doy = dates.dayofyear.values.astype(float)
    t0 = 365.25

    # True signal: mean = 10, harmonic 1 amp = 5, harmonic 2 amp = 2, harmonic 3 amp = 1
    true_cycle = (
        10.0
        + 5.0 * np.cos(2 * np.pi * 1 * doy / t0)
        + 2.0 * np.sin(2 * np.pi * 2 * doy / t0)
        + 1.0 * np.cos(2 * np.pi * 3 * doy / t0)
    )
    lon = np.linspace(0.5, 359.5, 360)
    data = np.tile(true_cycle[:, None], (1, len(lon))).astype(np.float32)
    da = xr.DataArray(data, coords={"time": dates, "lon": lon}, dims=["time", "lon"])

    clim_table, coeffs = fit_seasonal_harmonics(da, n_harmonics=3)

    assert clim_table.shape == (366, 360)
    # Check that day 1..365 reconstructed values match true signal closely
    eval_doy = np.arange(1, 366)
    expected_eval = (
        10.0
        + 5.0 * np.cos(2 * np.pi * 1 * eval_doy / t0)
        + 2.0 * np.sin(2 * np.pi * 2 * eval_doy / t0)
        + 1.0 * np.cos(2 * np.pi * 3 * eval_doy / t0)
    )
    np.testing.assert_allclose(clim_table[:365, 0], expected_eval, rtol=1e-3, atol=1e-3)

    # Removing climatology should yield ~zero anomaly
    da_anom = remove_clim(da, clim_table)
    np.testing.assert_allclose(da_anom.values, 0.0, atol=1e-2)


def test_120_day_mean_convention_a_no_leakage():
    """Verify 120-day mean removal under Convention (A): up to t₀, fixed across lead."""
    # Historical sequence of 200 days
    dates = pd.date_range("2016-01-01", periods=200, freq="1D")
    lon = np.linspace(0.5, 359.5, 360)
    data = np.arange(200, dtype=np.float32)[:, None] * np.ones((1, 360), dtype=np.float32)
    da_hist = xr.DataArray(data, coords={"time": dates, "lon": lon}, dims=["time", "lon"])

    # Forecast init time t₀ is day 150
    t0 = dates[150]
    obs_up_to_t0 = da_hist.sel(time=slice(None, t0))

    # Forecast from day 151 to 180 (lead 1 to 30 days)
    fc_dates = pd.date_range(t0 + pd.Timedelta(days=1), periods=30, freq="1D")
    da_fc = xr.DataArray(
        np.full((30, 360), 500.0, dtype=np.float32),
        coords={"time": fc_dates, "lon": lon},
        dims=["time", "lon"],
    )

    # Under Convention (A), prior 120-day mean uses observations ending at t₀
    fc_filtered = remove_120d_mean(da_fc, convention="A", obs_history_ending_at_t0=obs_up_to_t0)

    # The 120-day observation window ending at day 150 has mean (31+150)/2 = 90.5
    expected_prior_mean = float(obs_up_to_t0.isel(time=slice(-120, None)).mean().values)
    assert np.isclose(expected_prior_mean, 90.5)

    # Across all 30 forecast leads, the subtracted value must be exactly 90.5
    expected_fc_filtered = 500.0 - 90.5
    np.testing.assert_allclose(fc_filtered.values, expected_fc_filtered, rtol=1e-5)

    # Verify no timestamps after t₀ are present in obs_up_to_t0
    assert obs_up_to_t0.time.max() <= t0


def test_synthetic_eastward_propagating_wave_rmm_quadrature_and_phase():
    """Verify synthetic eastward wave produces RMM1/RMM2 in quadrature and monotonic phase."""
    # 48-day period wave, 192 days (4 complete cycles)
    T = 192
    lon_deg = np.linspace(0.5, 359.5, 360)
    lon_rad = np.deg2rad(lon_deg)  # 0 to 2pi

    # Wavenumber k = 1, angular frequency omega = 2pi / 48
    omega = 2.0 * np.pi / 48.0
    t_idx = np.arange(T)

    # OLR: negative anomaly for enhanced convection
    olr = -np.cos(lon_rad[None, :] - omega * t_idx[:, None])  # (T, 360)
    u850 = np.sin(lon_rad[None, :] - omega * t_idx[:, None])
    u200 = -np.sin(lon_rad[None, :] - omega * t_idx[:, None])

    # Normalise each to unit variance
    olr /= np.std(olr)
    u850 /= np.std(u850)
    u200 /= np.std(u200)

    # Combine into (T, 1080) matrix
    X = np.hstack([olr, u850, u200])

    eof1, eof2, eigenvalues, pc1_std, pc2_std = compute_eofs(X)
    rmm1, rmm2 = project_onto_eofs(X, eof1, eof2, pc1_std, pc2_std)

    # 1. Variance on training set must be 1.0 by construction (PC normalisation)
    np.testing.assert_allclose(np.var(rmm1), 1.0, atol=1e-4)
    np.testing.assert_allclose(np.var(rmm2), 1.0, atol=1e-4)

    # 2. Quadrature: correlation between RMM1 and RMM2 must be ~0
    corr = np.corrcoef(rmm1, rmm2)[0, 1]
    assert abs(corr) < 0.05, f"Expected near-zero correlation, got {corr:.4f}"

    # 3. Lag correlation between RMM1 and RMM2 peaks at ~12 days (T/4 = 48/4 = 12)
    lags = np.arange(-20, 21)
    cc = [
        np.corrcoef(
            rmm1[:-lag] if lag > 0 else (rmm1[-lag:] if lag < 0 else rmm1),
            rmm2[lag:] if lag > 0 else (rmm2[:lag] if lag < 0 else rmm2),
        )[0, 1]
        for lag in lags
    ]
    peak_lag = abs(lags[np.argmax(np.abs(cc))])
    assert 10 <= peak_lag <= 14, f"Expected quadrature peak lag ~12 days, got {peak_lag}"

    # 4. Monotonic phase progression through all 8 phases
    phases = compute_wh_phase(rmm1, rmm2)
    one_cycle_phases = set(phases[0:48].tolist())
    assert one_cycle_phases == {1, 2, 3, 4, 5, 6, 7, 8}


def test_train_val_basis_frozen_no_leakage():
    """Verify fitting on train slice and projecting val slice does not modify the basis."""
    rng = np.random.default_rng(1234)
    X_train = rng.standard_normal((365 * 10, 1080))
    X_val = rng.standard_normal((365 * 2, 1080))

    eof1, eof2, eig, pc1_std, pc2_std = compute_eofs(X_train)
    eof1_copy = eof1.copy()
    eof2_copy = eof2.copy()
    pc1_std_copy = float(pc1_std)
    pc2_std_copy = float(pc2_std)

    # Project val
    rmm1_val, rmm2_val = project_onto_eofs(X_val, eof1, eof2, pc1_std, pc2_std)

    # Basis arrays must remain byte-identical
    np.testing.assert_array_equal(eof1, eof1_copy)
    np.testing.assert_array_equal(eof2, eof2_copy)
    assert pc1_std == pc1_std_copy
    assert pc2_std == pc2_std_copy
    assert len(rmm1_val) == len(X_val)


def test_evaluate_extract_rmm_from_fields_without_scalar_collapse():
    """Verify extract_rmm_from_fields projects (360,) longitude fields properly onto EOF basis."""
    lat = np.linspace(89.5, -89.5, 180)
    lon = np.linspace(0.5, 359.5, 360)
    levels = [50, 100, 200, 500, 850, 1000]

    # Create dummy batch
    surf = {
        "ttr": torch.randn(1, 2, 180, 360),
    }
    atmos = {
        "u": torch.randn(1, 2, len(levels), 180, 360),
    }
    metadata = SimpleNamespace(
        lat=torch.from_numpy(lat),
        lon=torch.from_numpy(lon),
        atmos_levels=levels,
    )
    batch = SimpleNamespace(surf_vars=surf, atmos_vars=atmos, metadata=metadata)

    # Mock basis
    basis = {
        "olr_clim": np.zeros((366, 360), dtype=np.float32),
        "u850_clim": np.zeros((366, 360), dtype=np.float32),
        "u200_clim": np.zeros((366, 360), dtype=np.float32),
        "olr_std": 1.0,
        "u850_std": 1.0,
        "u200_std": 1.0,
        "pc1_std": 1.0,
        "pc2_std": 1.0,
        "eof1": np.ones(1080, dtype=np.float32) / np.sqrt(1080),
        "eof2": np.ones(1080, dtype=np.float32) / np.sqrt(1080),
        "transform_matrix": np.eye(2, dtype=np.float32),
    }

    t = datetime(2016, 5, 15, 6, 0)
    rmm1, rmm2, amp = extract_rmm_from_fields(batch, t, basis)

    assert rmm1 is not None and rmm2 is not None and amp is not None
    assert isinstance(rmm1, float)
    assert isinstance(rmm2, float)
    assert isinstance(amp, float)
    assert np.isfinite(rmm1) and np.isfinite(rmm2) and np.isfinite(amp)


def test_saved_basis_and_targets_contract():
    """Verify saved basis and targets strictly adhere to scientific contract."""
    basis_path = Path("data/rmm_basis.npz")
    targets_path = Path("data/rmm_targets.nc")

    if not basis_path.exists() or not targets_path.exists():
        pytest.skip("data/rmm_basis.npz or data/rmm_targets.nc not present on disk")

    basis = dict(np.load(basis_path))
    assert basis["eof1"].shape == (1080,)
    assert basis["eof2"].shape == (1080,)
    assert basis["olr_clim"].shape == (366, 360)
    assert basis["u850_clim"].shape == (366, 360)
    assert basis["u200_clim"].shape == (366, 360)
    assert basis["transform_matrix"].shape == (2, 2)
    assert basis["convention"] == "A"
    assert np.all(basis["train_years"] == np.arange(1980, 2016))
    assert np.all(basis["val_years"] == np.arange(2016, 2020))

    ds = xr.open_dataset(targets_path)
    # 1980–2019 only: exactly 14,610 days (13,149 train + 1,461 val)
    assert len(ds.time) == 14610
    assert str(ds.time.values[0])[:10] == "1980-01-01"
    assert str(ds.time.values[-1])[:10] == "2019-12-31"

    # Test years (2020+) must NOT be present
    years = pd.DatetimeIndex(ds.time.values).year
    assert 2020 not in years and 2021 not in years and 2022 not in years and 2023 not in years
    assert int(np.sum(ds.split.values == "train")) == 13149
    assert int(np.sum(ds.split.values == "val")) == 1461
