"""tests/test_stats.py — Unit tests for Welford accumulator and parallel reduction."""

from __future__ import annotations

import numpy as np

from aurora_mjo.stats import Welford


def test_welford_single_element() -> None:
    """Verify single-element initialization and update."""
    w = Welford()
    assert w.n == 0
    assert w.mean == 0.0
    assert w.std == 0.0

    w.update(42.5)
    assert w.n == 1
    assert np.isclose(w.mean, 42.5, atol=1e-12)
    assert w.var == 0.0
    assert w.std == 0.0


def test_welford_known_distribution_chunks() -> None:
    """Verify streaming Welford matches numpy.mean and numpy.std across arbitrary chunk sizes."""
    rng = np.random.default_rng(seed=12345)
    # Generate 50,000 samples from Normal(mu=285.0, sigma=15.0)
    data = rng.normal(loc=285.0, scale=15.0, size=50000)

    expected_mean = float(np.mean(data))
    expected_std = float(np.std(data, ddof=0))  # population std
    expected_var = float(np.var(data, ddof=0))

    w = Welford()

    # Stream in variable-sized chunks
    chunk_sizes = [1, 7, 13, 100, 500, 2048, 10000, 5000]
    idx = 0
    c_idx = 0
    while idx < len(data):
        cs = chunk_sizes[c_idx % len(chunk_sizes)]
        chunk = data[idx : idx + cs]
        w.update_batch(chunk)
        idx += cs
        c_idx += 1

    assert w.n == len(data)
    assert np.isclose(w.mean, expected_mean, atol=1e-11)
    assert np.isclose(w.var, expected_var, atol=1e-10)
    assert np.isclose(w.std, expected_std, atol=1e-10)


def test_welford_chunk_boundaries_and_empty() -> None:
    """Verify robust handling of empty chunks, 1-element chunks, and multi-dim arrays."""
    w = Welford()

    # Empty array
    w.update_batch(np.array([]))
    assert w.n == 0

    # 1-element array
    w.update_batch(np.array([10.0]))
    assert w.n == 1
    assert w.mean == 10.0
    assert w.std == 0.0

    # Multi-dimensional array
    arr_2d = np.array([[12.0, 14.0], [16.0, 18.0]])
    w.update_batch(arr_2d)
    assert w.n == 5

    expected = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    assert np.isclose(w.mean, float(np.mean(expected)), atol=1e-12)
    assert np.isclose(w.std, float(np.std(expected, ddof=0)), atol=1e-12)


def test_welford_merge_parallel_reduction() -> None:
    """Verify Chan's merge formula produces exact results compared to unified numpy reduction."""
    rng = np.random.default_rng(seed=999)
    d1 = rng.normal(loc=100.0, scale=20.0, size=15000)
    d2 = rng.normal(loc=105.0, scale=25.0, size=25000)
    d3 = rng.normal(loc=95.0, scale=18.0, size=10000)

    all_data = np.concatenate([d1, d2, d3])
    expected_mean = float(np.mean(all_data))
    expected_std = float(np.std(all_data, ddof=0))

    w1 = Welford()
    w1.update_batch(d1)

    w2 = Welford()
    w2.update_batch(d2)

    w3 = Welford()
    w3.update_batch(d3)

    # Merge w2 and w3 into w1
    w1.merge(w2)
    w1.merge(w3)

    assert w1.n == len(all_data)
    assert np.isclose(w1.mean, expected_mean, atol=1e-11)
    assert np.isclose(w1.std, expected_std, atol=1e-10)

    # Merging with empty accumulator is a no-op
    w_empty = Welford()
    w1.merge(w_empty)
    assert w1.n == len(all_data)
    assert np.isclose(w1.mean, expected_mean, atol=1e-11)

    # Merging empty into non-empty
    w_empty.merge(w1)
    assert w_empty.n == len(all_data)
    assert np.isclose(w_empty.mean, expected_mean, atol=1e-11)


def test_welford_non_finite_filtering() -> None:
    """Verify non-finite values (NaN, Inf) are filtered out without poisoning the accumulator."""
    w = Welford()
    data_with_nans = np.array([1.0, 2.0, np.nan, 3.0, np.inf, -np.inf, 4.0])

    w.update_batch(data_with_nans)
    assert w.n == 4
    expected = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(w.mean, float(np.mean(expected)), atol=1e-12)
    assert np.isclose(w.std, float(np.std(expected, ddof=0)), atol=1e-12)

    # Scalar update with NaN
    w.update(float("nan"))
    assert w.n == 4
