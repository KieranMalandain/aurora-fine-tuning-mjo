"""tests/test_dataset_index.py — Regression tests for Lesson 5: A glob is not a year filter.

Reference:
    00_CONTEXT.md §3 (Lesson 5)
    02_UPSTREAM_CONTRACT.md §4.2
    03_DOMAIN_PRIORS.md §1

Background:
    The pre-v3 dataset built one global index from the reference surface variable
    and reused it across all variables. It returned 54,060 timesteps for 1980–2015,
    which is exactly 1980–2016 inclusive (leaking 2016, the first validation year).
    v3 fixes this by:
      1. Building independent per-variable timestamp maps from each variable's own
         `time` coordinate.
      2. Dropping timestamps outside [start_year, end_year] at build time.
      3. Creating a sample timeline as the sorted intersection across all variables.
      4. Enforcing exact 6-hour spacing for valid sample starts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aurora_mjo.dataset import LANLMJODataset


def test_leap_year_1980_sample_count(synthetic_root: Path, synthetic_slt: Path) -> None:
    """Verify sample count for leap year 1980 is exactly 1,462 at k=1 rollout.

    Derivation:
        1980 is a leap year -> 366 days.
        4 timesteps per day (6-hourly instantaneous) -> 366 * 4 = 1,464 timesteps.
        Single-step rollout (k=1) requires 3 consecutive timesteps: (t-6h, t, t+6h).
        Valid starts = timesteps - (k + 1) = 1,464 - 2 = 1,462 samples.
        Provenance: MEASURED in docs/verify_output.txt and 03_DOMAIN_PRIORS.md §1.
    """
    ds = LANLMJODataset(
        start_year=1980,
        end_year=1980,
        root_dir=synthetic_root,
        slt_path=synthetic_slt,
        max_rollout_steps=1,
    )
    assert len(ds._timeline) == 1464, f"Expected 1,464 timesteps in 1980, got {len(ds._timeline)}"
    assert len(ds) == 1462, f"Expected 1,462 samples for 1980, got {len(ds)}"


def test_non_leap_year_1981_sample_count(synthetic_root: Path, synthetic_slt: Path) -> None:
    """Verify sample count for non-leap year 1981 is exactly 1,458 at k=1 rollout.

    Derivation:
        1981 is a non-leap year -> 365 days.
        4 timesteps per day (6-hourly instantaneous) -> 365 * 4 = 1,460 timesteps.
        Single-step rollout (k=1) requires 3 consecutive timesteps: (t-6h, t, t+6h).
        Valid starts = timesteps - (k + 1) = 1,460 - 2 = 1,458 samples.
        Provenance: DERIVED per 03_DOMAIN_PRIORS.md §1.
    """
    ds = LANLMJODataset(
        start_year=1981,
        end_year=1981,
        root_dir=synthetic_root,
        slt_path=synthetic_slt,
        max_rollout_steps=1,
    )
    assert len(ds._timeline) == 1460, f"Expected 1,460 timesteps in 1981, got {len(ds._timeline)}"
    assert len(ds) == 1458, f"Expected 1,458 samples for 1981, got {len(ds)}"


def test_year_range_enforcement_prevents_validation_leak(
    synthetic_root: Path, synthetic_slt: Path
) -> None:
    """Verify year-range filtering strictly drops out-of-range timestamps.

    The synthetic archive root contains data for both 1980 and 1981.
    A naive glob without year filtering would discover both years, yielding
    1,464 + 1,460 = 2,924 timesteps (and 2,922 samples).
    This test asserts that when constructed for 1980 only, no timestamp in the
    returned timeline or sample starts falls in 1981.
    This directly covers the bug class that allowed 2016 to leak into 1980–2015.
    """
    ds = LANLMJODataset(
        start_year=1980,
        end_year=1980,
        root_dir=synthetic_root,
        slt_path=synthetic_slt,
        max_rollout_steps=1,
    )

    # 1. Total samples must match 1980 only, not 1980+1981
    assert len(ds) == 1462, f"Expected 1,462 samples, got {len(ds)} (leak suspected if ~2,922)"

    # 2. Every timestamp in the timeline must belong strictly to 1980
    for ts in ds._timeline:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        assert dt.year == 1980, f"Timestamp {dt} leaked into 1980 dataset timeline"

    # 3. First and last sample init times
    first_sample_batch, _, _ = ds[0]
    last_sample_batch, _, _ = ds[len(ds) - 1]
    assert first_sample_batch.metadata.time[0] == datetime(1980, 1, 1, 6, 0)
    assert last_sample_batch.metadata.time[0] == datetime(1980, 12, 31, 12, 0)


def test_per_variable_index_independence_and_chunking_asymmetry(
    synthetic_root: Path, synthetic_slt: Path
) -> None:
    """Verify that each variable maintains an independent timestamp map.

    In the synthetic fixture:
      - `t2` is chunked in 1 yearly file per year.
      - `q` is chunked in 2 half-yearly files per year (`_h1.nc` and `_h2.nc`).
    This asymmetry mimics real-world ERA5 chunking differences where a shared
    `(file_idx, local_idx)` index would silently produce catastrophic misalignment.
    """
    ds = LANLMJODataset(
        start_year=1980,
        end_year=1980,
        root_dir=synthetic_root,
        slt_path=synthetic_slt,
        max_rollout_steps=1,
    )

    # 1. All 11 variables must exist in the map
    all_vars = {"2t", "10u", "10v", "msl", "ttr", "tcwv", "z", "q", "t", "u", "v"}
    assert set(ds._var_ts_map.keys()) == all_vars

    # 2. Every variable map must be a distinct dictionary object in memory
    map_ids = [id(m) for m in ds._var_ts_map.values()]
    assert len(map_ids) == len(set(map_ids)), "Variable timestamp maps must not be shared objects"

    # 3. File count asymmetry between t2 and q
    t2_files = ds.surf_file_map["2t"][0]
    q_files = ds.atmos_file_map["q"][0]
    assert len(t2_files) == 1, f"Expected 1 file for t2, got {len(t2_files)}"
    assert len(q_files) == 2, f"Expected 2 files for q, got {len(q_files)}"

    # 4. Despite chunking differences, both maps report exactly 1,464 timestamps in 1980
    assert len(ds._var_ts_map["2t"]) == 1464
    assert len(ds._var_ts_map["q"]) == 1464


def test_gapped_fixture_sample_reduction_exact(
    synthetic_root: Path, synthetic_root_gapped: Path, synthetic_slt: Path
) -> None:
    """Verify that a data gap reduces sample count by exactly the derived arithmetic.

    Derivation:
        The synthetic_root_gapped archive is missing 56 consecutive timesteps of `tcwv`
        (14 days * 4 steps/day = 56 steps, from 1980-06-01 00:00 to 1980-06-14 18:00).
        Total common intersection timeline drops from 1,464 to 1,408 timesteps (1,464 - 56).
        The gap divides the timeline into two contiguous segments of lengths N1 and N2,
        with N1 + N2 = 1,408.
        Each contiguous segment loses (k + 1) = 2 boundary steps at its end because
        a 1-step sample needs 3 consecutive points: [t-6h, t, t+6h].
        Total valid starts in gapped dataset:
            (N1 - 2) + (N2 - 2) = (N1 + N2) - 4 = 1,408 - 4 = 1,404 samples.
        Total valid starts in standard dataset:
            1,464 - 2 = 1,462 samples.
        Exact sample reduction:
            1,462 - 1,404 = 58 samples (56 missing steps + 2 boundary steps before the gap).
    """
    ds_std = LANLMJODataset(
        start_year=1980,
        end_year=1980,
        root_dir=synthetic_root,
        slt_path=synthetic_slt,
        max_rollout_steps=1,
    )
    ds_gap = LANLMJODataset(
        start_year=1980,
        end_year=1980,
        root_dir=synthetic_root_gapped,
        slt_path=synthetic_slt,
        max_rollout_steps=1,
    )

    assert len(ds_std) == 1462
    assert len(ds_gap) == 1404
    sample_reduction = len(ds_std) - len(ds_gap)
    assert sample_reduction == 58, f"Expected reduction of 58 samples, got {sample_reduction}"

    # Verify no returned sample spans the gap
    six_hours_s = 6 * 3600
    timeline = ds_gap._timeline
    for start_idx in ds_gap._valid_starts:
        # A 1-step sample requires 3 timestamps: t-6h, t, t+6h
        step_times = timeline[start_idx : start_idx + 3]
        diffs = np.diff(step_times)
        assert np.all(diffs == six_hours_s), f"Start {start_idx} spans non-6h gap: {diffs}"


def test_six_hour_consecutive_spacing(synthetic_dataset: LANLMJODataset) -> None:
    """Verify consecutive timesteps in returned samples are exactly 21,600 seconds apart."""
    six_hours_s = 21600  # 6 hours in seconds
    timeline = synthetic_dataset._timeline
    sample_indices = [
        0,
        len(synthetic_dataset) // 4,
        len(synthetic_dataset) // 2,
        3 * len(synthetic_dataset) // 4,
        len(synthetic_dataset) - 1,
    ]

    for idx in sample_indices:
        start_idx = int(synthetic_dataset._valid_starts[idx])
        chain = timeline[start_idx : start_idx + 3]
        diffs = np.diff(chain)
        assert np.all(diffs == six_hours_s), f"Timesteps in sample {idx} not 6-hourly: {diffs}"


def test_train_val_split_disjointness(synthetic_root: Path, synthetic_slt: Path) -> None:
    """Verify train (1980) and val (1981) datasets have completely disjoint timestamp sets.

    In accordance with 01_TARGET_STATE.md §9, chronological splits must prevent
    any evaluation leakage.
    """
    train_ds = LANLMJODataset(
        start_year=1980,
        end_year=1980,
        root_dir=synthetic_root,
        slt_path=synthetic_slt,
        max_rollout_steps=1,
    )
    val_ds = LANLMJODataset(
        start_year=1981,
        end_year=1981,
        root_dir=synthetic_root,
        slt_path=synthetic_slt,
        max_rollout_steps=1,
    )

    train_ts_set = set(train_ds._timeline.tolist())
    val_ts_set = set(val_ds._timeline.tolist())

    overlap = train_ts_set & val_ts_set
    assert len(overlap) == 0, f"Train and validation splits overlap by {len(overlap)} timestamps"
