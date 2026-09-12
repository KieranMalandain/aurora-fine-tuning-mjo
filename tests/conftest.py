"""tests/conftest.py — Global test fixtures and configuration for aurora_mjo test suite.

See docs/campaigns/refactor/tasks/D1_pytest_scaffold.md for fixture specifications.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

# 02_UPSTREAM_CONTRACT.md §4.5:
# HDF5 advisory file locking against parallel filesystems (Lustre / GPFS / CFS)
# triggers OSError: [Errno -101] NetCDF: HDF error unless disabled.
# Setting it here ensures tests never fail on file locking, independent of package init.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from aurora_mjo.cli_support import load_config  # noqa: E402
from aurora_mjo.dataset import LANLMJODataset  # noqa: E402


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip tests marked with needs_gpu when running in a CPU-only environment."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(
            reason=(
                "CUDA device not available (torch.cuda.is_available() is False); "
                "auto-skipping needs_gpu test."
            )
        )
        for item in items:
            if "needs_gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def synthetic_root() -> Path:
    """Path to the standard 18x36 synthetic archive."""
    root = Path(__file__).parent / "fixtures" / "synthetic_archive"
    if not root.exists():
        pytest.fail(
            f"Synthetic archive directory not found at {root}. Run scripts/make_test_fixtures.py."
        )
    return root


@pytest.fixture(scope="session")
def synthetic_root_gapped() -> Path:
    """Path to the 18x36 synthetic archive with 56 timesteps of tcwv missing."""
    root = Path(__file__).parent / "fixtures" / "synthetic_archive_gapped"
    if not root.exists():
        pytest.fail(
            f"Gapped archive directory not found at {root}. Run scripts/make_test_fixtures.py."
        )
    return root


@pytest.fixture(scope="session")
def synthetic_slt() -> Path:
    """Path to the 720x1440 synthetic slt_data_synthetic.nc file."""
    path = Path(__file__).parent / "fixtures" / "slt_data_synthetic.nc"
    if not path.exists():
        pytest.fail(
            f"Synthetic slt fixture not found at {path}. Run scripts/make_test_fixtures.py."
        )
    return path


@pytest.fixture(scope="session")
def synthetic_dataset(synthetic_root: Path, synthetic_slt: Path) -> LANLMJODataset:
    """Constructed LANLMJODataset over 1980 only using synthetic fixtures."""
    return LANLMJODataset(
        start_year=1980,
        end_year=1980,
        root_dir=synthetic_root,
        slt_path=synthetic_slt,
        max_rollout_steps=1,
    )


@pytest.fixture(scope="session")
def unified_config() -> dict[str, Any]:
    """Resolved baseline config dictionary from configs/unified.yaml."""
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "unified.yaml"
    return load_config(cfg_path, mode="baseline")


@pytest.fixture(scope="session")
def baseline_fingerprint() -> dict[str, Any]:
    """B1 baseline JSON fixtures loaded into a dictionary keyed by filename stem."""
    baseline_dir = Path(__file__).parent / "fixtures" / "baseline"
    fingerprints: dict[str, Any] = {}
    for json_file in sorted(baseline_dir.glob("*.json")):
        with open(json_file) as f:
            fingerprints[json_file.stem] = json.load(f)
    return fingerprints
