"""tests/test_env.py — Tests for process environment configuration and guards.

Reference:
    docs/campaigns/refactor/tasks/E1_env_guards_hard_fail.md
    00_CONTEXT.md §3 (Lesson 3)
    02_UPSTREAM_CONTRACT.md §4.5
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

import pytest

from aurora_mjo.env import (
    HDF5_USE_FILE_LOCKING_DEFAULT,
    HDF5_USE_FILE_LOCKING_KEY,
    configure_environment,
)


def test_configure_environment_sets_unset_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify configure_environment sets HDF5_USE_FILE_LOCKING=FALSE when unset."""
    monkeypatch.delenv(HDF5_USE_FILE_LOCKING_KEY, raising=False)
    assert HDF5_USE_FILE_LOCKING_KEY not in os.environ

    changed = configure_environment()

    assert os.environ.get(HDF5_USE_FILE_LOCKING_KEY) == HDF5_USE_FILE_LOCKING_DEFAULT
    assert changed == {HDF5_USE_FILE_LOCKING_KEY: HDF5_USE_FILE_LOCKING_DEFAULT}


def test_configure_environment_preserves_existing_value(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify configure_environment does NOT overwrite a pre-existing custom setting and warns."""
    monkeypatch.setenv(HDF5_USE_FILE_LOCKING_KEY, "TRUE")

    with caplog.at_level(logging.WARNING):
        changed = configure_environment()

    assert os.environ.get(HDF5_USE_FILE_LOCKING_KEY) == "TRUE"
    assert changed == {}
    assert "already set to 'TRUE'" in caplog.text


def test_configure_environment_noop_if_already_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify configure_environment is a no-op if already set to FALSE."""
    monkeypatch.setenv(HDF5_USE_FILE_LOCKING_KEY, HDF5_USE_FILE_LOCKING_DEFAULT)

    changed = configure_environment()

    assert os.environ.get(HDF5_USE_FILE_LOCKING_KEY) == HDF5_USE_FILE_LOCKING_DEFAULT
    assert changed == {}


def test_import_aurora_mjo_sets_env_var() -> None:
    """Verify importing aurora_mjo package sets HDF5_USE_FILE_LOCKING in a fresh process."""
    env = {k: v for k, v in os.environ.items() if k != HDF5_USE_FILE_LOCKING_KEY}
    code = (
        "import os\n"
        "assert 'HDF5_USE_FILE_LOCKING' not in os.environ, 'Variable must be unset initially'\n"
        "import aurora_mjo\n"
        "assert os.environ.get('HDF5_USE_FILE_LOCKING') == 'FALSE', 'Failed to set on import'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Subprocess import failed with code {result.returncode}:\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )


def test_direct_submodule_import_sets_env_var() -> None:
    """Verify directly importing aurora_mjo.dataset sets HDF5_USE_FILE_LOCKING."""
    env = {k: v for k, v in os.environ.items() if k != HDF5_USE_FILE_LOCKING_KEY}
    code = (
        "import os\n"
        "assert 'HDF5_USE_FILE_LOCKING' not in os.environ, 'Variable must be unset initially'\n"
        "from aurora_mjo.dataset import LANLMJODataset\n"
        "assert os.environ.get('HDF5_USE_FILE_LOCKING') == 'FALSE', "
        "'Failed to set on submodule import'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Subprocess submodule import failed with code {result.returncode}:\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )


def test_env_var_set_before_netcdf4_import() -> None:
    """Verify environment variable is active before netCDF4 library initialises."""
    env = {k: v for k, v in os.environ.items() if k != HDF5_USE_FILE_LOCKING_KEY}
    code = (
        "import os\n"
        "assert 'HDF5_USE_FILE_LOCKING' not in os.environ\n"
        "import aurora_mjo\n"
        "assert os.environ.get('HDF5_USE_FILE_LOCKING') == 'FALSE'\n"
        "import netCDF4\n"
        "assert os.environ.get('HDF5_USE_FILE_LOCKING') == 'FALSE'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Subprocess netCDF4 ordering check failed:\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )
