"""Process-start environment configuration and error types for aurora_mjo.

Incident & Rationale (Lesson 3 / 02_UPSTREAM_CONTRACT.md §4.5):
    On NERSC Perlmutter, attempting to open NetCDF4 files on the CFS parallel filesystem
    (Lustre / GPFS) with xarray or netCDF4 frequently failed with:

        OSError: [Errno -101] NetCDF: HDF error: '.../e5.oper.an.sfc.128_167_2t....nc'

    These files exist, have correct permissions, and are fully readable. The underlying
    failure is HDF5 library advisory file locking across network/parallel filesystem mounts.

    On 8 September 2026, measurement confirmed that setting the environment variable:

        export HDF5_USE_FILE_LOCKING=FALSE

    completely resolves the error, allowing the exact same file to open cleanly
    (e.g., shape 180x360x124, variable 't2').

    Previously, this environment variable was set only in `slurm_scripts/train_auto.slurm`.
    A guard that lives only in a shell script is not a guard: interactive sessions,
    direct python calls, non-auto SLURM scripts (eval.slurm, test_train.slurm), and
    subprocesses will fail if they omit this variable. Furthermore, HDF5 reads this
    environment variable during C-library initialisation (when netCDF4/h5py is first
    imported), so setting it inside Python must happen before any netCDF4 or h5py
    module import.

    This module provides `configure_environment()` to set load-bearing environment variables
    at process startup with zero external dependencies (stdlib only, no torch, no logging
    handlers configured).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

HDF5_USE_FILE_LOCKING_KEY = "HDF5_USE_FILE_LOCKING"
HDF5_USE_FILE_LOCKING_DEFAULT = "FALSE"


class StaticVarLoadError(RuntimeError):
    """Raised when an invariant or static variable (z, lsm, slt) fails to load.

    Replaces dangerous zero-tensor fallbacks that silently degrade model physics
    into learning a planet with no topography and no continents (Lesson 4).
    """


def configure_environment() -> dict[str, str]:
    """Configure critical process-level environment variables before library initialisation.

    Sets `HDF5_USE_FILE_LOCKING=FALSE` if unset in `os.environ`. If already set to a
    different value, issues a warning instead of overwriting, respecting deliberate
    user overrides.

    Returns:
        A dictionary of environment variable names and their applied values for variables
        that were newly set or modified by this call.
    """
    changed: dict[str, str] = {}

    current = os.environ.get(HDF5_USE_FILE_LOCKING_KEY)
    if current is None:
        os.environ[HDF5_USE_FILE_LOCKING_KEY] = HDF5_USE_FILE_LOCKING_DEFAULT
        changed[HDF5_USE_FILE_LOCKING_KEY] = HDF5_USE_FILE_LOCKING_DEFAULT
    elif current != HDF5_USE_FILE_LOCKING_DEFAULT:
        logger.warning(
            "Environment variable %s is already set to %r (default is %r). "
            "Preserving user override; parallel filesystem NetCDF reads may encounter "
            "Errno -101 HDF locking errors.",
            HDF5_USE_FILE_LOCKING_KEY,
            current,
            HDF5_USE_FILE_LOCKING_DEFAULT,
        )

    return changed
