"""aurora_mjo package."""

from __future__ import annotations

from aurora_mjo.env import StaticVarLoadError, configure_environment

# Configure process environment guards (specifically HDF5_USE_FILE_LOCKING=FALSE)
# at package import before any netCDF4 / h5py C-level library initialisation or
# subsequent submodule imports.
configure_environment()

__version__ = "0.1.0"

__all__ = ["StaticVarLoadError", "__version__", "configure_environment"]
