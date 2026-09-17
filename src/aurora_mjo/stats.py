# src/aurora_mjo/stats.py
"""Numerically stable running statistics accumulator (Welford / Chan) and dataset normalization.

This module extracts and packages the Welford accumulator (previously private in
scripts/calc_norm_stats.py) to enable full testability, chunk streaming, and
parallel reduction via Chan's merge formula.

It provides routines to compute true mean and standard deviation over specified
year ranges (e.g., 1980–2015 training set) for surface variables and per-level
atmospheric variables, adhering strictly to anti-leakage year constraints (Lesson 5).
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

# Ensure CFS file locking is disabled across worker processes
if "HDF5_USE_FILE_LOCKING" not in os.environ:
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

log = logging.getLogger(__name__)

# Aurora's standard 13 pressure levels (hPa)
AURORA_PLEVS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

SURFACE_VAR_MAP: dict[str, tuple[str, str]] = {
    "2t": ("Step02/ERA5.remap_180x360MODIS_6hrInst/T2", "t2"),
    "10u": ("Step02/ERA5.remap_180x360MODIS_6hrInst/U10", "u10"),
    "10v": ("Step02/ERA5.remap_180x360MODIS_6hrInst/V10", "v10"),
    "msl": ("Step02/ERA5.remap_180x360MODIS_6hrInst/PS", "ps"),
    "ttr": ("Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX", "mtnlwrf"),
    "tcwv": ("Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv", "tcwv"),
}

ATMOS_VAR_MAP: dict[str, tuple[str, str]] = {
    "z": ("Step01/ERA5.remap_180x360MODIS_6hrInst/gopt", "z"),
    "q": ("Step01/ERA5.remap_180x360MODIS_6hrInst/sphu", "q"),
    "t": ("Step01/ERA5.remap_180x360MODIS_6hrInst/tprt", "t"),
    "u": ("Step01/ERA5.remap_180x360MODIS_6hrInst/uWnd", "u"),
    "v": ("Step01/ERA5.remap_180x360MODIS_6hrInst/vWnd", "v"),
}

# Regex to safely extract YYYYMM from filename patterns like .198001_ or _198001.
# Prevents false positive substring matches (e.g. *2001*.nc matching 202001, Lesson 5)
_DATE_REGEX = re.compile(r"[\._](\d{4})(\d{2})[\._]")


class Welford:
    """Numerically stable single-pass mean and variance accumulator.

    Supports streaming batch updates and parallel reduction via Chan's
    algorithm (1979) for combining sample sets:
      delta = mu_B - mu_A
      mu = mu_A + delta * n_B / n
      M2 = M2_A + M2_B + delta^2 * n_A * n_B / n
    """

    __slots__ = ("m2", "mean", "n")

    def __init__(self, n: int = 0, mean: float = 0.0, m2: float = 0.0) -> None:
        self.n: int = int(n)
        self.mean: float = float(mean)
        self.m2: float = float(m2)

    def update(self, val: float | np.ndarray) -> None:
        """Update accumulator with a scalar or numpy array."""
        if isinstance(val, int | float | np.floating | np.integer):
            v = float(val)
            if not np.isfinite(v):
                return
            new_n = self.n + 1
            delta = v - self.mean
            self.mean += delta / new_n
            self.m2 += delta * (v - self.mean)
            self.n = new_n
        else:
            self.update_batch(val)

    def update_batch(self, values: np.ndarray) -> None:
        """Vectorized update from an array of arbitrary shape, ignoring non-finites."""
        vals = np.asarray(values, dtype=np.float64).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return

        batch_n = vals.size
        batch_mean = float(vals.mean())
        batch_var = float(vals.var())

        new_n = self.n + batch_n
        delta = batch_mean - self.mean
        self.mean += delta * batch_n / new_n
        self.m2 += batch_var * batch_n + (delta**2) * self.n * batch_n / new_n
        self.n = new_n

    def merge(self, other: Welford) -> Welford:
        """In-place merge with another Welford accumulator (Chan's algorithm)."""
        if other.n == 0:
            return self
        if self.n == 0:
            self.n = other.n
            self.mean = other.mean
            self.m2 = other.m2
            return self

        new_n = self.n + other.n
        delta = other.mean - self.mean
        self.mean += delta * other.n / new_n
        self.m2 += other.m2 + (delta**2) * self.n * other.n / new_n
        self.n = new_n
        return self

    @property
    def var(self) -> float:
        """Population variance (ddof=0)."""
        return float(self.m2 / max(self.n, 1))

    @property
    def std(self) -> float:
        """Population standard deviation (ddof=0)."""
        return float(np.sqrt(self.var))

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": round(float(self.mean), 4),
            "std": round(float(self.std), 4),
            "n": self.n,
        }

    def __repr__(self) -> str:
        return f"Welford(n={self.n}, mean={self.mean:.4f}, std={self.std:.4f})"


def collect_variable_files(
    root: Path,
    step_subdir: str,
    start_year: int,
    end_year: int,
) -> list[Path]:
    """Discover files strictly belonging to [start_year, end_year] via regex.

    Guarantees that files from outside the year range (especially quarantined
    test years 2020–2023) are never matched or processed (Lesson 5).
    """
    var_dir = root / step_subdir
    if not var_dir.exists():
        return []

    matched: list[tuple[int, int, Path]] = []
    for f in var_dir.glob("*.nc"):
        m = _DATE_REGEX.search(f.name)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            if start_year <= year <= end_year:
                matched.append((year, month, f))

    # Also check per-year subdirectories if present
    for y in range(start_year, end_year + 1):
        ydir = var_dir / str(y)
        if ydir.is_dir():
            for f in ydir.glob("*.nc"):
                m = _DATE_REGEX.search(f.name)
                if m:
                    year = int(m.group(1))
                    month = int(m.group(2))
                    if start_year <= year <= end_year:
                        matched.append((year, month, f))

    # Sort deterministically by (year, month, filename) and deduplicate
    seen: set[Path] = set()
    ordered_files: list[Path] = []
    for _, _, f in sorted(matched, key=lambda x: (x[0], x[1], x[2].name)):
        if f not in seen:
            seen.add(f)
            ordered_files.append(f)

    return ordered_files


def _process_surface_file(args: tuple[Path, str]) -> Welford:
    """Worker task: compute Welford accumulator for a single surface NetCDF file."""
    fpath, var_name = args
    w = Welford()
    with xr.open_dataset(str(fpath), engine="netcdf4") as ds:
        if var_name in ds:
            w.update_batch(ds[var_name].values)
        else:
            log.warning("Variable '%s' missing from %s", var_name, fpath.name)
    return w


def _process_atmos_file(args: tuple[Path, str, list[int]]) -> dict[int, Welford]:
    """Worker task: compute Welford accumulator for a single 3D NetCDF file across 13 levels."""
    fpath, var_name, level_indices = args
    with xr.open_dataset(str(fpath), engine="netcdf4") as ds:
        if var_name not in ds:
            log.warning("Variable '%s' missing from %s", var_name, fpath.name)
            return {p: Welford() for p in AURORA_PLEVS}
        data = ds[var_name].values

    out: dict[int, Welford] = {}
    for p, idx in zip(AURORA_PLEVS, level_indices, strict=True):
        w = Welford()
        w.update_batch(data[:, idx, :, :])
        out[p] = w
    return out


def _fmt(val: float) -> float:
    """Format floating point numbers preserving precision for small values (e.g. q)."""
    if val == 0.0:
        return 0.0
    if abs(val) < 0.001:
        return float(f"{val:.6e}")
    return round(float(val), 4)


def compute_norm_stats(
    root: Path,
    start_year: int = 1980,
    end_year: int = 2015,
    variables: list[str] | None = None,
    num_workers: int = 16,
) -> dict[str, Any]:
    """Compute normalization statistics across surface and atmospheric variables.

    Args:
        root: CFS archive root directory.
        start_year: First year of training period (inclusive).
        end_year: Last year of training period (inclusive).
        variables: Optional subset of variable names to compute. Defaults to all 11.
        num_workers: Worker pool size for concurrent NetCDF reading.

    Returns:
        Dictionary containing metadata, surface statistics, atmospheric per-level
        statistics, and a flat unified dict.
    """
    requested_vars = set(variables) if variables else None

    surface_results: dict[str, dict[str, Any]] = {}
    atmos_results: dict[str, dict[int, dict[str, Any]]] = {}
    flat_norm_stats: dict[str, dict[str, float]] = {}
    total_files_read = 0

    # 1. Process Surface Variables
    for aurora_name, (step_subdir, native_name) in SURFACE_VAR_MAP.items():
        if requested_vars and (
            aurora_name not in requested_vars and native_name not in requested_vars
        ):
            continue

        files = collect_variable_files(root, step_subdir, start_year, end_year)
        if not files:
            log.warning(
                "[%s] No files found for %d–%d in %s",
                aurora_name,
                start_year,
                end_year,
                step_subdir,
            )
            continue

        total_files_read += len(files)
        tasks = [(f, native_name) for f in files]
        acc = Welford()

        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for file_acc in executor.map(_process_surface_file, tasks, chunksize=4):
                    acc.merge(file_acc)
        else:
            for task in tasks:
                acc.merge(_process_surface_file(task))

        stat_dict = {
            "mean": round(float(acc.mean), 4),
            "std": round(float(acc.std), 4),
            "n": acc.n,
        }
        surface_results[aurora_name] = stat_dict
        flat_norm_stats[aurora_name] = {"mean": stat_dict["mean"], "std": stat_dict["std"]}

        # If msl, also mirror to ps
        if aurora_name == "msl":
            surface_results["ps"] = stat_dict

        log.info(
            "[%s] mean=%.4f std=%.4f (n=%d values, %d files)",
            aurora_name, acc.mean, acc.std, acc.n, len(files)
        )

    # 2. Process Atmospheric Variables
    for aurora_name, (step_subdir, native_name) in ATMOS_VAR_MAP.items():
        if requested_vars and (
            aurora_name not in requested_vars and native_name not in requested_vars
        ):
            continue

        files = collect_variable_files(root, step_subdir, start_year, end_year)
        if not files:
            log.warning(
                "[%s] No files found for %d–%d in %s",
                aurora_name,
                start_year,
                end_year,
                step_subdir,
            )
            continue

        total_files_read += len(files)

        # Probe level indices from the first file
        with xr.open_dataset(str(files[0]), engine="netcdf4") as ds:
            lev_coord = "lev" if "lev" in ds.coords else "level"
            all_levs = ds[lev_coord].values
            level_indices = [int(np.argmin(np.abs(all_levs - p))) for p in AURORA_PLEVS]

        tasks_atmos = [(f, native_name, level_indices) for f in files]
        var_total: dict[int, Welford] = {p: Welford() for p in AURORA_PLEVS}

        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for file_dict in executor.map(_process_atmos_file, tasks_atmos, chunksize=4):
                    for p in AURORA_PLEVS:
                        var_total[p].merge(file_dict[p])
        else:
            for task in tasks_atmos:
                file_dict = _process_atmos_file(task)
                for p in AURORA_PLEVS:
                    var_total[p].merge(file_dict[p])
        atmos_results[aurora_name] = {}
        for p in AURORA_PLEVS:
            acc_p = var_total[p]
            stat_p = {"mean": _fmt(acc_p.mean), "std": _fmt(acc_p.std), "n": acc_p.n}
            atmos_results[aurora_name][p] = stat_p
            flat_norm_stats[f"{aurora_name}_{p}"] = {"mean": stat_p["mean"], "std": stat_p["std"]}

        log.info(
            "[%s] completed across 13 levels (50 hPa std=%e, 1000 hPa std=%e, %d files)",
            aurora_name, var_total[50].std, var_total[1000].std, len(files)
        )

    return {
        "surface": surface_results,
        "atmos": atmos_results,
        "norm_stats": flat_norm_stats,
        "total_files_read": total_files_read,
    }
