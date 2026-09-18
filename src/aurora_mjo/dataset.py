# src/dataset.py
#
# NERSC/LANL ERA5 dataset for Aurora MJO fine-tuning - v3, timestamp-aligned.
#
# WHAT CHANGED IN v3 AND WHY (read this before touching indexing code)
# ====================================================================
# The previous design built ONE global index map from the *reference* surface
# variable's file list and reused (file_idx, local_time_idx) pairs for every
# other variable - silently assuming all 11 variables have identical file
# chunking and identical time axes.  Verification against Perlmutter logs
# proved this assumption false in at least one way:
#
#   * The 1980–2015 "train" dataset reported 54,060 timesteps.  That is
#     EXACTLY 1980–2016 inclusive (13,515 days x 4).  The requested range is
#     52,596.  The glob was pulling one extra year - 2016, the first
#     VALIDATION year - into training (train/val leakage), and nothing
#     enforced the year range on what the glob returned.
#   * If per-variable file lists can differ by a whole year, they can differ
#     in chunking too, which is the suspected mechanism behind the 100%
#     non-finite validation losses (cross-variable timestep misalignment).
#
# v3 makes both failure modes STRUCTURALLY IMPOSSIBLE:
#
#   1. Every variable gets its OWN {timestamp -> (file_idx, local_idx)} map,
#      built from the actual `time` coordinates in its own files.
#   2. Timestamps outside [start_year, end_year] are dropped at build time
#      (hard year-range enforcement, independent of file naming/globbing).
#   3. The sample timeline is the SORTED INTERSECTION of all variables'
#      timestamps, and a sample is only valid if every timestep it needs
#      (t-6h, t, t+6h, ..., t+(k)*6h) exists in that intersection at exact
#      6-hour spacing — gaps in any variable simply remove those samples
#      instead of producing misaligned or fill-value reads.
#   4. __init__ prints a per-variable alignment report (files, timesteps,
#      dropped-out-of-range, coverage) so any future data drift is visible
#      in the first 90 seconds of every job instead of surfacing as NaN.
#
# Fork-safety (unchanged from the known-good version): only file paths and
# plain-Python index structures are stored on `self`; every __getitem__
# opens, reads, and closes NetCDF files fresh.  Safe for any num_workers.
#
# Native 1° ingestion (G1): data and static variables are returned at native
# 1deg (180x360). Both upsamplers are retired.

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from aurora import Batch, Metadata
from torch.utils.data import Dataset

from aurora_mjo.env import StaticVarLoadError

_DEFAULT_NERSC_ROOT = "/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results"

AURORA_PLEVS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# {aurora_name: (step_subdir, glob_pattern, native_var_name)}
# NOTE — msl proxy: 'Ps' (surface pressure) stands in for 'msl' (missing from
# the LANL dataset).  Swap step_subdir + native name when real MSL lands.
SURFACE_VAR_MAP = {
    "2t": ("Step02/ERA5.remap_180x360MODIS_6hrInst/T2", "*", "t2"),
    "10u": ("Step02/ERA5.remap_180x360MODIS_6hrInst/U10", "*", "u10"),
    "10v": ("Step02/ERA5.remap_180x360MODIS_6hrInst/V10", "*", "v10"),
    "msl": ("Step02/ERA5.remap_180x360MODIS_6hrInst/PS", "*", "ps"),
    "ttr": ("Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX", "*", "mtnlwrf"),
    "tcwv": ("Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv", "*", "tcwv"),
}

ATMOS_VAR_MAP = {
    "z": ("Step01/ERA5.remap_180x360MODIS_6hrInst/gopt", "*", "z"),
    "q": ("Step01/ERA5.remap_180x360MODIS_6hrInst/sphu", "*", "q"),
    "t": ("Step01/ERA5.remap_180x360MODIS_6hrInst/tprt", "*", "t"),
    "u": ("Step01/ERA5.remap_180x360MODIS_6hrInst/uWnd", "*", "u"),
    "v": ("Step01/ERA5.remap_180x360MODIS_6hrInst/vWnd", "*", "v"),
}

STEP_HOURS = 6  # dataset cadence; must match Aurora's 6 h step
_DEFAULT_SLT_PATH = Path(__file__).resolve().parents[2] / "data/static/slt_1deg.nc"
_DEFAULT_SST_DIR = Path(__file__).resolve().parents[2] / "data/static/sst"
_DEFAULT_SYNTHETIC_SST = Path(__file__).resolve().parents[2] / "tests/fixtures/sst_synthetic.nc"


def _to_seconds_i64(times) -> np.ndarray:
    """Normalize any datetime64/cftime-decodable time axis to int64 unix seconds.

    int64 seconds (not datetime64 objects) are the map keys: exact, hashable,
    cheap to store 54k x 11 of, and immune to ns-vs-s resolution mismatches.
    """
    arr = np.asarray(times)
    if not np.issubdtype(arr.dtype, np.datetime64):
        # cftime or object datetimes → go through pandas-free conversion
        arr = np.array([np.datetime64(t) for t in arr.tolist()])
    return arr.astype("datetime64[s]").astype(np.int64)


class LANLMJODataset(Dataset):
    """Fork-safe, timestamp-aligned ERA5 dataset for Aurora MJO fine-tuning."""

    native_resolution = (180, 360)

    def __init__(
        self,
        start_year: int,
        end_year: int,
        root_dir: str | Path | None = None,
        slt_path: str | Path | None = None,
        sst_dir: str | Path | None = None,
        max_rollout_steps: int = 1,
    ):
        if root_dir is None:
            warnings.warn(
                f"root_dir not provided; falling back to default NERSC path: "
                f"{_DEFAULT_NERSC_ROOT}.",
                stacklevel=2,
            )
            root_dir = _DEFAULT_NERSC_ROOT
        self.root_dir = Path(root_dir)

        if slt_path is None:
            warnings.warn(
                f"slt_path not provided; falling back to default { _DEFAULT_SLT_PATH }.",
                stacklevel=2,
            )
            slt_path = _DEFAULT_SLT_PATH
        self.slt_path = Path(slt_path)

        if sst_dir is None:
            sst_dir = _DEFAULT_SST_DIR
        self.sst_dir = Path(sst_dir)

        self.start_year = start_year
        self.end_year = end_year
        self.max_rollout_steps = max(1, max_rollout_steps)

        print(
            f"Initializing LANL MJO Dataset ({start_year}-{end_year}) [timestamp-aligned v3]..."
        )

        # Hard year-range bounds in unix seconds — enforced on timestamps,
        # NOT on file names.  This is what closes the 2016-leakage hole.
        self._t_min = int(
            np.datetime64(f"{start_year}-01-01T00:00:00")
            .astype("datetime64[s]")
            .astype(np.int64)
        )
        self._t_max = int(
            np.datetime64(f"{end_year}-12-31T23:59:59")
            .astype("datetime64[s]")
            .astype(np.int64)
        )

        # 1. Per-variable file lists (collected first for coordinate probe)
        self.surf_file_map = self._collect_var_files(SURFACE_VAR_MAP)
        self.atmos_file_map = self._collect_var_files(ATMOS_VAR_MAP)

        # 2. Native coordinates read from archive reference file (no hard-coded linspace)
        self.lat, self.lon, self._flip_lat = self._load_grid_coordinates()
        self.atmos_levels = tuple(AURORA_PLEVS)

        # 3. Static variables (small, loaded once, fork-safe tensors)
        self.static_vars = self._load_static_vars()

        # 3. Per-variable {ts_seconds -> (file_idx, local_idx)} maps +
        #    intersection timeline + contiguity-checked sample starts.
        self._build_aligned_index()

        # 4. Pressure-level subset indices (29 LANL levels → 13 Aurora levels)
        self._plev_indices = self._probe_pressure_level_indices()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _collect_var_files(self, var_map: dict) -> dict:
        """Glob candidate files per variable.  Over-matching here is now
        harmless: the timestamp filter in _build_aligned_index() is what
        decides which timesteps are actually used."""
        result = {}
        for aurora_name, (step_subdir, glob_pattern, native_name) in var_map.items():
            var_dir = self.root_dir / step_subdir
            files: list[Path] = []
            for year in range(self.start_year, self.end_year + 1):
                year_dir = var_dir / str(year)
                matched = sorted(year_dir.glob(glob_pattern + ".nc"))
                if not matched:
                    matched = sorted(var_dir.glob(f"*{year}*" + ".nc"))
                files.extend(matched)
            # De-duplicate while preserving order (fallback glob can re-match).
            seen, uniq = set(), []
            for f in files:
                if f not in seen:
                    seen.add(f)
                    uniq.append(f)
            if not uniq:
                warnings.warn(
                    f"[LANLMJODataset] No files found for '{aurora_name}' in {var_dir} "
                    f"for years {self.start_year}–{self.end_year}. Skipping variable.",
                    stacklevel=2,
                )
                continue
            result[aurora_name] = (uniq, native_name)
        return result

    def _build_aligned_index(self):
        """Build per-variable timestamp maps and the common sample timeline.

        This is the heart of the v3 fix.  For each variable independently:
          probe each file's `time` coordinate (coords only - cheap, no data),
          convert to int64 unix seconds, DROP anything outside the requested
          year range, and record ts -> (file_idx, local_idx).  If the same
          timestamp appears in two files (overlapping files), the FIRST
          occurrence wins and the duplicate is counted + reported.

        The usable timeline is the sorted intersection across all variables;
        a sample with start index i is valid only if timeline contains the
        exact 6-hourly chain [t_i, t_i+6h, ..., t_i+(1+max_rollout_steps)*6h].
        """
        all_maps: dict[str, dict[int, tuple[int, int]]] = {}
        report_rows = []

        for aurora_name, (files, _native) in {
            **self.surf_file_map,
            **self.atmos_file_map,
        }.items():
            ts_map: dict[int, tuple[int, int]] = {}
            n_total, n_dropped, n_dupes = 0, 0, 0
            for fi, f in enumerate(files):
                with xr.open_dataset(str(f), engine="netcdf4", decode_times=True) as ds:
                    secs = _to_seconds_i64(ds.time.values)
                n_total += len(secs)
                for li, s in enumerate(secs.tolist()):
                    if s < self._t_min or s > self._t_max:
                        n_dropped += 1  # out-of-range (this was the leak)
                        continue
                    if s in ts_map:
                        n_dupes += 1  # overlapping/duplicate file content
                        continue
                    ts_map[s] = (fi, li)
            all_maps[aurora_name] = ts_map
            report_rows.append(
                (aurora_name, len(files), n_total, len(ts_map), n_dropped, n_dupes)
            )

        # Intersection across all variables.
        keysets = [set(m.keys()) for m in all_maps.values()]
        common = sorted(set.intersection(*keysets)) if keysets else []

        # Contiguity-checked sample starts:  a sample at timeline position i
        # consumes timestamps  t_i (history t-6h), t_i+6h (current t),
        # then targets t_i+12h ... t_i+(1+max_rollout_steps)*6h.
        step_s = STEP_HOURS * 3600
        need = 1 + self.max_rollout_steps + 1  # 2 inputs + max_rollout targets
        common_arr = np.asarray(common, dtype=np.int64)
        common_pos = {s: i for i, s in enumerate(common)}
        valid_starts: list[int] = []
        for i in range(len(common) - need + 1):
            # exact 6-hourly chain check (vectorized slice compare)
            if np.all(np.diff(common_arr[i : i + need]) == step_s):
                valid_starts.append(i)

        self._var_ts_map = all_maps
        self._timeline = common_arr  # int64 seconds, sorted
        self._valid_starts = np.asarray(valid_starts, dtype=np.int64)
        self.num_samples = len(self._valid_starts)

        # ---- Alignment report (prints once per rank at init) --------------
        union = set.union(*keysets) if keysets else set()
        print(
            f"  [align] requested range {self.start_year}-{self.end_year} | "
            f"common timeline: {len(common)} steps | union: {len(union)} | "
            f"valid {self.max_rollout_steps}-step samples: {self.num_samples}"
        )
        for name, nf, ntot, nkept, ndrop, ndup in report_rows:
            miss = len(common) and (len(union) - nkept)
            flag = ""
            if ndrop:
                flag += f"  DROPPED-OUT-OF-RANGE={ndrop}"
            if ndup:
                flag += f"  DUPLICATES={ndup}"
            if nkept != len(common):
                flag += f"  (this variable limits/differs from intersection by {nkept - len(common):+d})"
            print(
                f"  [align]   {name:>5}: files={nf:<4} raw_steps={ntot:<7} kept={nkept:<7}{flag}"
            )
        gaps = (len(common) - need + 1) - self.num_samples if len(common) >= need else 0
        if gaps > 0:
            print(
                f"  [align]   NOTE: {gaps} potential sample starts excluded by 6-hourly gaps."
            )

        if self.num_samples <= 0:
            raise RuntimeError(
                "[LANLMJODataset] Zero valid samples after timestamp alignment. "
                "Run tools/diagnose_val_nan.py for the per-variable audit."
            )

    def _probe_pressure_level_indices(self) -> list[int]:
        if not self.atmos_file_map:
            return []
        ref_files = next(iter(self.atmos_file_map.values()))[0]
        with xr.open_dataset(str(ref_files[0]), engine="netcdf4") as ds:
            if "lev" in ds.coords:
                all_levs = ds.lev.values
            elif "level" in ds.coords:
                all_levs = ds.level.values
            else:
                warnings.warn(
                    "[LANLMJODataset] No 'lev'/'level' coord found; using all levels.",
                    stacklevel=2,
                )
                return []
        return [int(np.argmin(np.abs(all_levs - p))) for p in AURORA_PLEVS]

    def _load_grid_coordinates(self) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Load native grid coordinates from an archive reference NetCDF file.

        Enforces Aurora's convention: latitudes must be strictly decreasing.
        If the archive stores latitudes ascending (e.g. -89.5 to 89.5 as in
        the LANL 180x360 remap), we invert them to descending (89.5 to -89.5)
        and record _flip_lat=True so that spatial slices are flipped along
        the latitude dimension (axis=-2) to keep physical orientation intact.
        """
        ref_file: Path | None = None
        static_dir = self.root_dir / "Step00/ERA5.invariant"
        z_files = list(static_dir.glob("*_z.*.nc")) if static_dir.exists() else []
        if z_files:
            ref_file = z_files[0]
        else:
            for _, (files, _) in self.surf_file_map.items():
                if files:
                    ref_file = Path(files[0])
                    break

        if ref_file is None:
            raise FileNotFoundError(
                f"No reference NetCDF file found in {self.root_dir} to read coordinates."
            )

        with xr.open_dataset(str(ref_file), engine="netcdf4") as ds:
            lat_k = "lat" if "lat" in ds.coords else "latitude"
            lon_k = "lon" if "lon" in ds.coords else "longitude"
            if lat_k not in ds.coords or lon_k not in ds.coords:
                raise KeyError(
                    f"Coordinate '{lat_k}' or '{lon_k}' not found in {ref_file}"
                )
            raw_lat = ds[lat_k].values.astype(np.float32)
            raw_lon = ds[lon_k].values.astype(np.float32)

        flip_lat = False
        if raw_lat[0] < raw_lat[-1]:
            # Archive stores latitudes ascending (South to North). Invert to descending.
            lat = raw_lat[::-1].copy()
            flip_lat = True
        else:
            lat = raw_lat.copy()

        lat_t = torch.from_numpy(lat)
        lon_t = torch.from_numpy(raw_lon.copy())

        # Validate strictly decreasing latitude and strictly increasing longitude
        if not torch.all(lat_t[1:] - lat_t[:-1] < 0):
            raise ValueError("Latitudes must be strictly decreasing.")
        if not torch.all(lon_t[1:] - lon_t[:-1] > 0):
            raise ValueError("Longitudes must be strictly increasing.")

        return lat_t, lon_t, flip_lat

    def _load_static_vars(self):
        """Load native invariant and static fields into memory.

        Static variables are small (180x360 at 1deg) and time-invariant, so we
        read them once at __init__ and reuse them in every sample.

        v3: NaN/Inf are zeroed for ALL three statics (previously only slt was sanitized).
        E1: Invariant and static variable loads fail loudly with StaticVarLoadError
        rather than substituting zero tensors on exception.
        G1: Statics are loaded at native 1° without upsampling or [:720, :] truncation.
        """
        static_dir = self.root_dir / "Step00/ERA5.invariant"

        z_files = list(static_dir.glob("*_z.*.nc"))
        if not z_files:
            raise FileNotFoundError(f"Invariant Z file not found in {static_dir}")
        lsm_files = list(static_dir.glob("*_lsm.*.nc"))
        if not lsm_files:
            raise FileNotFoundError(f"Invariant LSM file not found in {static_dir}")

        def _ensure_2d(t):
            while t.ndim > 2:
                t = t.squeeze(0)
            return t

        def _clean(t):
            return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

        z_path = z_files[0]
        try:
            with xr.open_dataset(z_path, engine="netcdf4") as ds_z:
                z_arr = ds_z["Z"].values
            if getattr(self, "_flip_lat", False):
                z_arr = np.flip(z_arr, axis=-2)
            z_tensor = _clean(torch.from_numpy(z_arr.copy()).float())
        except Exception as e:
            msg = (
                f"Failed to load invariant variable 'z' from {z_path}: {e}. "
                "If this error mentions 'NetCDF: HDF error' or 'Errno -101', "
                "ensure HDF5_USE_FILE_LOCKING=FALSE is set in your environment "
                "to disable advisory file locking on parallel filesystems."
            )
            raise StaticVarLoadError(msg) from e

        lsm_path = lsm_files[0]
        try:
            with xr.open_dataset(lsm_path, engine="netcdf4") as ds_lsm:
                lsm_arr = ds_lsm["LSM"].values
            if getattr(self, "_flip_lat", False):
                lsm_arr = np.flip(lsm_arr, axis=-2)
            lsm_tensor = _clean(torch.from_numpy(lsm_arr.copy()).float())
        except Exception as e:
            msg = (
                f"Failed to load invariant variable 'lsm' from {lsm_path}: {e}. "
                "If this error mentions 'NetCDF: HDF error' or 'Errno -101', "
                "ensure HDF5_USE_FILE_LOCKING=FALSE is set in your environment "
                "to disable advisory file locking on parallel filesystems."
            )
            raise StaticVarLoadError(msg) from e

        try:
            with xr.open_dataset(self.slt_path, engine="netcdf4") as ds_slt:
                slt_arr = ds_slt["slt"].values
                lat_k = (
                    "lat"
                    if "lat" in ds_slt.coords
                    else ("latitude" if "latitude" in ds_slt.coords else None)
                )
                if lat_k is not None:
                    slt_lat = ds_slt[lat_k].values
                    if len(slt_lat) > 1 and slt_lat[0] < slt_lat[-1]:
                        slt_arr = np.flip(slt_arr, axis=-2)
            slt_tensor = _clean(torch.from_numpy(slt_arr.copy()).float())
            slt_tensor = _ensure_2d(slt_tensor)
        except Exception as e:
            msg = (
                f"Failed to load static soil type 'slt' from {self.slt_path}: {e}. "
                "Note that 'slt_data.nc' is NOT in the LANL archive; it lives on "
                "purgeable /pscratch (default /pscratch/sd/k/kam352/Aurora/slt/slt_data.nc). "
                "scripts/download_slt.py is the only record of how it was produced. "
                "If this error mentions 'NetCDF: HDF error' or 'Errno -101', "
                "ensure HDF5_USE_FILE_LOCKING=FALSE is set in your environment "
                "to disable advisory file locking on parallel filesystems."
            )
            raise StaticVarLoadError(msg) from e

        return {
            "z": _ensure_2d(z_tensor),
            "lsm": _ensure_2d(lsm_tensor),
            "slt": slt_tensor,
        }

    # ------------------------------------------------------------------
    # Per-item read (forked workers — must stay fork-safe: open/close fresh)
    # ------------------------------------------------------------------

    def _read_var_at_times(
        self, aurora_name: str, files, native_name: str, ts_list: list[int]
    ) -> np.ndarray:
        """Read one variable at explicit timestamps via ITS OWN (fi, li) map."""
        ts_map = self._var_ts_map[aurora_name]
        arrays = []
        for s in ts_list:
            fi, li = ts_map[s]  # guaranteed present: timeline ⊆ every var's map
            with xr.open_dataset(str(files[fi]), engine="netcdf4") as ds:
                arr = ds[native_name].isel(time=li).values
            if self._flip_lat:
                arr = np.flip(arr, axis=-2)
            arrays.append(arr)
        return np.stack(arrays, axis=0).astype(np.float32)

    def _read_sst_at_time(self, init_time: datetime) -> torch.Tensor:
        """Fork-safe read of daily SST at t0.

        Opens, reads, and closes the SST NetCDF file fresh per sample.
        Extracts the daily slice corresponding to init_time (00:00 UTC).
        Flips along latitude if ascending to align with strictly decreasing
        Aurora grid coordinates.
        """
        year = init_time.year
        sst_file = self.sst_dir / f"sst_1deg_{year}.nc"
        if len(self.lat) != 180 or not sst_file.exists():
            # Fallback for synthetic test fixtures
            syn_file = self.sst_dir / "sst_synthetic.nc"
            if syn_file.exists():
                sst_file = syn_file
            elif _DEFAULT_SYNTHETIC_SST.exists():
                sst_file = _DEFAULT_SYNTHETIC_SST
            else:
                msg = (
                    f"Failed to load static SST for year {year}: file not found at {sst_file}. "
                    "Ensure scripts/fetch_sst.py has downloaded the required SST files into data/static/sst/."
                )
                raise StaticVarLoadError(msg)

        try:
            with xr.open_dataset(str(sst_file), engine="netcdf4") as ds_sst:
                day_idx = init_time.timetuple().tm_yday - 1
                n_days = len(ds_sst.time)
                day_idx = min(max(0, day_idx), n_days - 1)
                raw = ds_sst["sst"].isel(time=day_idx).values

                lat_k = (
                    "lat"
                    if "lat" in ds_sst.coords
                    else ("latitude" if "latitude" in ds_sst.coords else None)
                )
                if lat_k is not None:
                    sst_lat = ds_sst[lat_k].values
                    if len(sst_lat) > 1 and sst_lat[0] < sst_lat[-1]:
                        raw = np.flip(raw, axis=-2)
                elif self._flip_lat:
                    raw = np.flip(raw, axis=-2)

            tensor = torch.nan_to_num(
                torch.from_numpy(raw.copy()).float(),
                nan=285.5980,
                posinf=285.5980,
                neginf=285.5980,
            )
            while tensor.ndim > 2:
                tensor = tensor.squeeze(0)
            return tensor
        except Exception as e:
            if isinstance(e, StaticVarLoadError):
                raise
            msg = (
                f"Failed to load static SST from {sst_file} at {init_time}: {e}. "
                "Ensure HDF5_USE_FILE_LOCKING=FALSE is set in your environment "
                "to disable advisory file locking on parallel filesystems."
            )
            raise StaticVarLoadError(msg) from e

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        i0 = int(self._valid_starts[idx])
        step_s = STEP_HOURS * 3600
        t_hist = int(self._timeline[i0])  # t - 6h
        t_curr = t_hist + step_s  # t
        input_ts = [t_hist, t_curr]

        def process_var(arr):
            # Zero (not clamp-to-3.4e38) BOTH NaN and Inf — the default
            # nan_to_num turns +Inf into the largest finite float, which is a
            # ready-made overflow bomb one matmul later.
            return torch.nan_to_num(
                torch.from_numpy(arr), nan=0.0, posinf=0.0, neginf=0.0
            )

        # --- Surface inputs ---
        surf_in = {}
        for aurora_name, (files, native_name) in self.surf_file_map.items():
            raw = self._read_var_at_times(aurora_name, files, native_name, input_ts)
            surf_in[aurora_name] = process_var(raw)[None]  # (1, 2, H, W)

        # --- Atmospheric inputs ---
        atmos_in = {}
        for aurora_name, (files, native_name) in self.atmos_file_map.items():
            raw = self._read_var_at_times(aurora_name, files, native_name, input_ts)
            if self._plev_indices:
                raw = raw[:, self._plev_indices, :, :]
            atmos_in[aurora_name] = process_var(raw)[None]  # (1, 2, 13, H, W)

        # --- Multi-step targets ---
        surf_targets_list, atmos_targets_list = [], []
        for step in range(self.max_rollout_steps):
            tgt_ts = [t_curr + (step + 1) * step_s]
            surf_out = {}
            for aurora_name, (files, native_name) in self.surf_file_map.items():
                raw = self._read_var_at_times(aurora_name, files, native_name, tgt_ts)
                surf_out[aurora_name] = process_var(raw)  # (1, H, W)
            surf_targets_list.append(surf_out)

            atmos_out = {}
            for aurora_name, (files, native_name) in self.atmos_file_map.items():
                raw = self._read_var_at_times(aurora_name, files, native_name, tgt_ts)
                if self._plev_indices:
                    raw = raw[:, self._plev_indices, :, :]
                atmos_out[aurora_name] = process_var(raw)  # (1, 13, H, W)
            atmos_targets_list.append(atmos_out)

        # --- Time tag straight from the timeline (no extra file open) ---
        init_time = datetime.utcfromtimestamp(t_curr)

        # --- Static variables (invariants + time-varying SST at t0) ---
        static_vars = dict(self.static_vars)
        static_vars["sst"] = self._read_sst_at_time(init_time)

        in_batch = Batch(
            surf_vars=surf_in,
            atmos_vars=atmos_in,
            static_vars=static_vars,
            metadata=Metadata(
                lat=self.lat,
                lon=self.lon,
                time=(init_time,),
                atmos_levels=self.atmos_levels,
                rollout_step=0,
            ),
        )
        return in_batch, surf_targets_list, atmos_targets_list


def collate_fn(batch_list):
    return batch_list[0]
