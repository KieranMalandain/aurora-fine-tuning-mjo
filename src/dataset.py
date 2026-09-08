# src/dataset.py
#
# NERSC/LANL ERA5 dataset for Aurora MJO fine-tuning.
# All path roots are passed via constructor args or config — nothing is hardcoded here.
# Use src/dummy_dataset.py for Bouchet smoke tests.
#
# Fork-safety
# -----------
# PyTorch DataLoader with num_workers > 0 uses os.fork() to create worker
# processes.  xarray Datasets backed by dask or holding open netCDF4 file
# handles are NOT safe to fork — they contain internal locks and file
# descriptors that become corrupted in the child process, causing silent
# deadlocks.
#
# This module avoids the problem by storing ONLY file paths and a pre-built
# global-time-index → (file_index, local_time_index) mapping at __init__
# time.  Each __getitem__ call opens the required NetCDF file(s) from
# scratch, reads the needed slices, and closes them immediately.  This
# makes the Dataset fully fork-safe and compatible with any num_workers
# value.

import warnings
import torch
import torch.nn.functional as F
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from aurora import Batch, Metadata

# DEFAULT_NERSC_ROOT is provided as a fallback hint only.
# Always override via config['data']['root'] or the root_dir constructor arg.
_DEFAULT_NERSC_ROOT = "/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results"

# Required Aurora Pressure Levels (hPa)
AURORA_PLEVS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Variable maps: {aurora_name: (step_subdir, glob_pattern, native_var_name)}
#
# Each entry identifies:
#   - the StepXX subdirectory under root_dir that holds the files
#   - the glob pattern used to find per-year .nc files in that directory
#   - the explicit internal NetCDF variable name for xarray extraction
#
# NOTE — msl proxy: The LANL dataset does not include mean sea-level pressure.
# 'Ps' (surface pressure) is used as a stand-in for 'msl'.
# TO REPLACE WITH TRUE MSL: change the step_subdir and native_var_name to the actual MSL data
# once those files are available. No other code change should be required.
SURFACE_VAR_MAP = {
    '2t':   ('Step02/ERA5.remap_180x360MODIS_6hrInst/T2',          '*', 't2'),
    '10u':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/U10',         '*', 'u10'),
    '10v':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/V10',         '*', 'v10'),
    'msl':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/PS',          '*', 'ps'),   # Temporary proxy: LANL dataset lacks MSL. Swap 'ps' → real msl var when available
    'ttr':  ('Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX', '*', 'mtnlwrf'),  # Compatibility mapping: OLR in W/m² used for ttr
    'tcwv': ('Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv',        '*', 'tcwv'),
    # Phase-2 physics-loss variables (do not enable until baseline is stable):
    # 'evap':   ('Step02/ERA5.remap_180x360MODIS_6hrInst/EFLX',  '*', 'EFLX'),
    # 'precip': ('Step06/ERA5.remap_180x360MODIS_6hrAccu/TP6H',  '*', 'tp6h'),
}

ATMOS_VAR_MAP = {
    'z': ('Step01/ERA5.remap_180x360MODIS_6hrInst/gopt', '*', 'z'),
    'q': ('Step01/ERA5.remap_180x360MODIS_6hrInst/sphu', '*', 'q'),
    't': ('Step01/ERA5.remap_180x360MODIS_6hrInst/tprt', '*', 't'),
    'u': ('Step01/ERA5.remap_180x360MODIS_6hrInst/uWnd', '*', 'u'),
    'v': ('Step01/ERA5.remap_180x360MODIS_6hrInst/vWnd', '*', 'v'),
}


def _upsample_to_aurora(tensor):
    """
    CRITICAL: Upsamples 1-degree LANL data (180x360) to 0.25-degree Aurora data (720x1440).
    Tensor shape expected: (Lat, Lon) or (Time, Levels, Lat, Lon) or (Time, Lat, Lon)
    """
    original_shape = tensor.shape

    # Ensure tensor has exactly 4 dimensions (Batch, Channel, Lat, Lon) for interpolation
    if len(original_shape) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, Lat, Lon)
    elif len(original_shape) == 3:
        tensor = tensor.unsqueeze(0)  # (1, C, Lat, Lon)

    # Bilinear interpolation up to 720x1440
    upsampled = F.interpolate(tensor, size=(720, 1440), mode='bilinear', align_corners=False)

    # Strip the dummy dimensions back to the expected output shape
    if len(original_shape) == 2:
        return upsampled.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        return upsampled.squeeze(0)
    return upsampled


class LANLMJODataset(Dataset):
    """Fork-safe ERA5 dataset for Aurora MJO fine-tuning.

    Stores only file paths and a pre-built index map at init time.
    Each ``__getitem__`` opens the required NetCDF files from scratch,
    making this dataset fully compatible with PyTorch DataLoader using
    any ``num_workers`` value (including > 0).

    Args:
        start_year: First year (inclusive) to include in the dataset.
        end_year:   Last year (inclusive) to include in the dataset.
        root_dir:   Filesystem root of the LANL Results/ directory.
                    Pass via config['data']['root'] or the --data-root CLI flag.
                    Falls back to _DEFAULT_NERSC_ROOT with a warning if omitted.
        slt_path:   Path to the static Soil Type data file.
    """

    def __init__(self, start_year: int, end_year: int,
                 root_dir: str | Path | None = None,
                 slt_path: str | Path | None = None):
        if root_dir is None:
            warnings.warn(
                f"root_dir not provided; falling back to default NERSC path: {_DEFAULT_NERSC_ROOT}. "
                "Pass root_dir explicitly or set config['data']['root'] to suppress this warning.",
                stacklevel=2,
            )
            root_dir = _DEFAULT_NERSC_ROOT
        self.root_dir = Path(root_dir)

        if slt_path is None:
            warnings.warn(
                "slt_path not provided; falling back to default.",
                stacklevel=2,
            )
            slt_path = "/pscratch/sd/k/kam352/Aurora/slt/slt_data.nc"
        self.slt_path = Path(slt_path)

        self.start_year = start_year
        self.end_year = end_year

        print(f"Initializing LANL MJO Dataset ({start_year}-{end_year})...")

        # Aurora requires latitude/longitude in 0.25 deg format
        # Since we upsample the data, we must provide the UPSAMPLED coordinates to the Metadata
        self.lat = torch.linspace(90, -90, 720)
        self.lon = torch.linspace(0, 360, 1441)[:-1]
        self.atmos_levels = tuple(AURORA_PLEVS)

        # 1. Load static variables — these are small tensors, fork-safe once loaded
        self.static_vars = self._load_static_vars()

        # 2. Build per-variable file lists (NO xarray stored on self)
        self.surf_file_map = self._collect_var_files(SURFACE_VAR_MAP)
        self.atmos_file_map = self._collect_var_files(ATMOS_VAR_MAP)

        # 3. Build the global-index → (file_index, local_time_index) mapping.
        #    We use the first surface variable as the reference for time alignment.
        #    All variables share the same 6-hourly time grid.
        ref_var = next(iter(self.surf_file_map))
        ref_files = self.surf_file_map[ref_var][0]
        self._index_map = self._build_index_map(ref_files)
        self.num_samples = len(self._index_map) - 2

        # 4. Determine pressure-level indices for subsetting the 29-level LANL
        #    data to the 13 Aurora levels.  Only needs one file probe.
        self._plev_indices = self._probe_pressure_level_indices()

        print(f"  Total timesteps: {len(self._index_map)} → {self.num_samples} samples "
              f"({len(self.surf_file_map)} surf vars, {len(self.atmos_file_map)} atmos vars)")

    # ------------------------------------------------------------------
    # Init helpers (run once, store only serialisable state)
    # ------------------------------------------------------------------

    def _collect_var_files(self, var_map: dict) -> dict:
        """Glob files for each variable across the year range.

        Returns:
            Dict mapping aurora_name → (list[Path], native_var_name).
            No xarray objects are created — just file paths.
        """
        result = {}
        for aurora_name, (step_subdir, glob_pattern, native_name) in var_map.items():
            var_dir = self.root_dir / step_subdir
            files: list[Path] = []
            for year in range(self.start_year, self.end_year + 1):
                year_dir = var_dir / str(year)
                matched = sorted(year_dir.glob(glob_pattern + ".nc"))
                if not matched:
                    # Fallback: files directly in var_dir without year sub-directory
                    matched = sorted(var_dir.glob(f"*{year}*" + ".nc"))
                files.extend(matched)

            if not files:
                warnings.warn(
                    f"[LANLMJODataset] No files found for aurora variable '{aurora_name}' "
                    f"in {var_dir} for years {self.start_year}–{self.end_year}. "
                    "Skipping this variable.",
                    stacklevel=2,
                )
                continue

            result[aurora_name] = (files, native_name)
        return result

    def _build_index_map(self, files: list[Path]) -> list[tuple[int, int]]:
        """Map global time index → (file_index, local_time_index).

        Probes each file for its time dimension length without loading data.
        """
        mapping: list[tuple[int, int]] = []
        for fi, f in enumerate(files):
            with xr.open_dataset(str(f), engine="netcdf4") as ds:
                n = len(ds.time)
                for li in range(n):
                    mapping.append((fi, li))
        return mapping

    def _probe_pressure_level_indices(self) -> list[int]:
        """Read pressure levels from the first atmospheric file and find the
        indices of the 13 Aurora-required levels."""
        if not self.atmos_file_map:
            return []
        ref_var = next(iter(self.atmos_file_map))
        ref_files = self.atmos_file_map[ref_var][0]
        with xr.open_dataset(str(ref_files[0]), engine="netcdf4") as ds:
            if 'lev' in ds.coords:
                all_levs = ds.lev.values
            elif 'level' in ds.coords:
                all_levs = ds.level.values
            else:
                warnings.warn(
                    "[LANLMJODataset] Could not find pressure level coordinate "
                    "('lev' or 'level') in atmospheric data. Using all levels.",
                    stacklevel=2,
                )
                return list(range(len(next(iter(ds.data_vars.values())).dims)))
        return [int(np.argmin(np.abs(all_levs - p))) for p in AURORA_PLEVS]

    def _load_static_vars(self):
        """Loads Z, LSM, and SLT.  Returns pure tensors (fork-safe)."""
        static_dir = self.root_dir / "Step00/ERA5.invariant"

        z_files = list(static_dir.glob("*_z.*.nc"))
        if not z_files:
            raise FileNotFoundError(f"Invariant Z file not found in {static_dir}")
        lsm_files = list(static_dir.glob("*_lsm.*.nc"))
        if not lsm_files:
            raise FileNotFoundError(f"Invariant LSM file not found in {static_dir}")

        # Helper: Aurora expects static_vars to be exactly 2D (H, W).
        # Its forward pass does v[None, None].repeat(B, T, 1, 1), which
        # requires a 2D starting tensor.  Squeeze away any leading dims.
        def _ensure_2d(t):
            while t.ndim > 2:
                t = t.squeeze(0)
            return t

        try:
            with xr.open_dataset(z_files[0], engine="netcdf4") as ds_z:
                z_arr = ds_z['Z'].values
            z_tensor = _upsample_to_aurora(torch.from_numpy(z_arr).float())
        except Exception as e:
            warnings.warn(f"[LANLMJODataset] Failed to load invariant Z file: {e}. Using dummy zero tensor.")
            z_tensor = torch.zeros(720, 1440)

        try:
            with xr.open_dataset(lsm_files[0], engine="netcdf4") as ds_lsm:
                lsm_arr = ds_lsm['LSM'].values
            lsm_tensor = _upsample_to_aurora(torch.from_numpy(lsm_arr).float())
        except Exception as e:
            warnings.warn(f"[LANLMJODataset] Failed to load invariant LSM file: {e}. Using dummy zero tensor.")
            lsm_tensor = torch.zeros(720, 1440)

        # Load Soil Type (slt) natively at 0.25-degree
        with xr.open_dataset(self.slt_path, engine="netcdf4") as ds_slt:
            slt_arr = ds_slt['slt'].values

        slt_tensor = torch.nan_to_num(torch.from_numpy(slt_arr).float())
        slt_tensor = _ensure_2d(slt_tensor)     # squeeze to (Lat, Lon)
        slt_tensor = slt_tensor[:720, :]         # slice 721 to 720 to match Aurora grid

        return {
            "z":   _ensure_2d(z_tensor),
            "lsm": _ensure_2d(lsm_tensor),
            "slt": slt_tensor,
        }

    # ------------------------------------------------------------------
    # Per-item helpers (called inside forked workers — must be fork-safe)
    # ------------------------------------------------------------------

    def _read_var_at_indices(self, files, native_name, global_indices):
        """Read a variable at specific global time indices.

        Opens/closes files per call — no file handles are cached.
        Returns a numpy array of shape (len(global_indices), ...).
        """
        arrays = []
        for gi in global_indices:
            fi, li = self._index_map[gi]
            with xr.open_dataset(str(files[fi]), engine="netcdf4") as ds:
                arr = ds[native_name].isel(time=li).values
            arrays.append(arr)
        return np.stack(arrays, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_indices = [idx, idx + 1]
        target_indices = [idx + 2]

        def process_var(arr):
            """Clean NaNs, convert to tensor, upsample to Aurora resolution."""
            tensor = torch.nan_to_num(torch.from_numpy(arr))
            return _upsample_to_aurora(tensor)

        # --- Surface variables ---
        surf_in = {}
        surf_out = {}
        for aurora_name, (files, native_name) in self.surf_file_map.items():
            raw_in = self._read_var_at_indices(files, native_name, input_indices)
            surf_in[aurora_name] = process_var(raw_in)[None]   # (1, 2, H, W) with batch dim

            raw_tgt = self._read_var_at_indices(files, native_name, target_indices)
            surf_out[aurora_name] = process_var(raw_tgt)       # (1, H, W)

        # --- Atmospheric variables ---
        atmos_in = {}
        atmos_out = {}
        for aurora_name, (files, native_name) in self.atmos_file_map.items():
            raw_in = self._read_var_at_indices(files, native_name, input_indices)
            # Subset pressure levels: (T, 29, Lat, Lon) → (T, 13, Lat, Lon)
            if self._plev_indices:
                raw_in = raw_in[:, self._plev_indices, :, :]
            atmos_in[aurora_name] = process_var(raw_in)[None]  # (1, 2, 13, H, W)

            raw_tgt = self._read_var_at_indices(files, native_name, target_indices)
            if self._plev_indices:
                raw_tgt = raw_tgt[:, self._plev_indices, :, :]
            atmos_out[aurora_name] = process_var(raw_tgt)      # (1, 13, H, W)

        # --- Time tag ---
        # ADDRESSING THE ADDENDUM: Time Tags
        # ClimaX loses time tags. Aurora requires them. We explicitly pass the initialization time here.
        # batch.metadata.time[0] will be the exact initialization time.
        fi, li = self._index_map[input_indices[1]]
        ref_files = list(self.surf_file_map.values())[0][0]
        with xr.open_dataset(str(ref_files[fi]), engine="netcdf4") as ds:
            init_time = ds.time.values[li].astype('datetime64[s]').tolist()

        in_batch = Batch(
            surf_vars=surf_in,
            atmos_vars=atmos_in,
            static_vars=self.static_vars,
            metadata=Metadata(
                lat=self.lat,
                lon=self.lon,
                time=(init_time,),  # The crucial Initialization Time Tag
                atmos_levels=self.atmos_levels,
                rollout_step=0
            )
        )

        return in_batch, surf_out, atmos_out


def collate_fn(batch_list):
    return batch_list[0]