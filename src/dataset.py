# src/dataset.py
#
# NERSC/LANL ERA5 dataset for Aurora MJO fine-tuning.
# All path roots are passed via constructor args or config — nothing is hardcoded here.
# Use src/dummy_dataset.py for Bouchet smoke tests.

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
AURORA_PLEVS =[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Variable maps: {aurora_name: (step_subdir, glob_pattern)}
#
# Each entry identifies:
#   - the StepXX subdirectory under root_dir that holds the files
#   - the glob pattern used to find per-year .nc files in that directory
#
# NOTE — msl proxy: The LANL dataset does not include mean sea-level pressure.
# 'Ps' (surface pressure) is used as a stand-in for 'msl'.
# TO REPLACE WITH TRUE MSL: change 'Ps' below to the actual MSL glob pattern
# once those files are available. No other code change should be required.
SURFACE_VAR_MAP = {
    '2t':   ('Step02/ERA5.remap_180x360MODIS_6hrInst/T2',          't2*'),
    '10u':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/U10',         'u10*'),
    '10v':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/V10',         'v10*'),
    'msl':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/PS',          'Ps*'),   # MSL_PROXY: swap 'Ps*' → real msl glob when available
    'ttr':  ('Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX', 'mtnlwrf*'),  # OLR in W/m²
    'tcwv': ('Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv',        'tcwv*'),
    # Phase-2 physics-loss variables (do not enable until baseline is stable):
    # 'evap':   ('Step02/ERA5.remap_180x360MODIS_6hrInst/EFLX',  'EFLX*'),
    # 'precip': ('Step06/ERA5.remap_180x360MODIS_6hrAccu/TP6H',  'tp6h*'),
}

ATMOS_VAR_MAP = {
    'z': ('Step01/ERA5.remap_180x360MODIS_6hrInst/gopt', 'z*'),
    'q': ('Step01/ERA5.remap_180x360MODIS_6hrInst/sphu', 'q*'),
    't': ('Step01/ERA5.remap_180x360MODIS_6hrInst/tprt', 't*'),
    'u': ('Step01/ERA5.remap_180x360MODIS_6hrInst/uWnd', 'u*'),
    'v': ('Step01/ERA5.remap_180x360MODIS_6hrInst/vWnd', 'v*'),
}

class LANLMJODataset(Dataset):
    def __init__(self, start_year: int, end_year: int, root_dir: str | Path | None = None):
        """
        Dataset for the 1-degree NERSC/LANL ERA5 ERA5 preprocessing output.
        Handles on-the-fly upsampling to Aurora's native 0.25-degree resolution.

        Args:
            start_year: First year (inclusive) to include in the dataset.
            end_year:   Last year (inclusive) to include in the dataset.
            root_dir:   Filesystem root of the LANL Results/ directory.
                        Pass via config['data']['root'] or the --data-root CLI flag.
                        Falls back to _DEFAULT_NERSC_ROOT with a warning if omitted.
        """
        if root_dir is None:
            warnings.warn(
                f"root_dir not provided; falling back to default NERSC path: {_DEFAULT_NERSC_ROOT}. "
                "Pass root_dir explicitly or set config['data']['root'] to suppress this warning.",
                stacklevel=2,
            )
            root_dir = _DEFAULT_NERSC_ROOT
        self.root_dir = Path(root_dir)
        self.start_year = start_year
        self.end_year = end_year
        
        print(f"Initializing LANL MJO Dataset ({start_year}-{end_year})...")
        
        # 1. Load Static Variables
        self.static_vars = self._load_static_vars()
        
        # 2. Lazy-load dynamic datasets by globbing files per variable and year.
        self.surface_ds = self._build_virtual_dataset(SURFACE_VAR_MAP)
        self.pressure_ds = self._build_virtual_dataset(ATMOS_VAR_MAP)
        
        # 3. Filter Pressure Levels to the exact 13 Aurora requires
        # LANL data has 29 levels. We must slice them.
        plev_coord = 'isobaricInhPa' if 'isobaricInhPa' in self.pressure_ds.coords else 'level'
        self.pressure_ds = self.pressure_ds.sel({plev_coord: AURORA_PLEVS})
        self.atmos_levels = tuple(AURORA_PLEVS)
        
        self.num_samples = len(self.surface_ds['time']) - 2

        # Aurora requires latitude/longitude in 0.25 deg format
        # Since we upsample the data, we must provide the UPSAMPLED coordinates to the Metadata
        self.lat = torch.linspace(90, -90, 720)
        self.lon = torch.linspace(0, 360, 1441)[:-1]

    def _load_static_vars(self):
        """Loads Z and LSM, and creates a dummy SLT (Soil Type) to prevent Aurora crashes."""
        # Note: Actual file paths will need to be globbed based on the Step00 directory
        static_dir = self.root_dir / "Step00/ERA5.invariant"
        
        # NOTE FOR AGENT: Implement actual globbing here for the specific static files.
        # For now, we simulate the logic.
        try:
            z_file = list(static_dir.glob("*_z.*.nc"))[0]
            lsm_file = list(static_dir.glob("*_lsm.*.nc"))[0]
            z_arr = xr.open_dataset(z_file)['z'].values
            lsm_arr = xr.open_dataset(lsm_file)['lsm'].values
        except IndexError:
            # Fallback for testing if files aren't found immediately
            z_arr = np.zeros((180, 360))
            lsm_arr = np.zeros((180, 360))
            
        z_tensor = self._upsample_to_aurora(torch.from_numpy(z_arr).float())
        lsm_tensor = self._upsample_to_aurora(torch.from_numpy(lsm_arr).float())
        
        # Soil Type (slt) is missing from the LANL PDF. 
        # We create a dummy tensor of zeros so Aurora doesn't crash.
        slt_tensor = torch.zeros_like(z_tensor)
        
        return {"z": z_tensor, "lsm": lsm_tensor, "slt": slt_tensor}

    def _build_virtual_dataset(self, var_map: dict) -> xr.Dataset:
        """
        Glob files for every variable in `var_map` across the requested year
        range and merge them into a single xr.Dataset aligned on the 'time'
        dimension.

        Directory layout assumed under self.root_dir:
            <root_dir>/<step_subdir>/<year>/<glob_pattern>.nc
        e.g.:
            Results/Step02/ERA5.remap_180x360MODIS_6hrInst/T2/2000/t2_2000*.nc

        If a variable's files are absent on this host the variable is silently
        skipped with a warning so that other variables can still load.

        Returns:
            xr.Dataset with one data variable per aurora variable name, merged
            along the 'time' axis and sliced to [start_year, end_year].
        """
        per_var_datasets: list[xr.Dataset] = []

        for aurora_name, (step_subdir, glob_pattern) in var_map.items():
            var_dir = self.root_dir / step_subdir

            # Collect files year-by-year so we stay within the requested range.
            files: list[Path] = []
            for year in range(self.start_year, self.end_year + 1):
                year_dir = var_dir / str(year)
                matched = sorted(year_dir.glob(glob_pattern + ".nc"))
                if not matched:
                    # Some datasets store files directly in the var dir without
                    # a year sub-directory — try that layout as a fallback.
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

            # Open all files for this variable as a single time-concatenated dataset.
            ds_var = xr.open_mfdataset(
                [str(f) for f in files],
                combine="by_coords",
                concat_dim="time",
                engine="netcdf4",
                parallel=True,    # dask-parallel open; actual I/O stays lazy
            )

            # Rename the netCDF variable (whatever LANL called it) to the
            # canonical Aurora name so __getitem__ can use a uniform key.
            # We take whichever data variable appears first — each subdirectory
            # is expected to hold exactly one variable.
            native_name = next(iter(ds_var.data_vars))
            if native_name != aurora_name:
                ds_var = ds_var.rename({native_name: aurora_name})

            per_var_datasets.append(ds_var[[aurora_name]])

        if not per_var_datasets:
            raise RuntimeError(
                f"[LANLMJODataset] _build_virtual_dataset found no files at all under "
                f"{self.root_dir} for years {self.start_year}–{self.end_year}. "
                "Check root_dir and the LANL directory layout."
            )

        # Merge all per-variable datasets on their shared time/lat/lon grid.
        merged = xr.merge(per_var_datasets, join="inner")
        return merged

    def _upsample_to_aurora(self, tensor):
        """
        CRITICAL: Upsamples 1-degree LANL data (180x360) to 0.25-degree Aurora data (720x1440).
        Tensor shape expected: (Lat, Lon) or (Time, Levels, Lat, Lon)
        """
        original_shape = tensor.shape
        
        # Ensure tensor has exactly 4 dimensions (Batch, Channel, Lat, Lon) for interpolation
        if len(original_shape) == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0) # (1, 1, Lat, Lon)
        elif len(original_shape) == 3:
            tensor = tensor.unsqueeze(0) # (1, C, Lat, Lon)
            
        # Bilinear interpolation up to 720x1440
        upsampled = F.interpolate(tensor, size=(720, 1440), mode='bilinear', align_corners=False)
        
        # Strip the dummy dimensions back to the expected output shape
        if len(original_shape) == 2:
            return upsampled.squeeze(0).squeeze(0)
        elif len(original_shape) == 3:
            return upsampled.squeeze(0)
        return upsampled

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_slice = slice(idx, idx + 2)
        target_slice = slice(idx + 2, idx + 3)
        
        # Lazy Load from Disk
        surf_data = self.surface_ds.isel(time=input_slice).load()
        surf_target = self.surface_ds.isel(time=target_slice).load()
        pres_data = self.pressure_ds.isel(time=input_slice).load()
        pres_target = self.pressure_ds.isel(time=target_slice).load()

        def process_var(arr):
            # Clean NaNs and convert to tensor
            tensor = torch.nan_to_num(torch.from_numpy(arr.astype(np.float32)))
            # Upsample 1-degree to 0.25-degree
            return self._upsample_to_aurora(tensor)

        # Build Aurora dictionaries.
        # Variables are keyed by their canonical Aurora name (set in _build_virtual_dataset
        # via the rename step), so we look them up directly — no brittle string-replace.
        surf_in  = {
            k: process_var(surf_data[k].values)[None]
            for k in SURFACE_VAR_MAP
            if k in surf_data
        }
        atmos_in = {
            k: process_var(pres_data[k].values)[None]
            for k in ATMOS_VAR_MAP
            if k in pres_data
        }
        surf_out  = {
            k: process_var(surf_target[k].values)
            for k in SURFACE_VAR_MAP
            if k in surf_target
        }
        atmos_out = {
            k: process_var(pres_target[k].values)
            for k in ATMOS_VAR_MAP
            if k in pres_target
        }

        # ADDRESSING THE ADDENDUM: Time Tags
        # ClimaX loses time tags. Aurora requires them. We explicitly pass the initialization time here.
        # batch.metadata.time[0] will be the exact initialization time.
        init_time = surf_data.time.values.astype('datetime64[s]').tolist()[1]

        in_batch = Batch(
            surf_vars=surf_in,
            atmos_vars=atmos_in,
            static_vars=self.static_vars,
            metadata=Metadata(
                lat=self.lat,
                lon=self.lon,
                time=(init_time,), # The crucial Initialization Time Tag
                atmos_levels=self.atmos_levels,
                rollout_step=0 
            )
        )

        return in_batch, surf_out, atmos_out

def collate_fn(batch_list):
    return batch_list[0]