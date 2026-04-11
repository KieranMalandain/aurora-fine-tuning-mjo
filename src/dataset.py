# src/dataset.py
import torch
import torch.nn.functional as F
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from aurora import Batch, Metadata

# --- LANL DATA CONFIGURATION ---
# Base directory on NERSC Perlmutter
LANL_DIR = Path("/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results")

# Required Aurora Pressure Levels (hPa)
AURORA_PLEVS =[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Mapping: {Aurora_Name: (LANL_Step_Folder, LANL_Variable_Name)}
SURFACE_VAR_MAP = {
    '2t':   ('Step02/ERA5.remap_180x360MODIS_6hrInst/T2', 't2*'), # Note: xarray will load actual var name, e.g., t2m
    '10u':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/U10', 'u10*'),
    '10v':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/V10', 'v10*'),
    'msl':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/PS', 'Ps'),  # Using Surface Pressure as MSL proxy
    'ttr':  ('Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX', 'mtnlwrf'), # OLR in W/m^2
    'tcwv': ('Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv', 'tcwv'),
    # 'evap': ('Step02/ERA5.remap_180x360MODIS_6hrInst/EFLX', 'EFLX'), # For Phase 2 Physics Loss
    # 'precip': ('Step06/ERA5.remap_180x360MODIS_6hrAccu/TP6H', 'tp6h'), # For Phase 2 Physics Loss
}

ATMOS_VAR_MAP = {
    'z': ('Step01/ERA5.remap_180x360MODIS_6hrInst/gopt', 'z*'),
    'q': ('Step01/ERA5.remap_180x360MODIS_6hrInst/sphu', 'q*'),
    't': ('Step01/ERA5.remap_180x360MODIS_6hrInst/tprt', 't*'),
    'u': ('Step01/ERA5.remap_180x360MODIS_6hrInst/uWnd', 'u*'),
    'v': ('Step01/ERA5.remap_180x360MODIS_6hrInst/vWnd', 'v*'),
}

class LANLMJODataset(Dataset):
    def __init__(self, start_year, end_year, root_dir=LANL_DIR):
        """
        Dataloader engineered for the 1-degree NERSC LANL dataset.
        Handles on-the-fly upsampling to Aurora's native 0.25-degree resolution.
        """
        self.root_dir = Path(root_dir)
        self.start_year = start_year
        self.end_year = end_year
        
        print(f"Initializing LANL MJO Dataset ({start_year}-{end_year})...")
        
        # 1. Load Static Variables
        self.static_vars = self._load_static_vars()
        
        # 2. Lazy Load Dynamic Datasets
        # In a full production script, we would glob the files for the specific years.
        # For simplicity here, we assume a unified loader function exists.
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

    def _build_virtual_dataset(self, var_map):
        """
        NOTE FOR AGENT: This is a placeholder for the actual xarray merging logic.
        The agent should expand this to glob files from the respective StepXX directories
        based on `self.start_year` and `self.end_year`, and xr.merge() them.
        """
        # Placeholder return for structural purposes
        return xr.Dataset() 

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

        # Build Aurora Dictionaries
        # Note for Agent: Extract actual variable names dynamically based on the dataset keys
        surf_in = {aurora_k: process_var(surf_data[lanl_k.replace('*','')].values)[None] for aurora_k, (folder, lanl_k) in SURFACE_VAR_MAP.items() if lanl_k.replace('*','') in surf_data}
        atmos_in = {aurora_k: process_var(pres_data[lanl_k.replace('*','')].values)[None] for aurora_k, (folder, lanl_k) in ATMOS_VAR_MAP.items() if lanl_k.replace('*','') in pres_data}
        
        surf_out = {aurora_k: process_var(surf_target[lanl_k.replace('*','')].values) for aurora_k, (folder, lanl_k) in SURFACE_VAR_MAP.items() if lanl_k.replace('*','') in surf_target}
        atmos_out = {aurora_k: process_var(pres_target[lanl_k.replace('*','')].values) for aurora_k, (folder, lanl_k) in ATMOS_VAR_MAP.items() if lanl_k.replace('*','') in pres_target}

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