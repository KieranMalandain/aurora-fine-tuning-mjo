# src/dummy_dataset.py
# this is used to run on our local testing batch
# which consists of one month of ERA5 data
# from 2015-01-01 to 2015-01-31
# real data is stored on LANL NERSC HPC, use src/dataset.py instead for that
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from pathlib import Path
from aurora import Batch, Metadata

# variable mappings
# note: add in evap and precip later for pi-training

SURFACE_VAR_MAP = {
    't2': '2t',          # LANL name -> Aurora name
    'u10': '10u',
    'v10': '10v',
    'Ps': 'msl',         # using surface pressure for MSL (confirm w group)
    'mtnlwrf': 'ttr',    # LANL name for OLR
    'tcwv': 'tcwv',
    # 'EFLX': 'evap',    # Evaporation
    # 'tp6h': 'precip',  # Total Precipitation
}

ATMOS_VAR_MAP = {
    'z': 'z',
    'u': 'u',
    'v': 'v',
    't': 't',
    'q': 'q',
}

def load_and_combine_files(file_list, engine='netcdf4'):
    """
    Loads a list of NetCDF files and handles common issues.
    """
    print(f"Lazy loading {len(file_list)} files using engine='{engine}'...")

    try:
        ds = xr.open_mfdataset(
            file_list,
            engine=engine,
            combine='by_coords',
            parallel=False,
            chunks={'valid_time': 1}
        )
        if 'valid_time' in ds.dims:
            ds = ds.rename({'valid_time': 'time'})
        
        ds = ds.sortby('time')

        if 'latitude' in ds.dims and ds.dims['latitude'] == 721:
            print("    -> Trimming latitude from 721 to 720 points")
            ds = ds.isel(latittude=slice(0,720))

        return ds
    except Exception as e:
        print(f"CRITICAL ERROR opening dataset: {e}")
        raise e

class MJODataset(Dataset):
    def __init__(self, surface_ds, pressure_ds, static_file_path):
        """
        Args:
            surface_ds (xr.Dataset): Time-sliced surface data
            pressure_ds (xr.Dataset): Time-sliced atmospheric data
            static_file_path (str or Path): Path to static.nc data file
        """
        self.surface_ds = surface_ds
        self.pressure_ds = pressure_ds
        self.static_vars = self._load_static_vars(static_file_path)

        self.num_samples = len(self.surface_ds['time']) - 2
        self.lat = torch.from_numpy(surface_ds['latitude'].values)
        self.lon = torch.from_numpy(surface_ds['longitude'].values)

        if 'pressure_level' in pressure_ds.coords:
            p_levels = pressure_ds['pressure_level'].values
        elif 'isobaricInhPa' in pressure_ds.coords:
            p_levels = pressure_ds['isobaricInhPa'].values
        else:
            raise ValueError("Could not find pressure level coordinate in the dataset.")
        
        self.atmos_levels = tuple(int(l) for l in p_levels)
    
    def _load_static_vars(self, path):
        """Helper to load static variables."""
        ds = xr.open_dataset(path, engine='netcdf4')

        if 'latitude' in ds.dims and ds.dims['latitude'] == 721:
            ds = ds.isel(latitude=slice(0,720))
        
        if 'time' in ds.dims:
            z = ds['z'].isel(time=0).values
            lsm = ds['lsm'].isel(time=0).values
            slt = ds['slt'].isel(time=0).values
        else:
            z = ds['z'].values
            lsm = ds['lsm'].values
            slt = ds['slt'].values
        
        return {
            'z': torch.nan_to_num(torch.from_numpy(z).float().squeeze()),
            'lsm': torch.nan_to_num(torch.from_numpy(lsm).float().squeeze()),
            'slt': torch.nan_to_num(torch.from_numpy(slt).float().squeeze()),
        }
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input_slice = slice(idx, idx+2)
        target_slice = slice(idx+2, idx+3)

        surf_data = self.surface_ds.isel(time=input_slice).load()
        surf_target = self.surface_ds.isel(time=target_slice).load()
        pres_data = self.pressure_ds.isel(time=input_slice).load()
        pres_target = self.pressure_ds.isel(time=target_slice).load()

        def clean(arr):
            return torch.nan_to_num(torch.from_numpy(arr.astype(np.float32)))
        
        def process_var(name, arr):
            tensor = torch.nan_to_num(torch.from_numpy(arr.astype(np.float32)))
            # Add necessary batch and channel dims for interpolation: (1, 1, 180, 360)
            tensor = tensor.unsqueeze(0).unsqueeze(0) 
            # UPSAMPLE TO 0.25 DEGREE (180x360 -> 720x1440)
            tensor = F.interpolate(tensor, size=(720, 1440), mode='bilinear', align_corners=False)
            # Remove the dummy channel dim, keep batch dim: (1, 720, 1440)
            tensor = tensor.squeeze(1)
            # unit convert
            if name == 'mtnlwrf':
                return tensor / 3600.
            return tensor
        
        
        surf_tensors = {}
        for era_name, aurora_name in SURFACE_VAR_MAP.items():
            if era_name in surf_data:
                surf_tensors[aurora_name] = process_var(
                    era_name, surf_data[era_name].values
                )[None]
        
        atmos_tensors = {}
        for era_name, aurora_name in ATMOS_VAR_MAP.items():
            if era_name in pres_data:
                atmos_tensors[aurora_name] = process_var(
                    era_name, pres_data[era_name].values
                )[None]

        in_batch = Batch(
            surf_vars = surf_tensors,
            atmos_vars = atmos_tensors,
            static_vars = self.static_vars,
            metadata = Metadata(
                lat = self.lat,
                lon = self.lon,
                time = (surf_data.time.values.astype('datetime64[s]').tolist()[1],),
                atmos_levels = self.atmos_levels,
                rollout_step = 0
            )
        )        

        target_dict = {}
        for era_name, aurora_name in SURFACE_VAR_MAP.items():
            if era_name in surf_target:
                target_dict[aurora_name] = process_var(
                    era_name, surf_target[era_name].values
                )
        for era_name, aurora_name in ATMOS_VAR_MAP.items():
            if era_name in pres_target:
                target_dict[aurora_name] = process_var(
                    era_name, pres_target[era_name].values
                )
        
        return in_batch, target_dict
    
    def collate_fn(batch_list):
        """
        Collates a list of (batch, target) tuples. We take the first element assuming batch_size=1
        """
        return batch_list[0]