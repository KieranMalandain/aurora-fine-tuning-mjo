import torch
import xarray as xr
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
import time

class MJODataset(Dataset):
    def __init__(self, surface_ds, pressure_ds):
        self.surface_ds = surface_ds
        self.pressure_ds = pressure_ds
        self.num_samples = len(self.surface_ds['time']) - 2
        self.lat = torch.from_numpy(surface_ds['latitude'].values)
        self.long = torch.from_numpy(surface_ds['longitude'].values)
        self.atmos_levels = tuple(int(l) for l in pressure_ds['pressure_level'].values)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        in_slice = slice(idx, idx + 2)
        surf_data = self.surface_ds.isel(time=in_slice).load()
        return torch.from_numpy(surf_data['t2m'].values).float()
        
def verify():
    DATA_DIR = Path("/gpfs/gibbs/project/lu_lu/kam352/era5_jan2015_daily")
    
    print(f"Checking Data Directory: {DATA_DIR}")
    surface_files = sorted(DATA_DIR.glob("surface_*.nc"))
    if not surface_files:
        print("    FAIL. No surface files.")
        sys.exit(1)
    print(f"    SUCCESS. Found {len(surface_files)} surface files.")
    
    print("Init lazy xr dataset... (with chunks)")
    try:
        # This mirrors your main script exactly
        surface_ds = xr.open_mfdataset(
            surface_files, 
            combine='by_coords', 
            engine='netcdf4', 
            parallel=False, 
            chunks={'valid_time': 1} 
        )
        
        if 'valid_time' in surface_ds.dims:
            print("    Renaming valid time")
            surface_ds = surface_ds.rename({'valid_time': 'time'})
        surface_ds = surface_ds.sortby('time')
        
        pressure_ds = xr.open_mfdataset(
            sorted(DATA_DIR.glob("pressure_*.nc")), 
            combine='by_coords', 
            engine='netcdf4', 
            parallel=False, 
            chunks={'valid_time': 1} 
        )
        
        if 'valid_time' in pressure_ds:
            pressure_ds = pressure_ds.rename({'valid_time':'time'})
        pressure_ds = pressure_ds.sortby('time')
        
        
        print("   Success. Metadata loaded.")
        print(f"   Dataset dimensions: {surface_ds.dims}")
    except Exception as e:
        print(f"   FAIL. Error initializing dataset: {e}")
        try:
            # Open just the first file to peek at it
            temp_ds = xr.open_dataset(surface_files[0], engine='netcdf4')
            print(f"   Variables in file: {list(temp_ds.keys())}")
            print(f"   Coordinates in file: {list(temp_ds.coords)}")
        except:
            pass
        sys.exit(1)

    print(f"4. Testing DataLoader (fetching one real batch)...")
    try:
        ds = MJODataset(surface_ds, pressure_ds)
        loader = DataLoader(ds, batch_size=1, num_workers=0)
        
        start = time.time()
        # Try to get the first item. This triggers the actual file read.
        first_batch = next(iter(loader))
        end = time.time()
        
        print(f"   Success! Loaded one batch in {end-start:.2f} seconds.")
        print(f"   Batch shape: {first_batch.shape}")
        
    except Exception as e:
        print(f"   FAIL. Error fetching batch: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n------------------------------------------------")
    print("VERIFICATION PASSED. You are safe to submit the job.")
    print("------------------------------------------------")

if __name__ == "__main__":
    verify()