# scripts/precompute_stats.py

import xarray as xr
from pathlib import Path
import numpy as np

# Configuration
DATA_DIR = Path("/gpfs/gibbs/project/lu_lu/kam352/era5_jan2015_daily")
# Limit to the training days (Day 1 to 20)
TRAIN_SPLIT_DAY = 21 

def compute_stats():
    print(f"Loading data from {DATA_DIR}...")
    
    # We use 'chunks' here to ensure Dask processes this in parallel/streams it
    # rather than trying to load 100GB into RAM.
    surface_files = sorted(DATA_DIR.glob("surface_*.nc"))
    
    ds = xr.open_mfdataset(
        surface_files, 
        combine='by_coords', 
        engine='netcdf4', 
        parallel=True, 
        chunks={'time': 24} # Chunk by time to keep memory low
    )
    
    # Filter for Training Data only
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    
    train_ds = ds.sel(time=ds.time.dt.day < TRAIN_SPLIT_DAY)
    
    print("Computing stats for TTR (Top Thermal Radiation)...")
    # ERA5 TTR is in Joules (accumulated). Divide by 3600 to get Watts.
    ttr_mean = train_ds['ttr'].mean().compute() / 3600.0
    ttr_std = train_ds['ttr'].std().compute() / 3600.0
    
    print("Computing stats for TCWV (Total Column Water Vapor)...")
    tcwv_mean = train_ds['tcwv'].mean().compute()
    tcwv_std = train_ds['tcwv'].std().compute()
    
    print("-" * 30)
    print("COPY THESE VALUES INTO YOUR TRAINING SCRIPT:")
    print(f"TTR_MEAN = {float(ttr_mean)}")
    print(f"TTR_STD  = {float(ttr_std)}")
    print(f"TCWV_MEAN = {float(tcwv_mean)}")
    print(f"TCWV_STD  = {float(tcwv_std)}")
    print("-" * 30)

if __name__ == "__main__":
    compute_stats()