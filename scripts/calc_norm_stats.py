#!/usr/bin/env python3
"""
calc_norm_stats.py

Calculates normalization statistics (mean and standard deviation) for injected 
non-native Aurora variables (e.g., ttr, tcwv) over the specified training split.
Outputs the results in YAML format so they can be easily pasted into the config.
"""

import argparse
import yaml
from pathlib import Path
import xarray as xr
import numpy as np

# Map aurora variable names to their explicit paths and internal NetCDF variable names.
# This should match what we have in `src/dataset.py`.
VARIABLES_TO_CALC = {
    'ttr':  ('Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX', 'mtnlwrf'),
    'tcwv': ('Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv', 'tcwv'),
}

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def calculate_stats_for_var(var_name, subdir, native_name, root_dir, train_years):
    var_dir = root_dir / subdir
    
    files = []
    for year in range(train_years[0], train_years[1] + 1):
        year_dir = var_dir / str(year)
        matched = sorted(year_dir.glob("*.nc"))
        if not matched:
            matched = sorted(var_dir.glob(f"*{year}*.nc"))
        files.extend(matched)
        
    if not files:
        print(f"  [!] No files found for {var_name} in years {train_years}.")
        return None
        
    print(f"  Found {len(files)} files. Computing stats (this may take a minute)...")
    
    # We iterate file-by-file to prevent out-of-core memory issues
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for f in files:
        try:
            ds = xr.open_dataset(f, engine='netcdf4')
            if native_name not in ds:
                print(f"  [!] Native variable {native_name} not found in {f}.")
                ds.close()
                continue
                
            arr = ds[native_name].values
            
            # Avoid NaNs if they exist
            mask = ~np.isnan(arr)
            valid_arr = arr[mask]
            
            count = valid_arr.size
            if count == 0:
                ds.close()
                continue
                
            total_sum += np.sum(valid_arr, dtype=np.float64)
            total_sq_sum += np.sum(valid_arr ** 2, dtype=np.float64)
            total_count += count
            
            ds.close()
        except Exception as e:
            print(f"  [!] Error processing {f}: {e}")
            
    if total_count == 0:
        return None
        
    mean = total_sum / total_count
    variance = (total_sq_sum / total_count) - (mean ** 2)
    std = np.sqrt(variance)
    
    return float(mean), float(std)

def main():
    parser = argparse.ArgumentParser(description="Calculate normalization stats for injected variables.")
    parser.add_argument("--config", required=True, help="Path to the phase config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    root_str = cfg.get("data", {}).get("root")
    if not root_str:
        print("Error: data.root not found in config.")
        return
        
    root_dir = Path(root_str)
    
    train_years = cfg.get("data", {}).get("real", {}).get("train_years")
    if not train_years or len(train_years) != 2:
        print("Error: data.real.train_years missing or invalid in config.")
        return
        
    print(f"Dataset root: {root_dir}")
    print(f"Training split: {train_years[0]} to {train_years[1]}\n")
    
    results = {}
    
    for aurora_var, (subdir, native_name) in VARIABLES_TO_CALC.items():
        print(f"Processing '{aurora_var}' (native: {native_name})")
        stats = calculate_stats_for_var(aurora_var, subdir, native_name, root_dir, train_years)
        if stats:
            mean, std = stats
            print(f"  => Mean: {mean:.4f}, Std: {std:.4f}\n")
            results[aurora_var] = {"mean": round(mean, 4), "std": round(std, 4)}
            
    print("-" * 40)
    print("YAML OUTPUT FOR CONFIG:")
    print("-" * 40)
    print("norm_stats:")
    for var, stats in results.items():
        print(f"  {var}:")
        print(f"    mean: {stats['mean']}")
        print(f"    std: {stats['std']}")

if __name__ == "__main__":
    main()
