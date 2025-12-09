import os
import zipfile
import xarray as xr
from pathlib import Path
import shutil

# Path to your data
DATA_DIR = Path("/gpfs/gibbs/project/lu_lu/kam352/era5_jan2015_daily")

def repair_file(filepath):
    print(f"Checking {filepath.name}...")
    
    # 1. Check if it's actually a zip file
    if not zipfile.is_zipfile(filepath):
        print(f"  -> Valid NetCDF (not a zip). Skipping.")
        return

    print(f"  -> It is a zip archive! Repairing...")
    
    # 2. Rename .nc to .zip to avoid confusion
    zip_path = filepath.with_suffix(".zip")
    os.rename(filepath, zip_path)
    
    # 3. Create a temp directory to extract
    temp_dir = DATA_DIR / f"temp_{filepath.stem}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 4. Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            extracted_files = [temp_dir / f for f in zip_ref.namelist()]
        
        # 5. Merge the extracted files
        # We use xarray to merge the 'instant' and 'accum' files into one
        datasets = [xr.open_dataset(f) for f in extracted_files]
        merged_ds = xr.merge(datasets)
        
        # 6. Save the merged file back to the original name
        # We use netcdf4 engine explicitly here
        merged_ds.to_netcdf(filepath, engine='netcdf4')
        
        # Close datasets
        for ds in datasets: ds.close()
        merged_ds.close()
        
        print(f"  -> Successfully repaired and saved to {filepath.name}")
        
        # 7. Cleanup
        os.remove(zip_path) # Remove the zip
        shutil.rmtree(temp_dir) # Remove temp folder
        
    except Exception as e:
        print(f"  -> FAILED to repair {filepath.name}: {e}")
        # Restore original if failed
        if zip_path.exists() and not filepath.exists():
            os.rename(zip_path, filepath)

def main():
    # Check all surface and pressure files
    files = sorted(list(DATA_DIR.glob("*.nc")))
    print(f"Found {len(files)} files to check.")
    
    for f in files:
        repair_file(f)

if __name__ == "__main__":
    main()