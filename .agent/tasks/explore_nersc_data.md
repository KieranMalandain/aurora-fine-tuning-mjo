# Task: Explore NERSC NetCDF Data Structure

## Objective

Create an exploratory Python script to inspect the internal structure, dimensions, and variables of the LANL array NetCDF files on the NERSC cluster. This is to ensure that our future implementation of `src/dataset.py` will accurately map data variables and axes.

## Working rules
- Read only the files listed below unless you discover a direct dependency.
- If you need to expand scope, stop and report why.
- Make the smallest possible coherent patch.
- Keep all changes confined to the modify list unless explicitly authorized.
- Ensure the command `module load python` is run, and the conda/mamba environment `aurora_mjo` is activated before running Python code (directory `/pscratch/sd/k/kam352/conda/envs/aurora_mjo`). You could do this, for example, by `module load python && mamba run -n aurora_mjo python your_script.py`.

## Read

- `src/dataset.py` (to loosely understand what `dataset.py` expects in terms of names, but purely for context)

## Modify

- `scripts/explore_nersc_data.py` (New file)

## Do not touch

- `src/dataset.py` or any training logic. This task is strictly for creating an analysis script and reporting findings.

## Expected output

- A standalone python script `scripts/explore_nersc_data.py`
- Executing the script will output to terminal the dimensions, coordinates, and exact variable names to answer key data engineering questions (e.g., `time` vs `valid_time`, `lev` vs `level`).
- A brief report or execution of the script showing the printouts.

## Steps

1. Create a new script `scripts/explore_nersc_data.py`.
2. Define the base path as `/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results`.
3. Use `pathlib.Path` to search the following paths for exactly **one** `.nc` file (e.g. using `next(path.glob('...'))`):
   - Surface variable: `Step02/ERA5.remap_180x360MODIS_6hrInst/T2/*.nc`
   - Atmos variable: `Step01/ERA5.remap_180x360MODIS_6hrInst/uWnd/*.nc`
   - Static/Invariant variable: `Step00/ERA5.invariant/*z*.nc` 
4. For each file encountered, use `xarray.open_dataset(file_path, engine='netcdf4')` to load the dataset cleanly and print the following information securely to the console:
   - The verified absolute filename.
   - The dimensions: `ds.dims`
   - The coordinate keys: `list(ds.coords.keys())`
   - The data variable keys: `list(ds.data_vars.keys())`
   - Furthermore, extract arbitrary data variable elements `var_name = list(ds.data_vars.keys())[0]` and print out its `dtype` to ensure we recognize `float64` vs `float32` loading requirements, and `ds[var_name].attrs` to inspect for physical scaling factors like missing values or units.
5. Provide the exact command to run the script.

## Risks & Reminders

- **Memory/Operation Constraints**: Do not attempt to merge or build the entire dataset. This is simply reading metadata from 3 specific files.
- **Dependency context**: The LANL data assumes `Ps` is used instead of Mean Sea Level Pressure (`msl`) and lacks `slt`. Keep this in mind when reading the metadata output.
