# Task: Update NERSC Dataset Adapter

## Objective

Adjust `src/dataset.py` to correctly map the NERSC ERA5 NetCDF layout based on exploratory findings documented in `docs/nersc_data.md`. This is critical before the training loop can properly load and ingest data.

## Working rules
- Read only the files listed below.
- Do not attempt to run training or evaluate the model.
- Make the smallest possible coherent patch.
- Keep all changes confined to the modify list unless explicitly authorized.

## Read

- `docs/nersc_data.md`
- `src/dataset.py`

## Modify

- `src/dataset.py`

## Expected output

- A modified `src/dataset.py` that implements the three actionable changes to handle NERSC data files natively.
- A summary of the modification choice made regarding the discovery of the native variable name in `_build_virtual_dataset()`.

## Steps

1. **Vertical Coordinate Bug Fix**:
   - In `LANLMJODataset.__init__`, locate the pressure level coordinate fallback assignment: `plev_coord = 'isobaricInhPa' if 'isobaricInhPa' in self.pressure_ds.coords else 'level'`.
   - Update this logic to account for NERSC's internal vertical axis name. It must check for `lev` (e.g. `plev_coord = 'lev' if 'lev' in self.pressure_ds.coords else ...`).
2. **Static Geo-Potential Keys**:
   - In `LANLMJODataset._load_static_vars()`, the dataloader currently performs `z_arr = xr.open_dataset(z_file)['z'].values`.
   - Change the hardcoded key from `'z'` to capitalized `'Z'`, as the NERSC invariant file uses `'Z'`.
3. **Data Variable Cleanup Selection**:
   - In `LANLMJODataset._build_virtual_dataset()`, locate the block handling native names: `native_name = next(iter(ds_var.data_vars))`.
   - This `next(iter(...))` assumes the physical variable is the first item. NERSC files include `area`, `gw`, `lat_bnds`, `lon_bnds`, and `utc_date`. This selection mechanism is therefore unsafe and will likely pull a proxy coordinate scalar instead of the (Time, Lat, Lon) array.
   - Refactor this step. The easiest way to identify the native name is to declare it in `SURFACE_VAR_MAP` and `ATMOS_VAR_MAP` (e.g., adding it as a 3rd tuple element) or attempt to dynamically strip the `*` suffix from `glob_pattern`.
   - If something is unknown (e.g. the exact name of MSL once available), do not guess; rely on the given dictionary mappings to extract what is needed.

## Risks & Reminders

- The native variable name for some components is unconfirmed for edge cases. If `glob` pattern stripping fails or yields a key not in `ds_var.data_vars`, your code must cleanly error and state what it attempted to find.
- Keep memory footprints low; no evaluation is required for this step.
