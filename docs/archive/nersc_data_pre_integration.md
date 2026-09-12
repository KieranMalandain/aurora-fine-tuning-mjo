# NERSC ERA5 Data Structure and Integration Guide

This document outlines the internal structure, dimensions, and variables of the LANL ERA5 NetCDF files stored on the NERSC cluster. This information is critical for integrating the dataset properly into the Aurora fine-tuning pipeline, specifically via `src/dataset.py`.

## Overview 

The data consists of ERA5 samples remapped to a 180x360 resolution (1-degree equivalent), sampled every 6 hours (`6hrInst`). The base directory on NERSC where this data resides is:
`/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results`

### General Notes
- **Grid Size**: Native dimension is `lat: 180, lon: 360`. Aurora relies on a 0.25-degree resolution, so this data must be upsampled in the dataloader to `720 x 1440` using bilinear interpolation.
- **Coordinates**: Standard spatial coordinate axes are `'lat'` and `'lon'`.
- **Temporal**: The main time axis is recorded as `'time'`. Each file typically contains an axis for `utc_date` as a data variable, but `time` is the proper coordinate to index against.
- **Precision**: Physical variables generally use `float32`, while certain geometric factors (e.g., `area`) are `float64`.

---

## Variable Categories

The dataset is broken up into sub-directories (`StepXX`) based on variable types. Below are the key characteristics of each.

### 1. Surface Variables

- **Sample Path**: `Step02/ERA5.remap_180x360MODIS_6hrInst/T2/e5.oper.an.sfc.128_167_2t.ll025sc.198007_remap_180x360.nc`
- **Dimensions**: `lat` (180), `lon` (360), `time` (e.g., 124), `nbnd` (2)
- **Coordinates keys**: `['lat', 'lon', 'time']`
- **Data Variable keys**: `['area', 'gw', 'lat_bnds', 'lon_bnds', 't2', 'utc_date']`
- **Implications**: The target physical variable is often co-located with geometric variables (`area`, bounds, `gw`). The internal NetCDF variable key exactly matches the physical name string (e.g., `'t2'`).

### 2. Atmospheric (Pressure Level) Variables

- **Sample Path**: `Step01/ERA5.remap_180x360MODIS_6hrInst/uWnd/e5.oper.an.pl.128_131_u.ll025uv.201311_remap_180x360MODIS.nc`
- **Dimensions**: `lat` (180), `lon` (360), `lev` (29), `time` (e.g., 120), `nbnd` (2)
- **Coordinates keys**: `['lat', 'lev', 'lon', 'time']`
- **Data Variable keys**: `['area', 'gw', 'lat_bnds', 'lon_bnds', 'u', 'utc_date']`
- **Important**: The vertical coordinate is named `'lev'`, meaning any logic checking for `'level'` or `'isobaricInhPa'` will fail and must be updated. NERSC ERA5 provides 29 levels, but Aurora only uses 13, so interpolation or slicing is required.

### 3. Static / Invariant

- **Sample Path**: `Step00/ERA5.invariant/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100_remap_180x360MODIS.nc`
- **Dimensions**: `time` (1), `lat` (180), `lon` (360), `nbnd` (2)
- **Coordinates keys**: `['lat', 'lon', 'time']`
- **Data Variable keys**: `['Z', 'area', 'gw', 'lat_bnds', 'lon_bnds', 'utc_date']`
- **Important**: The geopotential variable key is capitalized as `'Z'`. 

---

## Actionable Changes for `src/dataset.py`

If you are modifying the dataloader to consume this NERSC data correctly, the following specific fixes must be applied to `src/dataset.py`:

1. **Vertical Coordinate Bug Fix**:
   - Near the pressure dataset filtering, the current code assumes:
     `plev_coord = 'isobaricInhPa' if 'isobaricInhPa' in self.pressure_ds.coords else 'level'`
   - **Correction**: It must be updated to check for `lev`:
     `plev_coord = 'lev'`

2. **Static Geo-Potential Keys**:
   - In `_load_static_vars()`, the dataloader may attempt to look up `['z']`.
   - **Correction**: The invariant file holds this value under the capitalized key `['Z']`. It must be explicitly mapped/accessed as `xr.open_dataset(z_file)['Z'].values`.

3. **Data Variable Cleanup Selection**:
   - NetCDF files include several non-physical proxy variables (`area`, `gw`, `utc_date`, etc.). Dataloaders concatenating natively via `xarray.open_mfdataset()` alongside physical data arrays must ensure they slice accurately so the Aurora system explicitly feeds only the correctly named variable tensor (e.g. `u`, `t2`), tossing the rest of the layout boundaries.
