# Sea Surface Temperature (SST) Static Boundary Condition

## Provenance Record
- **Source:** Copernicus Climate Data Store (CDS)
- **Dataset:** `reanalysis-era5-single-levels`
- **Product type:** `reanalysis`
- **Variable:** `sea_surface_temperature` (`sst`)
- **Temporal Resolution:** Daily at 00:00 UTC
- **Licence:** Copernicus Open Access Licence (Creative Commons Attribution 4.0 International, CC BY 4.0)
- **Retrieval Date:** 2026-09-17 (Task G5, campaign `science-baseline`, resolving Q-20)
- **Script:** `scripts/fetch_sst.py`

## Grid Specification
- **Native Resolution:** 0.25° regular latitude-longitude (721 × 1440 points).
- **Target Resolution:** 1.0° cell-centred regular latitude-longitude (180 × 360 points).
  - Latitudes: 89.5° to -89.5° (strictly descending, 180 points).
  - Longitudes: 0.5° to 359.5° (strictly ascending, 360 points).
  - Verified exact coordinate equality with CFS invariants `z`, `lsm`, and static `slt_1deg.nc`.

## Land Fill Decision & Justification
- **Method:** Zonal-mean SST per latitude band. Latitudes without any ocean points (e.g. Antarctic interior) are linearly interpolated from adjacent ocean-bearing latitude bands.
- **Justification:** The Aurora model already receives the land-sea mask (`lsm`) as a static condition and can learn to ignore land values. However, using 0 K or NetCDF fill values (e.g. -32767) would create catastrophic outliers under normalisation (-24σ to -2800σ). With zonal-mean fill:
  - Worst land point deviation across sampled years: **1.7452 σ** (Value: 306.12 K, Mean: 285.5980 K, Std: 11.7613 K).
  - 100% of all land points fall strictly within [-1.36 σ, +1.75 σ] of the global mean.

## Normalisation Statistics (1980–2015)
- **Mean:** 285.5980 K
- **Std:** 11.7613 K
- **Total Samples Measured:** 71,020,800 grid points across representative years [1980, 1998, 2015].

## Checksums and File Sizes
The NetCDF4 files stored in this directory are excluded from git tracking per protocol §9 (> 1 MB per file):
- `sst_1deg_1980.nc`:
  - Size: 24,435,016 bytes (23.30 MB)
  - SHA256: `cfe4319e35b4706410e3a2ccbdb45e5e6a8bd36eb3c1781b0b43b3ad3228aff0`
- `sst_1deg_1998.nc`:
  - Size: 24,499,350 bytes (23.36 MB)
  - SHA256: `6a423cca3368ca063375f7e140843b83b5f0a3cebf935df5765500e64f932ccc`
- `sst_1deg_2015.nc`:
  - Size: 24,611,678 bytes (23.47 MB)
  - SHA256: `fadffce03682f53f67c7a59a8d2882150a86677047f87ef3bec2f5a7c51c6e8e`
