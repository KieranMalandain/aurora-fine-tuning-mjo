"""scripts/download_slt.py — Download CDS ERA5 soil type (slt).

Provenance Record:
    Original 0.25° dataset `slt_data.nc` was downloaded via this script from the
    Copernicus Climate Data Store (CDS) ERA5 single levels archive for 1980-01-01.
    
    In task G1 (2026-09-17), `slt_data.nc` was regridded to the project's native 1°
    grid (180x360, latitudes 89.5 to -89.5, longitudes 0.5 to 359.5) using
    nearest-neighbor interpolation (preserving discrete categorical classes 0–7)
    and committed directly to `data/static/slt_1deg.nc` as int8 NetCDF4.
    
    This replaces the uncommitted /pscratch dependency and closes Q-07 and Q-17.
    This script is retained solely for archival provenance and reproducibility.
"""

from pathlib import Path

import cdsapi

DATA_DIR = Path("/pscratch/sd/k/kam352/Aurora/slt/")
DATA_DIR.mkdir(parents=True, exist_ok=True)

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": "soil_type",
        "year": "1980",
        "month": "01",
        "day": "01",
        "time": "00:00",
        "format": "netcdf",
        "grid": "0.25/0.25",
    },
    DATA_DIR / "slt_data.nc",
)

print("Static data successfully downloaded to", DATA_DIR)
