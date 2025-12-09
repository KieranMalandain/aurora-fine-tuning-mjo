import cdsapi
import os
from pathlib import Path

DATA_DIR = Path("/gpfs/gibbs/project/lu_lu/kam352/era5_static_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['land_sea_mask', 'geopotential', 'soil_type'],
        'year': '2015', 'month': '01', 'day': '01', 'time': '00:00',
        'format': 'netcdf',
        'grid': '0.25/0.25',
    },
    DATA_DIR / 'static_data.nc'
)

print("Static data successfully downloaded to", DATA_DIR)