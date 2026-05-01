import cdsapi
import os
from pathlib import Path

DATA_DIR = Path("/pscratch/sd/k/kam352/Aurora/slt/")
DATA_DIR.mkdir(parents=True, exist_ok=True)

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': 'soil_type',
        'year': '1980', 'month': '01', 'day': '01', 'time': '00:00',
        'format': 'netcdf',
        'grid': '0.25/0.25',
    },
    DATA_DIR / 'slt_data.nc'
)

print("Static data successfully downloaded to", DATA_DIR)