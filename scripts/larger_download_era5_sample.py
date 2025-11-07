# scripts/larger_download_era5_sample.py

# import cdsapi
# import os
# from pathlib import Path

# DATA_DIR = Path("/gpfs/gibbs/project/lu_lu/kam352/era5_sample_jan2015")
# os.makedirs(DATA_DIR, exist_ok=True)

# c = cdsapi.Client()

# common_request_params = {
#     'product_type': 'reanalysis',
#     'year': '2015',
#     'month': '01',
#     'day': [f'{day:02d}' for day in range(1, 32)],
#     'time': ['00:00', '06:00', '12:00', '18:00'],
#     'format': 'netcdf',
#     'grid': '0.25/0.25',
# }

# print("Downloading surface level data...")

# c.retrieve(
#     'reanalysis-era5-single-levels',
#     {
#         **common_request_params,
#         'variable': [
#             '2m_temperature',
#             '10m_u_component_of_wind', 
#             '10m_v_component_of_wind',
#             'mean_sea_level_pressure', 
#             'top_net_thermal_radiation', 
#             'total_column_water_vapour'
#         ],
#     },
#     DATA_DIR / 'surface_jan2015.nc'
# )

# print("Surface level data download initiated.")

# print("Downloading pressure level data...")

# c.retrieve(
#     'reanalysis-era5-pressure-levels',
#     {
#         **common_request_params, 
#         'variable': [
#             'geopotential', 
#             'specific_humidity', 
#             'temperature', 
#             'u_component_of_wind', 
#             'v_component_of_wind'
#         ],
#         'pressure_level': [
#             '50', '100', '150', '200', '250', '300', '400', 
#             '500', '600', '700', '850', '925', '1000'
#         ], 
#     },
#     DATA_DIR / 'pressure_jan2015.nc'
# )

# print(f"Data successfully downloaded to {DATA_DIR}")

# scripts/larger_download_era5_sample.py (Robust Version)

import cdsapi
import os
from pathlib import Path

# Use a descriptive directory name
DATA_DIR = Path("/gpfs/gibbs/project/lu_lu/kam352/era5_jan2015_daily")
os.makedirs(DATA_DIR, exist_ok=True)

c = cdsapi.Client()

YEAR = '2015'
MONTH = '01'
DAYS = [f'{d:02d}' for d in range(1, 32)] # All 31 days of January

# --- Download Surface Data (Day by Day) ---
print("--- Downloading Surface Data (day-by-day) ---")
for day in DAYS:
    target_file = DATA_DIR / f"surface_{YEAR}-{MONTH}-{day}.nc"
    if target_file.exists():
        print(f"Skipping {target_file.name}, already exists.")
        continue

    print(f"Requesting surface data for {YEAR}-{MONTH}-{day}...")
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'year': YEAR, 'month': MONTH, 'day': day,
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'format': 'netcdf',
            'grid': '0.25/0.25',
            'variable': [
                '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
                'mean_sea_level_pressure', 'top_net_thermal_radiation', 'total_column_water_vapour'
            ],
        },
        target_file
    )
    print(f"Finished downloading {target_file.name}")

# --- Download Pressure Level Data (Day by Day) ---
print("\n--- Downloading Pressure Level Data ---")
for day in DAYS:
    target_file = DATA_DIR / f"pressure_{YEAR}-{MONTH}-{day}.nc"
    if target_file.exists():
        print(f"Skipping {target_file.name}, already exists.")
        continue

    print(f"Requesting pressure data for {YEAR}-{MONTH}-{day}...")
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'year': YEAR, 'month': MONTH, 'day': day,
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'format': 'netcdf',
            'grid': '0.25/0.25',
            'variable': [
                'geopotential', 'specific_humidity', 'temperature', 
                'u_component_of_wind', 'v_component_of_wind'
            ],
            'pressure_level': [
                '50', '100', '150', '200', '250', '300', '400', 
                '500', '600', '700', '850', '925', '1000'
            ],
        },
        target_file
    )
    print(f"Finished downloading {target_file.name}")

print(f"\nAll data successfully downloaded to {DATA_DIR}")