import cdsapi
import os

DATA_DIR = "/gpfs/gibbs/project/lu_lu/kam352/era5_sample"
os.makedirs(DATA_DIR, exist_ok=True)

c = cdsapi.Client()

print("Downloading surface level data...")

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure'],
        'year': '2023', 'month': '01', 'day': '01',
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'format': 'grib',
    },
    os.path.join(DATA_DIR, 'surface.grib'))

print("Downloading pressure level data...")
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind'],
        'pressure_level': ['50', '250', '500', '850', '1000'],
        'year': '2023', 'month': '01', 'day': '01',
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'format': 'grib',
    },
    os.path.join(DATA_DIR, 'pressure.grib'))

print(f"Data successfully downloaded to {DATA_DIR}")