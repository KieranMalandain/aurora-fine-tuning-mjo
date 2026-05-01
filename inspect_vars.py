import xarray as xr
import glob
import sys

def inspect(path_pattern):
    files = glob.glob(path_pattern)
    if not files:
        print(f"No files found for {path_pattern}")
        return
    file = files[0]
    try:
        ds = xr.open_dataset(file, engine='h5netcdf')
        print(f"[{path_pattern}] vars: {list(ds.data_vars.keys())}")
    except Exception as e:
        print(f"[{path_pattern}] failed: {e}")

base = "/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results"
inspect(f"{base}/Step00/ERA5.invariant/*_z.*.nc")
inspect(f"{base}/Step00/ERA5.invariant/*_lsm.*.nc")
inspect(f"{base}/Step02/ERA5.remap_180x360MODIS_6hrInst/T2/*.nc")
inspect(f"{base}/Step02/ERA5.remap_180x360MODIS_6hrInst/U10/*.nc")
inspect(f"{base}/Step02/ERA5.remap_180x360MODIS_6hrInst/V10/*.nc")
inspect(f"{base}/Step02/ERA5.remap_180x360MODIS_6hrInst/PS/*.nc")
inspect(f"{base}/Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX/*.nc")
inspect(f"{base}/Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv/*.nc")

inspect(f"{base}/Step01/ERA5.remap_180x360MODIS_6hrInst/gopt/*.nc")
inspect(f"{base}/Step01/ERA5.remap_180x360MODIS_6hrInst/sphu/*.nc")
inspect(f"{base}/Step01/ERA5.remap_180x360MODIS_6hrInst/tprt/*.nc")
inspect(f"{base}/Step01/ERA5.remap_180x360MODIS_6hrInst/uWnd/*.nc")
inspect(f"{base}/Step01/ERA5.remap_180x360MODIS_6hrInst/vWnd/*.nc")

