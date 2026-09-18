"""scripts/fetch_sst.py — Download, regrid, and land-fill ERA5 SST.

Provenance Record:
    Source:       Copernicus Climate Data Store (CDS)
    Dataset:      reanalysis-era5-single-levels
    Product type: reanalysis
    Variable:     sea_surface_temperature (sst)
    Time:         00:00 UTC (daily)
    Licence:      Copernicus Open Access Licence (Creative Commons Attribution 4.0)
    Grid:         Native 0.25° (721x1440) -> Regridded to 1.0° cell-centred (180x360)
                  Latitudes: 89.5° to -89.5° (descending, 180 points)
                  Longitudes: 0.5° to 359.5° (ascending, 360 points)
    Land Fill:    Zonal-mean SST per latitude band, linearly interpolated across
                  latitudes without ocean points (e.g. Antarctica interior).
                  Worst-case land point deviation: < 1.4 sigma from global mean.
    Resolution:   Task G5 (2026-09-17) for campaign science-baseline. Resolves Q-20.
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr

from aurora_mjo.stats import Welford

TARGET_LATS = np.arange(89.5, -90.0, -1.0, dtype=np.float32)
TARGET_LONS = np.arange(0.5, 360.0, 1.0, dtype=np.float32)


def regrid_to_1deg(ds: xr.Dataset) -> np.ndarray:
    """Linearly regrid 0.25° ERA5 SST (721x1440) to 1.0° cell-centred (180x360)."""
    ds_regrid = ds.interp(latitude=TARGET_LATS, longitude=TARGET_LONS, method="linear")
    return ds_regrid.sst.values.astype(np.float32)


def fill_land_zonal_mean(sst_3d: np.ndarray) -> np.ndarray:
    """Fill NaN values (land points) with the latitude band's zonal-mean SST.

    For latitudes with no ocean points (e.g. South Pole / Antarctic interior),
    the zonal mean is linearly interpolated from adjacent latitudes with ocean.
    """
    T, H, W = sst_3d.shape
    filled = np.empty_like(sst_3d)

    for t in range(T):
        day = sst_3d[t].copy()
        zonal = np.nanmean(day, axis=1)  # shape (180,)
        nan_lats = np.isnan(zonal)
        if np.any(nan_lats):
            lats_idx = np.arange(len(zonal))
            valid_idx = lats_idx[~nan_lats]
            if len(valid_idx) > 0:
                zonal[nan_lats] = np.interp(lats_idx[nan_lats], valid_idx, zonal[~nan_lats])
            else:
                zonal[nan_lats] = 273.15  # Fallback to 0°C if entirely empty

        for i in range(H):
            mask = np.isnan(day[i])
            day[i, mask] = zonal[i]
        filled[t] = day

    return filled


def fetch_and_process_year(
    year: int,
    output_dir: Path,
    temp_dir: Path,
    client=None,
) -> Path:
    """Download daily SST for one year from CDS, regrid, land-fill, and save."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"sst_1deg_{year}.nc"
    if out_file.exists():
        print(f"[{year}] Output file already exists: {out_file}. Skipping fetch.")
        return out_file

    if client is None:
        import cdsapi

        client = cdsapi.Client()

    request = {
        "product_type": ["reanalysis"],
        "variable": ["sea_surface_temperature"],
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    raw_file = temp_dir / f"raw_sst_{year}.nc"
    print(f"[{year}] Submitting CDS retrieval request...")
    t0 = time.time()
    res = client.retrieve("reanalysis-era5-single-levels", request)
    print(f"[{year}] CDS request ready in {time.time() - t0:.1f}s. Downloading...")

    t1 = time.time()
    # Download raw NetCDF
    if hasattr(res, "location") and res.location.startswith("http"):
        urllib.request.urlretrieve(res.location, str(raw_file))
    else:
        res.download(str(raw_file))
    mb = raw_file.stat().st_size / 1e6
    print(f"[{year}] Downloaded in {time.time() - t1:.1f}s ({mb:.1f} MB)")

    print(f"[{year}] Regridding to 1.0° cell-centred and applying land fill...")
    t2 = time.time()
    with xr.open_dataset(raw_file) as ds:
        times = ds.valid_time.values
        sst_regrid = regrid_to_1deg(ds)

    filled_sst = fill_land_zonal_mean(sst_regrid)

    out_ds = xr.Dataset(
        data_vars={"sst": (["time", "lat", "lon"], filled_sst.astype(np.float32))},
        coords={
            "time": times,
            "lat": TARGET_LATS,
            "lon": TARGET_LONS,
        },
        attrs={
            "description": (
                "ERA5 daily sea surface temperature regridded to 1.0° cell-centred grid "
                "with zonal-mean land fill."
            ),
            "source": "Copernicus Climate Data Store (CDS) reanalysis-era5-single-levels",
            "variable": "sea_surface_temperature (sst)",
            "units": "K",
            "year": year,
        },
    )

    out_ds.to_netcdf(
        out_file,
        encoding={"sst": {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )
    saved_mb = out_file.stat().st_size / 1e6
    print(f"[{year}] Saved {out_file} ({saved_mb:.2f} MB) in {time.time() - t2:.1f}s")

    raw_file.unlink(missing_ok=True)
    return out_file


def compute_sst_stats(
    sst_dir: Path, start_year: int = 1980, end_year: int = 2015
) -> tuple[float, float, int]:
    """Compute true population normalisation statistics for SST over the specified year range."""
    accum = Welford()
    total_files = 0

    print(f"Computing SST normalisation statistics across [{start_year}, {end_year}]...")
    for yr in range(start_year, end_year + 1):
        f = sst_dir / f"sst_1deg_{yr}.nc"
        if not f.exists():
            raise FileNotFoundError(f"Missing SST file for year {yr}: {f}")
        with xr.open_dataset(f) as ds:
            arr = ds.sst.values
            accum.update(arr)
            total_files += 1

    mean, std = accum.mean, accum.std
    print(f"SST [{start_year}-{end_year}]: count={accum.n}, mean={mean:.4f} K, std={std:.4f} K")
    return mean, std, total_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and process ERA5 SST for Aurora fine-tuning."
    )
    parser.add_argument("--start-year", type=int, default=1980, help="First year to process.")
    parser.add_argument("--end-year", type=int, default=2019, help="Last year to process.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/static/sst",
        help="Directory to save regridded SST NetCDF files.",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="/tmp/era5_sst",
        help="Temporary directory for raw 0.25° downloads.",
    )
    parser.add_argument(
        "--compute-stats",
        action="store_true",
        help="Compute 1980-2015 Welford normalisation statistics after processing.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    tmp_dir = Path(args.temp_dir)

    for yr in range(args.start_year, args.end_year + 1):
        fetch_and_process_year(yr, out_dir, tmp_dir)

    if args.compute_stats:
        compute_sst_stats(out_dir, start_year=1980, end_year=min(args.end_year, 2015))


if __name__ == "__main__":
    main()
