#!/usr/bin/env python3
"""scripts/make_test_fixtures.py — Generate synthetic NetCDF fixtures for aurora_mjo.

This script generates synthetic ERA5 NetCDF archives for offline testing of
the aurora_mjo dataset loader, models, and evaluation pipelines without
requiring access to NERSC CFS storage or GPU hardware.

Target State & Domain Priors Alignment:
======================================
See:
  - docs/campaigns/refactor/01_TARGET_STATE.md D6 (synthetic fixtures)
  - docs/campaigns/refactor/02_UPSTREAM_CONTRACT.md §2 (archive layout and native var names)
  - docs/campaigns/refactor/03_DOMAIN_PRIORS.md §1 (sample count arithmetic)
  - docs/campaigns/refactor/03_DOMAIN_PRIORS.md §5 (geophysical ranges)

Real-data properties reproduced:
--------------------------------
1. Complete upstream directory structure:
     <root>/Step00/ERA5.invariant/
     <root>/Step01/ERA5.remap_180x360MODIS_6hrInst/{gopt,sphu,tprt,uWnd,vWnd}/
     <root>/Step02/ERA5.remap_180x360MODIS_6hrInst/{T2,U10,V10,PS,tcwv}/
     <root>/Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX/
2. Native variable names matching LANLMJODataset mappings:
     Surface: 't2', 'u10', 'v10', 'ps' (for msl), 'mtnlwrf' (for ttr), 'tcwv'
     Atmospheric: 'z', 'q', 't', 'u', 'v'
     Statics: 'Z', 'LSM' (in Step00), 'slt' (in slt_data_synthetic.nc)
3. Real `time` coordinates at exact 6-hour spacing with standard CF datetime64[ns]
   encoding, enabling timestamp extraction and unix second indexing.
4. Leap-year asymmetry across two consecutive years:
     - 1980: 366 days x 4 = 1,464 timesteps
     - 1981: 365 days x 4 = 1,460 timesteps
     Total: 2,924 timesteps
5. Asymmetric per-variable file chunking:
     - Surface variables (e.g. t2): 1 file per year (2 files total)
     - Atmospheric q (sphu): 2 files per year (half-year chunks, 4 files total)
     This directly exercises the v3 timestamp alignment multi-file indexing logic.
6. Full 13 Aurora atmospheric pressure levels in `lev` coordinate:
     (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000) hPa.
7. Physically plausible mean values and ranges per 03_DOMAIN_PRIORS.md §5:
     - 2t: ~281 K (unweighted literature prior: ~278–282 K)
     - ps (msl proxy): ~97,250 Pa (prior: 96,000–99,000 Pa)
     - mtnlwrf (ttr): negative, ~ -228 W m⁻² (prior: -225 to -240 W m⁻²)
     - tcwv: ~20 kg m⁻² (prior: 18–25 kg m⁻²)
     - 10u, 10v: ~ -2 to +2 m s⁻¹ (range -15 to +15 m s⁻¹)
     - t @ 500 hPa: ~253 K, t @ 200 hPa: ~218 K
     - z @ 500 hPa: ~54,000 m² s⁻², z @ 1000 hPa: ~1,000 m² s⁻²
     - q @ 1000 hPa: ~0.014 kg kg⁻¹, q @ 200 hPa: ~1e-5 kg kg⁻¹
     - Static Z: ~3,709 m² s⁻², LSM: ~0.33, slt: ~0.67
8. High-resolution synthetic soil type `slt_data_synthetic.nc` at 720x1440.
9. Deliberately gapped archive variant `tests/fixtures/synthetic_archive_gapped/`
   with exactly 56 timesteps (14 days = 1 fortnight) of tcwv missing in 1980.

Deliberate divergences from real data (WHAT TESTS CANNOT ASSERT):
-----------------------------------------------------------------
1. Reduced horizontal resolution: 18x36 grid (10 deg spacing) instead of 180x360
   (1 deg spacing). Tests running against synthetic fixtures CANNOT assert native
   spatial shapes of (180, 360) or (720, 1440) for surface/atmospheric input/target
   tensors. They must assert (..., 18, 36) instead.
2. Vertical level representation: 13 Aurora pressure levels directly, rather than
   the 29 raw levels present in upstream LANL ERA5 files. Level subset probing
   asserts indices [0, 1, 2, ..., 12] rather than [0, 2, 4, ..., 28].
3. Deterministic synthetic fields: generated using smooth analytical trigonometry
   and level profiles, not real geophysical fluid dynamics or meteorological chaos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

# Aurora pressure levels (hPa)
AURORA_PLEVS = np.array(
    [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 850.0, 925.0, 1000.0],
    dtype=np.float64,
)

# Variable definitions: (aurora_name, step_subdir, native_name, is_atmos)
DATA_VARS = [
    ("2t", "Step02/ERA5.remap_180x360MODIS_6hrInst/T2", "t2", False),
    ("10u", "Step02/ERA5.remap_180x360MODIS_6hrInst/U10", "u10", False),
    ("10v", "Step02/ERA5.remap_180x360MODIS_6hrInst/V10", "v10", False),
    ("msl", "Step02/ERA5.remap_180x360MODIS_6hrInst/PS", "ps", False),
    ("ttr", "Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX", "mtnlwrf", False),
    ("tcwv", "Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv", "tcwv", False),
    ("z", "Step01/ERA5.remap_180x360MODIS_6hrInst/gopt", "z", True),
    ("q", "Step01/ERA5.remap_180x360MODIS_6hrInst/sphu", "q", True),
    ("t", "Step01/ERA5.remap_180x360MODIS_6hrInst/tprt", "t", True),
    ("u", "Step01/ERA5.remap_180x360MODIS_6hrInst/uWnd", "u", True),
    ("v", "Step01/ERA5.remap_180x360MODIS_6hrInst/vWnd", "v", True),
]


def make_grid() -> tuple[np.ndarray, np.ndarray]:
    """Generate 18x36 reduced lat/lon coordinates."""
    lat = np.linspace(-85.0, 85.0, 18, dtype=np.float64)
    lon = np.linspace(5.0, 355.0, 36, dtype=np.float64)
    return lat, lon


def make_timesteps(start_year: int, end_year: int) -> dict[int, np.ndarray]:
    """Generate 6-hourly timestamps for requested years."""
    timesteps: dict[int, np.ndarray] = {}
    for year in range(start_year, end_year + 1):
        t0 = np.datetime64(f"{year}-01-01T00:00:00")
        t1 = np.datetime64(f"{year + 1}-01-01T00:00:00")
        timesteps[year] = np.arange(t0, t1, np.timedelta64(6, "h"))
    return timesteps


def generate_var_data(
    var_name: str,
    times: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lev: np.ndarray | None = None,
) -> np.ndarray:
    """Generate physically plausible synthetic values for a given variable.

    All fields are analytical, deterministic, smooth functions of lat, lon,
    pressure level, and time, with means centered on 03_DOMAIN_PRIORS.md §5 values.
    """
    n_time = len(times)
    n_lat = len(lat)
    n_lon = len(lon)
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)

    # 2D spatial basis: (lat, lon)
    cos_phi = np.cos(phi)[:, None]
    sin_phi = np.sin(phi)[:, None]
    sin_2phi = np.sin(2.0 * phi)[:, None]
    cos_lam = np.cos(lam)[None, :]
    sin_lam = np.sin(lam)[None, :]

    # Surface variables
    if var_name == "t2":
        # Mean ~281 K, equator ~291 K, poles ~271 K
        base = 275.0 + 10.0 * cos_phi + 0.5 * sin_lam
        arr = np.broadcast_to(base[None, :, :], (n_time, n_lat, n_lon))
        return np.round(arr, 2).astype(np.float32)

    if var_name == "u10":
        # Tropical easterlies (~ -4 m/s), mid-latitude westerlies (~ +5 m/s)
        base = -5.0 * cos_phi + 8.0 * (sin_2phi**2)
        arr = np.broadcast_to(base[None, :, :], (n_time, n_lat, n_lon))
        return np.round(arr, 2).astype(np.float32)

    if var_name == "v10":
        # Mean ~0 m/s, range -5 to +5 m/s
        base = 2.5 * sin_phi * cos_lam
        arr = np.broadcast_to(base[None, :, :], (n_time, n_lat, n_lon))
        return np.round(arr, 2).astype(np.float32)

    if var_name == "ps":
        # Mean ~97,250 Pa, range 96,000 to 98,000 Pa
        base = 98000.0 - 1500.0 * (sin_phi**2) + 200.0 * sin_lam
        arr = np.broadcast_to(base[None, :, :], (n_time, n_lat, n_lon))
        return np.round(arr, 1).astype(np.float32)

    if var_name == "mtnlwrf":
        # Mean ~ -228 W/m^2, always negative, equator ~ -218, poles ~ -235
        base = -235.0 + 10.0 * cos_phi + 1.0 * cos_lam
        arr = np.broadcast_to(base[None, :, :], (n_time, n_lat, n_lon))
        return np.round(arr, 2).astype(np.float32)

    if var_name == "tcwv":
        # Mean ~20 kg/m^2, tropics ~40, poles ~5, strictly non-negative
        base = 5.0 + 35.0 * (cos_phi**3)
        arr = np.broadcast_to(base[None, :, :], (n_time, n_lat, n_lon))
        return np.round(arr, 2).astype(np.float32)

    # Atmospheric variables (time, lev, lat, lon)
    assert lev is not None
    n_lev = len(lev)

    if var_name == "t":
        # Vertical profile: 50 hPa: 215 K, 200 hPa: 218 K, 500 hPa: 253 K, 1000 hPa: 288 K
        t_prof = {
            50.0: 212.0,
            100.0: 209.0,
            150.0: 211.0,
            200.0: 215.0,
            250.0: 223.0,
            300.0: 232.0,
            400.0: 242.0,
            500.0: 250.0,
            600.0: 257.0,
            700.0: 265.0,
            850.0: 275.0,
            925.0: 280.0,
            1000.0: 285.0,
        }
        prof = np.array([t_prof[p] for p in lev], dtype=np.float32)
        base = prof[None, :, None, None] + (5.0 * cos_phi)[None, None, :, :]
        arr = np.broadcast_to(base, (n_time, n_lev, n_lat, n_lon))
        return np.round(arr, 1).astype(np.float32)

    if var_name == "q":
        # Specific humidity: 1000 hPa: ~0.014 kg/kg, 200 hPa: ~1e-5 kg/kg, 50 hPa: ~2e-6 kg/kg
        q_prof = {
            50.0: 2e-6,
            100.0: 3e-6,
            150.0: 5e-6,
            200.0: 1e-5,
            250.0: 5e-5,
            300.0: 2e-4,
            400.0: 8e-4,
            500.0: 0.002,
            600.0: 0.004,
            700.0: 0.006,
            850.0: 0.009,
            925.0: 0.011,
            1000.0: 0.014,
        }
        prof = np.array([q_prof[p] for p in lev], dtype=np.float32)
        factor = (0.7 + 0.3 * cos_phi)[None, None, :, :]
        base = prof[None, :, None, None] * factor
        arr = np.broadcast_to(base, (n_time, n_lev, n_lat, n_lon))
        return arr.astype(np.float32)

    if var_name == "z":
        # Geopotential: 1000 hPa: ~1,000 m^2/s^2, 500 hPa: ~54,000, 50 hPa: ~200,000
        z_prof = {
            50.0: 200500.0,
            100.0: 160500.0,
            150.0: 135500.0,
            200.0: 116500.0,
            250.0: 102500.0,
            300.0: 90500.0,
            400.0: 70500.0,
            500.0: 54500.0,
            600.0: 41500.0,
            700.0: 29500.0,
            850.0: 14500.0,
            925.0: 8000.0,
            1000.0: 1500.0,
        }
        prof = np.array([z_prof[p] for p in lev], dtype=np.float32)
        base = prof[None, :, None, None] - (1000.0 * (sin_phi**2))[None, None, :, :]
        arr = np.broadcast_to(base, (n_time, n_lev, n_lat, n_lon))
        return np.round(arr, 0).astype(np.float32)

    if var_name == "u":
        # Jet stream at 200 hPa (~35 m/s), weaker elsewhere
        u_prof = {
            50.0: 5.0,
            100.0: 10.0,
            150.0: 25.0,
            200.0: 35.0,
            250.0: 30.0,
            300.0: 25.0,
            400.0: 18.0,
            500.0: 12.0,
            600.0: 8.0,
            700.0: 5.0,
            850.0: 0.0,
            925.0: -2.0,
            1000.0: -4.0,
        }
        prof = np.array([u_prof[p] for p in lev], dtype=np.float32)
        jet = (sin_2phi**2)[None, None, :, :]
        base = prof[None, :, None, None] * jet
        arr = np.broadcast_to(base, (n_time, n_lev, n_lat, n_lon))
        return np.round(arr, 1).astype(np.float32)

    if var_name == "v":
        # Meridional wind, mean ~0 m/s
        v_prof = np.linspace(-3.0, 3.0, n_lev, dtype=np.float32)
        base = v_prof[None, :, None, None] * (sin_phi * cos_lam)[None, None, :, :]
        arr = np.broadcast_to(base, (n_time, n_lev, n_lat, n_lon))
        return np.round(arr, 2).astype(np.float32)

    raise ValueError(f"Unknown variable name: {var_name}")


def write_netcdf(
    path: Path,
    data: np.ndarray,
    var_name: str,
    times: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lev: np.ndarray | None = None,
) -> None:
    """Save array to NetCDF4 with zlib compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if lev is not None:
        da = xr.DataArray(
            data,
            dims=["time", "lev", "lat", "lon"],
            coords={"time": times, "lev": lev, "lat": lat, "lon": lon},
        )
    else:
        da = xr.DataArray(
            data,
            dims=["time", "lat", "lon"],
            coords={"time": times, "lat": lat, "lon": lon},
        )
    ds = xr.Dataset({var_name: da})
    encoding = {var_name: {"zlib": True, "complevel": 9}}
    ds.to_netcdf(str(path), engine="netcdf4", encoding=encoding)


def generate_archive(
    root_dir: Path,
    years: list[int] | None = None,
    gapped_tcwv: bool = False,
) -> None:
    """Generate all Step00, Step01, Step02, Step03 variables into root_dir."""
    if years is None:
        years = [1980, 1981]
    lat, lon = make_grid()
    year_times = make_timesteps(min(years), max(years))

    # 1. Step00 Invariants
    inv_dir = root_dir / "Step00/ERA5.invariant"
    inv_dir.mkdir(parents=True, exist_ok=True)
    inv_time = np.array(["1979-01-01T00:00:00"], dtype="datetime64[ns]")

    phi = np.deg2rad(lat)
    cos_phi = np.cos(phi)[:, None]

    # Invariant Z (surface geopotential, mean ~3709 m^2/s^2)
    z_base = (3709.0 + 500.0 * np.sin(phi)[:, None] + np.zeros((1, len(lon)))).astype(np.float32)
    da_z = xr.DataArray(
        z_base[None, :, :],
        dims=["time", "lat", "lon"],
        coords={"time": inv_time, "lat": lat, "lon": lon},
    )
    xr.Dataset({"Z": da_z}).to_netcdf(
        str(
            inv_dir
            / "e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100_remap_180x360MODIS.nc"
        ),
        engine="netcdf4",
        encoding={"Z": {"zlib": True, "complevel": 9}},
    )

    # Invariant LSM (land-sea mask, fraction in [0, 1], mean ~0.3357)
    lsm_base = np.where(cos_phi > 0.6, 0.44, 0.20).astype(np.float32) + np.zeros(
        (1, len(lon)), dtype=np.float32
    )
    da_lsm = xr.DataArray(
        lsm_base[None, :, :],
        dims=["time", "lat", "lon"],
        coords={"time": inv_time, "lat": lat, "lon": lon},
    )
    xr.Dataset({"LSM": da_lsm}).to_netcdf(
        str(
            inv_dir
            / "e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100_remap_180x360MODIS.nc"
        ),
        engine="netcdf4",
        encoding={"LSM": {"zlib": True, "complevel": 9}},
    )

    # 2. Time-series variables across all years
    for _aurora_name, step_subdir, native_name, is_atmos in DATA_VARS:
        var_dir = root_dir / step_subdir
        var_lev = AURORA_PLEVS if is_atmos else None

        for year in years:
            times = year_times[year]

            # Variable-specific chunking: q is split into half-years; all others 1 file/year
            if native_name == "q":
                mid = len(times) // 2
                h1_times = times[:mid]
                h2_times = times[mid:]

                h1_data = generate_var_data(native_name, h1_times, lat, lon, var_lev)
                h2_data = generate_var_data(native_name, h2_times, lat, lon, var_lev)

                write_netcdf(
                    var_dir / f"e5.oper.an.pl.128_133_q.{year}_h1.nc",
                    h1_data,
                    native_name,
                    h1_times,
                    lat,
                    lon,
                    var_lev,
                )
                write_netcdf(
                    var_dir / f"e5.oper.an.pl.128_133_q.{year}_h2.nc",
                    h2_data,
                    native_name,
                    h2_times,
                    lat,
                    lon,
                    var_lev,
                )
            else:
                if gapped_tcwv and native_name == "tcwv" and year == 1980:
                    # Drop exactly 56 timesteps (14 days = 1 fortnight: June 1 to June 14)
                    gap_mask = (times < np.datetime64("1980-06-01T00:00:00")) | (
                        times > np.datetime64("1980-06-14T18:00:00")
                    )
                    use_times = times[gap_mask]
                else:
                    use_times = times

                data = generate_var_data(native_name, use_times, lat, lon, var_lev)
                prefix = "pl" if is_atmos else "sfc"
                write_netcdf(
                    var_dir / f"e5.oper.an.{prefix}.128_{native_name}.{year}.nc",
                    data,
                    native_name,
                    use_times,
                    lat,
                    lon,
                    var_lev,
                )


def generate_slt(slt_path: Path) -> None:
    """Generate synthetic slt_data.nc matching reduced synthetic grid (18x36)."""
    slt_path.parent.mkdir(parents=True, exist_ok=True)
    grid_lat, grid_lon = make_grid()
    lat = grid_lat[::-1].copy() if grid_lat[0] < grid_lat[-1] else grid_lat.copy()
    lon = grid_lon.copy()
    H = len(lat)
    W = len(lon)

    # Deterministic categorical 0..7 field: ocean (~72%) is 0; land (1..7)
    np.random.seed(42)
    slt = np.zeros((1, H, W), dtype=np.float32)
    ocean = np.random.rand(1, H, W) < 0.72
    land_cats = np.random.choice(
        [1, 2, 3, 4, 5, 6, 7],
        size=(1, H, W),
        p=[0.35, 0.30, 0.15, 0.10, 0.04, 0.03, 0.03],
    ).astype(np.float32)
    slt[~ocean] = land_cats[~ocean]

    valid_time = np.array(["1980-01-01T00:00:00"], dtype="datetime64[ns]")
    da = xr.DataArray(
        slt,
        dims=["valid_time", "latitude", "longitude"],
        coords={"valid_time": valid_time, "latitude": lat, "longitude": lon},
    )
    xr.Dataset({"slt": da}).to_netcdf(
        str(slt_path),
        engine="netcdf4",
        encoding={"slt": {"zlib": True, "complevel": 9}},
    )


def compute_directory_size(path: Path) -> int:
    """Return total recursive size in bytes."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic NetCDF fixtures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/fixtures/synthetic_archive"),
        help="Target directory for standard synthetic archive.",
    )
    parser.add_argument(
        "--gapped-output-dir",
        type=Path,
        default=Path("tests/fixtures/synthetic_archive_gapped"),
        help="Target directory for gapped synthetic archive.",
    )
    parser.add_argument(
        "--slt-output-path",
        type=Path,
        default=Path("tests/fixtures/slt_data_synthetic.nc"),
        help="Target path for synthetic slt file.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print summary size report.",
    )
    args = parser.parse_args()

    print(f"Generating synthetic archive at: {args.output_dir}")
    generate_archive(args.output_dir, years=[1980, 1981], gapped_tcwv=False)

    print(f"Generating gapped synthetic archive at: {args.gapped_output_dir}")
    generate_archive(args.gapped_output_dir, years=[1980, 1981], gapped_tcwv=True)

    print(f"Generating synthetic SLT at: {args.slt_output_path}")
    generate_slt(args.slt_output_path)

    std_size = compute_directory_size(args.output_dir)
    gap_size = compute_directory_size(args.gapped_output_dir)
    slt_size = compute_directory_size(args.slt_output_path)
    total_size = std_size + gap_size + slt_size

    if args.report or True:
        print("\n=== Synthetic Fixture Generation Report ===")
        print(
            f"Standard Archive ({args.output_dir}): "
            f"{std_size / (1024*1024):.2f} MB ({std_size:,} bytes)"
        )
        print(
            f"Gapped Archive ({args.gapped_output_dir}):   "
            f"{gap_size / (1024*1024):.2f} MB ({gap_size:,} bytes)"
        )
        print(
            f"Synthetic SLT ({args.slt_output_path}):      "
            f"{slt_size / (1024*1024):.2f} MB ({slt_size:,} bytes)"
        )
        print(
            f"Total Fixtures Size:                    "
            f"{total_size / (1024*1024):.2f} MB ({total_size:,} bytes)"
        )
        print("Hard Ceiling: 20.00 MB | Target: ~5.00 MB")


if __name__ == "__main__":
    main()
