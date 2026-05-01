# Finetuning Dataset: ERA5

## Dataset Overview and Variables Table

This table outlines the variables included in the ERA5 finetuning dataset (45 years of data from 1980-2024). Information provided by Dr. Xiaoming Sun, PI at LANL.

Note that `dirERA5/` is a placeholder for the directory `/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results/`.

| Variable | Description | Time Variant | Levels | Location / Filename |
| :--- | :--- | :--- | :--- | :--- |
| **Z** | Geopotential at the surface (m²/s²) (divide by g=9.8 m/s² for terrain height in m) | No | Single | `dirERA5/Step00/ERA5.invariant/e5.oper.invariant.128_129_z...` |
| **LSM** | Land-sea mask (0-1) | No | Single | `dirERA5/Step00/ERA5.invariant/e5.oper.invariant.128_172_lsm...` |
| **gopt** | Geopotential (m²/s²) | Yes | Multiple | `dirERA5/.../ERA5.remap_180x360MODIS_6hrInst/gopt.nc` |
| **pv** | Potential vorticity (Km²/kg/s) | Yes | Multiple | `dirERA5/.../ERA5.remap_180x360MODIS_6hrInst/pvor.nc` |
| **rhum** | Relative humidity (%) | Yes | Multiple | `dirERA5/.../ERA5.remap_180x360MODIS_6hrInst/rhum.nc` |
| **sphu** | Specific humidity (kg/kg) | Yes | Multiple | `dirERA5/.../ERA5.remap_180x360MODIS_6hrInst/sphu.nc` |
| **t** | Temperature (K) | Yes | Multiple | `dirERA5/.../ERA5.remap_180x360MODIS_6hrInst/t.nc` |
| **u** | U component of wind (m/s) (west-east) | Yes | Multiple | `dirERA5/.../ERA5.remap_180x360MODIS_6hrInst/uWnd.nc` |
| **VO** | Vorticity (relative) (s⁻¹) | Yes | Multiple | `dirERA5/.../ERA5.remap_180x360MODIS_6hrInst/vort.nc` |
| **v** | V component of wind (m/s) (south-north) | Yes | Multiple | `dirERA5/.../ERA5.remap_180x360MODIS_6hrInst/vWnd.nc` |
| **W** | Vertical velocity (Pa/s) | Yes | Multiple | `dirERA5/.../ERA5.remap_180x360MODIS_6hrInst/wWnd.nc` |
| **EFLX** | Instantaneous moisture flux (kg/m²/s) | Yes | Single | `dirERA5/Step02/.../EFLX.nc` |
| **P (PS)** | Surface pressure (Pa) | Yes | Single | `dirERA5/Step02/.../PS/.nc` |
| **SHFLX** | Instantaneous surface sensible heat flux (W/m²) | Yes | Single | `dirERA5/Step02/.../SHFLX/.nc` |
| **t2** | 2 meter temperature (K) | Yes | Single | `dirERA5/Step02/.../T2/.nc` |
| **tcc** | Total cloud cover (0-1) | Yes | Single | `dirERA5/Step02/.../TCC/.nc` |
| **tcwv** | Total column water vapor (kg/m²) | Yes | Single | `dirERA5/Step02/.../tcwv/.nc` |
| **td2** | 2 meter dewpoint temperature (K) | Yes | Single | `dirERA5/Step02/.../Td2/.nc` |
| **u10** | 10 meter U wind component (m/s) (west-east) | Yes | Single | `dirERA5/Step02/.../U10.nc` |
| **v10** | 10 meter V wind component (m/s) (south-north) | Yes | Single | `dirERA5/Step02/.../V10.nc` |
| **mslhf** | Mean surface latent heat flux in the previous 1 hour (W/m²) | Yes | Single | `dirERA5/.../meanSLHFLX/.nc` |
| **msnlwrf** | Mean surface net long-wave radiation flux in previous 1 hr (W/m²) | Yes | Single | `dirERA5/.../meanSNLWFLX/.nc` |
| **msnswrf** | Mean surface net short-wave radiation flux in previous 1 hr (W/m²) | Yes | Single | `dirERA5/.../meanSNSWFLX/.nc` |
| **msshf** | Mean surface sensible heat flux in the previous 1 hour (W/m²) | Yes | Single | `dirERA5/.../meanSSHFLX.nc` |
| **mtnlwrf** | Mean top net long-wave radiation flux in the previous 1 hour (W/m²) | Yes | Single | `dirERA5/.../meanTNLWFLX/.nc` |
| **mtnswrf** | Mean top net short-wave radiation flux in the previous 1 hour (W/m²) | Yes | Single | `dirERA5/.../meanTNSWFLX.nc` |
| **tp1h** | Total precipitation accumulated in the previous 1 hour (m) | Yes | Single | `dirERA5/.../TP1H/.nc` |
| **tp6h** | Total precipitation accumulated in the previous 6 hour (m) | Yes | Single | `dirERA5/Step06/.../TP6H/.nc` |
| **t850** | Temperature at 850 hPa (K) | Yes | Single | `dirERA5/Step07/.../t850/.nc` |
| **u200** | U component of wind at 200 hPa (m/s) (west-east) | Yes | Single | `dirERA5/Step07/.../u200.nc` |
| **u850** | U component of wind at 850 hPa (m/s) (west-east) | Yes | Single | `dirERA5/Step07/.../u850.nc` |
| **z500** | Geopotential at 500 hPa (m²/s²) | Yes | Single | `dirERA5/Step07/.../z500.nc` |
| **h_mvi** | Mass weighted vertical integral of moist static energy (J/m²) | Yes | Single | `dirERA5/Step11/.nc` |
| **dhdt_mvi** | Mass weighted vertical integral of dh/dt (W/m²) | Yes | Single | `dirERA5/Step11/.nc` |

*(Note: Certain Variable names and paths were reconstructed from document context for clarity.)*

---

## Important Notes & Specifications

### Resolution and Temporal Characteristics
* **Data Resolution:** The dataset is 6-hourly and 1-degree resolution, stored month by month. It is remapped from the original hourly and 0.25-degree ERA5 data on Perlmutter.
* **Instantaneous vs. Averaged/Accumulated:** All values are instantaneous **except** for:
  * Mean value in the previous 1 hour: `mslhf`, `msnlwrf`, `msnswrf`, `msshf`, `mtnlwrf`, `mtnswrf`.
  * Accumulation in the previous 1 hour: `tp1h`.
  * Accumulation in the previous 6 hour: `tp6h`.
* **Time Variant Status:** * **No:** Static and not changing with time.
  * **Yes:** Transient and changing with time.

### Levels Explained
* **Single Levels:** Represent data either at the surface, close to the surface, at a specific pressure level, or vertically integrated.
  * Specific single-level variables like `t`, `u`, `v`, and `z` are identical to their multiple-level counterparts but are isolated at specified pressure levels to enable fast data reading for certain finetuning tasks.
* **Multiple Levels:** Indicates data mapped across 29 pressure levels (hPa): 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000. 
  * *Note: A greater hPa is closer to the surface. Make sure to read the data in the vertical exactly the same order for all multiple-level variables.*

### Variables and Data Quality Warnings
* **Latent Heat Flux (`EFLX` vs `mslhf`):** Both represent surface latent heat flux. `EFLX` contains instantaneous values, while `mslhf` contains averages in the previous hour. **Recommendation:** Suggest using `mslhf` to remain consistent with upcoming MSE budget analysis.
* **Sensible Heat Flux (`SHFLX` vs `msshf`):** Both represent surface sensible heat flux. `SHFLX` contains instantaneous values, while `msshf` contains averages in the previous hour. **Recommendation:** Suggest using `msshf` to remain consistent with upcoming MSE budget analysis.
* **Precipitation (`tp1h` and `tp6h`):** * **WARNING:** The original data is missing the file `e5.accumulated_tp_1h.202206.nc`. DO NOT use this before fixing.
  * **WARNING:** The unit is inconsistent in the original data. It shows `kg/m²/s` starting from `e5.accumulated_tp_1h.202204.nc`, but uses `m` before that point (`m` is expected to be the correct unit).
* **Moist Static Energy Budget (`h_mvi` and `dhdt_mvi`):** * Available at every 6 hours, but the `dh/dt` inside the current file is based on hourly data (both `h_mvi` and `dhdt_mvi` are stored in the same `.nc` files). 
  * **Upcoming Data:** When the MSE budget analysis is fully provided, *another* `dhdt_mvi` file (likely named `dhdt_mvi_budget`) will be provided at every 6 hours where `dh/dt` will be based specifically on 6-hourly data for consistency.

### References
* Officially used in ClimaX finetuning in Nguyen et al. (2023, Table 9).
* **Citation:** Nguyen et al., 2023: *ClimaX: A foundation model for weather and climate* (https://arxiv.org/abs/2301.10343).
