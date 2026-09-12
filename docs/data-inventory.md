# Data Inventory

**Status:** Authoritative Data Lineage Catalog  
**Primary Data Archive:** NERSC Community File System (CFS)  

---

## 1. Primary Production Data: NERSC / LANL ERA5 Archive

- **Filesystem Path:**
  ```text
  /global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results
  ```
- **Access Discipline:** **STRICTLY READ-ONLY.** Owned by another research allocation (`m4946`).
- **Format:** NetCDF4, 1.0° regular lat/lon grid (180 latitude × 360 longitude), 6-hourly instantaneous (`6hrInst`), structured in `Step00/` through `Step11/` subdirectories covering 1980–2024.
- **Dynamic Upsampling:** All 1.0° fields are dynamically upsampled to Aurora's native 0.25° grid (720 × 1440) on GPU inside `src/aurora_mjo/dataset.py`.
- **Parallel Filesystem Locking:** Requires `HDF5_USE_FILE_LOCKING=FALSE` (enforced at Python startup by `src/aurora_mjo/env.py`).

### Active Production Variables
- **Surface Variables (6):**
  - `2t`: 2m air temperature (K)
  - `10u`: 10m zonal wind component (m s⁻¹)
  - `10v`: 10m meridional wind component (m s⁻¹)
  - `msl`: Surface pressure proxy (`ps`) (Pa) — *see [`docs/SPEC.md`](SPEC.md) §6*
  - `ttr`: Top net long-wave radiation flux proxy (`mtnlwrf`) (W m⁻²)
  - `tcwv`: Total column water vapor (kg m⁻²)
- **Atmospheric Variables (5 variables × 13 vertical levels):**
  - `z`: Geopotential (m² s⁻²)
  - `q`: Specific humidity (kg kg⁻¹)
  - `t`: Air temperature (K)
  - `u`: Zonal wind component (m s⁻¹)
  - `v`: Meridional wind component (m s⁻¹)
  - *Pressure levels (hPa):* 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000.
- **Static Invariant Variables (3):**
  - Surface geopotential (`z`) from `Step00/ERA5.invariant`
  - Land-sea mask (`lsm`) from `Step00/ERA5.invariant`
  - Soil type (`slt`) from `/pscratch/sd/k/kam352/Aurora/slt/slt_data.nc` (0.25° native)

---

## 2. Derived Research Artifacts

Derived data files are stored in project scratch or repository paths, never in upstream CFS:
- **RMM Targets & Basis:**
  - `data/rmm_basis.npz`: EOF eigenvectors and normalisation scalings computed from training split.
  - `data/rmm_targets.nc`: Precomputed ground-truth RMM indices across evaluation years.
- **Evaluation Outputs:**
  - `evaluation/mjo_skill/`: Anomaly correlation coefficient (ACC) and RMSE curves vs. forecast lead days.

---

## 3. Historical & Retired Datasets

- **Early Prototype (January 2015 Sample):** An early 1-month subset of ERA5 data (January 2015) was used during initial proof-of-concept testing. That dataset is retired and is not used in production training or evaluation.