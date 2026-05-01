# Task: Dataset Loader Implementation

## Objective
Harden `src/dataset.py` by verifying assumptions against the NERSC ERA5 dataset, fixing the xarray multi-file loading call, and narrowing the baseline variable set based on the provided diagnostic.

## Working rules
- Read only the files listed below unless you discover a direct dependency.
- If you need to expand scope, stop and report why.
- Make the smallest possible coherent patch.
- Keep all changes confined to the modify list unless explicitly authorized.
- Ensure conda/mamba environment `aurora_mjo` is activated before running Python code (directory `/pscratch/sd/k/kam352/conda/envs/aurora_mjo/bin/python`).

## Read
- `src/dataset.py`
- `docs/nersc-dataset-information.md`

## Modify
- `src/dataset.py`

## Do not touch
- Model architecture (`models/`)
- Training loop (`train.py` or `training/`)

## Expected output
- `src/dataset.py` updated with correct native variable names, fixed xarray loading, and restricted baseline variables.
- A report detailing exactly which variable naming and path assumptions changed.

## Steps
1. **Inspect Mapped Variables**: For each mapped variable family in `src/dataset.py`, inspect one real file on NERSC (e.g., using `ncdump -h <file>` or `xr.open_dataset` in a temporary script). Confirm the directory path, filename pattern, internal data variable name, vertical coordinate, and time dtype.
2. **Correct Path/Variable Assumptions**: Update `src/dataset.py` so that native variable names and directory structures match reality instead of guesses (especially for invariant `Z`, `LSM`, and physical fields). Do not leave guessed names.
3. **Fix Xarray Loading**: Replace `xr.open_mfdataset(..., combine="by_coords", concat_dim="time")` so it does **not** pass `concat_dim` together with `combine="by_coords"`. If necessary, use `combine="nested"` based on real-file testing.
4. **Harden Static Loading**: Replace placeholder static-file loading with actual verified globbing for `Z` and `LSM`. Remove silent fallback zero-arrays for NERSC training.
5. **Keep Baseline Minimal**: Retain only variables needed for Aurora MJO fine-tuning (`z`, `q`, `t`, `u`, `v`, `2t`, `10u`, `10v`, `msl`, `tcwv`, `ttr`, static `z`, static `lsm`). Comment out physics variables (`EFLX`, `tp6h`, `SHFLX`, `mslhf`, `msshf`, `tp1h`, etc.).
6. **Explicit Proxies**: Ensure `Ps` used for Aurora `msl` is clearly documented as a temporary proxy, and `mtnlwrf` used for `ttr` is documented as a compatibility mapping.

## Risks & Reminders
- Do not mix `combine="by_coords"` with `concat_dim="time"`.
- Do not assume `LSM` or `ps` capitalization until confirmed on disk.
- Keep memory usage low during file inspection.
