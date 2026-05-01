# Task: Include Soil Type (slt)

## Objective
Update the dataloader logic to incorporate actual soil type (`slt`) data instead of a dummy tensor. The data is available at `/pscratch/sd/k/kam352/Aurora/slt/slt_data.nc` and is natively in 0.25-degree format.

## Working rules
- Read only the files listed below unless you discover a direct dependency.
- If you need to expand scope, stop and report why.
- Make the smallest possible coherent patch.
- Keep all changes confined to the modify list unless explicitly authorized.
- Ensure conda/mamba environment `aurora_mjo` is activated before running Python code (directory `/pscratch/sd/k/kam352/conda/envs/aurora_mjo/bin/python`).

## Read
- `src/dataset.py`
- `configs/phase1_baseline.yaml`
- `docs/known-gaps.md`

## Modify
- `src/dataset.py`
- `configs/phase1_baseline.yaml`
- `configs/phase2_physics.yaml`
- `configs/phase2_rollout.yaml`
- `configs/phase3_longrun.yaml`
- `docs/known-gaps.md`

## Expected output
- `src/dataset.py` updated to load `slt` using xarray from the provided path.
- `__init__` signature in `LANLMJODataset` updated to accept `slt_path`.
- Config files updated under `data` block to include `slt_path: "/pscratch/sd/k/kam352/Aurora/slt/slt_data.nc"`.
- Dummy zero-tensor logic removed from `_load_static_vars`.
- `docs/known-gaps.md` updated to remove the item about `slt` being missing.

## Steps
1. Add `slt_path: "/pscratch/sd/k/kam352/Aurora/slt/slt_data.nc"` to the `data` block of all 4 config files in `configs/`.
2. Update `src/dataset.py` `LANLMJODataset.__init__` to accept `slt_path` as an argument. Make sure it properly defaults to the config if omitted, or logs a warning similar to `root_dir`.
3. Update `_load_static_vars` in `src/dataset.py` to open the provided `slt_path`. Extract the soil type variable (usually named `slt` internally).
4. Since the `slt` data is already at 0.25-degree resolution, directly convert it to a float tensor of shape `(1, 720, 1440)` without calling `_upsample_to_aurora`. Ensure NaN values are handled if any.
5. Remove the `slt` line from `docs/known-gaps.md` indicating it is missing.

## Risks & Reminders
- Do not upsample the `slt` data since it is already natively 0.25-degree.
- Keep memory usage in mind and ensure the xarray file is closed after reading (use `with xr.open_dataset(...) as ds:`).
