# Task: Dataset Loader Verification

## Objective
Verify that the updated `src/dataset.py` correctly loads the baseline NERSC ERA5 variables without crashing or using placeholder arrays.

## Working rules
- Read only the files listed below unless you discover a direct dependency.
- If you need to expand scope, stop and report why.
- Make the smallest possible coherent patch.
- Keep all changes confined to the modify list unless explicitly authorized.
- Ensure conda/mamba environment `aurora_mjo` is activated before running Python code (directory `/pscratch/sd/k/kam352/conda/envs/aurora_mjo/bin/python`).

## Read
- `src/dataset.py`

## Modify
- `scripts/verify_dataset_loader.py` (New file)

## Do not touch
- `src/dataset.py` (unless a bug is found during testing)
- Training scripts or model configs

## Expected output
- A simple test script `scripts/verify_dataset_loader.py` that successfully instantiates the dataset and loads one batch.
- Terminal output confirming the exact shape and variables returned by the loader.
- Full update to `docs/data-inventory.md` with the discovered information.

## Steps
1. Create `scripts/verify_dataset_loader.py`.
2. Import the dataset class from `src.dataset`.
3. Instantiate the dataset pointing to the NERSC data directory for a small time slice (e.g., one month of one year).
4. Fetch a single item (`__getitem__` or via a PyTorch `DataLoader`).
5. Print the shape and available variables in the returned tensor/dictionary.
6. Run the script on a compute node or interactive session on NERSC.

## Risks & Reminders
- This test should only load a small time slice to avoid memory/OOM issues on a login node.
- Ensure that the static `Z` and `LSM` variables are actually populated, and not filled with zeros from a fallback path.
