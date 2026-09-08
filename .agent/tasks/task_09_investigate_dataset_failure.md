# Task: Investigate Dataset Loader Failure & Set HF_HOME

## Objective

1. **Investigate Data Failure:** Identify why `scripts/verify_dataset_loader.py` failed with `[Errno -101] NetCDF: HDF error` on the `z` invariant file (`e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100_remap_180x360MODIS.nc`). Determine whether this is a corrupted file on the NERSC cluster, a permissions issue, or a logic failure in `src/dataset.py`.
2. **Understand Success Discrepancy:** Note that the other smoke tests succeeded because `train.py --smoke-test` bypasses `src/dataset.py` completely by generating synthetic tensors in-memory, and `compute_rmm.py` successfully reads different variables (`Step01`, `Step03`).
3. **Address File Lock Crashes:** Ensure that the `HF_HOME` issue discovered in Smoke Test 2 (where Hugging Face's `filelock` crashes on the NERSC GPFS system) is permanently resolved so that it does not cause failures when running jobs on a GPU node.

## Working rules
- Follow standard execution protocols: activate the `aurora_mjo` conda environment before running.
- Ensure conda/mamba environment `aurora_mjo` is activated before running Python code (directory `/pscratch/sd/k/kam352/conda/envs/aurora_mjo/bin/python`).
- Read only the files listed below unless you discover a direct dependency.
- If you need to expand scope, stop and report why.
- Make the smallest possible coherent patch.

## Read

- `src/dataset.py` (focus on the `_load_static_vars` method)
- `slurm_scripts/train.slurm`
- `slurm_scripts/test_train.slurm`
- `docs/project-brief.md`

## Modify

- `src/dataset.py` (if a fallback or fix is required)
- `slurm_scripts/train.slurm` (if `HF_HOME` needs to be set)
- `slurm_scripts/test_train.slurm` (if `HF_HOME` needs to be set)
- `docs/known-gaps.md` (to document the invariant file issue if it is fundamentally corrupted)

## Expected output

- A definitive diagnosis of the NetCDF HDF error. If the file is corrupted, update `docs/known-gaps.md` to note this and implement a fallback in `src/dataset.py` (e.g., using `slt` or a dummy tensor for `z` if necessary, or skipping it if it's purely a path error).
- `HF_HOME=/tmp/hf_home_${USER}` is explicitly set and exported in all Slurm submission scripts (`train.slurm`, `test_train.slurm`, `eval.slurm`) to prevent GPFS file-lock errors during Hugging Face cache access.

## Steps

1. **Verify the Problematic File:** Use the terminal (`ncdump -h` or a quick Python script with `xarray.open_dataset`) to attempt to read `/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results/Step00/ERA5.invariant/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100_remap_180x360MODIS.nc`.
2. **Diagnose and Fix:** If the file is corrupted, document this in `docs/known-gaps.md` and implement a temporary workaround in `src/dataset.py`'s `_load_static_vars` so that training is not permanently blocked. If the file is fine, fix the loading logic in `src/dataset.py` that caused the error.
3. **Fix `HF_HOME`:** Open `slurm_scripts/train.slurm` and `slurm_scripts/eval.slurm` (and `test_train.slurm`) and ensure that `export HF_HOME=/tmp/hf_home_${USER}` is uncommented, active, and placed before the execution command.
4. **Final Verification:** Run `export HF_HOME=/tmp/hf_home_${USER}; python scripts/verify_dataset_loader.py` again. It must succeed and print the dataset variables correctly.
