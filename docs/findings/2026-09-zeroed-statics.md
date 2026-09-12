# Zeroed Static Invariant Fields in ERA5 Dataset Loader

**Date:** 2026-09-11  
**Scope:** `src/dataset.py`, `docs/findings/2026-09-zeroed-statics.md`, `results/B2_result.md`  
**Status:** UNRESOLVED — logs purged  

---

## 1. Symptom

In `src/dataset.py::_load_static_vars` (lines 346–365), invariant loads for surface geopotential (`z`) and land-sea mask (`lsm`) were wrapped in broad exception handlers:

```python
try:
    with xr.open_dataset(z_files[0], engine="netcdf4") as ds_z:
        z_arr = ds_z["Z"].values
    z_tensor = _upsample_to_aurora(_clean(torch.from_numpy(z_arr).float()))
except Exception as e:
    warnings.warn(
        f"[LANLMJODataset] Failed to load invariant Z: {e}. Using zeros."
    )
    z_tensor = torch.zeros(720, 1440)

try:
    with xr.open_dataset(lsm_files[0], engine="netcdf4") as ds_lsm:
        lsm_arr = ds_lsm["LSM"].values
    lsm_tensor = _upsample_to_aurora(_clean(torch.from_numpy(lsm_arr).float()))
except Exception as e:
    warnings.warn(
        f"[LANLMJODataset] Failed to load invariant LSM: {e}. Using zeros."
    )
    lsm_tensor = torch.zeros(720, 1440)
```

If an exception occurred during invariant loading, execution did not halt. Instead, the dataloader silently substituted all-zero tensors (`torch.zeros(720, 1440)`) for topography and land-sea boundaries, emitting only a `UserWarning`. Training would proceed on a planet with flat topography ($Z = 0$) and 100% ocean ($LSM = 0$).

Note: `slt` (soil type from `slt_data.nc`) carried no such fallback; any failure to load `slt` raised an unhandled exception immediately.

---

## 2. Root cause

The root cause of invariant load failures is parallel filesystem advisory file locking on NERSC Community File System (CFS) mounts (`/global/cfs/cdirs/m4946/...`).

When `xarray.open_dataset(..., engine="netcdf4")` attempts to open an HDF5-backed NetCDF4 file on CFS without disabling file locking, the underlying HDF5 library fails with:

```text
OSError: [Errno -101] NetCDF: HDF error: '.../Step00/ERA5.invariant/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100_remap_180x360MODIS.nc'
```

Setting `export HDF5_USE_FILE_LOCKING=FALSE` in the process environment completely disables HDF5 advisory locking and allows the files to open cleanly.

---

## 3. Evidence

### 3.1 Directory Search and Log Availability

A forensic search was conducted across `/pscratch/sd/k/kam352/` and `/global/homes/k/kam352/` to locate all historical SLURM job execution logs from July 2026:

| Directory Path | Expected? | Status / Contents |
| --- | --- | --- |
| `/pscratch/sd/k/kam352/Aurora/aurora-fine-tuning-mjo/slurm_logs/` | Yes | **Missing** (`No such file or directory`) |
| `/pscratch/sd/k/kam352/Aurora/worktrees/campaign-fixes/slurm_logs/` | Yes | **Missing** (Worktree pruned in task 09 consolidation) |
| `/pscratch/sd/k/kam352/Aurora/worktrees/human-integration/slurm_logs/` | Yes | **Missing** (Worktree pruned in task 09 consolidation) |
| `/pscratch/sd/k/kam352/Aurora/worktrees/agent-simul-training-one/slurm_logs/` | Yes | **Missing** (Worktree pruned in task 09 consolidation) |
| `/pscratch/sd/k/kam352/Aurora/worktrees/` | Yes | Exists, **empty** (8 bytes, `.` and `..`) |
| `/pscratch/sd/k/kam352/Aurora/cleanup-reports/` | Yes | Exists; contains consolidation reports 01–09 and dumps |
| `/pscratch/sd/k/kam352/Aurora/cleanup-reports/_dumps-06/` | Yes | Exists; 9 `.out` git-diff inspection files from 2026-09-08 |
| `/pscratch/sd/k/kam352/Aurora/cleanup-preserve/` | Yes | Exists; contains `untracked/` catalogs and `patches/` from task 05 |
| `/pscratch/sd/k/kam352/hf_home/xet/logs/` | Unanticipated | Exists; 2 Hugging Face client `.log` files from 2026-07-10 and 2026-07-14 |
| `/global/homes/k/kam352/aurora-backup/` | Yes | Exists; git bundles and env exports; no slurm logs |
| `/global/homes/k/kam352/` | Yes | No batch log files found |

The July 2026 batch execution logs were not preserved on disk because:
1. Interactive `salloc` jobs streamed standard output and standard error directly to the interactive terminal rather than redirecting to log files.
2. The single July batch job (`55806263`) ran in `/pscratch/sd/k/kam352/Aurora/worktrees/campaign-fixes`. That directory was pruned during the September 8 git consolidation (task 09), and `slurm_logs/` was in `.gitignore`, meaning task 05 preservation did not capture it.

### 3.2 Target String Search Across All Preserved Repositories and Dumps

Searching for all five target search strings across all available files on disk yielded:

```text
=== TARGET: /pscratch/sd/k/kam352/Aurora/cleanup-reports ===
--- 1. Using zeros ---
/pscratch/sd/k/kam352/Aurora/cleanup-reports/build_09_report.py:4
/pscratch/sd/k/kam352/Aurora/cleanup-reports/09-consolidated.md:4
/pscratch/sd/k/kam352/Aurora/cleanup-reports/06-content-comparison.md:2
/pscratch/sd/k/kam352/Aurora/cleanup-reports/_dumps-06/step1_dataset.txt:2
/pscratch/sd/k/kam352/Aurora/cleanup-reports/_dumps-06/step01.out:2
/pscratch/sd/k/kam352/Aurora/cleanup-reports/_dumps-06/S_vs_K_full.diff:2
--- 2. Failed to load invariant ---
/pscratch/sd/k/kam352/Aurora/cleanup-reports/build_09_report.py:4
/pscratch/sd/k/kam352/Aurora/cleanup-reports/09-consolidated.md:4
/pscratch/sd/k/kam352/Aurora/cleanup-reports/06-content-comparison.md:4
/pscratch/sd/k/kam352/Aurora/cleanup-reports/_dumps-06/step09_dataset_diff.txt:2
/pscratch/sd/k/kam352/Aurora/cleanup-reports/_dumps-06/step1_dataset.txt:4
/pscratch/sd/k/kam352/Aurora/cleanup-reports/_dumps-06/step01.out:4
/pscratch/sd/k/kam352/Aurora/cleanup-reports/_dumps-06/S_vs_K_full.diff:4
--- 3. NetCDF: HDF error ---
/pscratch/sd/k/kam352/Aurora/cleanup-reports/build_09_report.py:3
/pscratch/sd/k/kam352/Aurora/cleanup-reports/09-consolidated.md:3
--- 4. Errno -101 ---
/pscratch/sd/k/kam352/Aurora/cleanup-reports/build_09_report.py:3
/pscratch/sd/k/kam352/Aurora/cleanup-reports/09-consolidated.md:3
--- 5. HDF5_USE_FILE_LOCKING ---
/pscratch/sd/k/kam352/Aurora/cleanup-reports/06-content-comparison.md:3
/pscratch/sd/k/kam352/Aurora/cleanup-reports/_dumps-06/step06.out:3
/pscratch/sd/k/kam352/Aurora/cleanup-reports/_dumps-06/S_vs_K_full.diff:3
```

All occurrences in `06-content-comparison.md` and `_dumps-06/` are static code listings and diffs of `src/dataset.py` and `slurm_scripts/train_auto.slurm`.

The **only** instance where `Using zeros` was logged during code execution occurred on **September 8, 2026** during task 09 of the git consolidation (`09-consolidated.md:600–603` and `build_09_report.py:343–346`):

```text
Initializing LANL MJO Dataset (2016-2016) [timestamp-aligned v3]...
/pscratch/sd/k/kam352/Aurora/worktrees/consolidation/src/dataset.py:318: UserWarning: [LANLMJODataset] Failed to load invariant Z: [Errno -101] NetCDF: HDF error: '/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results/Step00/ERA5.invariant/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100_remap_180x360MODIS.nc'. Using zeros.
  warnings.warn(f"[LANLMJODataset] Failed to load invariant Z: {e}. Using zeros.")
/pscratch/sd/k/kam352/Aurora/worktrees/consolidation/src/dataset.py:326: UserWarning: [LANLMJODataset] Failed to load invariant LSM: [Errno -101] NetCDF: HDF error: '/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results/Step00/ERA5.invariant/e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100_remap_180x360MODIS.nc'. Using zeros.
  warnings.warn(f"[LANLMJODataset] Failed to load invariant LSM: {e}. Using zeros.")
Traceback (most recent call last):
  File "/pscratch/sd/k/kam352/conda_envs/aurora_mjo/lib/python3.10/site-packages/xarray/backends/file_manager.py", line 211, in _acquire_with_cache_info
    file = self._cache[self._key]
...
  File "/pscratch/sd/k/kam352/Aurora/worktrees/consolidation/src/dataset.py", line 217, in _build_aligned_index
    with xr.open_dataset(str(f), engine="netcdf4", decode_times=True) as ds:
...
OSError: [Errno -101] NetCDF: HDF error: '/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results/Step02/ERA5.remap_180x360MODIS_6hrInst/T2/e5.oper.an.sfc.128_167_2t.ll025sc.201601_remap_180x360.nc'
```

### 3.3 Deductive Proof of Execution Integrity

The traceback above demonstrates a crucial structural property of `src/dataset.py`:
1. `self.static_vars = self._load_static_vars()` runs first (lines 169).
2. Immediately afterward, `self._build_aligned_index()` runs (line 176+). In line 217, it opens raw CFS ERA5 NetCDF files using `xr.open_dataset(str(f), engine="netcdf4")`.
3. There is **no** exception handler around `_build_aligned_index()`.
4. Therefore, if `HDF5_USE_FILE_LOCKING=FALSE` is absent, invariant loading fails with `[Errno -101]` and prints `Using zeros`, but milliseconds later, `_build_aligned_index()` crashes unconditionally with `OSError: [Errno -101] NetCDF: HDF error`.
5. A training job **cannot draw a single training batch or execute step 0** if `HDF5_USE_FILE_LOCKING=FALSE` is omitted.

### 3.4 SLURM Accounting Correlation with Metric Traces

Cross-referencing SLURM accounting (`sacct`) against the preserved metrics traces in `docs/archive/metrics/`:

| Job ID | Type | Start Time (PDT) | Node | Metric Trace File | Step / Epoch Progress |
| --- | --- | --- | --- | --- | --- |
| `55755699.6` | Interactive (`salloc`) | 2026-07-10 10:40:38 | `nid008565` | `baseline-2026-07.jsonl:1` | Step 0 completed (loss 28.144 at 10:41:48 PDT) |
| `55830479.0` | Interactive (`salloc`) | 2026-07-12 13:44:24 | `nid008304` | `lora-2026-07.jsonl:1–2` | Step 0 completed (loss 6.896 at 13:55:56 PDT); val step 10 evaluated 200 batches |
| `55806263.0` | Batch (`sbatch`) | 2026-07-13 15:12:39 | `nid008228` | `baseline-2026-07.jsonl:2–77` | **3,750 steps completed** across 3 epochs (15:22:55 to 16:44:26 PDT) |

Findings from this correlation:
1. Batch job `55806263` ran `slurm_scripts/train_auto.slurm`. Inspection of that script at commit `b9ffe63` proves that line 58 explicitly set `export HDF5_USE_FILE_LOCKING=FALSE`.
2. Batch job `55806263` executed for 1 hour 32 minutes and completed 3,750 training steps. It could not have run past dataset initialization without `HDF5_USE_FILE_LOCKING=FALSE`.
3. Because `HDF5_USE_FILE_LOCKING=FALSE` was active, invariant files `128_129_z` and `128_172_lsm` opened cleanly without error, meaning `_load_static_vars()` loaded true topography and land-sea mask fields.

---

## 4. What this RULES OUT

| Hypothesis | Evidence against |
| --- | --- |
| **July 2026 training runs trained on zero topography and zero land-sea mask (`Using zeros`)** | **Deductive structural proof.** If `HDF5_USE_FILE_LOCKING=FALSE` were missing, `_build_aligned_index()` would have crashed on `[Errno -101]` during dataset initialization before any batch was drawn (observed in `09-consolidated.md:639`). Both `baseline-2026-07.jsonl` (3,750 steps) and `lora-2026-07.jsonl` (step 0 and 200 val batches) successfully progressed past initialization. Furthermore, `train_auto.slurm` at commit `b9ffe63` explicitly set `HDF5_USE_FILE_LOCKING=FALSE` at line 58. |
| **Zeroed statics contributed to the July 100% non-finite validation loss** | The non-finite validation loss is fully explained by Lesson 1: `msl` channel fed surface pressure `ps` normalised with MSL constants (`location=100958, scale=1332`), producing −36 σ inputs across high terrain globally (`AURORA_MJO_GAMEPLAN.md` Finding 1). |
| **`Using zeros` occurred silently during normal batch runs** | The only observed firing of `Using zeros` occurred during task 09 interactive testing without the locking environment variable, where it was immediately accompanied by a fatal NetCDF crash on the next line. |

---

## 5. The fix

1. **Remove permissive exception handling (Task E1):**  
   In `src/aurora_mjo/dataset.py::_load_static_vars`, replace `try...except Exception as e` with strict, unhandled errors that fail immediately upon unreadable invariant files:
   ```python
   # Do not catch Exception and substitute zeros.
   # If HDF5 locking fails, raise with an explicit message pointing to HDF5_USE_FILE_LOCKING.
   ```
2. **Set environment variable at Python startup (Task E1):**  
   In `src/aurora_mjo/env.py`, enforce `os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"` at process initialization before importing `xarray` or `netCDF4`, ensuring interactive sessions and CLI entry points cannot accidentally run without it.

---

## 6. Open

1. **Relocate `slt_data.nc` (Q-07):**  
   Soil type data currently lives on purgeable scratch space at `/pscratch/sd/k/kam352/Aurora/slt/slt_data.nc`. While `slt` never had a zero-substitution fallback (it raises if missing), moving it to durable storage and verifying `scripts/download_slt.py` remains an open item.
