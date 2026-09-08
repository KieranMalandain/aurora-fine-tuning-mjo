# Comprehensive Smoke Test Results

This document contains the execution results of the comprehensive smoke test suite for the Aurora MJO project.

## 1. Main Training Pipeline
- **Command:** `python train.py --config configs/phase1_baseline.yaml --smoke-test`
- **Timestamp:** 2026-05-01 12:21:41
- **Status:** **PASS**
- **Output Snippet:**
  ```text
  2026-05-01 12:21:38,980 | INFO | __main__ | Starting training: experiment=phase1_baseline epochs=1 device=cpu
  2026-05-01 12:21:40,033 | INFO | src.trainer | Epoch 001 | VAL loss=4752.8906
  2026-05-01 12:21:41,811 | INFO | src.trainer | Saved checkpoint: checkpoints/phase1_baseline/epoch_001_val4752.8906.pt
  2026-05-01 12:21:41,811 | INFO | __main__ | Done.
  ```

## 2. MJO Head Architecture
- **Command:** `export HF_HOME=/tmp/hf_home_${USER}; python scripts/smoke_test_mjo_head.py`
- **Timestamp:** 2026-05-01 12:22:45
- **Status:** **PASS** (Note: requires HF_HOME in `/tmp` to avoid filelock errors on the NERSC GPFS file system)
- **Output Snippet:**
  ```text
  Test 2: head enabled
  ...
  mjo_head                        trainable=    33,795 /     33,795

  [PASS] returned (Batch, Tensor[1, 3])
  MJO prediction: RMM1=0.0000  RMM2=0.0000  Amp=0.0000
  ...
  All smoke tests passed.
  ```

## 3. Parameter Freezing Logic
- **Command:** `export HF_HOME=/tmp/hf_home_${USER}; python scripts/smoke_test_freeze.py`
- **Timestamp:** 2026-05-01 12:23:14
- **Status:** **PASS**
- **Output Snippet:**
  ```text
  PASS [1] backbone has 336 frozen parameter tensors.
  PASS [2] 160 LoRA adapter param tensors are trainable.
  PASS [3] injected variable embeddings ['ttr', 'tcwv'] are trainable.
  ...
  All assertions passed. Freezing logic is correct.
  ```

## 4. Autoregressive Rollout Logic
- **Command:** `export HF_HOME=/tmp/hf_home_${USER}; python scripts/smoke_test_rollout.py`
- **Timestamp:** 2026-05-01 12:23:38
- **Status:** **PASS**
- **Output Snippet:**
  ```text
  INFO Running _compute_loss with rollout k=2 …
  INFO Loss breakdown: total=0.338511  grid=0.338511  spectral=0.000000  mjo_head=0.000000
  INFO Verifying backward pass …
  INFO Gradient on dummy_param: -0.000000
  INFO === Smoke test PASSED ===
  ```

## 5. NERSC Dataset Loader Verification
- **Command:** `export PYTHONPATH=.; export HF_HOME=/tmp/hf_home_${USER}; python scripts/verify_dataset_loader.py`
- **Timestamp:** 2026-05-01 12:24:22
- **Status:** **FAIL**
- **Output Snippet:**
  ```text
  Initializing dataset for year 1980...
  Initializing LANL MJO Dataset (1980-1980)...
  Failed to initialize dataset: [Errno -101] NetCDF: HDF error: '/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results/Step00/ERA5.invariant/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100_remap_180x360MODIS.nc'
  ```

## 6. RMM Basis Computation
- **Command:** `export PYTHONPATH=.; export HF_HOME=/tmp/hf_home_${USER}; python scripts/compute_rmm.py --smoke-test`
- **Timestamp:** 2026-05-01 12:24:43
- **Status:** **PASS**
- **Output Snippet:**
  ```text
  === Smoke Test: Verifying pipeline logic with synthetic data ===

  Training matrix shape : (13140, 3)
  EOF1 shape            : (3,)
  ...
  Amplitude mean (train): 1.041
  Active MJO fraction  : 54.8%

  === Smoke Test PASSED ===
  ```

## 7. MJO Evaluation Metrics
- **Command:** `export PYTHONPATH=.; export HF_HOME=/tmp/hf_home_${USER}; python scripts/evaluate_mjo.py --smoke-test`
- **Timestamp:** 2026-05-01 12:25:05
- **Status:** **PASS**
- **Output Snippet:**
  ```text
  === Smoke Test: evaluate_mjo.py ===

  bivariate_acc (noisy perfect forecast): 0.9504  (expected > 0.8)
  RMSE RMM1=0.3372  RMM2=0.2773
  Amplitude error=-0.0368  Phase error=16.88°
  project_fields_to_rmm → RMM1=0.2741  RMM2=0.5584
  load_basis OK  eof1=[0.6092077  0.50767308 0.6092077 ]
  save_summary_csv OK  shape=(30, 7)
  save_skill_plots OK  wrote 3 PNG(s)
  ...
  === Smoke Test PASSED ===
  ```

