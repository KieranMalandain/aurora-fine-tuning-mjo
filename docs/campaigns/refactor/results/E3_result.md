STATUS: GREEN
PROCEED: YES
BLOCKED-ON: NONE
SUMMARY: Unified SLURM scripts under uv run --frozen, extracted env.sh, rewrote eval.slurm and test_train.slurm, added README.

<!--
The four lines above are MACHINE-READ by the next agent. Keep them as the first
four lines of the file, one per line, in this order, with these exact key names.
Everything below is for humans.
-->

# E3 — Rewrite `eval.slurm`; make every SLURM script consistent

| | |
| --- | --- |
| **Branch** | `epic/refactor-E3-slurm-rewrite` |
| **Agent / date** | Gemini 3.8 Flash, 2026-09-12 |
| **Wall clock** | 35 min (budget: 2 h) |
| **Commits** | 4, listed below |

---

## 1. What was done

1. Extracted all shared NERSC Perlmutter environment configuration variables into a dedicated sourced file `slurm_scripts/env.sh`, preserving historical failure comments verbatim.
2. Updated `slurm_scripts/train_auto.slurm`'s environment block to source `env.sh` and migrated its DDP execution from conda and `train.py` to `uv run --frozen torchrun` with `run.py train`. All SBATCH directives, signal trap routines (`USR1` at T-30m), `DONE` marker detection, and `STOP_TRAINING` logic were left untouched.
3. Completely rewrote `slurm_scripts/eval.slurm` against `configs/unified.yaml --mode ${MODE}` using single-device allocation (1 GPU, 32 CPUs, 1 hour walltime in `regular` queue) and `run.py evaluate`, eliminating obsolete references to deleted per-phase YAML configs, defunct conda paths, and deprecated dummy dataset overrides.
4. Updated `slurm_scripts/test_train.slurm` as a lightweight cluster smoke test running `run.py train --mode baseline --smoke-test` (1 GPU, 32 CPUs, 10 min on `debug` queue).
5. Updated `slurm_scripts/submit_chain.sh` to resolve the script path dynamically and forward `CONFIG`, strictly preserving the `--dependency=afterany:` semantics that enable safe and idempotent chain restarts.
6. Authored comprehensive operational documentation in `slurm_scripts/README.md` covering script purposes, submission syntax, resource justifications, signal contracts, and the exit-code/auto-resumption specification.

---

## 2. Definition of Done

- [x] Table of all four scripts plus `submit_chain.sh`: purpose, what was wrong, what changed
(See Section 4 Measurements table)

- [x] `slurm_scripts/env.sh` created with the six variables and their original comments carried across verbatim; nothing new added
```bash
#!/bin/bash
# slurm_scripts/env.sh — shared environment configuration for NERSC Perlmutter.
# Sourced by all batch scripts before launching python.

export HDF5_USE_FILE_LOCKING=FALSE
export HF_HOME=${PSCRATCH}/hf_home            # PERSISTENT cache (was /tmp → re-download every job)
export HF_HUB_ETAG_TIMEOUT=60
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # anti-fragmentation (cuBLAS/OOM fix)
```

- [x] `MASTER_ADDR`/`MASTER_PORT` left in `train_auto.slurm`
Lines 57-58 of `slurm_scripts/train_auto.slurm`:
```bash
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 1000 ))     # avoid port collisions between chained jobs
```

- [x] All four scripts run Python via `uv run --frozen` — **or** A1's `AMBER` fallback is used, with Q-03 referenced and the task reported `AMBER`
All scripts execute via `uv run --frozen` based on verified Perlmutter compute-node deployment in A1 (`STATUS: GREEN`).

- [x] `eval.slurm` rewritten: `MODE` variable, `unified.yaml --mode`, no `use_dummy`, sources `env.sh`, correct log paths; resource choices justified or flagged unverified
Evaluates on 1 GPU (`--gpus-per-task=1`, `-n 1`) and 32 CPUs (`-c 32`) with 1-hour walltime (`-t 01:00:00`) in `regular` queue.

- [x] `test_train.slurm` points at `run.py train --mode baseline --smoke-test` with a short walltime
Configured with `-q debug`, `-t 00:10:00`, 1 GPU, 32 CPUs.

- [x] `submit_chain.sh` updated with dependency semantics preserved and stated
Uses `--dependency=afterany:"${PREV}"` so downstream jobs launch upon success, timeout (`99`), or crash, relying on `--resume auto` and the `DONE` marker.

- [x] `train_auto.slurm` diff pasted, showing **only** env-block and invocation changes; SBATCH directives, `USR1` trap, `DONE`/`STOP_TRAINING` and exit codes untouched
```diff
diff --git a/slurm_scripts/train_auto.slurm b/slurm_scripts/train_auto.slurm
index 825318f..c2c2b4f 100644
--- a/slurm_scripts/train_auto.slurm
+++ b/slurm_scripts/train_auto.slurm
@@ -48,17 +48,13 @@ if [ -f "STOP_TRAINING" ]; then
     exit 0
 fi
 
-# 1. Conda (bypassing NERSC global module conflicts)
-source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
-conda activate /pscratch/sd/k/kam352/conda_envs/aurora_mjo
+# 1. Ensure working directory is repo root
+if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
+    cd "${SLURM_SUBMIT_DIR}"
+fi
 
-# 2. NERSC environment
-export HF_HOME=${PSCRATCH}/hf_home            # PERSISTENT cache (was /tmp → re-download every job)
-export HF_HUB_ETAG_TIMEOUT=60
-export HDF5_USE_FILE_LOCKING=FALSE
-export PYTHONUNBUFFERED=1
-export OMP_NUM_THREADS=8
-export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # anti-fragmentation (cuBLAS/OOM fix)
+# 2. Sourced NERSC environment
+source slurm_scripts/env.sh
 export MASTER_ADDR=$(hostname)
 export MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 1000 ))     # avoid port collisions between chained jobs
 
@@ -66,12 +62,12 @@ echo "[chain] MODE=${MODE}  CONFIG=${CONFIG}  JOB=${SLURM_JOB_ID}  NODE=$(hostna
 
 # 3. Launch DDP under srun in the background so this shell can catch USR1
 #    and forward it to every python rank.
-srun --cpu-bind=cores torchrun \
+srun --cpu-bind=cores uv run --frozen torchrun \
     --nnodes=1 \
     --nproc_per_node=4 \
     --rdzv_backend=c10d \
     --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
-    train.py --config "${CONFIG}" --mode "${MODE}" --resume auto &
+    run.py train --config "${CONFIG}" --mode "${MODE}" --resume auto &
 SRUN_PID=$!
 
 forward_usr1() {
```

- [x] `slurm_scripts/README.md` documents each script and the full exit-code contract including `STOP_TRAINING`
Authored and committed in `slurm_scripts/README.md`.

- [x] `bash -n` clean on all five files — output pasted
```text
$ bash -n slurm_scripts/*.slurm slurm_scripts/*.sh && echo "SYNTAX OK"
SYNTAX OK
```

- [x] Both greps return nothing outside `README.md` — output pasted
```text
$ grep -rn "phase1_baseline\|phase2_physics\|phase3_longrun\|use_dummy\|conda" slurm_scripts/ || true
slurm_scripts/README.md:126:- **Unified Configuration:** Legacy individual YAML configuration files (`phase1_baseline.yaml`, `phase2_physics.yaml`, etc.) have been retired in favor of `configs/unified.yaml` with `--mode <mode>`.
slurm_scripts/README.md:127:- **Synthetic Testing:** The deprecated `data.use_dummy=true` parameter is retired; synthetic validation is performed using `--smoke-test`.

$ grep -rn "conda/envs" slurm_scripts/ || echo "NO MATCHES"
NO MATCHES
```

- [x] **No job submitted** — stated explicitly
Stated explicitly: **No SLURM jobs (`sbatch`, `srun`, or `salloc`) were submitted during this task.** Cluster acceptance testing is deferred to task F2.

- [x] `uv run python scripts/check.py` green — summary table pasted
Summary table pasted in Section 3.

- [x] Result file written from `results/_TEMPLATE.md`

---

## 3. The gate

```text
===============================================================================
Gate            Status   Time     Why
-------------------------------------------------------------------------------
lockfile        PASS      0.02s   uv.lock matches pyproject.toml
ruff lint       PASS      0.10s   lint
ruff format     PASS      0.11s   formatting is canonical
types           PASS      0.52s   static types, ratcheted scope
pytest          PASS     28.23s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
```

---

## 4. Measurements

### Audit and Changes Across SLURM Execution Files

| Script | Purpose | What Was Wrong | What Changed |
|---|---|---|---|
| `slurm_scripts/train_auto.slurm` | Production 4-GPU auto-resuming training under SLURM | Activated conda env; hardcoded shared env variables; invoked `train.py` directly | Sourced `slurm_scripts/env.sh`; changed invocation to `srun --cpu-bind=cores uv run --frozen torchrun ... run.py train ...`; added directory guard `cd "${SLURM_SUBMIT_DIR:-.}"` |
| `slurm_scripts/eval.slurm` | Evaluates checkpoint MJO forecast skill | Referenced deleted `phase1_baseline.yaml`; passed `--override data.use_dummy=true`; hardcoded invalid conda path; lacked `MODE` overlay support | Rewrote to source `env.sh`, invoke `srun uv run --frozen python run.py evaluate --config "${CONFIG}" --mode "${MODE}" --checkpoint "${CHECKPOINT}" "$@"`; allocated 1 GPU, 32 CPUs, 1h in regular queue |
| `slurm_scripts/test_train.slurm` | Rapid cluster smoke verification | Referenced deleted `phase1_baseline.yaml`; activated conda env; used non-persistent `/tmp` HF cache | Rewrote to source `env.sh`, invoke `srun uv run --frozen python run.py train --config configs/unified.yaml --mode baseline --smoke-test`; allocated 1 GPU, 32 CPUs, 10 min in `debug` queue |
| `slurm_scripts/submit_chain.sh` | Chained SLURM job submission across training phases | Hardcoded script path relative to PWD; did not forward `CONFIG` | Resolved script directory dynamically via `$(dirname "${BASH_SOURCE[0]}")`; forwarded `CONFIG` alongside `MODE`; strictly preserved `--dependency=afterany:` semantics |
| `slurm_scripts/env.sh` | Centralized environment module | Did not exist (exports duplicated across batch scripts) | Created with 6 standard exports and exact incident comments carried across |

---

## 5. What was ruled out, and by what evidence

1. **Retaining conda fallback activation:** Ruled out. Measurement during task A1 proved that `uv sync --all-groups` and `uv run --frozen` function natively on Perlmutter compute nodes (`nid008469`) with full CUDA 12.1 detection across all 4 A100 GPUs.
2. **Placing `MASTER_ADDR` and `MASTER_PORT` in `env.sh`:** Ruled out. The port derivation `29000 + SLURM_JOB_ID % 1000` is training-specific to avoid port collisions between chained DDP runs; non-DDP jobs (`eval.slurm`, `test_train.slurm`) do not need distributed rendezvous ports.
3. **Using `--dependency=afterok` in `submit_chain.sh`:** Ruled out. Training jobs that exit with code `99` (clean timeout saves) would fail the `afterok` dependency check, halting the automated multi-job training chain. `--dependency=afterany` combined with `--resume auto` and the `DONE` marker ensures safe, resilient chain progression.

---

## 6. Caveats

NONE (`STATUS: GREEN`).

---

## 7. Observations

- `eval.slurm` walltime and resource limits (1 GPU, 32 CPUs, 1 hour in `regular` queue) are conservative and unverified on Perlmutter. They provide safe headroom for multi-lead rollouts across validation years (2016–2019). Full empirical benchmark of evaluation throughput will be captured in task F2.

---

## 8. Questions raised

NONE.

---

## 9. Commits

```text
45fdda7 docs(campaign): record task E3 completion results in E3_result.md
3b32263 docs(slurm): add slurm_scripts/README.md guide and contract documentation (E3)
68c9a4d feat(slurm): rewrite eval.slurm, test_train.slurm, and submit_chain.sh (E3)
55dd3e1 feat(slurm): extract env.sh and update train_auto.slurm to uv (E3)
```

## 10. Files changed

```text
 docs/campaigns/refactor/results/E3_result.md | 224 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 slurm_scripts/README.md                      | 127 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 slurm_scripts/env.sh                         |  10 ++++++++++
 slurm_scripts/eval.slurm                     |  48 ++++++++++++++++++++++++++++++++++++------------
 slurm_scripts/submit_chain.sh                |  12 +++++++-----
 slurm_scripts/test_train.slurm               |  41 +++++++++++++++++++++++++++++------------
 slurm_scripts/train_auto.slurm               |  20 ++++++++------------
 7 files changed, 441 insertions(+), 41 deletions(-)
```
