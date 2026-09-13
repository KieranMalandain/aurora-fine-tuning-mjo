# Aurora–MJO: Campaign Synthesis and Forward Handoff

**Status:** draft handoff note, September 2026. Written to prime a fresh working context.
Not authoritative — the "Recommended next steps" section in particular is a synthesis and
proposal to be decided by the human.

**Base branch:** `epic/refactor`  
**Base commit:** `85b0db5` (Task F2 complete and accepted; Task F3 added)  
**Hardware target:** NERSC Perlmutter, 1 node × 4 × NVIDIA A100 80GB SXM4  

This document is a synthesis and forward plan. It is explicitly **not** the source of truth.
Durable truth lives in three places:

| Truth | Lives in |
| :--- | :--- |
| What the system is and what it guarantees | `docs/SPEC.md` |
| What the code actually does | `tests/` |
| What was measured, by whom, when | `docs/campaigns/refactor/results/` |

---

## 2. What the project is

`aurora-fine-tuning-mjo` fine-tunes Microsoft Aurora (1.3B-parameter 3D Swin Transformer)
for sub-seasonal prediction of the Madden–Julian Oscillation (MJO). The approach treats MJO
forecasting as a physics-consistent initial-value prognostic state stepping problem using
ERA5 6-hourly data, targeting an RMM bivariate correlation $r > 0.5$ at 30-day forecast lead.
Training runs on NERSC Perlmutter on 1 node × 4 × NVIDIA A100 80GB under SLURM.

Input data is the 1° NERSC/LANL ERA5 preprocessing archive on Community File System (CFS),
upsampled on GPU to Aurora's native 0.25° ($720 \times 1440$) grid. The project combines surface
and atmospheric variables stepping forward autoregressively, with frozen backbone transfer
learning, LoRA adaptation, and a Wheeler–Hendon RMM evaluation pipeline.

---

## 3. What this campaign did

The `refactor` campaign took a consolidated but untested codebase (73 tracked files, ~6,950 lines
of scientific Python, no tests, no lockfile, and broken entry points) and brought it to a verified,
production-ready state.

| Task | Outcome | Status |
| :--- | :--- | :--- |
| **A1** | Migrated dependency management from conda to `uv`; pinned PyTorch 2.5.1+cu121; retired `environment.yml`. | **GREEN** |
| **A2** | Untracked run outputs; archived July metrics to `docs/archive/metrics/`; configured pre-commit and ruff format. | **GREEN** |
| **A3** | Implemented stdlib verification gate `scripts/check.py`; configured ruff, pyrefly 1.2.0, and pytest. | **GREEN** |
| **A4** | Vendored `REPO_BUILD_GUIDE.md`; unified agent rules in `AGENTS.md`; deleted `.agent/` after folding in. | **GREEN** |
| **B1** | Captured machine-comparable baseline fingerprints (configs, model params, 1980 dataset, GPU smoke test). | **GREEN** |
| **B2** | Investigated July logs; proved structurally and via `train_auto.slurm` that past runs were unaffected by zeroed statics. | **AMBER** |
| **C1** | Moved `src/*.py` to `src/aurora_mjo/`, resolving collision with third-party `aurora`; re-pointed 27 imports. | **GREEN** |
| **C2** | Created thin `run.py` CLI (`train`, `show-config`); moved logic to `cli_support.py`; added 7-line `train.py` shim. | **GREEN** |
| **C3** | Folded `tools/` into `scripts/`; purged debris; extracted `aurora_mjo.rmm`; resolved dangling `dummy_dataset`. | **GREEN** |
| **C4** | Re-asserted B1 fingerprint post-refactor: 0 byte config diffs, 0 parameter delta, 0.0 smoke loss deviation. | **GREEN** |
| **D1** | Generated synthetic NetCDF fixtures (< 8 MB); built `conftest.py` with session fixtures and GPU auto-skip. | **GREEN** |
| **D2** | Converted 6 smoke scripts into structured pytest modules; 33 passing CI tests, 5 deselected on CPU. | **GREEN** |
| **D3** | Implemented regression suites for all 7 paid-for lessons across 5 modules; locked sigma arithmetic and indexing. | **GREEN** |
| **E1** | Configured `HDF5_USE_FILE_LOCKING=FALSE` at startup; inverted silent zero-statics fallback to `StaticVarLoadError`. | **GREEN** |
| **E2** | Implemented boundary `Config` validation via Pydantic v2; `extra="forbid"` rejects typos; enforced 9 validity rules. | **GREEN** |
| **E3** | Unified SLURM scripts under `uv run --frozen`; extracted `slurm_scripts/env.sh`; rewrote `eval.slurm`. | **GREEN** |
| **F1** | Overhauled docs (`SPEC`, `SETUP`, `ARCHITECTURE`, `CLI`, `PROJECT_STATE`); archived pre-refactor docs and gameplan. | **GREEN** |
| **F2** | Executed acceptance testing on Perlmutter GPU nodes: fresh clone synced, gate green, smoke identical, SLURM exit 0. | **GREEN** |

The campaign was governed by two named gates:
1. **B1 / C4 Fingerprint Gate:** B1 captured exact config serializations, model parameter counts,
   1980 dataset alignment, and GPU smoke test loss trajectories before any code moved. C4 re-asserted
   all properties after the package move, CLI redesign, and script consolidation. All five config diffs
   were empty, integer parameter counts had $\Delta = 0$, and smoke loss relative deviation was $0.0$ (**MEASURED**, `C4_result.md`).
2. **F2 Perlmutter Acceptance Gate:** Tested on a fresh clone on Perlmutter compute nodes (`nid008244`,
   `nid008256`). Confirmed 100% passes across 5 CFS data tests, 2 GPU tests, both entry points (`run.py`
   and `train.py`), and batch job `test_train.slurm` exiting 0 (**MEASURED**, `F2_result.md`).

---

## 4. The state now

When a contributor or agent clones `epic/refactor` (or `main` post-merge):
- **Layout:** First-party code lives in `src/aurora_mjo/`, imported and tested, never run directly.
  Scripts live in `scripts/`, run directly, never imported. `run.py` is the CLI entry point.
- **Installation:** Single package manager `uv`. On Perlmutter or local Linux:
  ```bash
  uv sync --all-groups
  ```
  `uv` works natively on Perlmutter compute nodes without conda (**MEASURED**, `A1_result.md`, `F2_result.md`),
  provided cache directories are exported to `/pscratch` to avoid Lustre locking error 524.
- **The verification gate:** One command defines green:
  ```bash
  uv run python scripts/check.py
  ```
  Runs lockfile validation, ruff lint, ruff format, pyrefly types, and pytest (excluding data/GPU markers).
- **Smoke test execution:**
  ```bash
  uv run python run.py train --mode baseline --smoke-test --resume none
  ```
- **SLURM job submission:**
  ```bash
  sbatch slurm_scripts/test_train.slurm   # 1 GPU debug smoke test (20s)
  sbatch slurm_scripts/train_auto.slurm   # 4-GPU 11.5h production training
  ```
- **Test coverage:** 92 total tests: 84 passing, 1 xfailed (msl placeholder detection), 7 deselected
  (4 `needs_data`, 2 `needs_gpu`, 1 `slow`) on default CI (**MEASURED**, `PROJECT_STATE.md`).

---

## 5. The numbers

### 5.1 Behavioural Fingerprints (B1 vs C4 vs F2)

| Property | B1 Fixture | C4 Post-refactor | F2 Acceptance | Source / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Config Diff** | Baseline | 0 bytes diff | 0 bytes diff | `B1_result.md`, `C4_result.md`, `F2_result.md` |
| **Physics Config Diff** | Baseline | 0 bytes diff | 0 bytes diff | `B1_result.md`, `C4_result.md`, `F2_result.md` |
| **LoRA Config Diff** | Baseline | 0 bytes diff | 0 bytes diff | `B1_result.md`, `C4_result.md`, `F2_result.md` |
| **Combined Config Diff** | Baseline | 0 bytes diff | 0 bytes diff | `B1_result.md`, `C4_result.md`, `F2_result.md` |
| **Override Config Diff** | Baseline | 0 bytes diff | 0 bytes diff | `training.optimizer.lr=1e-5` (`B1_result.md`, `C4_result.md`) |
| **Train Loss (Step 1)** | 7472.024414 | 7472.024414 | 7472.024414 | Abs $\Delta = 0.0$, Rel $\Delta = 0.0\%$ (`F2_result.md`) |
| **Val Loss (Step 1)** | 7421.924805 | 7421.924805 | 7421.924805 | Abs $\Delta = 0.0$, Rel $\Delta = 0.0\%$ (`F2_result.md`) |
| **Non-finite Grad Skips** | 0 / 1 (0.0) | 0 / 1 (0.0) | 0 / 1 (0.0) | Single-step synthetic smoke test (`F2_result.md`) |

### 5.2 Model Parameter Counts by Mode

| Mode | Total Parameters | Trainable Parameters | Frozen Parameters | Trainable % | LoRA Active | Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline` | 112,830,384 | 41,008 | 112,789,376 | 0.036% | False | `B1_result.md`, CPU count |
| `physics_informed` | 112,830,384 | 41,008 | 112,789,376 | 0.036% | False | `B1_result.md`, CPU count |
| `lora` | 113,371,056 | 581,680 | 112,789,376 | 0.513% | True | `B1_result.md`, CPU count |
| `combined` | 113,371,056 | 581,680 | 112,789,376 | 0.513% | True | `B1_result.md`, CPU count |

*Note:* Trainable parameter delta for LoRA is exactly $581,680 - 41,008 = 540,672$ attention projection weights.

### 5.3 Sample Counts and Temporal Arithmetic

| Split / Range | Days | Timesteps | Samples ($k=1$) | Boundary Formula | Source / Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1980 (leap)** | 366 | 1,464 | **1,462** | $1464 - 2 = 1462$ | **MEASURED** (`B1_result.md`, `F2_result.md`) |
| **1981 (non-leap)** | 365 | 1,460 | **1,458** | $1460 - 2 = 1458$ | **MEASURED** (`D1_result.md`, `D3_result.md`) |
| **1980 Gapped ($tcwv$)** | 366 | 1,408 | **1,404** | $(N_1-2) + (N_2-2) = 1404$ | **MEASURED** (`D1_result.md`, `D3_result.md`) |
| **1980–2015 (train)** | 13,149 | 52,596 | **52,594** | $52596 - 2 = 52594$ | **DERIVED** (`03_DOMAIN_PRIORS.md`) |
| **1980–2016 (leaked)** | 13,515 | 54,060 | — | Naive glob bug signature | **MEASURED as bug** (`03_DOMAIN_PRIORS.md`) |

### 5.4 Static Invariant Variables

| Variable | Tensor Shape | Mean | Std Dev | Finite Fraction | Provenance / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`z` (geopotential)** | `[720, 1440]` | 3709.2466 | 8216.3809 | 100% | **MEASURED** (`B1_result.md`, `F2_result.md`), $m^2 s^{-2}$ |
| **`lsm` (land-sea)** | `[720, 1440]` | 0.3357 | 0.4484 | 100% | **MEASURED** (`B1_result.md`, `F2_result.md`), fraction |
| **`slt` (soil type)** | `[720, 1440]` | 0.6708 | 1.1682 | 100% | **MEASURED** (`B1_result.md`, `F2_result.md`), categories 0–7 |

### 5.5 Hardware Compute, Memory, and Suite Metrics

| Metric | Measured Value | Context / Hardware | Source |
| :--- | :--- | :--- | :--- |
| `small` peak memory | 39.71 GiB | 4 × A100 80GB, batch size 1 | `03_DOMAIN_PRIORS.md` §7, `tools/probe_results/` |
| `huge` peak memory | 78.41 GiB (**OOM**) | 4 × A100 80GB without grad ckpt | `03_DOMAIN_PRIORS.md` §7, `tools/probe_results/` |
| `small` mean step time | 0.694 s | 4 × A100 80GB, single-step | `03_DOMAIN_PRIORS.md` §7 |
| Full test suite count | 92 tests | 84 pass, 1 xfail, 7 deselected | `scripts/check.py`, `PROJECT_STATE.md` |
| Gate execution runtime | ~33–42 s | Perlmutter login node | `scripts/check.py` (`F2_result.md`) |
| Type exclusion count | 8 entries | `src/aurora_mjo/{model,trainer,dataset...}` | `pyproject.toml` (reduced from 13 in A3) |

---

## 6. What was ruled out

A hypothesis eliminated is worth as much as one confirmed:
1. **July 2026 runs trained on zeroed static fields:** **RULED OUT** (**INFERRED**, `B2_result.md`, `docs/findings/2026-09-zeroed-statics.md`).
   While July batch logs were purged on scratch, structural analysis of `LANLMJODataset.__init__` proves
   that omitting `HDF5_USE_FILE_LOCKING=FALSE` causes an immediate fatal NetCDF crash on variable `2t`
   at step 0. Batch job `55806263` executed 3,750 steps and `train_auto.slurm` at commit `b9ffe63` explicitly
   set the variable at line 58. The runs trained on real topography.
2. **Zeroed statics caused July non-finite validation loss:** **RULED OUT** (**INFERRED**, `B2_result.md`).
   Non-finite validation loss is fully explained by Lesson 1: surface pressure normalised with MSL constants
   producing $-36 \sigma$ inputs over terrain (`AURORA_MJO_GAMEPLAN.md` Finding 1).
3. **`microsoft-aurora` name collision was a false alarm:** **RULED OUT** (**MEASURED**, `C1_result.md`).
   `microsoft-aurora` installs top-level `aurora` in site-packages. If first-party code remained `src/aurora`,
   it would shadow the vendor package and crash imports on `Batch`. `aurora_mjo` cleanly separates them.
4. **Permissive config typing (`extra="allow"`):** **RULED OUT** (**MEASURED**, `E2_result.md`).
   Permissive typing allowed typos like `--override training.optimzer.lr=1e-5` to silently execute without
   applying learning rates. `extra="forbid"` catches typos at process startup.
5. **Interactive cluster smoke tests experience noise:** **RULED OUT** (**MEASURED**, `B1_result.md`, `C4_result.md`).
   Single-step smoke tests on A100 GPUs under fixed seed produced bitwise identical train ($7472.024414$)
   and validation ($7421.924805$) losses across 4 consecutive runs ($\Delta = 0.0$).
6. **Pretrained weights required for unit testing:** **RULED OUT** (**MEASURED**, `D2_result.md`).
   Monkeypatching `Aurora.load_checkpoint` allows 100% offline instantiation on CPU in < 0.5s without HuggingFace network access.

---

## 7. What is still broken

### 7.1 Inherited and Deliberately Not Fixed (Out of Scope for Refactor)

1. **Placeholder `msl` normalisation statistics:** In `configs/unified.yaml`, `norm_stats.msl` overrides
   MSL constants using Aurora's built-in `sp` values (`location=96667.9822, scale=9504.6359`).
   While this prevents $-36 \sigma$ Tibetan spikes (pulling inputs to $-4.7 \sigma$), they are
   **placeholders** and not computed from this dataset (**MEASURED**, `D3_result.md`).
   *Blocks:* Scientifically trustworthy production fine-tuning.
2. **Finding 5 (Physics-informed mode parameter freezing):** `model.py` freezes the entire Aurora backbone.
   Consequently, the moisture-budget loss evaluates fields whose decoders cannot learn.
   *Blocks:* Physics-informed training mode (Finding 5 requires a human scientific decision).
3. **Finding 6 (`msl` output head miscalibration):** Aurora's pretrained output head for `msl` is frozen
   and calibrated for true MSL. Predictions mapped back to surface pressure denormalise with the wrong
   scale (**INFERRED**, `AURORA_MJO_GAMEPLAN.md`).
   *Blocks:* Multi-step autoregressive rollout accuracy for surface pressure.
4. **Finding 3 (Unclamped rollout feedback):** In multi-step autoregressive rollouts, predictions fed back
   into subsequent timesteps bypass Aurora's `apply_rollout_input_clipping` (**INFERRED**, `AURORA_MJO_GAMEPLAN.md`).
   *Blocks:* Multi-step rollout stability beyond lead step 2.
5. **Small vs Full Aurora (1.3B) scale:** `model_type: "small"` was adopted under deadline pressure because
   gradient checkpointing crashes Perlmutter A100s with illegal memory access errors.
   *Blocks:* Training the full 1.3B Aurora foundation model.
6. **`slt_data.nc` on purgeable scratch:** Soil type data lives at `/pscratch/sd/k/kam352/Aurora/slt/slt_data.nc`
   (Q-07). Scratch is subject to periodic purge.
   *Blocks:* Long-term headless pipeline execution without manual data restoration.

### 7.2 Discovered During the Campaign (Observations Harvest)

The eighteen task result files yielded **35 raw Observations** and **1 AMBER caveat**. Four were dropped
as trivial/transitional (e.g. log ignore rules in fixtures, git fast-forwards), merging into **11 substantive findings**:
- **AMBER Caveat (Task B2):** July 2026 SLURM execution logs were purged on scratch and unredirected from
  interactive sessions; core safety proved deductively via `_build_aligned_index()` structure.
- **Config Immutability Wart (`cli_support.py:auto_scale_memory`):** (Tasks E2, F1)
  Mutates the resolved configuration dictionary after boundary validation to adjust batch sizing for visible GPUs.
  Must be refactored to return an updated `Config` instance.
- **RMM Climatology Leakage Risk (`aurora_mjo.rmm.compute`):** (Task C3)
  `calc_climatology` computes day-of-year means across whatever time axis is passed. Upstream callers must
  ensure climatology is fitted strictly on training years (1980–2015) before subtracting from test data.
- **`slt` Grid Boundary Truncation (`dataset.py`):** (Task F1)
  `slt` data is truncated with `[:720, :]`. Coordinate boundaries must be verified against Aurora's 0.25° grid latitudes.
- **Multi-step Target List Structure:** (Task D2)
  `LANLMJODataset.__getitem__` returns lists of target dicts for multi-step rollouts, requiring callers to index `targets[step][var]`.
- **Multi-rank DDP Grad-Guard Verification:** (Task D3)
  `_collective_all_finite` executes single-rank path in CPU CI; collective multi-GPU synchronization requires SLURM test harness.
- **Interactive Session Logging Absence:** (Task B2)
  Interactive `salloc` commands lack stdout/stderr persistence unless wrapped with `tee` or script redirection.
- **Non-importability of `scripts/calc_norm_stats.py`:** (Task D3)
  Welford accumulation logic sits in `scripts/` and cannot be unit-tested without package extraction.
- **Auto-resumption on Smoke Tests:** (Tasks B1, C4, F2)
  Existing checkpoints in save directories cause single-epoch smoke tests to skip epoch 1; `--resume none` is mandatory for deterministic tests.
- **Perlmutter Filesystem Locking Error 524 on `$HOME`:** (Tasks A1, A2, C4, F2)
  Lustre home directories reject `flock`. Solved by exporting `UV_CACHE_DIR`, `PRE_COMMIT_HOME`, and `HF_HOME` to `/pscratch`.
- **Git Branch Ref Collision:** (Task A1)
  Git ref storage rejects `epic/refactor/<id>` because `epic/refactor` exists as a file. Standardized to `epic/refactor-<id>-<slug>`.

---

## 8. What the campaign got wrong

Honest accounting of planning assumptions versus measured reality:
1. **Sample-count arithmetic:** **HELD EXACTLY** (**MEASURED**, `D1_result.md`, `D3_result.md`, `F2_result.md`).
   The formula in `03_DOMAIN_PRIORS.md` §1 survived contact with real data: 1,462 samples for 1980,
   1,458 for 1981, and exact 58-sample reduction on gapped data.
2. **Third-party package collision:** **HELD EXACTLY** (**MEASURED**, `C1_result.md`).
   `microsoft-aurora` occupied `aurora`. Renaming the first-party package to `aurora_mjo` prevented a catastrophic shadowing bug.
3. **Wall-clock budgets:** **HEAVILY OVERESTIMATED**.
   Nearly every task was budgeted for 2.0 to 3.5 hours and completed in 15 to 45 minutes:
   B1 took 15 min (budget 3h); C1 took 15 min (budget 2h); D1 took 25 min (budget 3h); E2 took 45 min (budget 3h);
   F2 took 35 min (budget 3h). Clear specs, single-branch discipline, and working sandboxes made execution rapid.
4. **Touch lists vs tool configurations:** **SLIGHT TENSION IN A3**.
   In A3, `Must not touch` forbade modifying `src/` and `scripts/`, but the gate check ran formatting and type checks
   across the repo. A3 resolved this cleanly via `pyproject.toml` exclude ratchets rather than modifying forbidden files.
5. **Did C4 or F2 catch regressions?** **NO, PASSED TRIVIALLY**.
   Both C4 and F2 passed with 0 delta. They did not catch regressions because C1–C3 and E1–E3 were executed
   with strict backward compatibility. However, they provided the quantitative proof that zero drift occurred.

---

## 9. Open questions

All eleven questions in `QUESTIONS.md` were answered by the human on 2026-09-13:
- **Q-01 (Package name `aurora_mjo`):** Human approved. Proven sound in C1, C4, F2.
- **Q-02 (Agent execution on Perlmutter):** Human approved. Proven sound in B1, B2, F2.
- **Q-03 (`uv` on Perlmutter):** Human approved. Proven sound in A1, F2 (`uv run --frozen`).
- **Q-04 (`train.py` shim):** Human approved. 7-line shim working cleanly; deprecate in later campaign.
- **Q-05 (`pyrefly` type checker):** Human approved. Pyrefly 1.2.0 active with ratcheted excludes.
- **Q-06 (CI availability):** Human answered: research repo not in an org; GitHub Actions CI not required.
  *Current status:* `.github/workflows/ci.yml` exists, but `scripts/check.py` run locally is the gate.
- **Q-07 (`slt_data.nc` location):** Human answered: lives in `/pscratch/sd/k/kam352/Aurora/slt/`.
  *Current status:* Unchanged on purgeable scratch. Needs relocation or committed provenance before long runs.
- **Q-08 (RMM package extraction):** Human approved mechanical split. Completed in C3.
- **Q-09 (Branch naming):** Human approved `epic/refactor-<id>-<slug>`. Used throughout.
- **Q-10 & Q-11 (Perlmutter scratch caches):** Human approved. Standardized in `slurm_scripts/env.sh`.

---

## 10. Recommended next steps

### 10.1 The Single Most Critical Next Action

> **Compute the true 1980–2015 surface pressure (`ps`) normalisation statistics across CFS ERA5 data on Perlmutter using `scripts/calc_norm_stats.py` to replace the placeholder `msl` normalisation constants.** (**RECOMMENDED**)

**Argument:**
`02_UPSTREAM_CONTRACT.md` §4.1 and Lesson 1 establish that normalising surface pressure with MSL constants
triggers $-36 \sigma$ spikes and 100% NaN loss. The current override uses placeholder values derived from
Aurora's built-in `sp` statistics. While this unblocks single-step synthetic smoke tests, it is not computed
from this dataset. In 30-step runs on real ERA5, 100% of gradient steps were skipped (**MEASURED**, `03_DOMAIN_PRIORS.md` §7).
No training run or LoRA rollout can be scientifically meaningful until true training statistics exist.
Cost: ~1 hour of compute on Perlmutter via SLURM.

### 10.2 Prioritised Campaign Roadmap

1. **Replace Placeholder `msl` Normalisation Statistics:**
   - *Why:* The foundational scientific blocker.
   - *Cost:* Small (1 hour SLURM CPU job + updating `configs/unified.yaml`).
   - *Confidence:* High (**MEASURED**, `D3_result.md`).
   - *What would change mind:* If real stats produced non-finite validation loss (unlikely).
2. **Scientific Decision on Physics-Informed Mode & Head Unfreezing (Findings 5 & 6):**
   - *Why:* Physics-informed mode is scientifically incoherent while backbone decoders are frozen.
   - *Cost:* Low (1 human scientific decision on whether to unfreeze decoders or retire mode).
   - *Confidence:* High (**INFERRED**, `AURORA_MJO_GAMEPLAN.md`).
3. **Autoregressive Rollout Input Clipping (Finding 3):**
   - *Why:* Rollout stability collapses beyond lead step 2 without clamping fed-back predictions.
   - *Cost:* Small (apply Aurora's `apply_rollout_input_clipping` in `src/aurora_mjo/trainer.py`).
   - *Confidence:* High (**INFERRED**, `AURORA_MJO_GAMEPLAN.md`).
4. **Baseline Fine-Tuning Execution & RMM Evaluation:**
   - *Why:* Measure true 30-day MJO forecast skill ($r > 0.5$ target).
   - *Cost:* Medium (1 node × 4 × A100 for 11 hours under `train_auto.slurm`, followed by `run.py evaluate`).
   - *Confidence:* High (all infrastructure accepted in F2).
5. **Aurora 1.3B Full vs Small Scaling Benchmark:**
   - *Why:* Settle whether the 1.3B foundation model can run without illegal memory access crashes.
   - *Cost:* Small (benchmarking `scripts/probe_model_size.py` with SDPA math backend).
   - *Confidence:* Medium (depends on PyTorch SDPA kernel stability on Perlmutter).
6. **Technical Debt & Storage Hardening:**
   - *Why:* Prevent scratch purge loss and eliminate config mutation warts.
   - *Cost:* Small (refactor `auto_scale_memory`, relocate `slt_data.nc`, retire `train.py` shim).

### 10.3 Next Campaign Proposal

- **Campaign Name:** `2026-09-mjo-science-baseline`
- **Base Branch:** `epic/science-baseline` (cut from `main` after human merges `epic/refactor`)
- **Phases:**
  - *Phase 1 (Norm Stats):* Run `scripts/calc_norm_stats.py` over 1980–2015; update `configs/unified.yaml`; invert `test_placeholder_norm_stats_guard` from xfail to pass.
  - *Phase 2 (Scientific Alignment):* Resolve Findings 5 & 6 (unfreeze decoders) and Finding 3 (rollout input clipping).
  - *Phase 3 (Baseline Training):* Execute 3-epoch baseline fine-tuning run on 4 × A100 under SLURM chain.
  - *Phase 4 (Evaluation & Analysis):* Run RMM evaluation over 2016–2019 validation years; generate bivariate correlation $r$ lead curve.

---

## 11. Recovery points

Carried forward from `cleanup-2026-09.md` §8 and updated with campaign artifacts. **Do not delete, overwrite, or rebase any ref in this table:**

| Ref | Points at | Description / Reason to Retain |
| :--- | :--- | :--- |
| `origin/archive/pre-antigravity-baseline` | `800455a` | Remote-only, never merged. Historical baseline. **DO NOT DELETE.** |
| `backup/pre-cleanup/main` | `07a9ce0` | Pre-cleanup main state. |
| `backup/pre-cleanup/antigravity` | `b0cff2d` | Pre-cleanup integration branch state. |
| `backup/pre-cleanup/campaign-fixes` | `b9ffe63` | Pre-cleanup campaign-fixes branch state. |
| `backup/pre-cleanup/simul-training` | `6862fdf` | Pre-cleanup simul-training branch state. |
| `wip/snapshot/human-integration` | `05e8515` | Snapshot of dirty worktree state. |
| `wip/snapshot/campaign-fixes` | commit | Snapshot of pre-restore metrics. |
| `wip/snapshot/simul-training-one` | commit | Snapshot of dirty worktree state. |
| `v0.1.0-pre-agents` | tag | Pre-agent baseline tag. |
| `v0.1.1-agent-scaffold` | tag | Scaffold baseline tag. |
| `v0.1.2-pre-campaign-fixes` | tag | Pre-campaign fixes baseline tag. |
| `epic/refactor` | `85b0db5` | The completed refactor integration branch, ready for merge to `main`. |
| `docs/archive/AURORA_MJO_GAMEPLAN.md` | file | Preserved July 2026 gameplan analysis (byte-identical). |
| `docs/archive/metrics/*.jsonl` | files | Preserved July loss traces (`baseline-2026-07.jsonl`, `lora-2026-07.jsonl`). |
| `tests/fixtures/baseline/` | files | Canonical JSON fixtures capturing pre-refactor behavioural fingerprints. |

---

## 12. Where to go from here

- **For what the system is and what it guarantees:** Read [`docs/SPEC.md`](file:///pscratch/sd/k/kam352/Aurora/aurora-fine-tuning-mjo/docs/SPEC.md).
- **For how to work in this codebase:** Read [`AGENTS.md`](file:///pscratch/sd/k/kam352/Aurora/aurora-fine-tuning-mjo/AGENTS.md) and [`docs/REPO_BUILD_GUIDE.md`](file:///pscratch/sd/k/kam352/Aurora/aurora-fine-tuning-mjo/docs/REPO_BUILD_GUIDE.md).
- **For the current next action and status:** Read [`docs/PROJECT_STATE.md`](file:///pscratch/sd/k/kam352/Aurora/aurora-fine-tuning-mjo/docs/PROJECT_STATE.md).
- **For evidence behind any measurement:** Read [`docs/campaigns/refactor/results/`](file:///pscratch/sd/k/kam352/Aurora/aurora-fine-tuning-mjo/docs/campaigns/refactor/results/).
