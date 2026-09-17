# Living Project State

**Last Updated:** 2026-09-17  
**Active Phase:** Campaign `science-baseline` underway (Task G1 implemented); preparing for Task G2 (`model_type: full` model scale).  

---

## 1. Next Action

> **Implement Task G2 (`model_type: full` model scale): configure production modes to instantiate `AuroraPretrained` (1.3B Swin3D backbone), run IMA matrix probe, benchmark memory/throughput at 1°, and ensure debug/smoke tests fall back to `AuroraSmallPretrained`.**

**Justification:** Task G1 successfully eliminated the 16× spatial upsampling overhead, enabling the full 1.3B model to fit comfortably within the 80 GB A100 budget without illegal memory access crashes.

*Campaign Synthesis & Next Steps:* See [`docs/campaigns/science-baseline/README.md`](campaigns/science-baseline/README.md) for full campaign plan and execution roadmap.

---

## 2. Status at a Glance

| Component / Subsystem | Status | Evidence / Reference | Notes |
| :--- | :--- | :--- | :--- |
| **Native Resolution Ingestion** | **GREEN** | Task G1 (`src/aurora_mjo/dataset.py`, `trainer.py`) | Both upsamplers deleted; 1° grid coordinates dynamically loaded from CFS archive; slt regridded to 1° (`data/static/slt_1deg.nc`); forward pass finite. |
| **Package Structure** | **GREEN** | Task C1 (`src/aurora_mjo/`) | First-party package installed via Hatchling; third-party `aurora` collision resolved. |
| **Dependency Management** | **GREEN** | Task A1 (`pyproject.toml`, `uv.lock`) | `uv` is the sole manager; PyTorch 2.5.1+cu121 links correctly with CUDA on Perlmutter. |
| **Verification Gate** | **GREEN** | Task A3 / G1 (`scripts/check.py`) | Stdlib runner checking lockfile, ruff lint/format, pyrefly types, and pytest (all PASS). |
| **Type Checking Ratchet** | **GREEN** | Task A3 / `pyproject.toml` | `pyrefly 1.2.0` active. Ratchet exclusion list holds 8 entries; entries come off, never on. |
| **Test Suite Coverage** | **GREEN** | Tasks D1–D3, E1–E2, G1 | **84 passed**, **1 xfailed**, **7 deselected** (needs data/gpu/slow) on default CI path; **5 passed** on `needs_data`. |
| **Environment Guards** | **GREEN** | Task E1 (`src/aurora_mjo/env.py`) | `HDF5_USE_FILE_LOCKING=FALSE` auto-set; zero-fallbacks replaced with `StaticVarLoadError`. |
| **Config Validation** | **GREEN** | Task E2 (`src/aurora_mjo/config.py`) | Pydantic v2 boundary validation with `extra="forbid"`; 9 validity rules enforced; `unified.yaml` free of `/pscratch`. |
| **RMM Core Extraction** | **GREEN** | Task C3 (`src/aurora_mjo/rmm/`) | Pure math extracted into `compute.py` and `evaluate.py`; scripts serve as thin CLI. |
| **SLURM Automation** | **GREEN** | Task E3 (`slurm_scripts/`) | All scripts unified under `uv run --frozen`; `env.sh` extracted; signal traps preserved. |
| **July 2026 Run Forensics** | **RESOLVED** | Task B2 / [`docs/findings/2026-09-zeroed-statics.md`](findings/2026-09-zeroed-statics.md) | Logs purged on scratch, but structural proof shows past runs did not run on zero statics. |
| **Documentation Suite** | **GREEN** | Task F1 (`README`, `SPEC`, `SETUP`, etc.) | Outdated configs and legacy cluster references purged; durable lessons promoted into `SPEC.md`. |
| **Perlmutter Acceptance** | **GREEN** | Task F2 (`results/F2_result.md`) | Fresh scratch clone; gate green; 5 data tests pass; 2 GPU tests pass; smoke test bitwise-identical to B1; SLURM debug job completed exit 0. |

---

## 3. Test Suite Metrics by Category

Measured locally via `uv run pytest --collect-only`:

| Category / Marker | Count | Execution Context | Verified In Tasks |
| :--- | :--- | :--- | :--- |
| **Default CI Path** (offline, synthetic CPU) | **85** (84 pass, 1 xfail) | Local dev, GitHub Actions (`ubuntu-latest`) | A3, D1, D2, D3, E1, E2, G1 |
| **`needs_data`** (requires CFS archive) | 5 | NERSC Perlmutter login / compute node | D2, D3, E1, G1 |
| **`needs_gpu`** (requires CUDA device) | 2 | NERSC Perlmutter GPU compute node | D2 |
| **`slow`** (rollout simulation > 5s) | 1 | NERSC Perlmutter GPU compute node | D2 |
| **`live`** (external network / HF Hub) | 0 | N/A (all offline) | D2 |
| **Total Test Suite** | **92** | Full suite passes across environments | F1, G1 |

*Note: The single `xfail` test (`tests/test_norm_stats.py::test_placeholder_stats_warning_or_failure`) tracks the open requirement to replace the `msl` placeholder constants.*

---

## 4. Open Checklist

- [ ] **Compute True Normalisation Statistics (OPEN-01):** Run `scripts/calc_norm_stats.py` over training years 1980–2015 on Perlmutter; update `configs/unified.yaml` and resolve `test_placeholder_stats_warning_or_failure` (Task G3).
- [x] **Relocate `slt_data.nc` (Q-07 / Q-17):** Regridded categorical soil type to 1° via nearest neighbour; saved and committed to `data/static/slt_1deg.nc` (~19.6 kB); removed purgeable scratch dependency; updated `configs/unified.yaml`.
- [ ] **Enable GitHub Branch Protection (Human Action):** Configure GitHub repository rulesets for `main` requiring pull requests and `scripts/check.py` CI status check approval.
- [ ] **Address Config Immutability Wart (Observation from E2):** Refactor `cli_support.py:auto_scale_memory` to remove post-validation dictionary mutation.
- [ ] **Scientific Campaign: Rollout & Loss Dynamics:**
  - *Finding 3:* Clamp fed-back predictions in autoregressive rollouts before feeding into subsequent input windows.
  - *Finding 5:* Resolve parameter freezing in physics-informed mode so moisture-budget loss can update relevant decoders.
  - *Finding 6:* Unfreeze or recalibrate `msl` output head so predicted fields match physical surface pressure scales.
  - *Model Scale (Task G2):* Conduct formal throughput/memory scaling benchmarks on 1.3B full Aurora vs. small variant.

---

## 5. Unverified Assumptions

1. **`slt` Grid Truncation Linearity (RESOLVED):** Truncation defect retired in Task G1 by nearest-neighbor regridding of categorical soil types directly to the 1.0° cell-centred grid (`data/static/slt_1deg.nc`).
2. **CFS File Locking Permanence:** Assumes CFS will continue to require `HDF5_USE_FILE_LOCKING=FALSE` indefinitely; guarded automatically in `env.py`.
3. **Wheeler–Hendon Projection Completeness:** Assumes training-period EOF basis (`rmm_basis.npz`) computed over 1980–2015 remains fully stationary across 2016–2023 evaluation splits.

---

## 6. Superseded Artifacts & Decisions

| Historical State / Concept | Superseded By | Reason for Supersession |
| :--- | :--- | :--- |
| **Git Worktrees per Agent** | Single branch per task (`epic/<slug>`) | Worktree model generated multi-branch divergence and detached stashes. |
| **`environment.yml` / Conda** | `pyproject.toml` + `uv.lock` | Unpinned specs allowed silent dependency drift; `uv` provides deterministic reproducibility. |
| **Flat `src/*.py` Layout** | `src/aurora_mjo/` package | Flat namespace collided with `microsoft-aurora` site-packages. |
| **Per-phase YAML Configs** | `configs/unified.yaml` with `--mode` | Multiple divergent files caused configuration drift and broken references. |
| **`train.py` Entry Point** | `run.py train` (shim preserved) | Thin CLI consolidation under Typer with validated boundary config. |
| **Broad `try...except` Statics** | `StaticVarLoadError` hard failure | Silent zero-substitutions masked missing data and filesystem errors. |
