# Documentation Archive (`docs/archive/`)

This directory preserves historical documents, pre-consolidation analyses, superseded verification traces, and early project artifacts. These files are kept for provenance, debugging forensics, and historical context; they are **not** active specifications.

Active documentation lives in `docs/` (`SPEC.md`, `SETUP.md`, `ARCHITECTURE.md`, `CLI.md`, `PROJECT_STATE.md`) and at the repository root (`README.md`, `AGENTS.md`).

---

## Catalog of Archived Artifacts

### 1. `AURORA_MJO_GAMEPLAN.md`
- **Original Location:** Repository root (`/AURORA_MJO_GAMEPLAN.md`).
- **Date & Origin:** July 2026, authored by Orchestrator (Opus 4.8).
- **Historical Significance:** The most substantive analytical document produced during initial model fine-tuning. It established the authoritative root-cause diagnosis of the 100% non-finite validation loss observed in July runs:
  - *Finding 1 (The Trigger):* Aurora's `msl` channel actually ingests surface pressure (`ps`) from the LANL archive, but was normalised using Aurora's built-in mean-sea-level constants (`location=100958, scale=1332`), pushing high-terrain inputs to −36 σ globally.
  - *Finding 2 (The Corruption Path):* Under bf16, `GradScaler` is disabled, allowing non-finite gradient updates to corrupt Adam moment buffers permanently even at learning rate ≈ 0.
- **Why Archived Unedited:** Preserved verbatim for scientific and engineering forensics. Config references (`phase1_baseline.yaml`, `phase2_physics.yaml`, `phase3_longrun.yaml`) and script names predated the September 2026 repository consolidation.
- **Superseded By:** Findings 1–6 and the architectural invariants are promoted into [`docs/SPEC.md`](../SPEC.md) and [`docs/PROJECT_STATE.md`](../PROJECT_STATE.md).

### 2. `verify_output_pre_slt.txt`
- **Original Location:** `docs/verify_output.txt`.
- **Date & Origin:** Pre-consolidation dataset verification output.
- **Historical Significance:** Shows `slt` (soil type) fully zeroed (`mean=0.0000, std=0.0000`) because `slt_data.nc` was not yet integrated into the static variable loader.
- **Superseded By:** [`docs/verify_output_slt.txt`](../verify_output_slt.txt), which demonstrates real physical soil type data loading (`mean=0.6708, std=1.1682`).

### 3. `nersc_data_pre_integration.md`
- **Original Location:** `docs/nersc_data.md`.
- **Date & Origin:** Early 2026 NERSC ERA5 data exploration note.
- **Historical Significance:** Documented preliminary data discovery and recommended fixes for coordinate mappings (`'lev'` vs `'level'`, capitalized `'Z'`).
- **Superseded By:** Consolidated into [`docs/SPEC.md`](../SPEC.md), [`docs/nersc-dataset-information.md`](../nersc-dataset-information.md), and verified by dataset unit tests.

### 4. `documentation_pre_refactor.md`
- **Original Location:** `docs/documentation.md`.
- **Date & Origin:** January 2026, authored by Kieran Malandain.
- **Historical Significance:** Early documentation describing deleted per-phase configs (`phase1_baseline.yaml`, `phase2_physics.yaml`, `phase3_longrun.yaml`), the obsolete `environment.yml` environment, and legacy training workflows.
- **Superseded By:** [`docs/SPEC.md`](../SPEC.md), [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md), [`docs/SETUP.md`](../SETUP.md), and [`docs/CLI.md`](../CLI.md).

### 5. `known_gaps_pre_refactor.md`
- **Original Location:** `docs/known-gaps.md`.
- **Date & Origin:** Early pre-consolidation notes.
- **Historical Significance:** Listed informal gaps and legacy Yale vs LANL assumptions.
- **Superseded By:** [`docs/PROJECT_STATE.md`](../PROJECT_STATE.md) and [`docs/SPEC.md`](../SPEC.md).

### 6. `development_workflow_pre_refactor.md`
- **Original Location:** `docs/development-workflow.md`.
- **Date & Origin:** Early guidelines for agent iteration.
- **Superseded By:** [`AGENTS.md`](../../AGENTS.md), [`docs/REPO_BUILD_GUIDE.md`](../REPO_BUILD_GUIDE.md), and [`docs/SETUP.md`](../SETUP.md).

### 7. `antigravity_usage_pre_refactor.md`
- **Original Location:** `docs/antigravity-usage.md`.
- **Date & Origin:** Initial multi-agent role description.
- **Superseded By:** [`AGENTS.md`](../../AGENTS.md) and [`docs/campaigns/refactor/04_AGENT_PROTOCOL.md`](../campaigns/refactor/04_AGENT_PROTOCOL.md).

### 8. `metrics/`
- **Contents:** `baseline-2026-07.jsonl` and `lora-2026-07.jsonl`.
- **Date & Origin:** Preserved during task A2 from tracked `checkpoints/{baseline,lora}/metrics.jsonl`.
- **Historical Significance:** The sole surviving operational traces of the July 2026 non-finite training runs.
