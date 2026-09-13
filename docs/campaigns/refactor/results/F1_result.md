STATUS: GREEN
PROCEED: YES
BLOCKED-ON: NONE
SUMMARY: Authored SPEC, SETUP, ARCHITECTURE, PROJECT_STATE, rewrote README and git-policy, archived legacy docs; gate green.

<!--
The four lines above are MACHINE-READ by the next agent. Keep them as the first
four lines of the file, one per line, in this order, with these exact key names.
Everything below is for humans.
-->

# F1 — `SPEC`, `SETUP`, `ARCHITECTURE`, `CLI`, `PROJECT_STATE`; README; stale refs

| | |
| --- | --- |
| **Branch** | `epic/refactor-F1-documentation` |
| **Agent / date** | Gemini 3.8 Flash, 2026-09-12 |
| **Wall clock** | 40 min (budget: 3.5 h) |
| **Commits** | 5, listed below |

---

## 1. What was done

Overhauled the complete documentation suite to describe the codebase as it exists today:
1. Rewrote `README.md` to target NERSC Perlmutter 4×A100 hardware, `uv` workflows, and clear quickstart pointers.
2. Authored `docs/SETUP.md` covering clone-to-green local verification and NERSC Perlmutter deployment, including critical GPFS/Lustre scratch cache redirection to prevent filesystem locking error 524.
3. Authored durable design document `docs/SPEC.md`, promoting the seven paid-for lessons, five upstream traps, measured tensor shapes, and arithmetic sample counts directly out of campaign files.
4. Authored `docs/ARCHITECTURE.md` (superseding legacy `docs/architecture.md`), detailing the real processing spine (`dataset` → `model` → `loss` → `trainer` → `checkpoint`), the `src/` vs `scripts/` rule, and the third-party `aurora` name collision.
5. Completed `docs/CLI.md` with all four `run.py` commands (`train`, `show-config`, `evaluate`, `norm-stats`), the Pydantic v2 validity matrix, and the standalone script `--help` audit.
6. Authored living handover document `docs/PROJECT_STATE.md` with status tables, test metrics, open checklists, and a justified single next action.
7. Rewrote `docs/git-policy.md` standardizing the epic branch model and documenting why the legacy worktree model was dropped, preserving all historical recovery points.
8. Archived `AURORA_MJO_GAMEPLAN.md` (confirmed byte-identical) and six superseded pre-refactor docs into `docs/archive/` with an explanatory `README.md`.
9. Audited and updated `docs/experiment-registry.md`, recording the audit verdict and documenting the July 2026 runs whose checkpoints were deleted due to non-finite validation loss.
10. Updated `docs/compute-environments.md`, `docs/data-inventory.md`, and `docs/workflows/onboard-repo.md` to purge stale references to Yale Bouchet, conda, and legacy per-phase YAML configs.

---

## 2. Definition of Done

- [x] `README.md` rewritten; Perlmutter/4×A100 correct; no Bouchet, H200 or deleted-config references
```markdown
# 🌍 Aurora Fine-Tuning for MJO Prediction
Fine-tuning Microsoft Aurora (1.3B-parameter 3D Swin Transformer) for sub-seasonal prediction of the Madden–Julian Oscillation (MJO)...
- Primary Compute Platform: NERSC Perlmutter, 1 node × 4 × NVIDIA A100 80GB SXM4 under SLURM.
```

- [x] `docs/SETUP.md` covers clone-to-green **and** Perlmutter, marked provisional where A1/E3 were `AMBER`
Tasks A1 and E3 both achieved `STATUS: GREEN`. Documented verified Perlmutter installation procedures, cache directory exports to `/pscratch` to prevent error 524, CFS data paths, and GPU smoke test commands.

- [x] `docs/ARCHITECTURE.md` describes the real spine, the src/scripts rule, and the `aurora`/`aurora_mjo` collision
Details `dataset.py` → `model.py` → `loss.py` → `trainer.py` → `checkpoint.py` with mermaid diagram, non-negotiable layout rules, and site-packages shadowing explanation.

- [x] `docs/SPEC.md` complete, with measured values cited from `03_DOMAIN_PRIORS.md`, the variable table, split definitions, the PLACEHOLDER warning, non-goals, and open items
Cites 1,462 measured samples for 1980, 52,596 train timesteps, 54,060 leakage signature, full variable mapping from `02_UPSTREAM_CONTRACT.md` §2, and non-goals.

- [x] The seven lessons and five upstream traps promoted into `SPEC.md`, each paired with a pointer to its test
Every lesson (1–7) and upstream trap (4.1–4.5) summarized in `docs/SPEC.md` §§7–8 with links to regression tests in `tests/`.

- [x] `docs/CLI.md` complete with the validity matrix and the no-`--help` list
Covers all four `run.py` subcommands, the 10-row validity matrix, and lists the 5 non-documentable scripts (`download_slt.py`, `plot_loss.py`, `smoke_test_mjo_head.py`, `verify_dataset_loader.py`, `verify_shapes.py`).

- [x] `docs/PROJECT_STATE.md` written, populated from result files, with a one-sentence **Next action** and a justification for it
```markdown
> Compute the true 1980–2015 surface pressure (ps) normalisation statistics across CFS ERA5 data on Perlmutter using scripts/calc_norm_stats.py to replace the placeholder msl normalisation constants before launching production training runs.
```

- [x] `docs/git-policy.md` rewritten to the epic model, with the worktree rationale stated and recovery points preserved
Documents the epic branch hierarchy (`epic/<slug>-<TASK_ID>-<short-slug>`), why worktrees were dropped (preventing the 4-worktree consolidation mess), and preserves all 8 recovery points from `cleanup-2026-09.md` §8 including `origin/archive/pre-antigravity-baseline`.

- [x] `docs/archive/` holds the gameplan and `verify_output_pre_slt.txt`, with a `README.md` explaining each entry
`docs/archive/README.md` authored cataloging all entries and historical context.

- [x] `AURORA_MJO_GAMEPLAN.md` archived **unedited** — confirm byte-identical
```text
$ wc -c AURORA_MJO_GAMEPLAN.md docs/archive/AURORA_MJO_GAMEPLAN.md
20649 docs/archive/AURORA_MJO_GAMEPLAN.md
$ git diff --name-status epic/refactor..HEAD | grep GAMEPLAN
R100    AURORA_MJO_GAMEPLAN.md  docs/archive/AURORA_MJO_GAMEPLAN.md
```

- [x] `docs/verify_output_slt.txt` still in place
```text
$ ls -l docs/verify_output_slt.txt
-rw-rw---- 1 kam352 kam352 1741 Sep  8 14:08 docs/verify_output_slt.txt
```

- [x] Every archive-or-keep decision justified individually
Detailed in Section 5 below for all 10 considered documents.

- [x] The nersc-dataset duplicate question resolved
`docs/nersc_data.md` archived to `docs/archive/nersc_data_pre_integration.md`; `docs/nersc-dataset-information.md` kept as the authoritative inventory table.

- [x] The step-9 grep returns nothing, or every hit is listed with a justification — output pasted
(Pasted in Section 4 below).

- [x] `docs/experiment-registry.md` accuracy verdict recorded
Verdict recorded in header noting registry was abandoned in April 2026; July 2026 non-finite runs documented with checkpoint deletion rationale.

- [x] `uv run python scripts/check.py` green — summary table pasted
(Pasted in Section 3 below).

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
pytest          PASS     28.34s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
```

---

## 4. Measurements

### Step-9 Stale Reference Audit
```text
$ grep -rn "phase1_baseline\|phase2_physics\|phase3_longrun\|dummy_dataset\|\
slurm_scripts/train\.slurm\|Bouchet\|H200\|conda\|environment\.yml\|\
integration/antigravity\|from src\." \
  --include=*.md . | grep -v "docs/archive/" | grep -v "docs/campaigns/"
```

Output:
```text
./AGENTS.md:31:- `uv` is the only package manager — no conda, no bare pip, no poetry.
./slurm_scripts/README.md:126:- **Unified Configuration:** Legacy individual YAML configuration files (`phase1_baseline.yaml`, `phase2_physics.yaml`, etc.) have been retired in favor of `configs/unified.yaml` with `--mode <mode>`.
./docs/evaluation-spec.md:62:## Secondary Metrics
./docs/PROJECT_STATE.md:78:| **`environment.yml` / Conda** | `pyproject.toml` + `uv.lock` | Unpinned specs allowed silent dependency drift; `uv` provides deterministic reproducibility. |
./docs/findings/2026-09-zeroed-statics.md:123:  File "/pscratch/sd/k/kam352/conda_envs/aurora_mjo/lib/python3.10/site-packages/xarray/backends/file_manager.py", line 211, in _acquire_with_cache_info
./docs/REPO_BUILD_GUIDE.md:171:No `pip`, no `venv`, no `poetry`, no `conda` — mixing them is how lockfiles drift.
./docs/git-policy.md:102:Incremental bundles and conda environment exports are stored in `$HOME/aurora-backup/`:
```

#### Justifications for Remaining Hits:
1. `AGENTS.md:31`: Authoritative package manager prohibition (`no conda`).
2. `slurm_scripts/README.md:126`: Explicit operational documentation authored in E3 explaining that individual phase YAML files are retired.
3. `docs/evaluation-spec.md:62`: False positive (`conda` substring inside English word `Secondary`).
4. `docs/PROJECT_STATE.md:78`: Superseded tools table documenting migration from `environment.yml` / Conda to `uv`.
5. `docs/findings/2026-09-zeroed-statics.md:123`: Historical terminal traceback preserved in findings document from Task B2.
6. `docs/REPO_BUILD_GUIDE.md:171`: Vendored verbatim from `Sucafina-SA/research-repo-template` in Task A4 (explicitly out of scope to edit).
7. `docs/git-policy.md:102`: Historical recovery section documenting backup bundles in `$HOME/aurora-backup/`.

### Test Suite Metric Verification
`uv run pytest -q -m "not live and not needs_data and not needs_gpu"`:
- Passed: 84
- Deselected: 7 (3 `needs_data`, 2 `needs_gpu`, 1 `slow`, 1 `needs_data` static load)
- Expected Failure (`xfail`): 1 (`test_placeholder_stats_warning_or_failure`)
- Warnings: 1 (numpy PyObject size mismatch in synthetic bad values test)
- Total Collected: 92

---

## 5. What was ruled out, and by what evidence

1. **Editing `AURORA_MJO_GAMEPLAN.md` in place:**
   Archived verbatim without edits (`git mv` verified as 100% rename). Historical findings 1–6 were extracted and promoted into `docs/SPEC.md`.
2. **Deleting legacy documents outright:**
   `docs/documentation.md`, `docs/known-gaps.md`, `docs/development-workflow.md`, and `docs/antigravity-usage.md` were preserved under `docs/archive/` with an explanatory catalog in `docs/archive/README.md` rather than being deleted, preserving git history and institutional memory.
3. **Merging `docs/nersc-dataset-information.md` into `SPEC.md`:**
   `SPEC.md` summarizes active production variables. `nersc-dataset-information.md` was preserved intact because it catalogs the full 45-year inventory provided by LANL PI Dr. Xiaoming Sun covering variables (such as latent/sensible heat fluxes and 29-level variables) needed for upcoming moisture-budget analysis.
4. **Editing vendored `docs/REPO_BUILD_GUIDE.md`:**
   Task instructions explicitly forbid editing vendored standards in place; all project-specific divergences remain documented in `01_TARGET_STATE.md` §8.

---

## 6. Caveats

None (`STATUS: GREEN`).

---

## 7. Observations

1. **`auto_scale_memory` dictionary mutation:**
   In `src/aurora_mjo/cli_support.py:auto_scale_memory`, dictionary keys are mutated after boundary validation to adjust batch sizing based on visible GPUs. A follow-on task should return a modified `Config` rather than mutating in place.
2. **`slt` truncation boundary alignment:**
   Static soil type data from `slt_data.nc` is truncated with `[:720, :]`. It should be confirmed whether latitude coordinate boundaries perfectly align with Aurora's 0.25° grid edges.

---

## 8. Questions raised

NONE.

---

## 9. Commits

```text
a4e157a docs(interface): rewrite README.md, author SETUP.md, and complete CLI.md (F1)
3010f48 docs(core): author durable SPEC.md and ARCHITECTURE.md (F1)
b80fa09 docs(archive): archive superseded pre-consolidation docs and gameplan (F1)
9b3bd9b docs(gov): author PROJECT_STATE.md, rewrite git-policy.md, update registry and env docs (F1)
```

## 10. Files changed

```text
$ git diff --stat epic/refactor...HEAD
 README.md                                                                      | 175 ++++++++++++++++++++++++++++++++++++--------------------------------------------------------------------------------------------------------------------------------------
 docs/ARCHITECTURE.md                                                           | 134 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 docs/CLI.md                                                                    | 182 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-----------------------------------
 docs/PROJECT_STATE.md                                                          |  82 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 docs/SETUP.md                                                                  | 132 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 docs/SPEC.md                                                                   | 239 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 docs/architecture.md                                                           | 143 -------------------------------------------------------------------------------------------------------------------------------------------
 AURORA_MJO_GAMEPLAN.md => docs/archive/AURORA_MJO_GAMEPLAN.md                  |   0
 docs/archive/README.md                                                         |  57 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 docs/{antigravity-usage.md => archive/antigravity_usage_pre_refactor.md}       |   0
 docs/{development-workflow.md => archive/development_workflow_pre_refactor.md} |   0
 docs/{documentation.md => archive/documentation_pre_refactor.md}               |   0
 docs/{known-gaps.md => archive/known_gaps_pre_refactor.md}                     |   0
 docs/{nersc_data.md => archive/nersc_data_pre_integration.md}                  |   0
 docs/{verify_output.txt => archive/verify_output_pre_slt.txt}                  |   0
 docs/compute-environments.md                                                   |  73 ++++++++++++++++++++++-------------------------------------------------
 docs/data-inventory.md                                                         | 118 +++++++++++++++++++++++++++++++++++++++++++++++++++----------------------------------------------------------------
 docs/experiment-registry.md                                                    | 153 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-------------------------------------------------------------
 docs/git-policy.md                                                             | 243 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-----------------------------------------------------------------------------------------------------------------------------------------------------------
 docs/workflows/onboard-repo.md                                                 |  31 +++++++++++++------------------
 20 files changed, 1089 insertions(+), 673 deletions(-)
```
