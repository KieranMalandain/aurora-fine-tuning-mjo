# Agent Instructions

<!--
docs/REPO_BUILD_GUIDE.md is vendored verbatim from research-repo-template.
It was written for quant research repos; docs/campaigns/refactor/01_TARGET_STATE.md §8
lists all deliberate divergences for this repository.
-->

## 1. Read order
1. This file (`AGENTS.md`)
2. `docs/REPO_BUILD_GUIDE.md`
3. `docs/PROJECT_STATE.md` (once authored in F1)
4. Your assigned campaign task file only (do not read ahead to other tasks)

## 2. The definition of done
One command defines done:
```bash
uv run python scripts/check.py
```
A task is never done with a red gate. If a Definition of Done item is not literally true, the status is not GREEN. Reporting green when it is red is the single most damaging thing an agent can do here.

## 3. What this project is
`aurora-fine-tuning-mjo` fine-tunes Microsoft Aurora (1.3B-parameter 3D Swin Transformer) for sub-seasonal prediction of the Madden–Julian Oscillation (target: RMM bivariate correlation r > 0.5 at 30-day lead).
The approach treats MJO forecasting as a physics-consistent initial-value prognostic state stepping problem using ERA5 6-hourly data.
Training runs on NERSC Perlmutter on 1 node × 4 × NVIDIA A100 80GB under SLURM.

## 4. Non-negotiable layout rules
- `src/aurora_mjo/` is imported and tested, never run directly.
- `scripts/` is run directly, never imported.
- `run.py` is a thin entry point: parses CLI flags and delegates to a single library function.
- `uv` is the only package manager — no conda, no bare pip, no poetry.
- Every runtime import must be a direct entry in `[project.dependencies]` in `pyproject.toml`.

## 5. Repo-specific prohibitions
- **Never write to `/global/cfs/cdirs/m4946/`**: It belongs to another group's CFS allocation; this repo reads it only.
- **Never flip `gradient_checkpointing` to `true`**: Checkpointing deterministically triggers an illegal memory access crash on Perlmutter A100s.
- **Never remove the `model.norm_stats.msl` override in `configs/unified.yaml`**: Removing it forces surface pressure through MSL normalisation, restoring a −36 σ input and 100% non-finite validation loss.
- **Never replace a probe with a constant in `dataset.py`**: Pressure-level indices and per-variable time axes are measured on purpose to prevent silent misalignment.
- **Never use a random train/val split**: Chronological splits only. Normalisation statistics and climatology must derive strictly from the training period to prevent evaluation leakage.
- **Never kill a running training process**: Never move or delete checkpoint directories without explicit user approval. Assume GPU time is scarce.
- **Never create a git worktree**: One clone, one branch per task, sequential execution.

## 6. Experiment registry rule
Every meaningful experiment change must update code, configuration, and `docs/experiment-registry.md`. Record: experiment name, date, objective, changed files, data split, variables used, losses used, rollout horizon, LoRA settings, expected outcome, and actual result after run.

## 7. Git discipline
- Branch from the campaign's epic branch (`epic/refactor`), never directly from `main`.
- Commit at internal milestones with descriptive commit messages explaining *why*.
- Push the branch and stop. Never merge your own branch; human merges after verification.

## 8. When stuck
Append your question to the active campaign's `QUESTIONS.md` as the next free `Q-nn` ID. State what it blocks, propose a concrete default, and immediately carry on with any work that does not depend on the answer.