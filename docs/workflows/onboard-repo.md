---
description: Rapid repository onboarding workflow
---

# Onboard Repo Workflow

Goal: Rapidly understand the repository architecture before proposing changes.

## Steps
1. Read `AGENTS.md`, `docs/REPO_BUILD_GUIDE.md`, `docs/SPEC.md`, and `docs/PROJECT_STATE.md`.
2. Inspect repository structure and identify:
   - Data pipeline entry points (`src/aurora_mjo/dataset.py`)
   - Training entry points (`run.py train`, `src/aurora_mjo/trainer.py`)
   - Evaluation entry points (`run.py evaluate`, `scripts/evaluate_mjo.py`)
   - Configuration system (`configs/unified.yaml`, `src/aurora_mjo/config.py`)
3. Summarize current architecture in 10 bullets max.
4. Verify the verification gate: `uv run python scripts/check.py`.
5. Propose the smallest next implementable step.
6. Commit at internal milestones and never merge your own branch.

*Note: Always verify compute host before submitting jobs (`hostname`). All active workloads target NERSC Perlmutter.*