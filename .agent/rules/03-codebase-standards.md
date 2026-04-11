---
trigger: always_on
---

# Codebase standards

- Keep training, evaluation, and data code separated.
- Prefer configuration-driven experiments.
- Avoid hidden constants in training scripts.
- New experiments should add config entries, not hard-coded branches.
- Use clear module boundaries:
  - data_local/
  - docs/
  - models/
  - training/
  - evaluation/
  - configs/
  - scripts/
  - slurm_scripts/
- If refactoring, preserve existing behavior unless the task explicitly changes it.
- investigation_phase1/ does not need to be edited at all, since it contains legacy code from an earlier iteration of the project which was proof-of-concept. But they can be referred to for specific logic.