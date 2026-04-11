---
trigger: always_on
---

# Research standards

- Separate baseline, ablation, and speculative work.
- Do not describe ideas as implemented unless code and configs exist.
- When changing metrics, document formulas and validation assumptions.
- When adding losses, state why each term exists and what failure mode it addresses.
- Avoid data leakage: chronological splits only, train-period normalization only.
- Prefer reproducible scripts/configs over notebook-only logic.