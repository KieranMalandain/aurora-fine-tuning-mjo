---
description: 
---

# Build RMM evaluation

Goal: add a correct, reproducible RMM evaluation pipeline.

Steps:
1. Locate current evaluation code and forecast output format.
2. Implement or verify:
   - climatology computation from training years only
   - anomaly generation
   - EOF projection pipeline
   - RMM1/RMM2 generation
3. Compute lead-dependent metrics:
   - bivariate correlation
   - amplitude error
   - phase error
   - active-event skill
4. Add tests or validation checks where feasible.
5. Document formulas and assumptions in docs/evaluation-spec.md.
6. Do not claim benchmark parity or SOTA without actual run results.