---
description: 
---

# Physics loss

Goal: add a moisture-centered auxiliary physics loss only after baseline rollout training exists.

Steps:
1. Identify available variables needed for the moisture residual.
2. State whether the loss is exact, approximate, or proxy-based.
3. Implement as an optional auxiliary term with a small default weight.
4. Add clear config flags and logging for the new term.
5. Update docs/architecture.md and docs/experiment-registry.md.
6. List expected failure modes and ablations to run.