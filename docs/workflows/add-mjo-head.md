---
description: 
---

# Add MJO head

Goal: implement an explicit MJO prediction head without destabilizing the existing backbone path.

Steps:
1. Identify the Aurora latent/output tensor shapes and current model API.
2. Propose the least invasive insertion point for an MJO head.
3. Implement:
   - tropical pooling
   - small MLP head
   - outputs for RMM1, RMM2, amplitude, active-MJO probability
4. Add config switches to enable/disable the head.
5. Add placeholder or real training loss wiring.
6. Update docs/architecture.md and docs/experiment-registry.md.
7. Return exact commands for a smoke test.