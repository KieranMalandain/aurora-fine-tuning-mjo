---
description: 
---

# Rollout training

Goal: convert single-step training into curriculum-based multi-step rollout training.

Steps:
1. Inspect current training loop and autoregressive usage.
2. Add configurable rollout horizons.
3. Support curriculum schedule:
   - 4 steps
   - 8 steps
   - 20 steps
   - variable horizon
4. Add teacher-forcing or scheduled-sampling controls if needed.
5. Ensure logging reports losses by lead.
6. Update configs and docs/experiment-registry.md.
7. Return exact train and resume commands.