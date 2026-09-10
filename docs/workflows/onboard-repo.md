---
description: 
---

# Onboard repo

Goal: rapidly understand the repository before proposing changes.

Steps:
1. Read GEMINI.md, AGENTS.md, docs/project-brief.md, and .agent/rules/*
2. Inspect repo structure and identify:
   - data pipeline entrypoints
   - training entrypoints
   - evaluation entrypoints
   - config system
3. Summarize current architecture in 10 bullets max.
4. Identify missing pieces relative to the intended roadmap:
   - MJO head
   - RMM evaluation
   - rollout training
   - LoRA
   - physics loss
5. Propose the smallest next implementable step.
6. Do not edit files unless explicitly asked.

Note: always verify at the beginning of a session whether we are ssh into Yale Bouchet or LANL NERSC.