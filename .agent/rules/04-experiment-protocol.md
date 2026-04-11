---
trigger: always_on
---

# Experiment protocol

Every meaningful experiment change must update:
- code
- config
- docs/experiment-registry.md

For each experiment, record:
- experiment name
- date
- objective
- changed files
- data split
- variables used
- losses used
- rollout horizon
- LoRA settings if any
- expected outcome
- actual result after run

Do not overwrite prior experiment definitions casually.