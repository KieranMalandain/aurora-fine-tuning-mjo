---
trigger: always_on
---

# Agent coordination

When multiple agents are used:
- Assign each agent a distinct file/domain ownership area.
- Do not have two agents edit the same file concurrently.
- One agent should act as coordinator and merge reviewer.
- Research agents may read broadly but should not rewrite training code directly.
- Implementation agents must report:
  - files changed
  - commands to run
  - unresolved assumptions

When creating task files in `.agent/tasks/`, the following format must be used as a baseline:

# Task: [task]

## Objective

[objective]

## Working rules
- Read only the files listed above unless you discover a direct dependency.
- If you need to expand scope, stop and report why.
- Make the smallest possible coherent patch.
- Keep all changes confined to the modify list unless explicitly authorized.
- Ensure conda/mamba environment `aurora_mjo` is activated before running Python code (directory `~/miniforge3/envs/aurora_mjo/bin/python`).

## Read

[list of files to read]

## Modify

[list of files to modify]

## Do not touch

[list of what not to touch, e.g., "model architecture" or "training loop"]

## Expected output

[list of expected outputs, e.g., "RMM basis / projection logic" or "brief explanation of climatology assumptions" or "smoke test command"]

## Steps

[clear, numbered steps]

## Risks & Reminders

[any risks and reminders, e.g., about data leakage]