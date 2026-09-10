# <TASK_ID> — <title>

| | |
| --- | --- |
| **Phase** | <A/B/C/D> |
| **Depends on** | <task IDs with `PROCEED: YES`, or "none"> |
| **Base branch** | `epic/refactor` |
| **Budget** | <wall-clock estimate> |
| **Touches** | <the files this task is allowed to change> |
| **Must not touch** | <the files it must not> |

## Objective

One paragraph. What is true at the end that is not true now.

## Why this is a separate task

<So an agent understands the boundary and does not helpfully absorb the next
task.>

## What you may assume

<Facts established by prior tasks, with the result file that established them.
This is what stops re-derivation.>

## Steps

1. …

## Definition of Done

Copy this verbatim into your result file and tick each box. For every item with
a command, paste the actual output.

- [ ] …
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] `docs/PROJECT_STATE.md` updated
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

<Explicitly. Record anything you notice here as an Observation instead.>
