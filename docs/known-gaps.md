# Known Gaps

This file lists parts of the repository that are known to be incomplete, outdated, or in transition.

## Known / suspected gaps

- current code may reflect an earlier project plan
- Yale vs LANL/NERSC data assumptions may differ

## Rule for agents

Do not silently assume these gaps are already resolved.
Inspect first, then report.

## Human Notes

- The current code implements the MJO head and rollout training in the main branch.
- `investigation_phase1/` contains legacy code from a previous iteration of the project. Code in here can be referred to, but nothing in this directory should be edited. Note this directory only exists on user's local storage, so ignore if not found.
- `Ps` (Surface Pressure) is currently being used as a proxy for `msl` (Mean Sea Level Pressure) because `msl` is missing from the LANL dataset.