# Known Gaps

This file lists parts of the repository that are known to be incomplete, outdated, or in transition.

## Known / suspected gaps

- current code may reflect an earlier project plan
- explicit MJO head may not yet exist
- RMM evaluation may be partial or outdated
- rollout curriculum may not yet be implemented
- LoRA integration may not match current intended design
- physics-informed loss may still reflect an earlier formulation
- Yale vs LANL/NERSC data assumptions may differ

## Rule for agents

Do not silently assume these gaps are already resolved.
Inspect first, then report.

## Human Notes

- The current code is a bit outdated relative to the roadmap. In particular, the MJO head and rollout training are not yet implemented in the main branch.
- `investigation_phase1/` contains legacy code from a previous iteration of the project. Code in here can be referred to, but nothing in this directory should be edited. 
- `src/dataset.py` LANL globbing logic (`_build_virtual_dataset`) is currently a placeholder and needs actual implementation to merge the `StepXX` directories.
- `Ps` (Surface Pressure) is currently being used as a proxy for `msl` (Mean Sea Level Pressure) because `msl` is missing from the LANL dataset.
- The dual-head architecture (State + MJO) is not yet implemented in `src/model.py`.
- `compute_rmm.py` does not exist, meaning we currently have no ground-truth targets to train the MJO head against.