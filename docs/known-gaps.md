# Known Gaps

This file lists parts of the repository that are known to be incomplete, outdated, or in transition.

## Known / suspected gaps

- current code may reflect an earlier project plan
- Yale vs LANL/NERSC data assumptions may differ

## Rule for agents

Do not silently assume these gaps are already resolved.
Inspect first, then report.

## Human Notes

- The current code is a bit outdated relative to the roadmap. In particular, the MJO head and rollout training are not yet implemented in the main branch.
- `investigation_phase1/` contains legacy code from a previous iteration of the project. Code in here can be referred to, but nothing in this directory should be edited. Note this directory only exists on user's local storage, so ignore if not found.
- `src/dataset.py` LANL globbing logic (`_build_virtual_dataset`) is currently a placeholder and needs actual implementation to merge the `StepXX` directories.
- `Ps` (Surface Pressure) is currently being used as a proxy for `msl` (Mean Sea Level Pressure) because `msl` is missing from the LANL dataset.
- `slt` (Soil Type) is entirely missing from the NERSC ERA5 dataset. We currently inject a dummy zero-tensor into the batch to prevent Aurora embedding initialization from crashing. We can investigate the best way to resolve this (e.g., extracting from Yale Bouchet or finding the right static map) later.
- The dual-head architecture (State + MJO) is not yet implemented in `src/model.py`.