# Task: Update Documentation for Phase 2

## Objective
Update the core repository documentation to reflect the successful implementation and integration of the Phase 2 roadmap. Specifically, the system now fully features a functioning MJO Head, Autoregressive $k$-step Rollouts, LoRA parameter freezing, Moisture-Budget Physics Loss, and formal RMM Evaluation scripts.

## Working rules
- Read only the files listed below unless you discover a direct dependency.
- If you need to expand scope, stop and report why.
- Keep all changes confined to the modify list unless explicitly authorized.

## Read
- `docs/architecture.md`
- `docs/documentation.md`
- `docs/known-gaps.md`
- `docs/project-brief.md`
- `README.md`
- `scripts/`
- `src/`
- `train.py`

## Modify
- `docs/architecture.md`
- `docs/documentation.md`
- `docs/known-gaps.md`
- `docs/project-brief.md`
- `README.md`

## Do not touch
- **ANY SECTION ENTITLED `## Human Notes`** (You must entirely ignore and preserve these sections across all files).
- Python source code (`src/`, `scripts/`).
- SLURM scripts or `configs/`.

## Expected output
- `docs/architecture.md`: Updated to indicate that the MJO Head, LoRA strategy, Physics Loss phase, Evaluation script, and Rollout loop are now fully *implemented* rather than planned scaffolding.
- `docs/documentation.md`: Significantly rewritten directory tree to include Phase 2 files (`phase2_rollout.yaml`, `phase2_physics.yaml`, `scripts/evaluate_mjo.py`, etc.). Detailed explanations added for new trainer loop and physics loss math.
- `docs/known-gaps.md`: Cleared of the "Known / suspected gaps" that have now been resolved (e.g., MJO head, rollout curriculum, physics-loss formulation, RMM targeting logic are completed).
- `docs/project-brief.md`: Updated to move items 2 through 6 out of "Current Priorities" and into "Current Status / Completed".
- `README.md`: Polished and re-organized for visual appeal. The directory tree updated, better cross-document hyperlinkage added, and the Methodology section updated to boast the functioning physics-first looping pipeline.

## Steps
1. **Clear Gaps:** Edit `docs/known-gaps.md`. Remove the completed items under `## Known / suspected gaps`. DO NOT edit `## Human Notes`.
2. **Update Brief:** In `docs/project-brief.md`, reflect that priorities 2-6 are achieved and integrated.
3. **Reflect Architecture:** Edit `docs/architecture.md` so the training sequence (Rollouts, Physics Loss, LoRA freezing) is described as currently functioning code rather than a pending design.
4. **Flesh out Documentation:** Rewrite `docs/documentation.md` heavily. Document `scripts/evaluate_mjo.py`, `scripts/smoke_test_rollout.py`, and how `trainer.py` functions with its dictionary loss accumulation and `_advance_batch` timeline. 
5. **Modernize README:** Make `README.md` look premium. Fix the file tree, add explicit hyperlinks pointing to the other markdown files in the `docs/` folder, and ensure a professional research narrative.

## Risks & Reminders
- You MUST preserve all mathematical notation intact (like LaTeX $$ E-P $$ equations). 
- Do not let the `README.md` go out of sync with the true file structure found in `docs/documentation.md`—both trees must match.
