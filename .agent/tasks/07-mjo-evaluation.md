# Task: RMM Bivariate Correlation Evaluation Script

## Objective
Create a standalone evaluation script that systematically measures lead-dependent MJO skill (Anomalous Correlation Coefficient, Phase Error, Amplitude Error) across valid rollout ranges (1-30 days), matching Wheeler-Hendon methodology.

## Working rules
- Read only the files listed above unless you discover a direct dependency.
- If you need to expand scope, stop and report why.
- Make the smallest possible coherent patch.
- Keep all changes confined to the modify list unless explicitly authorized.

## Read
- `docs/evaluation-spec.md`
- `scripts/compute_rmm.py`
- `src/trainer.py`

## Modify
- `scripts/evaluate_mjo.py` (new file)
- `docs/experiment-registry.md` (to document the baseline metrics)

## Do not touch
- Training logic
- MJO head definitions

## Expected output
- A script pulling validation subset samples, unrolling forecasts up to 30 days, extracting MJO targets out of the head, and computing Bivariate ACC against `rmm_targets.nc`.
- Plots and tables generated representing MJO skill.
- Smoke-test command.

## Risks & Reminders
- Only evaluate active MJO cases (amplitude > 1.0) when specified by the spec.
- Be extremely careful about indexing between model rollout steps (6-hourly) and Wheeler-Hendon validation targets.
