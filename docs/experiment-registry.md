# Experiment Registry

This file tracks all meaningful experiments.

## Template

### EXP-YYYYMMDD-XX
- Objective:
- Status:
- Branch:
- Config file:
- Data source:
- Split:
- Variables:
- Model variant:
- Losses:
- Rollout horizon:
- LoRA settings:
- Physics loss:
- Metrics tracked:
- Expected outcome:
- Actual outcome:
- Notes:

---

## Entries

### EXP-20260414-01
- Objective: Implement Phase 1 single-step supervised training loop (training scaffolding only; no actual training run yet)
- Status: Code complete — awaiting first run on NERSC or local dummy data
- Branch: agent-training-loop
- Config file: `configs/phase1_baseline.yaml`
- Data source: dummy (Yale Bouchet one-month ERA5) or real (NERSC LANL via `src/dataset.py`)
- Split: Train 1980–2015 / Val 2016–2019 / Test 2020–2023 (real); single-month Jan 2015 (dummy)
- Variables: 2t, 10u, 10v, msl, ttr, tcwv (surface); z, u, v, t, q (atmospheric, 13 levels)
- Model variant: AuroraSmallPretrained; use_lora=false
- Losses: TropicalWeightedL1Loss (grid); spectral/mjo_head/moisture_budget disabled
- Rollout horizon: 1 step (single-step supervised)
- LoRA settings: disabled
- Physics loss: disabled
- Metrics tracked: train loss, val loss per epoch
- Expected outcome: stable single-step training loop; decreasing val loss
- Actual outcome: TBD
- Notes: Smoke-test available via `python train.py --config configs/phase1_baseline.yaml --smoke-test`

---

### EXP-20260416-01
- Objective: Define baseline MJO evaluation protocol and implement `scripts/evaluate_mjo.py`
- Status: Script implemented — no real training run executed yet; baseline metrics TBD after first NERSC run
- Branch: agent-mjo-eval
- Config file: any `configs/*.yaml` (passed via `--config`)
- Data source: `data/rmm_targets.nc` + `data/rmm_basis.npz` (from `compute_rmm.py`); real data on NERSC
- Split: Val 2016–2019 (default); Test 2020–2023 (via `--split test`)
- Variables evaluated: RMM1, RMM2 (from MJO head or EOF projection of OLR/U850/U200)
- Model variant: Any AuroraMJO checkpoint (with or without MJO head)
- Losses: N/A (evaluation only)
- Rollout horizon: 1–30 days (120 × 6-hourly steps); controlled by `--max-lead-days`
- LoRA settings: N/A
- Physics loss: N/A
- Metrics tracked:
  - Bivariate RMM ACC vs lead day (Wheeler & Hendon 2004 formula)
  - RMSE of RMM1 and RMM2 vs lead day
  - Amplitude bias vs lead day
  - Mean absolute phase error (°) vs lead day
  - Active-MJO-only skill (amplitude₀ > 1.0) via `--also-active`
- Expected outcome: ACC > 0.5 at day 30 after Phase 2 rollout training
- Actual outcome: TBD
- Notes:
  - Smoke-test: `python scripts/evaluate_mjo.py --smoke-test`
  - Full run: `python scripts/evaluate_mjo.py --config configs/phase1_baseline.yaml --checkpoint <ckpt.pt> --targets data/rmm_targets.nc --basis data/rmm_basis.npz`
  - RMM extraction uses MJO head output (preferred) or EOF field projection (fallback)
  - All outputs written to `--out-dir` (default: `evaluation/mjo_skill/`)

---

