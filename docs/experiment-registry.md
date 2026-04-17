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

### EXP-20260415-01
- Objective: Phase 2 autoregressive rollout training — multi-step curriculum, k grows 1→4 over epochs
- Status: Code complete — not yet trained
- Branch: agent-rollout-training
- Config file: `configs/phase2_rollout.yaml`
- Data source: same as phase1 (dummy or real depending on `use_dummy` flag)
- Split: Train 1980–2015 / Val 2016–2019 (real); single-month dummy otherwise
- Variables: same as phase1 (2t, 10u, 10v, msl, ttr, tcwv; z, u, v, t, q; 13 levels)
- Model variant: AuroraSmallPretrained; use_lora=false
- Losses: TropicalWeightedL1Loss (grid) accumulated uniformly across rollout steps; others disabled
- Rollout horizon: start=1, max=4; grows by 1 every 2 epochs (curriculum)
- LoRA settings: disabled
- Physics loss: disabled
- Metrics tracked: train loss, val loss per epoch; rollout_k logged per batch
- Expected outcome: loss-per-step should decrease as rollout horizon grows; subseasonal skill improvement
- Actual outcome: TBD
- Notes: |
    Smoke-test (no data required): `python scripts/smoke_test_rollout.py`
    Full training: `python train.py --config configs/phase2_rollout.yaml`
    State advance logic: _advance_batch() rolls the 2-step history window and increments metadata.time by 6h per step.
    Weighting strategy configurable: "uniform" (default) or "final_heavy" (last step = 50% of loss).

---

