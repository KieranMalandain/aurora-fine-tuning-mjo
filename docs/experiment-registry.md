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

### EXP-20260416-02
- Objective: LoRA frozen-subnetwork verification — ensure backbone is frozen and only LoRA adapters + new-var embeddings + MJO head train
- Status: Code complete — not yet run on NERSC
- Branch: agent-lora-spec
- Config file: any config with `use_lora: true` and `freeze_backbone: true` (default)
- Data source: N/A (freezing logic is architecture-only; no training run)
- Split: N/A
- Variables: surf: 2t, 10u, 10v, msl, ttr, tcwv; atmos: z, u, v, t, q (13 levels)
- Model variant: AuroraSmallPretrained; use_lora=true
- Losses: N/A (infrastructure change only)
- Rollout horizon: N/A
- LoRA settings: r=8, alpha=8, mode="single"; adapters on Swin3D qkv and proj
- Physics loss: disabled
- Metrics tracked: trainable / total parameter counts at model load time
- Expected outcome: ~95–99% of params frozen; LoRA + ttr/tcwv embeddings + MJO head trainable
- Actual outcome: TBD
- Notes: |
    - `freeze_backbone()` in `src/model.py`: blanket freeze → unfreeze LoRA → unfreeze new-var patch embeds
    - `_log_param_counts()` prints per-child audit on every `load_model()` call
    - Smoke test: `conda run -n aurora_mjo python scripts/smoke_test_freeze.py`
    - Set `config['freeze_backbone'] = false` to do full fine-tuning (not recommended pre-baseline)

---

