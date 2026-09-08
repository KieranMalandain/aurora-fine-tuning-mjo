# Task: Fix Optimizer Crash and Update Phase 1 Config

## Objective

Resolve the `ValueError: optimizer got an empty parameter list` crash during the smoke test (`train.py --smoke-test`) and ensure `configs/phase1_baseline.yaml` correctly specifies the training parameters for Phase 1 fine-tuning.

## Working rules
- Read only the files listed below unless you discover a direct dependency.
- If you need to expand scope, stop and report why.
- Make the smallest possible coherent patch.
- Keep all changes confined to the modify list unless explicitly authorized.
- Ensure conda/mamba environment `aurora_mjo` is activated before running Python code (directory `/pscratch/sd/k/kam352/conda/envs/aurora_mjo/bin/python`).

## Read

- `src/model.py`
- `src/trainer.py`
- `train.py`
- `configs/phase1_baseline.yaml`

## Modify

- `configs/phase1_baseline.yaml`
- `train.py`

## Do not touch

- Model architecture internals in `src/model.py`
- Dataset loading logic in `src/dataset.py`

## Expected output

- `configs/phase1_baseline.yaml` explicitly includes `freeze_backbone: false` under the `model:` block to allow full fine-tuning.
- Smoke test command `python train.py --config configs/phase1_baseline.yaml --smoke-test` runs to completion without throwing the optimizer error.

## Steps

1. In `configs/phase1_baseline.yaml`, add `freeze_backbone: false` to the `model` section so that the Aurora backbone is not completely frozen when `use_lora` is false.
2. In `train.py`, check the `_patch_config_for_smoke_test` function. If the smoke test still strips `ttr` and `tcwv`, ensure the model retains at least some trainable parameters so the AdamW optimizer doesn't crash. (Adding `freeze_backbone: false` should be sufficient since all backbone parameters will become trainable).
3. Run the smoke test: `/pscratch/sd/k/kam352/conda/envs/aurora_mjo/bin/python train.py --config configs/phase1_baseline.yaml --smoke-test` to verify it passes.

## Risks & Reminders

- Ensure `freeze_backbone: false` is intended for the Phase 1 baseline. If Phase 1 was meant to use LoRA instead, change `use_lora: false` to `use_lora: true` instead.
- Be careful not to leak real data dependencies into the smoke test.
