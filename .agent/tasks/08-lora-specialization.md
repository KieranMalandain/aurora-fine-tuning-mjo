# Task: LoRA Frozen Subnetwork Verification

## Objective
Ensure that the LoRA specialization efficiently trains the newly initialized parts of the model (the MJO Head, TTR/TCWV embeddings) while keeping the massive Aurora backbone frozen properly against catastrophic forgetting.

## Working rules
- Read only the files listed above unless you discover a direct dependency.
- If you need to expand scope, stop and report why.
- Make the smallest possible coherent patch.
- Keep all changes confined to the modify list unless explicitly authorized.

## Read
- `docs/architecture.md`
- `src/model.py`
- `src/trainer.py` (where optimizer is built)

## Modify
- `src/model.py` (weight freezing logic)

## Do not touch
- Loss logic
- Evaluation scripts
- Dataloader

## Expected output
- Precise freezing logic in `load_model` iterating over `named_parameters()`.
- Unfrozen parameters check logging (e.g. printing trainable parameter counts vs total).
- Smoke-test command printing parameter states.

## Risks & Reminders
- Aurora might have a `use_lora=True` flag in its native codebase, but confirm whether it actually freezes the appropriate un-LoRA'd layers automatically or requires manual `requires_grad=False`.
- We MUST train the custom `MJOHead`.
