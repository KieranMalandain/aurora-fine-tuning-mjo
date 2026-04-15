# Task: Explicit MJO Head

## Objective
Extend `src/model.py` to support a dual-head architecture, predicting the standard Aurora rollout states plus an MJO-specific tuple `(RMM1, RMM2, Amplitude)`.

## Working rules
- Read only the files listed above unless you discover a direct dependency.
- If you need to expand scope, stop and report why.
- Make the smallest possible coherent patch.
- Keep all changes confined to the modify list unless explicitly authorized.

## Read
- `docs/architecture.md`
- `src/model.py`
- `src/trainer.py`
- `src/loss.py`
- `configs/phase1_baseline.yaml`
- any relevant model init code

## Modify
- `src/model.py`
- optionally config entries needed for the head

## Do not touch
- evaluation scripts
- dataset loading
- RMM computation
- unrelated docs

## Expected output
- optional head behind a flag
- correct checkpoint loading behavior
- smoke-test command

## Steps
1. **Pool features**: Extract the relevant tropical latent features or pre-decoder features from the Aurora backbone structure.
2. **Add MLP**: Attach a lightweight Multi-Layer Perceptron (MLP) to output `[RMM1, RMM2, Amplitude]`.
3. **Modify `forward`**: Adjust the forward pass of the wrapper to return `(decoded_states, mjo_predictions)`. Ensure `strict=False` in checkpoint loading continues to work properly with these new uninitialized weights.

## Risks & Reminders
- Make sure to review Aurora's internal forward pass logic so that pooling is applied intelligently (e.g. over the tropical lat/lon bounds rather than global bounds if appropriate).
- This is a blocker for the training loop having something to predict.
