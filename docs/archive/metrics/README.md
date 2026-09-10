# July 2026 Metrics Traces

These files record the loss and metric traces from the July 2026 runs on NERSC Perlmutter:
- `baseline-2026-07.jsonl`: metric trace from the baseline fine-tuning run.
- `lora-2026-07.jsonl`: metric trace from the LoRA fine-tuning run.

These runs are declared fully non-finite by `AURORA_MJO_GAMEPLAN.md`, so the model
checkpoints themselves were deliberately discarded. These traces are retained as
historical evidence of *how* the runs went non-finite (e.g. tracking loss spikes and
grad norms), which is valuable context for ongoing debugging and regression tests.

Previously, these files were tracked inside an otherwise gitignored `checkpoints/`
directory via `.gitignore` negations (`!checkpoints/*/metrics.jsonl`). That
configuration survives only until someone cleans up or simplifies `.gitignore`.
They are archived here permanently so they remain safely tracked outside `checkpoints/`.
