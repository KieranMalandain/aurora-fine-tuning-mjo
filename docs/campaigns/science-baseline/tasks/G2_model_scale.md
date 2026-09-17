# G2 — `small` / `full`; retire `huge`; measure the 1.3B model at 1°

| | |
| --- | --- |
| **Phase** | G |
| **Depends on** | G1 (`PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 4 h, dominated by queue wait |
| **Compute** | **4 GPU-hours.** `sbatch` permitted for the memory/throughput probe only, with human approval per `04_AGENT_PROTOCOL.md` §10. |
| **Touches** | `src/aurora_mjo/model.py`, `src/aurora_mjo/config.py`, `src/aurora_mjo/cli_support.py`, `configs/unified.yaml`, `scripts/probe_model_size.py`, `tests/test_config_validation.py`, `tests/test_freeze.py`, `docs/PROJECT_STATE.md`, `03_DOMAIN_PRIORS.md` (§6 only) |
| **Must not touch** | `src/aurora_mjo/dataset.py`. `src/aurora_mjo/loss.py`. `src/aurora_mjo/rmm/`. Any `results/*.md`. |

## Objective

`model_type: full` selects `AuroraPretrained` and is what every production run
uses; `small` selects `AuroraSmallPretrained` and is confined to smoke tests and
CI; `huge` raises at config validation. The memory and step time of the 1.3B
model at 1° are **measured**, and Lesson 6's gradient-checkpointing constraint is
re-tested under the 16× lower memory pressure.

## Why this is a separate task

`00_CONTEXT.md` R4: every result this project has produced came from a model
whose vendor docstring says *"Should only be used for debugging."* while every
claim in `docs/papers/` is about a 1.3B model. Correcting that is one decision,
and it interacts with exactly one other thing — whether the 1.3B model fits —
which is why the benchmark lives here and not in a separate task.

## What you may assume

- G1 landed: the model receives 180×360, patch grid `(4, 45, 90)`, 16× fewer grid
  points than before (`results/G1_result.md`).
- `AuroraSmallPretrained` with six surface variables is **112,830,384**
  parameters (**MEASURED**, review 2026-09-13, matching B1/C4/F2).
- Checkpoint names, measured from `microsoft-aurora==1.8.0`:
  `Aurora` → `aurora-0.25-finetuned.ckpt`; `AuroraPretrained` →
  `aurora-0.25-pretrained.ckpt`; `AuroraSmallPretrained` →
  `aurora-0.25-small-pretrained.ckpt`.
- `03_DOMAIN_PRIORS.md` §6 gives **DERIVED, WEAK** estimates of ~5–12 GiB and
  ~1–3 s/step for `full` at 1°. Your job is to replace them with measurements.

## Steps

1. **Resolve Q-13 first.** `02_SCIENTIFIC_CONTRACT.md` §2.2 argues
   `AuroraPretrained` is the correct ERA5 base but could not confirm it. Check
   `https://microsoft.github.io/aurora/example_era5.html` and the Aurora docs,
   and **quote the confirming text verbatim** in your result file. If the docs
   are ambiguous, open an issue on microsoft/aurora as was done for #184 and mark
   this task `BLOCKED`. **Do not proceed on a guess** — this is hard-blocking
   under `04_AGENT_PROTOCOL.md` §8.
2. Change the mapping in `model.py:424`. `full` → `AuroraPretrained`,
   anything else → `AuroraSmallPretrained`. Remove `Aurora` from the import.
3. Add a Pydantic validator in `config.py` rejecting `huge` with a message that
   names `full` as the replacement **and says why** — that `huge` pointed at the
   HRES-analysis fine-tune. A bare `ValueError` teaches nobody.
4. Add `training.require_full_model: bool` and set it `true` in the production
   modes. Startup aborts if it is `true` and `model_type` resolved to `small`.
   `--smoke-test` forces `small` regardless of config.
5. Write resolved `model_type`, resolved checkpoint filename and the
   `microsoft-aurora` version into the checkpoint payload and the header line of
   `metrics.jsonl`.
6. **Measure the exact parameter count** of `AuroraPretrained` with the six
   surface variables, total and trainable, for both `use_lora` settings. Record
   the integer; `03_DOMAIN_PRIORS.md` §2.3 currently carries "~1.3 × 10⁹" as
   DERIVED and it should become MEASURED.
7. Run `scripts/probe_model_size.py` at 1° for `small` and `full`, with and
   without gradient checkpointing, batch size 1, on 4 × A100. Record peak
   memory and mean step time for each of the four cells.
8. **Re-test Lesson 6 explicitly.** Does gradient checkpointing still trigger the
   illegal memory access at 1°? If `full` fits *without* checkpointing, record
   that the constraint has dissolved rather than been worked around. If it still
   crashes, record the exact error and leave `gradient_checkpointing: false`.
   **This is the task's headline question.**
9. Update `03_DOMAIN_PRIORS.md` §6 with the measured values, relabelled
   **MEASURED**, and note which of the WEAK derived estimates were wrong and by
   how much.
10. Confirm `freeze_backbone` behaves identically on `AuroraPretrained` — the
    LoRA-module detection and `decoder.surf_heads` lookup are structural and
    should carry over, but `tests/test_freeze.py` currently only exercises
    `small`. Add a `needs_gpu`-marked case.

## Definition of Done

- [ ] **Q-13 resolved with a verbatim quote**, or task marked `BLOCKED` with the
      opened issue linked
- [ ] `model_type: full` → `AuroraPretrained`; `huge` raises with an explanatory
      message; test added in `tests/test_config_validation.py`
- [ ] `--smoke-test` forces `small`; production launch with `small` aborts; both
      tested
- [ ] Model provenance written into checkpoints and `metrics.jsonl`; sample
      header pasted
- [ ] **Exact** parameter count for `AuroraPretrained` (6 surf vars), total and
      trainable, LoRA on and off, pasted
- [ ] Four-cell memory and step-time table at 1° pasted: `{small, full}` ×
      `{ckpt on, ckpt off}`
- [ ] Lesson 6 verdict stated plainly: does checkpointing still trigger the IMA
      at 1°, yes or no, with the error text if yes
- [ ] `03_DOMAIN_PRIORS.md` §6 updated to MEASURED; wrong estimates named
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] Discontinuity section written — parameter count, peak memory, step time
- [ ] `docs/PROJECT_STATE.md` updated
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- Any training run. This task measures; it does not learn.
- Changing the LoRA rank, `lora_mode`, or which modules are adapted — that is H4.
- Re-running the full `repro_ima_matrix` sweep. You are answering one question
  (does it crash at 1°), not characterising the crash.
- Enabling gradient checkpointing in `configs/unified.yaml` even if it works. G2
  records the finding; a later task acts on it if there is a reason to.
