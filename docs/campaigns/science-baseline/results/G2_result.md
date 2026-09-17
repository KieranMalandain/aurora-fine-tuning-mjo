STATUS: GREEN
PROCEED: YES
BLOCKED-ON: NONE
SUMMARY: model_type: full mapped to AuroraPretrained (1.3B); 1° peak mem 14.69 GiB; Lesson 6 constraint dissolved.

<!--
The four lines above are MACHINE-READ by the next agent. Keep them as the first
four lines of the file, one per line, in this order, with these exact key names.
Everything below is for humans.
-->

# G2 — `small` / `full`; retire `huge`; measure the 1.3B model at 1°

| | |
| --- | --- |
| **Branch** | `epic/science-g2-model-scale` |
| **Agent / date** | Antigravity, 2026-09-17 |
| **Wall clock** | 1 h 15 min (budget: 4 h) |
| **Commits** | 1 milestone commit, listed below |

---

## 1. What was done

1. **Resolved Q-13 via official Microsoft Aurora documentation:**
   Confirmed that `AuroraPretrained` is the authentic ERA5 1.3B base model (`aurora-0.25-pretrained.ckpt`), whereas `Aurora` (`aurora-0.25-finetuned.ckpt`, previously labeled `huge`) was fine-tuned specifically on operational IFS HRES T0 analysis and is unsuitable for ERA5 initial-value forecasting.
2. **Mapped `model_type: full` to `AuroraPretrained` and retired `huge`:**
   In `src/aurora_mjo/model.py`, mapped `model_type: full` directly to `AuroraPretrained` (1.3B parameters) and all other variants (e.g. `small`) to `AuroraSmallPretrained`. Retired `huge` in `src/aurora_mjo/config.py` with an actionable Pydantic validator explaining the HRES vs ERA5 distinction.
3. **Enforced model scale in production and smoke tests:**
   Added `training.require_full_model: bool = False` to `TrainingConfig` and enabled it across all production modes (`baseline`, `physics_informed`, `lora`, `combined`) in `configs/unified.yaml`. Validated that starting a production training run with `model_type: small` immediately aborts, while `--smoke-test` automatically forces `model_type: small` and disables `require_full_model`.
4. **Recorded provenance tracking:**
   Attached `model.model_type`, `model.checkpoint_name`, and `model.aurora_version` to loaded models. Guaranteed persistence into both the checkpoint dictionary payload (`CheckpointManager.save`) and the initial JSON record of `metrics.jsonl`.
5. **Benchmarked memory, throughput, and Lesson 6 at 1° native resolution:**
   Executed 4-cell matrix (`{small, full} × {ckpt on, ckpt off}`) using `scripts/probe_model_size.py` on an A100-SXM4-80GB GPU. Re-tested Lesson 6 (the gradient-checkpointing illegal memory access crash): at 1° native resolution, full 1.3B training with checkpointing completed 30 forward/backward steps with 0 errors. More importantly, `full` requires only **14.69 GiB** peak VRAM with checkpointing OFF (81.6% memory headroom), completely dissolving the checkpointing constraint.
6. **Updated documentation and test suite:**
   Updated `03_DOMAIN_PRIORS.md` §6 with measured memory and step times. Updated `PROJECT_STATE.md`. Added comprehensive unit tests in `tests/test_config_validation.py` and a GPU test `test_freeze_backbone_full_model_gpu` in `tests/test_freeze.py`.

---

## 2. Definition of Done

- [x] **Q-13 resolved with a verbatim quote**, or task marked `BLOCKED` with the
      opened issue linked
- [x] `model_type: full` → `AuroraPretrained`; `huge` raises with an explanatory
      message; test added in `tests/test_config_validation.py`
- [x] `--smoke-test` forces `small`; production launch with `small` aborts; both
      tested
- [x] Model provenance written into checkpoints and `metrics.jsonl`; sample
      header pasted
- [x] **Exact** parameter count for `AuroraPretrained` (6 surf vars), total and
      trainable, LoRA on and off, pasted
- [x] Four-cell memory and step-time table at 1° pasted: `{small, full}` ×
      `{ckpt on, ckpt off}`
- [x] Lesson 6 verdict stated plainly: does checkpointing still trigger the IMA
      at 1°, yes or no, with the error text if yes
- [x] `03_DOMAIN_PRIORS.md` §6 updated to MEASURED; wrong estimates named
- [x] `uv run python scripts/check.py` is green; summary table pasted
- [x] Discontinuity section written — parameter count, peak memory, step time
- [x] `docs/PROJECT_STATE.md` updated
- [x] Result file written from `results/_TEMPLATE.md`

### Proof of DoD items

#### 1. Q-13 Resolution (Verbatim Quote)
From Microsoft Aurora official documentation (`https://microsoft.github.io/aurora/example_era5.html`):
> *"The fine-tuned version of Aurora specifically only works with IFS HRES T0, so we use the non-fine-tuned version of Aurora in this example."*

And from vendor code docstrings (`microsoft-aurora` v1.8.0):
- `AuroraSmallPretrained`: *"Aurora (small) with pre-trained weights. Should only be used for debugging."* Checkpoint: `aurora-0.25-small-pretrained.ckpt`.
- `AuroraPretrained`: *"Aurora with pre-trained weights."* Checkpoint: `aurora-0.25-pretrained.ckpt`.
- `Aurora`: *"Aurora fine-tuned on operational IFS data."* Checkpoint: `aurora-0.25-finetuned.ckpt`.

#### 2. Config validation for `model_type: huge`
```bash
$ uv run python -c '
from aurora_mjo.config import Config
try:
    Config.from_dict({"model": {"model_type": "huge"}})
except Exception as e:
    print(e)
'
1 validation error for Config
model.model_type
  Value error, 'model_type: huge' is deprecated and invalid. 'huge' referred to Aurora fine-tuned on operational IFS HRES analysis ('aurora-0.25-finetuned.ckpt'), which cannot ingest ERA5 data. Use 'model_type: full' to instantiate the 1.3B pretrained foundation model ('aurora-0.25-pretrained.ckpt'). [type=value_error, input_value='huge', input_type=str]
```

#### 3. Production check with `model_type: small` aborts; `--smoke-test` forces `small`
```bash
$ uv run pytest tests/test_config_validation.py -k "model_type or require_full_model or smoke_test"
======================== test session starts ========================
tests/test_config_validation.py ...                                   [100%]
========================= 3 passed in 0.28s =========================
```

#### 4. Model provenance sample header
Written to `checkpoints/baseline/metrics.jsonl`:
```json
{"type": "header", "model_type": "small", "checkpoint_name": "aurora-0.25-small-pretrained.ckpt", "aurora_version": "1.8.0", "timestamp": 1789666616.043683}
```
And verified in checkpoint payload dictionary:
```python
['aurora_version', 'checkpoint_name', 'model_state_dict', 'model_type']
model_type: 'full'
checkpoint_name: 'aurora-0.25-pretrained.ckpt'
aurora_version: '1.8.0'
```

#### 5. Exact Parameter Counts
Measured via parameter audit on `AuroraMJO` with 6 surface variables (`2t`, `10u`, `10v`, `msl`, `ttr`, `tcwv`):
- **Without LoRA (`use_lora: false`):**
  - Total Parameters: **1,256,365,744**
  - Trainable Parameters: **81,968** (0.007%)
  - Frozen Parameters: **1,256,283,776** (99.993%)
- **With LoRA (`use_lora: true`, `lora_mode: single`):**
  - Total Parameters: **1,259,216,560**
  - Trainable Parameters: **2,932,784** (0.233%)
  - Frozen Parameters: **1,256,283,776** (99.767%)

#### 6. Four-Cell Benchmark Table (1° Native Resolution on A100-SXM4-80GB)
Measured via `scripts/probe_model_size.py`:

| Model Size | Activation Checkpointing | Peak Memory | Mean Step Time | Trainable Params | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `small` (112M) | OFF | **3.12 GiB** | **76.8 ms** | 41,008 | OK (0 skips) |
| `small` (112M) | ON | **2.04 GiB** | **110.4 ms** | 41,008 | OK (0 skips) |
| `full` (1.3B) | OFF | **14.69 GiB** | **221.2 ms** | 81,968 | OK (0 skips) |
| `full` (1.3B) | ON | **12.52 GiB** | **302.1 ms** | 81,968 | OK (0 skips) |

#### 7. Lesson 6 Verdict
**Does gradient checkpointing still trigger the IMA crash at 1°?**
**No.** All 30 forward/backward steps with gradient checkpointing on `full` (1.3B) completed with 0 errors.

**Headline Verdict:**
The gradient checkpointing constraint has **dissolved**. At 1° native resolution, the full 1.3B model requires only **14.69 GiB** peak GPU memory with gradient checkpointing turned **OFF** on Perlmutter's 80 GiB A100 cards, leaving **81.6% memory headroom** (65.3 GiB unused). Checkpointing is therefore entirely unnecessary for single-step training.

---

## 3. The gate

```text
===============================================================================
Gate            Status   Time     Why
-------------------------------------------------------------------------------
lockfile        PASS      0.02s   uv.lock matches pyproject.toml
ruff lint       PASS      0.10s   lint
ruff format     PASS      0.10s   formatting is canonical
types           PASS      0.53s   static types, ratcheted scope
pytest          PASS     39.37s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
```

---

## 4. Measurements

### Comparison of Measured Values vs Prior Estimates

| Quantity | Prior Derived Estimate | Measured in G2 | Error / Discrepancy |
| :--- | :--- | :--- | :--- |
| `full` parameters | ~1.3 × 10⁹ (DERIVED) | **1,256,365,744** | Exact integer established |
| `full` peak mem (ckpt off) | ~5–12 GiB (WEAK) | **14.69 GiB** | Underestimated by 2.69 GiB (22%) due to static weight/head footprint |
| `full` step time (ckpt off) | ~1–3 s/step (WEAK) | **0.221 s/step** | Overestimated by 4.5×–13.5× (Swin3D windowed attention is fast at 1°) |
| Memory headroom (80 GB A100) | Unknown | **81.6% (65.3 GiB free)** | Massive headroom for batch size 1 |

---

## 5. Discontinuity Section

Per `04_AGENT_PROTOCOL.md` §5, transitioning from `small` (used across all prior tasks) to `full` creates a deliberate parameter, memory, and throughput discontinuity:

| Metric | Before G2 (`small`) | After G2 (`full`) | Factor Change |
| :--- | :--- | :--- | :--- |
| **Total Parameters (Base)** | 112,830,384 | **1,256,365,744** | **11.13× increase** |
| **Trainable Parameters (Base)** | 41,008 | **81,968** | **2.00× increase** (wider embedding/head channels) |
| **Peak VRAM (1°, ckpt off)** | 3.12 GiB | **14.69 GiB** | **4.71× increase** |
| **Step Time (1°, ckpt off)** | 76.8 ms | **221.2 ms** | **2.88× increase** |
| **Training Session Estimate (7.5k steps)** | ~10 min | **~28 min** | Well within single 3.5 h interactive session |

---

## 6. What was ruled out, and by what evidence

1. **`model_type: huge` ruled out:**
   Evidence from official Microsoft documentation (`https://microsoft.github.io/aurora/example_era5.html`) confirms `aurora-0.25-finetuned.ckpt` is trained on IFS HRES analysis, not ERA5.
2. **Gradient checkpointing ruled out for production:**
   Although checkpointing no longer crashes at 1°, it reduces peak memory from 14.69 GiB to 12.52 GiB (saving only 2.17 GiB) while penalizing step time by +36.5% (221 ms -> 302 ms). Since 65.3 GiB of VRAM is already free, enabling checkpointing provides zero benefit and degrades throughput.

---

## 7. Caveats

None. `STATUS: GREEN`.

---

## 8. Observations

1. **Lustre File Locking on Perlmutter:**
   `huggingface_hub`'s `WeakFileLock` relies on `fcntl.flock`, which triggers `OSError: [Errno 524]` on Perlmutter's scratch filesystem. In addition to setting `HF_HUB_DISABLE_FILE_LOCKING=1`, `cli_support.py` and `scripts/probe_model_size.py` patch `filelock` and `huggingface_hub.utils._fixes.WeakFileLock` with a dummy no-op lock. This ensures resilient model downloads on compute nodes.

---

## 9. Questions raised

None. **Q-13** was answered and closed in `QUESTIONS.md`.

---

## 10. Commits

```text
fabdcb4  feat(g2): map full to AuroraPretrained, retire huge, benchmark 1.3B at 1deg, update domain priors
```

---

## 11. Files changed

```text
 configs/unified.yaml                                |   6 +-
 docs/PROJECT_STATE.md                               |  22 +--
 docs/campaigns/science-baseline/03_DOMAIN_PRIORS.md |  32 +--
 docs/campaigns/science-baseline/QUESTIONS.md        |   7 +-
 docs/campaigns/science-baseline/results/G2_result.md| 233 ++++++++++++++++++++
 scripts/probe_model_size.py                         | 184 ++++++++--------
 src/aurora_mjo/cli_support.py                       | 109 +++++++--
 src/aurora_mjo/config.py                            |  18 ++
 src/aurora_mjo/model.py                             |  20 +-
 tests/test_config_validation.py                     |  53 +++++
 tests/test_freeze.py                                |  59 +++++
 11 files changed, 611 insertions(+), 132 deletions(-)
```
