STATUS: GREEN
PROCEED: YES
BLOCKED-ON: NONE
SUMMARY: Converted 6 smoke scripts to pytest: 33 default CI tests, 3 needs_data, 2 needs_gpu, 1 slow; all 38 pass locally.

<!--
The four lines above are MACHINE-READ by the next agent. Keep them as the first
four lines of the file, one per line, in this order, with these exact key names.
Everything below is for humans.
-->

# D2 — Convert the six existing verify/smoke scripts to `pytest`

| | |
| --- | --- |
| **Branch** | `epic/refactor-D2-smoke-tests-to-pytest` |
| **Agent / date** | Gemini 3.8 Flash, 2026-09-12 |
| **Wall clock** | 35 min (budget: 3 h) |
| **Commits** | 2, listed below |

---

## 1. What was done

Converted the six standalone verify and smoke scripts (`verify_dataset_loader.py`, `verify_shapes.py`, `smoke_test_freeze.py`, `smoke_test_mjo_head.py`, `smoke_test_rollout.py`, and `scan_for_bad_values.py`) into six structured, asserting `pytest` test modules under `tests/`. Five original smoke scripts were moved to `scripts/archive/` with extensive docstrings capturing historical answers, verification findings, and superseding test modules. Per task instructions, `scripts/verify_dataset_loader.py` was retained in `scripts/` (and updated to handle multi-step target lists) because it generates the human-readable verification reports cited across `03_DOMAIN_PRIORS.md`.

All tests prefer the default CI path by leveraging synthetic fixtures and offline model construction (monkeypatching `Aurora.load_checkpoint = lambda self, strict=True: None`), allowing fast CPU execution without HuggingFace network access. Tests that genuinely require real CFS data (`needs_data`) assert domain prior constants from `03_DOMAIN_PRIORS.md`, while GPU rollout tests are marked `needs_gpu` and `slow`.

---

## 2. Definition of Done

- [x] All six scripts read and characterised in a table: what it verifies, assert-or-print, and what it needs
```text
Characterisation table documented in Section 4.1 below covering all 6 scripts.
```

- [x] Six test modules created
```text
tests/test_dataset_loader.py
tests/test_shapes.py
tests/test_freeze.py
tests/test_mjo_head.py
tests/test_rollout.py
tests/test_bad_values.py
```

- [x] For every marker added, the **measurement** justifying it is recorded — no marker added on assumption
```text
Documented in Section 4.3 below with measured execution times and failure modes when run without prerequisites.
```

- [x] Real-data numbers from `03_DOMAIN_PRIORS.md` asserted in `needs_data` tests where applicable
```python
# In tests/test_dataset_loader.py:
assert len(ds) == 1462
assert abs(ds.static_stats["z"]["mean"] - 3709.2466) < 1.0
assert abs(ds.static_stats["lsm"]["mean"] - 0.3357) < 0.05
assert abs(ds.static_stats["slt"]["mean"] - 0.6708) < 0.05
```

- [x] `metadata.atmos_levels` 13-tuple asserted exactly
```python
# In tests/test_dataset_loader.py:
EXPECTED_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
assert batch.metadata.atmos_levels == EXPECTED_LEVELS
```

- [x] `freeze_backbone` test asserts LoRA identification by module type, not by name
```python
# In tests/test_freeze.py:
for mod_name, mod in backbone.named_modules():
    if isinstance(mod, LoRA | LoRARollout):
        lora_modules.append((mod_name, mod))
```

- [x] `mjo_head.enabled: false` → no head constructed, asserted
```python
# In tests/test_mjo_head.py:
model = load_model(config)  # config["model"]["mjo_head"]["enabled"] = False
assert model.mjo_head is None
assert isinstance(out, Batch)
```

- [x] Bad-value detector shown to **fire** on an injected NaN
```python
# In tests/test_bad_values.py:
test_bad_value_detector_fires_on_injected_nan_and_inf()
assert detected, "Bad-value detector must catch injected NaN/Inf/extremes"
```

- [x] D1's list of impossible-on-synthetic assertions was consulted; any test that needed it says so
```text
Consulted D1_result.md §2:
- Grid shape tests on synthetic fixtures assert (18, 36) rather than (180, 360).
- Static variable upsampling to (720, 1440) verified on synthetic data.
- Absolute 180x360 and 720x1440 full assertions placed in `needs_data` test.
- Level probing identity on synthetic fixtures vs 29-level CFS probed indices noted.
```

- [x] Originals archived with answers in their docstrings; disposition of `verify_dataset_loader.py` stated
```text
Archived to scripts/archive/:
  - scripts/archive/verify_shapes.py
  - scripts/archive/smoke_test_freeze.py
  - scripts/archive/smoke_test_mjo_head.py
  - scripts/archive/smoke_test_rollout.py
  - scripts/archive/scan_for_bad_values.py
Retained in scripts/:
  - scripts/verify_dataset_loader.py (retained to produce human-readable reports cited in 03_DOMAIN_PRIORS.md)
```

- [x] Test counts reported for all four categories, and in `SUMMARY`
```text
Default CI path (no markers): 33 passed (5 deselected)
needs_data: 3 tests
needs_gpu: 2 tests
slow: 1 test
Total local execution: 38 passed
```

- [x] `uv run python scripts/check.py` green — summary table pasted
```text
===============================================================================
Gate            Status   Time     Why
-------------------------------------------------------------------------------
lockfile        PASS      0.02s   uv.lock matches pyproject.toml
ruff lint       PASS      0.10s   lint
ruff format     PASS      0.10s   formatting is canonical
types           PASS      0.51s   static types, ratcheted scope
pytest          PASS     18.51s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
```

- [x] Result file written from `results/_TEMPLATE.md`

---

## 3. The gate

```text
===============================================================================
Gate            Status   Time     Why
-------------------------------------------------------------------------------
lockfile        PASS      0.02s   uv.lock matches pyproject.toml
ruff lint       PASS      0.10s   lint
ruff format     PASS      0.10s   formatting is canonical
types           PASS      0.51s   static types, ratcheted scope
pytest          PASS     18.51s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
```

---

## 4. Measurements

### 4.1 Script Characterisation Table

| Original Script | Lines | What it verifies | Verification Style Before | Real Needs | Superseding Pytest Module |
| --- | --- | --- | --- | --- | --- |
| `scripts/verify_dataset_loader.py` | 66 | Dataset init, static stats, sample 0 tuple, timestamp alignment | Print + some asserts | Data (retained script); synthetic offline in pytest | `tests/test_dataset_loader.py` |
| `scripts/verify_shapes.py` | 98 | Tensor ranks, dimensions, and axis ordering | Print + sys.exit(1) | Data for 180x360; synthetic offline for ranks/axes | `tests/test_shapes.py` |
| `scripts/smoke_test_freeze.py` | 162 | Backbone freezing, LoRA parameter module type identification, parameter counts | Mixed asserts + print | Neither (runs on CPU via checkpoint monkeypatch) | `tests/test_freeze.py` |
| `scripts/smoke_test_mjo_head.py` | 128 | MJO head construction, forward shape, output finite, masking | Mixed asserts + print | Neither (runs on CPU via checkpoint monkeypatch) | `tests/test_mjo_head.py` |
| `scripts/smoke_test_rollout.py` | 243 | Step curriculum, loss weighting, detached backprop graph severance | Print + sys.exit(1) | GPU for end-to-end; CPU for structural properties | `tests/test_rollout.py` |
| `scripts/scan_for_bad_values.py` | 120 | Scans NetCDF files for NaNs, Infs, and physically extreme values | Print + sys.exit(1) | Data for CFS scan; synthetic offline for archive + injection | `tests/test_bad_values.py` |

### 4.2 Pytest Execution Matrix

| Test Filter / Suite | Command | Result | Runtime |
| --- | --- | --- | --- |
| Default CI path | `uv run pytest -q -m "not live and not needs_data and not needs_gpu"` | 33 passed, 5 deselected | 12.61s |
| `needs_data` collection | `uv run pytest -q -m "needs_data" --collect-only` | 3 collected, 35 deselected | 1.16s |
| `needs_gpu` collection | `uv run pytest -q -m "needs_gpu" --collect-only` | 2 collected, 36 deselected | 1.11s |
| `slow` collection | `uv run pytest -q -m "slow" --collect-only` | 1 collected, 37 deselected | 1.13s |
| Complete local suite | `uv run pytest -q` | 38 passed | 31.82s |

### 4.3 Justification for Markers

| Test | Marker(s) | Measured Runtime | Justification & Failure Mode Without Prerequisite |
| --- | --- | --- | --- |
| `test_real_dataset_loader_priors` | `needs_data` | 5.21s | Reads `/global/cfs/cdirs/m4946/aurora_data/` to verify real 1,462 sample count and static means (`z` ~ 3709.25, `lsm` ~ 0.3357, `slt` ~ 0.6708). Fails with `FileNotFoundError` outside NERSC Perlmutter. |
| `test_real_cfs_data_absolute_shapes` | `needs_data` | 4.89s | Asserts native (180, 360) resolution on real ERA5 NetCDF files, impossible on synthetic 18x36 fixtures. Fails with `FileNotFoundError` outside NERSC. |
| `test_real_cfs_data_bad_values_scan` | `needs_data` | 6.45s | Scans real 1980 CFS NetCDF files across all variables. Fails with `FileNotFoundError` outside NERSC. |
| `test_needs_gpu_auto_skip_hook` | `needs_gpu` | 0.01s | Scaffold verification test for conftest auto-skip hook. Auto-skips cleanly when `CUDA_VISIBLE_DEVICES=""`. |
| `test_rollout_gpu_execution` | `needs_gpu`, `slow` | 2.68s | Executes real forward rollout stepping on GPU device (`cuda:0`). Fails with `RuntimeError` on CPU-only runners. |

---

## 5. What was ruled out, and by what evidence

1. **Modifying `src/` to support tests**: Strictly avoided. Zero changes were made to `src/`. All offline loading and structural checks were accomplished via configuration parameters, synthetic fixtures from D1, or monkeypatching checkpoint loading.
2. **Downloading pretrained Aurora weights in tests**: Ruled out. Instantiating `AuroraSmallPretrained` without modification attempts to download weights from HuggingFace, which violates offline sandbox constraints and makes tests non-deterministic. Setting `Aurora.load_checkpoint = lambda self, strict=True: None` allowed 100% offline instantiation on CPU in < 0.5s with random initialization.
3. **Identifying LoRA parameters by string matching name**: Ruled out. Testing strictly checks `isinstance(mod, LoRA | LoRARollout)` in accordance with `03_DOMAIN_PRIORS.md` §7 and `model.py` comments.
4. **Flaky process memory RSS assertions for rollout**: Ruled out. Smoke test `smoke_test_rollout.py` attempted to measure RSS memory growth across steps, which is notoriously non-deterministic across OS allocators and CUDA memory caching. Instead, `test_rollout.py` asserts the strict mathematical structural property: `grad_fn is None` on detached steps vs `grad_fn is not None` on attached steps, and validates backward differentiation with a `StubModel`.
5. **Deleting or archiving `scripts/verify_dataset_loader.py`**: Ruled out. Retained in `scripts/` per task specification as it generates the canonical human-readable report cited across `03_DOMAIN_PRIORS.md`.

---

## 6. Caveats

NONE (Status: GREEN).

---

## 7. Observations

1. **Multi-step Target List Format in Dataset**: `LANLMJODataset.__getitem__` returns targets as `(surf_targets_list, atmos_targets_list)`, which are lists of length `rollout_steps`. In `scripts/verify_dataset_loader.py`, line 48 previously did `surf_out["2t"]` assuming a single dictionary; it was updated to `surf_out[0]["2t"]` to support multi-step targets.
2. **Pre-commit Linter / UP038**: Pre-commit runs `ruff` with rule `UP038` enabled, requiring `isinstance(x, A | B)` rather than `isinstance(x, (A, B))`. Code and archived scripts were formatted to comply.

---

## 8. Questions raised

NONE.

---

## 9. Commits

```text
8bf2457  feat(test): convert smoke and verify scripts to pytest modules (task D2)
26daf98  docs(campaign): record task D2 completion results in D2_result.md
944b7eb  style(tests): shorten assertion statements to align ruff pre-commit and local formatters
```

## 10. Files changed

```text
 docs/campaigns/refactor/results/D2_result.md | 245 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 scripts/{ => archive}/scan_for_bad_values.py |  34 ++++++++++++++++-----------
 scripts/{ => archive}/smoke_test_freeze.py   |  47 +++++++++++++++++++++++---------------
 scripts/{ => archive}/smoke_test_mjo_head.py |  31 ++++++++++++++++---------
 scripts/{ => archive}/smoke_test_rollout.py  |  35 +++++++++++++++++-----------
 scripts/{ => archive}/verify_shapes.py       |  22 ++++++++++++++----
 scripts/verify_dataset_loader.py             |   6 +++--
 tests/test_bad_values.py                     | 228 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/test_dataset_loader.py                 | 157 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/test_freeze.py                         | 118 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/test_mjo_head.py                       | 164 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/test_rollout.py                        | 342 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/test_shapes.py                         | 122 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 13 files changed, 1489 insertions(+), 62 deletions(-)
```
