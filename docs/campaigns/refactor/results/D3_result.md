STATUS: GREEN
PROCEED: YES
BLOCKED-ON: NONE
SUMMARY: Implemented 7 lesson regression tests across 5 modules; 61 default CI tests + 1 xfail, 67 pass locally; gate green.

<!--
The four lines above are MACHINE-READ by the next agent. Keep them as the first
four lines of the file, one per line, in this order, with these exact key names.
Everything below is for humans.
-->

# D3 — A test for each of the seven paid-for lessons

| | |
| --- | --- |
| **Branch** | `epic/refactor-D3-regression-tests` |
| **Agent / date** | Gemini 3.8 Flash, 2026-09-12 |
| **Wall clock** | 45 min (budget: 3.5 h) |
| **Commits** | 6, listed below |

---

## 1. What was done

Implemented comprehensive regression test suites across five new test modules under `tests/` (`test_dataset_index.py`, `test_config_modes.py`, `test_norm_stats.py`, `test_grad_guard.py`, and `test_static_vars.py`), covering all seven paid-for lessons in `00_CONTEXT.md` §3. These tests lock down critical domain invariants: 6-hourly dataset indexing and leak-proof year filtering (Lesson 5), deterministic disabling of gradient checkpointing and `--override` coercion stability (Lesson 6), surface-pressure MSL proxy normalisation and placeholder detection (Lesson 1), gradient non-finiteness detection and Adam moment buffer preservation (Lesson 2), and HDF5 locking and zero-substitution fallback regression tracking (Lessons 3 & 4). Lesson 7 (plan against measured state) was identified as procedural and enforced by protocol rather than code.

All tests execute cleanly on CPU in the offline sandbox using synthetic fixtures, with real CFS archive verification preserved under `@pytest.mark.needs_data`.

---

## 2. Definition of Done

- [x] Five test modules created
```text
tests/test_dataset_index.py
tests/test_config_modes.py
tests/test_norm_stats.py
tests/test_grad_guard.py
tests/test_static_vars.py
```

- [x] 1980 → 1462 and 1981 → 1458 asserted, with the 1981 derivation shown
```python
# In tests/test_dataset_index.py:
# 1980 (leap year): 366 days * 4 = 1,464 timesteps. k=1 rollout -> 1,464 - 2 = 1,462 samples.
assert len(ds_1980._timeline) == 1464
assert len(ds_1980) == 1462

# 1981 (non-leap year): 365 days * 4 = 1,460 timesteps. k=1 rollout -> 1,460 - 2 = 1,458 samples.
assert len(ds_1981._timeline) == 1460
assert len(ds_1981) == 1458
```

- [x] Year-range enforcement asserted against a fixture containing an adjacent year — **the 54,060-class bug is covered**
```python
# In tests/test_dataset_index.py::test_year_range_enforcement_prevents_validation_leak:
# Fixture contains both 1980 and 1981 data. Naive glob would return 2,922 samples.
ds = LANLMJODataset(start_year=1980, end_year=1980, root_dir=synthetic_root, ...)
assert len(ds) == 1462
for ts in ds._timeline:
    assert datetime.fromtimestamp(int(ts), tz=timezone.utc).year == 1980
```

- [x] Per-variable index independence asserted against the chunking asymmetry
```python
# In tests/test_dataset_index.py::test_per_variable_index_independence_and_chunking_asymmetry:
map_ids = [id(m) for m in ds._var_ts_map.values()]
assert len(map_ids) == len(set(map_ids))  # all distinct dict objects
assert len(ds.surf_file_map["2t"][0]) == 1  # 1 yearly file
assert len(ds.atmos_file_map["q"][0]) == 2   # 2 half-yearly files
assert len(ds._var_ts_map["2t"]) == 1464
assert len(ds._var_ts_map["q"]) == 1464
```

- [x] Gapped-fixture sample reduction asserted as an **exact** number, with the computation shown
```python
# In tests/test_dataset_index.py::test_gapped_fixture_sample_reduction_exact:
# Derivation: 56 missing timesteps split timeline into 2 blocks (N1 + N2 = 1,408).
# Each block loses 2 boundary steps: (N1-2) + (N2-2) = 1,408 - 4 = 1,404 samples.
# Sample reduction = 1,462 - 1,404 = 58 samples (56 missing + 2 preceding boundary).
assert len(ds_std) == 1462
assert len(ds_gap) == 1404
assert len(ds_std) - len(ds_gap) == 58
```

- [x] 6-hour spacing asserted; train/val timestamp disjointness asserted
```python
# In tests/test_dataset_index.py:
assert np.all(diffs == 21600)  # exact 6-hour spacing across samples
assert len(set(train_ds._timeline.tolist()) & set(val_ds._timeline.tolist())) == 0
```

- [x] `gradient_checkpointing == false` asserted in all four modes
```python
# In tests/test_config_modes.py::test_gradient_checkpointing_false_in_all_modes:
for mode in ("baseline", "physics_informed", "lora", "combined"):
    cfg = load_config(config_path, mode=mode)
    assert cfg.get("model", {}).get("gradient_checkpointing") is False
```

- [x] Distinct `save_dir` per mode asserted; `init_from` chain asserted
```python
# In tests/test_config_modes.py:
assert save_dirs == {
    "baseline": "checkpoints/baseline",
    "physics_informed": "checkpoints/physics_informed",
    "lora": "checkpoints/lora",
    "combined": "checkpoints/combined",
}
assert cfg_physics["experiment"]["init_from"] == "latest:checkpoints/baseline"
assert cfg_lora["experiment"]["init_from"] == "latest:checkpoints/baseline"
assert cfg_combined["experiment"]["init_from"] == "latest:checkpoints/lora"
```

- [x] `--override` coercion behaviour locked against B1's fingerprint
```python
# In tests/test_config_modes.py::test_apply_overrides_locked_against_b1_fingerprint:
overridden = apply_overrides(raw_cfg, ["training.optimizer.lr=1e-5"])
assert overridden["training"]["optimizer"]["lr"] == "1e-5"  # PyYAML coercion to string
assert overridden == baseline_fingerprint["config_baseline_override"]
```

- [x] `msl` override presence asserted, **and** shown to reach Aurora's `locations`/`scales`
```python
# In tests/test_norm_stats.py:
for mode in MODES:
    assert "msl" in load_config(config_path, mode=mode)["model"]["norm_stats"]
load_model(model_cfg, norm_stats=model_cfg.get("norm_stats"))
assert locations["msl"] == model_cfg["norm_stats"]["msl"]["mean"]
assert scales["msl"] == model_cfg["norm_stats"]["msl"]["std"]
```

- [x] Placeholder detection present as `xfail`/warning, not a hard failure, with the reasoning stated
```python
# In tests/test_norm_stats.py::test_placeholder_norm_stats_guard:
if is_placeholder:
    pytest.xfail(
        "msl norm_stats in configs/unified.yaml are PLACEHOLDER values "
        "(96667.9822, 9504.6359). Must be replaced by computing real statistics "
        "over 1980–2015 using `scripts/calc_norm_stats.py` before executing production runs."
    )
```

- [x] The −30 σ / −6 σ sigma arithmetic asserted
```python
# In tests/test_norm_stats.py::test_sigma_arithmetic_tibetan_plateau:
sigma_builtin = (52000.0 - 100958.0) / 1332.0  # -36.75 sigma
assert sigma_builtin < -30.0
sigma_override = (52000.0 - 96667.9822) / 9504.6359  # -4.70 sigma
assert -6.0 < sigma_override < -4.0
assert (9504.6359 / 1332.0) > 7.0  # scale is >7x larger due to topography variance
```

- [x] Grad guard: skip on NaN, Adam buffers unchanged, scheduler not stepped, counter incremented, **and** finite step still steps
```python
# In tests/test_grad_guard.py:
assert trainer._nonfinite_grad_steps == 1
assert torch.equal(trainer.model.fc.weight, init_weight)
assert trainer.optimizer.param_groups[0]["lr"] == init_lr
assert "exp_avg" not in param_state  # Adam buffer not allocated/poisoned
# In test_grad_guard_selective_finite_step_advances:
assert not torch.equal(trainer.model.fc.weight, init_weight)
assert torch.isfinite(param_state["exp_avg"]).all()
```

- [x] DDP-collective coverage gap recorded explicitly if untestable
```text
Documented in test_grad_guard.py::test_collective_all_finite_single_rank_and_ddp_gap:
"DDP COVERAGE GAP: In a distributed multi-GPU training setup, `_collective_all_finite` executes:
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
to ensure all ranks step-or-skip collectively... In the single-process CPU test environment,
dist.is_initialized() is False, so the method directly returns local_finite. Testing the collective
reduction requires a multi-GPU torchrun environment."
```

- [x] `slt` truncate-not-upsample asymmetry asserted
```python
# In tests/test_static_vars.py::test_slt_truncate_not_upsample_asymmetry:
assert ds_z["Z"].shape[-2:] == (18, 36)
assert ds_lsm["LSM"].shape[-2:] == (18, 36)
assert ds_slt["slt"].shape[-2:] == (720, 1440)  # already 0.25 deg on disk
```

- [x] `_clean` NaN-zeroing asserted for all three statics
```python
# In tests/test_static_vars.py::test_clean_zeroes_nan_and_inf_across_all_three_statics:
for var_name in ("z", "lsm", "slt"):
    assert not torch.isnan(cleaned_statics[var_name]).any()
    assert not torch.isinf(cleaned_statics[var_name]).any()
```

- [x] Current zero-fallback behaviour asserted, with a docstring saying E1 inverts it
```python
# In tests/test_static_vars.py::test_current_zero_fallback_on_unreadable_invariant:
with pytest.warns(UserWarning, match=r"Using zeros"):
    statics = synthetic_dataset._load_static_vars()
assert (statics["z"] == 0.0).all()
# Docstring specifies: "CURRENT BEHAVIOUR (WRONG): Lesson 4 warns that substituting zeros hides
# I/O and locking failures. Task E1 will turn this into a hard failure... WHEN TASK E1 IMPLEMENTS
# THE FIX, THIS TEST MUST BE INVERTED TO ASSERT THAT AN EXCEPTION IS RAISED."
```

- [x] Lesson 7's no-test decision stated
```text
Lesson 7 (plan against measured state, not remembered state) is procedural rather than code.
It is enforced by 04_AGENT_PROTOCOL.md §2 and gate C4's verification, rather than a synthetic test.
```

- [x] Test counts by category reported and in `SUMMARY`
```text
Default CI path (no markers): 61 passed, 1 xfailed (6 deselected)
needs_data: 4 tests
needs_gpu: 2 tests
slow: 1 test
Total local execution: 67 passed, 1 xfailed
```

- [x] `uv run python scripts/check.py` green — summary table pasted
```text
===============================================================================
Gate            Status   Time     Why
-------------------------------------------------------------------------------
lockfile        PASS      0.02s   uv.lock matches pyproject.toml
ruff lint       PASS      0.11s   lint
ruff format     PASS      0.11s   formatting is canonical
types           PASS      0.50s   static types, ratcheted scope
pytest          PASS     23.09s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
```

- [x] Result file written from `results/_TEMPLATE.md`

---

## 3. The gate

```text
=== Gate: lockfile (uv lock --check) ===
Resolved 100 packages in 1ms

=== Gate: ruff lint (uv run ruff check .) ===
All checks passed!

=== Gate: ruff format (uv run ruff format --check .) ===
15 files already formatted

=== Gate: types (uv run pyrefly check) ===
 WARN PYTHONPATH environment variable is set to `/opt/nersc/pymon`. Checks in other environments may not include these paths.
 INFO Checking project configured at `/pscratch/sd/k/kam352/Aurora/aurora-fine-tuning-mjo/pyproject.toml`
 INFO 0 errors (1 suppressed, 2 warnings not shown)                                                                                                                                                                                                                                                                                 

=== Gate: pytest (uv run pytest -q -m not live and not needs_data and not needs_gpu) ===
..............................................x...............                                                                                                                                                                                                                                                               [100%]
========================================================================================================================================================= warnings summary =========================================================================================================================================================
tests/test_bad_values.py::test_scan_synthetic_archive_has_no_bad_values
  <frozen importlib._bootstrap>:241: RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility. Expected 16 from C header, got 96 from PyObject

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
61 passed, 6 deselected, 1 xfailed, 1 warning in 16.87s

===============================================================================
Gate            Status   Time     Why
-------------------------------------------------------------------------------
lockfile        PASS      0.02s   uv.lock matches pyproject.toml
ruff lint       PASS      0.11s   lint
ruff format     PASS      0.11s   formatting is canonical
types           PASS      0.50s   static types, ratcheted scope
pytest          PASS     23.09s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
```

---

## 4. Measurements

### 4.1 Sample Count and Gap Arithmetic
| Dataset Variant | Year Range | Days | Raw Timesteps | Usable Timeline | Valid k=1 Starts | Derivation / Provenance |
| --- | --- | --- | --- | --- | --- | --- |
| Leap year standard | 1980 | 366 | 1,464 | 1,464 | 1,462 | 1,464 - 2 = 1,462 (MEASURED) |
| Non-leap standard | 1981 | 365 | 1,460 | 1,460 | 1,458 | 1,460 - 2 = 1,458 (DERIVED) |
| 1980 gapped variant | 1980 | 366 | 1,464 (tcwv: 1,408) | 1,408 | 1,404 | (N1 - 2) + (N2 - 2) = 1,408 - 4 = 1,404 |
| Sample gap reduction | — | — | -56 | -56 | -58 | 56 gap timesteps + 2 boundary steps |

### 4.2 Sigma Arithmetic (Lesson 1)
| Condition | Normalisation Location | Normalisation Scale | 52,000 Pa Tibetan Input | Normalized Output | Effect |
| --- | --- | --- | --- | --- | --- |
| Aurora built-in MSL | 100,958.0 Pa | 1,332.0 Pa | 52,000.0 Pa | **-36.755 σ** | 100% NaN validation loss |
| Config override (placeholder) | 96,667.9822 Pa | 9,504.6359 Pa | 52,000.0 Pa | **-4.700 σ** | Stable, finite forward pass |

### 4.3 Pytest Execution Matrix
| Test Filter / Suite | Command | Result | Runtime |
| --- | --- | --- | --- |
| Default CI path | `uv run pytest -q -m "not live and not needs_data and not needs_gpu"` | 61 passed, 1 xfailed, 6 deselected | 16.87s |
| `needs_data` collection | `uv run pytest -q -m "needs_data" --collect-only` | 4 collected, 64 deselected | 1.12s |
| `needs_gpu` collection | `uv run pytest -q -m "needs_gpu" --collect-only` | 2 collected, 66 deselected | 1.11s |
| `slow` collection | `uv run pytest -q -m "slow" --collect-only` | 1 collected, 67 deselected | 0.98s |
| Complete local suite | `uv run pytest -q` | 67 passed, 1 xfailed | 42.48s |

---

## 5. What was ruled out, and by what evidence

1. **Modifying `src/` or `configs/unified.yaml`**: Strictly ruled out per task instructions. Zero modifications were made to library or config code.
2. **Treating Lesson 7 as an executable code test**: Ruled out. Lesson 7 ("plan against measured state, not remembered state") is a procedural discipline enforced by agent protocol and verified at C4, not a unit test.
3. **Hard-failing the placeholder normalisation test**: Ruled out. Generating real normalisation statistics over 1980–2015 is explicitly out of scope for Phase D (`00_CONTEXT.md` §5). Making the placeholder detection a hard red gate would violate the zero-broken-windows rule and train developers to ignore red CI. It was properly implemented as `pytest.xfail` referencing `scripts/calc_norm_stats.py`.
4. **Fixing the static variables zero-substitution fallback in this task**: Ruled out. E1 owns the fix. D3 strictly asserts current behavior (emitting `UserWarning` matching `Using zeros` and returning zero tensors) with clear docstrings alerting E1's implementer to invert the test when raising an exception.

---

## 6. Caveats

NONE (Status: GREEN).

---

## 7. Observations

1. **Assertion statement formatting between pre-commit ruff (0.6.9) and local ruff (0.16.6)**: Multiline assert statements with trailing comma messages (e.g. `assert (cond), "msg"`) are formatted differently by ruff 0.6.9 (parenthesizing condition) vs ruff 0.16.6 (parenthesizing message). Shortening assertion messages to keep `assert cond, "msg"` under 100 characters on a single line resolves the formatting divergence completely.
2. **Entanglement of `scripts/calc_norm_stats.py`**: Per `04_AGENT_PROTOCOL.md` layout rules (`scripts/` is run, never imported), `calc_norm_stats.py` is not directly importable from tests. Its pure logic could be extracted into the package in a future campaign if interactive testing of Welford accumulation is required.
3. **Multi-rank DDP collective grad-guard test**: In single-process CI, `_collective_all_finite` executes the single-rank branch (`dist.is_initialized() == False`). A complete multi-rank test verifying that all ranks synchronize a skip decision requires a multi-GPU SLURM or `torchrun` test harness.

---

## 8. Questions raised

NONE.

---

## 9. Commits

```text
af24256  test(d3): add Lesson 5 dataset indexing and year range regression tests
6bebebc  test(d3): add Lesson 6 checkpointing and config modes regression tests
a713e77  test(d3): add Lesson 1 normalisation stats and msl proxy regression tests
cd8265d  test(d3): add Lesson 2 non-finite gradient guard regression tests
2fd21f6  test(d3): add Lessons 3 and 4 static vars and locking regression tests
f46d07a  style(tests): shorten assertion statements to align ruff pre-commit and local formatters
```

## 10. Files changed

```text
 tests/test_config_modes.py  | 171 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/test_dataset_index.py | 239 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/test_grad_guard.py    | 231 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/test_norm_stats.py    | 195 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/test_static_vars.py   | 200 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 5 files changed, 1036 insertions(+)
```
