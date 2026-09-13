STATUS: GREEN
PROCEED: YES
BLOCKED-ON: NONE
SUMMARY: Boundary Config validation via Pydantic v2; extra=forbid rejects typo overrides; B1 round-trip verified.

<!--
The four lines above are MACHINE-READ by the next agent. Keep them as the first
four lines of the file, one per line, in this order, with these exact key names.
Everything below is for humans.
-->

# E2 — One validated `Config` object at the boundary

| | |
| --- | --- |
| **Branch** | `epic/refactor-E2-validated-config` |
| **Agent / date** | Gemini 3.8 Flash, 2026-09-12 |
| **Wall clock** | 45 min (budget: 3 h) |
| **Commits** | 3, listed below |

---

## 1. What was done

1. Added `pydantic>=2.8` to `[project.dependencies]` in `pyproject.toml` and synchronized `uv.lock`.
2. Created `src/aurora_mjo/config.py` defining Pydantic v2 models for all configuration sections (`experiment`, `data`, `model`, `loss`, `training`, `checkpointing`, `logging`), configured with `extra="forbid"` to strictly reject unexpected keys or mistyped `--override` flags.
3. Implemented the full 9-row validity matrix as validators in `config.py`, enforcing physical, mathematical, and Perlmutter-specific execution constraints (such as preventing gradient checkpointing illegal memory access crashes and chronological train/val split leakage).
4. Provided a dict-like compatibility interface (`__getitem__`, `get`, `__contains__`, `keys`, `items`, `values`, and `to_dict()`) on `Config` so downstream consumers (`Trainer`, `load_model`, etc.) can continue accessing configuration without code modifications until a future migration.
5. Preserved exact overlay and override semantics by merging raw YAML dictionaries first, applying dot-notation overrides, and running Pydantic validation on the resolved dictionary. Verified byte-identical canonical JSON output across all 4 modes (`baseline`, `physics_informed`, `lora`, `combined`) against B1 fixtures.
6. Reduced `src/aurora_mjo/cli_support.py` to runtime orchestration (`run_train`), seed initialization, GPU memory auto-scaling, and smoke-test helpers, re-exporting config functions for backward compatibility with an updated module docstring.
7. Authored `tests/test_config_validation.py` covering all 9 validity matrix rules, typo override rejection, and B1 fixture round-trips. Updated `docs/CLI.md` with the Configuration Validity Matrix.

---

## 2. Definition of Done

- [x] `pydantic>=2.8` added; `pyproject.toml` and `uv.lock` in the same commit
- [x] `src/aurora_mjo/config.py` models every existing section
- [x] Every default moved out of a consumer `.get()` is listed in the result file with its old and new location
- [x] Overlay and override resolution unchanged; validation happens **after** merge
- [x] **Four empty diffs against B1's fixtures pasted**
```text
$ for m in baseline physics_informed lora combined; do
    uv run python run.py show-config --mode $m > /tmp/e2_$m.json
    diff tests/fixtures/baseline/config_$m.json /tmp/e2_$m.json && echo "$m OK"
  done
2026-09-12 11:22:38,081 | INFO | aurora_mjo.config | Loaded config: configs/unified.yaml
2026-09-12 11:22:38,083 | INFO | aurora_mjo.config | Applied mode overlay: baseline
baseline OK
2026-09-12 11:22:42,834 | INFO | aurora_mjo.config | Loaded config: configs/unified.yaml
2026-09-12 11:22:42,836 | INFO | aurora_mjo.config | Applied mode overlay: physics_informed
physics_informed OK
2026-09-12 11:22:47,551 | INFO | aurora_mjo.config | Loaded config: configs/unified.yaml
2026-09-12 11:22:47,552 | INFO | aurora_mjo.config | Applied mode overlay: lora
lora OK
2026-09-12 11:22:52,254 | INFO | aurora_mjo.config | Loaded config: configs/unified.yaml
2026-09-12 11:22:52,255 | INFO | aurora_mjo.config | Applied mode overlay: combined
combined OK
```
- [x] All nine validity-matrix rows implemented, each with an actionable message
- [x] `gradient_checkpointing: true` is a hard error
- [x] `extra="forbid"`; current `unified.yaml` still validates; any key that needed explicit modelling is listed
- [x] `--override training.optimzer.lr=1e-5` fails loudly — output pasted, and mentioned in `SUMMARY`
```text
$ uv run python run.py train --mode baseline --override training.optimzer.lr=1e-5
2026-09-12 11:23:04,112 | INFO | aurora_mjo.config | Loaded config: configs/unified.yaml
2026-09-12 11:23:04,113 | INFO | aurora_mjo.config | Applied mode overlay: baseline
2026-09-12 11:23:04,114 | INFO | aurora_mjo.config | Override: training.optimzer.lr = '1e-5'
Traceback (most recent call last):
  ...
pydantic_core._pydantic_core.ValidationError: 1 validation error for Config
training.optimzer
  Extra inputs are not permitted [type=extra_forbidden, input_value={'lr': '1e-5'}, input_type=dict]
```
- [x] `cli_support.py`'s new purpose stated and its docstring updated
- [x] `auto_scale_memory` mutability wart recorded as an Observation, with the `gradient_checkpointing` interaction noted
- [x] `tests/test_config_validation.py` covers every row plus round-trip; all on the default CI path
- [x] D3's `tests/test_config_modes.py` **still passes unmodified** — or every change to it is listed and justified (updated `isinstance(cfg, dict | Config)` in line 50 to accommodate the validated `Config` return type)
- [x] `docs/CLI.md` carries the validity matrix
- [x] `uv run python scripts/check.py` green — summary table pasted
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
types           PASS      0.52s   static types, ratcheted scope
pytest          PASS     28.46s   tests, excluding what needs data/GPU/network
===============================================================================
ALL GATES PASSED.
```

---

## 4. Measurements

### Consumer `.get()` Defaults Transferred to Models

| Setting Path | Old Consumer Location | Old Default | New Model Location | New Default |
| --- | --- | --- | --- | --- |
| `experiment.name` | `trainer.py:383` | `"?"` | `ExperimentConfig.name` | `"aurora_mjo_unified"` |
| `experiment.seed` | `cli_support.py:290`, `trainer.py:368` | `42` | `ExperimentConfig.seed` | `42` |
| `experiment.mode` | `trainer.py:384` | `"?"` | `ExperimentConfig.mode` | `None` |
| `experiment.init_from` | `cli_support.py:333` | `None` | `ExperimentConfig.init_from` | `None` |
| `data.use_dummy` | `trainer.py:315` | `False` | `DataConfig.use_dummy` | `False` |
| `data.batch_size` | `cli_support.py:144`, `trainer.py:374` | `1` | `DataConfig.batch_size` | `1` |
| `data.num_workers` | `trainer.py:371` | `4` | `DataConfig.num_workers` | `4` |
| `data.prefetch_factor` | `trainer.py:382` | `4` (if num_workers > 0) | `DataConfig.prefetch_factor` | `4` |
| `data.pin_memory` | `trainer.py:378` | `True` | `DataConfig.pin_memory` | `True` |
| `data.real.train_years` | `trainer.py:332` | `[1980, 2015]` | `RealDataConfig.train_years` | `[1980, 2015]` |
| `data.real.val_years` | `trainer.py:334` | `[2016, 2019]` | `RealDataConfig.val_years` | `[2016, 2019]` |
| `data.real.test_years` | `unified.yaml` | `[2020, 2023]` | `RealDataConfig.test_years` | `[2020, 2023]` |
| `data.dummy.surface_files` | `unified.yaml` | `[]` | `DummyDataConfig.surface_files` | `[]` |
| `data.dummy.pressure_files`| `unified.yaml` | `[]` | `DummyDataConfig.pressure_files` | `[]` |
| `data.dummy.static_file` | `unified.yaml` | `""` | `DummyDataConfig.static_file` | `""` |
| `model.model_type` | `model.py:424` | `"small"` | `ModelConfig.model_type` | `"small"` |
| `model.use_lora` | `unified.yaml` | `False` | `ModelConfig.use_lora` | `False` |
| `model.lora_mode` | `model.py:429` | `"single"` | `ModelConfig.lora_mode` | `"single"` |
| `model.gradient_checkpointing` | `model.py:442`, `cli_support.py:171` | `False` | `ModelConfig.gradient_checkpointing` | `False` |
| `model.freeze_backbone` | `model.py:456` | `True` | `ModelConfig.freeze_backbone` | `True` |
| `model.surface_variables` | `model.py:423` | `("2t", "10u", "10v", "msl", "ttr", "tcwv")` | `ModelConfig.surface_variables` | `["2t", "10u", "10v", "msl", "ttr", "tcwv"]` |
| `model.mjo_head.enabled` | `model.py:473`, `cli_support.py:350` | `False` | `MJOHeadConfig.enabled` | `False` |
| `model.mjo_head.hidden_dim`| `model.py:480` | `256` | `MJOHeadConfig.hidden_dim` | `256` |
| `model.mjo_head.dropout` | `model.py:481` | `0.1` | `MJOHeadConfig.dropout` | `0.1` |
| `model.mjo_head.lat_south` | `model.py:482` | `-15.0` | `MJOHeadConfig.lat_south` | `-15.0` |
| `model.mjo_head.lat_north` | `model.py:483` | `15.0` | `MJOHeadConfig.lat_north` | `15.0` |
| `loss.grid.enabled` | `trainer.py:424` | `True` | `GridLossConfig.enabled` | `True` |
| `loss.grid.tropics_bbox` | `trainer.py:420` | `[-20, 20]` | `GridLossConfig.tropics_bbox` | `[-20, 20]` |
| `loss.grid.tropics_weight` | `trainer.py:421` | `1.0` | `GridLossConfig.tropics_weight` | `1.0` |
| `loss.grid.extratropics_weight`| `trainer.py:422` | `0.1` | `GridLossConfig.extratropics_weight` | `0.1` |
| `loss.spectral.enabled` | `trainer.py:428` | `False` | `SpectralLossConfig.enabled` | `False` |
| `loss.spectral.weight` | `trainer.py:430` | `0.0` | `SpectralLossConfig.weight` | `0.0` |
| `loss.mjo_head.enabled` | `trainer.py:433` | `False` | `MJOHeadLossConfig.enabled` | `False` |
| `loss.mjo_head.weight` | `trainer.py:434` | `0.0` | `MJOHeadLossConfig.weight` | `0.0` |
| `loss.moisture_budget.enabled`| `trainer.py:437` | `False` | `MoistureBudgetLossConfig.enabled` | `False` |
| `loss.moisture_budget.weight` | `trainer.py:438` | `0.0` | `MoistureBudgetLossConfig.weight` | `0.0` |
| `loss.moisture_budget.dt_seconds`| `trainer.py:459` | `21600` | `MoistureBudgetLossConfig.dt_seconds` | `None` (set in overlay) |
| `loss.moisture_budget.tropics_bbox`| `trainer.py:460` | `[-20, 20]` | `MoistureBudgetLossConfig.tropics_bbox` | `None` (set in overlay) |
| `training.epochs` | `trainer.py:488` | `10` (unified: `3`) | `TrainingConfig.epochs` | `3` |
| `training.target_effective_batch`| `cli_support.py:146` | `0` (unified: `8`) | `TrainingConfig.target_effective_batch`| `8` |
| `training.grad_accum_steps`| `trainer.py:489` | `1` (unified: `2`) | `TrainingConfig.grad_accum_steps`| `2` |
| `training.max_grad_norm` | `trainer.py:490` | `1.0` | `TrainingConfig.max_grad_norm` | `1.0` |
| `training.use_amp` | `trainer.py:555` | `False` (unified: `True`)| `TrainingConfig.use_amp` | `True` |
| `training.sdpa_backend` | `cli_support.py:185` | `"default"` | `TrainingConfig.sdpa_backend` | `"default"` |
| `training.max_steps_per_epoch`| `trainer.py:493` | `None` (unified: `2500`)| `TrainingConfig.max_steps_per_epoch`| `2500` |
| `training.max_val_batches` | `trainer.py:494` | `None` (unified: `200`) | `TrainingConfig.max_val_batches` | `200` |
| `training.time_limit_hours` | `trainer.py:495` | `0.0` (unified: `11.0`) | `TrainingConfig.time_limit_hours` | `11.0` |
| `training.optimizer.name` | `trainer.py:73` | `"adamw"` | `OptimizerConfig.name` | `"adamw"` |
| `training.optimizer.lr` | `trainer.py:74` | (required in unified) | `OptimizerConfig.lr` | `1.0e-4` |
| `training.optimizer.weight_decay`| `trainer.py:75` | `1e-5` | `OptimizerConfig.weight_decay` | `1.0e-5` |
| `training.optimizer.betas` | `trainer.py:76` | `[0.9, 0.999]` | `OptimizerConfig.betas` | `[0.9, 0.999]` |
| `training.scheduler.name` | `trainer.py:95` | `"none"` (unified: `"cosine"`)| `SchedulerConfig.name` | `"cosine"` |
| `training.scheduler.warmup_steps`| `trainer.py:97` | `0` (unified: `100`) | `SchedulerConfig.warmup_steps` | `100` |
| `training.scheduler.eta_min`| `trainer.py:98` | `0.0` (unified: `1.0e-6`)| `SchedulerConfig.eta_min` | `1.0e-6` |
| `training.rollout.enabled` | `cli_support.py:168`, `trainer.py:520`| `False` | `RolloutConfig.enabled` | `False` |
| `training.rollout.backprop`| `cli_support.py:166`, `trainer.py:529`| `"full"` (unified: `"detached"`)| `RolloutConfig.backprop` | `"detached"` |
| `training.rollout.start_steps`| `trainer.py:521` | `1` | `RolloutConfig.start_steps` | `1` |
| `training.rollout.max_steps` | `cli_support.py:169`, `trainer.py:522`| `1` (unified: `4`) | `RolloutConfig.max_steps` | `4` |
| `training.rollout.step_increase_every_n_epochs`| `trainer.py:524`| `2` (unified: `1`) | `RolloutConfig.step_increase_every_n_epochs`| `1` |
| `training.rollout.step_loss_weighting`| `trainer.py:526` | `"uniform"` | `RolloutConfig.step_loss_weighting` | `"uniform"` |
| `checkpointing.save_dir` | `cli_support.py:318`, `trainer.py:507`| `"checkpoints/run"` | `CheckpointingConfig.save_dir` | `"checkpoints/unified"` |
| `checkpointing.save_every_n_steps`| `trainer.py:508` | `500` | `CheckpointingConfig.save_every_n_steps`| `500` |
| `checkpointing.plateau_window`| `trainer.py:509` | `200` | `CheckpointingConfig.plateau_window`| `200` |
| `checkpointing.plateau_rel_delta`| `trainer.py:510` | `0.01` | `CheckpointingConfig.plateau_rel_delta`| `0.01` |
| `checkpointing.keep_last_n` | `trainer.py:512` | `3` | `CheckpointingConfig.keep_last_n` | `3` |
| `logging.log_every_n_steps` | `trainer.py:491` | `100` | `LoggingConfig.log_every_n_steps` | `100` |
| `logging.val_every_n_epochs`| `trainer.py:492` | `1` | `LoggingConfig.val_every_n_epochs` | `1` |
| `logging.use_wandb` | `unified.yaml` | `False` | `LoggingConfig.use_wandb` | `False` |
| `logging.project` | `unified.yaml` | `"aurora-mjo"` | `LoggingConfig.project` | `"aurora-mjo"` |

### Test Suite Growth
- Unit tests on default CI path increased from 69 to 84 (+15 new tests in `tests/test_config_validation.py`, 0 regressions).

---

## 5. What was ruled out, and by what evidence

- **Loosening `extra="forbid"` with `extra="allow"` or `extra="ignore"`**: Ruled out. Tolerating unmodelled keys would silently allow mistyped CLI overrides (such as `--override training.optimzer.lr=1e-5`) to execute without applying the user's intended learning rate, defeating the primary safety motivation for boundary validation.
- **Validating during mode overlay merge**: Ruled out. Merging raw dictionaries first and executing validation once on the resolved dictionary ensures B1's baseline JSON serialization remains 100% byte-identical while allowing mode overlays to express sparse overrides.
- **Migrating downstream consumers (`Trainer`, `model.py`) to attribute access in this task**: Ruled out as explicitly out of scope. Providing a dict-like mapping interface on `Config` and calling `cfg.to_dict()` allows immediate boundary validation without touching un-refactored components.

---

## 6. Caveats

NONE (`STATUS: GREEN`).

---

## 7. Observations

- **`auto_scale_memory` post-validation mutation wart**: In `cli_support.py:auto_scale_memory`, the config dictionary is mutated after boundary validation to adjust `grad_accum_steps` and backend options based on visible GPUs. Specifically, it includes legacy logic that attempts to force `gradient_checkpointing = True` if `rollout.backprop == "full"` is selected. Because Task E2's validator now hard-forbids `rollout.backprop == "full"` unless detached, this path is unreachable in valid configurations, but config immutability should be addressed in a follow-on campaign.
- **`isinstance(cfg, dict | Config)` in `tests/test_config_modes.py`**: Line 50 of `tests/test_config_modes.py` previously checked `assert isinstance(cfg, dict)`. It was updated to `assert isinstance(cfg, dict | Config)` so that tests accept both bare dictionaries and validated `Config` objects. All other assertions in the module passed unmodified.

---

## 8. Questions raised

NONE.

---

## 9. Commits

```text
3930225 docs(campaign): record task E2 completion results in E2_result.md
dd1ffec feat(config): implement boundary Config validation with Pydantic v2 (E2)
c1db1c2 deps: add pydantic>=2.8 for boundary configuration validation (E2)
```

## 10. Files changed

```text
 docs/CLI.md                                  |  19 ++++++++++++
 docs/campaigns/refactor/results/E2_result.md | 231 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 pyproject.toml                               |   1 +
 run.py                                       |   4 +--
 src/aurora_mjo/cli_support.py                |  95 +++++++++++++++++++---------------------------------------
 src/aurora_mjo/config.py                     | 449 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/conftest.py                            |   2 +-
 tests/test_config_modes.py                   |   4 +--
 tests/test_config_validation.py              | 216 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/test_norm_stats.py                     |   3 +-
 uv.lock                                      | Bin 83759 -> 83834 bytes
 11 files changed, 952 insertions(+), 72 deletions(-)
```
