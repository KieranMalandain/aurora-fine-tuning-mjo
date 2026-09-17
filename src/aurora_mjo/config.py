"""Configuration loading, validation, and schema definitions for aurora_mjo.

This module is the single boundary validation layer for configuration.
Configs are loaded from YAML, overlayed with the requested operational mode,
mutated by CLI overrides, and validated once into a strongly-typed `Config`
object using Pydantic v2.

All models enforce `extra='forbid'` to catch typos in `--override` paths
(e.g., `training.optimzer.lr` will fail loudly at load time).
The validity matrix enforces critical geophysics, hardware, and safety constraints
(e.g. Perlmutter gradient checkpointing prohibition, non-overlapping train/val splits).
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Config Model supporting dict-like access for consumer compatibility
# ---------------------------------------------------------------------------


class BaseConfigModel(BaseModel):
    """Base model enforcing forbidden extra keys and dict-like mapping access."""

    model_config = ConfigDict(extra="forbid")

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item) from None

    def get(self, item: str, default: Any = None) -> Any:
        if item in self.model_fields_set:
            val = getattr(self, item)
            return val if val is not None else default
        val = getattr(self, item, None)
        if val is not None:
            return val
        return default

    def __contains__(self, item: str) -> bool:
        if item in self.model_fields_set:
            return getattr(self, item) is not None
        val = getattr(self, item, None)
        return val is not None


# ---------------------------------------------------------------------------
# Section Schemas
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseConfigModel):
    """Experiment metadata and initialization provenance."""

    name: str = "aurora_mjo_unified"
    seed: int = 42
    mode: str | None = None
    init_from: str | None = None


class RealDataConfig(BaseConfigModel):
    """Year splits for real ERA5 dataset."""

    train_years: list[int] = Field(default_factory=lambda: [1980, 2015])
    val_years: list[int] = Field(default_factory=lambda: [2016, 2019])
    test_years: list[int] = Field(default_factory=lambda: [2020, 2023])

    @model_validator(mode="after")
    def check_chronological_splits(self) -> RealDataConfig:
        """Enforce strict chronological splits without overlap (Lesson 5 leakage prevention)."""
        train_set = set(range(self.train_years[0], self.train_years[1] + 1))
        val_set = set(range(self.val_years[0], self.val_years[1] + 1))
        test_set = set(range(self.test_years[0], self.test_years[1] + 1))

        if train_set.intersection(val_set):
            raise ValueError(
                f"val_years {self.val_years} overlapping train_years {self.train_years}: "
                "chronological splits only; this is the 54,060 leakage class"
            )
        if test_set.intersection(train_set) or test_set.intersection(val_set):
            raise ValueError(
                f"test_years {self.test_years} overlapping either train_years or val_years: "
                "chronological splits only; this is the 54,060 leakage class"
            )
        return self


class DummyDataConfig(BaseConfigModel):
    """Legacy placeholder paths from deleted dummy dataset.

    Retained for YAML backwards compatibility.
    """

    surface_files: list[str] = Field(default_factory=list)
    pressure_files: list[str] = Field(default_factory=list)
    static_file: str = ""


class DataConfig(BaseConfigModel):
    """Data loading, caching, and split configuration."""

    use_dummy: bool = False
    root: str
    slt_path: str
    dummy: DummyDataConfig = Field(default_factory=DummyDataConfig)
    real: RealDataConfig = Field(default_factory=RealDataConfig)
    batch_size: int = 1
    num_workers: int = 4
    prefetch_factor: int | None = 4
    pin_memory: bool = True

    @field_validator("use_dummy")
    @classmethod
    def check_use_dummy(cls, v: bool) -> bool:
        if v:
            raise ValueError("the dummy dataset was deleted; use `--smoke-test`")
        return v


class NormStatConfig(BaseConfigModel):
    """Per-variable mean and standard deviation normalisation constants."""

    mean: float
    std: float


class MJOHeadConfig(BaseConfigModel):
    """MJO linear projection head configuration."""

    enabled: bool = False
    hidden_dim: int = 256
    dropout: float = 0.1
    lat_south: float = -15.0
    lat_north: float = 15.0


class ModelConfig(BaseConfigModel):
    """Backbone architecture, LoRA adaptation, and normalisation overrides."""

    model_type: str = "small"
    use_lora: bool = False
    lora_mode: str = "single"
    gradient_checkpointing: bool = False
    surface_variables: list[str] = Field(
        default_factory=lambda: ["2t", "10u", "10v", "msl", "ttr", "tcwv"]
    )
    norm_stats: dict[str, NormStatConfig]
    freeze_backbone: bool = True
    mjo_head: MJOHeadConfig = Field(default_factory=MJOHeadConfig)

    @field_validator("model_type")
    @classmethod
    def check_model_type(cls, v: str) -> str:
        if v == "huge":
            raise ValueError(
                "model_type 'huge' is retired; use 'full' instead. 'huge' pointed at "
                "aurora-0.25-finetuned.ckpt (the IFS HRES operational analysis fine-tune), "
                "which is a different input distribution from ERA5 reanalysis."
            )
        return v

    @field_validator("gradient_checkpointing")
    @classmethod
    def check_gradient_checkpointing(cls, v: bool) -> bool:
        if v:
            raise ValueError(
                "it deterministically triggers an illegal memory access on Perlmutter; "
                "see the gameplan; the full model does not fit on 4×A100 without it, "
                "so use `model_type: small`"
            )
        return v

    @field_validator("norm_stats")
    @classmethod
    def check_norm_stats(cls, v: dict[str, NormStatConfig]) -> dict[str, NormStatConfig]:
        if "msl" not in v:
            raise ValueError(
                "model.norm_stats.msl absent: removing it restores a −36 σ input and "
                "100% non-finite validation"
            )
        return v


class GridLossConfig(BaseConfigModel):
    """Latitude-weighted spatial grid loss."""

    enabled: bool = True
    tropics_bbox: list[int] = Field(default_factory=lambda: [-20, 20])
    tropics_weight: float = 1.0
    extratropics_weight: float = 0.1


class SpectralLossConfig(BaseConfigModel):
    """Spherical harmonic / spectral domain loss."""

    enabled: bool = False
    weight: float = 0.0


class MJOHeadLossConfig(BaseConfigModel):
    """Direct RMM index supervision loss."""

    enabled: bool = False
    weight: float = 0.0


class MoistureBudgetLossConfig(BaseConfigModel):
    """Column-integrated moisture conservation physics loss."""

    enabled: bool = False
    weight: float = 0.0
    tropics_bbox: list[int] | None = None
    dt_seconds: int | None = None


class LossConfig(BaseConfigModel):
    """Composite loss function configuration."""

    grid: GridLossConfig = Field(default_factory=GridLossConfig)
    spectral: SpectralLossConfig = Field(default_factory=SpectralLossConfig)
    mjo_head: MJOHeadLossConfig = Field(default_factory=MJOHeadLossConfig)
    moisture_budget: MoistureBudgetLossConfig = Field(default_factory=MoistureBudgetLossConfig)


class OptimizerConfig(BaseConfigModel):
    """Optimization hyperparameters."""

    name: str = "adamw"
    lr: float | str = 1.0e-4
    weight_decay: float = 1.0e-5
    betas: list[float] = Field(default_factory=lambda: [0.9, 0.999])

    @field_validator("lr")
    @classmethod
    def check_lr(cls, v: float | str) -> float | str:
        try:
            float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid learning rate value: {v!r}") from None
        return v


class SchedulerConfig(BaseConfigModel):
    """Learning rate scheduler settings."""

    name: str = "cosine"
    warmup_steps: int = 100
    eta_min: float = 1.0e-6


class RolloutConfig(BaseConfigModel):
    """Autoregressive multi-step rollout curriculum."""

    enabled: bool = False
    backprop: str = "detached"
    start_steps: int = 1
    max_steps: int = 4
    step_increase_every_n_epochs: int = 1
    step_loss_weighting: str = "uniform"

    @model_validator(mode="after")
    def check_rollout_settings(self) -> RolloutConfig:
        if self.enabled and self.backprop.lower() == "full":
            raise ValueError(
                'rollout.enabled: true with backprop: "full": only "detached" has been '
                "verified crash-free; see probe_ima_matrix.py"
            )
        if self.start_steps > self.max_steps:
            raise ValueError(
                f"rollout.start_steps ({self.start_steps}) > rollout.max_steps ({self.max_steps}): "
                "nonsensical curriculum"
            )
        return self


class TrainingConfig(BaseConfigModel):
    """Training run loop, AMP, and step pacing."""

    require_full_model: bool = False
    epochs: int = 3
    target_effective_batch: int = 8
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0
    use_amp: bool = True
    sdpa_backend: str = "default"
    max_steps_per_epoch: int | None = 2500
    max_val_batches: int | None = 200
    time_limit_hours: float = 11.0
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)


class CheckpointingConfig(BaseConfigModel):
    """Checkpoint directory and cadence settings."""

    save_dir: str = "checkpoints/unified"
    save_every_n_steps: int = 500
    plateau_window: int = 200
    plateau_rel_delta: float = 0.01
    keep_last_n: int = 3


class LoggingConfig(BaseConfigModel):
    """Console and telemetry logging settings."""

    log_every_n_steps: int = 100
    val_every_n_epochs: int = 1
    use_wandb: bool = False
    project: str = "aurora-mjo"


class Config(BaseConfigModel):
    """Root configuration object representing fully validated state."""

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig
    model: ModelConfig
    loss: LossConfig = Field(default_factory=LossConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def check_cross_section_rules(self) -> Config:
        """Validate cross-section constraints."""
        if not self.model.mjo_head.enabled and self.loss.mjo_head.enabled:
            raise ValueError(
                "mjo_head.enabled: false with loss.mjo_head.enabled: true: "
                "a loss term scoring a head that does not exist"
            )
        if self.training.require_full_model and self.model.model_type == "small":
            raise ValueError(
                "training.require_full_model is true, but model_type resolved to 'small'. "
                "Production launches must use the 1.3B model ('full') to prevent running "
                "the debug model."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Produce canonical dictionary matching B1 baseline fingerprint."""
        return self.model_dump(exclude_unset=True)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, dict):
            return self.to_dict() == other
        return super().__eq__(other)


# ---------------------------------------------------------------------------
# Loading, Merge, and Override Helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `overlay` into a copy of `base` (overlay wins)."""
    out = deepcopy(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def apply_overrides(
    cfg: Config | dict[str, Any],
    overrides: list[str],
) -> Config:
    """Apply dot-notation overrides (e.g. 'training.optimizer.lr=1e-5') and validate.

    Fails loudly if any override key is invalid or introduces an unknown field.
    """
    raw = cfg.to_dict() if isinstance(cfg, Config) else deepcopy(cfg)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override (expected key=value): {override!r}")
        key_path, raw_value = override.split("=", 1)
        keys = key_path.strip().split(".")
        value = yaml.safe_load(raw_value)
        node = raw
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
        log.info(f"Override: {key_path} = {value!r}")

    return Config.model_validate(raw)


def load_config(
    path: str | Path,
    mode: str | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load YAML config, resolve mode overlay, apply overrides, and validate once.

    Args:
        path: Path to YAML configuration file.
        mode: Operational mode name (required for unified configs).
        overrides: Optional list of dot-notation override strings ('key.path=value').

    Returns:
        Validated `Config` instance.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    log.info(f"Loaded config: {path}")

    if "modes" in raw:
        modes = raw.pop("modes")
        if mode is None:
            raise SystemExit(
                f"--mode is required with a unified config (available: {list(modes)})"
            )
        if mode not in modes:
            raise SystemExit(f"Unknown mode {mode!r}; available: {list(modes)}")

        # Rule 8: check that no two modes share a save_dir
        save_dirs: dict[str, str] = {}
        for m_name, m_overlay in modes.items():
            m_resolved = _deep_merge(raw, m_overlay)
            sd = m_resolved.get("checkpointing", {}).get("save_dir")
            if sd in save_dirs:
                raise ValueError(
                    f"two modes sharing a save_dir causes silent checkpoint overwrite: "
                    f"modes '{save_dirs[sd]}' and '{m_name}' both use save_dir '{sd}'"
                )
            if sd:
                save_dirs[sd] = m_name

        cfg_dict = _deep_merge(raw, modes[mode])
        cfg_dict.setdefault("experiment", {})["mode"] = mode
        log.info(f"Applied mode overlay: {mode}")
    else:
        cfg_dict = raw

    if overrides:
        return apply_overrides(cfg_dict, overrides)

    return Config.model_validate(cfg_dict)


def print_config(cfg: Config | dict[str, Any]) -> None:
    """Print canonical JSON (sorted keys, indent 2) to stdout."""
    dumped = cfg.to_dict() if isinstance(cfg, Config) else cfg
    print(json.dumps(dumped, indent=2, sort_keys=True))
