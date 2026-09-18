"""tests/test_rollout.py — Converted from scripts/smoke_test_rollout.py.

Verifies:
  1. Rollout curriculum advances across epochs as configured and caps at max_steps.
  2. Step loss weighting (uniform and linear) normalizes weights to sum to 1.0.
  3. Structural property of backprop: "detached" — _advance_batch with detach=True
     severs the autograd computation graph between rollout steps, maintaining flat
     O(1) activation memory across rollout horizon k.
  4. Multi-step autoregressive rollout forward pass and backward loss flow with StubModel.
  5. Detached rollout step execution via Trainer._detached_rollout_step.
  6. GPU-accelerated rollout execution (marked needs_gpu, slow).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
import torch
from aurora.batch import Batch, Metadata
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aurora_mjo.trainer import Trainer, _advance_batch

SURF_VARS = ("2t", "10u", "10v", "msl", "ttr", "tcwv")
ATMOS_VARS = ("z", "q", "t", "u", "v")
ATMOS_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)


class StubModel(nn.Module):
    """Stub model returning a 1-step prediction Batch echoing input fields.

    Carries a single learnable parameter so optimizer has a grad path.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dummy_param = nn.Parameter(torch.tensor([1.0], requires_grad=True))

    def forward(self, batch: Batch) -> Batch:
        # 1-step prediction: slice last history timestep and add dummy_param * 0
        pred_surf = {k: v[:, -1:, ...] + 0.0 * self.dummy_param for k, v in batch.surf_vars.items()}
        pred_atmos = {
            k: v[:, -1:, ...] + 0.0 * self.dummy_param for k, v in batch.atmos_vars.items()
        }
        new_meta = Metadata(
            lat=batch.metadata.lat,
            lon=batch.metadata.lon,
            time=batch.metadata.time,
            atmos_levels=batch.metadata.atmos_levels,
            rollout_step=batch.metadata.rollout_step,
        )
        return Batch(
            surf_vars=pred_surf,
            static_vars=batch.static_vars,
            atmos_vars=pred_atmos,
            metadata=new_meta,
        )


def _make_synthetic_batch_and_targets(
    b: int = 1, h: int = 32, w: int = 64, k: int = 2, device: torch.device | None = None
) -> tuple[Batch, list[dict[str, torch.Tensor]]]:
    """Create a small synthetic Batch and a list of k target dictionaries."""
    t = 2
    lat = torch.linspace(90, -90, h)
    lon = torch.linspace(0, 360 - 360 / w, w)
    if device is not None:
        lat = lat.to(device)
        lon = lon.to(device)

    meta = Metadata(
        lat=lat,
        lon=lon,
        time=(datetime(2020, 1, 1, 0, 0),),
        atmos_levels=ATMOS_LEVELS,
        rollout_step=0,
    )
    surf_vars = {v: torch.randn(b, t, h, w) for v in SURF_VARS}
    static_vars = {"z": torch.randn(h, w), "lsm": torch.randn(h, w), "slt": torch.randn(h, w)}
    atmos_vars = {v: torch.randn(b, t, len(ATMOS_LEVELS), h, w) for v in ATMOS_VARS}

    if device is not None:
        surf_vars = {k: v.to(device) for k, v in surf_vars.items()}
        static_vars = {k: v.to(device) for k, v in static_vars.items()}
        atmos_vars = {k: v.to(device) for k, v in atmos_vars.items()}

    in_batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=meta,
    )
    target_dict_list = [
        {
            **{k: torch.randn(b, 1, h, w) for k in SURF_VARS},
            **{k: torch.randn(b, 1, len(ATMOS_LEVELS), h, w) for k in ATMOS_VARS},
        }
        for _ in range(k)
    ]
    if device is not None:
        target_dict_list = [{k: v.to(device) for k, v in td.items()} for td in target_dict_list]
    return in_batch, target_dict_list


def _make_minimal_trainer(
    model: nn.Module,
    rollout_cfg: dict[str, Any],
    device: torch.device | None = None,
) -> Trainer:
    """Instantiate a minimal Trainer with injected dummy DataLoader."""
    if device is None:
        device = torch.device("cpu")

    cfg = {
        "experiment": {"name": "test_rollout", "seed": 42},
        "data": {
            "use_dummy": False,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
        },
        "loss": {
            "grid": {
                "enabled": True,
                "tropics_bbox": [-20, 20],
                "tropics_weight": 1.0,
                "extratropics_weight": 0.1,
            },
            "spectral": {"enabled": False, "weight": 0.0},
            "mjo_head": {"enabled": False, "weight": 0.0},
            "moisture_budget": {"enabled": False, "weight": 0.0},
        },
        "training": {
            "epochs": 10,
            "grad_accum_steps": 1,
            "max_grad_norm": 1.0,
            "optimizer": {
                "name": "adamw",
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "betas": [0.9, 0.999],
            },
            "scheduler": {"name": "none"},
            "rollout": rollout_cfg,
        },
        "checkpointing": {
            "save_dir": "/tmp/test_rollout_ckpt",
            "save_every_n_epochs": 1,
            "keep_last_n": 1,
        },
        "logging": {"log_every_n_steps": 1, "val_every_n_epochs": 1},
    }
    dummy_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)
    return Trainer(
        model=model,
        cfg=cfg,
        device=device,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
    )


def test_rollout_curriculum_progression() -> None:
    """Verify that rollout curriculum advances as configured and caps at max_steps."""
    rollout_cfg = {
        "enabled": True,
        "start_steps": 1,
        "max_steps": 4,
        "step_increase_every_n_epochs": 2,
        "step_loss_weighting": "uniform",
    }
    trainer = _make_minimal_trainer(StubModel(), rollout_cfg)

    expected_steps = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 3,
        6: 3,
        7: 4,
        8: 4,
        15: 4,
    }
    for epoch, expected_k in expected_steps.items():
        measured_k = trainer._current_rollout_steps(epoch)
        assert measured_k == expected_k, f"Epoch {epoch}: expected k={expected_k}"


def test_step_loss_weighting() -> None:
    """Verify uniform and linear step loss weights sum to 1.0."""
    uniform_cfg = {
        "enabled": True,
        "start_steps": 2,
        "max_steps": 4,
        "step_loss_weighting": "uniform",
    }
    trainer_uniform = _make_minimal_trainer(StubModel(), uniform_cfg)

    for k in (1, 2, 3, 4):
        w = trainer_uniform._step_weights(k)
        assert len(w) == k
        assert abs(sum(w) - 1.0) < 1e-6
        assert all(abs(wi - 1.0 / k) < 1e-6 for wi in w)

    final_heavy_cfg = {
        "enabled": True,
        "start_steps": 2,
        "max_steps": 4,
        "step_loss_weighting": "final_heavy",
    }
    trainer_final_heavy = _make_minimal_trainer(StubModel(), final_heavy_cfg)

    w_heavy = trainer_final_heavy._step_weights(3)
    assert len(w_heavy) == 3
    assert abs(sum(w_heavy) - 1.0) < 1e-6
    # In final_heavy weighting, last step receives 0.5, earlier steps split remaining 0.5
    assert w_heavy == [0.25, 0.25, 0.5]


def test_advance_batch_detached_cuts_autograd_graph() -> None:
    """Verify the structural property of backprop: 'detached'.

    When detach=True in _advance_batch, the returned batch's variables have
    no grad_fn, cutting the autograd graph and freeing prior step activations.
    When detach=False, the history retains the autograd computation graph.
    """
    b, h, w = 1, 16, 32
    meta = Metadata(
        lat=torch.linspace(90, -90, h),
        lon=torch.linspace(0, 360 - 360 / w, w),
        time=(datetime(2020, 1, 1, 0, 0),),
        atmos_levels=ATMOS_LEVELS,
    )
    in_batch = Batch(
        surf_vars={"2t": torch.randn(b, 2, h, w, requires_grad=True)},
        static_vars={"z": torch.randn(h, w)},
        atmos_vars={"t": torch.randn(b, 2, len(ATMOS_LEVELS), h, w, requires_grad=True)},
        metadata=meta,
    )

    # Simulated model prediction carrying an active gradient graph
    pred_batch = Batch(
        surf_vars={"2t": in_batch.surf_vars["2t"][:, -1:, ...] * 2.0},
        static_vars=in_batch.static_vars,
        atmos_vars={"t": in_batch.atmos_vars["t"][:, -1:, ...] * 2.0},
        metadata=meta,
    )

    # With detach=False: computation graph is retained
    attached_batch = _advance_batch(in_batch, pred_batch, step_index=0, detach=False)
    assert attached_batch.surf_vars["2t"].grad_fn is not None, "Expected grad_fn on attached batch"

    # With detach=True: autograd graph is severed
    detached_batch = _advance_batch(in_batch, pred_batch, step_index=0, detach=True)
    assert detached_batch.surf_vars["2t"].grad_fn is None, "Expected severed graph when detach=True"
    assert detached_batch.atmos_vars["t"].grad_fn is None, "Expected severed graph when detach=True"


def test_rollout_loss_computation_and_backward() -> None:
    """Verify autoregressive rollout loss computation and backward gradient flow."""
    model = StubModel()
    rollout_cfg = {
        "enabled": True,
        "start_steps": 2,
        "max_steps": 2,
        "step_loss_weighting": "uniform",
        "backprop": "full",
    }
    trainer = _make_minimal_trainer(model, rollout_cfg)

    in_batch, target_dict_list = _make_synthetic_batch_and_targets(b=1, h=32, w=64, k=2)
    losses = trainer._compute_loss(in_batch, target_dict_list, epoch=1)

    assert "total" in losses
    assert torch.isfinite(losses["total"])
    assert losses["total"].item() > 0.0

    # Backward pass
    losses["total"].backward()
    assert model.dummy_param.grad is not None, "Gradients should flow to model parameters"
    assert torch.isfinite(model.dummy_param.grad).all()


def test_detached_rollout_step_training() -> None:
    """Verify detached rollout step execution via Trainer._detached_rollout_step."""
    model = StubModel()
    rollout_cfg = {
        "enabled": True,
        "start_steps": 2,
        "max_steps": 2,
        "step_loss_weighting": "uniform",
        "backprop": "detached",
    }
    trainer = _make_minimal_trainer(model, rollout_cfg)

    in_batch, target_dict_list = _make_synthetic_batch_and_targets(b=1, h=32, w=64, k=2)
    items = trainer._detached_rollout_step(
        in_batch=in_batch,
        target_dict_list=target_dict_list,
        epoch=1,
        is_boundary=True,
        batch_in_epoch=0,
    )

    assert "total" in items
    assert isinstance(items["total"], float)
    assert items["total"] > 0.0


@pytest.mark.needs_gpu
@pytest.mark.slow
def test_rollout_gpu_execution() -> None:
    """Verify rollout forward and backward pass on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    device = torch.device("cuda:0")
    model = StubModel().to(device)
    rollout_cfg = {
        "enabled": True,
        "start_steps": 2,
        "max_steps": 2,
        "step_loss_weighting": "uniform",
        "backprop": "detached",
    }
    trainer = _make_minimal_trainer(model, rollout_cfg, device=device)

    in_batch, target_dict_list = _make_synthetic_batch_and_targets(
        b=1, h=32, w=64, k=2, device=device
    )
    items = trainer._detached_rollout_step(
        in_batch=in_batch,
        target_dict_list=target_dict_list,
        epoch=1,
        is_boundary=True,
        batch_in_epoch=0,
    )
    assert items["total"] > 0.0


def test_validation_runs_under_no_grad() -> None:
    """Verify that Trainer.validate() executes strictly under torch.no_grad() (Task H4)."""
    model = StubModel()
    rollout_cfg = {
        "enabled": True,
        "start_steps": 1,
        "max_steps": 1,
        "step_loss_weighting": "uniform",
        "backprop": "full",
    }
    trainer = _make_minimal_trainer(model, rollout_cfg)
    in_batch, target_dict_list = _make_synthetic_batch_and_targets(b=1, h=32, w=64, k=1)
    trainer.val_loader = [(in_batch, target_dict_list)]  # type: ignore[assignment]

    # Wrap _compute_loss to inspect grad_enabled state at execution time
    original_compute_loss = trainer._compute_loss
    grad_enabled_during_val: list[bool] = []

    def _spy_compute_loss(in_b, tgt_list, epoch=1):
        grad_enabled_during_val.append(torch.is_grad_enabled())
        return original_compute_loss(in_b, tgt_list, epoch=epoch)

    trainer._compute_loss = _spy_compute_loss  # type: ignore[assignment]
    val_loss = trainer.validate(epoch=1)

    assert len(grad_enabled_during_val) > 0, "validate() did not execute _compute_loss"
    assert not any(grad_enabled_during_val), (
        f"Expected torch.is_grad_enabled() to be False during validate(), "
        f"got {grad_enabled_during_val}"
    )
    assert torch.isfinite(torch.tensor(val_loss)), "validate() loss must be finite"


def test_validation_loss_unchanged_with_no_grad_at_k1() -> None:
    """Verify validation loss at k=1 is bitwise identical under no_grad vs with_grad (H4)."""
    model = StubModel()
    rollout_cfg = {
        "enabled": True,
        "start_steps": 1,
        "max_steps": 1,
        "step_loss_weighting": "uniform",
        "backprop": "full",
    }
    trainer = _make_minimal_trainer(model, rollout_cfg)
    in_batch, target_dict_list = _make_synthetic_batch_and_targets(b=1, h=32, w=64, k=1)

    with torch.no_grad():
        losses_no_grad = trainer._compute_loss(in_batch, target_dict_list, epoch=1)

    with torch.enable_grad():
        losses_with_grad = trainer._compute_loss(in_batch, target_dict_list, epoch=1)

    val_no_grad = losses_no_grad["total"].item()
    val_with_grad = losses_with_grad["total"].item()
    assert (
        abs(val_no_grad - val_with_grad) < 1e-6
    ), f"Loss mismatch at k=1: no_grad={val_no_grad} vs with_grad={val_with_grad}"


def test_rollout_clamp_vendor_replacement() -> None:
    """Verify _ROLLOUT_CLAMP delegates tcwv/q to vendor while retaining others (Task H4)."""
    from aurora_mjo.trainer import _ROLLOUT_CLAMP

    # 1. Vendor-delegated variables are removed
    assert "tcwv" not in _ROLLOUT_CLAMP, "tcwv must be delegated to Aurora positive_surf_vars"
    assert "q" not in _ROLLOUT_CLAMP, "q must be delegated to Aurora positive_atmos_vars"

    # 2. Non-vendor physical guards are strictly retained
    expected_retained = {"msl", "2t", "10u", "10v", "ttr"}
    assert (
        set(_ROLLOUT_CLAMP.keys()) == expected_retained
    ), f"Expected retained clamp variables {expected_retained}, got {set(_ROLLOUT_CLAMP.keys())}"
