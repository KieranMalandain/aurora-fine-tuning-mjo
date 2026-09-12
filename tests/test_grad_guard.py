"""tests/test_grad_guard.py — Regression tests for Lesson 2: Non-finite gradient guard.

Reference:
    00_CONTEXT.md §3 (Lesson 2)
    01_TARGET_STATE.md §9
    src/aurora_mjo/trainer.py ~line 1080

Background:
    Under bf16 mixed-precision training, GradScaler is disabled, so `scaler.step()`
    degrades to a bare `optimizer.step()` with no Inf/NaN checking.
    A finite loss can still produce non-finite gradients (e.g. saturated attention
    softmax Jacobian producing finite activations but NaN gradients). `clip_grad_norm_`
    propagates a NaN total norm without raising.
    Critically, AdamW updates its moment buffers (`exp_avg`, `exp_avg_sq`) *before*
    applying the learning rate multiplier. Therefore, a single bad batch poisons
    Adam's buffers permanently, even during warmup at lr ≈ 0, causing the observed
    NaN-forever training failure mode.

    The gradient guard in `trainer.py` checks `torch.isfinite(p.grad).all()` across all
    parameters and synchronizes via `_collective_all_finite`. If any gradient is non-finite:
      1. The optimizer step is skipped.
      2. The scheduler step is skipped.
      3. `_nonfinite_grad_steps` counter is incremented.
      4. Adam's moment buffers are NOT updated or corrupted.
      5. Poisoned gradients are cleared with `zero_grad(set_to_none=True)`.

    This test module verifies the guard on CPU using a lightweight `nn.Module`.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aurora_mjo.trainer import Trainer


class TinyLinearModel(nn.Module):
    """Minimal nn.Module for testing gradient guard mechanics on CPU in CI."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _build_test_trainer(
    tmp_path: Path,
    batches: list[torch.Tensor],
    warmup_steps: int = 0,
    lr: float = 1e-2,
) -> Trainer:
    """Build a lightweight Trainer instance running on CPU with mock losses."""
    model = TinyLinearModel()
    cfg: dict[str, Any] = {
        "training": {
            "epochs": 1,
            "grad_accum_steps": 1,
            "max_grad_norm": 1.0,
            "max_steps_per_epoch": len(batches),
            "optimizer": {
                "name": "adamw",
                "lr": lr,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
            },
            "scheduler": {
                "name": "cosine",
                "warmup_steps": warmup_steps,
                "eta_min": 1e-4,
            },
        },
        "checkpointing": {
            "save_dir": str(tmp_path / "ckpt"),
            "save_every_n_steps": 1000,
        },
        "loss": {"grid": {"enabled": False}},
    }

    trainer = Trainer(
        model=model,
        cfg=cfg,
        device=torch.device("cpu"),
        train_loader=batches,
        val_loader=batches,
        is_main=True,
    )

    # Mock data prep and loss computation for TinyLinearModel
    trainer._prep_batch = lambda batch: (batch, [])

    def mock_compute_loss(in_batch: Any, target_dict_list: Any, epoch: int = 1) -> dict[str, Any]:
        out = trainer.model(in_batch)
        loss = out.sum()
        return {
            "total": loss,
            "grid": loss.detach(),
            "spectral": torch.tensor(0.0),
            "mjo_head": torch.tensor(0.0),
            "moisture_budget": torch.tensor(0.0),
        }

    trainer._compute_loss = mock_compute_loss
    return trainer


def test_grad_guard_skips_step_and_preserves_adam_buffers_on_nan(tmp_path: Path) -> None:
    """Assert guard skips step on NaN gradient, leaving Adam buffers untouched."""
    batch = torch.ones(2, 2)
    trainer = _build_test_trainer(tmp_path, [batch], warmup_steps=10, lr=1e-3)

    # Hook backward to inject NaN into fc.weight.grad
    orig_backward = trainer._guarded_backward

    def bad_backward(*args: Any, **kwargs: Any) -> bool:
        res = orig_backward(*args, **kwargs)
        with torch.no_grad():
            assert trainer.model.fc.weight.grad is not None
            trainer.model.fc.weight.grad[0, 0] = float("nan")
        return res

    trainer._guarded_backward = bad_backward

    init_weight = trainer.model.fc.weight.clone()
    init_lr = trainer.optimizer.param_groups[0]["lr"]
    assert trainer._nonfinite_grad_steps == 0

    trainer.train_epoch(1)

    # 1. Counter incremented
    assert trainer._nonfinite_grad_steps == 1, f"Skips: {trainer._nonfinite_grad_steps}"

    # 2. Weights unchanged (optimizer step skipped)
    assert torch.equal(trainer.model.fc.weight, init_weight), "Weights modified on NaN step"

    # 3. Scheduler not advanced
    assert trainer.optimizer.param_groups[0]["lr"] == init_lr, "Scheduler stepped on NaN step"

    # 4. Adam moment buffers uninitialized / uncorrupted
    param_state = trainer.optimizer.state.get(trainer.model.fc.weight, {})
    assert "exp_avg" not in param_state, "Adam buffer allocated on skipped step"


def test_grad_guard_selective_finite_step_advances(tmp_path: Path) -> None:
    """Assert guard is selective: valid finite gradients step optimizer and scheduler normally."""
    batch = torch.ones(2, 2)
    trainer = _build_test_trainer(tmp_path, [batch], warmup_steps=0, lr=1e-2)

    init_weight = trainer.model.fc.weight.clone()

    trainer.train_epoch(1)

    # 1. Counter remains 0
    assert trainer._nonfinite_grad_steps == 0

    # 2. Weights updated by optimizer step
    assert not torch.equal(trainer.model.fc.weight, init_weight), "Weights not updated"

    # 3. Adam moment buffers initialized with finite values
    param_state = trainer.optimizer.state[trainer.model.fc.weight]
    assert "exp_avg" in param_state
    assert torch.isfinite(param_state["exp_avg"]).all()
    assert torch.isfinite(param_state["exp_avg_sq"]).all()


def test_adam_buffers_unpoisoned_when_nan_follows_clean_step(tmp_path: Path) -> None:
    """Assert Adam moment buffers are not poisoned when NaN follows a good step.

    This directly tests the core failure mode: buffer poisoning where Adam's m/v
    state is corrupted by an invalid gradient, ruining all future steps.
    """
    batches = [torch.ones(2, 2), torch.ones(2, 2)]
    trainer = _build_test_trainer(tmp_path, batches, warmup_steps=0, lr=1e-2)

    batch_count = 0
    orig_backward = trainer._guarded_backward

    def mixed_backward(*args: Any, **kwargs: Any) -> bool:
        nonlocal batch_count
        batch_count += 1
        res = orig_backward(*args, **kwargs)
        # Inject NaN only on the second batch
        if batch_count == 2:
            with torch.no_grad():
                assert trainer.model.fc.weight.grad is not None
                trainer.model.fc.weight.grad[0, 0] = float("nan")
        return res

    trainer._guarded_backward = mixed_backward

    # Run batch 1 (clean) and capture Adam buffer state
    # We simulate by running step 1 with clean backward, inspecting state, then running step 2
    trainer_clean = _build_test_trainer(tmp_path / "t1", [torch.ones(2, 2)], warmup_steps=0)
    trainer_clean.train_epoch(1)
    clean_state = deepcopy(trainer_clean.optimizer.state[trainer_clean.model.fc.weight])

    # Now run the 2-step mixed trainer
    trainer.train_epoch(1)

    assert trainer._nonfinite_grad_steps == 1, "Expected exactly 1 skipped step"
    param_state_mixed = trainer.optimizer.state[trainer.model.fc.weight]

    # Adam buffers must remain strictly finite and match the state from step 1
    assert torch.isfinite(param_state_mixed["exp_avg"]).all(), "Adam exp_avg poisoned by NaN!"
    assert torch.isfinite(param_state_mixed["exp_avg_sq"]).all(), "Adam exp_avg_sq poisoned by NaN!"
    assert torch.equal(param_state_mixed["exp_avg"], clean_state["exp_avg"])
    assert torch.equal(param_state_mixed["exp_avg_sq"], clean_state["exp_avg_sq"])


def test_collective_all_finite_single_rank_and_ddp_gap(tmp_path: Path) -> None:
    """Verify _collective_all_finite single-rank fallback and document multi-rank DDP gap.

    DDP COVERAGE GAP:
    In a distributed multi-GPU training setup, `_collective_all_finite` executes:
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
    to ensure all ranks step-or-skip collectively, preventing deadlocks where one rank skips
    while others attempt to sync.
    In the single-process CPU test environment, `dist.is_initialized()` is False, so the
    method directly returns `local_finite`. This test validates the single-rank contract.
    """
    trainer = _build_test_trainer(tmp_path, [torch.ones(2, 2)])

    # When dist is uninitialized (single-process / CPU test path)
    assert trainer._collective_all_finite(True) is True
    assert trainer._collective_all_finite(False) is False
