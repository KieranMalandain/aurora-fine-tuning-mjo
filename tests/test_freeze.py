"""tests/test_freeze.py — Converted from scripts/smoke_test_freeze.py.

Verifies:
  1. Standard Aurora backbone parameters are frozen (requires_grad=False).
  2. LoRA adapter parameters are identified strictly by containing module type
     (isinstance(mod, (LoRA, LoRARollout))) rather than by parameter name, and are all trainable.
  3. Patch-embedding weights for injected variables (ttr, tcwv) are trainable.
  4. Pretrained surface embeddings (2t, 10u, 10v, msl) remain frozen.
  5. The MJO head MLP is fully trainable when enabled.
  6. Trainable parameter counts match domain priors (41,008 for small baseline per
     03_DOMAIN_PRIORS.md §7) and constitute a tiny fraction of total parameters (< 1%).
"""

from __future__ import annotations

from typing import Any

import pytest
from aurora import Aurora
from aurora.model.lora import LoRA, LoRARollout

from aurora_mjo.model import _AURORA_DEFAULT_SURF_VARS, AuroraMJO, load_model


@pytest.fixture(autouse=True)
def mock_aurora_load_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make load_checkpoint a no-op so models instantiate offline on CPU without HF downloads."""
    monkeypatch.setattr(Aurora, "load_checkpoint", lambda self, strict=True: None)


def test_lora_and_backbone_freezing_by_module_type() -> None:
    """Verify LoRA parameters are identified by module type and backbone is frozen."""
    config: dict[str, Any] = {
        "model_type": "small",
        "surface_variables": ["2t", "10u", "10v", "msl", "ttr", "tcwv"],
        "use_lora": True,
        "lora_mode": "single",
        "freeze_backbone": True,
        "gradient_checkpointing": False,
        "mjo_head": {
            "enabled": True,
            "hidden_dim": 64,
            "dropout": 0.0,
            "lat_south": -15.0,
            "lat_north": 15.0,
        },
    }

    model: AuroraMJO = load_model(config)
    backbone = model.backbone
    mjo_head = model.mjo_head

    # 1. Backbone has frozen parameters
    backbone_frozen = [p for p in backbone.parameters() if not p.requires_grad]
    assert len(backbone_frozen) > 0, "Expected frozen parameters in backbone after freeze_backbone"

    # 2. LoRA adapter params identified strictly BY MODULE TYPE, not by name
    lora_modules = []
    lora_params = []
    for mod_name, mod in backbone.named_modules():
        if isinstance(mod, LoRA | LoRARollout):
            lora_modules.append((mod_name, mod))
            for p in mod.parameters():
                lora_params.append((mod_name, p))

    assert len(lora_modules) > 0, "Expected LoRA modules in backbone when use_lora=True"
    assert len(lora_params) > 0, "Expected parameters in LoRA modules"
    assert all(p.requires_grad for _, p in lora_params), "All LoRA adapter params must be trainable"

    # 3. Injected surface variable embeddings (ttr, tcwv) are trainable
    surf_embed = backbone.encoder.surf_token_embeds
    injected_vars = [v for v in config["surface_variables"] if v not in _AURORA_DEFAULT_SURF_VARS]
    assert set(injected_vars) == {"ttr", "tcwv"}
    for var in injected_vars:
        assert var in surf_embed.weights, f"Injected var '{var}' missing"
        assert surf_embed.weights[var].requires_grad, f"Injected '{var}' not trainable"

    # 4. Standard pretrained surface variable embeddings are frozen
    for var in _AURORA_DEFAULT_SURF_VARS:
        if var in surf_embed.weights:
            assert not surf_embed.weights[var].requires_grad, f"Pretrained '{var}' not frozen"

    # 5. MJO head is fully trainable
    assert mjo_head is not None, "MJO head should be constructed when enabled=True"
    head_params = list(mjo_head.parameters())
    assert len(head_params) > 0, "MJO head has no parameters"
    assert all(p.requires_grad for p in head_params), "All MJO head params must be trainable"

    # 6. Overall trainable ratio is small (< 1%)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_fraction = trainable_params / total_params
    assert trainable_params > 0, "Model must have trainable parameters"
    assert trainable_fraction < 0.01, f"Trainable fraction {trainable_fraction:.2%} >= 1%"


def test_baseline_trainable_param_counts_match_domain_priors() -> None:
    """Verify small baseline parameter counts match 03_DOMAIN_PRIORS.md §7 (41,008 trainable)."""
    # Baseline configuration: all 6 surface variables (ttr, tcwv unfreeze decoder heads),
    # freeze_backbone=True, mjo_head disabled, no LoRA.
    config: dict[str, Any] = {
        "model_type": "small",
        "surface_variables": ["2t", "10u", "10v", "msl", "ttr", "tcwv"],
        "use_lora": False,
        "freeze_backbone": True,
        "gradient_checkpointing": False,
        "mjo_head": {"enabled": False},
    }

    model: AuroraMJO = load_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    # 03_DOMAIN_PRIORS.md §7: Trainable params = 41,008
    assert trainable_params == 41008, f"Expected 41,008 trainable params, got {trainable_params:,}"
    assert frozen_params == 112789376, f"Expected 112,789,376 frozen params, got {frozen_params:,}"
    assert total_params == 112830384, f"Expected 112,830,384 total params, got {total_params:,}"
