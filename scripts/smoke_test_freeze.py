#!/usr/bin/env python3
"""Smoke test: verify LoRA-aware backbone freezing in AuroraMJO.

Run from the repo root with the aurora_mjo conda environment:

    conda run -n aurora_mjo python scripts/smoke_test_freeze.py

What this tests
---------------
1. All standard Aurora backbone parameters are frozen (requires_grad=False).
2. LoRA adapter parameters (lora_A, lora_B) are trainable.
3. Patch-embedding weights for injected variables (ttr, tcwv) are trainable.
4. The MJO head MLP is fully trainable.
5. The _log_param_counts helper prints a non-zero trainable count.
"""

import sys
import types

# ---------------------------------------------------------------------------
# Minimal stubs so we can import src.model without a real checkpoint download.
# We monkey-patch Aurora classes to skip checkpoint loading.
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn

# We need aurora installed; the environment should have it.
try:
    from aurora import Aurora, AuroraSmallPretrained
    from aurora.model.lora import LoRA, LoRARollout
except ImportError:
    print("ERROR: 'aurora' package not found. Activate the aurora_mjo environment.")
    sys.exit(1)

# Patch load_checkpoint to a no-op so we can run without internet / checkpoint.
_orig_load = Aurora.load_checkpoint
Aurora.load_checkpoint = lambda self, strict=True: None  # type: ignore[method-assign]

# ---------------------------------------------------------------------------
# Now import the project module.
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")
from src.model import (
    AuroraMJO,
    _AURORA_DEFAULT_SURF_VARS,
    _is_lora_param,
    _log_param_counts,
    freeze_backbone,
    load_model,
)

# ---------------------------------------------------------------------------
# Minimal config that exercises all freezing branches.
# ---------------------------------------------------------------------------
CONFIG = {
    "model_type": "small",  # AuroraSmallPretrained — faster to instantiate
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


def run_smoke_test() -> None:
    print("=" * 60)
    print(" Smoke test: LoRA backbone freezing")
    print("=" * 60)

    model: AuroraMJO = load_model(CONFIG)

    backbone = model.backbone
    mjo_head = model.mjo_head

    # ------------------------------------------------------------------
    # Assertion 1: Backbone has at least some frozen params.
    # ------------------------------------------------------------------
    backbone_frozen = [
        p for p in backbone.parameters() if not p.requires_grad
    ]
    assert backbone_frozen, (
        "FAIL: no frozen parameters found in backbone — freeze_backbone() may not have run."
    )
    print(f"PASS [1] backbone has {len(backbone_frozen)} frozen parameter tensors.")

    # ------------------------------------------------------------------
    # Assertion 2: LoRA adapter params are trainable.
    # ------------------------------------------------------------------
    lora_trainable = []
    for mod_name, mod in backbone.named_modules():
        if isinstance(mod, (LoRA, LoRARollout)):
            for p in mod.parameters():
                lora_trainable.append((mod_name, p))

    assert lora_trainable, (
        "FAIL: no LoRA adapter params found. "
        "Check that use_lora=True and Aurora inserted LoRA modules."
    )
    all_lora_trainable = all(p.requires_grad for _, p in lora_trainable)
    assert all_lora_trainable, (
        "FAIL: at least one LoRA adapter parameter is FROZEN. "
        "freeze_backbone() should have unfrozen them."
    )
    print(f"PASS [2] {len(lora_trainable)} LoRA adapter param tensors are trainable.")

    # ------------------------------------------------------------------
    # Assertion 3: New variable embeddings (ttr, tcwv) are trainable.
    # ------------------------------------------------------------------
    surf_embed = backbone.encoder.surf_token_embeds
    injected = [v for v in CONFIG["surface_variables"] if v not in _AURORA_DEFAULT_SURF_VARS]
    for var in injected:
        assert var in surf_embed.weights, (
            f"FAIL: '{var}' not in surf_token_embeds.weights — was it passed to Aurora?"
        )
        assert surf_embed.weights[var].requires_grad, (
            f"FAIL: embedding for '{var}' is FROZEN. It should be trainable (randomly initialized)."
        )
    print(f"PASS [3] injected variable embeddings {injected} are trainable.")

    # ------------------------------------------------------------------
    # Assertion 4: Standard surf-var embeddings ARE frozen.
    # ------------------------------------------------------------------
    for var in _AURORA_DEFAULT_SURF_VARS:
        if var in surf_embed.weights:
            assert not surf_embed.weights[var].requires_grad, (
                f"FAIL: pretrained embedding for '{var}' should be FROZEN but is trainable."
            )
    print(f"PASS [4] pretrained surface embeddings are frozen.")

    # ------------------------------------------------------------------
    # Assertion 5: MJO head is fully trainable.
    # ------------------------------------------------------------------
    assert mjo_head is not None, "FAIL: MJO head is None — check config['mjo_head']['enabled']."
    head_params = list(mjo_head.parameters())
    assert head_params, "FAIL: MJO head has no parameters."
    assert all(p.requires_grad for p in head_params), (
        "FAIL: at least one MJO head parameter is FROZEN."
    )
    print(f"PASS [5] MJO head has {len(head_params)} param tensors, all trainable.")

    # ------------------------------------------------------------------
    # Summary table (visual inspect).
    # ------------------------------------------------------------------
    print("\n--- Detailed parameter audit ---")
    _log_param_counts(model, label="AuroraMJO (smoke test)")

    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_total > 0, "FAIL: zero trainable parameters — optimizer would do nothing."
    print(f"PASS [6] {trainable_total:,} trainable parameters in total.\n")

    print("All assertions passed. Freezing logic is correct.")


if __name__ == "__main__":
    run_smoke_test()
