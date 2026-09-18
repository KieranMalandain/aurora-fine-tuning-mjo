# src/model.py
"""Aurora wrapper with an optional dual-head extension for MJO prediction.

Architecture
============
The model has two heads:

1. **State head** – unchanged Aurora decoder output (a ``Batch`` of predicted
   gridded fields).
2. **MJO head** – a lightweight MLP attached to the encoder latent space.
   Features are mean-pooled over the tropical lat/lon patch region before
   being passed to the MLP, which predicts ``[RMM1, RMM2, Amplitude]``.

The MJO head is controlled by the config key ``mjo_head.enabled`` (bool).
When disabled the model is a transparent Aurora wrapper; the ``forward``
signature and return type do not change (returns ``Batch`` only).
When enabled ``forward`` returns ``(Batch, Tensor[B, 3])``.

Checkpoint loading
==================
``load_checkpoint(strict=False)`` is preserved: the new MLP weights are
randomly initialized and silently absent from the pretrained checkpoint.
"""

import torch
from aurora import AuroraPretrained, AuroraSmallPretrained
from aurora.batch import Batch
from aurora.model.lora import LoRA, LoRARollout
from aurora.normalisation import locations, scales
from torch import nn

# Aurora's built-in default surface variables (from Aurora.__init__ signature).
# Any variable in our config's surface_variables list that is NOT here was
# newly added and its embedding weights were randomly initialized  - those
# must remain trainable.
_AURORA_DEFAULT_SURF_VARS: frozenset[str] = frozenset({"2t", "10u", "10v", "msl"})
# Aurora's built-in default static variables (from Aurora.__init__ signature).
# Any variable in static_variables that is NOT here was newly added (e.g. 'sst')
# and its embedding weights must remain trainable.
_AURORA_DEFAULT_STATIC_VARS: frozenset[str] = frozenset({"lsm", "z", "slt"})


# ---------------------------------------------------------------------------
# MJO head
# ---------------------------------------------------------------------------


class MJOHead(nn.Module):
    """Lightweight MLP that predicts (RMM1, RMM2, Amplitude) from pooled
    encoder features extracted over the tropical band.

    Args:
        embed_dim (int): Dimensionality of Aurora's encoder output token (``D``).
        hidden_dim (int): Width of the hidden layer.  Defaults to ``256``.
        dropout (float): Dropout probability.  Defaults to ``0.1``.
        lat_south (float): Southern boundary of the tropical pool (degrees).
        lat_north (float): Northern boundary of the tropical pool (degrees).
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        lat_south: float = -15.0,
        lat_north: float = 15.0,
    ) -> None:
        super().__init__()
        self.lat_south = lat_south
        self.lat_north = lat_north

        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # [RMM1, RMM2, Amplitude]
        )

        # Zero-init the final projection so the head starts neutral.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        patch_res: tuple[int, int, int],
        lat: torch.Tensor,
    ) -> torch.Tensor:
        """Pool tropical encoder tokens and predict MJO indices.

        Args:
            x (torch.Tensor): Encoder output, shape ``(B, latent_levels *
                H_patches * W_patches, D)``.
            patch_res (tuple[int, int, int]): ``(latent_levels, H_patches,
                W_patches)`` as computed in Aurora's ``forward``.
            lat (torch.Tensor): Latitude coordinates of the *full-resolution*
                grid, shape ``(H,)``.  Aurora crops+patches this internally,
                so we reduce it to the patch-centre latitudes here.

        Returns:
            torch.Tensor: Shape ``(B, 3)`` – predicted RMM1, RMM2, Amplitude.
        """
        n_levels, n_h, n_w = patch_res
        B, _L, D = x.shape

        # Reshape to (B, latent_levels, H_patches, W_patches, D)
        x_spatial = x.view(B, n_levels, n_h, n_w, D)

        # Build patch-centre lats: Aurora patchifies at patch_size spacing.
        # ``lat`` is the *cropped* full-res lat grid; step = H // n_h = patch_size.
        patch_size = lat.shape[0] // n_h
        # Patch centres are at indices patch_size//2, 3*patch_size//2, ...
        patch_lat = lat[patch_size // 2 :: patch_size]  # shape (n_h,)
        patch_lat = patch_lat[:n_h]  # guard against rounding

        # Tropical mask over the patch-lat dimension.
        trop_mask = (patch_lat >= self.lat_south) & (
            patch_lat <= self.lat_north
        )  # (n_h,)

        # Pool: mean over levels, tropical latitudes, and all longitudes.
        # Shape after tropical slice: (B, n_levels, n_trop, n_w, D)
        x_trop = x_spatial[:, :, trop_mask, :, :]  # (B, n_levels, n_trop, n_w, D)
        x_pooled = x_trop.mean(dim=(1, 2, 3))  # (B, D)

        return self.mlp(x_pooled)  # (B, 3)


# ---------------------------------------------------------------------------
# Wrapper model
# ---------------------------------------------------------------------------


class AuroraMJO(nn.Module):
    """Aurora backbone wrapped with an optional MJO prediction head.

    The backbone is either the full 1.3 B Aurora or the small pretrained
    variant, selected by ``config['model_type']``.

    Config keys recognised
    ----------------------
    model_type : str
        ``'full'`` -> ``AuroraPretrained``, anything else -> ``AuroraSmallPretrained``.
    surface_variables : list[str]
        Which surface variables to pass.
    use_lora : bool
    lora_mode : str, optional
    gradient_checkpointing : bool, optional
    mjo_head.enabled : bool, optional
        Whether to attach the MJO head.  Defaults to ``False``.
    mjo_head.hidden_dim : int, optional  (default 256)
    mjo_head.dropout : float, optional  (default 0.1)
    mjo_head.lat_south : float, optional  (default -15.0)
    mjo_head.lat_north : float, optional  (default 15.0)
    """

    def __init__(
        self,
        backbone: AuroraPretrained | AuroraSmallPretrained,
        mjo_head: MJOHead | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.mjo_head = mjo_head

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: Batch) -> Batch | tuple[Batch, torch.Tensor]:
        """Run the dual-head forward pass.

        When the MJO head is disabled this is a transparent wrapper around
        the Aurora backbone and returns a ``Batch``.

        When the MJO head is enabled the encoder latent is intercepted via a
        forward hook, the Aurora backbone still runs its full decoder, and the
        return value is ``(pred_batch, mjo_pred)`` where ``mjo_pred`` has
        shape ``(B, 3)`` -> ``[RMM1, RMM2, Amplitude]``.

        Args:
            batch (:class:`aurora.batch.Batch`): Input batch.

        Returns:
            ``Batch`` when MJO head disabled, ``(Batch, Tensor)`` otherwise.
        """
        if self.mjo_head is None:
            return self.backbone(batch)

        # ------------------------------------------------------------------
        # Intercept encoder output via a temporary forward hook.
        # We store the latent tensor and patch_res so the MJO head can use
        # them without modifying Aurora's internals.
        # ------------------------------------------------------------------
        _encoder_output: dict[str, object] = {}

        def _hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            _encoder_output["x"] = output

        hook_handle = self.backbone.encoder.register_forward_hook(_hook)

        try:
            # Run the full backbone (encoder -> swin -> decoder).
            pred_batch = self.backbone(batch)
        finally:
            hook_handle.remove()

        # Reconstruct patch_res the same way Aurora does it internally.
        # After normalise+crop the spatial shape matches the cropped batch.
        # We re-derive it from the stored latent token count.
        x_enc = _encoder_output["x"]  # (B, L', D) – L' = n_levels * H_p * W_p

        # Aurora's crop may reduce H/W slightly; back out the patch grid.
        # n_levels = backbone.encoder.latent_levels ( = 1 surf + C atmos latents)
        n_levels = self.backbone.encoder.latent_levels
        n_tokens = x_enc.shape[1]
        n_spatial = n_tokens // n_levels  # H_p * W_p
        # Infer n_h, n_w from the input batch lat/lon lengths.
        p = self.backbone.patch_size
        # Use the *original* batch lat/lon (before crop) to get H, W;
        # Aurora crops by dropping incomplete patches, so cropped H = (H // p) * p.
        H_full = batch.metadata.lat.shape[0]
        W_full = batch.metadata.lon.shape[-1]
        H_patch = H_full // p
        W_patch = W_full // p
        # Sanity guard: reconcile with actual token count.
        if H_patch * W_patch != n_spatial:
            # Fallback: try to factor n_spatial assuming W_patch = W_full // p.
            H_patch = n_spatial // W_patch

        patch_res = (n_levels, H_patch, W_patch)

        # Lat grid (full-res, from the *original* batch metadata, before crop).
        lat = batch.metadata.lat.float()

        mjo_pred = self.mjo_head(x_enc, patch_res, lat)

        return pred_batch, mjo_pred

    # ------------------------------------------------------------------
    # Passthrough helpers
    # ------------------------------------------------------------------

    def configure_activation_checkpointing(self) -> None:
        """Delegate to the backbone."""
        self.backbone.configure_activation_checkpointing()


# ---------------------------------------------------------------------------
# Freezing helpers
# ---------------------------------------------------------------------------


def _is_lora_param(name: str, module: nn.Module) -> bool:
    """Return True if *module* is a LoRA adapter layer.

    Aurora's ``LoRA`` / ``LoRARollout`` contain ``lora_A`` and ``lora_B``
    parameters.  We match by direct module-type rather than name-suffix to
    avoid false positives if user code happens to share the same suffix.
    """
    return isinstance(module, (LoRA, LoRARollout))


def freeze_backbone(
    backbone: AuroraPretrained | AuroraSmallPretrained,
    new_surf_vars: tuple[str, ...],
    use_lora: bool,
    static_vars: tuple[str, ...] = ("lsm", "z", "slt", "sst"),
) -> None:
    """Freeze the Aurora backbone except for LoRA adapters and new-variable embeddings.

    Calling strategy
    ----------------
    1. Freeze **everything** in the backbone with ``requires_grad = False``.
    2. Unfreeze LoRA adapter parameters (``lora_A``, ``lora_B``) when
       ``use_lora=True``.  These are identified by their containing module
       type (:class:`aurora.model.lora.LoRA` / ``LoRARollout``), not by name,
       to avoid any accidental matches.
    3. Unfreeze the patch-embedding weights of *newly injected* surface and static
       variables (i.e. any variable in ``new_surf_vars`` or ``static_vars`` that is
       not among Aurora's built-in defaults).  These weights were randomly initialized
       on construction because the pretrained checkpoint has no entry for them.
       Note: Static variables share ``surf_token_embeds`` with surface variables.
    4. Unfreeze the decoder heads of *newly injected* surface variables. Static
       variables do NOT have decoder heads.

    Args:
        backbone: The Aurora model instance.
        new_surf_vars: The full tuple of surface variable names passed to
            Aurora, as read from ``config['surface_variables']``.
        use_lora: Whether LoRA adapters are inserted (i.e.
            ``config['use_lora']``).
        static_vars: The full tuple of static variable names passed to
            Aurora, as read from ``config['static_variables']``.
    """
    # --- Step 1: blanket freeze ---
    for param in backbone.parameters():
        param.requires_grad_(False)

    # --- Step 2: unfreeze LoRA adapter parameters ---
    if use_lora:
        for mod_name, mod in backbone.named_modules():
            if _is_lora_param(mod_name, mod):
                for param in mod.parameters():
                    param.requires_grad_(True)

    # --- Step 3: unfreeze new-variable patch embeddings ---
    # Static and surface variables share surf_token_embeds (LevelPatchEmbed).
    # Newly added surface variables AND static variables (such as 'sst') have
    # randomly initialized embedding weights that must be trained.
    injected_surf_vars = [v for v in new_surf_vars if v not in _AURORA_DEFAULT_SURF_VARS]
    injected_static_vars = [v for v in static_vars if v not in _AURORA_DEFAULT_STATIC_VARS]
    injected_vars = injected_surf_vars + injected_static_vars
    if injected_vars:
        surf_embed = backbone.encoder.surf_token_embeds
        for var in injected_vars:
            if var in surf_embed.weights:
                surf_embed.weights[var].requires_grad_(True)
            else:
                print(
                    f"[freeze_backbone] WARNING: '{var}' not found in "
                    "surf_token_embeds.weights; skipping unfreeze."
                )

    # --- Step 4 (BUG FIX): unfreeze DECODER heads of injected surface variables ---
    # Note: Static variables do NOT have decoder heads (only surf_vars do).
    if injected_surf_vars:
        surf_heads = getattr(backbone.decoder, "surf_heads", None)
        if surf_heads is not None:
            for var in injected_surf_vars:
                if var in surf_heads:
                    for p in surf_heads[var].parameters():
                        p.requires_grad_(True)
                    print(f"[freeze_backbone] Unfroze decoder head for '{var}'")
                else:
                    print(
                        f"[freeze_backbone] WARNING: '{var}' not found in "
                        "decoder.surf_heads; skipping unfreeze."
                    )
        else:
            print(
                "[freeze_backbone] WARNING: backbone.decoder.surf_heads "
                "not found; injected-variable decoder heads NOT unfrozen."
            )

    # --- Step 5 (FIX 6, AURORA_MJO_GAMEPLAN §Finding 6): unfreeze the `msl`
    # decoder head. ---
    # `msl` is a *default* Aurora surface variable (in
    # `_AURORA_DEFAULT_SURF_VARS`), so it is never touched by the
    # injected-vars logic in Steps 3-4 above. But our dataset feeds `msl`
    # real surface-pressure (`ps`) values as a proxy for mean-sea-level
    # pressure (LANL has no true MSL field) - see FIX 2 / dataset.py. Once
    # the *input* normalization stats are corrected from MSL to ps
    # statistics, the *output* head is still frozen at its pretrained,
    # MSL-calibrated weights: it would keep emitting MSL-scaled values
    # (denormalized with ps stats -> ~7x too spread out), inflating grid
    # loss over high terrain even though the input trigger is fixed.
    # Unfreezing lets the head re-learn ps-scaled output so the channel is
    # coherent end-to-end. Parameter count is small; risk is low.
    msl_surf_heads = getattr(backbone.decoder, "surf_heads", None)
    if msl_surf_heads is not None and "msl" in msl_surf_heads:
        for p in msl_surf_heads["msl"].parameters():
            p.requires_grad_(True)
        print(
            "[freeze_backbone] Unfroze decoder head for 'msl' (FIX 6: "
            "renormalized as surface pressure proxy)"
        )
    else:
        print(
            "[freeze_backbone] WARNING: 'msl' not found in "
            "decoder.surf_heads; msl decoder head NOT unfrozen (FIX 6 "
            "not applied)."
        )


def _log_param_counts(model: nn.Module, label: str = "model") -> None:
    """Print a summary of trainable vs total parameter counts.

    Breakdown is shown per top-level named child so it is easy to spot
    which sub-module is contributing trainable parameters.

    Args:
        model: The module to inspect.
        label: Human-readable name printed in the header line.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(
        f"\n{'=' * 60}\n"
        f" Parameter audit: {label}\n"
        f"{'=' * 60}\n"
        f"  Total      : {total:>12,}\n"
        f"  Trainable  : {trainable:>12,}  ({100 * trainable / max(total, 1):.2f}%)\n"
        f"  Frozen     : {frozen:>12,}  ({100 * frozen / max(total, 1):.2f}%)\n"
        f"{'=' * 60}"
    )

    # Per-child breakdown (only if they have params)
    for child_name, child in model.named_children():
        c_total = sum(p.numel() for p in child.parameters())
        if c_total == 0:
            continue
        c_train = sum(p.numel() for p in child.parameters() if p.requires_grad)
        print(f"  {child_name:<30}  trainable={c_train:>10,} / {c_total:>10,}")
    print()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_model(config: dict, norm_stats: dict | None = None) -> AuroraMJO:
    """Build and return an :class:`AuroraMJO` model.

    Args:
        config (dict): Configuration dictionary.  See :class:`AuroraMJO` for
            the full list of recognised keys.
        norm_stats (dict, optional): Per-variable normalisation statistics for
            any injected surface variables (e.g. ``ttr``, ``tcwv``).  Each
            entry is ``{var_name: {'mean': float, 'std': float}}``.

    Returns:
        :class:`AuroraMJO`: Initialized model with loaded pretrained weights.
    """
    print(f"Initializing Aurora model. Type: {config['model_type']}")

    extended_surf_vars = tuple(config["surface_variables"])
    static_vars = tuple(config.get("static_variables", ("lsm", "z", "slt")))
    model_class = AuroraPretrained if config["model_type"] == "full" else AuroraSmallPretrained

    # Construct surf_stats dict for Aurora constructor to avoid global dict mutation.
    # Note: Aurora uses surf_stats to normalise both surf_vars and static_vars.
    surf_stats: dict[str, tuple[float, float]] = {}
    if norm_stats:
        for var_name, stats in norm_stats.items():
            if var_name in extended_surf_vars or var_name in static_vars:
                surf_stats[var_name] = (float(stats["mean"]), float(stats["std"]))
                m, s = stats["mean"], stats["std"]
                print(f"   - {var_name} (surf_stats): mean={m:.4f}, std={s:.4f}")

    backbone = model_class(
        surf_vars=extended_surf_vars,
        static_vars=static_vars,
        use_lora=config["use_lora"],
        lora_mode=config.get("lora_mode", "single"),
        surf_stats=surf_stats or None,
    )

    print("Loading pre-trained weights (strict=False)")
    backbone.load_checkpoint(strict=False)

    # Note on normalisation injection:
    # Surface variables and static variables are normalised via Aurora's native
    # `surf_stats` constructor hook above, preventing process-global mutation
    # of `aurora.normalisation.locations` / `scales`.
    # Aurora exposes no equivalent constructor hook for atmospheric variables (which are
    # normalised via module-level `locations`/`scales`). If any atmospheric variables are
    # passed in norm_stats (e.g. "z_50"), global mutation is unavoidable and applied here.
    if norm_stats:
        for var_name, stats in norm_stats.items():
            if var_name not in extended_surf_vars and var_name not in static_vars:
                locations[var_name] = float(stats["mean"])
                scales[var_name] = float(stats["std"])
                m, s = stats["mean"], stats["std"]
                print(f"   - {var_name} (global atmos fallback): mean={m:.4f}, std={s:.4f}")

    if config.get("gradient_checkpointing", False):
        print("Enabling gradient checkpointing")
        backbone.configure_activation_checkpointing(
            module_names=("Swin3DTransformerBackbone",)
        )

    # ------------------------------------------------------------------
    # LoRA-aware weight freezing
    # ------------------------------------------------------------------
    # Aurora's use_lora=True inserts LoRA adapter modules but does NOT
    # call requires_grad_(False) on any backbone weights.  We do that here.
    # Freezing is applied regardless of use_lora so that even in the
    # full-fine-tune path the caller can opt in by setting
    # config['freeze_backbone'] = True explicitly.
    if config.get("freeze_backbone", True):
        print(
            "Freezing backbone (keeping LoRA adapters + new-var embeddings trainable)"
        )
        freeze_backbone(
            backbone=backbone,
            new_surf_vars=tuple(config["surface_variables"]),
            use_lora=config["use_lora"],
            static_vars=static_vars,
        )
    else:
        print("WARNING: freeze_backbone=False  - full backbone will be trained.")

    # ------------------------------------------------------------------
    # MJO head (optional)
    # ------------------------------------------------------------------
    mjo_head: MJOHead | None = None
    head_cfg = config.get("mjo_head", {})
    if head_cfg.get("enabled", False):
        # embed_dim differs by model size:
        #   AuroraSmallPretrained -> 256
        #   AuroraPretrained (full) -> 512
        embed_dim = backbone.encoder.embed_dim
        mjo_head = MJOHead(
            embed_dim=embed_dim,
            hidden_dim=head_cfg.get("hidden_dim", 256),
            dropout=head_cfg.get("dropout", 0.1),
            lat_south=head_cfg.get("lat_south", -15.0),
            lat_north=head_cfg.get("lat_north", 15.0),
        )
        print(
            f"MJO head enabled  embed_dim={embed_dim}  "
            f"hidden_dim={head_cfg.get('hidden_dim', 256)}  "
            f"tropical band=[{head_cfg.get('lat_south', -15.0)}, "
            f"{head_cfg.get('lat_north', 15.0)}]°"
        )
    else:
        print("MJO head disabled  (set config['mjo_head']['enabled']=True to activate)")

    model = AuroraMJO(backbone=backbone, mjo_head=mjo_head)
    model.model_type = config["model_type"]
    model.checkpoint_name = getattr(backbone, "default_checkpoint_name", "unknown")
    try:
        import importlib.metadata

        model.aurora_version = importlib.metadata.version("microsoft-aurora")
    except Exception:
        model.aurora_version = "unknown"

    # ------------------------------------------------------------------
    # Trainable parameter audit
    # ------------------------------------------------------------------
    _log_param_counts(model, label="AuroraMJO")

    return model
