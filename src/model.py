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

import dataclasses
from typing import Optional, Union

import torch
import torch.nn as nn
from aurora import Aurora, AuroraSmallPretrained
from aurora.batch import Batch
from aurora.normalisation import locations, scales


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
        patch_lat = patch_lat[:n_h]                     # guard against rounding

        # Tropical mask over the patch-lat dimension.
        trop_mask = (patch_lat >= self.lat_south) & (patch_lat <= self.lat_north)  # (n_h,)

        # Pool: mean over levels, tropical latitudes, and all longitudes.
        # Shape after tropical slice: (B, n_levels, n_trop, n_w, D)
        x_trop = x_spatial[:, :, trop_mask, :, :]  # (B, n_levels, n_trop, n_w, D)
        x_pooled = x_trop.mean(dim=(1, 2, 3))       # (B, D)

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
        ``'huge'`` → ``Aurora``, anything else → ``AuroraSmallPretrained``.
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
        backbone: Aurora,
        mjo_head: Optional[MJOHead] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.mjo_head = mjo_head

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, batch: Batch
    ) -> Union[Batch, tuple[Batch, torch.Tensor]]:
        """Run the dual-head forward pass.

        When the MJO head is disabled this is a transparent wrapper around
        the Aurora backbone and returns a ``Batch``.

        When the MJO head is enabled the encoder latent is intercepted via a
        forward hook, the Aurora backbone still runs its full decoder, and the
        return value is ``(pred_batch, mjo_pred)`` where ``mjo_pred`` has
        shape ``(B, 3)`` → ``[RMM1, RMM2, Amplitude]``.

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

        def _hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:  # noqa: ANN001
            _encoder_output["x"] = output

        hook_handle = self.backbone.encoder.register_forward_hook(_hook)

        try:
            # Run the full backbone (encoder → swin → decoder).
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
        n_spatial = n_tokens // n_levels            # H_p * W_p
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
# Factory
# ---------------------------------------------------------------------------

def load_model(config: dict, norm_stats: Optional[dict] = None) -> AuroraMJO:
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
    model_class = Aurora if config["model_type"] == "huge" else AuroraSmallPretrained

    backbone = model_class(
        surf_vars=extended_surf_vars,
        use_lora=config["use_lora"],
        lora_mode=config.get("lora_mode", "single"),
    )

    print("Loading pre-trained weights (strict=False)")
    backbone.load_checkpoint(strict=False)

    if norm_stats:
        print("Injecting normalisation statistics for new variables")
        for var_name, stats in norm_stats.items():
            locations[var_name] = stats["mean"]
            scales[var_name] = stats["std"]
            print(f"   - {var_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    if config.get("gradient_checkpointing", False):
        print("Enabling gradient checkpointing")
        backbone.configure_activation_checkpointing()

    # ------------------------------------------------------------------
    # MJO head (optional)
    # ------------------------------------------------------------------
    mjo_head: Optional[MJOHead] = None
    head_cfg = config.get("mjo_head", {})
    if head_cfg.get("enabled", False):
        # embed_dim differs by model size:
        #   AuroraSmallPretrained → 256
        #   Aurora (huge)         → 512
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

    return AuroraMJO(backbone=backbone, mjo_head=mjo_head)