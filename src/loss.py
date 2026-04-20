# src/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TropicalWeightedL1Loss(nn.Module):
    """
    Standard L1 (MAE) loss, but applies a multiplier to the tropical 
    region to force the model to focus on MJO-relevant latitudes.
    """
    def __init__(self, lat_coords, tropics_bbox=[-20, 20], tropics_weight=1.0, extratropics_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')

        weights = torch.ones_like(lat_coords)
        tropical_mask = (lat_coords >= tropics_bbox[0]) & (lat_coords <= tropics_bbox[1])
        
        weights[tropical_mask] = tropics_weight
        weights[~tropical_mask] = extratropics_weight
        
        # Reshape to broadcast over (Batch, Time, Lat, Lon)
        self.register_buffer('spatial_weights', weights.view(1, 1, -1, 1))

    def forward(self, pred, target):
        loss = self.l1(pred, target)
        weighted_loss = loss * self.spatial_weights
        return weighted_loss.mean()


class SpectralLoss(nn.Module):
    """
    Calculates L1 loss in the 2D frequency domain using FFT.
    Prevents the model from producing 'blurry' predictions by penalizing
    discrepancies in the power spectrum (texture and sharp gradients).
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Apply 2D Real FFT to spatial dimensions (Lat, Lon)
        pred_fft = torch.fft.rfft2(pred.float(), norm='ortho')
        target_fft = torch.fft.rfft2(target.float(), norm='ortho')
        
        # Return Mean Absolute Error of the complex amplitudes
        return torch.abs(pred_fft - target_fft).mean()

class MoistureBudgetLoss(nn.Module):
    """
    Physics-Informed Loss: Column-Integrated Moisture Conservation.

    Mathematical basis
    ==================
    The vertically-integrated moisture budget over an atmospheric column is:

        d<q>/dt + div(<v·q>) = E − P

    where:
        <·>     = vertical integral ∫(·) dp/g  over pressure levels
        q       = specific humidity  (kg/kg)
        v       = (u, v) horizontal wind  (m/s)
        E       = surface evaporation  (kg/m²/s)
        P       = precipitation  (kg/m²/s)

    Aurora does NOT predict E and P directly.  Instead we compute the
    **implied E−P residual**:

        R  =  d<q>/dt  +  div(<v·q>)

    and penalise its magnitude.  A model that oversmooths convection
    creates artificially large moisture sources / sinks (large |R|);
    penalising |R| encourages column-wise moisture conservation.

    The residual is converted from SI (kg/m²/s) to mm/day (×86 400) so
    the loss magnitude is O(1–10), compatible with the grid L1 loss.

    Only the tropical band (configurable, default ±20°) is included in
    the loss, since MJO convection lives in the tropics.

    Numerical details
    -----------------
    * Spherical divergence with circular longitude padding.
    * cos(lat) clamped ≥ 1e-5 to avoid pole singularities.
    * Central finite differences for both lat and lon derivatives.

    Input contract
    --------------
    ``forward(in_batch, pred_batch)`` where both arguments are Aurora
    ``Batch`` objects whose ``.atmos_vars`` contain ``'q'``, ``'u'``,
    ``'v'`` with shape ``(B, T, levels, H, W)``.
    """

    def __init__(
        self,
        pressure_levels,
        latitudes,
        longitudes,
        dt_seconds=21600,
        tropics_bbox=(-20, 20),
    ):
        super().__init__()
        self.dt = dt_seconds
        self.g = 9.80665
        self.R = 6371000.0

        # ---- Grid tensors ------------------------------------------------
        plevs = torch.tensor(pressure_levels, dtype=torch.float32) * 100.0  # hPa→Pa
        self.register_buffer("plevs", plevs)

        lats = torch.tensor(latitudes, dtype=torch.float32)
        lons = torch.tensor(longitudes, dtype=torch.float32)
        self.register_buffer("lats", lats)
        self.register_buffer("lons", lons)

        # Layer-thickness weights (central differences at interior, one-sided at edges)
        dp = torch.zeros_like(plevs)
        dp[0] = plevs[1] - plevs[0]
        dp[1:-1] = (plevs[2:] - plevs[:-2]) / 2.0
        dp[-1] = plevs[-1] - plevs[-2]
        self.register_buffer("dp", dp.view(1, 1, -1, 1, 1))  # (1,1,L,1,1)

        # ---- Spherical geometry ------------------------------------------
        dlat_rad = torch.deg2rad(torch.abs(lats[0] - lats[1]))
        dlon_rad = torch.deg2rad(torch.abs(lons[1] - lons[0]))

        self.dy = self.R * dlat_rad  # constant

        cos_lat_raw = torch.cos(torch.deg2rad(lats))
        cos_lat_safe = torch.clamp(cos_lat_raw, min=1e-5).view(1, 1, -1, 1)
        self.register_buffer("cos_lat", cos_lat_safe)  # (1,1,H,1)
        self.register_buffer("dx", self.R * cos_lat_safe * dlon_rad)

        # ---- Tropical mask -----------------------------------------------
        lat_south, lat_north = tropics_bbox
        trop_mask = (lats >= lat_south) & (lats <= lat_north)  # (H,)
        self.register_buffer("trop_mask", trop_mask)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def vertical_integral(self, x):
        """Column integral  <X> = Σ X · dp / g  over pressure-level dim (dim=2)."""
        return torch.sum(x * self.dp / self.g, dim=2)  # (B,T,H,W)

    def spherical_divergence(self, u_flux, v_flux):
        """
        div = (1 / R cos φ) [∂u/∂λ  +  ∂(v cos φ)/∂φ]

        Uses circular padding in longitude and replicate padding in latitude.
        Inputs/outputs have shape (B, T, H, W).
        """
        # ∂u/∂λ — circular in longitude
        u_padded = F.pad(u_flux, pad=(1, 1, 0, 0), mode="circular")
        du_dlon = (u_padded[..., 2:] - u_padded[..., :-2]) / (2.0 * self.dx)

        # ∂(v cos φ)/∂φ — replicate at poles
        v_cos_lat = v_flux * self.cos_lat
        v_padded = F.pad(v_cos_lat, pad=(0, 0, 1, 1), mode="replicate")
        # lats decrease (90→−90): index 0 = North
        dv_dlat = (v_padded[..., :-2, :] - v_padded[..., 2:, :]) / (2.0 * self.dy)

        return du_dlon + dv_dlat / self.cos_lat

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, in_batch, pred_batch):
        """Compute the tropical moisture-budget residual loss.

        Args:
            in_batch:  Aurora ``Batch`` for the current time step.
            pred_batch: Aurora ``Batch`` for the predicted next time step.

        Returns:
            Scalar loss (mean |residual| in mm/day over the tropical band).
            Returns ``0`` (with grad) if required atmos vars are missing.
        """
        # Guard: need q, u, v in both batches
        required = {"q", "u", "v"}
        for name, batch in [("in_batch", in_batch), ("pred_batch", pred_batch)]:
            if not hasattr(batch, "atmos_vars"):
                return torch.tensor(0.0, device=self.plevs.device, requires_grad=True)
            if not required.issubset(batch.atmos_vars.keys()):
                return torch.tensor(0.0, device=self.plevs.device, requires_grad=True)

        q_curr = in_batch.atmos_vars["q"]     # (B, T, L, H, W)
        q_next = pred_batch.atmos_vars["q"]
        u_next = pred_batch.atmos_vars["u"]
        v_next = pred_batch.atmos_vars["v"]

        # Column integrals  →  (B, T, H, W)
        int_q_curr = self.vertical_integral(q_curr)
        int_q_next = self.vertical_integral(q_next)
        int_uq = self.vertical_integral(u_next * q_next)
        int_vq = self.vertical_integral(v_next * q_next)

        # Moisture tendency + divergence  (kg/m²/s)
        dq_dt = (int_q_next - int_q_curr) / self.dt
        div_flux = self.spherical_divergence(int_uq, int_vq)
        residual_si = dq_dt + div_flux  # implied E−P

        # Convert to mm/day for O(1) optimizer visibility
        residual_mm_day = residual_si * 86400.0

        # Restrict to tropical band
        residual_trop = residual_mm_day[:, :, self.trop_mask, :]

        return torch.abs(residual_trop).mean()