# src/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TropicalWeightedL1Loss(nn.Module):
    """
    Standard L1 (MAE) loss, but applies a multiplier to the tropical
    region to force the model to focus on MJO-relevant latitudes.

    Accepts tensors of any shape as long as one dimension matches the
    latitude axis length (720 for Aurora 0.25deg grid).  Weights are
    stored as 1-D and reshaped dynamically so the loss works with
    per-variable inputs of different ranks:
      - Surface:  (B, H, W)      or (B, 1, H, W)
      - Atmos:    (B, levels, H, W)
    """
    def __init__(self, lat_coords, tropics_bbox=[-20, 20], tropics_weight=1.0, extratropics_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')
        self.n_lat = lat_coords.shape[0]

        weights = torch.ones_like(lat_coords)
        tropical_mask = (lat_coords >= tropics_bbox[0]) & (lat_coords <= tropics_bbox[1])

        weights[tropical_mask] = tropics_weight
        weights[~tropical_mask] = extratropics_weight

        self.register_buffer('lat_weights', weights)  # (H,)

    def forward(self, pred, target):
        loss = self.l1(pred, target)

        lat_dim = None
        for d in range(1, loss.ndim - 1):
            if loss.shape[d] == self.n_lat:
                lat_dim = d
                break

        if lat_dim is not None:
            shape = [1] * loss.ndim
            shape[lat_dim] = self.n_lat
            w = self.lat_weights.view(*shape)
            loss = loss * w

        return loss.mean()


class SpectralLoss(nn.Module):
    """
    Calculates L1 loss in the 2D frequency domain using FFT.
    Prevents the model from producing 'blurry' predictions by penalizing
    discrepancies in the power spectrum (texture and sharp gradients).
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred.float(), norm='ortho')
        target_fft = torch.fft.rfft2(target.float(), norm='ortho')
        return torch.abs(pred_fft - target_fft).mean()


class MoistureBudgetLoss(nn.Module):
    """
    Physics-Informed Loss: Column-Integrated Moisture Conservation.

    Mathematical basis
    ==================
    The vertically-integrated moisture budget over an atmospheric column is:

        d<q>/dt + div(<v dot q>) = E - P

    where:
        <dot>     = vertical integral ∫(dot) dp/g  over pressure levels
        q       = specific humidity  (kg/kg)
        v       = (u, v) horizontal wind  (m/s)
        E       = surface evaporation  (kg/m^2/s)
        P       = precipitation  (kg/m^2/s)

    Aurora does NOT predict E and P directly.  We compute the **implied
    E−P residual**  R = d<q>/dt + div(<v dot q>)  and penalise its magnitude.
    A model that oversmooths convection creates artificially large moisture
    sources/sinks (large |R|); penalising |R| encourages column-wise
    moisture conservation.  Converted to mm/day (x86 400) so the loss is
    O(1–10) and compatible with the grid L1 loss.  Only the tropical band
    (default +-20deg) is included, since MJO convection lives in the tropics.

    v2 FIXES relative to the original implementation
    ================================================
    1. **History-dim bug (silently wrong physics).**  Aurora input batches
       carry a 2-step history window: q_curr had shape (B, 2, L, H, W)
       while the prediction has (B, 1, L, H, W).  The old code broadcast
       (B,1,dot) − (B,2,dot) -> (B,2,dot), producing TWO residuals: one against
       time t (correct, Δt = 6 h) and one against t−6 h (a 12 h difference
       divided by 6 h  - physically wrong and doubling the loss signal).
       We now explicitly slice the LAST history step:  q_curr[:, -1:].
    2. **Mixed-precision safety.**  q ~ 1e-3 kg/kg; the tendency is a small
       difference of small numbers.  Under bf16 autocast this is mostly
       rounding noise, and it contributed to the cuBLAS/bf16 instability in
       job 52464118.  The whole residual is now computed with autocast
       DISABLED in float32.  (Gradients flow back into the autocast region
       normally  - this is fully differentiable.)
    3. **NaN/Inf hygiene.**  Inputs pass through nan_to_num, the residual is
       clamped to +-1e4 mm/day (physical |E−P| is <300 mm/day; the clamp only
       stops a transient blow-up from flooding the optimizer with inf grads),
       and a final non-finite guard returns a zero loss for that step
       instead of poisoning the whole batch.
    4. **Grid-mismatch guard.**  If the incoming field's lat size doesn't
       match the configured grid (e.g. 64x128 smoke tests), we return a
       zero (grad-carrying) loss instead of a shape error.

    Numerical scheme (unchanged, verified correct):
      * Spherical divergence  div = (1/(R cos phi ))[partial u/partial  lambda  + partial (v cos phi )/partial  phi ]
        with circular longitude padding and replicate latitude padding;
        the latitude axis runs 90->−90 so the north-minus-south central
        difference keeps the correct sign.
      * cos(lat) clamped  geq 1e-5 to avoid pole singularities.
      * Layer thicknesses dp via central differences (one-sided at edges).
    """

    def __init__(
        self,
        pressure_levels,
        latitudes,
        longitudes,
        dt_seconds=21600,
        tropics_bbox=(-20, 20),
        residual_clamp_mm_day: float = 1.0e4,
    ):
        super().__init__()
        self.dt = float(dt_seconds)
        self.g = 9.80665
        self.R = 6371000.0
        self.residual_clamp = float(residual_clamp_mm_day)

        # ---- Grid tensors ------------------------------------------------
        plevs = torch.tensor(pressure_levels, dtype=torch.float32) * 100.0  # hPa->Pa
        self.register_buffer("plevs", plevs)

        lats = torch.tensor(latitudes, dtype=torch.float32)
        lons = torch.tensor(longitudes, dtype=torch.float32)
        self.register_buffer("lats", lats)
        self.register_buffer("lons", lons)

        # Layer-thickness weights (central differences interior, one-sided edges)
        dp = torch.zeros_like(plevs)
        dp[0] = plevs[1] - plevs[0]
        dp[1:-1] = (plevs[2:] - plevs[:-2]) / 2.0
        dp[-1] = plevs[-1] - plevs[-2]
        self.register_buffer("dp", dp.view(1, 1, -1, 1, 1))  # (1,1,L,1,1)

        # ---- Spherical geometry ------------------------------------------
        dlat_rad = torch.deg2rad(torch.abs(lats[0] - lats[1]))
        dlon_rad = torch.deg2rad(torch.abs(lons[1] - lons[0]))

        self.dy = self.R * dlat_rad  # constant (meters per lat grid step x2 applied below)

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
        """Column integral  <X> = Σ X dot dp / g  over pressure-level dim (dim=2)."""
        return torch.sum(x * self.dp / self.g, dim=2)  # (B,T,H,W)

    def spherical_divergence(self, u_flux, v_flux):
        """div = (1 / R cos phi ) [partial u/partial  lambda  + partial (v cos phi )/partial  phi ] on (B,T,H,W) fields."""
        # partial u/partial  lambda  term  - circular in longitude; dx already contains Rdotcos phi dotd lambda .
        u_padded = F.pad(u_flux, pad=(1, 1, 0, 0), mode="circular")
        du_dlon = (u_padded[..., 2:] - u_padded[..., :-2]) / (2.0 * self.dx) 

        # partial (v cos phi )/partial  phi - replicate at poles.  lats run 90->−90 so
        # (north − south) is the correct + phi  direction.
        v_cos_lat = v_flux * self.cos_lat
        v_padded = F.pad(v_cos_lat, pad=(0, 0, 1, 1), mode="replicate")
        dv_dlat = (v_padded[..., :-2, :] - v_padded[..., 2:, :]) / (2.0 * self.dy)

        return du_dlon + dv_dlat / self.cos_lat

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, in_batch, pred_batch):
        """Tropical moisture-budget residual loss (scalar, mm/day,  geq0).

        Args:
            in_batch:   Aurora ``Batch`` fed into the current forward pass
                        (atmos_vars shaped (B, T_hist, L, H, W), T_hist = 2).
            pred_batch: Aurora ``Batch`` predicted for the next time step
                        (atmos_vars shaped (B, 1, L, H, W)).
        """
        device = self.plevs.device

        # Guard: need q, u, v in both batches
        required = {"q", "u", "v"}
        for batch in (in_batch, pred_batch):
            if not hasattr(batch, "atmos_vars") or not required.issubset(batch.atmos_vars.keys()):
                return torch.zeros((), device=device, requires_grad=True)

        # FIX 1: take ONLY the latest history step so the tendency is a
        # true 6-h difference. (B, T_hist, L, H, W) -> (B, 1, L, H, W).
        q_curr = in_batch.atmos_vars["q"][:, -1:, ...]
        q_next = pred_batch.atmos_vars["q"][:, -1:, ...]
        u_next = pred_batch.atmos_vars["u"][:, -1:, ...]
        v_next = pred_batch.atmos_vars["v"][:, -1:, ...]

        # Grid-mismatch guard (smoke tests / low-res debugging).
        if q_next.shape[-2] != self.lats.shape[0] or q_next.shape[-1] != self.lons.shape[0]:
            return torch.zeros((), device=device, requires_grad=True)

        # FIX 2: force float32 outside autocast  - the residual is a small
        # difference of small numbers and is garbage in bf16.
        with torch.autocast(device_type=q_next.device.type, enabled=False):
            q_curr = torch.nan_to_num(q_curr.float())
            q_next_f = torch.nan_to_num(q_next.float())
            u_next_f = torch.nan_to_num(u_next.float())
            v_next_f = torch.nan_to_num(v_next.float())

            # Column integrals -> (B, 1, H, W)
            int_q_curr = self.vertical_integral(q_curr)
            int_q_next = self.vertical_integral(q_next_f)
            int_uq = self.vertical_integral(u_next_f * q_next_f)
            int_vq = self.vertical_integral(v_next_f * q_next_f)

            # Moisture tendency + flux divergence (kg/m²/s)
            dq_dt = (int_q_next - int_q_curr) / self.dt
            div_flux = self.spherical_divergence(int_uq, int_vq)
            residual_si = dq_dt + div_flux  # implied E−P

            # Convert to mm/day, clamp against transient blow-ups (FIX 3).
            residual_mm_day = residual_si * 86400.0
            residual_mm_day = torch.clamp(
                residual_mm_day, -self.residual_clamp, self.residual_clamp
            )

            # Restrict to the tropical band and reduce.
            residual_trop = residual_mm_day[:, :, self.trop_mask, :]
            loss = torch.abs(residual_trop).mean()

            # Final non-finite guard  - never let NaN/Inf reach the optimizer.
            if not torch.isfinite(loss):
                return torch.zeros((), device=device, requires_grad=True)

        return loss
