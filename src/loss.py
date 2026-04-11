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
    Physics-Informed Loss: Integrated Column Moisture Budget.
    d<q>/dt + div(<v*q>) - (E - P) = 0
    Optimized for numerical stability (NaN prevention) and spherical topology.
    """
    def __init__(self, pressure_levels, latitudes, longitudes, dt_seconds=21600):
        super().__init__()
        self.dt = dt_seconds  
        self.g = 9.80665      
        self.R = 6371000.0    
        self.rho_w = 1000.0   
        
        # Grid setup
        self.register_buffer('plevs', torch.tensor(pressure_levels, dtype=torch.float32) * 100.0) # Pa
        self.register_buffer('lats', torch.tensor(latitudes, dtype=torch.float32))
        self.register_buffer('lons', torch.tensor(longitudes, dtype=torch.float32))
        
        # DP calculation
        dp = torch.zeros_like(self.plevs)
        dp[0] = self.plevs[1] - self.plevs[0]
        dp[1:-1] = (self.plevs[2:] - self.plevs[:-2]) / 2.0
        dp[-1] = self.plevs[-1] - self.plevs[-2]
        self.register_buffer('dp', dp.view(1, 1, -1, 1, 1))

        # --- SPHERICAL MATH FIXES ---
        
        # 1. dlat and dlon in radians
        dlat_rad = torch.deg2rad(torch.abs(self.lats[0] - self.lats[1]))
        dlon_rad = torch.deg2rad(torch.abs(self.lons[1] - self.lons[0]))
        
        # 2. dy is constant
        self.dy = self.R * dlat_rad
        
        # 3. cos(lat) - Clamped to prevent division by zero at the poles!
        cos_lat_raw = torch.cos(torch.deg2rad(self.lats))
        cos_lat_safe = torch.clamp(cos_lat_raw, min=1e-5).view(1, 1, -1, 1)
        self.register_buffer('cos_lat', cos_lat_safe)
        
        # dx depends on latitude
        self.register_buffer('dx', self.R * self.cos_lat * dlon_rad)

    def vertical_integral(self, x):
        """ <X> = int(X dp/g) """
        return torch.sum(x * self.dp / self.g, dim=2) 

    def spherical_divergence(self, u_flux, v_flux):
        """
        Computes div = 1/(R*cos(lat)) * [d(u)/d(lon) + d(v*cos(lat))/d(lat)]
        Uses Circular Padding for Longitude to respect Earth's topology.
        """
        # --- LONGITUDE DERIVATIVE (With Circular Padding) ---
        # Pad the last dimension (longitude) by 1 on both sides
        u_padded = F.pad(u_flux, pad=(1, 1, 0, 0), mode='circular')
        # Central difference: (right - left) / (2 * dx)
        du_dlon = (u_padded[..., 2:] - u_padded[..., :-2]) / (2.0 * self.dx)
        
        # --- LATITUDE DERIVATIVE (With Replication Padding) ---
        v_cos_lat = v_flux * self.cos_lat
        # Pad latitude (dim=-2) with replication (poles don't wrap around)
        v_padded = F.pad(v_cos_lat, pad=(0, 0, 1, 1), mode='replicate')
        # Central difference
        # Note: If lats are decreasing (90 to -90), index 0 is North, index 2 is South.
        # We need the gradient in the positive y (North) direction.
        dv_dlat = (v_padded[..., :-2, :] - v_padded[..., 2:, :]) / (2.0 * self.dy)
        
        # Combine and divide by cos(lat)
        divergence = du_dlon + (dv_dlat / self.cos_lat)
        return divergence

    def forward(self, q_curr, preds_next):
        # 1. Extract and integrate
        q_next, u_next, v_next = preds_next['q'], preds_next['u'], preds_next['v']
        
        # If the model didn't output E and P directly, we gracefully return 0 loss 
        # so the code doesn't crash during Phase 1 (before we add E and P).
        if 'evap' not in preds_next or 'precip' not in preds_next:
            return torch.tensor(0.0, device=q_curr.device, requires_grad=True)

        E_next = preds_next['evap']
        P_next = (preds_next['precip'] * self.rho_w) / self.dt 

        int_q_curr = self.vertical_integral(q_curr)
        int_q_next = self.vertical_integral(q_next)
        int_uq_next = self.vertical_integral(u_next * q_next)
        int_vq_next = self.vertical_integral(v_next * q_next)

        # 2. Kinematics
        dq_dt = (int_q_next - int_q_curr) / self.dt
        div_flux = self.spherical_divergence(int_uq_next, int_vq_next)

        # 3. Residual calculation (kg / m^2 / s)
        residual_si = dq_dt + div_flux - (E_next - P_next)

        # --- THE ML SCALING FIX ---
        # Convert kg/m^2/s to mm/day. 
        # 1 kg/m^2 of water is 1 mm depth. 1 day is 86400 seconds.
        # This scales the loss from ~10^-5 up to ~O(1), making it visible to the optimizer.
        residual_mm_day = residual_si * 86400.0

        return torch.abs(residual_mm_day).mean()