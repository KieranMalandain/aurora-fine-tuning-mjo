# src/loss.py
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn

DEFAULT_AURORA_PLEVS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

# Default variable weights w_v matching 02_SCIENTIFIC_CONTRACT.md §4.1:
# MJO convective variables: 2.0; dynamical wind/temp: 1.0; state/surface: 0.5.
# Set a priori based on physical domain priors and NOT tuned against validation skill.
DEFAULT_VARIABLE_WEIGHTS: dict[str, float] = {
    "ttr": 2.0,
    "tcwv": 2.0,
    "q": 2.0,
    "u": 1.0,
    "v": 1.0,
    "t": 1.0,
    "z": 0.5,
    "msl": 0.5,
    "2t": 0.5,
    "10u": 0.5,
    "10v": 0.5,
}


class TropicalWeightedL1Loss(nn.Module):
    """Normalised, weighted, area-correct grid loss (02_SCIENTIFIC_CONTRACT.md §4).

    Mathematical specification
    ==========================
    L_grid = Σ_v w_v · ( 1 / (H · W) ) · Σ_ℓ Σ_φ Σ_λ c_ℓ · a(φ) · m(φ) · | x̂_vℓφλ − x_vℓφλ |

    where:
      - x̂, x are normalised by the training-period constants (1980–2015 Welford stats from G3).
        Aurora denormalises its output, so both prediction and target are re-normalised before
        loss calculation (the denormalise-then-renormalise path).
      - w_v is the per-variable weight (config-visible, defaults in §4.1).
      - c_ℓ is the per-level weight (default: Δp_ℓ / Σ Δp, summing to 1.0; c_ℓ ≡ 1 for surface vars).
      - a(φ) = cos(φ) / mean(cos(φ)) is area weighting, integrating to 1 over the sphere.
      - m(φ) is the tropical emphasis (1.0 inside ±20°, 0.1 outside).
      - a(φ) and m(φ) are separate, applied multiplicatively.
    """

    def __init__(
        self,
        lat_coords: torch.Tensor | list[float] | None = None,
        tropics_bbox: list[int] | tuple[int, int] = (-20, 20),
        tropics_weight: float = 1.0,
        extratropics_weight: float = 0.1,
        level_weighting: str = "pressure_delta",
        pressure_levels: list[int | float] | tuple[int | float, ...] | None = None,
        variable_weights: dict[str, float] | None = None,
        norm_stats: dict | None = None,
        norm_stats_file: str | Path | None = None,
    ):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")
        self.level_weighting = level_weighting.lower()

        # ---- Coordinates & Grid Geometry ---------------------------------
        if lat_coords is None:
            lat_coords = torch.linspace(89.5, -89.5, 180)
        lat_tensor = torch.as_tensor(lat_coords, dtype=torch.float32)
        self.n_lat = lat_tensor.shape[0]

        # Cos-latitude area weighting: a(φ) = cos(φ) / mean(cos(φ))
        cos_lat = torch.cos(torch.deg2rad(lat_tensor))
        cos_lat = torch.clamp(cos_lat, min=0.0)
        a_phi = cos_lat / cos_lat.mean()  # mean is 1.0 (integrates to 1 over the sphere)

        # Tropical emphasis mask: m(φ)
        m_phi = torch.full_like(lat_tensor, float(extratropics_weight))
        trop_mask = (lat_tensor >= tropics_bbox[0]) & (lat_tensor <= tropics_bbox[1])
        m_phi[trop_mask] = float(tropics_weight)

        # Multiplicative separation: store a(φ) and m(φ) separately
        self.register_buffer("area_weights", a_phi)
        self.register_buffer("tropical_weights", m_phi)
        self.register_buffer("spatial_weights", a_phi * m_phi)

        # ---- Vertical Level Weighting c_ℓ --------------------------------
        plevs_list = list(pressure_levels or DEFAULT_AURORA_PLEVS)
        plevs_t = torch.tensor(plevs_list, dtype=torch.float32)
        self.register_buffer("plevs", plevs_t)

        if self.level_weighting == "pressure_delta":
            dp = torch.zeros_like(plevs_t)
            dp[0] = plevs_t[1] - plevs_t[0]
            dp[1:-1] = (plevs_t[2:] - plevs_t[:-2]) / 2.0
            dp[-1] = plevs_t[-1] - plevs_t[-2]
            c_l = dp / dp.sum()
        elif self.level_weighting == "uniform":
            c_l = torch.ones_like(plevs_t) / len(plevs_t)
        else:
            raise ValueError(
                f"Unknown level_weighting: {level_weighting!r}. Must be 'pressure_delta' or 'uniform'."
            )
        self.register_buffer("c_l", c_l)

        # ---- Variable Weights w_v ----------------------------------------
        self.variable_weights: dict[str, float] = dict(DEFAULT_VARIABLE_WEIGHTS)
        if variable_weights:
            self.variable_weights.update(variable_weights)

        # ---- Normalisation Constants (G3 1980–2015 Welford Statistics) ---
        stats_dict = self._load_norm_stats(norm_stats, norm_stats_file)
        self.surf_vars = ("2t", "10u", "10v", "msl", "ttr", "tcwv", "sst", "ps")
        self.atmos_vars = ("z", "q", "t", "u", "v")

        # Register normalisation buffers for surface variables
        for v in self.surf_vars:
            mean_val, std_val = self._extract_surf_stats(v, stats_dict)
            self.register_buffer(f"surf_mean_{v}", torch.tensor(mean_val, dtype=torch.float32))
            self.register_buffer(f"surf_std_{v}", torch.tensor(std_val, dtype=torch.float32))

        # Register normalisation buffers for atmospheric variables across 13 levels
        for v in self.atmos_vars:
            mean_vec, std_vec = self._extract_atmos_stats(v, stats_dict, plevs_list)
            self.register_buffer(f"atmos_mean_{v}", mean_vec)
            self.register_buffer(f"atmos_std_{v}", std_vec)

    @staticmethod
    def _load_norm_stats(
        norm_stats: dict | None,
        norm_stats_file: str | Path | None,
    ) -> dict:
        if norm_stats is not None:
            return norm_stats

        candidate_paths = []
        if norm_stats_file is not None:
            candidate_paths.append(Path(norm_stats_file))
        repo_root = Path(__file__).resolve().parent.parent.parent
        candidate_paths.extend([
            repo_root / "configs" / "norm_stats_1980_2015.yaml",
            Path("configs/norm_stats_1980_2015.yaml"),
        ])

        for p in candidate_paths:
            if p.is_file():
                with open(p) as f:
                    return yaml.safe_load(f) or {}
        return {}

    def _extract_surf_stats(self, var_name: str, stats_dict: dict) -> tuple[float, float]:
        # 1. From passed G3 norm stats dict
        surf_stats = stats_dict.get("surface", {})
        if var_name in surf_stats:
            entry = surf_stats[var_name]
            return float(entry["mean"]), float(entry["std"])
        # 2. Check top-level (if unified.yaml model.norm_stats shape)
        if var_name in stats_dict and isinstance(stats_dict[var_name], dict):
            entry = stats_dict[var_name]
            if "mean" in entry and "std" in entry:
                return float(entry["mean"]), float(entry["std"])
        # 3. Fallback to aurora.normalisation
        import aurora.normalisation as norm

        loc = float(norm.locations.get(var_name, 0.0))
        scale = float(norm.scales.get(var_name, 1.0))
        return loc, scale

    def _extract_atmos_stats(
        self,
        var_name: str,
        stats_dict: dict,
        plevs_list: list[int | float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        means, stds = [], []
        atmos_stats = stats_dict.get("atmos", {}).get(var_name, {})

        import aurora.normalisation as norm

        for lev in plevs_list:
            lev_int = int(lev)
            if lev_int in atmos_stats or str(lev_int) in atmos_stats:
                entry = atmos_stats.get(lev_int) or atmos_stats[str(lev_int)]
                means.append(float(entry["mean"]))
                stds.append(float(entry["std"]))
            else:
                aurora_key = f"{var_name}_{lev_int}"
                means.append(float(norm.locations.get(aurora_key, 0.0)))
                stds.append(float(norm.scales.get(aurora_key, 1.0)))

        return torch.tensor(means, dtype=torch.float32), torch.tensor(stds, dtype=torch.float32)

    def get_var_weight(self, var_name: str) -> float:
        """Return the scalar weight w_v for variable var_name."""
        return float(self.variable_weights.get(var_name, 1.0))

    def loss_per_var(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        var_name: str | None = None,
    ) -> torch.Tensor:
        """Compute normalised, area-weighted, level-weighted L1 loss for a single variable.

        Denormalise-then-renormalise path (02_SCIENTIFIC_CONTRACT.md §4):
        Aurora outputs prognostic variables in denormalised physical units.
        To prevent large-magnitude variables like msl (~9500 Pa) and z (~3800 m^2/s^2)
        from dominating 99% of the gradient budget over small-magnitude moisture
        variables like q (~0.0016 kg/kg), both prediction and target are re-normalised
        using the training population constants (1980–2015 Welford stats from G3)
        prior to computing the L1 error.
        """
        # --- Surface Variable Path ---
        if var_name is not None and hasattr(self, f"surf_mean_{var_name}"):
            mean = getattr(self, f"surf_mean_{var_name}")
            std = getattr(self, f"surf_std_{var_name}")
            p_norm = (pred - mean) / std
            tgt_norm = (target - mean) / std
            diff = torch.abs(p_norm - tgt_norm)

            lat_dim = self._find_lat_dim(diff)
            if lat_dim is not None:
                shape = [1] * diff.ndim
                shape[lat_dim] = self.n_lat
                w = self.spatial_weights.view(*shape)
                return (diff * w).mean()
            return diff.mean()

        # --- Atmospheric Variable Path ---
        if var_name is not None and hasattr(self, f"atmos_mean_{var_name}"):
            mean = getattr(self, f"atmos_mean_{var_name}")
            std = getattr(self, f"atmos_std_{var_name}")

            level_dim = self._find_level_dim(pred)
            if level_dim is not None:
                level_shape = [1] * pred.ndim
                level_shape[level_dim] = len(self.plevs)

                p_norm = (pred - mean.view(*level_shape)) / std.view(*level_shape)
                tgt_norm = (target - mean.view(*level_shape)) / std.view(*level_shape)
                diff = torch.abs(p_norm - tgt_norm)

                # Weight levels by c_ℓ (pressure-delta or uniform)
                c_l_view = self.c_l.view(*level_shape)
                level_diff = (diff * c_l_view).sum(dim=level_dim)
            else:
                diff = torch.abs(pred - target)
                level_diff = diff.mean(dim=1) if pred.ndim >= 4 else diff

            # Area weighting and tropical mask on remaining spatial dimensions
            lat_dim = self._find_lat_dim(level_diff)
            if lat_dim is not None:
                spatial_shape = [1] * level_diff.ndim
                spatial_shape[lat_dim] = self.n_lat
                w = self.spatial_weights.view(*spatial_shape)
                return (level_diff * w).mean()
            return level_diff.mean()

        # --- Fallback / Unnormalised Path (e.g. legacy/probe calls without var_name) ---
        diff = torch.abs(pred - target)
        level_dim = self._find_level_dim(diff)
        if level_dim is not None:
            level_shape = [1] * diff.ndim
            level_shape[level_dim] = len(self.plevs)
            c_l_view = self.c_l.view(*level_shape)
            diff = (diff * c_l_view).sum(dim=level_dim)

        lat_dim = self._find_lat_dim(diff)
        if lat_dim is not None:
            spatial_shape = [1] * diff.ndim
            spatial_shape[lat_dim] = self.n_lat
            w = self.spatial_weights.view(*spatial_shape)
            return (diff * w).mean()
        return diff.mean()

    def _find_lat_dim(self, t: torch.Tensor) -> int | None:
        for d in range(t.ndim):
            if t.shape[d] == self.n_lat:
                return d
        return None

    def _find_level_dim(self, t: torch.Tensor) -> int | None:
        n_levs = len(self.plevs)
        for d in range(t.ndim):
            if t.shape[d] == n_levs:
                return d
        return None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        var_name: str | None = None,
    ) -> torch.Tensor:
        """Forward pass for a single variable prediction and target."""
        return self.loss_per_var(pred, target, var_name=var_name)

    def compute_batch(
        self,
        pred_batch,
        target_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute weighted composite grid loss and per-variable components for a batch.

        Returns:
            (total_weighted_loss, dict_of_per_var_losses)
        """
        grid_var_losses: dict[str, torch.Tensor] = {}
        weighted_losses: list[torch.Tensor] = []

        for attr in ("surf_vars", "atmos_vars"):
            if hasattr(pred_batch, attr):
                for var_name, pred_t in getattr(pred_batch, attr).items():
                    if var_name in target_dict:
                        tgt_t = target_dict[var_name].to(pred_t.device).float()
                        p = pred_t.float()
                        while p.ndim > tgt_t.ndim and p.shape[1] == 1:
                            p = p.squeeze(1)
                        while tgt_t.ndim > p.ndim and (tgt_t.shape[0] == 1 or tgt_t.shape[1] == 1):
                            tgt_t = tgt_t.squeeze(0) if tgt_t.shape[0] == 1 else tgt_t.squeeze(1)

                        v_loss = self.loss_per_var(p, tgt_t, var_name=var_name)
                        w_v = self.get_var_weight(var_name)
                        grid_var_losses[var_name] = v_loss
                        weighted_losses.append(w_v * v_loss)

        zero_dev = self.c_l.device
        total_loss = sum(weighted_losses) if weighted_losses else torch.zeros((), device=zero_dev)
        return total_loss, grid_var_losses


class SpectralLoss(nn.Module):
    """Per-variable 2-D spatial amplitude-spectrum loss (H2).

    Motivation and defect corrected
    ================================
    Before H2, `SpectralLoss.forward` received a `(B, N)` tensor produced by
    `_extract_batch_outputs`, which flattened *every variable at every level*
    into a single row-major vector and then called `rfft2` over
    `(batch, concatenated-everything)`.  At B=1 that was a 1-D FFT over a
    mixture of `z`, `q`, `t`, `u`, `v` across 13 levels and the surface
    variables — not a spatial spectrum.  `docs/papers/` §IV-B's attribution of
    Day-10 grittiness to this term is therefore **not supported by the pre-H2
    code** (`00_CONTEXT.md` R6).  This class corrects the defect.

    Design: `rfft2` over (lat, lon) only, per variable, per level
    =============================================================
    For each variable `v` and pressure level `ℓ` the field `x[b, ℓ, φ, λ]` is
    a 2-D spatial map.  `rfft2` is applied over the last two axes `(φ, λ)`,
    yielding a genuine spatial-frequency representation.

    Latitude non-periodicity: Hann window
    ======================================
    Longitude is periodic (the grid wraps at 0°/360°), so a plain DFT along
    that axis is exact.  Latitude is NOT periodic: the field values at the
    north and south poles are not constrained to be equal, so naïvely treating
    latitude as periodic introduces a spurious high-wavenumber "wrap-around"
    signal at the pole edges.

    Two remedies exist: (a) restrict to the tropical band only, or (b) apply a
    smooth tapering window in latitude.  We choose (b) — a per-row Hann window:

        w(φ) = 0.5 × (1 − cos(2π i / (H−1)))   for row i=0..H−1

    Rationale: tropical restriction discards all extratropical power, which
    matters for the mid-latitude Rossby-wave response that the MJO drives and
    which Aurora is expected to learn.  The Hann window smoothly tapers both
    poles to zero (removing the discontinuity) while preserving full-globe
    spatial information up to a 6 dB spectral-dynamic-range cost — the
    standard accepted trade-off in signal processing.  The window is
    normalised so that its mean squared value is 1, preserving the overall
    energy scale.

    Amplitude spectrum vs. complex coefficients
    ============================================
    The term is implemented as:

        L_s = Σ_v w_v · mean_{b,ℓ,k} | |FFT(x̂_{vℓ})|_k − |FFT(x_{vℓ})|_k |

    i.e. we penalise differences in **amplitude spectra**, not in complex
    coefficients.  The alternative `|FFT(x̂) − FFT(x)|` (complex formulation)
    penalises phase error, which the grid loss already does.  The docstring's
    own stated purpose — "match the texture and spatial variance rather than
    just the position" — argues for amplitude-only: a field spatially shifted
    by one grid cell has identical power spectrum but a very large complex
    coefficient difference; amplitude-only correctly returns zero (or near zero)
    for such a shift.  The `test_spectral_shift_distinguishes_amplitude_vs_complex`
    test quantifies this distinction.

    Normalisation before the transform
    ====================================
    Each variable is normalised using the same G3 1980–2015 Welford training
    statistics as `TropicalWeightedL1Loss` (denormalise-then-renormalise path).
    Without this, `z` (~3 800 m²/s²) and `msl` (~9 500 Pa) would dominate the
    spectrum term with the same 99:1 gradient imbalance documented in R1.

    The term ships **disabled by default** (`enabled: false, weight: 0.0` in
    `configs/unified.yaml`).  `02_SCIENTIFIC_CONTRACT.md` reserves it for a
    controlled ablation in a later campaign.
    """

    def __init__(
        self,
        lat_coords: torch.Tensor | list[float] | None = None,
        variable_weights: dict[str, float] | None = None,
        norm_stats: dict | None = None,
        norm_stats_file: str | Path | None = None,
    ):
        super().__init__()

        # ---- Latitude Hann window ----------------------------------------
        # Applied over the latitude (row) dimension to suppress the spurious
        # high-wavenumber signal from the north→south pole discontinuity.
        # Normalised so that mean(w²) = 1, preserving energy scale.
        if lat_coords is None:
            lat_coords = torch.linspace(89.5, -89.5, 180)
        lat_tensor = torch.as_tensor(lat_coords, dtype=torch.float32)
        H = lat_tensor.shape[0]
        i = torch.arange(H, dtype=torch.float32)
        hann = 0.5 * (1.0 - torch.cos(2.0 * torch.pi * i / max(H - 1, 1)))
        # Normalise: divide by RMS so energy is preserved on average
        hann_norm = hann / (hann.pow(2).mean().sqrt() + 1e-12)
        # Shape (1, 1, H, 1) for broadcasting over (B, L, H, W)
        self.register_buffer("hann_window", hann_norm.view(1, 1, H, 1))
        self.H = H

        # ---- Variable weights w_v (same as H1 TropicalWeightedL1Loss) ----
        self.variable_weights: dict[str, float] = dict(DEFAULT_VARIABLE_WEIGHTS)
        if variable_weights:
            self.variable_weights.update(variable_weights)

        # ---- Normalisation constants (G3 1980–2015 Welford statistics) ---
        stats_dict = TropicalWeightedL1Loss._load_norm_stats(norm_stats, norm_stats_file)
        self._surf_vars = ("2t", "10u", "10v", "msl", "ttr", "tcwv", "sst", "ps")
        self._atmos_vars = ("z", "q", "t", "u", "v")
        plevs_list = list(DEFAULT_AURORA_PLEVS)

        for v in self._surf_vars:
            mean_val, std_val = TropicalWeightedL1Loss._extract_surf_stats(
                None, v, stats_dict  # type: ignore[arg-type]
            )
            self.register_buffer(
                f"surf_std_{v}", torch.tensor(std_val, dtype=torch.float32)
            )

        for v in self._atmos_vars:
            _, std_vec = TropicalWeightedL1Loss._extract_atmos_stats(
                None, v, stats_dict, plevs_list  # type: ignore[arg-type]
            )
            self.register_buffer(f"atmos_std_{v}", std_vec)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise_surf(self, x: torch.Tensor, var_name: str) -> torch.Tensor:
        """Normalise a surface field (B, 1, H, W) or (B, H, W) by its std."""
        std = getattr(self, f"surf_std_{var_name}", None)
        if std is None or std.item() == 0.0:
            return x.float()
        return x.float() / std

    def _normalise_atmos(self, x: torch.Tensor, var_name: str) -> torch.Tensor:
        """Normalise an atmospheric field (B, 1, L, H, W) or (B, L, H, W) by its per-level std."""
        std = getattr(self, f"atmos_std_{var_name}", None)
        if std is None:
            return x.float()
        # std shape: (L,) — broadcast over (B, [1,] L, H, W)
        xf = x.float()
        if xf.ndim == 5:  # (B, T=1, L, H, W)
            return xf / std.view(1, 1, -1, 1, 1).clamp(min=1e-12)
        elif xf.ndim == 4:  # (B, L, H, W)
            return xf / std.view(1, -1, 1, 1).clamp(min=1e-12)
        return xf

    def _amplitude_spectrum_loss(
        self, pred_norm: torch.Tensor, tgt_norm: torch.Tensor
    ) -> torch.Tensor:
        """Compute amplitude-spectrum L1 loss on a (B, H, W) or (B, L, H, W) tensor.

        Applies the Hann latitude window, then rfft2 over the last two axes,
        then computes mean |‖FFT(x̂)‖ − ‖FFT(x)‖|.
        """
        p = pred_norm
        t = tgt_norm

        # Ensure (B, L, H, W) shape — surface vars have L=1
        if p.ndim == 3:  # (B, H, W)
            p = p.unsqueeze(1)
            t = t.unsqueeze(1)
        elif p.ndim == 5:  # (B, T=1, L, H, W) — squeeze time
            p = p[:, 0]
            t = t[:, 0]

        # Guard: latitude dimension must match the registered window
        if p.shape[-2] != self.H:
            # Smoke tests with small grids — return zero with grad path
            return torch.zeros((), device=p.device, dtype=p.dtype)

        # Apply Hann window along latitude (dim -2)
        win = self.hann_window  # (1, 1, H, 1)
        p_win = p * win
        t_win = t * win

        # rfft2 over (lat=last-2, lon=last-1) — the genuine spatial transform
        p_fft = torch.fft.rfft2(p_win, norm="ortho")  # (B, L, H, W//2+1) complex
        t_fft = torch.fft.rfft2(t_win, norm="ortho")

        # Amplitude (power-spectrum formulation): |·| is amplitude, not complex diff
        p_amp = torch.abs(p_fft)
        t_amp = torch.abs(t_fft)

        return torch.abs(p_amp - t_amp).mean()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward_per_var(
        self,
        pred_batch,
        target_dict: dict[str, torch.Tensor],
        device: torch.device | str,
    ) -> torch.Tensor:
        """Compute the weighted per-variable spectral loss over all variables in the batch.

        Args:
            pred_batch: Aurora ``Batch`` predicted for the next time step.
            target_dict: Mapping of variable name → target tensor.
            device: Device to move tensors to before computing the loss.

        Returns:
            Scalar loss (weighted sum over variables).
        """
        weighted_losses: list[torch.Tensor] = []

        for attr, var_type in (("surf_vars", "surf"), ("atmos_vars", "atmos")):
            if not hasattr(pred_batch, attr):
                continue
            for var_name, pred_t in getattr(pred_batch, attr).items():
                if var_name not in target_dict:
                    continue
                p = pred_t.to(device).float()
                tgt = target_dict[var_name].to(device).float()
                # Squeeze any trailing time dimension added by the trainer
                while p.ndim > tgt.ndim and p.shape[1] == 1:
                    p = p.squeeze(1)
                while tgt.ndim > p.ndim and (tgt.shape[0] == 1 or tgt.shape[1] == 1):
                    tgt = tgt.squeeze(0) if tgt.shape[0] == 1 else tgt.squeeze(1)

                # Normalise before the transform (same G3 constants as grid loss)
                if var_type == "surf" and hasattr(self, f"surf_std_{var_name}"):
                    p_norm = self._normalise_surf(p, var_name)
                    tgt_norm = self._normalise_surf(tgt, var_name)
                elif var_type == "atmos" and hasattr(self, f"atmos_std_{var_name}"):
                    p_norm = self._normalise_atmos(p, var_name)
                    tgt_norm = self._normalise_atmos(tgt, var_name)
                else:
                    p_norm = p
                    tgt_norm = tgt

                amp_loss = self._amplitude_spectrum_loss(p_norm, tgt_norm)
                w_v = float(self.variable_weights.get(var_name, 1.0))
                weighted_losses.append(w_v * amp_loss)

        if not weighted_losses:
            dev = next(iter(self.buffers()), None)
            zero_dev = dev.device if dev is not None else torch.device("cpu")
            return torch.zeros((), device=zero_dev)
        return sum(weighted_losses)  # type: ignore[return-value]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Single-field convenience forward for unit tests.

        Takes (B, H, W) or (B, L, H, W) normalised tensors and returns the
        amplitude-spectrum L1 loss.  The per-variable normalisation and
        weighting are applied in `forward_per_var` during training.
        """
        return self._amplitude_spectrum_loss(pred.float(), target.float())


class MoistureBudgetLoss(nn.Module):
    """Physics-Informed Loss: Column-Integrated Moisture Conservation Supervised by ERA5 E − P.

    Mathematical basis
    ==================
    The vertically-integrated moisture budget over an atmospheric column is:

        ∂⟨q⟩/∂t + ∇·⟨v q⟩ = E − P

    where:
        ⟨·⟩     = vertical column integral ∫(·) dp/g over pressure levels
        q       = specific humidity (kg/kg)
        v       = (u, v) horizontal wind (m/s)
        E       = surface evaporation (kg/m²/s)
        P       = precipitation (kg/m²/s)

    Scientific defect R2 resolution (00_CONTEXT.md R2, 02_SCIENTIFIC_CONTRACT.md §3):
    --------------------------------------------------------------------------------
    The pre-H3 loss penalized mean(|R|) toward zero, where R = ∂⟨q⟩/∂t + ∇·⟨vq⟩.
    In the tropics, active MJO convection produces E − P ≈ -15 to -35 mm/day.
    Driving R toward zero penalised the active convective envelope the model is fine-tuned
    to predict and rewarded oversmoothing (smoothed fields have small gradients and small R).
    H3 supervises the implied residual R against ERA5 observed (E − P)_ERA5:
        L_moisture = mean(|R − (E − P)_ERA5|)  over the tropical band (±20°).

    ERA5 downward-positive sign convention & unit conversions:
    ----------------------------------------------------------
    - Vertical fluxes in ERA5 are positive DOWNWARD (atmosphere -> surface).
    - Evaporation transfers latent heat UPWARD from surface into atmosphere, so
      ERA5 mslhf (mean surface latent heat flux, W m⁻²) is NEGATIVE for evaporation
      (tropical ocean mean ~ -113 W m⁻²).
      Therefore, upward evaporation water mass flux is:
          E (kg m⁻² s⁻¹) = -mslhf / L_v   where L_v = 2.501 × 10⁶ J kg⁻¹
          E (mm/day) = (-mslhf / L_v) × 86400.0
    - Precipitation is total 6-hour accumulation tp6h in metres:
          P (kg m⁻² s⁻¹) = (tp6h / Δt) × 1000.0   where Δt = 21600 s (6 h)
          P (mm/day) = (tp6h / Δt) × 1000.0 × 86400.0 = tp6h × 4000.0
    - Target:
          (E − P)_ERA5 = E_mm_day − P_mm_day

    Surface-pressure masking (Task H3 Step 4):
    ------------------------------------------
    The column integral is masked at surface pressure ps: pressure levels with p > ps
    sit below the topography and are excluded from the integral (mask = 0), not extrapolated.
    This prevents artificial sub-surface moisture inflation over land and mountain barriers
    (such as the Maritime Continent).
    """

    def __init__(
        self,
        pressure_levels: list[float] | tuple[float, ...] | torch.Tensor = DEFAULT_AURORA_PLEVS,
        latitudes: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        longitudes: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        dt_seconds: float = 21600.0,
        tropics_bbox: tuple[float, float] = (-20.0, 20.0),
        residual_clamp_mm_day: float = 1.0e4,
        latent_heat_vaporization: float = 2.501e6,
        plevs: list[float] | tuple[float, ...] | torch.Tensor | None = None,
    ):
        super().__init__()
        if plevs is not None:
            pressure_levels = plevs
        self.dt = float(dt_seconds)
        self.g = 9.80665
        self.R = 6371000.0
        self.L_v = float(latent_heat_vaporization)
        self.residual_clamp = float(residual_clamp_mm_day)

        if latitudes is None:
            latitudes = torch.linspace(89.5, -89.5, 180).tolist()
        if longitudes is None:
            longitudes = torch.linspace(0.5, 359.5, 360).tolist()

        # ---- Grid tensors ------------------------------------------------
        plevs_t = torch.as_tensor(pressure_levels, dtype=torch.float32)
        if plevs_t.max() <= 1100.0:  # If in hPa, convert to Pa
            plevs_t = plevs_t * 100.0
        self.register_buffer("plevs", plevs_t)

        lats = torch.as_tensor(latitudes, dtype=torch.float32)
        lons = torch.as_tensor(longitudes, dtype=torch.float32)
        self.register_buffer("lats", lats)
        self.register_buffer("lons", lons)

        # Layer-thickness weights (central differences interior, one-sided edges)
        dp = torch.zeros_like(plevs_t)
        dp[0] = plevs_t[1] - plevs_t[0]
        dp[1:-1] = (plevs_t[2:] - plevs_t[:-2]) / 2.0
        dp[-1] = plevs_t[-1] - plevs_t[-2]
        self.register_buffer("dp", dp)  # 1D (L,)

        # ---- Spherical geometry ------------------------------------------
        dlat_rad = torch.deg2rad(torch.abs(lats[0] - lats[1]))
        dlon_rad = torch.deg2rad(torch.abs(lons[1] - lons[0]))

        self.dy = self.R * dlat_rad

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

    def vertical_integral(
        self, x: torch.Tensor, ps: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Column integral <X> = ∫ X dp/g over pressure-level dim (dim=-3), masked at ps.

        Args:
            x: Tensor of shape (..., L, H, W).
            ps: Optional surface pressure tensor in Pa. Levels with p > ps are excluded.
        """
        dp_shape = [1] * (x.ndim - 3) + [-1, 1, 1]
        dp_view = self.dp.view(*dp_shape)

        if ps is None:
            return torch.sum(x * dp_view / self.g, dim=-3)

        # Prepare ps to broadcast over (..., L, H, W)
        ps_view = ps.float()
        while ps_view.ndim < x.ndim:
            if ps_view.ndim == 2:  # (H, W) -> (... 1, H, W)
                ps_view = ps_view.view(*([1] * (x.ndim - 2)), ps_view.shape[0], ps_view.shape[1])
            elif ps_view.ndim == 3:  # (B, H, W) -> (B, 1, H, W)
                ps_view = ps_view.unsqueeze(-3)
            elif ps_view.ndim == 4 and x.ndim == 5:  # (B, 1, H, W) -> (B, 1, 1, H, W)
                ps_view = ps_view.unsqueeze(-3)
            else:
                ps_view = ps_view.unsqueeze(0)

        plevs_view = self.plevs.view(*dp_shape)
        mask = (plevs_view <= ps_view).float()
        return torch.sum(x * mask * dp_view / self.g, dim=-3)

    def spherical_divergence(self, u_flux: torch.Tensor, v_flux: torch.Tensor) -> torch.Tensor:
        """div = (1 / R cos phi ) [∂u/∂λ + ∂(v cos phi)/∂φ] on (B,T,H,W) fields."""
        u_padded = F.pad(u_flux, pad=(1, 1, 0, 0), mode="circular")
        du_dlon = (u_padded[..., 2:] - u_padded[..., :-2]) / (2.0 * self.dx)

        v_cos_lat = v_flux * self.cos_lat
        v_padded = F.pad(v_cos_lat, pad=(0, 0, 1, 1), mode="replicate")
        dv_dlat = (v_padded[..., :-2, :] - v_padded[..., 2:, :]) / (2.0 * self.dy)

        return du_dlon + dv_dlat / self.cos_lat

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        in_batch,
        pred_batch,
        target_dict: dict[str, torch.Tensor] | None = None,
        emp_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Tropical moisture-budget residual loss (scalar, mm/day, ≥ 0).

        Args:
            in_batch: Aurora Batch fed into the current forward pass.
            pred_batch: Aurora Batch predicted for the next time step.
            target_dict: Optional dict containing loss targets (mslhf, tp6h, msl).
            emp_target: Optional direct E−P target tensor in mm/day.
        """
        device = self.plevs.device

        # Guard: need q, u, v in both batches
        required = {"q", "u", "v"}
        for batch in (in_batch, pred_batch):
            if not hasattr(batch, "atmos_vars") or not required.issubset(
                batch.atmos_vars.keys()
            ):
                return torch.zeros((), device=device, requires_grad=True)

        # Slice latest history step for true 6-h difference: (B, 1, L, H, W)
        q_curr = in_batch.atmos_vars["q"][:, -1:, ...]
        q_next = pred_batch.atmos_vars["q"][:, -1:, ...]
        u_next = pred_batch.atmos_vars["u"][:, -1:, ...]
        v_next = pred_batch.atmos_vars["v"][:, -1:, ...]

        # Grid-mismatch guard (smoke tests / low-res debugging)
        if (
            q_next.shape[-2] != self.lats.shape[0]
            or q_next.shape[-1] != self.lons.shape[0]
        ):
            return torch.zeros((), device=device, requires_grad=True)

        # Retrieve surface pressure (msl) for topographic masking if available
        ps_curr = None
        ps_next = None
        if hasattr(in_batch, "surf_vars") and "msl" in in_batch.surf_vars:
            ps_curr = in_batch.surf_vars["msl"][:, -1:, ...]
        if hasattr(pred_batch, "surf_vars") and "msl" in pred_batch.surf_vars:
            ps_next = pred_batch.surf_vars["msl"][:, -1:, ...]
        elif target_dict is not None and "msl" in target_dict:
            ps_next = target_dict["msl"].to(device).float()
            while ps_next.ndim < 4:
                ps_next = ps_next.unsqueeze(0) if ps_next.ndim == 2 else ps_next.unsqueeze(1)
            if ps_curr is None:
                ps_curr = ps_next

        # Float32 outside autocast (residual is small difference of small numbers)
        with torch.autocast(device_type=q_next.device.type, enabled=False):
            q_curr = torch.nan_to_num(q_curr.float())
            q_next_f = torch.nan_to_num(q_next.float())
            u_next_f = torch.nan_to_num(u_next.float())
            v_next_f = torch.nan_to_num(v_next.float())

            # Column integrals masked at surface pressure ps
            int_q_curr = self.vertical_integral(q_curr, ps=ps_curr)
            int_q_next = self.vertical_integral(q_next_f, ps=ps_next)
            int_uq = self.vertical_integral(u_next_f * q_next_f, ps=ps_next)
            int_vq = self.vertical_integral(v_next_f * q_next_f, ps=ps_next)

            # Moisture tendency + flux divergence (kg/m²/s)
            dq_dt = (int_q_next - int_q_curr) / self.dt
            div_flux = self.spherical_divergence(int_uq, int_vq)
            residual_si = dq_dt + div_flux  # implied E−P in kg/m²/s

            # Convert to mm/day (× 86400) and clamp against transient blow-ups
            residual_mm_day = residual_si * 86400.0
            residual_mm_day = torch.clamp(
                residual_mm_day, -self.residual_clamp, self.residual_clamp
            )

            # Restrict residual to tropical band: (B, 1, N_trop_lat, W)
            residual_trop = residual_mm_day[:, :, self.trop_mask, :]

            # Supervise against ERA5 observed (E - P):
            # ERA5 downward-positive convention: E = -mslhf / L_v, P = tp6h * 4000.0 (mm/day)
            if emp_target is not None:
                tgt = emp_target.to(device).float()
                while tgt.ndim < 4:
                    tgt = tgt.unsqueeze(0) if tgt.ndim == 2 else tgt.unsqueeze(1)
                target_trop = tgt[:, :, self.trop_mask, :]
            elif target_dict is not None and "mslhf" in target_dict and "tp6h" in target_dict:
                mslhf = target_dict["mslhf"].to(device).float()
                tp6h = target_dict["tp6h"].to(device).float()
                while mslhf.ndim < 4:
                    mslhf = mslhf.unsqueeze(0) if mslhf.ndim == 2 else mslhf.unsqueeze(1)
                while tp6h.ndim < 4:
                    tp6h = tp6h.unsqueeze(0) if tp6h.ndim == 2 else tp6h.unsqueeze(1)
                e_mm_day = (-mslhf / self.L_v) * 86400.0
                p_mm_day = tp6h * 4000.0
                emp_calc = e_mm_day - p_mm_day
                target_trop = emp_calc[:, :, self.trop_mask, :]
            else:
                # Fallback to zero target if targets are not provided
                target_trop = torch.zeros_like(residual_trop)

            diff = torch.abs(residual_trop - target_trop)
            loss = diff.mean()

            # Final non-finite guard
            if not torch.isfinite(loss):
                return torch.zeros((), device=device, requires_grad=True)

        return loss
