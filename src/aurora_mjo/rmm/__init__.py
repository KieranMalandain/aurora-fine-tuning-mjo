"""RMM computation and evaluation package for aurora_mjo."""

from __future__ import annotations

from aurora_mjo.rmm.compute import (
    LAT_N,
    LAT_S,
    N_LON,
    build_combined_vector,
    compute_daily_clim,
    compute_eofs,
    compute_wh_phase,
    compute_zonal_std,
    fit_seasonal_harmonics,
    project_onto_eofs,
    remove_120d_mean,
    remove_clim,
    to_daily_mean,
    tropical_mean,
)
from aurora_mjo.rmm.evaluate import (
    amplitude_error,
    bivariate_acc,
    extract_rmm_from_fields,
    extract_rmm_from_mjo_head,
    phase_error_deg,
    project_fields_to_rmm,
    rmse_pair,
)

__all__ = [
    "LAT_N",
    "LAT_S",
    "N_LON",
    "amplitude_error",
    "bivariate_acc",
    "build_combined_vector",
    "compute_daily_clim",
    "compute_eofs",
    "compute_wh_phase",
    "compute_zonal_std",
    "extract_rmm_from_fields",
    "extract_rmm_from_mjo_head",
    "fit_seasonal_harmonics",
    "phase_error_deg",
    "project_fields_to_rmm",
    "project_onto_eofs",
    "remove_120d_mean",
    "remove_clim",
    "rmse_pair",
    "to_daily_mean",
    "tropical_mean",
]
