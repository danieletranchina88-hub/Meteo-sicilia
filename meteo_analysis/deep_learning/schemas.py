"""Versioned tensor contracts shared by training and inference."""

from __future__ import annotations

import hashlib
import json

SCHEMA_VERSION = 1

# Class zero is intentionally background.  Stationary fronts are retained:
# omitting them would force a real stationary boundary into a wrong class.
FRONT_CLASSES = (
    "background",
    "cold",
    "warm",
    "occluded",
    "stationary",
)

# These are physically derived, unit-explicit fields already produced by the
# common ERA5/ICON feature code.  Latitude and longitude are excluded to make
# climatological location shortcuts harder to learn.
FRONT_FEATURES = (
    "theta_850_k",
    "theta_e_850_k",
    "grad_theta_850_20_k_per_100km",
    "grad_theta_850_50_k_per_100km",
    "grad_theta_850_100_k_per_100km",
    "thermal_advection_850_20_k_per_3h",
    "thermal_advection_850_50_k_per_3h",
    "thermal_advection_850_100_k_per_3h",
    "frontogenesis_850_20_k_per_100km_3h",
    "frontogenesis_850_50_k_per_100km_3h",
    "frontogenesis_850_100_k_per_100km_3h",
    "dewpoint_gradient_850_k_per_100km",
    "dewpoint_gradient_700_k_per_100km",
    "rh_850_pct",
    "rh_700_pct",
    "mixing_ratio_850_gkg",
    "mixing_ratio_700_gkg",
    "convergence_10m_1e5_s",
    "divergence_850_1e5_s",
    "vorticity_850_1e5_s",
    "pmsl_tendency_3h_hpa",
    "vorticity_advection_500_1e10_s2",
    "geopotential_height_500_m",
    "grad_theta_850_50_mean_3h",
    "grad_theta_850_50_std_3h",
    "theta_w_925_k",
    "theta_w_850_k",
    "theta_w_700_k",
    "theta_e_925_k",
    "theta_e_700_k",
    "u925_m_s",
    "v925_m_s",
    "u850_m_s",
    "v850_m_s",
    "u700_m_s",
    "v700_m_s",
    "pmsl_hpa",
    "u10_m_s",
    "v10_m_s",
    "omega700_pa_s",
)

STATIC_DOWNSCALING_FEATURES = (
    "elevation_m",
    "slope_east_m_per_m",
    "slope_north_m_per_m",
    "land_fraction",
    "distance_to_coast_km",
)

DOWNSCALING_OUTPUTS = (
    "temperature_2m_k",
    "precipitation_rate_mm_h",
    "wind_u10_m_s",
    "wind_v10_m_s",
)


def schema_hash(*parts) -> str:
    """Return a stable identifier for ordered channels and model settings."""
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
