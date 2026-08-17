"""Continuous physical-support field for objective front analysis (Fase B).

Biard & Kunkel (2019) let a CNN emit a per-pixel probability field and then
extract front lines from it. This module keeps the *idea* (a continuous
support field that separates the **existence** of a frontal structure from
its **type**) but builds the field **analytically from physics**, not from a
trained network: every component is a normalised membership of a diagnostic
the pipeline already trusts (theta_w gradient, ABZ, TFP, dry-theta gradient,
kinematics, vertical coherence, synoptic-scale gradient), combined with
explicit penalties (terrain, domain edge, missing data, moisture-only
boundary, mesoscale-only structure).

The output ``any_front_support`` is a physical-support heuristic in [0, 1],
NOT a calibrated probability. It answers "how strongly do the fields support
*some* front here", independent of cold/warm/stationary. Non-finite input is
handled explicitly and never silently clipped: invalid cells are marked in
``valid`` and contribute 0 support.

Nothing here publishes a line: the field is a diagnostic in Fase B and will
drive least-cost line extraction in Fase C.
"""

from __future__ import annotations

import numpy as np

import front_locator as fl
import front_physics as fp

EARTH_KM_PER_DEG = 111.32


def smoothstep_field(field: np.ndarray, weak: float, strong: float) -> np.ndarray:
    """Array fuzzy membership in [0, 1]; NaN -> 0. weak may exceed strong."""
    values = np.asarray(field, dtype=float)
    out = np.zeros_like(values)
    finite = np.isfinite(values)
    if strong == weak:
        out[finite] = (values[finite] >= strong).astype(float)
        return out
    frac = (values - weak) / (strong - weak)
    out[finite] = np.clip(frac[finite], 0.0, 1.0)
    return out


def _grad_mag_100(field: np.ndarray, metrics: dict, smooth_km: float) -> np.ndarray:
    """|grad(field)| in units/100 km after a physical pre-smoothing."""
    smoothed = fl.smooth_km(field, smooth_km, metrics)
    east, north = fl.gradient(smoothed, metrics)
    return np.hypot(east, north) * 100.0


def _edge_distance_km(longitudes, latitudes) -> np.ndarray:
    """Distance (km) of each cell to the nearest domain edge."""
    lon = np.asarray(longitudes, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    mean_lat = np.deg2rad(float(np.mean(lat)))
    dx_w = (lon - lon[0]) * EARTH_KM_PER_DEG * np.cos(mean_lat)
    dx_e = (lon[-1] - lon) * EARTH_KM_PER_DEG * np.cos(mean_lat)
    dy_s = (lat - lat[0]) * EARTH_KM_PER_DEG
    dy_n = (lat[-1] - lat) * EARTH_KM_PER_DEG
    dx = np.minimum(dx_w, dx_e)[None, :]
    dy = np.minimum(dy_s, dy_n)[:, None]
    return np.minimum(np.broadcast_to(dx, (lat.size, lon.size)),
                      np.broadcast_to(dy, (lat.size, lon.size)))


# Default membership intervals (physical units). ABZ/TFP come from the
# climatology when available; the rest are documented physical ranges, all
# configurable through ``config``.
_DEFAULTS = {
    "abz_weak": 0.90, "abz_full": 1.70,             # K/100 km (refined)
    "tfp_weak": -2.5e-5, "tfp_full": -9.0e-5,        # K/km^2 (negative warm edge)
    "dry_grad_weak": 0.7, "dry_grad_full": 3.0,      # K/100 km
    "moist_grad_weak": 1.5, "moist_grad_full": 6.0,  # K/100 km (theta_e)
    "convergence_weak": 0.05, "convergence_full": 2.0,   # 1e-5 s-1
    "vorticity_weak": -0.5, "vorticity_full": 4.0,       # 1e-5 s-1
    "frontogenesis_weak": -0.6, "frontogenesis_full": 2.8,
    "synoptic_grad_weak": 0.6, "synoptic_grad_full": 1.6,  # K/100 km (100 km smooth)
    "vertical_grad_weak": 0.6, "vertical_grad_full": 2.5,  # K/100 km (925 hPa)
    "pressure_lap_weak": 0.0, "pressure_lap_full": 8.0e-4,  # hPa/km^2 trough proxy
    "terrain_soft_m": 900.0, "terrain_hard_m": 1600.0,
    "edge_soft_km": 70.0, "edge_hard_km": 0.0,
}

# Positive-evidence weights for the combined field. Thermodynamic existence
# dominates; dynamics/pressure/vertical/synoptic reinforce.
_WEIGHTS = {
    "thermal": 1.3, "abz": 1.2, "tfp": 1.0, "dry_thermal": 1.1,
    "dynamic": 0.9, "pressure": 0.6, "vertical": 0.8, "synoptic": 1.0,
}


def physical_support_field(
    theta_w: np.ndarray,
    theta: np.ndarray,
    theta_e: np.ndarray,
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    *,
    terrain: np.ndarray | None = None,
    pressure_hpa: np.ndarray | None = None,
    theta_w_925: np.ndarray | None = None,
    theta_w_700: np.ndarray | None = None,
    synoptic_sigma_km: float = 100.0,
    refine_sigma_km: float = 45.0,
    derivative_sigma_km: float = 15.0,
    config: dict | None = None,
) -> dict:
    """Return the per-cell support components, penalties and any_front_support.

    All fields share the (lat, lon) grid of ``theta_w``. See module docstring
    for the meaning of the output; every value is in [0, 1].
    """
    cfg = dict(_DEFAULTS)
    if config:
        cfg.update(config)
    metrics = fl.grid_metrics(longitudes, latitudes)
    theta_w = np.asarray(theta_w, dtype=float)
    valid = np.isfinite(theta_w)

    # --- thermodynamic existence -----------------------------------------
    refined = fl.smooth_km(theta_w, refine_sigma_km, metrics)
    grad_e, grad_n = fl.gradient(refined, metrics)
    grad_mag = fl.smooth_km(np.hypot(grad_e, grad_n), derivative_sigma_km, metrics)
    grad_mag_100 = grad_mag * 100.0
    gm_e, gm_n = fl.gradient(grad_mag, metrics)
    safe = np.maximum(grad_mag, 1.0e-9)
    tfp = (gm_e * grad_e + gm_n * grad_n) / safe
    local_grid_km = np.sqrt(metrics["dx_km_col"] * metrics["dy_km"])
    abz = (grad_mag + (local_grid_km / np.sqrt(2.0)) * np.hypot(gm_e, gm_n)) * 100.0

    thermal = smoothstep_field(grad_mag_100, cfg["abz_weak"], cfg["abz_full"])
    abz_support = smoothstep_field(abz, cfg["abz_weak"], cfg["abz_full"])
    tfp_support = smoothstep_field(tfp, cfg["tfp_weak"], cfg["tfp_full"])

    dry_grad_100 = _grad_mag_100(theta, metrics, refine_sigma_km)
    dry_thermal = smoothstep_field(dry_grad_100, cfg["dry_grad_weak"], cfg["dry_grad_full"])
    moist_grad_100 = _grad_mag_100(theta_e, metrics, refine_sigma_km)
    moisture = smoothstep_field(moist_grad_100, cfg["moist_grad_weak"], cfg["moist_grad_full"])

    # --- dynamics (reuse the validated kinematic diagnostics) ------------
    kin = fp.kinematic_fields(theta_w, u_wind, v_wind, metrics, smoothing_km=refine_sigma_km)
    convergence = smoothstep_field(kin["convergence1e5"], cfg["convergence_weak"], cfg["convergence_full"])
    vorticity = smoothstep_field(kin["vorticity1e5"], cfg["vorticity_weak"], cfg["vorticity_full"])
    frontogenesis = smoothstep_field(kin["frontogenesis"], cfg["frontogenesis_weak"], cfg["frontogenesis_full"])
    dynamic = np.clip(0.5 * convergence + 0.25 * vorticity + 0.25 * frontogenesis, 0.0, 1.0)

    # --- synoptic-scale existence ----------------------------------------
    synoptic_grad_100 = _grad_mag_100(theta_w, metrics, synoptic_sigma_km)
    synoptic = smoothstep_field(synoptic_grad_100, cfg["synoptic_grad_weak"], cfg["synoptic_grad_full"])

    # --- vertical coherence (optional 925/700 hPa) -----------------------
    vertical_levels = []
    for level_field in (theta_w_925, theta_w_700):
        if level_field is not None:
            level_grad_100 = _grad_mag_100(
                np.asarray(level_field, float), metrics, refine_sigma_km
            )
            vertical_levels.append(smoothstep_field(
                level_grad_100, cfg["vertical_grad_weak"], cfg["vertical_grad_full"]
            ))
    vertical = (
        np.mean(vertical_levels, axis=0)
        if vertical_levels else np.full_like(theta_w, 0.5)
    )

    # --- pressure trough (optional PMSL): laplacian>0 is a relative trough
    if pressure_hpa is not None:
        pressure_lap = fl.laplacian(fl.smooth_km(np.asarray(pressure_hpa, float), 80.0, metrics), metrics)
        pressure = smoothstep_field(pressure_lap, cfg["pressure_lap_weak"], cfg["pressure_lap_full"])
    else:
        pressure = np.full_like(theta_w, 0.45)

    components = {
        "thermal": thermal, "abz": abz_support, "tfp": tfp_support,
        "dry_thermal": dry_thermal, "moisture": moisture, "dynamic": dynamic,
        "pressure": pressure, "vertical": vertical, "synoptic": synoptic,
    }

    # --- explicit penalties ----------------------------------------------
    if terrain is not None:
        terrain_penalty = smoothstep_field(np.asarray(terrain, float), cfg["terrain_soft_m"], cfg["terrain_hard_m"])
    else:
        terrain_penalty = np.zeros_like(theta_w)
    edge_km = _edge_distance_km(longitudes, latitudes)
    edge_penalty = 1.0 - smoothstep_field(edge_km, cfg["edge_hard_km"], cfg["edge_soft_km"])
    missing_data_penalty = (~valid).astype(float)
    # moisture-only boundary: strong theta_e gradient with weak dry & theta_w
    moisture_boundary_penalty = np.clip(
        moisture * (1.0 - dry_thermal) * (1.0 - thermal), 0.0, 1.0
    )
    # mesoscale-only: refined gradient present but no synoptic-scale support
    local_scale_penalty = np.clip(thermal * (1.0 - synoptic), 0.0, 1.0)

    penalties = {
        "terrain": terrain_penalty, "edge": edge_penalty,
        "missing_data": missing_data_penalty,
        "moisture_boundary": moisture_boundary_penalty,
        "local_scale": local_scale_penalty,
    }

    # --- combine ----------------------------------------------------------
    weight_sum = sum(_WEIGHTS.values())
    positive = sum(_WEIGHTS[name] * components[name] for name in _WEIGHTS) / weight_sum
    penalty = np.clip(
        0.55 * terrain_penalty + 0.9 * edge_penalty
        + 0.6 * moisture_boundary_penalty + 0.45 * local_scale_penalty,
        0.0, 1.0,
    )
    any_front_support = np.where(valid, np.clip(positive * (1.0 - penalty), 0.0, 1.0), 0.0)

    return {
        "components": components,
        "penalties": penalties,
        "any_front_support": any_front_support,
        "tfp": tfp,
        "abz": abz,
        "grad_mag_100": grad_mag_100,
        "valid": valid,
    }
