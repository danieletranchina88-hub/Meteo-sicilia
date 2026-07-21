"""Physical diagnostics and transparent evidence scores for front analysis.

This module contains no GRIB or tracking code.  It turns gridded ICON fields
into kinematic diagnostics, then maps independent pieces of evidence to
bounded scores.  The resulting ``evidenceScore`` is deliberately *not* a
probability: it measures internal physical consistency of one deterministic
forecast.  Keeping this distinction explicit avoids pretending that a
single ICON run can quantify forecast probability.

The frontogenesis equation is the horizontal, adiabatic Petterssen/Keyser
form for the material change in |grad(theta_w)|.  Positive values mean
frontogenesis and negative values frontolysis.
"""

from __future__ import annotations

import numpy as np

import front_locator as fl


def _finite(value, default=np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def smoothstep(value, weak: float, strong: float) -> float:
    """Linear fuzzy membership in [0, 1] with a safe finite fallback."""
    value = _finite(value)
    if not np.isfinite(value):
        return 0.0
    if strong == weak:
        return float(value >= strong)
    fraction = (value - weak) / (strong - weak)
    return float(np.clip(fraction, 0.0, 1.0))


def kinematic_fields(
    theta_w: np.ndarray,
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    metrics: dict,
    *,
    smoothing_km: float = 45.0,
) -> dict[str, np.ndarray]:
    """Return metric-aware wind and frontogenesis diagnostics.

    Output units:
      * vorticity / convergence: 1e-5 s-1
      * frontogenesis: K (100 km)-1 (3 h)-1
      * thermal advection: K (3 h)-1
    """
    theta = fl.smooth_km(theta_w, smoothing_km, metrics)
    u = fl.smooth_km(u_wind, smoothing_km, metrics)
    v = fl.smooth_km(v_wind, smoothing_km, metrics)

    theta_x, theta_y = fl.gradient(theta, metrics)  # K/km
    u_x_km, u_y_km = fl.gradient(u, metrics)        # (m/s)/km
    v_x_km, v_y_km = fl.gradient(v, metrics)
    # Velocity derivatives must be per metre to obtain s-1.
    u_x, u_y = u_x_km / 1000.0, u_y_km / 1000.0
    v_x, v_y = v_x_km / 1000.0, v_y_km / 1000.0

    magnitude = np.hypot(theta_x, theta_y)
    safe = np.maximum(magnitude * magnitude, 1.0e-12)
    divergence = u_x + v_y
    stretching = u_x - v_y
    shearing = v_x + u_y
    cos_two_beta = (theta_x * theta_x - theta_y * theta_y) / safe
    sin_two_beta = 2.0 * theta_x * theta_y / safe

    # D|grad(theta)|/Dt from horizontal deformation and divergence.
    frontogenesis_si = -0.5 * magnitude * (
        divergence + stretching * cos_two_beta + shearing * sin_two_beta
    )
    frontogenesis = frontogenesis_si * 100.0 * 10_800.0
    vorticity = (v_x - u_y) * 1.0e5
    convergence = -divergence * 1.0e5
    thermal_advection = -(
        u * theta_x + v * theta_y
    ) / 1000.0 * 10_800.0

    invalid = ~(
        np.isfinite(theta) & np.isfinite(u) & np.isfinite(v)
    )
    return {
        "thetaWSmooth": theta,
        "thetaWGradientEast": theta_x,
        "thetaWGradientNorth": theta_y,
        "vorticity1e5": np.where(invalid, np.nan, vorticity),
        "convergence1e5": np.where(invalid, np.nan, convergence),
        "frontogenesis": np.where(invalid, np.nan, frontogenesis),
        "thermalAdvection3h": np.where(invalid, np.nan, thermal_advection),
    }


def candidate_evidence(metrics: dict) -> dict:
    """Score independent thermodynamic, dynamic and structural evidence."""
    locator = smoothstep(metrics.get("locatorConfidence"), 0.15, 0.85)
    theta_w_contrast = smoothstep(metrics.get("deltaThetaW"), 1.2, 4.5)
    dry_contrast = smoothstep(metrics.get("deltaTemperature"), 0.45, 3.0)
    virtual_contrast = smoothstep(metrics.get("deltaThetaV"), 0.20, 2.2)
    dry_gradient = smoothstep(metrics.get("dryThermalGradient"), 0.75, 3.0)
    alignment = smoothstep(metrics.get("thermalAlignment"), 0.05, 0.75)
    thermal = float(np.average(
        [locator, theta_w_contrast, dry_contrast, virtual_contrast,
         dry_gradient, alignment],
        weights=[1.1, 1.2, 1.3, 1.2, 1.0, 0.8],
    ))

    wind_shift = smoothstep(metrics.get("windShiftMs"), 1.2, 7.0)
    convergence = smoothstep(metrics.get("convergenceMs"), 0.05, 2.0)
    vorticity = smoothstep(metrics.get("vorticity1e5"), -0.5, 4.0)
    frontogenesis = smoothstep(metrics.get("frontogenesis"), -0.6, 2.8)
    dynamic_members = sorted(
        [wind_shift, convergence, vorticity, frontogenesis], reverse=True
    )
    # A mature or frontolysing front need not satisfy every dynamic signal.
    # Requiring several independent members but not all avoids both a single-
    # signal false positive and a frontogenesis-only false negative.
    dynamic = float(np.mean(dynamic_members[:3]))

    pressure_value = _finite(metrics.get("pressureTroughHpa"))
    pressure = (
        smoothstep(pressure_value, -0.15, 1.8)
        if np.isfinite(pressure_value) else 0.45
    )
    isallobaric = _finite(metrics.get("isallobaricSupportHpa3h"))
    if np.isfinite(isallobaric):
        pressure = 0.72 * pressure + 0.28 * smoothstep(isallobaric, -0.3, 1.2)

    lower_support = _finite(metrics.get("lowerLevelSupport"))
    lower_contrast = _finite(metrics.get("deltaThetaW925"))
    if np.isfinite(lower_support) and np.isfinite(lower_contrast):
        vertical = 0.58 * smoothstep(lower_support, 0.35, 0.80) + 0.42 * smoothstep(
            lower_contrast, 0.7, 3.2
        )
    else:
        vertical = 0.45

    omega = _finite(metrics.get("omega700PaS"))
    activity = smoothstep(-omega, -0.03, 0.20) if np.isfinite(omega) else 0.50

    synoptic = smoothstep(metrics.get("synopticSupport"), 0.55, 0.95)
    length = smoothstep(metrics.get("lengthKm"), 220.0, 750.0)
    sinuosity = _finite(metrics.get("sinuosity"), 1.0)
    shape = smoothstep(2.30 - sinuosity, 0.0, 1.15)
    terrain = 1.0 - smoothstep(metrics.get("terrainFraction"), 0.20, 0.70)
    structural = float(np.average(
        [synoptic, length, shape, terrain], weights=[1.4, 0.8, 0.8, 0.6]
    ))

    evidence = float(np.clip(
        0.38 * thermal
        + 0.24 * dynamic
        + 0.10 * pressure
        + 0.10 * vertical
        + 0.04 * activity
        + 0.14 * structural,
        0.0,
        1.0,
    ))
    components = {
        "thermal": round(thermal, 3),
        "dynamic": round(dynamic, 3),
        "pressure": round(float(pressure), 3),
        "vertical": round(float(vertical), 3),
        "activity": round(float(activity), 3),
        "structural": round(structural, 3),
    }
    return {
        "candidateEvidence": round(evidence, 3),
        "evidenceComponents": components,
    }


def candidate_is_plausible(metrics: dict, evidence: dict | None = None) -> bool:
    """Conservative core gates; optional diagnostics stay soft evidence."""
    evidence = evidence or candidate_evidence(metrics)
    components = evidence["evidenceComponents"]
    required = (
        _finite(metrics.get("deltaTemperature"), -999.0) >= 0.45
        and _finite(metrics.get("deltaThetaV"), -999.0) >= 0.20
        and _finite(metrics.get("dryThermalGradient"), -999.0) >= 0.75
        and _finite(metrics.get("thermalAlignment"), -999.0) >= -0.05
        and _finite(metrics.get("terrainFraction"), 1.0) <= 0.70
        and _finite(metrics.get("locatorConfidence"), 0.0) >= 0.15
    )
    return bool(
        required
        and components["thermal"] >= 0.34
        and components["dynamic"] >= 0.18
        and components["structural"] >= 0.32
        and evidence["candidateEvidence"] >= 0.36
    )

