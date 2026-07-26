"""Surface fog and visibility diagnostics with explicit data limitations."""

from __future__ import annotations

import numpy as np


def _array(values):
    return np.asarray(values, dtype=float)


def _like(template, values, description, units):
    if hasattr(template, "dims") and hasattr(template, "copy"):
        result = template.copy(data=values)
        result.attrs.update(description=description, units=units)
        return result
    return values


def classify_fog_type(
    rh_sfc,
    wind_10m,
    clct,
    warm_advection=None,
    surface_minus_air_temperature=None,
):
    """Classify fog type: 0 none, 1 radiation, 2 advection.

    Advection fog is only labelled when real warm-advection and surface/air
    thermal-contrast fields are supplied.  Wind speed alone is not sufficient
    evidence and is no longer used to invent an advection-fog signal.
    """
    rh, wind, cloud = np.broadcast_arrays(
        _array(rh_sfc), _array(wind_10m), _array(clct)
    )
    valid = np.isfinite(rh) & np.isfinite(wind) & np.isfinite(cloud)
    result = np.full(rh.shape, np.nan, dtype=float)
    result[valid] = 0.0

    # Very high RH can itself create a low cloud fraction, therefore the
    # clear-sky condition is relaxed only near saturation.
    radiation = (
        valid
        & (rh >= 95.0)
        & (wind < 3.0)
        & ((cloud <= 35.0) | (rh >= 99.0))
    )
    result[radiation] = 1.0

    if warm_advection is not None and surface_minus_air_temperature is not None:
        advection, thermal_contrast = np.broadcast_arrays(
            _array(warm_advection), _array(surface_minus_air_temperature)
        )
        advection_fog = (
            valid
            & (rh >= 95.0)
            & (wind >= 2.0)
            & (wind <= 7.0)
            & (advection > 0.0)
            & (thermal_contrast < 0.0)
            & ~radiation
        )
        result[advection_fog] = 2.0

    return _like(
        rh_sfc,
        result,
        "Tipo di nebbia (0 assente, 1 radiazione, 2 avvezione)",
        "category",
    )


def calculate_fog_probability(rh_sfc, wind_10m, clct, fog_type=None):
    """Return a conservative surface fog probability from 0 to 1."""
    rh, wind, cloud = np.broadcast_arrays(
        _array(rh_sfc), _array(wind_10m), _array(clct)
    )
    valid = np.isfinite(rh) & np.isfinite(wind) & np.isfinite(cloud)
    saturation = np.clip((rh - 88.0) / 12.0, 0.0, 1.0)
    calm = np.clip((5.0 - wind) / 4.0, 0.0, 1.0)
    clear = np.clip((70.0 - cloud) / 70.0, 0.0, 1.0)
    probability = saturation * (0.55 * calm + 0.25 * clear + 0.20)
    if fog_type is not None:
        category = _array(fog_type)
        probability = np.where(category > 0.0, np.maximum(probability, 0.55), probability)
    probability = np.where(rh < 90.0, np.minimum(probability, 0.12), probability)
    probability = np.where(valid, np.clip(probability, 0.0, 0.95), np.nan)
    return _like(rh_sfc, probability, "Probabilità diagnostica di nebbia", "probability")


def estimate_visibility(rh_sfc, fog_type, fog_probability=None):
    """Estimate horizontal visibility in metres from near-surface saturation.

    Kunkel's extinction relation needs liquid-water content, which is not
    present in the current surface feed.  This continuous RH-based proxy is
    labelled as an estimate and avoids the former fixed 100/200/500-m jumps.
    """
    rh = _array(rh_sfc)
    category = _array(fog_type)
    valid = np.isfinite(rh) & np.isfinite(category)
    # 20 km around 80% RH, about 3.3 km at 90%, 1.35 km at 95% and 0.55 km
    # at 100%.  Fog classifications then enforce the WMO <1-km ceiling.
    visibility = 20_000.0 * np.exp(-0.18 * np.maximum(rh - 80.0, 0.0))
    visibility = np.clip(visibility, 80.0, 30_000.0)
    if fog_probability is not None:
        probability = _array(fog_probability)
        visibility *= np.clip(1.0 - 0.55 * probability, 0.35, 1.0)
    visibility = np.where(category > 0.0, np.minimum(visibility, 950.0), visibility)
    visibility = np.where(valid, visibility, np.nan)
    return _like(
        rh_sfc,
        visibility,
        "Visibilità orizzontale stimata da saturazione superficiale",
        "m",
    )
