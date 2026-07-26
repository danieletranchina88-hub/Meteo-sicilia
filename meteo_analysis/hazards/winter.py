"""Freezing-rain diagnostics from a coarse pressure-level profile."""

from __future__ import annotations

import numpy as np


def _array(values):
    return np.asarray(values, dtype=float)


def _to_celsius(values):
    array = _array(values)
    finite = array[np.isfinite(array)]
    if finite.size and float(np.nanmedian(finite)) > 150.0:
        return array - 273.15
    return array


def _like(template, values):
    if hasattr(template, "dims") and hasattr(template, "copy"):
        result = template.copy(data=values)
        result.attrs.update(
            description="Rischio gelicidio (0 assente, 1 possibile, 2 elevato)",
            units="category",
        )
        return result
    return values


def detect_freezing_rain(
    t_925,
    t_850,
    t_700,
    t_2m,
    precip_rate,
):
    """Detect a warm nose above a sub-freezing surface.

    Category 2 requires a warm layer at least 1 °C and 925 hPa above freezing,
    which is consistent with a shallow surface cold layer.  A warm layer aloft
    with 925 hPa still below freezing is category 1 because sleet cannot be
    excluded from these sparse levels.
    """
    t925, t850, t700, surface, precipitation = np.broadcast_arrays(
        _to_celsius(t_925),
        _to_celsius(t_850),
        _to_celsius(t_700),
        _to_celsius(t_2m),
        _array(precip_rate),
    )
    valid = (
        np.isfinite(t925)
        & np.isfinite(t850)
        & np.isfinite(t700)
        & np.isfinite(surface)
        & np.isfinite(precipitation)
    )
    result = np.full(t925.shape, np.nan)
    result[valid] = 0.0
    warm_max = np.maximum.reduce((t925, t850, t700))
    candidate = valid & (surface < 0.0) & (warm_max > 0.0) & (precipitation > 0.1)
    high = candidate & (warm_max >= 1.0) & (t925 > 0.0)
    result[candidate] = 1.0
    result[high] = 2.0
    return _like(t_2m, result)


def freezing_rain_duration(risk, step_hours=1.0, axis=0):
    """Return duration in hours for each contiguous freezing-rain episode."""
    values = _array(risk)
    moved = np.moveaxis(values, axis, 0)
    active = np.isfinite(moved) & (moved >= 1.0)
    duration = np.zeros_like(moved, dtype=float)
    running = np.zeros(moved.shape[1:], dtype=float)
    for index in range(moved.shape[0]):
        running = np.where(active[index], running + float(step_hours), 0.0)
        duration[index] = running
    # Second pass assigns the full episode duration to every active time.
    for index in range(moved.shape[0] - 2, -1, -1):
        duration[index] = np.where(
            active[index] & active[index + 1],
            np.maximum(duration[index], duration[index + 1]),
            duration[index],
        )
    duration[~np.isfinite(moved)] = np.nan
    return np.moveaxis(duration, 0, axis)
