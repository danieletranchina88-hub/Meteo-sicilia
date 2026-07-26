"""Alpine foehn diagnostic."""

from __future__ import annotations

import numpy as np


def _array(values):
    return np.asarray(values, dtype=float)


def _like(template, values):
    if hasattr(template, "dims") and hasattr(template, "copy"):
        result = template.copy(data=values)
        result.attrs.update(
            description="Foehn alpino (0 assente, 1 da nord, 2 da sud)",
            units="category",
        )
        return result
    return values


def detect_foehn(
    u_700,
    v_700,
    rh_sfc,
    north_minus_south_pressure_hpa=None,
    ridge_axis_degrees_from_east=10.0,
):
    """Detect cross-Alpine foehn with flow, pressure and lee-side dryness.

    ``north_minus_south_pressure_hpa`` is positive when pressure is higher on
    the northern side.  Without this real cross-barrier pressure difference
    the result is unavailable; multiplying surface wind is not a substitute.
    """
    u, v, humidity = np.broadcast_arrays(
        _array(u_700), _array(v_700), _array(rh_sfc)
    )
    if north_minus_south_pressure_hpa is None:
        return _like(rh_sfc, np.full(u.shape, np.nan))
    pressure_difference = np.broadcast_to(
        _array(north_minus_south_pressure_hpa), u.shape
    )
    valid = (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(humidity)
        & np.isfinite(pressure_difference)
    )

    axis = np.deg2rad(float(ridge_axis_degrees_from_east))
    # Unit normal points approximately north for the west-east Alpine ridge.
    normal_u = -np.sin(axis)
    normal_v = np.cos(axis)
    northward_normal_flow = u * normal_u + v * normal_v
    threshold_ms = 15.0 * 0.514444  # 15 kt

    result = np.full(u.shape, np.nan)
    result[valid] = 0.0
    # Higher pressure north + southward flow -> north foehn on the southern
    # slopes; the opposite configuration is south foehn.
    north_foehn = (
        valid
        & (northward_normal_flow < -threshold_ms)
        & (pressure_difference >= 2.0)
        & (humidity < 40.0)
    )
    south_foehn = (
        valid
        & (northward_normal_flow > threshold_ms)
        & (pressure_difference <= -2.0)
        & (humidity < 40.0)
    )
    result[north_foehn] = 1.0
    result[south_foehn] = 2.0
    return _like(rh_sfc, result)
