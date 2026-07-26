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


def cross_alpine_pressure_difference(
    pressure_hpa,
    latitudes,
    offset_degrees=1.35,
):
    """Return pressure north minus pressure south across the Alpine barrier.

    The input grid is regular in latitude.  Each grid row is compared with
    points roughly 150 km north and south, which is wide enough to span the
    main Alpine ridge without replacing the diagnostic with a local pressure
    gradient.
    """
    pressure = _array(pressure_hpa)
    latitude = _array(latitudes).squeeze()
    if pressure.ndim != 2 or latitude.ndim != 1 or pressure.shape[0] != latitude.size:
        raise ValueError("griglia pressione/latitudine non coerente")

    if latitude[0] > latitude[-1]:
        latitude_ascending = latitude[::-1]
        pressure_ascending = pressure[::-1]
    else:
        latitude_ascending = latitude
        pressure_ascending = pressure

    north = np.full_like(pressure, np.nan, dtype=float)
    south = np.full_like(pressure, np.nan, dtype=float)
    north_targets = latitude + float(offset_degrees)
    south_targets = latitude - float(offset_degrees)
    for column in range(pressure.shape[1]):
        values = pressure_ascending[:, column]
        finite = np.isfinite(latitude_ascending) & np.isfinite(values)
        if np.count_nonzero(finite) < 2:
            continue
        north[:, column] = np.interp(
            north_targets,
            latitude_ascending[finite],
            values[finite],
            left=np.nan,
            right=np.nan,
        )
        south[:, column] = np.interp(
            south_targets,
            latitude_ascending[finite],
            values[finite],
            left=np.nan,
            right=np.nan,
        )
    return north - south


def alpine_domain_mask(latitudes, longitudes):
    """Return a conservative mask covering the Alpine ridge and lee valleys."""
    latitude = _array(latitudes).squeeze()
    longitude = _array(longitudes).squeeze()
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("coordinate alpine non monodimensionali")
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    broad_arc = (
        (lat_grid >= 44.4)
        & (lat_grid <= 48.25)
        & (lon_grid >= 5.8)
        & (lon_grid <= 15.8)
    )
    # Exclude the broad Po Valley and the distant north-eastern lowlands.
    south_edge = 44.65 + 0.055 * np.maximum(lon_grid - 6.0, 0.0)
    north_edge = 48.2 - 0.025 * np.maximum(lon_grid - 6.0, 0.0)
    return broad_arc & (lat_grid >= south_edge) & (lat_grid <= north_edge)


def detect_foehn(
    u_700,
    v_700,
    rh_sfc,
    north_minus_south_pressure_hpa=None,
    ridge_axis_degrees_from_east=10.0,
    domain_mask=None,
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
    if domain_mask is not None:
        mask = np.broadcast_to(np.asarray(domain_mask, dtype=bool), result.shape)
        result[valid & ~mask] = 0.0
    return _like(rh_sfc, result)
