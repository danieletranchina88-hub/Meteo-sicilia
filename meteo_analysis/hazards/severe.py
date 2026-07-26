"""Severe-convection composite diagnostics.

These functions calculate ingredients only when the required physical fields
are available.  Missing temperature, helicity or shear is represented by NaN,
never by a climatological constant.
"""

from __future__ import annotations

import numpy as np


def _array(values):
    return np.asarray(values, dtype=float)


def _like(template, values, description):
    if hasattr(template, "dims") and hasattr(template, "copy"):
        result = template.copy(data=values)
        result.attrs.update(description=description, units="dimensionless")
        return result
    return values


def calculate_ship(
    mucape,
    lapse_rate_700_500,
    z_0c,
    mix_ratio,
    temperature_500_c=None,
):
    """Calculate a SHIP-style significant-hail diagnostic.

    The operational SHIP formulation requires 500-hPa temperature.  If that
    field is absent the result is unavailable (NaN), rather than deriving it
    from surface temperature or freezing-level height.
    """
    cape, lapse, freezing_level, mixing_ratio = np.broadcast_arrays(
        _array(mucape),
        _array(lapse_rate_700_500),
        _array(z_0c),
        _array(mix_ratio),
    )
    if temperature_500_c is None:
        return _like(
            mucape,
            np.full(cape.shape, np.nan),
            "SHIP non disponibile: temperatura a 500 hPa assente",
        )
    t500 = np.broadcast_to(_array(temperature_500_c), cape.shape)
    valid = (
        np.isfinite(cape)
        & np.isfinite(lapse)
        & np.isfinite(freezing_level)
        & np.isfinite(mixing_ratio)
        & np.isfinite(t500)
    )
    ship = (
        np.maximum(cape, 0.0)
        * np.maximum(mixing_ratio, 0.0)
        * np.maximum(lapse, 0.0)
        * np.maximum(-t500, 0.0)
        / 42_000_000.0
    )
    # Standard low-end adjustments keep marginal ingredients from producing
    # a large composite through multiplication alone.
    ship *= np.where(cape < 1_300.0, np.clip(cape / 1_300.0, 0.0, 1.0), 1.0)
    ship *= np.where(lapse < 5.8, np.clip(lapse / 5.8, 0.0, 1.0), 1.0)
    ship *= np.where(t500 > -23.0, np.clip((-t500) / 23.0, 0.0, 1.0), 1.0)
    # Very low freezing levels favour small/partly melted hail; exceptionally
    # high levels strongly reduce survival to the ground.
    freezing_factor = np.clip((freezing_level - 1_000.0) / 1_000.0, 0.0, 1.0)
    freezing_factor *= np.clip((5_000.0 - freezing_level) / 1_500.0, 0.0, 1.0)
    ship *= freezing_factor
    ship = np.where(valid, np.clip(ship, 0.0, 8.0), np.nan)
    return _like(mucape, ship, "Significant Hail Parameter (SHIP-style)")


def calculate_scp(mucape, srh_0_3km, bulk_shear_0_6km):
    """Calculate the right-moving Supercell Composite Parameter."""
    cape, srh, shear = np.broadcast_arrays(
        _array(mucape), _array(srh_0_3km), _array(bulk_shear_0_6km)
    )
    valid = np.isfinite(cape) & np.isfinite(srh) & np.isfinite(shear)
    shear_term = np.where(
        shear < 10.0,
        0.0,
        np.where(shear > 20.0, 1.0, (shear - 10.0) / 10.0),
    )
    shear_term = np.clip(shear_term, 0.0, 1.5)
    scp = (
        np.maximum(cape, 0.0) / 1_000.0
        * np.maximum(srh, 0.0) / 50.0
        * shear_term
    )
    scp = np.where(valid, np.clip(scp, 0.0, 30.0), np.nan)
    return _like(mucape, scp, "Supercell Composite Parameter")


def evaluate_hail_threat(ship, scp, dry_air_700hpa_rh):
    """Categorise hail threat: 0 low, 1 medium, 2 high."""
    ship_values, scp_values, rh700 = np.broadcast_arrays(
        _array(ship), _array(scp), _array(dry_air_700hpa_rh)
    )
    valid = np.isfinite(ship_values) & np.isfinite(scp_values) & np.isfinite(rh700)
    result = np.full(ship_values.shape, np.nan)
    result[valid] = 0.0
    dry_intrusion = (rh700 >= 15.0) & (rh700 < 45.0)
    high = valid & (ship_values >= 1.0) & (scp_values >= 1.0) & dry_intrusion
    medium = valid & ~high & ((ship_values >= 0.5) | (scp_values >= 0.5))
    result[medium] = 1.0
    result[high] = 2.0
    return _like(ship, result, "Minaccia grandinigena (0 bassa, 1 media, 2 alta)")


def require_temporal_persistence(threat, minimum_category=1, hours=2, axis=0):
    """Keep categories present for at least ``hours`` consecutive time steps."""
    values = _array(threat)
    if values.shape[axis] < hours:
        return np.full_like(values, np.nan)
    moved = np.moveaxis(values, axis, 0)
    qualifying = np.isfinite(moved) & (moved >= minimum_category)
    persistent = np.zeros_like(qualifying, dtype=bool)
    for start in range(0, moved.shape[0] - hours + 1):
        window = np.all(qualifying[start : start + hours], axis=0)
        for offset in range(hours):
            persistent[start + offset] |= window
    filtered = np.where(
        np.isfinite(moved), np.where(persistent, moved, np.minimum(moved, 0.0)), np.nan
    )
    return np.moveaxis(filtered, 0, axis)
