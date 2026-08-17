"""Independent, bounded confirmations for objective frontal candidates.

The operational geometry remains thermodynamically seeded and must pass all
existing air-mass, synoptic-scale and temporal gates.  These diagnostics may
only add a small amount of evidence and quantify disagreement; they can never
publish a wind-only or vorticity-only line.

Implemented signals:
* Parfitt et al. (2017) F = zeta |grad(T)| / (f * 0.45 K/100 km);
* a six-hour-normalised 10 m wind-change diagnostic inspired by Schemm et al.
  (2015), with the SW-to-NW transition kept separate from general rotation;
* symmetric geometric disagreement between independent locator families.
"""

from __future__ import annotations

import numpy as np

import front_detection as fd
import front_locator as fl

OMEGA_EARTH = 7.2921159e-5


def _smoothstep_field(values, weak: float, strong: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.zeros_like(values)
    finite = np.isfinite(values)
    result[finite] = np.clip(
        (values[finite] - weak) / max(strong - weak, 1.0e-12), 0.0, 1.0
    )
    return result


def parfitt_f_field(
    temperature: np.ndarray,
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    *,
    smoothing_km: float = 45.0,
) -> dict[str, np.ndarray]:
    """Return signed Parfitt-F and bounded support on a pressure surface."""
    metrics = fl.grid_metrics(longitudes, latitudes)
    temperature = np.asarray(temperature, dtype=float)
    u_wind = np.asarray(u_wind, dtype=float)
    v_wind = np.asarray(v_wind, dtype=float)
    input_valid = (
        np.isfinite(temperature) & np.isfinite(u_wind) & np.isfinite(v_wind)
    )
    temperature = fl.smooth_km(temperature, smoothing_km, metrics)
    u_wind = fl.smooth_km(u_wind, smoothing_km, metrics)
    v_wind = fl.smooth_km(v_wind, smoothing_km, metrics)
    temp_e, temp_n = fl.gradient(temperature, metrics)
    _, u_n_km = fl.gradient(u_wind, metrics)
    v_e_km, _ = fl.gradient(v_wind, metrics)
    vorticity = (v_e_km - u_n_km) / 1000.0
    coriolis = (
        2.0 * OMEGA_EARTH * np.sin(np.deg2rad(latitudes))
    )[:, None]
    thermal_gradient_100 = np.hypot(temp_e, temp_n) * 100.0
    denominator = np.maximum(np.abs(coriolis) * 0.45, 1.0e-10)
    signed_f = vorticity * thermal_gradient_100 / denominator
    valid = (
        input_valid
        & np.isfinite(temperature) & np.isfinite(u_wind) & np.isfinite(v_wind)
        & (np.abs(coriolis) > 1.0e-5)
    )
    signed_f = np.where(valid, signed_f, np.nan)
    return {
        "parfittF": signed_f,
        "parfittSupport": _smoothstep_field(signed_f, 0.50, 1.50),
        "parfittMask": valid & (signed_f >= 1.0),
        "relativeVorticity1e5": np.where(valid, vorticity * 1.0e5, np.nan),
        "temperatureGradient100Km": np.where(valid, thermal_gradient_100, np.nan),
    }


def temporal_wind_shift_field(
    u_before: np.ndarray,
    v_before: np.ndarray,
    u_after: np.ndarray,
    v_after: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    *,
    elapsed_hours: float,
    smoothing_km: float = 45.0,
) -> dict[str, np.ndarray]:
    """Six-hour-normalised 10 m wind change, never an existence gate.

    ``coldSupport`` is deliberately strict: in the Northern Hemisphere it
    recognises the classic wind-from-SW to wind-from-NW passage (model wind
    components change from northward to southward while remaining eastward).
    ``generalSupport`` also records a substantial rotation/change for fronts
    whose local orientation makes the quadrant test inappropriate.
    """
    if not np.isfinite(elapsed_hours) or elapsed_hours <= 0.0:
        raise ValueError("elapsed_hours deve essere positivo")
    metrics = fl.grid_metrics(longitudes, latitudes)
    fields = [
        fl.smooth_km(np.asarray(value, float), smoothing_km, metrics)
        for value in (u_before, v_before, u_after, v_after)
    ]
    u0, v0, u1, v1 = fields
    scale = 6.0 / float(elapsed_hours)
    change = np.hypot(u1 - u0, v1 - v0) * scale
    speed0, speed1 = np.hypot(u0, v0), np.hypot(u1, v1)
    dot = u0 * u1 + v0 * v1
    cross = u0 * v1 - v0 * u1
    turn = np.degrees(np.arctan2(np.abs(cross), dot))
    directional = (speed0 >= 1.5) & (speed1 >= 1.5)
    change_support = _smoothstep_field(change, 2.0, 6.0)
    turn_support = _smoothstep_field(turn, 15.0, 55.0)
    general = change_support * np.where(directional, turn_support, 0.45)
    north_to_south_change = (v0 - v1) * scale
    classic_cold = (
        (np.asarray(latitudes)[:, None] > 0.0)
        & (u0 >= 0.0) & (u1 >= 0.0) & (v0 > 0.0) & (v1 < 0.0)
    )
    cold = np.where(
        classic_cold,
        np.minimum(
            change_support,
            _smoothstep_field(north_to_south_change, 2.0, 6.0),
        ),
        0.0,
    )
    valid = np.isfinite(u0) & np.isfinite(v0) & np.isfinite(u1) & np.isfinite(v1)
    return {
        "windChangeMs6h": np.where(valid, change, np.nan),
        "windTurnDeg": np.where(valid & directional, turn, np.nan),
        "wndGeneralSupport": np.where(valid, general, 0.0),
        "wndColdSupport": np.where(valid, cold, 0.0),
        "valid": valid,
    }


def sample_line(field, coordinates, longitudes, latitudes) -> np.ndarray:
    metrics = fl.grid_metrics(longitudes, latitudes)
    return fl._sample(
        np.asarray(field, float), np.asarray(coordinates, float),
        np.asarray(longitudes, float), np.asarray(latitudes, float),
        metrics["dlon"], metrics["dlat"],
    )


def line_consensus_metrics(
    coordinates: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    *,
    parfitt: dict | None = None,
    wnd: dict | None = None,
    locator_sources: list[str] | tuple[str, ...] = (),
) -> dict:
    """Summarise independent confirmations along one thermal candidate."""
    output: dict[str, float | int | list] = {
        "locatorSources": sorted(set(locator_sources)),
        "locatorAgreementCount": len(set(locator_sources)),
    }
    supports = []
    available = 0
    if parfitt is not None:
        f_values = sample_line(
            parfitt["parfittF"], coordinates, longitudes, latitudes
        )
        p_support = sample_line(
            parfitt["parfittSupport"], coordinates, longitudes, latitudes
        )
        finite = np.isfinite(f_values)
        if np.any(finite):
            available += 1
            output["parfittF"] = float(np.nanmedian(f_values))
            output["parfittSupportFraction"] = float(np.mean(f_values[finite] >= 1.0))
            value = float(np.nanmedian(p_support[finite]))
            output["parfittSupport"] = value
            supports.append(value)
    if wnd is not None:
        change = sample_line(
            wnd["windChangeMs6h"], coordinates, longitudes, latitudes
        )
        turn = sample_line(wnd["windTurnDeg"], coordinates, longitudes, latitudes)
        general = sample_line(wnd["wndGeneralSupport"], coordinates, longitudes, latitudes)
        cold = sample_line(wnd["wndColdSupport"], coordinates, longitudes, latitudes)
        finite = np.isfinite(change)
        if np.any(finite):
            available += 1
            output["wndChangeMs6h"] = float(np.nanmedian(change))
            output["wndTurnDeg"] = (
                float(np.nanmedian(turn))
                if np.any(np.isfinite(turn)) else np.nan
            )
            output["wndSupportFraction"] = float(np.mean(general[finite] >= 0.5))
            output["wndColdSupportFraction"] = float(np.mean(cold[finite] >= 0.5))
            value = float(np.nanmedian(general[finite]))
            output["wndSupport"] = value
            supports.append(value)
    locator_count = len(set(locator_sources))
    if locator_count:
        available += 1
        supports.append(float(np.clip((locator_count - 1) / 2.0, 0.0, 1.0)))
    output["methodAvailability"] = available
    output["methodAgreementCount"] = int(
        (locator_count >= 2)
        + (output.get("parfittSupport", 0.0) >= 0.5)
        + (output.get("wndSupport", 0.0) >= 0.5)
    )
    output["consensusSupport"] = (
        float(np.mean(supports)) if supports else 0.5
    )
    return output


def symmetric_line_distance_km(first: np.ndarray, second: np.ndarray) -> float:
    """Robust symmetric median separation of two matching polylines."""
    first = fd._resample_km(np.asarray(first, float), 20.0)
    second = fd._resample_km(np.asarray(second, float), 20.0)
    mean_lat = np.deg2rad(float(np.mean(np.r_[first[:, 1], second[:, 1]])))
    scale_lon = fl.EARTH_KM_PER_DEG * np.cos(mean_lat)
    first_km = np.column_stack((
        first[:, 0] * scale_lon, first[:, 1] * fl.EARTH_KM_PER_DEG
    ))
    second_km = np.column_stack((
        second[:, 0] * scale_lon, second[:, 1] * fl.EARTH_KM_PER_DEG
    ))
    d1 = fd._points_to_segments_km(first_km, second_km)
    d2 = fd._points_to_segments_km(second_km, first_km)
    return float(0.5 * (np.median(d1) + np.median(d2)))
