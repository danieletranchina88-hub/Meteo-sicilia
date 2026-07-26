"""Diagnostics for convective initiation.

The probability produced here is an expert diagnostic, not a calibrated
ensemble probability.  It deliberately requires independent thermodynamic
and kinematic evidence before values above 70% are allowed.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

EARTH_RADIUS_M = 6_371_000.0


def _as_float(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _return_like(template, values: np.ndarray, description: str, units: str):
    """Return an xarray object when the input is one, otherwise an ndarray."""
    if hasattr(template, "dims") and hasattr(template, "copy"):
        result = template.copy(data=values)
        result.attrs["description"] = description
        result.attrs["units"] = units
        return result
    return values


def _nan_gaussian(values: np.ndarray, sigma: tuple[float, float]) -> np.ndarray:
    """Gaussian smoothing which does not spread NaNs into valid cells."""
    array = _as_float(values)
    valid = np.isfinite(array)
    if not valid.any():
        return np.full_like(array, np.nan)
    numerator = gaussian_filter(
        np.where(valid, array, 0.0), sigma=sigma, mode="nearest"
    )
    denominator = gaussian_filter(valid.astype(float), sigma=sigma, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        smoothed = numerator / denominator
    return np.where(denominator > 0.05, smoothed, np.nan)


def normalize_cin(cin) -> np.ndarray:
    """Normalise CIN to the meteorological signed convention (J/kg, <= 0).

    Some GRIB encoders publish CIN as a positive inhibition magnitude while
    others publish the signed negative energy.  ``-abs(CIN)`` makes the
    threshold ``CIN > -50 J/kg`` unambiguous for both encodings.
    """
    values = _as_float(cin)
    return np.where(np.isfinite(values), -np.abs(values), np.nan)


def horizontal_convergence(
    u_wind,
    v_wind,
    latitudes,
    longitudes,
    smoothing_km: float = 10.0,
) -> np.ndarray:
    """Return smoothed horizontal convergence ``-(du/dx + dv/dy)`` in s-1.

    The input grid must be regular latitude/longitude.  Latitude may be
    ascending or descending.  A roughly 10-km smoothing suppresses grid-scale
    noise without erasing mesoscale convergence lines.
    """
    u = _as_float(u_wind)
    v = _as_float(v_wind)
    lat = _as_float(latitudes).squeeze()
    lon = _as_float(longitudes).squeeze()
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError("u e v devono essere griglie 2D con la stessa forma")
    if lat.ndim != 1 or lon.ndim != 1 or u.shape != (lat.size, lon.size):
        raise ValueError("coordinate 1D non compatibili con la griglia del vento")
    if lat.size < 3 or lon.size < 3:
        return np.full_like(u, np.nan)

    dlat_rad = abs(float(np.nanmedian(np.diff(np.deg2rad(lat)))))
    dlon_rad = abs(float(np.nanmedian(np.diff(np.deg2rad(lon)))))
    mean_lat = float(np.nanmean(lat))
    dy_m = max(EARTH_RADIUS_M * dlat_rad, 1.0)
    dx_m = max(EARTH_RADIUS_M * np.cos(np.deg2rad(mean_lat)) * dlon_rad, 1.0)
    sigma = (
        max(float(smoothing_km) * 1000.0 / dy_m, 0.0),
        max(float(smoothing_km) * 1000.0 / dx_m, 0.0),
    )
    smooth_u = _nan_gaussian(u, sigma)
    smooth_v = _nan_gaussian(v, sigma)

    # x grows eastward.  np.gradient handles a descending latitude coordinate
    # correctly when the signed metric coordinate is supplied explicitly.
    x_m = (
        np.deg2rad(lon - lon[0])[None, :]
        * EARTH_RADIUS_M
        * np.cos(np.deg2rad(lat))[:, None]
    )
    y_m = np.deg2rad(lat - lat[0]) * EARTH_RADIUS_M
    du_di = np.gradient(smooth_u, axis=1)
    dx_di = np.gradient(x_m, axis=1)
    du_dx = np.divide(
        du_di,
        dx_di,
        out=np.full_like(du_di, np.nan),
        where=np.abs(dx_di) > 1.0,
    )
    dv_dy = np.gradient(smooth_v, y_m, axis=0)
    convergence = -(du_dx + dv_dy)
    return np.where(np.isfinite(u) & np.isfinite(v), convergence, np.nan)


def _front_coordinate_sequences(fronts: dict | None) -> Iterable[np.ndarray]:
    if not isinstance(fronts, dict):
        return
    for feature in fronts.get("features", []):
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates") or []
        if geometry_type == "LineString":
            sequences = [coordinates]
        elif geometry_type == "MultiLineString":
            sequences = coordinates
        else:
            continue
        for sequence in sequences:
            array = np.asarray(sequence, dtype=float)
            if array.ndim == 2 and array.shape[1] >= 2 and len(array) >= 2:
                yield array[:, :2]


def front_distance_km(
    latitudes,
    longitudes,
    fronts: dict | None,
    sampling_km: float = 8.0,
) -> np.ndarray:
    """Approximate distance from every grid cell to the nearest front line."""
    lat = _as_float(latitudes).squeeze()
    lon = _as_float(longitudes).squeeze()
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("latitudine e longitudine devono essere coordinate 1D")
    output_shape = (lat.size, lon.size)
    sequences = list(_front_coordinate_sequences(fronts))
    if not sequences:
        return np.full(output_shape, np.inf, dtype=float)

    reference_lat = float(np.nanmean(lat))
    km_per_lon = 111.195 * np.cos(np.deg2rad(reference_lat))
    points: list[np.ndarray] = []
    for sequence in sequences:
        xy = np.column_stack((sequence[:, 0] * km_per_lon, sequence[:, 1] * 111.195))
        for start, end in zip(xy[:-1], xy[1:]):
            length = float(np.hypot(*(end - start)))
            count = max(2, int(np.ceil(length / max(sampling_km, 1.0))) + 1)
            fraction = np.linspace(0.0, 1.0, count)
            points.append(start + fraction[:, None] * (end - start))
    if not points:
        return np.full(output_shape, np.inf, dtype=float)

    front_points = np.vstack(points)
    grid_lon, grid_lat = np.meshgrid(lon, lat)
    grid_points = np.column_stack(
        (grid_lon.ravel() * km_per_lon, grid_lat.ravel() * 111.195)
    )
    distances, _ = cKDTree(front_points).query(grid_points, workers=-1)
    return distances.reshape(output_shape)


def calculate_convection_probability(
    mucape,
    cin,
    convergence_10m,
    front_distance_km,
    omega_700=None,
    surface_rh=None,
):
    """Calculate a conservative convective-initiation probability.

    Inputs are ML/MU-CAPE (J/kg), signed or magnitude CIN (J/kg), low-level
    convergence (s-1), nearest-front distance (km), optional 700-hPa omega
    (Pa/s; negative is ascent) and optional surface RH (%).

    The expert high-probability rule is enforced exactly: CAPE > 800 J/kg,
    CIN > -50 J/kg, convergence > 1e-4 s-1 and a front within 50 km.  Without
    all four ingredients the result is capped below 70%.
    """
    cape = _as_float(mucape)
    inhibition = normalize_cin(cin)
    convergence = _as_float(convergence_10m)
    distance = _as_float(front_distance_km)
    cape, inhibition, convergence, distance = np.broadcast_arrays(
        cape, inhibition, convergence, distance
    )
    valid = np.isfinite(cape) & np.isfinite(inhibition) & np.isfinite(convergence)

    cape_score = np.clip((cape - 100.0) / 1_400.0, 0.0, 1.0)
    cin_score = np.clip((inhibition + 200.0) / 175.0, 0.0, 1.0)
    convergence_score = np.clip((convergence - 1.0e-5) / 1.9e-4, 0.0, 1.0)
    front_score = np.where(
        np.isfinite(distance), np.exp(-np.maximum(distance, 0.0) / 60.0), 0.0
    )

    if omega_700 is None:
        ascent_score = np.zeros_like(cape)
    else:
        omega = np.broadcast_to(_as_float(omega_700), cape.shape)
        ascent_score = np.where(
            np.isfinite(omega), np.clip((-omega - 0.02) / 0.28, 0.0, 1.0), 0.0
        )

    if surface_rh is None:
        moisture_multiplier = np.ones_like(cape)
    else:
        humidity = np.broadcast_to(_as_float(surface_rh), cape.shape)
        moisture_multiplier = np.where(
            np.isfinite(humidity),
            0.65 + 0.35 * np.clip((humidity - 45.0) / 40.0, 0.0, 1.0),
            1.0,
        )

    probability = (
        2.0
        + 32.0 * cape_score * cin_score
        + 24.0 * cape_score * convergence_score
        + 8.0 * cape_score * front_score
        + 6.0 * cape_score * ascent_score
    ) * moisture_multiplier

    high = (
        (cape > 800.0)
        & (inhibition > -50.0)
        & (convergence > 1.0e-4)
        & (distance < 50.0)
    )
    medium = (
        (cape > 400.0)
        & (inhibition > -100.0)
        & (convergence > 0.5e-4)
        & ~high
    )

    medium_floor = (
        35.0
        + 9.0 * np.clip((cape - 400.0) / 800.0, 0.0, 1.0)
        + 7.0 * np.clip((convergence - 0.5e-4) / 1.0e-4, 0.0, 1.0)
    )
    high_floor = (
        72.0
        + 10.0 * np.clip((cape - 800.0) / 1_200.0, 0.0, 1.0)
        + 7.0 * np.clip((convergence - 1.0e-4) / 1.5e-4, 0.0, 1.0)
        + 5.0 * np.clip((50.0 - distance) / 50.0, 0.0, 1.0)
    )
    probability = np.where(medium, np.maximum(probability, medium_floor), probability)
    probability = np.where(high, np.maximum(probability, high_floor), probability)

    # Scientific guardrails: high values cannot be produced by CAPE alone.
    probability = np.where(~high, np.minimum(probability, 69.0), probability)
    probability = np.where(~(medium | high), np.minimum(probability, 34.0), probability)
    probability = np.where(cape < 100.0, np.minimum(probability, 8.0), probability)
    probability = np.where(
        inhibition <= -200.0, np.minimum(probability, 12.0), probability
    )
    probability = np.where(
        convergence <= 0.0, np.minimum(probability, 25.0), probability
    )
    probability = np.where(valid, np.clip(probability, 0.0, 95.0), np.nan)
    return _return_like(
        mucape,
        probability / 100.0,
        "Probabilità esperta di innesco convettivo",
        "probability",
    )


def summarize_convection(probability_percent, mask=None) -> dict:
    """Return robust statistics used by QC and natural-language products."""
    values = _as_float(probability_percent)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    sample = values[valid]
    if not sample.size:
        return {
            "status": "unavailable",
            "maximum": None,
            "p95": None,
            "areaAbove40Pct": None,
            "areaAbove70Pct": None,
        }
    return {
        "status": "available",
        "maximum": round(float(np.max(sample)), 1),
        "p95": round(float(np.percentile(sample, 95)), 1),
        "areaAbove40Pct": round(float(np.mean(sample >= 40.0) * 100.0), 2),
        "areaAbove70Pct": round(float(np.mean(sample >= 70.0) * 100.0), 2),
    }
