"""Identical physical predictors for ERA5 training and ICON-2I inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

EARTH_RADIUS_M = 6_371_000.0
P0_PA = 100_000.0
KAPPA = 0.2854
EPSILON = 0.62197
GRAVITY = 9.80665

GRID_BOUNDS = (3.0, 22.0, 33.7, 48.9)
GRID_RESOLUTION_DEG = 0.20
SMOOTHING_SCALES_KM = (20, 50, 100)
MIN_FEATURE_COMPLETENESS = 0.80

FEATURE_COLUMNS = [
    "theta_850_k", "theta_e_850_k",
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
    "rh_850_pct", "rh_700_pct",
    "mixing_ratio_850_gkg", "mixing_ratio_700_gkg",
    "convergence_10m_1e5_s", "divergence_850_1e5_s",
    "vorticity_850_1e5_s", "bulk_shear_0_6km_ms",
    "pmsl_tendency_3h_hpa", "vorticity_advection_500_1e10_s2",
    "geopotential_height_500_m", "elevation_m", "ruggedness_10km_m",
    "grad_theta_850_50_mean_3h", "grad_theta_850_50_std_3h",
    "lat", "lon",
]

ERA5_TRANSFER_FEATURE_COLUMNS = [
    name for name in FEATURE_COLUMNS
    if name not in {"bulk_shear_0_6km_ms", "elevation_m", "ruggedness_10km_m"}
]
ERA5_TRANSFER_NO_COORDINATES_FEATURE_COLUMNS = [
    name for name in ERA5_TRANSFER_FEATURE_COLUMNS if name not in {"lat", "lon"}
]


def feature_schema_hash(features: Sequence[str] = FEATURE_COLUMNS) -> str:
    payload = json.dumps(list(features), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def regular_grid(bounds=GRID_BOUNDS, resolution=GRID_RESOLUTION_DEG):
    west, east, south, north = map(float, bounds)
    if not (west < east and south < north and resolution > 0):
        raise ValueError("bounds o risoluzione non validi")
    lon = np.arange(west, east + resolution * 0.25, resolution)
    lat = np.arange(south, north + resolution * 0.25, resolution)
    return lat, lon


def _spacing(latitude, longitude):
    dy = EARTH_RADIUS_M * abs(float(np.nanmedian(np.diff(np.deg2rad(latitude)))))
    dx = (
        EARTH_RADIUS_M
        * np.cos(np.deg2rad(float(np.nanmean(latitude))))
        * abs(float(np.nanmedian(np.diff(np.deg2rad(longitude)))))
    )
    return max(dy, 1.0), max(dx, 1.0)


def _nan_gaussian(values, sigma):
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    if not valid.any():
        return np.full_like(values, np.nan)
    numerator = gaussian_filter(np.where(valid, values, 0.0), sigma, mode="nearest")
    denominator = gaussian_filter(valid.astype(float), sigma, mode="nearest")
    result = numerator / np.maximum(denominator, 1.0e-9)
    return np.where(denominator >= 0.15, result, np.nan)


def _smooth_km(values, latitude, longitude, scale_km):
    dy, dx = _spacing(latitude, longitude)
    return _nan_gaussian(
        values, (scale_km * 1000.0 / dy, scale_km * 1000.0 / dx)
    )


def _gradient(values, latitude, longitude):
    values = np.asarray(values, float)
    y = np.deg2rad(latitude - latitude[0]) * EARTH_RADIUS_M
    x = (
        np.deg2rad(longitude - longitude[0])[None, :]
        * EARTH_RADIUS_M
        * np.cos(np.deg2rad(latitude))[:, None]
    )
    dx = np.gradient(x, axis=1)
    east = np.divide(
        np.gradient(values, axis=1), dx,
        out=np.full_like(values, np.nan), where=np.abs(dx) > 1.0,
    )
    north = np.gradient(values, y, axis=0)
    return east, north


def _thermodynamics(pressure_hpa, temperature, specific_humidity):
    temperature = np.asarray(temperature, float)
    if np.nanmedian(temperature) < 150:
        temperature = temperature + 273.15
    q = np.asarray(specific_humidity, float)
    if np.nanmedian(np.abs(q)) > 0.2:
        q = q / 1000.0
    valid = (
        np.isfinite(temperature) & np.isfinite(q)
        & (temperature > 180) & (temperature < 340)
        & (q >= 0) & (q < 0.06)
    )
    q = np.where(valid, q, np.nan)
    p = float(pressure_hpa) * 100.0
    mixing_ratio = q / np.maximum(1.0 - q, 1e-9)
    vapour_pressure = p * mixing_ratio / (EPSILON + mixing_ratio)
    logarithm = np.log(np.maximum(vapour_pressure / 100.0, 1e-12) / 6.112)
    dewpoint = 243.5 * logarithm / (17.67 - logarithm) + 273.15
    dewpoint = np.minimum(dewpoint, temperature)
    theta = temperature * (P0_PA / p) ** KAPPA
    t_lcl = 56.0 + 1.0 / (
        1.0 / (dewpoint - 56.0) + np.log(temperature / dewpoint) / 800.0
    )
    theta_dl = (
        temperature
        * (P0_PA / np.maximum(p - vapour_pressure, 1.0)) ** KAPPA
        * (temperature / t_lcl) ** (0.28 * mixing_ratio)
    )
    theta_e = theta_dl * np.exp(
        mixing_ratio * (1.0 + 0.448 * mixing_ratio)
        * (3036.0 / t_lcl - 1.78)
    )
    tc = temperature - 273.15
    saturation = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    rh = np.clip(100.0 * vapour_pressure / saturation, 0.0, 100.0)
    return {
        "theta": np.where(valid, theta, np.nan),
        "theta_e": np.where(valid, theta_e, np.nan),
        "dewpoint": np.where(valid, dewpoint, np.nan),
        "rh": np.where(valid, rh, np.nan),
        "mixing_ratio_gkg": np.where(valid, mixing_ratio * 1000.0, np.nan),
    }


def _kinematics(u, v, latitude, longitude):
    ux, uy = _gradient(u, latitude, longitude)
    vx, vy = _gradient(v, latitude, longitude)
    return ux + vy, vx - uy, ux - vy, vx + uy


def _frontogenesis(theta, u, v, latitude, longitude):
    tx, ty = _gradient(theta, latitude, longitude)
    magnitude = np.hypot(tx, ty)
    divergence, _vorticity, stretch, shear = _kinematics(
        u, v, latitude, longitude
    )
    alpha = np.arctan2(ty, tx)
    # Petterssen 2-D form; K/(100 km)/(3 h).
    deformation_projection = (
        stretch * np.cos(2.0 * alpha) + shear * np.sin(2.0 * alpha)
    )
    return (
        -0.5 * magnitude * (divergence + deformation_projection)
        * 100_000.0 * 10_800.0
    )


def theta_gradient_50(temperature, latitude, longitude):
    temperature = np.asarray(temperature, float)
    if np.nanmedian(temperature) < 150:
        temperature = temperature + 273.15
    theta = temperature * (P0_PA / 85_000.0) ** KAPPA
    theta = _smooth_km(theta, latitude, longitude, 50.0)
    return np.hypot(*_gradient(theta, latitude, longitude)) * 100_000.0


def compute_feature_frame(
    fields: Mapping[str, np.ndarray],
    latitude,
    longitude,
    *,
    valid_time,
    previous_pmsl_3h=None,
    gradient_history: Sequence[np.ndarray] | None = None,
):
    latitude = np.asarray(latitude, float)
    longitude = np.asarray(longitude, float)
    arrays = {name: np.asarray(value, float) for name, value in fields.items()}
    shape = (latitude.size, longitude.size)
    if any(value.shape != shape for value in arrays.values()):
        raise ValueError("campi con griglie non coincidenti")
    thermo850 = _thermodynamics(850.0, arrays["t850"], arrays["q850"])
    thermo700 = _thermodynamics(700.0, arrays["t700"], arrays["q700"])
    result = {
        "theta_850_k": thermo850["theta"],
        "theta_e_850_k": thermo850["theta_e"],
        "rh_850_pct": thermo850["rh"],
        "rh_700_pct": thermo700["rh"],
        "mixing_ratio_850_gkg": thermo850["mixing_ratio_gkg"],
        "mixing_ratio_700_gkg": thermo700["mixing_ratio_gkg"],
    }
    gradients = {}
    for scale in SMOOTHING_SCALES_KM:
        theta = _smooth_km(
            thermo850["theta"], latitude, longitude, float(scale)
        )
        gx, gy = _gradient(theta, latitude, longitude)
        gradient = np.hypot(gx, gy) * 100_000.0
        gradients[scale] = gradient
        result[f"grad_theta_850_{scale}_k_per_100km"] = gradient
        result[f"thermal_advection_850_{scale}_k_per_3h"] = (
            -(arrays["u850"] * gx + arrays["v850"] * gy) * 10_800.0
        )
        result[f"frontogenesis_850_{scale}_k_per_100km_3h"] = _frontogenesis(
            theta, arrays["u850"], arrays["v850"], latitude, longitude
        )
    for level, thermo in ((850, thermo850), (700, thermo700)):
        dx, dy = _gradient(thermo["dewpoint"], latitude, longitude)
        result[f"dewpoint_gradient_{level}_k_per_100km"] = (
            np.hypot(dx, dy) * 100_000.0
        )
    div10, _, _, _ = _kinematics(
        arrays["u10"], arrays["v10"], latitude, longitude
    )
    div850, vort850, _, _ = _kinematics(
        arrays["u850"], arrays["v850"], latitude, longitude
    )
    result["convergence_10m_1e5_s"] = -div10 * 1e5
    result["divergence_850_1e5_s"] = div850 * 1e5
    result["vorticity_850_1e5_s"] = vort850 * 1e5
    result["bulk_shear_0_6km_ms"] = np.hypot(
        arrays["wshear_u_0_6km"], arrays["wshear_v_0_6km"]
    )
    if previous_pmsl_3h is None:
        result["pmsl_tendency_3h_hpa"] = np.full(shape, np.nan)
    else:
        result["pmsl_tendency_3h_hpa"] = (
            arrays["pmsl"] - np.asarray(previous_pmsl_3h, float)
        ) / 100.0
    _div500, vort500, _, _ = _kinematics(
        arrays["u500"], arrays["v500"], latitude, longitude
    )
    zx, zy = _gradient(vort500, latitude, longitude)
    result["vorticity_advection_500_1e10_s2"] = (
        -(arrays["u500"] * zx + arrays["v500"] * zy) * 1e10
    )
    result["geopotential_height_500_m"] = arrays["fi500"] / GRAVITY
    result["elevation_m"] = arrays["hsurf"]
    result["ruggedness_10km_m"] = arrays["ruggedness_10km"]
    history = list(gradient_history or []) + [gradients[50]]
    stack = np.stack(history[-3:])
    result["grad_theta_850_50_mean_3h"] = np.nanmean(stack, axis=0)
    result["grad_theta_850_50_std_3h"] = (
        np.nanstd(stack, axis=0) if len(stack) >= 2 else np.full(shape, np.nan)
    )
    lon2d, lat2d = np.meshgrid(longitude, latitude)
    result["lat"], result["lon"] = lat2d, lon2d
    timestamp = pd.Timestamp(valid_time)
    timestamp = (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    )
    return pd.DataFrame({
        "time": np.repeat(timestamp, latitude.size * longitude.size),
        **{name: result[name].ravel() for name in FEATURE_COLUMNS},
    })


def fields_from_store(store, hour, *, feature_profile="all"):
    names = (
        "t850", "q850", "u850", "v850", "t700", "q700",
        "u500", "v500", "fi500", "u10", "v10", "pmsl",
    )
    fields = {name: store.field(name, int(hour)) for name in names}
    shape = fields["t850"].shape
    if feature_profile == "all":
        for name in ("wshear_u_0_6km", "wshear_v_0_6km", "hsurf"):
            fields[name] = store.field(name, int(hour))
        fields["ruggedness_10km"] = store.terrain_ruggedness_10km()
    elif feature_profile in {
        "era5-transfer", "era5-transfer-no-coordinates"
    }:
        for name in (
            "wshear_u_0_6km", "wshear_v_0_6km", "hsurf", "ruggedness_10km"
        ):
            fields[name] = np.zeros(shape)
    else:
        raise ValueError(f"profilo feature sconosciuto: {feature_profile}")
    return fields


def theta_gradient_50_from_store(store, hour):
    return theta_gradient_50(
        store.field("t850", int(hour)),
        store.target_latitudes,
        store.target_longitudes,
    )


def feature_frame_from_store(
    store, hour, *, gradient_history=None, feature_profile="all"
):
    fields = fields_from_store(store, hour, feature_profile=feature_profile)
    previous = (
        store.field("pmsl", hour - 3)
        if hour >= 3 and hour - 3 in store.available_hours("pmsl")
        else None
    )
    run_time = datetime.strptime(store.run_tag, "%Y%m%d%H").replace(
        tzinfo=timezone.utc
    )
    return compute_feature_frame(
        fields, store.target_latitudes, store.target_longitudes,
        valid_time=run_time + timedelta(hours=int(hour)),
        previous_pmsl_3h=previous,
        gradient_history=gradient_history,
    )
