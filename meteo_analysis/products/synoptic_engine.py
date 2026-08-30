"""Auditable, ingredients-based ICON-2I synoptic bulletin engine.

The engine reduces real gridded fields into regional evidence before writing
language.  It deliberately separates three states which are often confused:

* a diagnostic is available and supports a process;
* a diagnostic is available and does not support it;
* the diagnostic is not available.

No missing quantity is replaced by climatology or a fabricated constant.  The
output is deterministic model guidance, not an official warning or a
calibrated probability forecast.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


ENGINE_METHOD = "icon2i-multifield-synoptic-engine-v1"
EARTH_RADIUS_M = 6_371_000.0

# Overlap at the regional borders is intentional: this is a narrative
# diagnostic, not an administrative warning system.  A signal on the Po
# valley or central Apennines must not disappear because of a hard border.
REGIONS = {
    "Nord-Ovest": (43.5, 48.9, 3.0, 12.5),
    "Nord-Est": (43.5, 48.9, 10.5, 16.8),
    "Centro": (40.2, 44.7, 7.5, 16.8),
    "Sud peninsulare": (37.3, 42.4, 12.0, 19.2),
    "Sicilia": (35.0, 38.8, 11.2, 16.7),
    "Sardegna": (38.2, 41.7, 7.4, 10.4),
}

FIELD_LABELS = {
    "pressureMsl": "MSLP",
    "temperature2m": "temperatura 2 m",
    "relativeHumidity2m": "UR 2 m",
    "rainStep": "precipitazione oraria",
    "cloudCover": "copertura nuvolosa",
    "wind10": "vento 10 m",
    "gust10": "raffica 10 m",
    "convergence10": "convergenza 10 m",
    "frontDistanceKm": "distanza dal fronte pubblicato",
    "temperature925": "temperatura 925 hPa",
    "thetaE925": "theta-e 925 hPa",
    "relativeHumidity925": "UR 925 hPa",
    "wind925": "vento 925 hPa",
    "temperature850": "temperatura 850 hPa",
    "thetaE850": "theta-e 850 hPa",
    "thetaW850": "theta-w 850 hPa",
    "thetaEGradient850": "gradiente theta-e 850 hPa",
    "thermalAdvection850": "advezione theta-e 850 hPa",
    "relativeHumidity850": "UR 850 hPa",
    "wind850": "vento 850 hPa",
    "temperature700": "temperatura 700 hPa",
    "relativeHumidity700": "UR 700 hPa",
    "omega700": "omega 700 hPa",
    "wind700": "vento 700 hPa",
    "height500": "geopotenziale 500 hPa",
    "temperature500": "temperatura 500 hPa",
    "lapseRate700500": "gradiente termico 700–500 hPa",
    "vorticity500": "vorticità relativa 500 hPa",
    "vorticityAdvection500": "avvezione di vorticità 500 hPa",
    "omega500": "omega 500 hPa",
    "wind500": "vento 500 hPa",
    "wind300": "vento 300 hPa",
    "divergence300": "divergenza 300 hPa",
    "capeMl": "ML-CAPE",
    "capeMu": "MU-CAPE",
    "cinMl": "ML-CIN",
    "shear06": "bulk shear 0–6 km",
    "updraftHelicity": "UH_MAX",
    "lpi": "LPI",
    "precipitableWater": "acqua precipitabile",
    "stormScore": "score temporalesco",
    "stormCoherence": "coerenza temporalesca interna",
    "stormContradiction": "contraddizione temporalesca interna",
    "hailIndex": "indice grandine",
    "downburstIndex": "indice downburst",
    "freezingLevel": "quota zero termico",
    "freezingRain": "diagnostica gelicidio",
    "fogIndex": "indice nebbia",
    "visibility": "visibilità",
    "foehnIndex": "indice foehn",
    "triggerIndex": "indice d’innesco",
    "bowenRatio": "rapporto di Bowen",
    "upslopeFlow": "flusso di risalita orografica",
    "seaBreezeConvergence": "convergenza di brezza",
}

ALWAYS_UNAVAILABLE = (
    "Potential vorticity e tropopausa dinamica 2 PVU",
    "SRH 0–1/0–3 km e hodograph completo",
    "DCAPE",
    "effective bulk shear e storm-relative flow",
    "separazione precipitazione convettiva/stratiforme",
    "profilo verticale completo per neve e warm nose",
    "PBL height e contenuto di acqua di nube",
)


def _finite_array(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    return array if array.ndim == 2 else None


def _temperature_c(values: Any) -> np.ndarray | None:
    array = _finite_array(values)
    if array is None:
        return None
    finite = array[np.isfinite(array)]
    if finite.size and float(np.nanmedian(finite)) > 150.0:
        return array - 273.15
    return array


def _stats(values: Any, mask: np.ndarray | None = None) -> dict[str, float | None]:
    array = _finite_array(values)
    if array is None:
        return {
            "minimum": None, "p10": None, "median": None, "mean": None,
            "p90": None, "p95": None, "maximum": None, "validPct": 0.0,
        }
    valid = np.isfinite(array)
    denominator = int(np.count_nonzero(mask)) if mask is not None else array.size
    if mask is not None:
        valid &= mask
    sample = array[valid]
    if not sample.size:
        return {
            "minimum": None, "p10": None, "median": None, "mean": None,
            "p90": None, "p95": None, "maximum": None, "validPct": 0.0,
        }
    number = lambda value: round(float(value), 2)
    return {
        "minimum": number(np.min(sample)),
        "p10": number(np.percentile(sample, 10)),
        "median": number(np.median(sample)),
        "mean": number(np.mean(sample)),
        "p90": number(np.percentile(sample, 90)),
        "p95": number(np.percentile(sample, 95)),
        "maximum": number(np.max(sample)),
        "validPct": round(100.0 * sample.size / max(denominator, 1), 1),
    }


def _coverage(values: Any, threshold: float, mask: np.ndarray) -> float | None:
    array = _finite_array(values)
    if array is None:
        return None
    valid = np.isfinite(array) & mask
    if not valid.any():
        return None
    return round(float(np.mean(array[valid] >= threshold) * 100.0), 2)


def _region_masks(latitudes: np.ndarray, longitudes: np.ndarray) -> dict[str, np.ndarray]:
    lat2d, lon2d = np.meshgrid(latitudes, longitudes, indexing="ij")
    masks = {}
    for name, (south, north, west, east) in REGIONS.items():
        masks[name] = (
            (lat2d >= south) & (lat2d <= north)
            & (lon2d >= west) & (lon2d <= east)
        )
    masks["intero dominio"] = np.ones(lat2d.shape, dtype=bool)
    return masks


def _nan_smooth(values: np.ndarray, sigma: tuple[float, float]) -> np.ndarray:
    valid = np.isfinite(values)
    if not valid.any():
        return np.full_like(values, np.nan)
    numerator = gaussian_filter(
        np.where(valid, values, 0.0), sigma=sigma, mode="nearest"
    )
    denominator = gaussian_filter(valid.astype(float), sigma=sigma, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return np.where(denominator > 0.05, result, np.nan)


def _metric_coordinates(latitudes: np.ndarray, longitudes: np.ndarray):
    y = np.deg2rad(latitudes - latitudes[0]) * EARTH_RADIUS_M
    x = (
        np.deg2rad(longitudes - longitudes[0])[None, :]
        * EARTH_RADIUS_M
        * np.cos(np.deg2rad(latitudes))[:, None]
    )
    return x, y


def _sigma_km(latitudes: np.ndarray, longitudes: np.ndarray, kilometres: float):
    dy = max(abs(float(np.nanmedian(np.diff(latitudes)))) * 111.195, 0.1)
    dx = max(
        abs(float(np.nanmedian(np.diff(longitudes))))
        * 111.195 * np.cos(np.deg2rad(float(np.nanmean(latitudes)))),
        0.1,
    )
    return kilometres / dy, kilometres / dx


def _gradient(values: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray):
    x, y = _metric_coordinates(latitudes, longitudes)
    east_index = np.gradient(values, axis=1)
    east_spacing = np.gradient(x, axis=1)
    east = np.divide(
        east_index, east_spacing,
        out=np.full_like(east_index, np.nan), where=np.abs(east_spacing) > 1.0,
    )
    north = np.gradient(values, y, axis=0)
    return east, north


def _kinematics(
    u_wind: Any,
    v_wind: Any,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    smoothing_km: float,
):
    u = _finite_array(u_wind)
    v = _finite_array(v_wind)
    if u is None or v is None or u.shape != v.shape:
        return None, None
    sigma = _sigma_km(latitudes, longitudes, smoothing_km)
    u = _nan_smooth(u, sigma)
    v = _nan_smooth(v, sigma)
    du_dx, du_dy = _gradient(u, latitudes, longitudes)
    dv_dx, dv_dy = _gradient(v, latitudes, longitudes)
    divergence = (du_dx + dv_dy) * 1.0e5
    vorticity = (dv_dx - du_dy) * 1.0e5
    return divergence, vorticity


def _scalar_advection(
    scalar: Any,
    u_wind: Any,
    v_wind: Any,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    hours: float = 3.0,
):
    field = _finite_array(scalar)
    u = _finite_array(u_wind)
    v = _finite_array(v_wind)
    if field is None or u is None or v is None:
        return None
    east, north = _gradient(field, latitudes, longitudes)
    return -(u * east + v * north) * float(hours) * 3600.0


def _extreme_location(
    values: Any,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    mode: str,
    smoothing_km: float = 0.0,
):
    array = _finite_array(values)
    if array is None or not np.isfinite(array).any():
        return None
    if smoothing_km:
        array = _nan_smooth(array, _sigma_km(latitudes, longitudes, smoothing_km))
    flat = int(np.nanargmin(array) if mode == "min" else np.nanargmax(array))
    row, column = np.unravel_index(flat, array.shape)
    return {
        "value": round(float(array[row, column]), 1),
        "latitude": round(float(latitudes[row]), 2),
        "longitude": round(float(longitudes[column]), 2),
    }


def _front_summary(fronts: dict | None) -> dict[str, Any]:
    features = (fronts or {}).get("features", []) if isinstance(fronts, dict) else []
    types = Counter()
    supports = []
    uncertainties = []
    diagnostics = []
    for feature in features:
        props = feature.get("properties") or {}
        front_type = props.get("frontType")
        if front_type:
            types[str(front_type)] += 1
        for key, target in (
            ("existenceConfidence", supports), ("uncertaintyIndex", uncertainties)
        ):
            try:
                value = float(props.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                target.append(value)
        diag = props.get("diagnostics") or {}
        if isinstance(diag, dict):
            diagnostics.append(diag)

    def median_key(*keys):
        values = []
        for item in diagnostics:
            value = None
            for key in keys:
                try:
                    candidate = float(item.get(key))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(candidate):
                    value = candidate
                    break
            if value is None:
                continue
            if np.isfinite(value):
                values.append(value)
        return round(float(np.median(values)), 2) if values else None

    return {
        "count": len(features),
        "types": dict(types),
        "existenceSupportMedian": (
            round(float(np.median(supports)) * 100.0, 1) if supports else None
        ),
        "uncertaintyMedian": (
            round(float(np.median(uncertainties)) * 100.0, 1)
            if uncertainties else None
        ),
        "frontogenesisMedian": median_key("frontogenesis"),
        "vorticityMedian1e5": median_key("vorticity1e5"),
        "convergenceMedian1e5": median_key(
            "kinematicConvergence1e5", "convergence1e5"
        ),
        "deltaThetaWMedian": median_key("deltaThetaW"),
    }


def _convective_overlap(arrays: dict[str, np.ndarray | None], mask: np.ndarray):
    """Measure co-located ingredients, never independent regional maxima."""
    cape = arrays.get("capeMl")
    cin = arrays.get("cinMl")
    if cape is None or cin is None:
        return {
            "validCells": 0, "validPct": 0.0,
            "conditionalAreaPct": None, "robustAreaPct": None,
            "scoreP95OnRobustCells": None,
        }
    rh700 = arrays.get("relativeHumidity700")
    pwat = arrays.get("precipitableWater")
    convergence = arrays.get("convergence10")
    omega = arrays.get("omega700")
    front_distance = arrays.get("frontDistanceKm")

    moisture_available = np.zeros(cape.shape, dtype=bool)
    moisture_support = np.zeros(cape.shape, dtype=bool)
    if rh700 is not None:
        moisture_available |= np.isfinite(rh700)
        moisture_support |= np.isfinite(rh700) & (rh700 >= 50.0)
    if pwat is not None:
        moisture_available |= np.isfinite(pwat)
        moisture_support |= np.isfinite(pwat) & (pwat >= 20.0)

    lift_available = np.zeros(cape.shape, dtype=np.int8)
    lift_support = np.zeros(cape.shape, dtype=np.int8)
    for field, condition in (
        (convergence, lambda value: value >= 6.0),
        (omega, lambda value: value <= -0.10),
        (front_distance, lambda value: value <= 80.0),
    ):
        if field is None:
            continue
        finite = np.isfinite(field)
        lift_available += finite.astype(np.int8)
        lift_support += (finite & condition(field)).astype(np.int8)

    valid = (
        mask & np.isfinite(cape) & np.isfinite(cin)
        & moisture_available & (lift_available >= 1)
    )
    valid_count = int(np.count_nonzero(valid))
    denominator = max(int(np.count_nonzero(mask)), 1)
    if not valid_count:
        return {
            "validCells": 0,
            "validPct": round(100.0 * np.count_nonzero(valid) / denominator, 2),
            "conditionalAreaPct": None,
            "robustAreaPct": None,
            "scoreP95OnRobustCells": None,
        }
    conditional = (
        valid & (cape >= 500.0) & (cin >= -150.0)
        & moisture_support & (lift_support >= 1)
    )
    robust = (
        valid & (cape >= 800.0) & (cin >= -75.0)
        & moisture_support & (lift_support >= 2)
    )
    score = arrays.get("stormScore")
    robust_score = (
        score[robust & np.isfinite(score)]
        if score is not None else np.asarray([], dtype=float)
    )
    return {
        "validCells": valid_count,
        "validPct": round(100.0 * valid_count / denominator, 2),
        "conditionalAreaPct": round(100.0 * np.count_nonzero(conditional) / valid_count, 2),
        "robustAreaPct": round(100.0 * np.count_nonzero(robust) / valid_count, 2),
        "scoreP95OnRobustCells": (
            round(float(np.percentile(robust_score, 95)), 1)
            if robust_score.size else None
        ),
    }


def build_synoptic_frame(
    *,
    lead_hours: int,
    valid_time: str,
    latitudes,
    longitudes,
    fronts: dict | None,
    fields: dict[str, Any],
    rain_stride: int = 8,
) -> dict[str, Any]:
    """Build one auditable multi-level evidence frame."""
    lat = np.asarray(latitudes, dtype=float).squeeze()
    lon = np.asarray(longitudes, dtype=float).squeeze()
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("coordinate sinottiche 1D richieste")
    shape = (lat.size, lon.size)
    arrays: dict[str, np.ndarray | None] = {}
    for name, values in fields.items():
        array = _finite_array(values)
        arrays[name] = array if array is not None and array.shape == shape else None

    for name in ("temperature2m", "temperature925", "temperature850", "temperature700", "temperature500"):
        arrays[name] = _temperature_c(arrays.get(name))

    for output, u_name, v_name in (
        ("wind10", "u10", "v10"),
        ("wind925", "u925", "v925"),
        ("wind850", "u850", "v850"),
        ("wind700", "u700", "v700"),
        ("wind500", "u500", "v500"),
        ("wind300", "u300", "v300"),
    ):
        u = arrays.get(u_name)
        v = arrays.get(v_name)
        arrays[output] = np.hypot(u, v) if u is not None and v is not None else None

    # Il sito presenta il vento superficiale in km/h; i livelli isobarici
    # restano in m/s, unità standard per la diagnostica dinamica.
    if arrays.get("wind10") is not None:
        arrays["wind10"] = arrays["wind10"] * 3.6
    # La convergenza arriva in s-1. Statistiche, soglie e testo usano
    # esplicitamente 10^-5 s-1.
    if arrays.get("convergence10") is not None:
        arrays["convergence10"] = arrays["convergence10"] * 1.0e5

    theta_e = arrays.get("thetaE850")
    if theta_e is not None:
        smoothed = _nan_smooth(theta_e, _sigma_km(lat, lon, 45.0))
        east, north = _gradient(smoothed, lat, lon)
        arrays["thetaEGradient850"] = np.hypot(east, north) * 100_000.0
        arrays["thermalAdvection850"] = _scalar_advection(
            smoothed, arrays.get("u850"), arrays.get("v850"), lat, lon, 3.0
        )
    else:
        arrays["thetaEGradient850"] = None
        arrays["thermalAdvection850"] = None

    divergence300, _ = _kinematics(
        arrays.get("u300"), arrays.get("v300"), lat, lon, 100.0
    )
    arrays["divergence300"] = divergence300
    _, vorticity500 = _kinematics(
        arrays.get("u500"), arrays.get("v500"), lat, lon, 80.0
    )
    arrays["vorticity500"] = vorticity500
    arrays["vorticityAdvection500"] = _scalar_advection(
        vorticity500, arrays.get("u500"), arrays.get("v500"), lat, lon, 3.0
    )
    if arrays.get("temperature700") is not None and arrays.get("temperature500") is not None:
        # 700–500 hPa mean geometric separation is around 3 km.  The result is
        # a layer-mean environmental lapse rate, explicitly diagnostic.
        arrays["lapseRate700500"] = (
            arrays["temperature700"] - arrays["temperature500"]
        ) / 3.0
    else:
        arrays["lapseRate700500"] = None

    masks = _region_masks(lat, lon)
    regions = {}
    public_names = [name for name in FIELD_LABELS if name in arrays]
    for region, mask in masks.items():
        summary = {name: _stats(arrays.get(name), mask) for name in public_names}
        rain = arrays.get("rainStep")
        summary.setdefault("rainStep", _stats(rain, mask))
        summary["rainStep"]["areaAbove01Pct"] = _coverage(rain, 0.1, mask)
        summary["rainStep"]["areaAbove10Pct"] = _coverage(rain, 1.0, mask)
        summary["convectiveOverlap"] = _convective_overlap(arrays, mask)
        regions[region] = summary

    pressure = arrays.get("pressureMsl")
    frame = {
        "leadHours": int(lead_hours),
        "validTime": valid_time,
        "regions": regions,
        "fronts": _front_summary(fronts),
        "pressureMinimum": _extreme_location(pressure, lat, lon, "min", 60.0),
        "pressureMaximum": _extreme_location(pressure, lat, lon, "max", 60.0),
        "availability": {
            name: bool(array is not None and np.isfinite(array).any())
            for name, array in arrays.items() if name in FIELD_LABELS
        },
        # Private run-building payload.  It is removed before serialization.
        "_rain": (
            arrays["rainStep"][::rain_stride, ::rain_stride].astype(np.float32)
            if arrays.get("rainStep") is not None else None
        ),
        "_rainLatitudes": lat[::rain_stride].astype(np.float32),
        "_rainLongitudes": lon[::rain_stride].astype(np.float32),
    }
    return frame


def _metric(frame: dict, region: str, field: str, statistic: str = "p95"):
    try:
        value = frame["regions"][region][field][statistic]
    except (KeyError, TypeError):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _strongest_region(frame: dict, field: str, statistic: str = "p95", mode: str = "max"):
    candidates = []
    for region in REGIONS:
        value = _metric(frame, region, field, statistic)
        if value is not None:
            candidates.append((value, region))
    if not candidates:
        return None, None
    value, region = (min(candidates) if mode == "min" else max(candidates))
    return region, value


def _component_level(value, bands):
    if value is None:
        return "dato insufficiente"
    for threshold, label in bands:
        if value >= threshold:
            return label
    return bands[-1][1]


def _convection_components(frame: dict) -> dict[str, Any]:
    overlap_candidates = []
    for candidate_region in REGIONS:
        overlap = frame["regions"][candidate_region].get("convectiveOverlap") or {}
        robust_area = overlap.get("robustAreaPct")
        conditional_area = overlap.get("conditionalAreaPct")
        if robust_area is not None or conditional_area is not None:
            overlap_candidates.append((
                float(robust_area or 0.0),
                float(conditional_area or 0.0),
                candidate_region,
            ))
    region = max(overlap_candidates)[2] if overlap_candidates else None
    score = _metric(frame, region, "stormScore", "p95") if region else None
    if region is None:
        region, _ = _strongest_region(frame, "capeMl", "p90")
    if region is None:
        return {
            "area": "non determinabile", "status": "dato insufficiente",
            "components": {}, "evidence": [], "limits": ["ML-CAPE non disponibile"],
        }

    cape = _metric(frame, region, "capeMl", "p90")
    cin = _metric(frame, region, "cinMl", "p90")
    rh700 = _metric(frame, region, "relativeHumidity700", "mean")
    pwat = _metric(frame, region, "precipitableWater", "p90")
    convergence = _metric(frame, region, "convergence10", "p95")
    omega = _metric(frame, region, "omega700", "p10")
    front_distance = _metric(frame, region, "frontDistanceKm", "p10")
    shear = _metric(frame, region, "shear06", "p90")
    uh = _metric(frame, region, "updraftHelicity", "maximum")
    lpi = _metric(frame, region, "lpi", "maximum")
    contradiction = _metric(frame, region, "stormContradiction", "mean")

    instability = _component_level(cape, (
        (2500, "molto forte"), (1500, "forte"), (800, "moderata"),
        (300, "debole"), (-1e9, "molto debole"),
    ))
    if cin is None:
        inhibition = "dato insufficiente"
    elif cin >= -35:
        inhibition = "debole"
    elif cin >= -75:
        inhibition = "moderata"
    elif cin >= -150:
        inhibition = "forte"
    else:
        inhibition = "molto forte"
    moisture_value = max(
        rh700 if rh700 is not None else -1e9,
        (pwat * 2.0) if pwat is not None else -1e9,
    )
    moisture = _component_level(moisture_value if moisture_value > -1e8 else None, (
        (80, "molto forte"), (65, "forte"), (50, "moderata"),
        (35, "debole"), (-1e9, "molto debole"),
    ))
    lift_score = 0
    if convergence is not None and convergence >= 8.0:  # 1e-5 s-1 units
        lift_score += 1
    if omega is not None and omega <= -0.15:
        lift_score += 1
    if front_distance is not None and front_distance <= 80.0:
        lift_score += 1
    lift = ("forte" if lift_score >= 3 else "moderata" if lift_score == 2
            else "debole" if lift_score == 1 else "molto debole")
    organisation = _component_level(shear, (
        (25, "molto forte"), (18, "forte"), (10, "moderata"),
        (6, "debole"), (-1e9, "molto debole"),
    ))

    pva = _metric(frame, region, "vorticityAdvection500", "p90")
    omega500 = _metric(frame, region, "omega500", "p10")
    divergence300 = _metric(frame, region, "divergence300", "p90")
    forcing_score = sum((
        pva is not None and pva > 0.0,
        omega500 is not None and omega500 <= -0.10,
        divergence300 is not None and divergence300 >= 1.0,
    ))
    forcing = (
        "forte" if forcing_score >= 3 else
        "moderata" if forcing_score == 2 else
        "debole" if forcing_score == 1 else
        "molto debole"
    )

    available_core = all(value is not None for value in (cape, cin))
    overlap = frame["regions"][region].get("convectiveOverlap") or {}
    valid_overlap = int(overlap.get("validCells") or 0)
    robust_area = float(overlap.get("robustAreaPct") or 0.0)
    conditional_area = float(overlap.get("conditionalAreaPct") or 0.0)
    joint_score = overlap.get("scoreP95OnRobustCells")
    if not available_core or not valid_overlap:
        status = "dato insufficiente"
    elif robust_area > 0.0 and joint_score is not None and joint_score >= 55 and (contradiction or 0) < 35:
        status = "segnale abbastanza robusto"
    elif conditional_area > 0.0:
        status = "scenario condizionale"
    elif cape >= 300:
        status = "segnale debole"
    else:
        status = "segnale molto debole"

    evidence = []
    if cape is not None:
        evidence.append(f"ML-CAPE p90 {cape:.0f} J/kg")
    if cin is not None:
        evidence.append(f"ML-CIN p90 {cin:.0f} J/kg")
    if convergence is not None:
        evidence.append(f"convergenza p95 {convergence:.1f}×10⁻⁵ s⁻¹")
    if omega is not None:
        evidence.append(f"omega 700 p10 {omega:.2f} Pa/s")
    if shear is not None:
        evidence.append(f"shear 0–6 km p90 {shear:.1f} m/s")
    if rh700 is not None:
        evidence.append(f"UR 700 media {rh700:.0f}%")
    if pwat is not None:
        evidence.append(f"PWAT p90 {pwat:.1f} mm")
    if uh is not None:
        evidence.append(f"UH_MAX {uh:.0f} m²/s²")
    if lpi is not None:
        evidence.append(f"LPI massimo {lpi:.1f}")
    if valid_overlap:
        evidence.append(
            f"ingredienti co-localizzati: {conditional_area:.1f}% delle celle valide; "
            f"supporto robusto {robust_area:.1f}%"
        )

    limits = []
    if cin is not None and cin < -75:
        limits.append("inibizione convettiva significativa")
    if lift_score < 2:
        limits.append("innesco non corroborato da almeno due forzanti")
    if valid_overlap and robust_area <= 0.0:
        limits.append("ingredienti forti non sovrapposti nella stessa cella")
    if moisture in {"debole", "molto debole"}:
        limits.append("umidità troposferica limitante")
    if contradiction is not None and contradiction >= 35:
        limits.append("contraddizione elevata fra gli ingredienti interni")
    return {
        "area": region,
        "status": status,
        "score": round(score, 1) if score is not None else None,
        "components": {
            "instabilità": instability,
            "umidità": moisture,
            "innesco": lift,
            "forcing dinamico": forcing,
            "shear e organizzazione": organisation,
            "inibizione": inhibition,
        },
        "evidence": evidence,
        "limits": limits,
    }


def _rain_accumulation(frames: list[dict], start: int, hours: int):
    selected = frames[start + 1:min(len(frames), start + hours + 1)]
    arrays = [frame.get("_rain") for frame in selected if frame.get("_rain") is not None]
    if not arrays:
        return None
    stack = np.stack(arrays)
    total = np.nansum(stack, axis=0)
    total = np.where(np.isfinite(stack).any(axis=0), total, np.nan)
    lat = frames[start].get("_rainLatitudes")
    lon = frames[start].get("_rainLongitudes")
    if lat is None or lon is None:
        return None
    masks = _region_masks(np.asarray(lat), np.asarray(lon))
    regions = {}
    for name, mask in masks.items():
        stats = _stats(total, mask)
        stats["areaAbove10Pct"] = _coverage(total, 10.0, mask)
        stats["areaAbove30Pct"] = _coverage(total, 30.0, mask)
        regions[name] = stats
    return {"hours": int(hours), "regions": regions}


def _confidence_label(frame: dict, convection: dict) -> str:
    core = (
        "pressureMsl", "temperature2m", "rainStep", "wind10",
        "temperature850", "thetaE850", "height500", "capeMl", "cinMl",
    )
    available = sum(bool(frame["availability"].get(name)) for name in core)
    fraction = available / len(core)
    lead = int(frame["leadHours"])
    if fraction < 0.55:
        return "dato insufficiente"
    if lead > 48:
        return "scenario plausibile"
    if convection.get("status") == "scenario condizionale":
        return "scenario condizionale"
    if fraction >= 0.85 and lead <= 18:
        return "segnale abbastanza robusto"
    return "scenario plausibile"


def _fmt_location(location: dict | None, unit: str) -> str:
    if not location:
        return "non determinabile"
    return (
        f"{location['value']:.1f} {unit} presso "
        f"{location['latitude']:.2f}°N, {location['longitude']:.2f}°E"
    )


def _front_names(types: dict) -> str:
    names = {
        "cold": "freddi", "warm": "caldi", "stationary": "stazionari",
        "occluded": "occlusi", "uncertain": "non classificati",
    }
    return ", ".join(
        f"{count} {names.get(kind, kind)}" for kind, count in sorted(types.items())
    )


def _analysis_for_frame(frames: list[dict], index: int) -> dict[str, Any]:
    frame = frames[index]
    convection = _convection_components(frame)
    domain = frame["regions"]["intero dominio"]
    pmin = frame.get("pressureMinimum")
    pmax = frame.get("pressureMaximum")
    pressure_spread = None
    if pmin and pmax:
        pressure_spread = pmax["value"] - pmin["value"]
    wind_p95 = (domain.get("wind10") or {}).get("p95")
    front = frame["fronts"]

    synoptic_parts = []
    if pmin and pmax:
        synoptic_parts.append(
            "La MSLP filtrata su scala sinottica presenta un minimo relativo di "
            + _fmt_location(pmin, "hPa") + " e un massimo relativo di "
            + _fmt_location(pmax, "hPa") + "."
        )
        if pressure_spread is not None and pressure_spread >= 8 and wind_p95 is not None:
            synoptic_parts.append(
                f"Il gradiente di {pressure_spread:.1f} hPa è coerente con un vento "
                f"al 95° percentile di {wind_p95:.0f} km/h sul dominio."
            )
        else:
            synoptic_parts.append(
                "Il solo estremo barico non viene interpretato come ciclone o "
                "perturbazione attiva senza corroborazione dinamica."
            )
    else:
        synoptic_parts.append("MSLP non disponibile: configurazione barica non determinabile.")
    if front.get("count"):
        synoptic_parts.append(
            f"L’analisi frontale multilivello pubblica {front['count']} strutture "
            f"({_front_names(front.get('types', {}))}); supporto d’esistenza mediano "
            f"{(front.get('existenceSupportMedian') or 0.0):.0f}/100 e incertezza interna "
            f"{(front.get('uncertaintyMedian') or 0.0):.0f}/100."
        )
    else:
        synoptic_parts.append("Nessun fronte sinottico supera i controlli multilivello in questa scadenza.")

    zmin = (domain.get("height500") or {}).get("minimum")
    zmax = (domain.get("height500") or {}).get("maximum")
    pva_region, pva = _strongest_region(frame, "vorticityAdvection500", "p95")
    omega_region, omega = _strongest_region(frame, "omega500", "p10", "min")
    jet_region, jet = _strongest_region(frame, "wind300", "p95")
    div_region, upper_div = _strongest_region(frame, "divergence300", "p95")
    upper_parts = []
    if zmin is not None and zmax is not None:
        upper_parts.append(
            f"A 500 hPa il geopotenziale varia da {zmin:.0f} a {zmax:.0f} m."
        )
    else:
        upper_parts.append("Geopotenziale a 500 hPa non disponibile.")
    if pva is not None:
        upper_parts.append(
            f"La PVA più marcata ricade su {pva_region} ({pva:.2f} unità diagnostiche/3 h)."
        )
    else:
        upper_parts.append("Avvezione di vorticità a 500 hPa non determinabile.")
    if omega is not None:
        upper_parts.append(
            f"Il sollevamento a 500 hPa raggiunge il 10° percentile di {omega:.2f} Pa/s su {omega_region}."
        )
    if jet is not None:
        upper_parts.append(
            f"Vento a 300 hPa al 95° percentile di {jet:.1f} m/s su {jet_region}"
            + (f", con divergenza massima regionale {upper_div:.1f}×10⁻⁵ s⁻¹ su {div_region}." if upper_div is not None else ".")
        )
    else:
        upper_parts.append("Vento e divergenza a 300 hPa non disponibili: jet streak non classificabile.")

    gradient_region, gradient = _strongest_region(frame, "thetaEGradient850", "p95")
    warm_region, warm_adv = _strongest_region(frame, "thermalAdvection850", "p90")
    cold_region, cold_adv = _strongest_region(frame, "thermalAdvection850", "p10", "min")
    low_parts = []
    if gradient is not None:
        low_parts.append(
            f"Il gradiente di theta-e a 850 hPa è massimo su {gradient_region} "
            f"(p95 {gradient:.1f} K/100 km)."
        )
    else:
        low_parts.append("Theta-e e gradiente a 850 hPa non disponibili.")
    if warm_adv is not None and cold_adv is not None:
        low_parts.append(
            f"Advezione calda più marcata su {warm_region} ({warm_adv:.1f} K/3 h) "
            f"e fredda su {cold_region} ({cold_adv:.1f} K/3 h)."
        )
    if front.get("frontogenesisMedian") is not None:
        low_parts.append(
            f"Sulle linee frontali la frontogenesi mediana è {front['frontogenesisMedian']:.2f} "
            "K/(100 km)/(3 h), valutata insieme a convergenza, rotazione del vento e struttura 925–700 hPa."
        )

    components = ", ".join(
        f"{name} {value}" for name, value in convection.get("components", {}).items()
    )
    conv_parts = [
        f"Area con segnale convettivo più significativo: {convection['area']}. "
        f"Valutazione: {convection['status']}."
    ]
    if components:
        conv_parts.append("Scomposizione degli ingredienti: " + components + ".")
    if convection.get("limits"):
        conv_parts.append("Fattori limitanti: " + "; ".join(convection["limits"]) + ".")
    conv_parts.append(
        "UH_MAX, LPI e indice proprietario sono diagnostiche del singolo run: "
        "non equivalgono a SRH, osservazioni o probabilità calibrate."
    )

    accumulations = {}
    precip_parts = []
    for hours in (3, 6, 12, 24):
        accumulation = _rain_accumulation(frames, index, hours)
        if accumulation is None:
            continue
        accumulations[str(hours)] = accumulation
        candidates = []
        for region in REGIONS:
            value = accumulation["regions"][region].get("maximum")
            if value is not None:
                candidates.append((value, region))
        if candidates:
            maximum, region = max(candidates)
            precip_parts.append(
                f"Accumulo massimo modellistico nelle successive {hours} h: "
                f"{maximum:.1f} mm su {region}."
            )
    if not precip_parts:
        precip_parts.append("Accumuli temporali non determinabili dai dati disponibili.")
    precip_parts.append(
        "Gli accumuli modellistici non costituiscono da soli una valutazione del rischio idrogeologico."
    )

    significant = []
    gust_region, gust = _strongest_region(frame, "gust10", "p95")
    hail_region, hail = _strongest_region(frame, "hailIndex", "p95")
    down_region, down = _strongest_region(frame, "downburstIndex", "p95")
    fog_region, fog = _strongest_region(frame, "fogIndex", "p95")
    freeze_region, freeze = _strongest_region(frame, "freezingRain", "p95")
    visibility_region, minimum_visibility = _strongest_region(
        frame, "visibility", "p10", "min"
    )
    foehn_region, foehn = _strongest_region(frame, "foehnIndex", "p95")
    freezing_level_region, freezing_level = _strongest_region(
        frame, "freezingLevel", "p10", "min"
    )
    upslope_region, upslope = _strongest_region(frame, "upslopeFlow", "p95")
    breeze_region, breeze = _strongest_region(
        frame, "seaBreezeConvergence", "p95"
    )
    if gust is not None and gust >= 55:
        significant.append({
            "area": gust_region, "phenomenon": "vento/raffiche forti",
            "period": frame["validTime"], "confidence": "segnale abbastanza robusto",
            "reason": f"raffica ICON-2I p95 {gust:.0f} km/h",
        })
    if hail is not None and hail >= 55 and convection["status"] not in {"dato insufficiente", "segnale molto debole"}:
        significant.append({
            "area": hail_region, "phenomenon": "grandine condizionata alla presenza di celle",
            "period": frame["validTime"], "confidence": "scenario condizionale",
            "reason": f"indice grandine p95 {hail:.0f}/100 con ingredienti convettivi concomitanti",
        })
    if down is not None and down >= 55 and convection["status"] not in {"dato insufficiente", "segnale molto debole"}:
        significant.append({
            "area": down_region, "phenomenon": "raffiche convettive/downburst",
            "period": frame["validTime"], "confidence": "scenario condizionale",
            "reason": f"indice downburst p95 {down:.0f}/100, condizionato all’innesco",
        })
    if fog is not None and fog >= 65:
        significant.append({
            "area": fog_region, "phenomenon": "nebbia o forte riduzione di visibilità",
            "period": frame["validTime"], "confidence": "scenario plausibile",
            "reason": f"indice nebbia p95 {fog:.0f}/100 da UR, vento e copertura nuvolosa",
        })
    if freeze is not None and freeze >= 50:
        significant.append({
            "area": freeze_region, "phenomenon": "gelicidio",
            "period": frame["validTime"], "confidence": "scenario condizionale",
            "reason": "warm nose 925/850/700 hPa, superficie sottozero e precipitazione concomitante",
        })
    rain24 = accumulations.get("24")
    if rain24:
        rain_candidates = [
            (rain24["regions"][region].get("maximum"), region)
            for region in REGIONS
            if rain24["regions"][region].get("maximum") is not None
        ]
        if rain_candidates:
            rain_max, rain_region = max(rain_candidates)
            if rain_max >= 40:
                significant.append({
                    "area": rain_region, "phenomenon": "precipitazioni localmente abbondanti",
                    "period": "successive 24 ore", "confidence": "scenario plausibile",
                    "reason": f"accumulo massimo ICON-2I {rain_max:.1f} mm/24 h; rischio idrologico non valutato",
                })

    boundary_parts = []
    if minimum_visibility is not None:
        boundary_parts.append(
            f"Visibilità al 10° percentile {minimum_visibility:.0f} m su "
            f"{visibility_region}; indice nebbia p95 "
            f"{fog:.0f}/100." if fog is not None else
            f"Visibilità al 10° percentile {minimum_visibility:.0f} m su "
            f"{visibility_region}; indice nebbia non disponibile."
        )
    else:
        boundary_parts.append("Visibilità e diagnostica di nebbia non disponibili.")
    if freezing_level is not None:
        boundary_parts.append(
            f"Quota dello zero termico al 10° percentile {freezing_level:.0f} m "
            f"su {freezing_level_region}."
        )
    if freeze is not None:
        boundary_parts.append(
            f"Diagnostica di gelicidio p95 {freeze:.0f}/100 su {freeze_region}; "
            "la neve non viene classificata senza profilo wet-bulb completo."
        )
    else:
        boundary_parts.append(
            "Fase invernale non determinabile: manca un profilo verticale "
            "completo di temperatura e bulbo umido."
        )
    if foehn is not None and foehn > 0:
        boundary_parts.append(
            f"Segnale foehn p95 {foehn:.0f}/100 su {foehn_region}, derivato da "
            "flusso trasversale a 700 hPa, gradiente barico alpino e aria secca sottovento."
        )
    else:
        boundary_parts.append(
            "Nessun segnale foehn significativo nei campi disponibili alla scadenza."
        )
    if upslope is not None and upslope >= 0.3:
        boundary_parts.append(
            f"Risalita orografica p95 {upslope:.2f} m/s su {upslope_region}."
        )
    if breeze is not None and breeze >= 1.0:
        boundary_parts.append(
            f"Convergenza di brezza p95 {breeze:.1f}×10⁻⁵ s⁻¹ su {breeze_region}; "
            "è una possibile forzante locale, non forcing sinottico."
        )

    evolution = []
    for horizon in (6, 12, 24, 48, 72):
        target = next(
            (item for item in frames if item["leadHours"] == frame["leadHours"] + horizon),
            None,
        )
        if target is None:
            evolution.append({"hours": horizon, "status": "non disponibile"})
            continue
        target_conv = _convection_components(target)
        target_p = target.get("pressureMinimum")
        evolution.append({
            "hours": horizon,
            "status": "disponibile",
            "validTime": target["validTime"],
            "minimumPressureHpa": target_p.get("value") if target_p else None,
            "frontCount": target["fronts"].get("count", 0),
            "convectiveSignal": target_conv["status"],
            "convectiveArea": target_conv["area"],
        })

    evolution_text = []
    for item in evolution:
        if item["status"] != "disponibile":
            evolution_text.append(f"+{item['hours']} h: orizzonte non disponibile dal valid time selezionato.")
            continue
        pressure_text = (
            f"minimo MSLP {item['minimumPressureHpa']:.1f} hPa; "
            if item.get("minimumPressureHpa") is not None else "MSLP non disponibile; "
        )
        evolution_text.append(
            f"+{item['hours']} h: {pressure_text}{item['frontCount']} fronti pubblicati; "
            f"segnale convettivo {item['convectiveSignal']} su {item['convectiveArea']}."
        )

    unavailable = [
        FIELD_LABELS[name]
        for name, available in frame["availability"].items()
        if not available and name in FIELD_LABELS
    ] + list(ALWAYS_UNAVAILABLE)
    confidence = _confidence_label(frame, convection)
    uncertainty_text = [
        f"Classificazione complessiva del segnale: {confidence}. Il lead time è +{frame['leadHours']} h.",
        "Il prodotto usa un solo run deterministico ICON-2I: non misura la dispersione di un ensemble e non fornisce percentuali di affidabilità.",
    ]
    if unavailable:
        uncertainty_text.append("Diagnostiche non disponibili: " + "; ".join(unavailable) + ".")
    if convection.get("limits"):
        uncertainty_text.append("Contraddizioni o limiti convettivi: " + "; ".join(convection["limits"]) + ".")

    why = []
    if convection["status"] not in {"dato insufficiente", "segnale molto debole"}:
        why.append({
            "phenomenon": "Sviluppo convettivo",
            "status": convection["status"],
            "area": convection["area"],
            "evidence": convection["evidence"],
            "limitingFactors": convection["limits"],
            "conclusion": (
                "Ambiente favorevole all’innesco soltanto dove instabilità, umidità e sollevamento coincidono."
                if convection["status"] == "scenario condizionale"
                else "Concordanza multilivello sufficiente per un segnale convettivo strutturato nel singolo run."
            ),
        })
    if pva is not None and omega is not None and pva > 0 and omega < -0.1:
        why.append({
            "phenomenon": "Sollevamento dinamico",
            "status": "segnale abbastanza robusto" if pva_region == omega_region else "scenario plausibile",
            "area": pva_region if pva_region == omega_region else f"{pva_region}/{omega_region}",
            "evidence": [
                f"PVA 500 hPa {pva:.2f} unità diagnostiche/3 h",
                f"omega 500 hPa {omega:.2f} Pa/s",
            ],
            "limitingFactors": ([] if pva_region == omega_region else ["massimi non perfettamente sovrapposti"]),
            "conclusion": "PVA e moto ascendente forniscono una catena dinamica coerente; non vengono interpretati isolatamente.",
        })

    operational = []
    if significant:
        first = significant[0]
        operational.append(
            f"Nel periodo centrato su {frame['validTime']}, il segnale più rilevante riguarda "
            f"{first['phenomenon']} su {first['area']}: {first['reason']}."
        )
    else:
        operational.append(
            f"Per la scadenza {frame['validTime']} non emergono fenomeni significativi "
            "sufficientemente corroborati dai campi disponibili."
        )
    if convection["status"] not in {"dato insufficiente", "segnale molto debole"}:
        operational.append(
            f"Su {convection['area']} il quadro convettivo è classificato come {convection['status']}; "
            + ("i principali limiti sono " + ", ".join(convection["limits"]) + "." if convection["limits"] else "gli ingredienti principali risultano concordi nel singolo run.")
        )

    return {
        "leadHours": frame["leadHours"],
        "validTime": frame["validTime"],
        "signalConfidence": confidence,
        "sections": [
            {"id": "synoptic", "title": "Analisi sinottica", "paragraphs": synoptic_parts},
            {"id": "upper", "title": "Dinamica in quota", "paragraphs": upper_parts},
            {"id": "low", "title": "Bassa troposfera", "paragraphs": low_parts},
            {"id": "convection", "title": "Stabilità e convezione", "paragraphs": conv_parts, "components": convection.get("components", {})},
            {"id": "precipitation", "title": "Precipitazioni", "paragraphs": precip_parts},
            {"id": "boundary", "title": "Strato limite, fase e orografia", "paragraphs": boundary_parts},
            {"id": "significant", "title": "Fenomeni significativi", "items": significant, "paragraphs": ([] if significant else ["Nessun fenomeno significativo soddisfa contemporaneamente i criteri fisici richiesti."])},
            {"id": "evolution", "title": "Evoluzione", "paragraphs": evolution_text},
            {"id": "uncertainty", "title": "Incertezze", "paragraphs": uncertainty_text},
        ],
        "why": why,
        "operationalSummary": operational,
        "unavailableInputs": unavailable,
        "evolution": evolution,
        "rainAccumulations": accumulations,
    }


def generate_run_bulletin(
    frames: list[dict[str, Any]],
    *,
    run_time: str,
    model: str = "ICON-2I",
    area: str = "Italia e dominio ICON-2I",
    spatial_resolution_km: float = 2.2,
    temporal_resolution_hours: int = 1,
) -> dict[str, Any]:
    """Generate timeline-linked technical and operational bulletins."""
    ordered = sorted(frames, key=lambda item: item["leadHours"])
    if not ordered:
        raise ValueError("nessun frame sinottico disponibile")
    analyses = [_analysis_for_frame(ordered, index) for index in range(len(ordered))]
    clean_frames = []
    for frame in ordered:
        clean_frames.append({key: value for key, value in frame.items() if not key.startswith("_")})
    return {
        "schemaVersion": 1,
        "method": ENGINE_METHOD,
        "model": model,
        "runTime": run_time,
        "area": area,
        "spatialResolutionKm": spatial_resolution_km,
        "temporalResolutionHours": temporal_resolution_hours,
        "forecastHours": [int(item["leadHours"]) for item in ordered],
        "forecastHorizonHours": int(ordered[-1]["leadHours"]),
        "semantics": {
            "source": "single-deterministic-nwp-run",
            "confidence": "qualitative-internal-evidence-not-forecast-skill",
            "stormScore": "diagnostic-not-calibrated-probability",
            "missingData": "never-imputed-with-climatology",
        },
        "analyses": analyses,
        "evidenceFrames": clean_frames,
        "disclaimer": (
            "Analisi automatica deterministica ICON-2I: non è un’allerta, non "
            "sostituisce un previsore e non rappresenta la dispersione di un ensemble."
        ),
    }
