"""AI Observation Layer: a coherent, aggregated observational snapshot for
a geographic area, meant to be consumed by the Meteorologist AI instead of
raw per-station JSON (requirement 22).

Design rules enforced here:

* Nothing is invented. A variable missing from every nearby station stays
  absent from the output rather than being filled with a guess or a zero.
* An area aggregate (mean/min/max across stations) is a *summary
  statistic*, not a spatial reconstruction: it is tagged
  ``dataType = "observation_summary"``, distinct from
  :data:`meteo_analysis.observations.model.DATA_TYPE_ANALYSIS`, which is
  reserved for a genuine gridded/interpolated field (requirement 6/23 — do
  not blur the two).
* Every figure keeps its provenance (station id, source, observed_at,
  quality) so the AI can cite where a number came from (requirement 27).
* Confidence is computed from station density and observed quality, never
  invented by a downstream language model (requirement 26).
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

DATA_TYPE_OBSERVATION_SUMMARY = "observation_summary"

# Requirement 22's explicit list of derived quantities the AI should be
# able to ask for; anything not computable from present data stays absent.
_TREND_WINDOWS_MINUTES = {"1h": 60, "3h": 180, "6h": 360}
_TREND_TOLERANCE_MINUTES = 20


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return radius * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _epoch(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _station_mean_quality(station: dict[str, Any]) -> float | None:
    scores = [
        measurement.get("qualityScore")
        for measurement in (station.get("observations") or {}).values()
        if isinstance(measurement, dict) and isinstance(measurement.get("qualityScore"), (int, float))
    ]
    return round(sum(scores) / len(scores), 3) if scores else None


def _station_snapshot(station: dict[str, Any], *, distance_km: float, now_epoch: float) -> dict[str, Any]:
    health = station.get("health") or {}
    observations: dict[str, Any] = {}
    for variable, measurement in (station.get("observations") or {}).items():
        if not isinstance(measurement, dict) or measurement.get("value") is None:
            continue
        observed_at = _epoch(measurement.get("observedAt"))
        observations[variable] = {
            "value": measurement.get("value"),
            "unit": measurement.get("canonicalUnit"),
            "observedAt": measurement.get("observedAt"),
            "ageMinutes": (
                round((now_epoch - observed_at) / 60.0, 1)
                if observed_at is not None else None
            ),
            "qualityFlag": measurement.get("qualityFlags"),
            "qualityScore": measurement.get("qualityScore"),
        }
    return {
        "id": (
            station["sourceStationId"] if station.get("source") == "metar"
            else f"{station.get('source')}:{station.get('sourceStationId')}"
        ),
        "name": station.get("name"),
        "source": station.get("source"),
        "networkType": station.get("networkType"),
        "distanceKm": round(distance_km, 2),
        "elevationM": station.get("elevationM"),
        "health": health.get("status"),
        "qualityScore": _station_mean_quality(station),
        "observations": observations,
    }


def _nearby_stations(
    stations: list[dict[str, Any]],
    *,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    now: datetime,
) -> list[tuple[dict[str, Any], float]]:
    now_epoch = now.timestamp()
    nearby: list[tuple[dict[str, Any], float]] = []
    for station in stations:
        if station.get("duplicateOfStationId"):
            continue
        lat, lon = station.get("lat"), station.get("lon")
        if lat is None or lon is None:
            continue
        distance = _haversine_km(center_lat, center_lon, float(lat), float(lon))
        if distance <= radius_km:
            nearby.append((station, distance))
    nearby.sort(key=lambda pair: pair[1])
    return nearby


def _variable_values_at(
    stations: list[dict[str, Any]], variable: str,
) -> dict[str, float]:
    """Map ``station key -> value`` for stations currently reporting ``variable``."""

    values: dict[str, float] = {}
    for station in stations:
        measurement = (station.get("observations") or {}).get(variable)
        if isinstance(measurement, dict) and measurement.get("value") is not None:
            key = f"{station.get('source')}:{station.get('sourceStationId')}"
            values[key] = float(measurement["value"])
    return values


def _aggregate_variable(
    nearby: list[tuple[dict[str, Any], float]], variable: str,
) -> dict[str, Any] | None:
    contributors: list[tuple[dict[str, Any], float]] = []
    for station, distance in nearby:
        measurement = (station.get("observations") or {}).get(variable)
        if isinstance(measurement, dict) and measurement.get("value") is not None:
            contributors.append((station, distance))
    if not contributors:
        return None
    weighted_sum = 0.0
    weight_total = 0.0
    values: list[float] = []
    provenance: list[dict[str, Any]] = []
    for station, distance in contributors:
        measurement = station["observations"][variable]
        value = float(measurement["value"])
        values.append(value)
        quality = measurement.get("qualityScore")
        weight = float(quality) if isinstance(quality, (int, float)) else 0.5
        weighted_sum += value * weight
        weight_total += weight
        provenance.append({
            "stationId": (
                station["sourceStationId"] if station.get("source") == "metar"
                else f"{station.get('source')}:{station.get('sourceStationId')}"
            ),
            "source": station.get("source"),
            "value": value,
            "observedAt": measurement.get("observedAt"),
            "distanceKm": round(distance, 2),
        })
    mean_value = weighted_sum / weight_total if weight_total > 0 else sum(values) / len(values)
    return {
        "dataType": DATA_TYPE_OBSERVATION_SUMMARY,
        "mean": round(mean_value, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "stationCount": len(values),
        "provenance": provenance,
    }


def _trend_for_variable(
    current_nearby: list[tuple[dict[str, Any], float]],
    variable: str,
    history_stations_by_offset: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Average change over each requested window, using only stations that
    have both a current and a past reading (requirement: real trend, not a
    guess). ``insufficient_data`` when no historical snapshot is supplied
    for that window."""

    current_values = _variable_values_at([s for s, _ in current_nearby], variable)
    trend: dict[str, Any] = {}
    for label in _TREND_WINDOWS_MINUTES:
        past_stations = history_stations_by_offset.get(label)
        if not past_stations:
            trend[label] = "insufficient_data"
            continue
        past_values = _variable_values_at(past_stations, variable)
        deltas = [
            current_values[key] - past_values[key]
            for key in current_values
            if key in past_values
        ]
        trend[label] = round(sum(deltas) / len(deltas), 2) if deltas else "insufficient_data"
    return trend


def _confidence(nearby_count: int, mean_quality: float | None) -> dict[str, Any]:
    """Deterministic confidence heuristic (requirement 26): density of
    stations plus their average measured quality, never an LLM guess.
    Explicitly documented as a first-order heuristic pending validation
    against real verification statistics (see ``docs/rete_stazioni.md``)."""

    density_score = min(1.0, nearby_count / 5.0)
    quality_score = mean_quality if mean_quality is not None else 0.5
    score = round(0.5 * density_score + 0.5 * quality_score, 2)
    if nearby_count == 0:
        level = "insufficient_data"
    elif score >= 0.75:
        level = "high"
    elif score >= 0.45:
        level = "medium"
    else:
        level = "low"
    return {"score": score, "level": level, "stationCount": nearby_count}


def build_area_briefing(
    stations: list[dict[str, Any]],
    *,
    center_lat: float,
    center_lon: float,
    radius_km: float = 25.0,
    variables: tuple[str, ...] = (
        "temperature", "dewpoint", "relativeHumidity", "pressureMsl",
        "windSpeed", "windGust", "precipitation",
    ),
    history: dict[str, list[dict[str, Any]]] | None = None,
    now: datetime | None = None,
    max_stations: int = 25,
) -> dict[str, Any]:
    """Build one structured, area-scoped observational briefing.

    ``history`` maps a trend-window label (``"1h"``, ``"3h"``, ``"6h"``) to
    the registry ``stations`` list captured at that offset in the past (for
    example read from ``observation_archive/``). Omitted or empty windows
    simply yield ``insufficient_data`` for that lead — no value is
    fabricated.
    """

    now = now or datetime.now(timezone.utc)
    nearby = _nearby_stations(
        stations, center_lat=center_lat, center_lon=center_lon,
        radius_km=radius_km, now=now,
    )
    now_epoch = now.timestamp()

    station_snapshots = [
        _station_snapshot(station, distance_km=distance, now_epoch=now_epoch)
        for station, distance in nearby[:max_stations]
    ]

    quality_scores = [
        _station_mean_quality(station) for station, _ in nearby
    ]
    quality_scores = [score for score in quality_scores if score is not None]
    mean_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None

    variables_out: dict[str, Any] = {}
    for variable in variables:
        aggregate = _aggregate_variable(nearby, variable)
        if aggregate is None:
            variables_out[variable] = "insufficient_data"
            continue
        aggregate["trend"] = _trend_for_variable(
            nearby, variable, history or {},
        )
        variables_out[variable] = aggregate

    return {
        "schemaVersion": 1,
        "generatedAt": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "area": {
            "centerLat": center_lat,
            "centerLon": center_lon,
            "radiusKm": radius_km,
        },
        "stationCount": len(nearby),
        "stations": station_snapshots,
        "variables": variables_out,
        "confidence": _confidence(len(nearby), mean_quality),
    }
