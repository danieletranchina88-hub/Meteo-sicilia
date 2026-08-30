"""Deterministic forecast verification against time-matched observations."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 1
LEAD_BUCKETS = (
    (0, 6, "0-6h"),
    (7, 24, "7-24h"),
    (25, 48, "25-48h"),
    (49, 72, "49-72h"),
)


def _timestamp(value: Any) -> int | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        seconds = float(value)
        if seconds > 10_000_000_000:
            seconds /= 1000.0
        return int(seconds)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp())


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _rh_from_temperature_dewpoint(temp_c, dewpoint_c) -> float | None:
    temperature = _number(temp_c)
    dewpoint = _number(dewpoint_c)
    if temperature is None or dewpoint is None:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        numerator = math.exp(17.625 * dewpoint / (243.04 + dewpoint))
        denominator = math.exp(17.625 * temperature / (243.04 + temperature))
    return min(100.0, max(0.0, 100.0 * numerator / denominator))


def _wind_components(speed_kmh, direction_deg):
    speed = _number(speed_kmh)
    direction = _number(direction_deg)
    if speed is None or direction is None:
        return None, None
    radians = math.radians(direction % 360.0)
    speed_ms = speed / 3.6
    # Meteorological direction is the direction the wind comes from.
    return -speed_ms * math.sin(radians), -speed_ms * math.cos(radians)


def _observation_index(snapshots: Iterable[dict[str, Any]]):
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for snapshot in snapshots:
        for station in snapshot.get("stations") or []:
            station_id = str(station.get("id") or "").strip().upper()
            observed_at = _timestamp(station.get("obsTime"))
            if not station_id or observed_at is None:
                continue
            unique[(station_id, observed_at)] = station
    result: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for (station_id, observed_at), station in unique.items():
        result.setdefault(station_id, []).append((observed_at, station))
    for entries in result.values():
        entries.sort(key=lambda item: item[0])
    return result


def _closest(entries, valid_at, tolerance_seconds):
    best = None
    best_delta = tolerance_seconds + 1
    for observed_at, station in entries:
        delta = abs(observed_at - valid_at)
        if delta < best_delta:
            best = (observed_at, station)
            best_delta = delta
    return best if best is not None and best_delta <= tolerance_seconds else None


def _metric(errors: list[float]) -> dict[str, Any]:
    if not errors:
        return {"count": 0, "bias": None, "mae": None, "rmse": None}
    values = np.asarray(errors, dtype=float)
    return {
        "count": int(values.size),
        "bias": round(float(np.mean(values)), 4),
        "mae": round(float(np.mean(np.abs(values))), 4),
        "rmse": round(float(np.sqrt(np.mean(values ** 2))), 4),
    }


def _lead_bucket(lead_hours: int) -> str:
    for lower, upper, name in LEAD_BUCKETS:
        if lower <= lead_hours <= upper:
            return name
    return "outside-0-72h"


def verify_station_forecasts(
    forecast: dict[str, Any],
    observation_snapshots: Iterable[dict[str, Any]],
    *,
    tolerance_minutes: int = 45,
) -> dict[str, Any]:
    """Verify native-grid station samples with strict time matching.

    Errors are always ``forecast - observation``.  No spatial or temporal
    observation filling is performed.  ICON MSLP is compared only to METAR
    ``seaLevelPressureHpa``; altimeter settings are explicitly excluded.
    """

    snapshots = list(observation_snapshots)
    observations = _observation_index(snapshots)
    stations = forecast.get("stations") or []
    times = forecast.get("times") or []
    fields = forecast.get("fields") or {}
    tolerance_seconds = int(tolerance_minutes) * 60
    errors: dict[str, list[float]] = {}
    bucket_errors: dict[str, dict[str, list[float]]] = {}
    matched_pairs = 0
    matched_stations = set()

    def forecast_value(name, station_index, time_index):
        try:
            value = fields[name]["valuesByStation"][station_index][time_index]
        except (KeyError, IndexError, TypeError):
            return None
        return _number(value)

    for station_index, station_meta in enumerate(stations):
        station_id = str(station_meta.get("id") or "").strip().upper()
        entries = observations.get(station_id, [])
        if not entries:
            continue
        for time_index, time_meta in enumerate(times):
            valid_at = _timestamp(time_meta.get("validTime"))
            if valid_at is None:
                continue
            closest = _closest(entries, valid_at, tolerance_seconds)
            if closest is None:
                continue
            _, observed = closest
            lead = int(time_meta.get("leadHours") or 0)
            bucket = _lead_bucket(lead)
            pair_errors: dict[str, float] = {}

            mappings = (
                ("temperature2m", "tempC", "temperature2m"),
                ("dewpoint2m", "dewpC", "dewpoint2m"),
                ("pressureMsl", "seaLevelPressureHpa", "pressureMsl"),
            )
            for forecast_name, observation_name, metric_name in mappings:
                predicted = forecast_value(forecast_name, station_index, time_index)
                actual = _number(observed.get(observation_name))
                if predicted is not None and actual is not None:
                    pair_errors[metric_name] = predicted - actual

            predicted_rh = forecast_value(
                "relativeHumidity2m", station_index, time_index
            )
            actual_rh = _rh_from_temperature_dewpoint(
                observed.get("tempC"), observed.get("dewpC")
            )
            if predicted_rh is not None and actual_rh is not None:
                pair_errors["relativeHumidity2m"] = predicted_rh - actual_rh

            predicted_u = forecast_value("windU10", station_index, time_index)
            predicted_v = forecast_value("windV10", station_index, time_index)
            observed_u, observed_v = _wind_components(
                observed.get("wspdKmh"), observed.get("wdir")
            )
            if predicted_u is not None and observed_u is not None:
                pair_errors["windU10"] = predicted_u - observed_u
            if predicted_v is not None and observed_v is not None:
                pair_errors["windV10"] = predicted_v - observed_v
            if (
                predicted_u is not None
                and predicted_v is not None
                and observed.get("wspdKmh") is not None
            ):
                predicted_speed = math.hypot(predicted_u, predicted_v)
                actual_speed = float(observed["wspdKmh"]) / 3.6
                pair_errors["windSpeed10"] = predicted_speed - actual_speed

            if not pair_errors:
                continue
            matched_pairs += 1
            matched_stations.add(station_id)
            for name, error in pair_errors.items():
                errors.setdefault(name, []).append(error)
                bucket_errors.setdefault(bucket, {}).setdefault(name, []).append(error)

    return {
        "schemaVersion": SCHEMA_VERSION,
        "model": forecast.get("model"),
        "runTime": forecast.get("runTime"),
        "reference": "NOAA-AWC-METAR-time-matched",
        "errorConvention": "forecast-minus-observation",
        "timeToleranceMinutes": int(tolerance_minutes),
        "matchedForecastStationTimes": matched_pairs,
        "matchedStationCount": len(matched_stations),
        "observationSnapshotCount": len(snapshots),
        "pressurePolicy": "MSLP-compared-only-with-METAR-SLP-never-altimeter",
        "representativeness": (
            "Point observations versus a 2.2-km model grid; scores include "
            "unresolved exposure and elevation differences."
        ),
        "metrics": {name: _metric(values) for name, values in sorted(errors.items())},
        "metricsByLeadBucket": {
            bucket: {
                name: _metric(values)
                for name, values in sorted(bucket_errors.get(bucket, {}).items())
            }
            for _, _, bucket in LEAD_BUCKETS
        },
    }
