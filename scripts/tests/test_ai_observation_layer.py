#!/usr/bin/env python3
"""Regression tests for the AI Observation Layer (requirement 22).

No live network calls: stations are built by hand and run through the real
registry/QC/health pipeline stages so the briefing is exercised against the
same shapes it will see in production.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from meteo_analysis.observations.ai_layer import (  # noqa: E402
    DATA_TYPE_OBSERVATION_SUMMARY,
    build_area_briefing,
)
from meteo_analysis.observations.health import annotate_health  # noqa: E402
from meteo_analysis.observations.model import DATA_TYPE_OBSERVATION, empty_station  # noqa: E402
from meteo_analysis.observations.quality import apply_quality_control  # noqa: E402


def _station(source, station_id, name, lat, lon, elevation=10.0):
    return empty_station(
        source=source, source_station_id=station_id, name=name,
        lat=lat, lon=lon, elevation_m=elevation,
    )


def _measurement(value, observed_at_epoch, canonical_unit="degC"):
    return {
        "value": value,
        "rawValue": value,
        "rawUnit": canonical_unit,
        "canonicalUnit": canonical_unit,
        "observedAt": observed_at_epoch,
        "dataType": DATA_TYPE_OBSERVATION,
        "qualityFlag": None,
    }


def _prepared_stations(now: datetime) -> list[dict]:
    now_epoch = now.timestamp()

    palermo = _station("metar", "LICJ", "Palermo Punta Raisi", 38.18, 13.10, elevation=21)
    palermo["observations"] = {
        "temperature": _measurement(27.4, now_epoch - 300),
        "pressureMsl": _measurement(1013.2, now_epoch - 300, "hPa"),
    }

    trapani = _station("italiameteo", "IM777", "Trapani Birgi", 37.91, 12.49, elevation=6)
    trapani["observations"] = {
        "temperature": _measurement(26.8, now_epoch - 600),
    }

    # Far outside any reasonable Sicily-area radius: must never leak in.
    milano = _station("metar", "LIML", "Milano Linate", 45.45, 9.28, elevation=103)
    milano["observations"] = {"temperature": _measurement(19.0, now_epoch - 300)}

    # Broken sensor: physically impossible value, must be scored 0 and
    # excluded from the mean by construction of a very low weight.
    broken = _station("meteonetwork", "MN999", "Stazione guasta", 38.05, 13.30, elevation=50)
    broken["observations"] = {"temperature": _measurement(999.0, now_epoch - 120)}

    stations = [palermo, trapani, milano, broken]
    apply_quality_control(stations, now=now)
    annotate_health(stations, now=now)
    return stations


def test_briefing_only_includes_stations_within_radius():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    stations = _prepared_stations(now)

    briefing = build_area_briefing(
        stations, center_lat=38.1, center_lon=13.3, radius_km=80.0, now=now,
    )

    station_ids = {entry["id"] for entry in briefing["stations"]}
    assert "LIML" not in station_ids  # Milan is ~400 km away
    assert "LICJ" in station_ids
    assert "italiameteo:IM777" in station_ids


def test_temperature_aggregate_has_provenance_and_summary_datatype():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    stations = _prepared_stations(now)

    briefing = build_area_briefing(
        stations, center_lat=38.1, center_lon=13.3, radius_km=80.0, now=now,
        variables=("temperature", "windSpeed"),
    )

    temperature = briefing["variables"]["temperature"]
    assert temperature["dataType"] == DATA_TYPE_OBSERVATION_SUMMARY
    assert temperature["stationCount"] >= 2
    provenance_ids = {entry["stationId"] for entry in temperature["provenance"]}
    assert "LICJ" in provenance_ids

    # windSpeed was never reported by any nearby station: must stay
    # explicitly "insufficient_data", never a fabricated number.
    assert briefing["variables"]["windSpeed"] == "insufficient_data"


def test_out_of_bounds_reading_gets_near_zero_weight_not_deleted():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    stations = _prepared_stations(now)

    briefing = build_area_briefing(
        stations, center_lat=38.1, center_lon=13.3, radius_km=80.0, now=now,
        variables=("temperature",),
    )

    temperature = briefing["variables"]["temperature"]
    # The broken 999degC reading is still visible in the raw station list...
    broken_entry = next(s for s in briefing["stations"] if s["id"] == "meteonetwork:MN999")
    assert broken_entry["observations"]["temperature"]["value"] == 999.0
    assert broken_entry["observations"]["temperature"]["qualityScore"] == 0.0
    # ...but must not have pulled the weighted mean up towards it.
    assert temperature["mean"] < 40.0


def test_trend_is_insufficient_data_without_history():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    stations = _prepared_stations(now)

    briefing = build_area_briefing(
        stations, center_lat=38.1, center_lon=13.3, radius_km=80.0, now=now,
        variables=("temperature",),
    )
    trend = briefing["variables"]["temperature"]["trend"]
    assert trend == {
        "1h": "insufficient_data", "3h": "insufficient_data", "6h": "insufficient_data",
    }


def test_trend_uses_matching_stations_between_snapshots():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    stations = _prepared_stations(now)

    past = _station("metar", "LICJ", "Palermo Punta Raisi", 38.18, 13.10, elevation=21)
    past["observations"] = {
        "temperature": _measurement(24.4, now.timestamp() - 3900),
    }
    apply_quality_control([past], now=now)

    briefing = build_area_briefing(
        stations, center_lat=38.1, center_lon=13.3, radius_km=80.0, now=now,
        variables=("temperature",), history={"1h": [past]},
    )
    trend = briefing["variables"]["temperature"]["trend"]
    assert trend["1h"] == 3.0  # 27.4 - 24.4, LICJ is the only station in both
    assert trend["3h"] == "insufficient_data"


def test_confidence_reflects_station_density():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    stations = _prepared_stations(now)

    dense = build_area_briefing(
        stations, center_lat=38.1, center_lon=13.3, radius_km=80.0, now=now,
    )
    sparse = build_area_briefing(
        stations, center_lat=41.9, center_lon=12.5, radius_km=5.0, now=now,
    )
    assert sparse["confidence"]["level"] == "insufficient_data"
    assert sparse["confidence"]["stationCount"] == 0
    assert dense["confidence"]["stationCount"] >= 2
    assert dense["confidence"]["score"] > sparse["confidence"]["score"]


if __name__ == "__main__":
    test_briefing_only_includes_stations_within_radius()
    test_temperature_aggregate_has_provenance_and_summary_datatype()
    test_out_of_bounds_reading_gets_near_zero_weight_not_deleted()
    test_trend_is_insufficient_data_without_history()
    test_trend_uses_matching_stations_between_snapshots()
    test_confidence_reflects_station_density()
    print("AI observation layer tests passed")
