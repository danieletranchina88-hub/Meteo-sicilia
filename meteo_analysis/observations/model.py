"""Canonical schema shared by every observation provider.

Each provider returns stations and observations already translated to this
common vocabulary so the rest of the pipeline (registry, QC, health,
verification) never has to special-case a source.  Raw values and raw units
are always preserved alongside the canonical ones (requirement: never
destroy the original measurement while normalizing it).
"""

from __future__ import annotations

from typing import Any

# Canonical variable names used throughout the pipeline and in the archive.
# Each entry documents the canonical (SI-ish, meteorologically conventional)
# unit and a physically-impossible-value bound used by the QC layer.  Bounds
# are intentionally generous: they exist to reject broken sensors/parsers,
# not to second-guess genuine extremes.
CANONICAL_VARIABLES: dict[str, dict[str, Any]] = {
    "temperature": {"unit": "degC", "min": -60.0, "max": 55.0},
    "dewpoint": {"unit": "degC", "min": -60.0, "max": 40.0},
    "relativeHumidity": {"unit": "%", "min": 0.0, "max": 100.0},
    "pressureStation": {"unit": "hPa", "min": 800.0, "max": 1085.0},
    "pressureMsl": {"unit": "hPa", "min": 850.0, "max": 1080.0},
    "precipitation": {"unit": "mm", "min": 0.0, "max": 500.0},
    "precipitationIntensity": {"unit": "mm/h", "min": 0.0, "max": 400.0},
    "windSpeed": {"unit": "km/h", "min": 0.0, "max": 400.0},
    "windGust": {"unit": "km/h", "min": 0.0, "max": 500.0},
    "windDirection": {"unit": "deg", "min": 0.0, "max": 360.0},
    "solarRadiation": {"unit": "W/m2", "min": 0.0, "max": 1500.0},
    "snowDepth": {"unit": "cm", "min": 0.0, "max": 1200.0},
}

# Networks are not all equal in reliability or in reporting cadence.  These
# defaults drive both the station-health thresholds (requirement 10) and the
# default quality weight used by the cross-provider QC check (requirement 9).
NETWORK_TYPE_INSTITUTIONAL = "institutional"
NETWORK_TYPE_CROWDSOURCED = "crowdsourced"
NETWORK_TYPE_AVIATION = "aviation"

PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "metar": {
        "networkType": NETWORK_TYPE_AVIATION,
        "expectedLatencyMinutes": 60,
        "staleAfterMinutes": 180,
        "baseQualityWeight": 0.85,
    },
    "italiameteo": {
        "networkType": NETWORK_TYPE_INSTITUTIONAL,
        "expectedLatencyMinutes": 40,
        "staleAfterMinutes": 120,
        "baseQualityWeight": 0.95,
    },
    "meteonetwork": {
        "networkType": NETWORK_TYPE_CROWDSOURCED,
        "expectedLatencyMinutes": 20,
        "staleAfterMinutes": 90,
        "baseQualityWeight": 0.6,
    },
}

# data_type distinguishes a direct sensor reading from any reconstruction
# (spatial analysis/interpolation).  Nothing downstream may treat the two the
# same way (requirement 6).
DATA_TYPE_OBSERVATION = "observation"
DATA_TYPE_ANALYSIS = "analysis/interpolated"


def provider_defaults(source: str) -> dict[str, Any]:
    return PROVIDER_DEFAULTS.get(
        source,
        {
            "networkType": NETWORK_TYPE_INSTITUTIONAL,
            "expectedLatencyMinutes": 60,
            "staleAfterMinutes": 180,
            "baseQualityWeight": 0.7,
        },
    )


def empty_station(
    *,
    source: str,
    source_station_id: str,
    name: str,
    lat: float,
    lon: float,
    elevation_m: float | None = None,
    wmo_id: str | None = None,
    icao_id: str | None = None,
    station_type: str | None = None,
) -> dict[str, Any]:
    """Build a canonical station record (registry row before merge)."""

    defaults = provider_defaults(source)
    return {
        "source": source,
        "sourceStationId": source_station_id,
        "wmoId": wmo_id,
        "icaoId": icao_id,
        "name": name,
        "lat": round(float(lat), 5),
        "lon": round(float(lon), 5),
        "elevationM": None if elevation_m is None else round(float(elevation_m), 1),
        "stationType": station_type,
        "networkType": defaults["networkType"],
        "country": "IT",
        "region": None,
        "province": None,
        "active": True,
        "observations": {},
    }
