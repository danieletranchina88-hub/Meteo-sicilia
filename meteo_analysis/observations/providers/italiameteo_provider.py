"""Agenzia ItaliaMeteo / MeteoHub provider.

MeteoHub (``https://meteohub.agenziaitaliameteo.it``) distributes the
national in-situ network operated by ItaliaMeteo (~4500 telemetered stations
aggregated from the regional networks) as JSON/BUFR after a free
registration.  Machine-to-machine access requires an API token that this
sandbox cannot obtain or verify against a live account, so this provider is
implemented but **disabled by default**: it activates only when the
following environment variables are present.

Required environment variables
-------------------------------
``METEOHUB_BASE_URL``
    Base URL of the MeteoHub REST API (e.g.
    ``https://meteohub.agenziaitaliameteo.it/api/v1``).  Documented in the
    MeteoHub user guide after registration; kept configurable because the
    public sandbox used to write this integration has no network access to
    confirm the exact production path.
``METEOHUB_API_TOKEN``
    Access token issued by MeteoHub after registration
    (``DATASET > Estrazione Dati``).

Optional
--------
``METEOHUB_STATIONS_PATH`` (default ``observations/stations``)
``METEOHUB_OBSERVATIONS_PATH`` (default ``observations/latest``)

The response schema is normalised defensively: several plausible key names
are tried for every field because the exact production JSON shape could not
be inspected from this environment.  Any station whose identifier or
coordinates cannot be parsed is skipped rather than guessed.
"""

from __future__ import annotations

import os
from typing import Any

from meteo_analysis.observations.model import DATA_TYPE_OBSERVATION, empty_station
from meteo_analysis.observations.providers.base import ObservationProvider, ProviderResult
from meteo_analysis.observations.providers._utils import as_float, as_list, first


class ItaliaMeteoProvider(ObservationProvider):
    source = "italiameteo"

    def __init__(self, *, environ: dict[str, str] | None = None):
        self._environ = environ if environ is not None else os.environ

    def is_configured(self) -> bool:
        return bool(
            self._environ.get("METEOHUB_BASE_URL")
            and self._environ.get("METEOHUB_API_TOKEN")
        )

    def fetch(self, *, session=None, timeout: tuple[int, int] = (15, 45)) -> ProviderResult:
        if session is None:
            import requests

            session = requests
        base_url = self._environ["METEOHUB_BASE_URL"].rstrip("/")
        token = self._environ["METEOHUB_API_TOKEN"]
        stations_path = self._environ.get(
            "METEOHUB_STATIONS_PATH", "observations/stations"
        )
        observations_path = self._environ.get(
            "METEOHUB_OBSERVATIONS_PATH", "observations/latest"
        )
        headers = {
            "Authorization": "Bearer " + token,
            "User-Agent": "Meteo-Sicilia-Verification/1.0",
        }

        stations_response = session.get(
            f"{base_url}/{stations_path}", headers=headers, timeout=timeout
        )
        stations_response.raise_for_status()
        stations_raw = as_list(stations_response.json())

        observations_response = session.get(
            f"{base_url}/{observations_path}", headers=headers, timeout=timeout
        )
        observations_response.raise_for_status()
        observations_raw = as_list(observations_response.json())

        observations_by_id: dict[str, dict[str, Any]] = {}
        for record in observations_raw:
            if not isinstance(record, dict):
                continue
            station_id = str(
                first(record, "stationId", "station_id", "id", "code") or ""
            ).strip()
            if station_id:
                observations_by_id[station_id] = record

        stations: list[dict[str, Any]] = []
        for record in stations_raw:
            if not isinstance(record, dict):
                continue
            station_id = str(
                first(record, "stationId", "station_id", "id", "code") or ""
            ).strip()
            lat = as_float(record, "lat", "latitude")
            lon = as_float(record, "lon", "lng", "longitude")
            if not station_id or lat is None or lon is None:
                continue
            station = empty_station(
                source=self.source,
                source_station_id=station_id,
                name=str(first(record, "name", "stationName") or station_id),
                lat=lat,
                lon=lon,
                elevation_m=as_float(record, "elevation", "elevationM", "quota", "alt"),
                wmo_id=first(record, "wmoId", "wmo_id"),
                station_type=str(first(record, "network", "rete") or "italiameteo"),
            )
            station["region"] = first(record, "region", "regione")
            station["province"] = first(record, "province", "provincia")
            observation = observations_by_id.get(station_id)
            if observation:
                station["observations"] = _normalize_observation(observation)
            stations.append(station)

        return ProviderResult(source=self.source, ok=True, stations=stations)


_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "temperature": ("temperature", "temp", "t", "airTemperature"),
    "dewpoint": ("dewpoint", "dewPoint", "td"),
    "relativeHumidity": ("relativeHumidity", "humidity", "rh", "ur"),
    "pressureMsl": ("pressureMsl", "mslp", "pressureSeaLevel"),
    "pressureStation": ("pressureStation", "pressure", "qfe"),
    "precipitation": ("precipitation", "rain", "precip1h", "pioggia"),
    "windSpeed": ("windSpeed", "wind", "vento"),
    "windGust": ("windGust", "gust", "raffica"),
    "windDirection": ("windDirection", "windDir", "direzioneVento"),
    "solarRadiation": ("solarRadiation", "radiation", "radiazione"),
    "snowDepth": ("snowDepth", "snow", "neve"),
}


def _normalize_observation(record: dict[str, Any]) -> dict[str, Any]:
    observed_at = first(record, "observedAt", "timestamp", "obsTime", "time")
    result: dict[str, Any] = {}
    for canonical, aliases in _FIELD_ALIASES.items():
        raw_value = first(record, *aliases)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        result[canonical] = {
            "value": value,
            "rawValue": raw_value,
            "rawUnit": first(record, canonical + "Unit", "unit") or "unknown",
            "canonicalUnit": None,
            "observedAt": observed_at,
            "dataType": DATA_TYPE_OBSERVATION,
            "qualityFlag": first(record, "qcFlag", "qualityFlag"),
        }
    return result
