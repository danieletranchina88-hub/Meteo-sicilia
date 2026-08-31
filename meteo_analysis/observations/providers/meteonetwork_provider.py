"""MeteoNetwork (amateur/crowdsourced) provider.

MeteoNetwork exposes a documented REST v3 API
(``https://api.meteonetwork.it/v3``) with Swagger documentation.  Access
requires a free account: a token is obtained via ``POST /v3/login`` with an
email/password pair and then used as a bearer token on every subsequent
call.  This sandbox has no network access to validate the exact response
shape against a live account, so field extraction is defensive (several
plausible key names are tried) and any station that fails to parse is
skipped rather than guessed.

Required environment variables
-------------------------------
``METEONETWORK_EMAIL`` / ``METEONETWORK_PASSWORD``
    Credentials for ``POST /v3/login``.  Alternatively ``METEONETWORK_TOKEN``
    can be supplied directly if a long-lived token has already been issued.

MeteoNetwork is explicitly an *additional* densification network: its
stations must always carry ``source = "meteonetwork"`` and a lower default
quality weight than institutional networks (see
:mod:`meteo_analysis.observations.model`), never silently merged with or
promoted to institutional status.
"""

from __future__ import annotations

import os
from typing import Any

from meteo_analysis.observations.model import DATA_TYPE_OBSERVATION, empty_station
from meteo_analysis.observations.providers.base import ObservationProvider, ProviderResult

API_BASE = "https://api.meteonetwork.it/v3"


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _float(mapping: dict[str, Any], *keys: str) -> float | None:
    value = _first(mapping, *keys)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


class MeteoNetworkProvider(ObservationProvider):
    source = "meteonetwork"

    def __init__(self, *, environ: dict[str, str] | None = None):
        self._environ = environ if environ is not None else os.environ

    def is_configured(self) -> bool:
        has_token = bool(self._environ.get("METEONETWORK_TOKEN"))
        has_credentials = bool(
            self._environ.get("METEONETWORK_EMAIL")
            and self._environ.get("METEONETWORK_PASSWORD")
        )
        return has_token or has_credentials

    def _authorization_token(self, session, timeout) -> str:
        token = self._environ.get("METEONETWORK_TOKEN")
        if token:
            return token
        response = session.post(
            f"{API_BASE}/login",
            data={
                "email": self._environ["METEONETWORK_EMAIL"],
                "password": self._environ["METEONETWORK_PASSWORD"],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        issued_token = _first(payload, "token", "access_token", "jwt")
        if not issued_token:
            raise ValueError("login MeteoNetwork riuscito ma senza token in risposta")
        return str(issued_token)

    def fetch(self, *, session=None, timeout: tuple[int, int] = (15, 45)) -> ProviderResult:
        if session is None:
            import requests

            session = requests
        token = self._authorization_token(session, timeout)
        headers = {
            "Authorization": "Bearer " + token,
            "User-Agent": "Meteo-Sicilia-Verification/1.0",
        }

        stations_response = session.get(
            f"{API_BASE}/stations", headers=headers, timeout=timeout
        )
        stations_response.raise_for_status()
        stations_raw = _as_list(stations_response.json())

        realtime_response = session.get(
            f"{API_BASE}/realtime", headers=headers, timeout=timeout
        )
        realtime_by_id: dict[str, dict[str, Any]] = {}
        if realtime_response.ok:
            for record in _as_list(realtime_response.json()):
                if not isinstance(record, dict):
                    continue
                station_id = str(
                    _first(record, "idstation", "stationId", "id", "code") or ""
                ).strip()
                if station_id:
                    realtime_by_id[station_id] = record

        stations: list[dict[str, Any]] = []
        for record in stations_raw:
            if not isinstance(record, dict):
                continue
            station_id = str(
                _first(record, "idstation", "stationId", "id", "code") or ""
            ).strip()
            lat = _float(record, "lat", "latitude")
            lon = _float(record, "lon", "lng", "longitude")
            if not station_id or lat is None or lon is None:
                continue
            station = empty_station(
                source=self.source,
                source_station_id=station_id,
                name=str(_first(record, "name", "nickname", "city") or station_id),
                lat=lat,
                lon=lon,
                elevation_m=_float(record, "elevation", "alt", "quota"),
                station_type="meteonetwork-crowdsourced",
            )
            station["region"] = _first(record, "region", "regione")
            station["province"] = _first(record, "province", "provincia")
            realtime = realtime_by_id.get(station_id)
            if realtime:
                station["observations"] = _normalize_observation(realtime)
            stations.append(station)

        return ProviderResult(source=self.source, ok=True, stations=stations)


def _as_list(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "items", "results", "stations"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "temperature": ("temperature", "temp", "t"),
    "dewpoint": ("dewpoint", "dewPoint", "td"),
    "relativeHumidity": ("humidity", "rh", "relativeHumidity"),
    "pressureStation": ("pressure", "qfe"),
    "pressureMsl": ("pressureMsl", "qnh", "slp"),
    "precipitation": ("rain", "rain1h", "precipitation"),
    "windSpeed": ("windSpeed", "wind", "windavg"),
    "windGust": ("windGust", "gust", "windmax"),
    "windDirection": ("windDir", "winddir", "windDirection"),
    "solarRadiation": ("radiation", "solarRadiation"),
}


def _normalize_observation(record: dict[str, Any]) -> dict[str, Any]:
    observed_at = _first(record, "date", "timestamp", "obsTime", "time")
    result: dict[str, Any] = {}
    for canonical, aliases in _FIELD_ALIASES.items():
        raw_value = _first(record, *aliases)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        result[canonical] = {
            "value": value,
            "rawValue": raw_value,
            "rawUnit": _first(record, canonical + "Unit", "unit") or "unknown",
            "canonicalUnit": None,
            "observedAt": observed_at,
            "dataType": DATA_TYPE_OBSERVATION,
            "qualityFlag": _first(record, "qcFlag", "qualityFlag"),
        }
    return result
