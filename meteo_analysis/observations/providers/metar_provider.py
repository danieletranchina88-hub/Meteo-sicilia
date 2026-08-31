"""METAR/AWC provider: adapts the existing verified fetcher to the
canonical multi-provider schema without changing its network behaviour.

This is a thin wrapper: all the parsing/normalization logic that was already
reviewed and tested lives in ``meteo_analysis.verification.observations`` and
keeps working exactly as before, including for the pre-existing
``data_weather/observations.json`` (fusion / legacy station map) consumers.
"""

from __future__ import annotations

from typing import Any

from meteo_analysis.observations.model import DATA_TYPE_OBSERVATION, empty_station
from meteo_analysis.observations.providers.base import ObservationProvider, ProviderResult
from meteo_analysis.verification.observations import fetch_italy_metar_observations


class MetarProvider(ObservationProvider):
    source = "metar"

    def fetch(self, *, session=None, timeout: tuple[int, int] = (15, 45)) -> ProviderResult:
        payload = fetch_italy_metar_observations(session=session, timeout=timeout)
        network = payload.get("stationNetwork", {}) or {}
        reports_by_id: dict[str, dict[str, Any]] = {
            item["id"]: item for item in payload.get("stations", []) if item.get("id")
        }
        stations: list[dict[str, Any]] = []
        for entry in network.get("stations", []):
            station_id = str(entry.get("id") or "").strip().upper()
            if not station_id:
                continue
            report = reports_by_id.get(station_id)
            station = empty_station(
                source=self.source,
                source_station_id=station_id,
                name=entry.get("name") or station_id,
                lat=entry.get("lat"),
                lon=entry.get("lon"),
                elevation_m=entry.get("elevationM"),
                icao_id=station_id,
                station_type="aviation-metar",
            )
            if report and report.get("obsTime") is not None:
                station["observations"] = {
                    "temperature": _measure(report.get("tempC"), "degC", report["obsTime"]),
                    "dewpoint": _measure(report.get("dewpC"), "degC", report["obsTime"]),
                    "pressureMsl": _measure(report.get("pressHpa"), "hPa", report["obsTime"]),
                    "windSpeed": _measure(report.get("wspdKmh"), "km/h", report["obsTime"]),
                    "windGust": _measure(report.get("windGustKmh"), "km/h", report["obsTime"]),
                    "windDirection": _measure(report.get("wdir"), "deg", report["obsTime"]),
                }
                station["observations"] = {
                    key: value for key, value in station["observations"].items()
                    if value is not None
                }
                station["rawReport"] = report.get("rawReport")
            stations.append(station)
        return ProviderResult(
            source=self.source,
            ok=True,
            stations=stations,
            fetched_at=payload.get("capturedAt"),
        )


def _measure(value, unit: str, obs_time_epoch) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "value": value,
        "rawValue": value,
        "rawUnit": unit,
        "canonicalUnit": unit,
        "observedAt": obs_time_epoch,
        "dataType": DATA_TYPE_OBSERVATION,
        "qualityFlag": None,
    }
