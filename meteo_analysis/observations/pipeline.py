"""Orchestrates every observation provider into one coherent payload.

This is the single entry point the rest of the codebase (``process_data.py``,
``collect_observations.py``, future API layers) should call.  It keeps the
existing public JSON schema fully backward compatible (``stations``,
``stationNetwork``, ``count``, ``capturedAt``, ...) so nothing that already
reads ``data_weather/observations.json`` breaks, while adding the new
canonical, multi-provider fields described in the station-network audit
(``registry``, ``diagnostics``, ``coverage``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from meteo_analysis.observations.coverage import compute_coverage_report
from meteo_analysis.observations.health import annotate_health, network_diagnostics
from meteo_analysis.observations.providers.base import ObservationProvider, ProviderResult
from meteo_analysis.observations.providers.italiameteo_provider import ItaliaMeteoProvider
from meteo_analysis.observations.providers.metar_provider import MetarProvider
from meteo_analysis.observations.providers.meteonetwork_provider import MeteoNetworkProvider
from meteo_analysis.observations.quality import apply_quality_control
from meteo_analysis.observations.registry import build_registry

PIPELINE_SCHEMA_VERSION = 1


def default_providers(*, environ: dict[str, str] | None = None) -> list[ObservationProvider]:
    return [
        MetarProvider(),
        ItaliaMeteoProvider(environ=environ),
        MeteoNetworkProvider(environ=environ),
    ]


def _legacy_metar_view(stations: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild the pre-existing METAR-only schema for zero-risk compatibility.

    The client-side fusion/analysis code (``index.html``) and the ICON
    station-forecast archive (``meteo_analysis/verification/stations.py``)
    both expect this exact shape; they keep working unmodified.
    """

    live: list[dict[str, Any]] = []
    network: list[dict[str, Any]] = []
    for station in stations:
        if station.get("source") != "metar":
            continue
        network.append({
            "id": station["sourceStationId"],
            "name": station["name"],
            "lat": station["lat"],
            "lon": station["lon"],
            "elevationM": station.get("elevationM"),
            "stationTypes": ["METAR"],
            "country": "IT",
        })
        observations = station.get("observations") or {}
        if not observations:
            continue
        observed_at = None
        for measurement in observations.values():
            if isinstance(measurement, dict) and measurement.get("observedAt") is not None:
                observed_at = measurement["observedAt"]
                break
        wind = observations.get("windSpeed", {})
        gust = observations.get("windGust", {})
        direction = observations.get("windDirection", {})
        temp = observations.get("temperature", {})
        dewp = observations.get("dewpoint", {})
        pressure = observations.get("pressureMsl", {})
        live.append({
            "id": station["sourceStationId"],
            "name": station["name"],
            "lat": station["lat"],
            "lon": station["lon"],
            "elevationM": station.get("elevationM"),
            "obsTime": observed_at,
            "tempC": temp.get("value") if isinstance(temp, dict) else None,
            "dewpC": dewp.get("value") if isinstance(dewp, dict) else None,
            "wspdKmh": wind.get("value") if isinstance(wind, dict) else None,
            "windGustKmh": gust.get("value") if isinstance(gust, dict) else None,
            "wdir": direction.get("value") if isinstance(direction, dict) else None,
            "seaLevelPressureHpa": pressure.get("value") if isinstance(pressure, dict) else None,
            "pressHpa": pressure.get("value") if isinstance(pressure, dict) else None,
            "altimeterHpa": None,
            "rawReport": station.get("rawReport"),
        })
    return {
        "count": len(live),
        "stations": live,
        "stationNetwork": {
            "count": len(network),
            "stations": network,
        },
    }


def collect_national_observations(
    *,
    providers: list[ObservationProvider] | None = None,
    environ: dict[str, str] | None = None,
    session=None,
    timeout: tuple[int, int] = (15, 45),
    history: dict[str, dict[str, list[float]]] | None = None,
    compute_coverage: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch, merge, QC and classify observations from every configured
    provider.  A provider failure never raises past this function: it is
    recorded in ``providerStatus`` and the pipeline keeps going with
    whatever data remains available from the other sources.
    """

    now = now or datetime.now(timezone.utc)
    providers = providers if providers is not None else default_providers(environ=environ)

    provider_status: dict[str, Any] = {}
    all_stations: list[dict[str, Any]] = []
    for provider in providers:
        result: ProviderResult = provider.safe_fetch(session=session, timeout=timeout)
        provider_status[provider.source] = {
            "ok": result.ok,
            "configured": result.configured,
            "error": result.error,
            "stationCount": len(result.stations),
            "fetchedAt": result.fetched_at,
        }
        if result.ok:
            all_stations.extend(result.stations)

    registry = build_registry(all_stations)
    stations = registry["stations"]
    apply_quality_control(stations, history=history, now=now)
    annotate_health(stations, now=now)
    diagnostics = network_diagnostics(stations, now=now)
    diagnostics["duplicatesFlagged"] = registry["duplicatesFlagged"]
    diagnostics["providerStatus"] = provider_status

    active_sources = sorted(
        source for source, status in provider_status.items() if status["ok"]
    )
    source_labels = {
        "metar": "NOAA AWC METAR",
        "italiameteo": "Agenzia ItaliaMeteo / MeteoHub",
        "meteonetwork": "MeteoNetwork",
    }
    payload: dict[str, Any] = {
        "schemaVersion": PIPELINE_SCHEMA_VERSION,
        "capturedAt": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": " + ".join(source_labels.get(source, source) for source in active_sources)
            or "nessuna fonte disponibile in questo run",
        "sourceUrl": None,
        "providerStatus": provider_status,
        "diagnostics": diagnostics,
        "registry": {
            "stations": stations,
            "totalStations": len(stations),
        },
    }
    if compute_coverage:
        payload["coverage"] = compute_coverage_report(stations)
    payload.update(_legacy_metar_view(stations))
    return payload
