#!/usr/bin/env python3
"""Regression tests for the multi-provider observation network.

No live network calls are made: every provider fixture is a small, hand
built canonical station list or a fake ``ObservationProvider`` subclass, in
line with the requirement that provider parsers be testable from recorded
fixtures rather than live APIs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from meteo_analysis.observations.coverage import compute_coverage  # noqa: E402
from meteo_analysis.observations.health import (  # noqa: E402
    HEALTH_DELAYED,
    HEALTH_LIVE,
    HEALTH_OFFLINE,
    HEALTH_STALE,
    annotate_health,
    classify_health,
    network_diagnostics,
)
from meteo_analysis.observations.model import (  # noqa: E402
    DATA_TYPE_OBSERVATION,
    empty_station,
)
from meteo_analysis.observations.pipeline import (  # noqa: E402
    collect_national_observations,
    default_providers,
)
from meteo_analysis.observations.providers.base import (  # noqa: E402
    ObservationProvider,
    ProviderResult,
)
from meteo_analysis.observations.providers.italiameteo_provider import (  # noqa: E402
    ItaliaMeteoProvider,
)
from meteo_analysis.observations.providers.meteonetwork_provider import (  # noqa: E402
    MeteoNetworkProvider,
)
from meteo_analysis.observations.quality import (  # noqa: E402
    FLAG_OK,
    FLAG_OUT_OF_BOUNDS,
    FLAG_STUCK,
    apply_quality_control,
    evaluate_observation,
)
from meteo_analysis.observations.registry import build_registry, match_confidence  # noqa: E402


def _station(source, station_id, name, lat, lon, elevation=10.0, **kwargs):
    return empty_station(
        source=source,
        source_station_id=station_id,
        name=name,
        lat=lat,
        lon=lon,
        elevation_m=elevation,
        **kwargs,
    )


def _measurement(value, observed_at):
    return {
        "value": value,
        "rawValue": value,
        "rawUnit": "degC",
        "canonicalUnit": "degC",
        "observedAt": observed_at,
        "dataType": DATA_TYPE_OBSERVATION,
        "qualityFlag": None,
    }


class FakeProvider(ObservationProvider):
    def __init__(self, source, stations, *, configured=True, ok=True, error=None):
        self.source = source
        self._stations = stations
        self._configured = configured
        self._ok = ok
        self._error = error

    def is_configured(self):
        return self._configured

    def fetch(self, *, session=None, timeout=None):
        if not self._ok:
            raise RuntimeError(self._error or "provider fittizio in errore")
        return ProviderResult(source=self.source, ok=True, stations=self._stations)


def test_provider_isolation_never_empties_whole_network():
    """One provider failing must not take the others down (requirement 4/19)."""

    now = time.time()
    good = _station("metar", "LICJ", "Palermo Punta Raisi", 38.18, 13.10)
    good["observations"] = {"temperature": _measurement(27.4, now - 120)}
    broken_provider = FakeProvider("italiameteo", [], ok=False, error="rete irraggiungibile")
    not_configured_provider = FakeProvider("meteonetwork", [], configured=False)

    payload = collect_national_observations(
        providers=[
            FakeProvider("metar", [good]),
            broken_provider,
            not_configured_provider,
        ],
        compute_coverage=False,
    )

    assert payload["registry"]["totalStations"] == 1
    assert payload["providerStatus"]["italiameteo"]["ok"] is False
    assert "rete irraggiungibile" in payload["providerStatus"]["italiameteo"]["error"]
    assert payload["providerStatus"]["meteonetwork"]["configured"] is False
    # Legacy schema used by index.html and StationForecastArchive is intact.
    assert payload["count"] == 1
    assert payload["stations"][0]["id"] == "LICJ"
    assert payload["stationNetwork"]["stations"][0]["id"] == "LICJ"


def test_registry_deduplication_confidence_is_conservative():
    """Two nearby stations from different providers are only flagged, never
    silently merged, and a coincidental nearby-but-different site keeps a
    low confidence score (requirement 8)."""

    a = _station("metar", "LICJ", "Palermo Punta Raisi", 38.1824, 13.0910, icao_id="LICJ")
    b = _station("italiameteo", "16560", "Punta Raisi Aeroporto", 38.1825, 13.0912)
    unrelated = _station("meteonetwork", "MN001", "Erice Vetta", 38.038, 12.588)

    registry = build_registry([a, b, unrelated])
    stations = {item["internalStationId"]: item for item in registry["stations"]}

    metar_entry = stations["metar:LICJ"]
    im_entry = stations["italiameteo:16560"]
    assert metar_entry["duplicateCandidates"], "stazioni vicine devono essere annotate"
    assert metar_entry["duplicateCandidates"][0]["confidence"] >= 0.85
    # High confidence match: one of the two pairs is flagged as duplicateOf,
    # the other keeps existing (nothing is dropped from the registry).
    assert (
        metar_entry["duplicateOfStationId"] == "italiameteo:16560"
        or im_entry["duplicateOfStationId"] == "metar:LICJ"
    )
    assert registry["totalBeforeDeduplication"] == 3
    assert registry["duplicatesFlagged"] == 1

    far_confidence = match_confidence(a, unrelated)
    assert far_confidence == 0.0


def test_quality_control_physical_bounds_and_stuck_sensor():
    now = datetime.now(timezone.utc)
    now_epoch = now.timestamp()

    impossible = evaluate_observation(
        "temperature", 120.0, source="metar", observed_at_epoch=now_epoch
    )
    assert FLAG_OUT_OF_BOUNDS in impossible["qualityFlags"]
    assert impossible["qualityScore"] == 0.0

    ok = evaluate_observation(
        "temperature", 21.5, source="italiameteo", observed_at_epoch=now_epoch
    )
    assert ok["qualityFlags"] == [FLAG_OK]
    assert ok["qualityScore"] > 0.5

    stuck = evaluate_observation(
        "temperature", 18.0, source="meteonetwork", observed_at_epoch=now_epoch,
        recent_values=[18.0] * 8,
    )
    assert FLAG_STUCK in stuck["qualityFlags"]
    assert stuck["qualityScore"] < 0.5


def test_apply_quality_control_mutates_station_observations():
    station = _station("metar", "LIRF", "Roma Fiumicino", 41.80, 12.24)
    station["observations"] = {"temperature": _measurement(19.0, time.time())}
    apply_quality_control([station])
    measurement = station["observations"]["temperature"]
    assert "qualityScore" in measurement
    assert measurement["qualityFlags"] == [FLAG_OK]


def test_station_health_thresholds_are_provider_aware():
    """A network with 40-minute expected latency is not OFFLINE at 45
    minutes; METAR (60-minute expected) is not even DELAYED yet at the same
    age (requirement 10)."""

    now = datetime.now(timezone.utc)
    recent_epoch = now.timestamp() - 45 * 60

    italiameteo_station = _station("italiameteo", "X1", "Test", 42.0, 12.0)
    italiameteo_station["observations"] = {
        "temperature": _measurement(20.0, recent_epoch)
    }
    metar_station = _station("metar", "X2", "Test", 42.0, 12.0)
    metar_station["observations"] = {"temperature": _measurement(20.0, recent_epoch)}

    im_health = classify_health(italiameteo_station, now=now)
    metar_health = classify_health(metar_station, now=now)
    assert im_health["status"] == HEALTH_DELAYED
    assert metar_health["status"] == HEALTH_LIVE

    offline_station = _station("metar", "X3", "Test", 42.0, 12.0)
    offline_health = classify_health(offline_station, now=now)
    assert offline_health["status"] == HEALTH_OFFLINE

    stale_station = _station("metar", "X4", "Test", 42.0, 12.0)
    stale_station["observations"] = {
        "temperature": _measurement(20.0, now.timestamp() - 8 * 3600)
    }
    stale_health = classify_health(stale_station, now=now)
    assert stale_health["status"] == HEALTH_STALE


def test_network_diagnostics_counts_by_freshness_and_source():
    now = datetime.now(timezone.utc)
    live = _station("metar", "A", "A", 41.0, 12.0)
    live["observations"] = {"temperature": _measurement(20.0, now.timestamp() - 300)}
    offline = _station("meteonetwork", "B", "B", 41.1, 12.1)
    stations = [live, offline]
    annotate_health(stations, now=now)
    diagnostics = network_diagnostics(stations, now=now)
    assert diagnostics["stationsTotal"] == 2
    assert diagnostics["stationsReporting15m"] == 1
    assert diagnostics["stationsOffline"] == 1
    assert diagnostics["bySource"]["metar"]["total"] == 1
    assert diagnostics["bySource"]["meteonetwork"]["total"] == 1


def test_coverage_reports_gap_fraction_and_variable_filter():
    stations = [
        _station("metar", "A", "A", 41.9, 12.5),  # Roma area
    ]
    stations[0]["observations"] = {"temperature": _measurement(20.0, time.time())}
    coverage_all = compute_coverage(stations, grid_step_deg=1.0)
    assert coverage_all["cells"] > 0
    assert coverage_all["meanNearestDistanceKm"] > 0

    coverage_missing_variable = compute_coverage(
        stations, variable="precipitation", grid_step_deg=1.0
    )
    assert coverage_missing_variable["cells"] == 0
    assert coverage_missing_variable["meanNearestDistanceKm"] is None


def test_optional_providers_are_disabled_without_credentials():
    """ItaliaMeteo/MeteoNetwork must not attempt network calls or raise when
    their environment variables are absent (requirement: fail closed, not
    silently pretend to have data)."""

    empty_environ: dict[str, str] = {}
    italiameteo = ItaliaMeteoProvider(environ=empty_environ)
    meteonetwork = MeteoNetworkProvider(environ=empty_environ)
    assert italiameteo.is_configured() is False
    assert meteonetwork.is_configured() is False

    result_im = italiameteo.safe_fetch()
    result_mn = meteonetwork.safe_fetch()
    assert result_im.ok is False and result_im.configured is False
    assert result_mn.ok is False and result_mn.configured is False

    providers = default_providers(environ=empty_environ)
    assert {provider.source for provider in providers} == {
        "metar", "italiameteo", "meteonetwork",
    }


if __name__ == "__main__":
    test_provider_isolation_never_empties_whole_network()
    test_registry_deduplication_confidence_is_conservative()
    test_quality_control_physical_bounds_and_stuck_sensor()
    test_apply_quality_control_mutates_station_observations()
    test_station_health_thresholds_are_provider_aware()
    test_network_diagnostics_counts_by_freshness_and_source()
    test_coverage_reports_gap_fraction_and_variable_filter()
    test_optional_providers_are_disabled_without_credentials()
    print("Observation network tests passed")
