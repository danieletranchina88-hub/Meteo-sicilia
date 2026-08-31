"""Station health classification and network-wide diagnostics.

Health thresholds are provider-aware (requirement 10): a network that
normally reports with 30-40 minutes of latency must not be marked OFFLINE
after 15 minutes just because a faster network exists elsewhere.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from meteo_analysis.observations.model import provider_defaults

HEALTH_LIVE = "LIVE"
HEALTH_DELAYED = "DELAYED"
HEALTH_STALE = "STALE"
HEALTH_OFFLINE = "OFFLINE"
HEALTH_UNRELIABLE = "UNRELIABLE"


def _latest_observation_epoch(station: dict[str, Any]) -> float | None:
    latest: float | None = None
    for measurement in (station.get("observations") or {}).values():
        if not isinstance(measurement, dict):
            continue
        observed_at = measurement.get("observedAt")
        if observed_at is None:
            continue
        try:
            observed_at = float(observed_at)
        except (TypeError, ValueError):
            continue
        if latest is None or observed_at > latest:
            latest = observed_at
    return latest


def classify_health(station: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Return ``{status, lastObservationAgeMinutes, expectedLatencyMinutes}``."""

    now = now or datetime.now(timezone.utc)
    defaults = provider_defaults(station.get("source", ""))
    expected_latency = defaults["expectedLatencyMinutes"]
    stale_after = defaults["staleAfterMinutes"]
    latest_epoch = _latest_observation_epoch(station)

    if latest_epoch is None:
        return {
            "status": HEALTH_OFFLINE,
            "lastObservationAgeMinutes": None,
            "expectedLatencyMinutes": expected_latency,
        }

    observed = datetime.fromtimestamp(latest_epoch, tz=timezone.utc)
    age_minutes = (now - observed).total_seconds() / 60.0

    unreliable_flags = any(
        "out_of_bounds" in (measurement.get("qualityFlags") or [])
        for measurement in (station.get("observations") or {}).values()
        if isinstance(measurement, dict)
    )

    if age_minutes <= expected_latency:
        status = HEALTH_LIVE
    elif age_minutes <= expected_latency * 3:
        status = HEALTH_DELAYED
    elif age_minutes <= stale_after * 4:
        status = HEALTH_STALE
    else:
        status = HEALTH_OFFLINE

    if unreliable_flags and status in (HEALTH_LIVE, HEALTH_DELAYED):
        status = HEALTH_UNRELIABLE

    return {
        "status": status,
        "lastObservationAgeMinutes": round(age_minutes, 1),
        "expectedLatencyMinutes": expected_latency,
    }


def annotate_health(stations: list[dict[str, Any]], *, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    for station in stations:
        station["health"] = classify_health(station, now=now)


def network_diagnostics(
    stations: list[dict[str, Any]], *, now: datetime | None = None
) -> dict[str, Any]:
    """Aggregate diagnostics used by the audit/monitoring dashboard.

    Produces the exact counters requested by the station-network audit:
    ``stations_total``, ``stations_reporting_15m/30m/60m/3h``,
    ``stations_stale``, ``stations_offline``, plus per-source and
    per-health breakdowns.
    """

    now = now or datetime.now(timezone.utc)
    ages_minutes: list[float | None] = []
    for station in stations:
        health = station.get("health") or classify_health(station, now=now)
        ages_minutes.append(health.get("lastObservationAgeMinutes"))

    def count_within(minutes: float) -> int:
        return sum(
            1 for age in ages_minutes if age is not None and age <= minutes
        )

    by_health: dict[str, int] = {}
    by_source: dict[str, dict[str, int]] = {}
    for station in stations:
        status = (station.get("health") or {}).get("status", HEALTH_OFFLINE)
        by_health[status] = by_health.get(status, 0) + 1
        source = station.get("source", "unknown")
        bucket = by_source.setdefault(source, {"total": 0})
        bucket["total"] += 1
        bucket[status] = bucket.get(status, 0) + 1

    by_region: dict[str, int] = {}
    for station in stations:
        region = station.get("region") or "sconosciuta"
        by_region[region] = by_region.get(region, 0) + 1

    return {
        "stationsTotal": len(stations),
        "stationsReporting15m": count_within(15),
        "stationsReporting30m": count_within(30),
        "stationsReporting60m": count_within(60),
        "stationsReporting3h": count_within(180),
        "stationsStale": by_health.get(HEALTH_STALE, 0),
        "stationsOffline": by_health.get(HEALTH_OFFLINE, 0),
        "byHealth": by_health,
        "bySource": by_source,
        "byRegion": by_region,
        "computedAt": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
