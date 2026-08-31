"""Strict normalization of surface observations used for verification.

The Aviation Weather Center exposes both sea-level pressure (``slp``) and an
altimeter setting (``altim``).  They are different quantities.  Only ``slp``
is allowed to verify or analyse ICON mean sea-level pressure; the altimeter
setting is retained separately for provenance and future specialist uses.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Iterable


SOURCE_NAME = "NOAA Aviation Weather Center METAR"
SOURCE_URL = "https://aviationweather.gov/api/data/metar"
STATION_INFO_URL = "https://aviationweather.gov/api/data/stationinfo"

# AWC's station catalogue is broader and more stable than the subset that
# happens to issue a METAR in one hour.  These bounds include the national
# territory and islands while avoiding a claim that nearby foreign stations
# are part of the Italian observing network.
ITALY_DOMAIN = (35.0, 6.0, 48.0, 19.0)
ITALY_COUNTRY = "IT"
SCHEMA_VERSION = 3
STATION_NETWORK_SCHEMA_VERSION = 1


def _number(mapping: dict[str, Any], key: str) -> float | None:
    try:
        value = float(mapping.get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _epoch_seconds(value: Any) -> int | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        seconds = float(value)
        if seconds > 10_000_000_000:  # tolerate millisecond epochs
            seconds /= 1000.0
        return int(seconds) if seconds > 0 else None
    if isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.astimezone(timezone.utc).timestamp())
    return None


def _altimeter_hpa(value: float | None) -> float | None:
    if value is None:
        return None
    # AWC normally returns hPa in JSON, but accepting inHg makes archived
    # payloads robust to a provider-format change without confusing units.
    if 20.0 <= value <= 35.0:
        value *= 33.8638866667
    return value if 850.0 < value < 1080.0 else None


def _sea_level_pressure_hpa(value: float | None) -> float | None:
    return value if value is not None and 850.0 < value < 1080.0 else None


def normalize_metar_reports(
    reports: Iterable[dict[str, Any]],
    *,
    domain: tuple[float, float, float, float],
    captured_at: datetime | None = None,
    station_catalog: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return strict, unit-explicit observations inside ``domain``.

    ``domain`` is ``(south, west, north, east)``.  A timestamp is retained per
    station; the payload-level timestamp is only the latest included report
    and must never be assigned to every station during verification.
    """

    south, west, north, east = (float(value) for value in domain)
    catalog_by_id = {
        str(item.get("id") or "").strip().upper(): item
        for item in (station_catalog or [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    captured = captured_at or datetime.now(timezone.utc)
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=timezone.utc)
    stations: list[dict[str, Any]] = []
    latest = 0

    for report in reports:
        if not isinstance(report, dict):
            continue
        latitude = _number(report, "lat")
        longitude = _number(report, "lon")
        if latitude is None or longitude is None:
            continue
        if not (south <= latitude <= north and west <= longitude <= east):
            continue

        station_id = str(report.get("icaoId") or "").strip().upper()
        if not station_id:
            continue
        catalog = catalog_by_id.get(station_id, {})
        observed_at = _epoch_seconds(report.get("obsTime"))
        if observed_at is not None:
            latest = max(latest, observed_at)

        wind_kt = _number(report, "wspd")
        gust_kt = _number(report, "wgst")
        slp = _sea_level_pressure_hpa(_number(report, "slp"))
        altimeter = _altimeter_hpa(_number(report, "altim"))
        elevation = _number(report, "elev")
        if elevation is None:
            elevation = _number(catalog, "elevationM")
        if elevation is not None and not (-500.0 <= elevation <= 9000.0):
            elevation = None

        stations.append({
            "id": station_id,
            "name": str(report.get("name") or catalog.get("name") or ""),
            "lat": round(latitude, 5),
            "lon": round(longitude, 5),
            "elevationM": round(elevation, 1) if elevation is not None else None,
            "obsTime": observed_at,
            "tempC": _number(report, "temp"),
            "dewpC": _number(report, "dewp"),
            "wspdKmh": round(wind_kt * 1.852, 2) if wind_kt is not None else None,
            "windGustKmh": round(gust_kt * 1.852, 2) if gust_kt is not None else None,
            "wdir": _number(report, "wdir"),
            "seaLevelPressureHpa": slp,
            # Backward-compatible site field.  It deliberately mirrors only
            # true SLP, never the altimeter setting.
            "pressHpa": slp,
            "altimeterHpa": round(altimeter, 2) if altimeter is not None else None,
            "rawReport": str(report.get("rawOb") or "") or None,
        })

    stations.sort(key=lambda item: item["id"])
    return {
        "schemaVersion": SCHEMA_VERSION,
        "source": SOURCE_NAME,
        "sourceUrl": SOURCE_URL,
        "capturedAt": captured.astimezone(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "obsTime": latest or None,
        "count": len(stations),
        "pressureSemantics": {
            "seaLevelPressureHpa": "METAR SLP; comparable with model MSLP",
            "altimeterHpa": "altimeter setting; retained but not compared with MSLP",
        },
        "stations": stations,
    }


def normalize_italian_station_network(
    records: Iterable[dict[str, Any]],
    *,
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    """Return the official AWC Italian METAR catalogue, not only reporters.

    A station remains in the catalogue if its latest METAR is unavailable.
    That distinction is required on the public map: an absent report means
    "no current observation", not "no station exists here".
    """

    captured = captured_at or datetime.now(timezone.utc)
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=timezone.utc)
    south, west, north, east = ITALY_DOMAIN
    stations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        station_id = str(record.get("icaoId") or "").strip().upper()
        station_types = record.get("siteType") or []
        if isinstance(station_types, str):
            station_types = [station_types]
        station_types = [str(value).strip().upper() for value in station_types]
        latitude = _number(record, "lat")
        longitude = _number(record, "lon")
        if (
            not station_id
            or station_id in seen
            or str(record.get("country") or "").strip().upper() != ITALY_COUNTRY
            or "METAR" not in station_types
            or latitude is None
            or longitude is None
            or not (south <= latitude <= north and west <= longitude <= east)
        ):
            continue
        elevation = _number(record, "elev")
        if elevation is not None and not (-500.0 <= elevation <= 9000.0):
            elevation = None
        seen.add(station_id)
        stations.append({
            "id": station_id,
            "name": str(record.get("site") or station_id),
            "lat": round(latitude, 5),
            "lon": round(longitude, 5),
            "elevationM": round(elevation, 1) if elevation is not None else None,
            "stationTypes": station_types,
            "country": ITALY_COUNTRY,
        })
    stations.sort(key=lambda item: item["id"])
    return {
        "schemaVersion": STATION_NETWORK_SCHEMA_VERSION,
        "source": "NOAA Aviation Weather Center stationinfo",
        "sourceUrl": STATION_INFO_URL,
        "country": ITALY_COUNTRY,
        "domain": {
            "south": south, "west": west, "north": north, "east": east,
        },
        "capturedAt": captured.astimezone(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "count": len(stations),
        "stations": stations,
    }


def fetch_italy_station_network(
    *,
    session=None,
    timeout: tuple[int, int] = (15, 45),
) -> dict[str, Any]:
    """Fetch the current official catalogue of Italian METAR stations."""

    if session is None:
        import requests

        session = requests
    south, west, north, east = ITALY_DOMAIN
    response = session.get(
        STATION_INFO_URL,
        params={
            "format": "json",
            "bbox": f"{south},{west},{north},{east}",
        },
        timeout=timeout,
        headers={"User-Agent": "Meteo-Sicilia-Verification/1.0"},
    )
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, list):
        raise ValueError("catalogo AWC stationinfo non composto da una lista")
    network = normalize_italian_station_network(raw)
    if not network["stations"]:
        raise ValueError("catalogo AWC senza stazioni METAR italiane valide")
    return network


def fetch_italy_metar_observations(
    *,
    session=None,
    timeout: tuple[int, int] = (15, 45),
) -> dict[str, Any]:
    """Fetch live reports for the whole Italian METAR catalogue.

    A single bounded geographical query keeps the hourly collector fast.  Its
    reports are then matched against the official Italian catalogue; the API
    does not return a fabricated null report for a silent station, so the full
    catalogue is carried alongside the live subset for honest map rendering.
    """

    if session is None:
        import requests

        session = requests
    network = fetch_italy_station_network(session=session, timeout=timeout)
    station_ids = {item["id"] for item in network["stations"]}
    south, west, north, east = ITALY_DOMAIN
    response = session.get(
        SOURCE_URL,
        params={"format": "json", "bbox": f"{south},{west},{north},{east}"},
        timeout=timeout,
        headers={"User-Agent": "Meteo-Sicilia-Verification/1.0"},
    )
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, list):
        raise ValueError("risposta METAR JSON non composta da una lista")
    reports_by_id: dict[str, dict[str, Any]] = {}
    for report in raw:
        if not isinstance(report, dict):
            continue
        station_id = str(report.get("icaoId") or "").strip().upper()
        if station_id not in station_ids:
            continue
        previous = reports_by_id.get(station_id)
        if previous is None or (
            _epoch_seconds(report.get("obsTime")) or 0
        ) >= (_epoch_seconds(previous.get("obsTime")) or 0):
            reports_by_id[station_id] = report
    payload = normalize_metar_reports(
        reports_by_id.values(),
        domain=ITALY_DOMAIN,
        station_catalog=network["stations"],
    )
    payload["stationNetwork"] = network
    payload["coverage"] = {
        "country": ITALY_COUNTRY,
        "registeredMetarStations": network["count"],
        "reportingMetarStations": payload["count"],
        "reportingPolicy": (
            "Solo report METAR realmente ricevuti; una stazione registrata "
            "senza report corrente rimane visibile come non osservata."
        ),
    }
    return payload


def fetch_metar_observations(
    *,
    domain: tuple[float, float, float, float],
    session=None,
    timeout: tuple[int, int] = (15, 45),
) -> dict[str, Any]:
    """Fetch one bounded AWC snapshot with an explicit custom user agent."""

    if session is None:
        import requests

        session = requests
    south, west, north, east = domain
    response = session.get(
        SOURCE_URL,
        params={
            "format": "json",
            "bbox": f"{south},{west},{north},{east}",
        },
        timeout=timeout,
        headers={"User-Agent": "Meteo-Sicilia-Verification/1.0"},
    )
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, list):
        raise ValueError("risposta METAR JSON non composta da una lista")
    return normalize_metar_reports(raw, domain=domain)
