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
SCHEMA_VERSION = 2


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
) -> dict[str, Any]:
    """Return strict, unit-explicit observations inside ``domain``.

    ``domain`` is ``(south, west, north, east)``.  A timestamp is retained per
    station; the payload-level timestamp is only the latest included report
    and must never be assigned to every station during verification.
    """

    south, west, north, east = (float(value) for value in domain)
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
        observed_at = _epoch_seconds(report.get("obsTime"))
        if observed_at is not None:
            latest = max(latest, observed_at)

        wind_kt = _number(report, "wspd")
        gust_kt = _number(report, "wgst")
        slp = _sea_level_pressure_hpa(_number(report, "slp"))
        altimeter = _altimeter_hpa(_number(report, "altim"))
        elevation = _number(report, "elev")
        if elevation is not None and not (-500.0 <= elevation <= 9000.0):
            elevation = None

        stations.append({
            "id": station_id,
            "name": str(report.get("name") or ""),
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
