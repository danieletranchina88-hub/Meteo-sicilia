"""Public MeteoHub Sicily temperature export, normalized for the site.

Contract checked against MeteoHub maps_observed.py and a real JSONL export.
No account credentials are required for this recent public dataset.
"""
from datetime import datetime, timedelta, timezone
import json
import math

URL = "https://meteohub.agenziaitaliameteo.it/api/observations"
SOURCE = "MeteoHub · Regione Siciliana · Agenzia ItaliaMeteo / CINECA"
NETWORK = "dpcn-sicilia"


def number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def normalize_jsonl(lines, now=None):
    now = now or datetime.now(timezone.utc)
    latest = {}
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or record.get("network") != NETWORK:
            continue
        metadata = {}
        temperatures = []
        for group in record.get("data", []):
            values = group.get("vars", {})
            if "level" not in group:
                metadata.update(values)
            elif (group.get("level", [])[:2] == [103, 2000]
                  and group.get("timerange") == [254, 0, 0]):
                value = number((values.get("B12101") or {}).get("v"))
                if value is not None:
                    temperatures.append(value - 273.15)
        def value(key):
            return (metadata.get(key) or {}).get("v")
        lat, lon = number(value("B05001")), number(value("B06001"))
        if lat is None or lon is None or not (35 <= lat <= 39 and 11 <= lon <= 16):
            continue
        try:
            observed = datetime.fromisoformat(record["date"].replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            continue
        if observed.tzinfo is None or not 0 <= (now - observed).total_seconds() <= 7200:
            continue
        if len(temperatures) != 1 or not -50 <= temperatures[0] <= 55:
            continue
        sid = f"MH:SICILIA:{lat:.5f}:{lon:.5f}"
        elevation = number(value("B07030"))
        # The real export uses zero for many unknown heights. Keep these
        # visible, but do not treat them as verified sea-level stations.
        if elevation is None or not 0 < elevation <= 4000:
            elevation = None
        station = dict(id=sid, name=str(value("B01019") or sid), lat=lat, lon=lon,
                       elevationM=elevation, obsTime=int(observed.timestamp()),
                       tempC=round(temperatures[0], 2), source=SOURCE,
                       quality="provisional", network=NETWORK)
        if sid not in latest or station["obsTime"] > latest[sid]["obsTime"]:
            latest[sid] = station
    return sorted(latest.values(), key=lambda s: s["id"])


def fetch_meteohub_stations(session=None, now=None):
    if session is None:
        import requests
        session = requests
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(hours=2)
    query = (f"reftime:>={start:%Y-%m-%d %H:%M},<={now:%Y-%m-%d %H:%M};"
             "product:B12101;license:CCBY_COMPLIANT")
    response = session.post(URL, params={
        "q": query, "networks": NETWORK, "output_format": "json",
        "reliabilityCheck": "true", "lonmin": 11, "lonmax": 16,
        "latmin": 35, "latmax": 39,
    }, timeout=(15, 60), stream=True)
    try:
        response.raise_for_status()
        size = 0
        def lines():
            nonlocal size
            for line in response.iter_lines():
                size += len(line)
                if size > 20_000_000:
                    raise ValueError("export MeteoHub troppo grande")
                yield line
        stations = normalize_jsonl(lines(), now)
        if not stations:
            raise ValueError("nessuna temperatura MeteoHub recente e utilizzabile")
        return stations
    finally:
        response.close()


def fetch_site_observations():
    from .observations import fetch_italy_metar_observations
    now = datetime.now(timezone.utc)
    statuses = []
    try:
        payload = fetch_italy_metar_observations()
        statuses.append(dict(source="NOAA AWC METAR", status="ok", count=payload["count"]))
    except Exception as error:
        payload = {"schemaVersion": 3, "stations": [], "stationNetwork": {"stations": []}}
        statuses.append(dict(source="NOAA AWC METAR", status="unavailable", reason=type(error).__name__))
    for station in payload["stations"]:
        station["source"] = "NOAA AWC METAR"
    try:
        stations = fetch_meteohub_stations(now=now)
        payload["stations"].extend(stations)
        payload["stationNetwork"]["stations"].extend([
            {k: s[k] for k in ("id", "name", "lat", "lon", "elevationM", "source")}
            for s in stations
        ])
        statuses.append(dict(source=SOURCE, status="ok", count=len(stations)))
    except Exception as error:
        statuses.append(dict(source=SOURCE, status="unavailable", reason=type(error).__name__))
    if not payload["stations"]:
        raise ValueError("nessuna fonte osservativa disponibile")
    payload.update(source="NOAA AWC METAR + MeteoHub Sicilia", sourceUrl=URL,
                   capturedAt=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                   count=len(payload["stations"]), sourceStatus=statuses,
                   obsTime=max(s.get("obsTime") or 0 for s in payload["stations"]))
    payload["stationNetwork"]["count"] = len(payload["stationNetwork"]["stations"])
    payload["stationNetwork"]["source"] = payload["source"]
    payload["stationNetwork"]["coverageNote"] = "Catalogo METAR; MeteoHub: stazioni con temperatura ricevuta nelle ultime due ore, non catalogo completo."
    return payload
