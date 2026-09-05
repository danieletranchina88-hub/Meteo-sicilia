"""Public MeteoHub Sicily surface observations, normalized for the site.

Contract checked against MeteoHub maps_observed.py and a real JSONL export.
No account credentials are required for this recent public dataset.
"""
from datetime import datetime, timedelta, timezone
import json
import math

URL = "https://meteohub.agenziaitaliameteo.it/api/observations"
SOURCE = "MeteoHub · Regione Siciliana · Agenzia ItaliaMeteo / CINECA"
NETWORK = "dpcn-sicilia"
PRODUCTS = (
    "B12101",  # air temperature at 2 m
    "B12103",  # dew-point temperature at 2 m
    "B13003",  # relative humidity
    "B11001",  # wind direction
    "B11002",  # wind speed
    "B11041",  # maximum wind gust
    "B11042",  # maximum wind gust (alternate descriptor)
    "B10051",  # pressure reduced to mean sea level
    "B10004",  # station pressure (published separately, never as MSLP)
)


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
    field_times = {}
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or record.get("network") != NETWORK:
            continue
        metadata = {}
        observed_values = {}
        for group in record.get("data", []):
            values = group.get("vars", {})
            if "level" not in group:
                metadata.update(values)
            else:
                level = group.get("level", [])
                timerange = group.get("timerange")
                for code in PRODUCTS:
                    if code in {"B12101", "B12103", "B13003"} and not (
                        level[:2] == [103, 2000] and timerange == [254, 0, 0]
                    ):
                        continue
                    if code in {"B11001", "B11002", "B11041", "B11042"} and not (
                        level[:2] in ([103, 10000], [1, 0])
                    ):
                        continue
                    if code == "B10051" and level[:2] not in ([1, 0], [101, 0]):
                        continue
                    if code == "B10004":
                        level_type = number(level[0]) if level else None
                        level_value = number(level[1]) if len(level) > 1 else None
                        station_level = (
                            (level_type == 1 and level_value == 0)
                            or (level_type == 102 and level_value is not None
                                and -500_000 <= level_value <= 4_000_000)
                            or (level_type == 103 and level_value is not None
                                and 0 <= level_value <= 10_000)
                        )
                        if not station_level:
                            continue
                    value = number((values.get(code) or {}).get("v"))
                    if value is not None:
                        observed_values[code] = value
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
        temperature = observed_values.get("B12101")
        if temperature is not None and temperature > 120:
            temperature -= 273.15
        if temperature is not None and not -50 <= temperature <= 55:
            temperature = None
        dewpoint = observed_values.get("B12103")
        if dewpoint is not None and dewpoint > 120:
            dewpoint -= 273.15
        if dewpoint is not None and not -70 <= dewpoint <= 45:
            dewpoint = None
        humidity = observed_values.get("B13003")
        if humidity is not None and not 0 <= humidity <= 100:
            humidity = None
        wind_speed = observed_values.get("B11002")
        wind_direction = observed_values.get("B11001")
        gust = observed_values.get("B11041", observed_values.get("B11042"))
        wind_speed = wind_speed * 3.6 if wind_speed is not None and 0 <= wind_speed <= 70 else None
        gust = gust * 3.6 if gust is not None and 0 <= gust <= 100 else None
        if wind_direction is not None and not 0 <= wind_direction <= 360:
            wind_direction = None
        def pressure_hpa(code):
            value = observed_values.get(code)
            if value is None:
                return None
            value = value / 100 if value > 2000 else value
            return value if 800 <= value <= 1100 else None
        sea_level_pressure = pressure_hpa("B10051")
        station_pressure = pressure_hpa("B10004")
        if all(value is None for value in (
            temperature, dewpoint, humidity, wind_speed, gust,
            wind_direction, sea_level_pressure, station_pressure,
        )):
            continue
        sid = f"MH:SICILIA:{lat:.5f}:{lon:.5f}"
        elevation = number(value("B07030"))
        # The real export uses zero for many unknown heights. Keep these
        # visible, but do not treat them as verified sea-level stations.
        if elevation is None or not 0 < elevation <= 4000:
            elevation = None
        observed_epoch = int(observed.timestamp())
        station = dict(id=sid, name=str(value("B01019") or sid), lat=lat, lon=lon,
                       elevationM=elevation, obsTime=observed_epoch,
                       source=SOURCE, quality="provisional", network=NETWORK)
        optional = {
            "tempC": temperature, "dewpC": dewpoint, "rhPct": humidity,
            "wspdKmh": wind_speed, "windGustKmh": gust,
            "wdir": wind_direction, "pressHpa": sea_level_pressure,
            "stationPressureHpa": station_pressure,
        }
        if sid not in latest:
            latest[sid] = station
            field_times[sid] = {}
        elif observed_epoch > latest[sid]["obsTime"]:
            # A MeteoHub station can arrive as one JSONL record per product.
            # Refresh station metadata without discarding the other products.
            latest[sid].update(name=station["name"], lat=lat, lon=lon,
                               obsTime=observed_epoch)
            if elevation is not None:
                latest[sid]["elevationM"] = elevation
        for key, item in optional.items():
            if item is None or observed_epoch < field_times[sid].get(key, -1):
                continue
            latest[sid][key] = round(item, 2)
            field_times[sid][key] = observed_epoch
    for sid, station in latest.items():
        station["fieldObsTime"] = field_times[sid]
        station["obsTime"] = max(field_times[sid].values())
    return sorted(latest.values(), key=lambda s: s["id"])


def fetch_meteohub_stations(session=None, now=None):
    if session is None:
        import requests
        session = requests
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(hours=2)
    base_query = (f"reftime:>={start:%Y-%m-%d %H:%M},<={now:%Y-%m-%d %H:%M};"
                  "license:CCBY_COMPLIANT")
    # The public map serves pressure only after selecting that product even
    # when allStationProducts=true. Ask B10004/B10051 explicitly and merge the
    # JSONL records with the broad temperature/humidity/wind response.
    lines = []
    size = 0
    for product in (None, "B10004", "B10051"):
        query = base_query + (f";product:{product}" if product else "")
        response = session.post(URL, params={
            "q": query, "networks": NETWORK, "output_format": "JSON",
            "stationDetails": "true",
            "allStationProducts": "true" if product is None else "false",
            "reliabilityCheck": "true", "lonmin": 11, "lonmax": 16,
            "latmin": 35, "latmax": 39,
        }, timeout=(15, 60), stream=True)
        try:
            response.raise_for_status()
            for line in response.iter_lines():
                size += len(line)
                if size > 80_000_000:
                    raise ValueError("export MeteoHub troppo grande")
                lines.append(line)
        except Exception:
            if product is None:
                raise
        finally:
            response.close()
    stations = normalize_jsonl(lines, now)
    if not stations:
        raise ValueError("nessuna osservazione MeteoHub recente e utilizzabile")
    return stations


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
        response = getattr(error, "response", None)
        reason = type(error).__name__
        if response is not None:
            reason += f" HTTP {response.status_code}"
        statuses.append(dict(source=SOURCE, status="unavailable", reason=reason))
    if not payload["stations"]:
        raise ValueError("nessuna fonte osservativa disponibile")
    payload.update(source="NOAA AWC METAR + MeteoHub Sicilia", sourceUrl=URL,
                   capturedAt=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                   count=len(payload["stations"]), sourceStatus=statuses,
                   obsTime=max(s.get("obsTime") or 0 for s in payload["stations"]))
    payload["stationNetwork"]["count"] = len(payload["stationNetwork"]["stations"])
    payload["stationNetwork"]["source"] = payload["source"]
    payload["stationNetwork"]["coverageNote"] = "Catalogo METAR; MeteoHub: stazioni siciliane con almeno un parametro ricevuto nelle ultime due ore, non catalogo completo."
    return payload
