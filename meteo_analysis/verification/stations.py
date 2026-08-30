"""Native-grid forecast sampling at observing stations.

This archive is intentionally small: it preserves the model value interpolated
from the full 2.2-km grid at each METAR location for every forecast hour.  It
does not replace a full-field archive, but provides the first statistically
valid dataset for residual bias correction and independent scorecards.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1

FIELD_METADATA = {
    "temperature2m": {"label": "Temperatura 2 m", "unit": "degC", "decimals": 2},
    "dewpoint2m": {"label": "Punto di rugiada 2 m", "unit": "degC", "decimals": 2},
    "relativeHumidity2m": {"label": "Umidità relativa 2 m", "unit": "%", "decimals": 2},
    "pressureMsl": {"label": "Pressione media al livello del mare", "unit": "hPa", "decimals": 2},
    "windU10": {"label": "Componente zonale vento 10 m", "unit": "m s-1", "decimals": 3},
    "windV10": {"label": "Componente meridionale vento 10 m", "unit": "m s-1", "decimals": 3},
    "windGust10": {"label": "Raffica massima 10 m", "unit": "m s-1", "decimals": 3},
    "rainStep": {"label": "Precipitazione del passo", "unit": "mm", "decimals": 3},
    "cloudCover": {"label": "Copertura nuvolosa", "unit": "%", "decimals": 2},
    "terrainHeight": {"label": "Quota orografica della griglia", "unit": "m", "decimals": 1},
}


def _coordinates_and_values(values, latitudes, longitudes):
    array = np.asarray(values, dtype=float)
    latitude = np.asarray(latitudes, dtype=float)
    longitude = np.asarray(longitudes, dtype=float)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("coordinate non monodimensionali")
    if array.shape != (latitude.size, longitude.size):
        raise ValueError(
            f"forma campo {array.shape}, attesa {(latitude.size, longitude.size)}"
        )
    if latitude.size < 2 or longitude.size < 2:
        raise ValueError("griglia troppo piccola per interpolazione bilineare")
    if latitude[0] > latitude[-1]:
        latitude = latitude[::-1]
        array = array[::-1, :]
    if longitude[0] > longitude[-1]:
        longitude = longitude[::-1]
        array = array[:, ::-1]
    if not np.all(np.diff(latitude) > 0) or not np.all(np.diff(longitude) > 0):
        raise ValueError("coordinate non strettamente monotone")
    return array, latitude, longitude


def bilinear_sample(values, latitudes, longitudes, points) -> np.ndarray:
    """Sample a regular grid at ``(lat, lon)`` points.

    Missing corners are excluded and the remaining bilinear weights are
    renormalized.  A point outside the domain, or with no finite corner,
    remains NaN rather than being extrapolated or filled climatologically.
    """

    array, latitude, longitude = _coordinates_and_values(
        values, latitudes, longitudes
    )
    samples = np.full(len(points), np.nan, dtype=float)
    for position, point in enumerate(points):
        point_lat, point_lon = float(point[0]), float(point[1])
        if not (
            latitude[0] <= point_lat <= latitude[-1]
            and longitude[0] <= point_lon <= longitude[-1]
        ):
            continue
        iy = int(np.searchsorted(latitude, point_lat, side="right") - 1)
        ix = int(np.searchsorted(longitude, point_lon, side="right") - 1)
        iy = min(max(iy, 0), latitude.size - 2)
        ix = min(max(ix, 0), longitude.size - 2)
        y0, y1 = latitude[iy], latitude[iy + 1]
        x0, x1 = longitude[ix], longitude[ix + 1]
        fy = (point_lat - y0) / (y1 - y0)
        fx = (point_lon - x0) / (x1 - x0)
        corners = np.asarray([
            array[iy, ix],
            array[iy, ix + 1],
            array[iy + 1, ix],
            array[iy + 1, ix + 1],
        ])
        weights = np.asarray([
            (1.0 - fy) * (1.0 - fx),
            (1.0 - fy) * fx,
            fy * (1.0 - fx),
            fy * fx,
        ])
        finite = np.isfinite(corners)
        weight_sum = float(np.sum(weights[finite]))
        if weight_sum > 0.0:
            samples[position] = float(
                np.sum(corners[finite] * weights[finite]) / weight_sum
            )
    return samples


def _haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return radius * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _atomic_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    encoded = json.dumps(
        payload, separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode("utf-8")
    try:
        with partial.open("wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, compresslevel=7, mtime=0
            ) as output:
                output.write(encoded)
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


class StationForecastArchive:
    """Collect full-resolution ICON samples for a fixed station set."""

    def __init__(self, latitudes, longitudes, stations, *, run_time: str):
        self.latitudes = np.asarray(latitudes, dtype=float)
        self.longitudes = np.asarray(longitudes, dtype=float)
        if self.latitudes.ndim != 1 or self.longitudes.ndim != 1:
            raise ValueError("coordinate di griglia non monodimensionali")
        self.source_shape = (self.latitudes.size, self.longitudes.size)
        self.run_time = str(run_time)
        self.stations = []
        seen = set()
        for source in stations or []:
            station_id = str(source.get("id") or "").strip().upper()
            try:
                station_lat = float(source["lat"])
                station_lon = float(source["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            if not station_id or station_id in seen:
                continue
            seen.add(station_id)
            nearest_y = int(np.nanargmin(np.abs(self.latitudes - station_lat)))
            nearest_x = int(np.nanargmin(np.abs(self.longitudes - station_lon)))
            self.stations.append({
                "id": station_id,
                "name": str(source.get("name") or ""),
                "lat": round(station_lat, 5),
                "lon": round(station_lon, 5),
                "elevationM": source.get("elevationM"),
                "nearestGridDistanceKm": round(_haversine_km(
                    station_lat,
                    station_lon,
                    float(self.latitudes[nearest_y]),
                    float(self.longitudes[nearest_x]),
                ), 3),
            })
        self.times: dict[int, str] = {}
        self.data: dict[str, dict[int, list[float | None]]] = {}

    def add(self, lead_hours: int, valid_time: str, fields: dict[str, Any]) -> None:
        lead = int(lead_hours)
        if lead in self.times:
            return
        points = [(item["lat"], item["lon"]) for item in self.stations]
        self.times[lead] = str(valid_time)
        for name, values in fields.items():
            if name not in FIELD_METADATA or values is None:
                continue
            array = np.asarray(values, dtype=float)
            if array.shape != self.source_shape:
                continue
            sampled = bilinear_sample(
                array, self.latitudes, self.longitudes, points
            )
            decimals = int(FIELD_METADATA[name]["decimals"])
            self.data.setdefault(name, {})[lead] = [
                None if not np.isfinite(value) else round(float(value), decimals)
                for value in sampled
            ]

    def write(self, path) -> dict[str, Any]:
        if not self.times:
            raise ValueError("nessuna scadenza campionata")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        hours = sorted(self.times)
        fields: dict[str, Any] = {}
        for name in sorted(self.data):
            metadata = FIELD_METADATA[name]
            values_by_station = []
            for station_index in range(len(self.stations)):
                values_by_station.append([
                    self.data[name].get(hour, [None] * len(self.stations))[station_index]
                    for hour in hours
                ])
            fields[name] = {
                "label": metadata["label"],
                "unit": metadata["unit"],
                "valuesByStation": values_by_station,
            }
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "model": "ICON-2I",
            "runTime": self.run_time,
            "nominalResolutionKm": 2.2,
            "sampling": "bilinear-native-grid-finite-weight-renormalized",
            "missingDataPolicy": "preserve-null-no-extrapolation-no-climatology",
            "stations": self.stations,
            "times": [
                {"leadHours": hour, "validTime": self.times[hour]}
                for hour in hours
            ],
            "fields": fields,
        }
        _atomic_gzip_json(path, payload)
        return payload
