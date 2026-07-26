"""Compact, tiled ICON-2I time series for the static meteogram page."""

from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_SPACING_DEG = 0.20
DEFAULT_TILE_SIZE_DEG = 2.0

FIELD_METADATA = {
    "temperature2m": ("Temperatura 2 m", "°C", 1),
    "feelsLike": ("Temperatura percepita", "°C", 1),
    "rainStep": ("Precipitazione oraria", "mm", 2),
    "pressureMsl": ("Pressione al livello del mare", "hPa", 1),
    "relativeHumidity2m": ("Umidità relativa 2 m", "%", 0),
    "cloudCover": ("Copertura nuvolosa", "%", 0),
    "windU10": ("Componente zonale vento 10 m", "m/s", 1),
    "windV10": ("Componente meridionale vento 10 m", "m/s", 1),
    "convectionProbability": ("Probabilità temporali", "%", 1),
    "capeMl": ("ML-CAPE", "J/kg", 0),
    "cinMl": ("ML-CIN", "J/kg", 0),
    "omega700": ("Moto verticale 700 hPa", "Pa/s", 3),
    "frontDistanceKm": ("Distanza dal fronte", "km", 0),
    "visibility": ("Visibilità stimata", "m", 0),
    "fogProbability": ("Probabilità di nebbia", "%", 0),
    "freezingRainRisk": ("Rischio gelicidio", "classe", 0),
    "foehnIndex": ("Indice di foehn", "classe", 0),
    "temperature925": ("Temperatura 925 hPa", "°C", 1),
    "temperature850": ("Temperatura 850 hPa", "°C", 1),
    "temperature700": ("Temperatura 700 hPa", "°C", 1),
    "relativeHumidity925": ("Umidità relativa 925 hPa", "%", 0),
    "relativeHumidity850": ("Umidità relativa 850 hPa", "%", 0),
    "windU925": ("Componente zonale vento 925 hPa", "m/s", 1),
    "windV925": ("Componente meridionale vento 925 hPa", "m/s", 1),
    "windU850": ("Componente zonale vento 850 hPa", "m/s", 1),
    "windV850": ("Componente meridionale vento 850 hPa", "m/s", 1),
    "thetaW850": ("Theta-w 850 hPa", "K", 1),
    "thetaE850": ("Theta-e 850 hPa", "K", 1),
}

UPPER_FIELD_MAP = {
    "925": {
        "t": "temperature925",
        "rh": "relativeHumidity925",
        "u": "windU925",
        "v": "windV925",
    },
    "850": {
        "t": "temperature850",
        "rh": "relativeHumidity850",
        "u": "windU850",
        "v": "windV850",
        "thetaW": "thetaW850",
        "thetaE": "thetaE850",
    },
}


def _temperature_celsius(values):
    result = np.asarray(values, dtype=float)
    finite = result[np.isfinite(result)]
    if finite.size and float(np.nanmedian(finite)) > 150.0:
        result = result - 273.15
    return result


def _atomic_gzip_json(path: Path, payload):
    partial = path.with_suffix(path.suffix + ".part")
    encoded = json.dumps(
        payload, separators=(",", ":"), allow_nan=False
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


def _strict_values(values, decimals):
    rounded = np.round(np.asarray(values, dtype=float), decimals)
    return [
        None if not np.isfinite(float(value)) else float(value)
        for value in rounded.ravel()
    ]


class MeteogramArchive:
    """Collect coarse time slices and emit small 2° tiles for GitHub Pages."""

    def __init__(
        self,
        latitudes,
        longitudes,
        *,
        run_time,
        spacing_deg=DEFAULT_SPACING_DEG,
        tile_size_deg=DEFAULT_TILE_SIZE_DEG,
    ):
        latitudes = np.asarray(latitudes, dtype=float)
        longitudes = np.asarray(longitudes, dtype=float)
        if latitudes.ndim != 1 or longitudes.ndim != 1:
            raise ValueError("coordinate meteogrammi non monodimensionali")
        if latitudes.size < 2 or longitudes.size < 2:
            raise ValueError("griglia meteogrammi troppo piccola")

        lat_step = max(
            1,
            int(round(spacing_deg / float(np.nanmedian(np.abs(np.diff(latitudes)))))),
        )
        lon_step = max(
            1,
            int(round(spacing_deg / float(np.nanmedian(np.abs(np.diff(longitudes)))))),
        )
        self.y_indices = np.arange(0, latitudes.size, lat_step, dtype=int)
        self.x_indices = np.arange(0, longitudes.size, lon_step, dtype=int)
        if self.y_indices[-1] != latitudes.size - 1:
            self.y_indices = np.append(self.y_indices, latitudes.size - 1)
        if self.x_indices[-1] != longitudes.size - 1:
            self.x_indices = np.append(self.x_indices, longitudes.size - 1)

        self.latitudes = latitudes[self.y_indices]
        self.longitudes = longitudes[self.x_indices]
        self.source_shape = (latitudes.size, longitudes.size)
        self.run_time = str(run_time)
        self.spacing_deg = float(spacing_deg)
        self.tile_size_deg = float(tile_size_deg)
        self.entries = {}

    def _sample_surface(self, values):
        if values is None:
            return np.full(
                (self.latitudes.size, self.longitudes.size), np.nan, dtype=np.float32
            )
        array = np.asarray(values, dtype=float)
        if array.shape != self.source_shape:
            return np.full(
                (self.latitudes.size, self.longitudes.size), np.nan, dtype=np.float32
            )
        return np.asarray(
            array[np.ix_(self.y_indices, self.x_indices)], dtype=np.float32
        )

    def _sample_upper(self, payload, key):
        if not payload or key not in payload:
            return np.full(
                (self.latitudes.size, self.longitudes.size), np.nan, dtype=np.float32
            )
        nx = int(payload.get("nx", 0))
        ny = int(payload.get("ny", 0))
        raw = np.asarray(
            [np.nan if value is None else value for value in payload[key]], dtype=float
        )
        if nx < 2 or ny < 2 or raw.size != nx * ny:
            return np.full(
                (self.latitudes.size, self.longitudes.size), np.nan, dtype=np.float32
            )
        grid = raw.reshape(ny, nx)
        dx = abs(float(payload["dx"]))
        dy = abs(float(payload["dy"]))
        x = np.rint((self.longitudes - float(payload["lo1"])) / dx).astype(int)
        y = np.rint((float(payload["la1"]) - self.latitudes) / dy).astype(int)
        x = np.clip(x, 0, nx - 1)
        y = np.clip(y, 0, ny - 1)
        return np.asarray(grid[np.ix_(y, x)], dtype=np.float32)

    def add(self, lead_hours, valid_time, fields, *, upper_air=None):
        lead_hours = int(lead_hours)
        if lead_hours in self.entries:
            return
        sampled = {
            name: self._sample_surface(values)
            for name, values in fields.items()
            if name in FIELD_METADATA
        }
        levels = (upper_air or {}).get("levels", {})
        for level, mapping in UPPER_FIELD_MAP.items():
            payload = levels.get(level)
            for source, destination in mapping.items():
                if destination not in sampled:
                    sampled[destination] = self._sample_upper(payload, source)
        self.entries[lead_hours] = {
            "validTime": str(valid_time),
            "fields": sampled,
        }

    def _tile_groups(self):
        lon_origin = math.floor(float(np.nanmin(self.longitudes)))
        lat_origin = math.floor(float(np.nanmin(self.latitudes)))
        x_tiles = np.floor(
            (self.longitudes - lon_origin) / self.tile_size_deg
        ).astype(int)
        y_tiles = np.floor(
            (self.latitudes - lat_origin) / self.tile_size_deg
        ).astype(int)
        for tile_y in sorted(set(y_tiles.tolist())):
            rows = np.flatnonzero(y_tiles == tile_y)
            for tile_x in sorted(set(x_tiles.tolist())):
                columns = np.flatnonzero(x_tiles == tile_x)
                if rows.size and columns.size:
                    yield lon_origin, lat_origin, tile_x, tile_y, rows, columns

    def write(self, directory):
        if not self.entries:
            raise ValueError("nessuna scadenza per i meteogrammi")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        ordered_hours = sorted(self.entries)
        field_names = sorted({
            name
            for entry in self.entries.values()
            for name in entry["fields"]
        })
        tiles = []
        for lon_origin, lat_origin, tile_x, tile_y, rows, columns in self._tile_groups():
            filename = f"tile_{tile_x}_{tile_y}.json.gz"
            payload = {
                "schemaVersion": SCHEMA_VERSION,
                "model": "ICON-2I",
                "runTime": self.run_time,
                "grid": {
                    "latitudes": np.round(self.latitudes[rows], 4).tolist(),
                    "longitudes": np.round(self.longitudes[columns], 4).tolist(),
                },
                "times": [
                    {
                        "leadHours": hour,
                        "validTime": self.entries[hour]["validTime"],
                    }
                    for hour in ordered_hours
                ],
                "fields": {},
            }
            for name in field_names:
                label, unit, decimals = FIELD_METADATA[name]
                values = []
                for hour in ordered_hours:
                    field = self.entries[hour]["fields"].get(name)
                    if field is None:
                        tile = np.full((rows.size, columns.size), np.nan)
                    else:
                        tile = field[np.ix_(rows, columns)]
                    values.append(_strict_values(tile, decimals))
                payload["fields"][name] = {
                    "label": label,
                    "unit": unit,
                    "values": values,
                }
            _atomic_gzip_json(directory / filename, payload)
            tiles.append({
                "x": int(tile_x),
                "y": int(tile_y),
                "file": filename,
                "west": round(float(self.longitudes[columns].min()), 4),
                "east": round(float(self.longitudes[columns].max()), 4),
                "south": round(float(self.latitudes[rows].min()), 4),
                "north": round(float(self.latitudes[rows].max()), 4),
            })

        manifest = {
            "schemaVersion": SCHEMA_VERSION,
            "model": "ICON-2I",
            "runTime": self.run_time,
            "hours": ordered_hours,
            "spacingDegrees": self.spacing_deg,
            "tileSizeDegrees": self.tile_size_deg,
            "longitudeOrigin": math.floor(float(np.nanmin(self.longitudes))),
            "latitudeOrigin": math.floor(float(np.nanmin(self.latitudes))),
            "domain": {
                "west": round(float(np.nanmin(self.longitudes)), 4),
                "east": round(float(np.nanmax(self.longitudes)), 4),
                "south": round(float(np.nanmin(self.latitudes)), 4),
                "north": round(float(np.nanmax(self.latitudes)), 4),
            },
            "fields": {
                name: {
                    "label": FIELD_METADATA[name][0],
                    "unit": FIELD_METADATA[name][1],
                }
                for name in field_names
            },
            "tiles": tiles,
        }
        manifest_path = directory / "catalog.json"
        partial = manifest_path.with_suffix(".json.part")
        try:
            partial.write_text(
                json.dumps(manifest, separators=(",", ":"), allow_nan=False),
                encoding="utf-8",
            )
            os.replace(partial, manifest_path)
        finally:
            if partial.exists():
                partial.unlink()
        return manifest
