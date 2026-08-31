"""Territorial coverage metrics for the station network.

Implements a simplified version of requirement 11: Italy is divided into a
regular lat/lon grid (default ~10 km) and, for each cell, the distance to the
nearest station and the station counts within 5/10/20/50 km are computed.
The grid is coarser than the 5x5 km suggested in the specification to keep
the computation fast in pure NumPy without a spatial index; the resolution
is a constructor parameter so it can be tightened once the pipeline runs on
a real scheduler with more CPU budget.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ITALY_BBOX = {"south": 35.2, "north": 47.2, "west": 6.5, "east": 18.6}

_RADII_KM = (5.0, 10.0, 20.0, 50.0)


def _haversine_km_matrix(
    grid_lat: np.ndarray, grid_lon: np.ndarray, station_lat: np.ndarray, station_lon: np.ndarray
) -> np.ndarray:
    radius = 6371.0088
    lat1 = np.radians(grid_lat)[:, None]
    lat2 = np.radians(station_lat)[None, :]
    dlat = lat2 - lat1
    dlon = np.radians(station_lon)[None, :] - np.radians(grid_lon)[:, None]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return radius * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))


def compute_coverage(
    stations: list[dict[str, Any]],
    *,
    variable: str | None = None,
    grid_step_deg: float = 0.1,
    bbox: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return grid coverage statistics, optionally restricted to a variable.

    When ``variable`` is given only stations that actually measure it (i.e.
    carry a finite value for that canonical variable) count towards the
    density metrics, so gaps are computed per-parameter as requested.
    """

    bbox = bbox or ITALY_BBOX
    if variable is not None:
        stations = [
            station for station in stations
            if isinstance((station.get("observations") or {}).get(variable), dict)
            and (station["observations"][variable] or {}).get("value") is not None
        ]

    if not stations:
        return {
            "variable": variable,
            "gridStepDeg": grid_step_deg,
            "cells": 0,
            "cellsWithoutStationWithin20km": None,
            "meanNearestDistanceKm": None,
            "maxNearestDistanceKm": None,
        }

    station_lat = np.asarray([station["lat"] for station in stations], dtype=float)
    station_lon = np.asarray([station["lon"] for station in stations], dtype=float)

    grid_lat = np.arange(bbox["south"], bbox["north"], grid_step_deg)
    grid_lon = np.arange(bbox["west"], bbox["east"], grid_step_deg)
    grid_lat_mesh, grid_lon_mesh = np.meshgrid(grid_lat, grid_lon, indexing="ij")
    flat_lat = grid_lat_mesh.ravel()
    flat_lon = grid_lon_mesh.ravel()

    distances = _haversine_km_matrix(flat_lat, flat_lon, station_lat, station_lon)
    nearest = distances.min(axis=1)

    counts_within = {
        radius: int(np.count_nonzero((distances <= radius).any(axis=1)))
        for radius in _RADII_KM
    }
    gap_cells = int(np.count_nonzero(nearest > 20.0))

    return {
        "variable": variable,
        "gridStepDeg": grid_step_deg,
        "bbox": bbox,
        "cells": int(flat_lat.size),
        "cellsWithStationWithin5km": counts_within[5.0],
        "cellsWithStationWithin10km": counts_within[10.0],
        "cellsWithStationWithin20km": counts_within[20.0],
        "cellsWithStationWithin50km": counts_within[50.0],
        "cellsWithoutStationWithin20km": gap_cells,
        "coverageGapFraction20km": round(gap_cells / flat_lat.size, 4),
        "meanNearestDistanceKm": round(float(nearest.mean()), 2),
        "maxNearestDistanceKm": round(float(nearest.max()), 2),
    }


def compute_coverage_report(
    stations: list[dict[str, Any]], *, grid_step_deg: float = 0.1
) -> dict[str, Any]:
    """Coverage for the whole network plus one report per key variable."""

    key_variables = [
        "temperature", "precipitation", "windSpeed", "relativeHumidity", "pressureMsl",
    ]
    return {
        "overall": compute_coverage(stations, grid_step_deg=grid_step_deg),
        "byVariable": {
            variable: compute_coverage(
                stations, variable=variable, grid_step_deg=grid_step_deg
            )
            for variable in key_variables
        },
    }
