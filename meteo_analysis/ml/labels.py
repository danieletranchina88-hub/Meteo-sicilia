"""Geodesic gridding of manually analysed frontal polylines."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .features import EARTH_RADIUS_M, regular_grid

FRONT_CLASSES = {
    "cold": 1,
    "warm": 2,
    "occluded": 3,
    "stationary": 4,
}


def _unit_sphere(latitude, longitude):
    lat = np.deg2rad(np.asarray(latitude, float))
    lon = np.deg2rad(np.asarray(longitude, float))
    return np.column_stack((
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ))


def _densify(coords: Iterable[Iterable[float]], spacing_km=10.0):
    """Densify a lon/lat line enough for a nearest-neighbour distance mask."""
    coords = np.asarray(list(coords), float)
    if len(coords) < 2:
        return coords
    result = [coords[0]]
    for first, second in zip(coords[:-1], coords[1:]):
        mean_lat = np.deg2rad((first[1] + second[1]) / 2.0)
        dx = (second[0] - first[0]) * 111.32 * np.cos(mean_lat)
        dy = (second[1] - first[1]) * 110.57
        count = max(1, int(np.ceil(np.hypot(dx, dy) / spacing_km)))
        result.extend(
            first + (second - first) * fraction
            for fraction in np.linspace(1.0 / count, 1.0, count)
        )
    return np.asarray(result)


def grid_labels(
    fronts,
    *,
    valid_time,
    bounds=None,
    resolution=0.20,
    distance_km=40.0,
    multiclass=False,
):
    """Return one label for every point of the regular inference grid.

    ``fronts`` accepts GeoJSON-like mappings with ``type``/``tipo`` and a
    ``coordinates`` member, so the routine does not require GeoPandas.
    Overlapping class buffers are resolved using the physically nearest line.
    """
    latitude, longitude = regular_grid(
        bounds=bounds or (3.0, 22.0, 33.7, 48.9), resolution=resolution
    )
    lon2d, lat2d = np.meshgrid(longitude, latitude)
    points = _unit_sphere(lat2d.ravel(), lon2d.ravel())
    best_distance = np.full(points.shape[0], np.inf)
    labels = np.zeros(points.shape[0], dtype=np.uint8)
    chord_limit = 2.0 * np.sin((distance_km * 1000.0 / EARTH_RADIUS_M) / 2.0)

    for front in fronts:
        kind = str(front.get("tipo", front.get("type", ""))).lower()
        kind = {"occ": "occluded", "occlusion": "occluded"}.get(kind, kind)
        code = FRONT_CLASSES.get(kind)
        if code is None:
            continue
        coordinates = front.get("coordinates")
        if coordinates is None and isinstance(front.get("geometry"), dict):
            coordinates = front["geometry"].get("coordinates")
        dense = _densify(coordinates or [])
        if len(dense) < 2:
            continue
        tree = cKDTree(_unit_sphere(dense[:, 1], dense[:, 0]))
        distance, _ = tree.query(points, k=1)
        closer = (distance <= chord_limit) & (distance < best_distance)
        labels[closer] = code if multiclass else 1
        best_distance[closer] = distance[closer]

    timestamp = pd.Timestamp(valid_time)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return pd.DataFrame({
        "time": timestamp,
        "lat": lat2d.ravel(),
        "lon": lon2d.ravel(),
        "y": labels,
    })
