"""Geodesic terrain context and strict downscaling sample construction."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from .schemas import STATIC_DOWNSCALING_FEATURES

EARTH_RADIUS_M = 6_371_000.0


def _unit_sphere(latitude, longitude):
    latitude = np.deg2rad(np.asarray(latitude, dtype=np.float64))
    longitude = np.deg2rad(np.asarray(longitude, dtype=np.float64))
    return np.column_stack((
        np.cos(latitude) * np.cos(longitude),
        np.cos(latitude) * np.sin(longitude),
        np.sin(latitude),
    ))


def terrain_context(elevation_m, land_fraction, latitude, longitude):
    """Return terrain derivatives, coast distance and spherical cell area."""
    elevation = np.asarray(elevation_m, dtype=np.float64)
    land_fraction = np.asarray(land_fraction, dtype=np.float64)
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    expected = (latitude.size, longitude.size)
    if elevation.shape != expected or land_fraction.shape != expected:
        raise ValueError("DEM/terra non allineati alla griglia ad alta risoluzione")
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("coordinate terrain non monodimensionali")
    if latitude.size < 3 or longitude.size < 3:
        raise ValueError("servono almeno tre celle per asse")
    if not (
        np.all(np.isfinite(elevation)) and np.all(np.isfinite(land_fraction))
        and np.all(np.diff(latitude) > 0) and np.all(np.diff(longitude) > 0)
        and np.min(land_fraction) >= 0 and np.max(land_fraction) <= 1
    ):
        raise ValueError("terrain non finito, non ordinato o fuori scala")

    y_m = np.deg2rad(latitude - latitude[0]) * EARTH_RADIUS_M
    dx_rad = float(np.median(np.diff(np.deg2rad(longitude))))
    dx_m = EARTH_RADIUS_M * np.cos(np.deg2rad(latitude)) * dx_rad
    slope_north = np.gradient(elevation, y_m, axis=0)
    slope_east = np.divide(
        np.gradient(elevation, axis=1), dx_m[:, None],
        out=np.zeros_like(elevation), where=np.abs(dx_m[:, None]) > 1.0,
    )

    land = land_fraction >= 0.5
    coast = (land & binary_dilation(~land)) | (~land & binary_dilation(land))
    if not np.any(coast):
        raise ValueError("distanza dalla costa indeterminabile: maschera uniforme")
    lon2d, lat2d = np.meshgrid(longitude, latitude)
    coast_tree = cKDTree(_unit_sphere(lat2d[coast], lon2d[coast]))
    chord, _ = coast_tree.query(
        _unit_sphere(lat2d.ravel(), lon2d.ravel()), workers=-1
    )
    distance_km = (
        2.0 * EARTH_RADIUS_M
        * np.arcsin(np.clip(chord, 0.0, 2.0) / 2.0) / 1000.0
    ).reshape(expected)
    latitude_edges = np.empty(latitude.size + 1, dtype=np.float64)
    longitude_edges = np.empty(longitude.size + 1, dtype=np.float64)
    latitude_edges[1:-1] = (latitude[:-1] + latitude[1:]) / 2.0
    longitude_edges[1:-1] = (longitude[:-1] + longitude[1:]) / 2.0
    latitude_edges[0] = latitude[0] - (latitude[1] - latitude[0]) / 2.0
    latitude_edges[-1] = latitude[-1] + (latitude[-1] - latitude[-2]) / 2.0
    longitude_edges[0] = longitude[0] - (longitude[1] - longitude[0]) / 2.0
    longitude_edges[-1] = longitude[-1] + (
        longitude[-1] - longitude[-2]
    ) / 2.0
    latitude_factor = np.abs(np.diff(np.sin(np.deg2rad(latitude_edges))))
    longitude_width = np.abs(np.diff(np.deg2rad(longitude_edges)))
    cell_area = EARTH_RADIUS_M ** 2 * np.outer(latitude_factor, longitude_width)
    return np.stack((
        elevation, slope_east, slope_north, land_fraction, distance_km,
    )).astype(np.float32), cell_area.astype(np.float32)


def build_orographic_sample(
    coarse,
    target,
    *,
    elevation_m,
    land_fraction,
    latitude,
    longitude,
):
    """Create arrays for one NPZ without resampling unrelated source grids."""
    coarse = np.asarray(coarse, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if coarse.ndim != 3 or target.ndim != 3:
        raise ValueError("coarse e target devono avere forma [C,H,W]")
    static, cell_area = terrain_context(
        elevation_m, land_fraction, latitude, longitude
    )
    if target.shape[1:] != static.shape[1:]:
        raise ValueError("target e DEM non hanno la stessa griglia")
    scale_y = target.shape[1] / coarse.shape[1]
    scale_x = target.shape[2] / coarse.shape[2]
    if scale_y != scale_x or int(scale_y) != scale_y:
        raise ValueError("target non è un multiplo intero isotropo del coarse")
    if int(scale_y) < 2 or int(scale_y) & (int(scale_y) - 1):
        raise ValueError("fattore di scala non supportato: usare potenze di due")
    return {
        "coarse": coarse,
        "static": static,
        "target": target,
        "valid_mask": np.isfinite(target),
        "cell_area_m2": cell_area,
        "static_channels": STATIC_DOWNSCALING_FEATURES,
        "scale": int(scale_y),
    }
