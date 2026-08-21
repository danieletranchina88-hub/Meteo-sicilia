#!/usr/bin/env python3
"""Offline front/precipitation association, never an existence detector.

Following the impact-oriented use in Dagon et al. (2022), this module checks
whether independently detected fronts are associated with precipitation. It
is deliberately disconnected from the operational publication gates: dry
fronts remain valid and convective rain cannot manufacture a front.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.spatial import cKDTree

import front_benchmark as fb

EARTH_RADIUS_KM = 6371.0088


def _unit_sphere(latitude, longitude) -> np.ndarray:
    latitude = np.deg2rad(np.asarray(latitude, float))
    longitude = np.deg2rad(np.asarray(longitude, float))
    return np.column_stack((
        np.cos(latitude) * np.cos(longitude),
        np.cos(latitude) * np.sin(longitude),
        np.sin(latitude),
    ))


def _densify(coordinates: np.ndarray, spacing_km: float = 10.0) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    if len(coordinates) < 2:
        return coordinates
    output = [coordinates[0]]
    for first, second in zip(coordinates[:-1], coordinates[1:]):
        mean_lat = np.deg2rad(0.5 * (first[1] + second[1]))
        dx = (second[0] - first[0]) * 111.32 * np.cos(mean_lat)
        dy = (second[1] - first[1]) * 111.32
        count = max(1, int(np.ceil(np.hypot(dx, dy) / spacing_km)))
        output.extend(
            first + fraction * (second - first)
            for fraction in np.linspace(1.0 / count, 1.0, count)
        )
    return np.asarray(output)


def distance_to_front_grid(
    fronts: list[dict], longitudes: np.ndarray, latitudes: np.ndarray
) -> np.ndarray:
    """Approximate geodesic distance (km) from each grid cell to a front."""
    dense = [
        _densify(np.asarray(front["coordinates"], dtype=float))
        for front in fronts
        if np.asarray(front.get("coordinates", [])).ndim == 2
        and len(front.get("coordinates", [])) >= 2
    ]
    if not dense:
        return np.full((len(latitudes), len(longitudes)), np.inf)
    coordinates = np.vstack(dense)
    tree = cKDTree(_unit_sphere(coordinates[:, 1], coordinates[:, 0]))
    lon2d, lat2d = np.meshgrid(longitudes, latitudes)
    chord, _ = tree.query(_unit_sphere(lat2d.ravel(), lon2d.ravel()), k=1)
    angle = 2.0 * np.arcsin(np.clip(chord, 0.0, 2.0) / 2.0)
    return (EARTH_RADIUS_KM * angle).reshape(lat2d.shape)


def impact_association(
    fronts: list[dict],
    precipitation: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    *,
    precipitation_threshold: float = 5.0,
    near_radius_km: float = 100.0,
    far_radius_km: float = 250.0,
) -> dict:
    """Compare precipitation occurrence near fronts with the far background."""
    precipitation = np.asarray(precipitation, dtype=float)
    expected = (len(latitudes), len(longitudes))
    if precipitation.shape != expected:
        raise ValueError(
            f"griglia precipitazione {precipitation.shape}, attesa {expected}"
        )
    distance = distance_to_front_grid(fronts, longitudes, latitudes)
    valid = np.isfinite(precipitation)
    near = valid & (distance <= near_radius_km)
    far = valid & (distance >= far_radius_km)

    def region(mask: np.ndarray) -> dict:
        if not np.any(mask):
            return {"cells": 0, "eventRate": None, "medianPrecipitation": None}
        values = precipitation[mask]
        return {
            "cells": int(values.size),
            "eventRate": float(np.mean(values >= precipitation_threshold)),
            "medianPrecipitation": float(np.median(values)),
        }

    near_stats, far_stats = region(near), region(far)
    lift = None
    if (
        near_stats["eventRate"] is not None
        and far_stats["eventRate"] is not None
        and far_stats["eventRate"] > 0.0
    ):
        lift = near_stats["eventRate"] / far_stats["eventRate"]
    return {
        "role": "offline-impact-validation-only",
        "precipitationThreshold": float(precipitation_threshold),
        "nearRadiusKm": float(near_radius_km),
        "farRadiusKm": float(far_radius_km),
        "near": near_stats,
        "far": far_stats,
        "eventRateLift": float(lift) if lift is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fronts", help="GeoJSON frontale già prodotto")
    parser.add_argument(
        "precipitation_npz",
        help="NPZ con precipitation, longitudes e latitudes",
    )
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--near-km", type=float, default=100.0)
    parser.add_argument("--far-km", type=float, default=250.0)
    args = parser.parse_args()
    data = np.load(args.precipitation_npz)
    report = impact_association(
        fb.load_fronts(args.fronts), data["precipitation"],
        data["longitudes"], data["latitudes"],
        precipitation_threshold=args.threshold,
        near_radius_km=args.near_km,
        far_radius_km=args.far_km,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
