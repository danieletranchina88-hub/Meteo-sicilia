#!/usr/bin/env python3
"""Deterministic checks for the convective-initiation diagnostic."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from meteo_analysis.hazards.convection import (  # noqa: E402
    calculate_convection_probability,
    front_distance_km,
    horizontal_convergence,
    normalize_cin,
)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


lat = np.linspace(46.0, 36.0, 101)
lon = np.linspace(7.0, 18.0, 111)
grid_lon, grid_lat = np.meshgrid(lon, lat)

# A linear convergent flow has an analytical convergence of about 2e-4 s-1.
x_m = (grid_lon - np.mean(lon)) * 111_195.0 * np.cos(np.deg2rad(grid_lat))
y_m = (grid_lat - np.mean(lat)) * 111_195.0
u = -1.0e-4 * x_m
v = -1.0e-4 * y_m
convergence = horizontal_convergence(u, v, lat, lon, smoothing_km=10.0)
center = convergence[15:-15, 15:-15]
check(
    1.7e-4 < float(np.nanmedian(center)) < 2.3e-4,
    "convergenza metrica errata o dipendente dall'ordine della latitudine",
)

fronts = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[12.0, 36.0], [12.0, 46.0]],
            },
            "properties": {"frontType": "cold"},
        }
    ],
}
distance = front_distance_km(lat, lon, fronts)
near_column = int(np.argmin(np.abs(lon - 12.0)))
check(float(np.nanmax(distance[:, near_column])) < 6.0, "distanza dal fronte errata")
check(float(np.nanmin(distance[:, 0])) > 300.0, "fronte esteso artificialmente")

shape = distance.shape
cape = np.full(shape, 1_200.0)
cin_positive_magnitude = np.full(shape, 35.0)
conv = np.full(shape, 1.3e-4)
probability = np.asarray(
    calculate_convection_probability(
        cape, cin_positive_magnitude, conv, distance, surface_rh=np.full(shape, 75.0)
    )
) * 100.0
check(np.nanmax(probability[:, near_column]) >= 70.0, "regola esperta alta non applicata")
check(np.nanmax(probability[:, 0]) < 70.0, "probabilità alta lontano dai fronti")
check(np.all(normalize_cin(cin_positive_magnitude) == -35.0), "segno CIN non normalizzato")

stable = np.asarray(
    calculate_convection_probability(
        np.zeros(shape), np.full(shape, -300.0), np.zeros(shape), distance
    )
) * 100.0
check(float(np.nanmax(stable)) <= 8.0, "ambiente stabile sovrastimato")

# Regression for the former nationwide 80% field: even with widespread CAPE
# and weak CIN, only a narrow convergence/front intersection can exceed 70%.
stripe_convergence = np.zeros(shape)
stripe_convergence[:, near_column - 1 : near_column + 2] = 1.4e-4
anti_saturation = np.asarray(
    calculate_convection_probability(
        np.full(shape, 1_500.0),
        np.full(shape, -30.0),
        stripe_convergence,
        distance,
    )
) * 100.0
high_fraction = float(np.mean(anti_saturation >= 70.0))
check(high_fraction < 0.05, f"campo alto troppo esteso: {high_fraction:.3f}")
check(not np.allclose(anti_saturation, 80.0), "ricomparso il valore fisso 80%")

print("Convection tests passed")
