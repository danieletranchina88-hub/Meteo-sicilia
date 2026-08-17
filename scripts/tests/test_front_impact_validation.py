"""Precipitation validates impacts offline and never detects a front."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import front_impact_validation as fi

lon = np.linspace(3.0, 22.0, 150)
lat = np.linspace(34.0, 49.0, 120)
lon2d, lat2d = np.meshgrid(lon, lat)
fronts = [{
    "coordinates": np.column_stack((
        np.linspace(5.0, 20.0, 50), np.full(50, 42.0)
    ))
}]
distance = fi.distance_to_front_grid(fronts, lon, lat)
rng = np.random.default_rng(2026)
precipitation = rng.uniform(0.0, 1.0, distance.shape)
precipitation[distance <= 80.0] += 8.0
# Give the far field a small but non-zero event rate so lift is measurable.
precipitation[(distance >= 250.0) & (lon2d < 4.0)] += 6.0
report = fi.impact_association(
    fronts, precipitation, lon, lat,
    precipitation_threshold=5.0,
)
print(report)
if report["role"] != "offline-impact-validation-only":
    raise SystemExit("FAIL: ruolo operativo ambiguo")
if not (
    report["near"]["eventRate"] > report["far"]["eventRate"]
    and report["eventRateLift"] > 2.0
):
    raise SystemExit("FAIL: associazione di impatto non riconosciuta")
if not np.all(np.isinf(fi.distance_to_front_grid([], lon, lat))):
    raise SystemExit("FAIL: precipitazione senza fronti crea una geometria")
print("ESITO: SUPERATO")
