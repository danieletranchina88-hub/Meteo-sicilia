"""Synthetic tests for independent, track-level ECMWF confirmation."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import front_tracking as ft


def line(latitude, offset=0.0):
    longitude = np.linspace(5.0, 19.0, 80)
    return np.column_stack((longitude, np.full_like(longitude, latitude + offset)))


def feature(latitude, front_type="cold"):
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": line(latitude).tolist()},
        "properties": {"frontType": front_type},
    }


hours = list(range(0, 13))
track = {
    "hours": hours,
    "lines": {hour: line(42.0 - hour * 0.05) for hour in hours},
    "frontType": "cold",
}
close_reference = {
    hour: {"type": "FeatureCollection", "features": [
        feature(42.0 - hour * 0.05 + 0.25)
    ]}
    for hour in hours
}
distant_reference = {
    hour: {"type": "FeatureCollection", "features": [feature(47.0)]}
    for hour in hours
}
opposite_reference = {
    hour: {"type": "FeatureCollection", "features": [
        feature(42.0 - hour * 0.05 + 0.25, "warm")
    ]}
    for hour in hours
}

radius = lambda hour: 110.0
checks = [
    ("linee vicine", close_reference, True),
    ("linee lontane", distant_reference, False),
    ("guida vuota", {}, False),
    ("tipo opposto", opposite_reference, False),
]

ok = True
for label, reference, expected in checks:
    result = ft.cross_model_diagnostics(track, reference, radius)
    print(label, result)
    if result["confirmed"] is not expected:
        ok = False

print("ESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
