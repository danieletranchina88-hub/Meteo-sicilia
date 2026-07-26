#!/usr/bin/env python3
"""Physical guardrail tests for secondary hazard modules."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from meteo_analysis.hazards.severe import calculate_ship, calculate_scp  # noqa: E402
from meteo_analysis.hazards.visibility import (  # noqa: E402
    calculate_fog_probability,
    classify_fog_type,
    estimate_visibility,
)
from meteo_analysis.hazards.winter import detect_freezing_rain  # noqa: E402
from meteo_analysis.orography.foehn import (  # noqa: E402
    cross_alpine_pressure_difference,
    detect_foehn,
)


shape = (3, 4)
ship_missing = calculate_ship(
    np.full(shape, 2_000.0),
    np.full(shape, 7.0),
    np.full(shape, 2_500.0),
    np.full(shape, 12.0),
)
assert np.isnan(ship_missing).all(), "SHIP non deve inventare T500"

scp_no_shear = calculate_scp(
    np.full(shape, 2_000.0), np.full(shape, 150.0), np.full(shape, 8.0)
)
assert np.nanmax(scp_no_shear) == 0.0

fog_type = classify_fog_type(
    np.full(shape, 99.0), np.full(shape, 1.5), np.full(shape, 20.0)
)
fog_probability = calculate_fog_probability(
    np.full(shape, 99.0),
    np.full(shape, 1.5),
    np.full(shape, 20.0),
    fog_type,
)
visibility = estimate_visibility(np.full(shape, 99.0), fog_type, fog_probability)
assert np.all(fog_type == 1.0)
assert np.nanmax(visibility) < 1_000.0

freezing = detect_freezing_rain(
    np.full(shape, 274.5),
    np.full(shape, 276.0),
    np.full(shape, 268.0),
    np.full(shape, 271.5),
    np.full(shape, 1.0),
)
assert np.all(freezing == 2.0)

foehn_without_pressure = detect_foehn(
    np.zeros(shape), np.full(shape, -10.0), np.full(shape, 30.0)
)
assert np.isnan(foehn_without_pressure).all(), "foehn senza gradiente barico"

latitudes = np.array([49.0, 48.0, 47.0, 46.0, 45.0, 44.0])
pressure_gradient = np.repeat(
    np.array([1018.0, 1017.0, 1015.0, 1011.0, 1009.0, 1008.0])[:, None],
    4,
    axis=1,
)
cross_barrier = cross_alpine_pressure_difference(
    pressure_gradient, latitudes, offset_degrees=1.0
)
assert np.nanmedian(cross_barrier[2:4]) > 3.0
north_foehn = detect_foehn(
    np.zeros((2, 4)),
    np.full((2, 4), -9.0),
    np.full((2, 4), 30.0),
    north_minus_south_pressure_hpa=np.full((2, 4), 4.0),
)
assert np.all(north_foehn == 1.0), "foehn settentrionale non riconosciuto"

print("Hazard tests passed")
