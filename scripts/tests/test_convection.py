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
    relative_humidity_from_specific_humidity,
    summarize_convection,
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
sentinel_cin = normalize_cin(np.array([-999.9, 40.0, -25.0, np.nan]))
check(np.isnan(sentinel_cin[0]), "sentinel CIN MeteoHub non escluso")
check(sentinel_cin[1] == -40.0, "CIN positivo non convertito in magnitudine firmata")
check(sentinel_cin[2] == -25.0, "CIN già firmato alterato")
check(np.isnan(sentinel_cin[3]), "NaN CIN trasformato in valore valido")
sentinel_probability = np.asarray(
    calculate_convection_probability(
        np.array([1_500.0]),
        np.array([-999.9]),
        np.array([1.5e-4]),
        np.array([10.0]),
    )
)
check(
    np.isnan(sentinel_probability[0]),
    "cella con sentinel CIN pubblicata come probabilità valida",
)

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
summary = summarize_convection(anti_saturation)
check(summary["areaAbove70Pct"] < 5.0, "QC area alta non coerente")
check(summary["areaExactly80Pct"] < 5.0, "QC valore fisso 80% non coerente")

# --- New ingredients: deep-layer shear and mid-tropospheric moisture ---

# A marginal-but-real setup (medium tier): CAPE + weak lid + some convergence.
marg_cape = np.full(shape, 900.0)
marg_cin = np.full(shape, 60.0)  # magnitude -> -60 J/kg, weak lid
marg_conv = np.full(shape, 0.8e-4)
base = np.asarray(
    calculate_convection_probability(marg_cape, marg_cin, marg_conv, distance)
) * 100.0
sheared = np.asarray(
    calculate_convection_probability(
        marg_cape, marg_cin, marg_conv, distance,
        deep_layer_shear=np.full(shape, 25.0),
    )
) * 100.0
check(
    float(np.nanmax(sheared)) > float(np.nanmax(base)) + 1.0,
    "lo shear di strato profondo non rafforza un ambiente instabile",
)

# Shear without instability must not create a thunderstorm.
no_cape_sheared = np.asarray(
    calculate_convection_probability(
        np.zeros(shape), np.full(shape, -300.0), np.zeros(shape), distance,
        deep_layer_shear=np.full(shape, 30.0),
    )
) * 100.0
check(
    float(np.nanmax(no_cape_sheared)) <= 8.0,
    "lo shear ha inventato convezione senza CAPE",
)

# Dry mid-troposphere suppresses; a moist one does not.  Tested below the
# medium-tier convergence gate so the tier floor does not mask the moisture
# modulation (the floor deliberately overrides in the medium/high tiers).
weak_conv = np.full(shape, 0.3e-4)
dry_aloft = np.asarray(
    calculate_convection_probability(
        marg_cape, marg_cin, weak_conv, distance,
        mid_level_rh=np.full(shape, 12.0),
    )
) * 100.0
moist_aloft = np.asarray(
    calculate_convection_probability(
        marg_cape, marg_cin, weak_conv, distance,
        mid_level_rh=np.full(shape, 85.0),
    )
) * 100.0
check(
    float(np.nanmax(dry_aloft)) < float(np.nanmax(moist_aloft)),
    "l'aria secca in quota non riduce la probabilità",
)

# The expert >=70% rule still requires the four core ingredients even with
# generous shear and mid-level moisture supplied.
gated = np.asarray(
    calculate_convection_probability(
        np.full(shape, 1_500.0), np.full(shape, -30.0),
        np.zeros(shape),  # no convergence -> cannot be "high"
        distance,
        deep_layer_shear=np.full(shape, 30.0),
        mid_level_rh=np.full(shape, 90.0),
    )
) * 100.0
check(
    float(np.nanmax(gated)) < 70.0,
    "shear/umidità hanno superato la regola dei quattro ingredienti",
)

# Backward compatibility: omitting the new fields reproduces the old field.
legacy = np.asarray(
    calculate_convection_probability(
        cape, cin_positive_magnitude, conv, distance,
        surface_rh=np.full(shape, 75.0),
    )
)
legacy_explicit_none = np.asarray(
    calculate_convection_probability(
        cape, cin_positive_magnitude, conv, distance,
        surface_rh=np.full(shape, 75.0),
        deep_layer_shear=None, mid_level_rh=None,
    )
)
check(
    np.allclose(legacy, legacy_explicit_none, equal_nan=True),
    "i nuovi parametri opzionali alterano il comportamento predefinito",
)

# --- RH from specific humidity (mid-level moisture ingredient source) ---
rh_moist = relative_humidity_from_specific_humidity(275.15, 0.0045, 700.0)
rh_dry = relative_humidity_from_specific_humidity(275.15, 0.0004, 700.0)
check(float(rh_moist) > 65.0, f"RH umida a 700 hPa troppo bassa: {rh_moist}")
check(float(rh_dry) < 20.0, f"RH secca a 700 hPa troppo alta: {rh_dry}")
# Celsius input and g/kg input are auto-detected and give the same answer.
rh_celsius_gkg = relative_humidity_from_specific_humidity(2.0, 4.5, 700.0)
check(
    abs(float(rh_celsius_gkg) - float(rh_moist)) < 1.0,
    "auto-rilevamento unità (°C / g kg-1) incoerente",
)
check(
    np.isnan(relative_humidity_from_specific_humidity(np.nan, 0.004, 700.0)),
    "input non finito non propagato a NaN",
)

print("Convection tests passed")
