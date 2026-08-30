#!/usr/bin/env python3
"""Scientific and semantic regression tests for the expert bulletin."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from meteo_analysis.products.synoptic_engine import (  # noqa: E402
    ENGINE_METHOD,
    build_synoptic_frame,
    generate_run_bulletin,
)


LAT = np.linspace(33.7, 48.9, 31)
LON = np.linspace(3.0, 22.0, 39)
LON2D, LAT2D = np.meshgrid(LON, LAT)
SHAPE = LAT2D.shape


def constant(value):
    return np.full(SHAPE, float(value), dtype=float)


def fields(*, strong_cin=False, missing_upper=False, missing_rain=False):
    pressure = 1018.0 - 12.0 * np.exp(
        -((LAT2D - 43.0) ** 2 + (LON2D - 9.0) ** 2) / 8.0
    )
    theta_e = 315.0 + 12.0 * np.tanh((LON2D - 12.0) / 1.2)
    result = {
        "pressureMsl": pressure,
        "temperature2m": 18.0 + 0.3 * (LON2D - 12.0),
        "relativeHumidity2m": constant(72),
        "rainStep": (
            np.full(SHAPE, np.nan) if missing_rain
            else np.maximum(0.0, 2.0 - np.hypot(LAT2D - 41.0, LON2D - 13.0))
        ),
        "cloudCover": constant(68),
        "u10": constant(10),
        "v10": constant(0),
        "gust10": constant(55),
        "convergence10": constant(9.0e-5),
        "frontDistanceKm": constant(35),
        "temperature925": constant(288),
        "thetaE925": constant(318),
        "relativeHumidity925": constant(75),
        "u925": constant(11),
        "v925": constant(2),
        "temperature850": constant(282),
        "thetaE850": theta_e,
        "thetaW850": theta_e - 12,
        "relativeHumidity850": constant(70),
        "u850": constant(14),
        "v850": constant(4),
        "temperature700": constant(270),
        "relativeHumidity700": constant(68),
        "omega700": constant(-0.22),
        "u700": constant(17),
        "v700": constant(6),
        "height500": 5700.0 + 15.0 * (LAT2D - 40.0),
        "temperature500": constant(248),
        "u500": constant(24),
        "v500": constant(8),
        "omega500": constant(-0.16),
        "u300": constant(38),
        "v300": 4.0 * np.sin((LAT2D - 40.0) / 2.0),
        "capeMl": constant(1300),
        "capeMu": constant(1700),
        "cinMl": constant(-180 if strong_cin else -35),
        "shear06": constant(19),
        "updraftHelicity": constant(55),
        "lpi": constant(2.5),
        "precipitableWater": constant(32),
        "stormScore": constant(68),
        "stormCoherence": constant(82),
        "stormContradiction": constant(12),
        "hailIndex": constant(35),
        "downburstIndex": constant(42),
        "freezingLevel": constant(2700),
        "freezingRain": constant(0),
        "fogIndex": constant(18),
        "visibility": constant(14000),
        "foehnIndex": constant(0),
        "triggerIndex": constant(45),
        "bowenRatio": constant(0.8),
        "upslopeFlow": constant(0.35),
        "seaBreezeConvergence": constant(1.5),
    }
    if missing_upper:
        for name in ("temperature500", "u500", "v500", "omega500", "u300", "v300"):
            result[name] = None
    return result


def fronts():
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "frontType": "cold",
                "existenceConfidence": 0.82,
                "uncertaintyIndex": 0.17,
                "diagnostics": {
                    "frontogenesis": 1.8,
                    "vorticity1e5": 4.2,
                    "convergence1e5": 6.4,
                    "deltaThetaW": 5.1,
                },
            },
            "geometry": {"type": "LineString", "coordinates": [[8, 44], [14, 39]]},
        }],
    }


def make_frame(hour, **field_options):
    return build_synoptic_frame(
        lead_hours=hour,
        valid_time=f"2026-08-30T{hour:02d}:00:00Z",
        latitudes=LAT,
        longitudes=LON,
        fronts=fronts(),
        fields=fields(**field_options),
        rain_stride=3,
    )


def test_complete_run():
    frame = make_frame(0)
    domain = frame["regions"]["intero dominio"]
    assert abs(domain["wind10"]["p95"] - 36.0) < 0.01, domain["wind10"]
    assert abs(domain["convergence10"]["p95"] - 9.0) < 0.01
    assert frame["availability"]["divergence300"] is True
    assert frame["fronts"]["count"] == 1

    run = generate_run_bulletin(
        [make_frame(hour) for hour in range(7)],
        run_time="2026-08-30T00:00:00Z",
    )
    assert run["method"] == ENGINE_METHOD
    assert len(run["analyses"]) == 7
    section_titles = {item["title"] for item in run["analyses"][0]["sections"]}
    assert {
        "Analisi sinottica", "Dinamica in quota", "Bassa troposfera",
        "Stabilità e convezione", "Precipitazioni", "Evoluzione", "Incertezze",
    } <= section_titles
    assert run["semantics"]["stormScore"] == "diagnostic-not-calibrated-probability"
    json.dumps(run, allow_nan=False)


def test_cape_alone_never_becomes_robust():
    frame = make_frame(0, strong_cin=True)
    # Remove the independent lift evidence while retaining high CAPE and score.
    frame["regions"]["intero dominio"]["convergence10"]["p95"] = 0.0
    for region in frame["regions"]:
        frame["regions"][region]["convergence10"]["p95"] = 0.0
        frame["regions"][region]["omega700"]["p10"] = 0.1
        frame["regions"][region]["frontDistanceKm"]["p10"] = 300.0
        overlap = frame["regions"][region]["convectiveOverlap"]
        overlap["robustAreaPct"] = 0.0
        overlap["conditionalAreaPct"] = 0.0
        overlap["scoreP95OnRobustCells"] = None
    frame["fronts"]["count"] = 0
    frame["fronts"]["types"] = {}
    run = generate_run_bulletin([frame], run_time="2026-08-30T00:00:00Z")
    convection = next(
        item for item in run["analyses"][0]["sections"]
        if item["id"] == "convection"
    )
    text = " ".join(convection["paragraphs"]).lower()
    assert "segnale debole" in text
    assert "inibizione convettiva significativa" in text
    assert "segnale abbastanza robusto" not in text


def test_missing_diagnostics_are_declared_and_nan_rain_is_not_zero():
    frames = [
        make_frame(hour, missing_upper=True, missing_rain=True)
        for hour in range(2)
    ]
    run = generate_run_bulletin(frames, run_time="2026-08-30T00:00:00Z")
    analysis = run["analyses"][0]
    unavailable = " ".join(analysis["unavailableInputs"])
    assert "vento 300 hPa" in unavailable
    assert "Potential vorticity" in unavailable
    rain3 = analysis["rainAccumulations"]["3"]["regions"]["intero dominio"]
    assert rain3["maximum"] is None
    assert rain3["validPct"] == 0.0


if __name__ == "__main__":
    test_complete_run()
    test_cape_alone_never_becomes_robust()
    test_missing_diagnostics_are_declared_and_nan_rain_is_not_zero()
    print("Synoptic expert bulletin tests passed")
