#!/usr/bin/env python3
"""Checks that NLG claims remain tied to actual model evidence."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from meteo_analysis.products.nlg import (  # noqa: E402
    build_bulletin_inputs,
    generate_bulletin_details,
)


def make_inputs(fronts, convection):
    shape = (12, 14)
    return build_bulletin_inputs(
        valid_time="2026-07-26T12:00:00Z",
        fronts=fronts,
        convection_probability=convection,
        temperature=np.full(shape, 27.0),
        precipitation=np.zeros(shape),
        cloud=np.full(shape, 30.0),
        pressure=np.full(shape, 1013.0),
        u_wind=np.full(shape, 3.0),
        v_wind=np.zeros(shape),
        mask=np.ones(shape, dtype=bool),
        area="Italia",
    )


empty_fronts = {"type": "FeatureCollection", "features": []}
low = make_inputs(empty_fronts, np.full((12, 14), 12.0))
low_details = generate_bulletin_details(low)
assert "fronte caldo" not in low_details["text"].lower()
assert "fronte freddo" not in low_details["text"].lower()
assert "generalmente basso" in low_details["text"].lower()

cold_front = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[8, 40], [12, 43]]},
            "properties": {"frontType": "cold"},
        }
    ],
}
high_field = np.full((12, 14), 45.0)
high_field[:, :2] = 78.0
high = make_inputs(cold_front, high_field)
high_details = generate_bulletin_details(high)
assert "fronte freddo" in high_details["text"].lower()
assert "su Italia" in high_details["text"]
assert "picchi del 78%" in high_details["text"]
assert high_details["method"].endswith("-v2")
assert len(high_details["paragraphs"]) >= 4

unavailable = make_inputs(empty_fronts, np.full((12, 14), np.nan))
unavailable_details = generate_bulletin_details(unavailable)
assert "non sostituisce i campi mancanti con valori simulati" in unavailable_details["text"]
assert "probabilità temporali" in unavailable_details["unavailableInputs"]

print("NLG tests passed")
