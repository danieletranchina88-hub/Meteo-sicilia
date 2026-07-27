#!/usr/bin/env python3
"""Checks for the compact static meteogram product."""

import gzip
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from meteo_analysis.products.meteograms import MeteogramArchive  # noqa: E402


latitudes = np.linspace(42.0, 36.0, 31)
longitudes = np.linspace(10.0, 16.0, 31)
yy, xx = np.meshgrid(latitudes, longitudes, indexing="ij")
archive = MeteogramArchive(
    latitudes,
    longitudes,
    run_time="2026-07-26T12:00:00Z",
    spacing_deg=0.4,
    tile_size_deg=2.0,
)
for hour in (0, 1, 2):
    archive.add(
        hour,
        f"2026-07-26T{12 + hour:02d}:00:00Z",
        {
            "temperature2m": 10.0 + hour + yy * 0.1 + xx * 0.01,
            "rainStep": np.full_like(yy, hour * 0.25),
            "windU10": np.full_like(yy, 3.0),
            "windV10": np.full_like(yy, 4.0),
            "convectionProbability": np.full_like(yy, 20.0 + hour),
        },
    )

with tempfile.TemporaryDirectory() as temporary:
    manifest = archive.write(temporary)
    assert manifest["hours"] == [0, 1, 2]
    assert len(manifest["tiles"]) > 1
    assert manifest["domain"]["west"] == 10.0
    tile_path = os.path.join(temporary, manifest["tiles"][0]["file"])
    with gzip.open(tile_path, "rt", encoding="utf-8") as source:
        tile = json.load(source)
    assert len(tile["times"]) == 3
    point_count = (
        len(tile["grid"]["latitudes"]) * len(tile["grid"]["longitudes"])
    )
    for field in tile["fields"].values():
        assert len(field["values"]) == 3
        assert all(len(values) == point_count for values in field["values"])
    assert tile["fields"]["rainStep"]["values"][2][0] == 0.5
    assert tile["fields"]["windU10"]["unit"] == "m/s"
    assert "temperature850" not in manifest["fields"]
    assert "temperature925" not in manifest["fields"]

print("Meteogram archive tests passed")
