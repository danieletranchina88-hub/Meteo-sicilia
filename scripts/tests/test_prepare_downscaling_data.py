#!/usr/bin/env python3
"""End-to-end test for the aligned observation dataset packager."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from meteo_analysis.deep_learning.data import (
    TensorManifestDataset,
    load_manifest,
)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    latitude = np.linspace(37.0, 37.7, 8)
    longitude = np.linspace(12.0, 13.1, 12)
    elevation = np.linspace(0.0, 1200.0, 96).reshape(8, 12)
    land = np.zeros((8, 12), dtype=np.float32)
    land[:, 3:] = 1.0
    np.savez(
        root / "static.npz",
        elevation_m=elevation,
        land_fraction=land,
        latitude=latitude,
        longitude=longitude,
    )
    samples = []
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for index in range(30):
        name = f"aligned_{index:03d}.npz"
        coarse = np.zeros((5, 2, 3), dtype=np.float32)
        coarse[0] = 280.0 + index * 0.05
        target = np.zeros((4, 8, 12), dtype=np.float32)
        target[0] = 280.0 + index * 0.05 + elevation * 1.0e-4
        target[1] = index % 4
        target[2] = 3.0
        target[3] = -1.0
        np.savez(root / name, coarse=coarse, target=target)
        samples.append({
            "path": name,
            "validTime": (start + timedelta(days=index)).isoformat(),
            "runTime": (start + timedelta(days=index)).isoformat(),
            "forecastHour": 0,
        })
    source = root / "source.json"
    source.write_text(json.dumps({
        "schemaVersion": 1,
        "staticPath": "static.npz",
        "coarseChannels": ["t2m", "rain", "u10", "v10", "pmsl"],
        "outputChannels": [
            "temperature_2m_k", "precipitation_rate_mm_h",
            "wind_u10_m_s", "wind_v10_m_s",
        ],
        "targetAuthority": {
            "temperature": "synthetic-test-stations",
            "precipitation": "synthetic-test-radar",
            "wind": "synthetic-test-stations",
        },
        "grid": {"crs": "EPSG:4326", "scale": 4},
        "samples": samples,
    }), encoding="utf-8")
    output = root / "dataset"
    subprocess.run([
        sys.executable,
        str(ROOT / "scripts/prepare_downscaling_data.py"),
        str(source),
        "--output", str(output),
        "--patch-size-coarse", "2",
        "--stride-coarse", "2",
    ], check=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    manifest = load_manifest(
        output / "manifest.json", expected_task="orographic-downscaling"
    )
    assert manifest["scale"] == 4
    assert len(manifest["samples"]) == 60
    assert len(manifest["sharedStatic"]["sha256"]) == 64
    assert all(len(item["sha256"]) == 64 for item in manifest["samples"])
    assert {item["split"] for item in manifest["samples"]} == {
        "train", "validation", "test",
    }
    first_path = output / manifest["samples"][0]["path"]
    with np.load(first_path, allow_pickle=False) as archive:
        assert "static" not in archive.files
        assert "cell_area_m2" not in archive.files
    for split in ("train", "validation", "test"):
        dataset = TensorManifestDataset(
            output / "manifest.json",
            task="orographic-downscaling",
            split=split,
        )
        item = dataset[0]
        assert item["coarse"].shape == (5, 2, 2)
        assert item["target"].shape == (4, 8, 8)
        assert item["static"].shape == (5, 8, 8)

print("Downscaling dataset packager test passed")
