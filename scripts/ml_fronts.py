#!/usr/bin/env python3
"""Operational ICON-2I supervised front probabilities and vector contours."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meteo_analysis.ml.features import (
    ERA5_TRANSFER_FEATURE_COLUMNS,
    ERA5_TRANSFER_NO_COORDINATES_FEATURE_COLUMNS,
    feature_frame_from_store,
    theta_gradient_50_from_store,
)
from meteo_analysis.ml.fusion import MLFrontGuidance
from meteo_analysis.ml.icon2i import Icon2IStore
from meteo_analysis.ml.model import FrontModel


def _temporal_confirmation(probabilities, threshold):
    """Require support in 2 of 3 adjacent hours for stand-alone ML contours."""
    hours = sorted(probabilities)
    confirmed = {}
    for index, hour in enumerate(hours):
        neighbours = [
            probabilities[other] >= threshold
            for other in hours[max(0, index - 1):index + 2]
        ]
        count = np.sum(neighbours, axis=0)
        needed = 2 if len(neighbours) >= 2 else 1
        confirmed[hour] = np.where(count >= needed, probabilities[hour], 0.0)
    return confirmed


def _contours(probability, latitude, longitude, threshold):
    from contourpy import contour_generator

    generator = contour_generator(
        x=longitude, y=latitude, z=probability, line_type="Separate"
    )
    features = []
    for coordinates in generator.lines(float(threshold)):
        coordinates = np.asarray(coordinates, float)
        if len(coordinates) < 3:
            continue
        mean_lat = np.deg2rad(np.mean(coordinates[:, 1]))
        length = np.sum(np.hypot(
            np.diff(coordinates[:, 0]) * 111.32 * np.cos(mean_lat),
            np.diff(coordinates[:, 1]) * 110.57,
        ))
        if length < 100.0:
            continue
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": np.round(coordinates, 3).tolist(),
            },
            "properties": {
                "probabilityThreshold": round(float(threshold), 4),
                "lengthKm": round(float(length), 1),
                "diagnosticOnly": True,
            },
        })
    return features


def predict_store(store, model, *, hours=None, output_dir=None):
    common = set.intersection(*[
        store.available_hours(name) for name in (
            "t850", "q850", "u850", "v850", "t700", "q700",
            "u500", "v500", "fi500", "u10", "v10", "pmsl",
        )
    ])
    selected = sorted(common if hours is None else common & set(map(int, hours)))
    if not selected:
        raise ValueError("nessuna ora ICON comune alle feature ML")
    profile = str(model.metadata.get("featureProfile", "era5-transfer"))
    expected_features = {
        "era5-transfer": ERA5_TRANSFER_FEATURE_COLUMNS,
        "era5-transfer-no-coordinates":
            ERA5_TRANSFER_NO_COORDINATES_FEATURE_COLUMNS,
    }.get(profile)
    if expected_features is None or model.features != expected_features:
        raise ValueError("profilo/ordine feature del modello non operativo")
    probabilities = {}
    gradient_cache = {}
    for hour in selected:
        history = []
        for previous in (hour - 2, hour - 1):
            if previous in common:
                if previous not in gradient_cache:
                    gradient_cache[previous] = theta_gradient_50_from_store(
                        store, previous
                    )
                history.append(gradient_cache[previous])
        frame = feature_frame_from_store(
            store, hour, gradient_history=history, feature_profile=profile
        )
        values = model.predict(frame).reshape(
            len(store.target_latitudes), len(store.target_longitudes)
        )
        probabilities[hour] = values.astype(np.float32)
        print(
            f"   ML +{hour:02d}h: max={np.nanmax(values):.3f}, "
            f"area>soglia={100*np.mean(values >= model.threshold):.2f}%",
            flush=True,
        )
    confirmed = _temporal_confirmation(probabilities, model.threshold)
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for hour in selected:
            payload = {
                "type": "FeatureCollection",
                "features": _contours(
                    confirmed[hour], store.target_latitudes,
                    store.target_longitudes, model.threshold,
                ),
                "properties": {
                    "forecastHour": hour,
                    "method": "xgboost-dwd-era5-icon2i-v1",
                    "threshold": model.threshold,
                    "temporalConfirmation": "2-of-3 adjacent hours",
                    "diagnosticOnly": True,
                },
            }
            (output / f"fronts_ml_{hour:03d}.geojson").write_text(
                json.dumps(payload, separators=(",", ":"), allow_nan=False),
                encoding="utf-8",
            )
    # Fusion samples raw calibrated probabilities. Physical tracking supplies
    # its own stricter 3 h persistence; zeroing here would duplicate that gate.
    return MLFrontGuidance(
        store.target_latitudes, store.target_longitudes,
        probabilities, model.threshold,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="YYYYMMDDHH")
    parser.add_argument("--model", default="models/front_model.json")
    parser.add_argument("--paths-json", required=True)
    parser.add_argument("--output-dir", default="outputs/fronts_ml")
    parser.add_argument("--step", type=int, default=3)
    args = parser.parse_args()
    paths = json.loads(Path(args.paths_json).read_text(encoding="utf-8"))
    model = FrontModel.load(args.model)
    store = Icon2IStore(paths, args.run)
    try:
        common = set.intersection(*[
            store.available_hours(name) for name in paths
        ])
        predict_store(
            store, model, hours=range(min(common), max(common) + 1, args.step),
            output_dir=args.output_dir,
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
