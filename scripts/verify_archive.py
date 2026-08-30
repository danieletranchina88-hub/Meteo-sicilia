#!/usr/bin/env python3
"""Score an archived ICON station forecast against METAR snapshots."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meteo_analysis.verification.metrics import verify_station_forecasts


def load_forecast(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_observations(directory: Path):
    snapshots = []
    for path in sorted(directory.rglob("metar_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("stations"):
            snapshots.append(payload)
    return snapshots


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        partial.write_text(
            json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("forecast", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-minutes", type=int, default=45)
    arguments = parser.parse_args()
    forecast = load_forecast(arguments.forecast)
    observations = load_observations(arguments.observations)
    if not observations:
        raise RuntimeError("nessuno snapshot METAR valido trovato")
    report = verify_station_forecasts(
        forecast,
        observations,
        tolerance_minutes=arguments.tolerance_minutes,
    )
    write_json_atomic(arguments.output, report)
    print(
        f"{arguments.output}: {report['matchedForecastStationTimes']} "
        "coppie stazione/scadenza"
    )


if __name__ == "__main__":
    main()
