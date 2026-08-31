#!/usr/bin/env python3
"""Collect one immutable, time-stamped METAR verification snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meteo_analysis.observations.pipeline import collect_national_observations


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
    parser.add_argument("--output-dir", default="observation_archive")
    arguments = parser.parse_args()
    payload = collect_national_observations()
    total_stations = payload.get("registry", {}).get("totalStations", 0)
    if not total_stations:
        raise RuntimeError("snapshot vuoto: non archivio un falso successo")
    stamp = payload["capturedAt"].replace("-", "").replace(":", "")
    stamp = stamp.replace("T", "_").replace("Z", "Z")
    output = Path(arguments.output_dir) / f"observations_{stamp}.json"
    write_json_atomic(output, payload)
    print(f"{output}: {total_stations} stazioni totali (provider: "
          f"{', '.join(sorted(payload.get('providerStatus', {}).keys()))})")


if __name__ == "__main__":
    main()
