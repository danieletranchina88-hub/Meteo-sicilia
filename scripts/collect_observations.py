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

from meteo_analysis.verification.observations import fetch_italy_metar_observations


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
    payload = fetch_italy_metar_observations()
    if not payload.get("stations"):
        raise RuntimeError("snapshot METAR vuoto: non archivio un falso successo")
    stamp = payload["capturedAt"].replace("-", "").replace(":", "")
    stamp = stamp.replace("T", "_").replace("Z", "Z")
    output = Path(arguments.output_dir) / f"metar_{stamp}.json"
    write_json_atomic(output, payload)
    print(f"{output}: {payload['count']} stazioni")


if __name__ == "__main__":
    main()
