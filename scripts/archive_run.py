#!/usr/bin/env python3
"""Create a checksum-verified portable archive bundle for one ICON run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meteo_analysis.verification.archive import sha256_file


VERIFICATION_PATHS = {
    "archive_manifest.json",
    "catalog.json",
    "observations.json",
    "front_qc.json",
    "hazard_qc.json",
    "ai_agent_status.json",
    "ai_expert_bulletin.json.gz",
    "verification/forecast_samples.json.gz",
}


def _write_json_atomic(path: Path, payload) -> None:
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


def build_bundle(data_dir: Path, output_dir: Path, *, full: bool = False):
    manifest_path = data_dir / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive_id = str(manifest["archiveId"])
    destination = output_dir / archive_id
    destination.mkdir(parents=True, exist_ok=True)
    known_assets = {
        item["path"]: item for item in manifest.get("publishedAssets") or []
    }
    selected = set(known_assets) if full else set(VERIFICATION_PATHS)
    selected.discard("archive_manifest.json")
    copied = []
    for relative in sorted(selected):
        if relative not in known_assets:
            continue
        source = data_dir / relative
        expected = known_assets[relative]["sha256"]
        actual = sha256_file(source)
        if actual != expected:
            raise RuntimeError(f"checksum non valido prima della copia: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256_file(target) != expected:
            raise RuntimeError(f"checksum non valido dopo la copia: {relative}")
        copied.append(relative)
    shutil.copy2(manifest_path, destination / "archive_manifest.json")
    bundle = {
        "schemaVersion": 1,
        "archiveId": archive_id,
        "mode": "full-published-products" if full else "verification",
        "assets": ["archive_manifest.json", *copied],
        "rawGribIncluded": False,
        "scientificPurpose": (
            "Forecast-observation verification and residual-correction training; "
            "not a replacement for the full raw-GRIB archive."
        ),
    }
    _write_json_atomic(destination / "bundle.json", bundle)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_weather")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--full", action="store_true")
    arguments = parser.parse_args()
    destination = build_bundle(
        Path(arguments.data_dir), Path(arguments.output_dir), full=arguments.full
    )
    print(destination)


if __name__ == "__main__":
    main()
