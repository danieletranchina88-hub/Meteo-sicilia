"""Immutable run-manifest primitives for the future AI training archive."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


def sha256_file(path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def source_asset_record(
    *,
    name: str,
    url: str,
    path,
    role: str,
    required: bool,
    retained: bool = False,
    archive_object: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the exact GRIB that was accepted by the processor."""

    path = Path(path)
    record = {
        "name": str(name),
        "role": str(role),
        "url": str(url),
        "required": bool(required),
        "sizeBytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "retainedInArchive": bool(retained),
    }
    if archive_object is not None:
        record["archiveObject"] = dict(archive_object)
    return record


def _output_role(relative_path: str) -> str:
    if relative_path == "ai_agent_status.json":
        return "quality-control"
    if relative_path.startswith("verification/"):
        return "verification-sample"
    if relative_path.startswith("meteograms/"):
        return "point-timeseries"
    if relative_path.startswith("step_"):
        return "native-grid-surface-product"
    if relative_path.startswith("upper_"):
        return "upper-air-product"
    if relative_path.startswith("storm_"):
        return "convective-diagnostic-product"
    if relative_path.endswith("_qc.json"):
        return "quality-control"
    if "bulletin" in relative_path:
        return "expert-analysis"
    if relative_path == "observations.json":
        return "observation-snapshot"
    return "published-product"


def _inventory(directory: Path) -> list[dict[str, Any]]:
    assets = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative == "archive_manifest.json" or relative.endswith(".part"):
            continue
        assets.append({
            "path": relative,
            "role": _output_role(relative),
            "sizeBytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    return assets


def build_run_manifest(
    data_dir,
    *,
    run_time: str,
    catalog: Iterable[dict[str, Any]],
    source_assets: Iterable[dict[str, Any]] = (),
    algorithms: dict[str, str | None] | None = None,
    domain: dict[str, float] | None = None,
    object_storage: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a checksum inventory without claiming unretained GRIBs exist."""

    directory = Path(data_dir)
    catalog = sorted(list(catalog), key=lambda item: int(item["leadHours"]))
    sources = sorted(
        [dict(item) for item in source_assets],
        key=lambda item: (item.get("role", ""), item.get("name", "")),
    )
    created = created_at or datetime.now(timezone.utc)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    hours = [int(item["leadHours"]) for item in catalog]
    expected = list(range(0, 73))
    retained_sources = sum(bool(item.get("retainedInArchive")) for item in sources)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "archiveId": f"icon2i-{run_time.replace('-', '').replace(':', '').replace('T', '').replace('Z', '')}",
        "createdAt": created.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "name": "ICON-2I",
            "runTime": run_time,
            "nominalHorizontalResolutionKm": 2.2,
            "temporalResolutionHours": 1,
            "deterministic": True,
            "dataset": "ICON_2I_SURFACE_PRESSURE_LEVELS",
            "provider": "Agenzia ItaliaMeteo MeteoHub",
        },
        "domain": domain or {},
        "forecastHours": hours,
        "completeness": {
            "expectedForecastHours": expected,
            "missingForecastHours": sorted(set(expected) - set(hours)),
            "complete0To72Hourly": hours == expected,
        },
        "algorithms": algorithms or {},
        "sourceAssets": sources,
        "publishedAssets": _inventory(directory),
        "objectStorage": object_storage or {
            "schemaVersion": 1,
            "mode": "off",
            "kind": None,
            "immutability": None,
            "retention": None,
            "runPrefix": None,
        },
        "archiveCapability": {
            "verificationSamplesIncluded": (
                directory / "verification" / "forecast_samples.json.gz"
            ).exists(),
            "observationSnapshotIncluded": (
                directory / "observations.json"
            ).exists(),
            "rawGribAssetsRetained": retained_sources,
            "fullFieldHistoricalTrainingReady": (
                bool(sources) and retained_sources == len(sources)
            ),
            "note": (
                "Checksums and exact source URLs provide provenance. Full-field "
                "historical ML training additionally requires object storage for "
                "the raw GRIB assets."
            ),
        },
    }


def write_run_manifest(path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        partial.write_text(
            json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(partial, target)
    finally:
        if partial.exists():
            partial.unlink()
