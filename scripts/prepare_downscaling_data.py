#!/usr/bin/env python3
"""Patch pre-aligned ICON/observation arrays into a strict training dataset.

The script deliberately does not resample radar or station observations.  The
source manifest must point to arrays already aligned by an upstream,
source-specific and auditable ingestion step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meteo_analysis.deep_learning.preparation import terrain_context
from meteo_analysis.deep_learning.schemas import (
    DOWNSCALING_OUTPUTS,
    SCHEMA_VERSION,
    STATIC_DOWNSCALING_FEATURES,
    schema_hash,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("validTime sorgente privo di fuso UTC")
    return parsed


def _safe_path(root, value):
    path = (root / str(value)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("path sorgente esterno alla directory del manifest") from error
    return path


def _starts(size, patch, stride):
    if patch > size:
        raise ValueError("patch più grande del dominio coarse")
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _split_by_time(samples):
    times = sorted({_utc(item["validTime"]) for item in samples})
    if len(times) < 30:
        raise ValueError("servono almeno 30 valid time indipendenti")
    result = {}
    for index, valid_time in enumerate(times):
        fraction = index / len(times)
        result[valid_time] = (
            "train" if fraction < 0.70
            else "validation" if fraction < 0.85
            else "test"
        )
    return result


def _write_npz(path, **arrays):
    partial = path.with_suffix(path.suffix + ".part")
    try:
        with partial.open("wb") as output:
            np.savez_compressed(output, **arrays)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest")
    parser.add_argument("--output", default="training_data/downscaling")
    parser.add_argument("--patch-size-coarse", type=int, default=64)
    parser.add_argument("--stride-coarse", type=int, default=48)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.001)
    args = parser.parse_args()

    source_path = Path(args.source_manifest).resolve()
    source_root = source_path.parent
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("schemaVersion") != 1:
        raise ValueError("schema del manifest sorgente non supportato")
    coarse_channels = tuple(source.get("coarseChannels") or ())
    output_channels = tuple(source.get("outputChannels") or ())
    if not coarse_channels or output_channels != DOWNSCALING_OUTPUTS:
        raise ValueError("canali coarse/output sorgente non compatibili")
    target_authority = source.get("targetAuthority")
    if not isinstance(target_authority, dict):
        raise TypeError("targetAuthority sorgente mancante")
    samples = source.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest sorgente privo di campioni")
    split_by_time = _split_by_time(samples)

    if not source.get("staticPath"):
        raise ValueError("staticPath sorgente mancante")
    static_source = _safe_path(source_root, source["staticPath"])
    with np.load(static_source, allow_pickle=False) as archive:
        required = {"elevation_m", "land_fraction", "latitude", "longitude"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"statico sorgente privo di {sorted(missing)}")
        latitude = np.asarray(archive["latitude"], dtype=np.float64)
        longitude = np.asarray(archive["longitude"], dtype=np.float64)
        static, cell_area = terrain_context(
            archive["elevation_m"], archive["land_fraction"],
            latitude, longitude,
        )

    output = Path(args.output)
    shard_dir = output / "samples"
    shard_dir.mkdir(parents=True, exist_ok=True)
    static_output = output / "shared_static.npz"
    _write_npz(static_output, static=static, cell_area_m2=cell_area)
    records = []
    expected_scale = None
    patch = int(args.patch_size_coarse)
    stride = int(args.stride_coarse)
    if patch < 2 or stride < 1:
        raise ValueError("dimensione/stride della patch non validi")
    minimum_valid = float(args.minimum_valid_fraction)
    if not 0 <= minimum_valid <= 1:
        raise ValueError("minimum-valid-fraction fuori da 0..1")

    ordered = sorted(samples, key=lambda item: _utc(item["validTime"]))
    for sample_index, item in enumerate(ordered):
        sample_path = _safe_path(source_root, item.get("path"))
        with np.load(sample_path, allow_pickle=False) as archive:
            required = {"coarse", "target"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"{sample_path.name}: array mancanti {sorted(missing)}")
            coarse = np.asarray(archive["coarse"], dtype=np.float32)
            target = np.asarray(archive["target"], dtype=np.float32)
            valid = (
                np.asarray(archive["valid_mask"], dtype=bool)
                if "valid_mask" in archive else np.isfinite(target)
            )
        if coarse.ndim != 3 or target.ndim != 3 or valid.shape != target.shape:
            raise ValueError(f"{sample_path.name}: forme sorgente incoerenti")
        if not np.all(np.isfinite(coarse)):
            raise ValueError(f"{sample_path.name}: coarse contiene dati non finiti")
        if coarse.shape[0] != len(coarse_channels) or target.shape[0] != len(output_channels):
            raise ValueError(f"{sample_path.name}: numero canali incoerente")
        if target.shape[1:] != static.shape[1:]:
            raise ValueError(f"{sample_path.name}: target non allineato allo statico")
        scale_y = target.shape[1] / coarse.shape[1]
        scale_x = target.shape[2] / coarse.shape[2]
        if scale_y != scale_x or int(scale_y) != scale_y:
            raise ValueError(f"{sample_path.name}: scala coarse/fine non intera")
        scale = int(scale_y)
        if scale < 2 or scale & (scale - 1):
            raise ValueError(f"{sample_path.name}: scala non potenza di due")
        if expected_scale is None:
            expected_scale = scale
        elif scale != expected_scale:
            raise ValueError("fattore di scala variabile fra campioni")
        split = split_by_time[_utc(item["validTime"])]
        for y0 in _starts(coarse.shape[1], patch, stride):
            for x0 in _starts(coarse.shape[2], patch, stride):
                y1, x1 = y0 + patch, x0 + patch
                fy0, fy1 = y0 * scale, y1 * scale
                fx0, fx1 = x0 * scale, x1 * scale
                target_patch = target[:, fy0:fy1, fx0:fx1]
                valid_patch = valid[:, fy0:fy1, fx0:fx1] & np.isfinite(target_patch)
                fractions = valid_patch.reshape(len(output_channels), -1).mean(axis=1)
                if np.any(fractions < minimum_valid):
                    continue
                name = f"{sample_index:06d}_{y0:04d}_{x0:04d}.npz"
                shard_path = shard_dir / name
                _write_npz(
                    shard_path,
                    coarse=coarse[:, y0:y1, x0:x1],
                    target=target_patch,
                    valid_mask=valid_patch,
                )
                records.append({
                    "path": f"samples/{name}",
                    "sha256": _sha256(shard_path),
                    "validTime": str(item["validTime"]),
                    "runTime": item.get("runTime"),
                    "forecastHour": item.get("forecastHour"),
                    "split": split,
                    "coarseWindow": [y0, y1, x0, x1],
                    "fineWindow": [fy0, fy1, fx0, fx1],
                    "sourceSha256": _sha256(sample_path),
                    "validFractionByOutput": [float(value) for value in fractions],
                })
        print(
            f"[{sample_index + 1}/{len(ordered)}] {item['validTime']}: "
            f"{len(records)} patch cumulative", flush=True,
        )
    if not records:
        raise ValueError("nessuna patch supera i criteri di validità")

    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "task": "orographic-downscaling",
        "coarseChannels": list(coarse_channels),
        "staticChannels": list(STATIC_DOWNSCALING_FEATURES),
        "outputChannels": list(output_channels),
        "schemaHash": schema_hash(
            coarse_channels, STATIC_DOWNSCALING_FEATURES, output_channels
        ),
        "targetAuthority": target_authority,
        "sourceManifestSha256": _sha256(source_path),
        "sharedStatic": {
            "path": static_output.name,
            "sha256": _sha256(static_output),
            "sourceSha256": _sha256(static_source),
        },
        "grid": source.get("grid"),
        "scale": expected_scale,
        "splitPolicy": "chronological-70-15-15-by-complete-valid-time",
        "samples": records,
    }
    target_path = output / "manifest.json"
    partial = target_path.with_suffix(".json.part")
    partial.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, target_path)
    print(target_path)


if __name__ == "__main__":
    main()
