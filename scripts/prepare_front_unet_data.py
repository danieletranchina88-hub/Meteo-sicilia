#!/usr/bin/env python3
"""Build chronological, multi-class DWD/ERA5 tensors for FrontUNet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meteo_analysis.deep_learning.front_inputs import build_front_tensor
from meteo_analysis.deep_learning.schemas import (
    DWD_SUPERVISED_FRONT_CLASSES,
    FRONT_FEATURES,
    SCHEMA_VERSION,
    schema_hash,
)
from meteo_analysis.ml.dwd_labels import iter_archive
from meteo_analysis.ml.era5 import EarthmoverERA5
from meteo_analysis.ml.features import theta_gradient_50
from meteo_analysis.ml.labels import grid_labels

DWD_URL = (
    "https://zenodo.org/api/records/5785817/files/"
    "DWDFrontsNA.tar.gz/content"
)
DWD_MD5 = "e9a9c26a5d5d10b6f83d7d5726115a50"


def download_archive(destination):
    destination = Path(destination)
    if destination.exists():
        if hashlib.md5(destination.read_bytes()).hexdigest() == DWD_MD5:
            return destination
        raise ValueError("archivio DWD esistente con checksum errato")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(DWD_URL, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        output.write(chunk)
        if hashlib.md5(partial.read_bytes()).hexdigest() != DWD_MD5:
            raise ValueError("checksum MD5 dell'archivio DWD non valido")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def _split(index, count):
    fraction = index / count
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "validation"
    return "test"


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
    parser.add_argument("--archive", default="training_data/DWDFrontsNA.tar.gz")
    parser.add_argument("--output", default="training_data/front_unet")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2019-12-31")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    archive = download_archive(args.archive)
    analyses = sorted(
        iter_archive(archive, start=args.start, end=args.end, hours=(0,)),
        key=lambda item: item[0],
    )[::max(1, args.stride)]
    if args.limit:
        analyses = analyses[:args.limit]
    if len(analyses) < 30:
        raise ValueError("servono almeno 30 analisi per split temporali attendibili")

    output = Path(args.output)
    shard_dir = output / "samples"
    shard_dir.mkdir(parents=True, exist_ok=True)
    era5 = EarthmoverERA5()
    records = []
    class_counts_by_split = {
        split: np.zeros(len(DWD_SUPERVISED_FRONT_CLASSES), dtype=np.int64)
        for split in ("train", "validation", "test")
    }
    try:
        for index, (valid_time, fronts) in enumerate(analyses):
            unsupported = sorted({
                str(front.get("tipo", front.get("type", ""))).lower()
                for front in fronts
                if str(front.get("tipo", front.get("type", ""))).lower()
                not in DWD_SUPERVISED_FRONT_CLASSES
            })
            if unsupported:
                raise ValueError(
                    "l'archivio contiene classi non supervisionabili: "
                    + ", ".join(unsupported)
                )
            fields = era5.fields(valid_time)
            fields.update({
                f"{variable}{level}": era5.pressure_field(
                    variable, level, valid_time
                )
                for level in (925, 700)
                for variable in ("t", "q", "u", "v")
            })
            fields["omega700"] = era5.pressure_field("w", 700, valid_time)
            previous = era5.single_field("msl", valid_time - timedelta(hours=3))
            history = [
                theta_gradient_50(
                    era5.pressure_field(
                        "t", 850, valid_time - timedelta(hours=offset)
                    ),
                    era5.target_latitudes,
                    era5.target_longitudes,
                )
                for offset in (2, 1)
            ]
            inputs = build_front_tensor(
                fields, era5.target_latitudes, era5.target_longitudes,
                valid_time=valid_time, previous_pmsl_3h=previous,
                gradient_history=history,
            )
            labels = grid_labels(
                fronts, valid_time=valid_time, multiclass=True
            )
            shape = (era5.target_latitudes.size, era5.target_longitudes.size)
            target = labels.y.to_numpy(dtype=np.int64).reshape(shape)
            weights = labels.labelWeight.to_numpy(dtype=np.float32).reshape(shape)
            valid = np.mean(np.isfinite(inputs), axis=0) >= 0.80
            split = _split(index, len(analyses))
            labelled = target[valid]
            if labelled.size == 0 or np.max(labelled) >= len(
                DWD_SUPERVISED_FRONT_CLASSES
            ):
                raise ValueError("target DWD incompatibile con la tassonomia dichiarata")
            class_counts_by_split[split] += np.bincount(
                labelled, minlength=len(DWD_SUPERVISED_FRONT_CLASSES)
            )
            name = valid_time.strftime("%Y%m%d%H.npz")
            shard_path = shard_dir / name
            _write_npz(
                shard_path, inputs=inputs, target=target,
                label_weight=weights, valid_mask=valid,
            )
            records.append({
                "path": f"samples/{name}",
                "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                "validTime": valid_time.isoformat().replace("+00:00", "Z"),
                "split": split,
                "frontCount": len(fronts),
                "frontCountByClass": {
                    name: sum(front["tipo"] == name for front in fronts)
                    for name in DWD_SUPERVISED_FRONT_CLASSES[1:]
                },
            })
            print(
                f"[{index + 1}/{len(analyses)}] {valid_time:%Y-%m-%d}: "
                f"{len(fronts)} linee, split={records[-1]['split']}", flush=True,
            )
    finally:
        era5.close()

    missing = [
        DWD_SUPERVISED_FRONT_CLASSES[index]
        for index, count in enumerate(class_counts_by_split["train"])
        if count == 0
    ]
    if missing:
        raise ValueError(
            "classi supervisionate assenti dal training set: " + ", ".join(missing)
        )

    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "task": "front-segmentation",
        "channels": list(FRONT_FEATURES),
        "classes": list(DWD_SUPERVISED_FRONT_CLASSES),
        "unsupportedClasses": ["stationary"],
        "schemaHash": schema_hash(
            FRONT_FEATURES, DWD_SUPERVISED_FRONT_CLASSES
        ),
        "predictorSource": "ERA5",
        "operationalPredictor": "ICON-2I",
        "grid": {
            "resolutionDegrees": 0.20,
            "purpose": "synoptic recognition; native ICON grid refines geometry",
        },
        "targetAuthority": {
            "name": "DWD manual front polylines",
            "doi": "10.5281/zenodo.5785816",
            "license": "CC BY 4.0",
            "rasterBufferKm": 40.0,
            "supervisedClasses": ["cold", "warm", "occluded"],
            "unsupportedClasses": ["stationary"],
            "knownLimitation": (
                "DWD stationary fronts are drawn as alternating cold/warm "
                "segments and cannot be separated reliably in this archive"
            ),
            "methodReference": "https://doi.org/10.5194/wcd-3-113-2022",
        },
        "splitPolicy": "chronological-70-15-15-by-complete-valid-time",
        "classPixelCountsBySplit": {
            split: {
                name: int(counts[index])
                for index, name in enumerate(DWD_SUPERVISED_FRONT_CLASSES)
            }
            for split, counts in class_counts_by_split.items()
        },
        "samples": records,
    }
    target = output / "manifest.json"
    partial = target.with_suffix(".json.part")
    partial.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, target)
    print(target)


if __name__ == "__main__":
    main()
