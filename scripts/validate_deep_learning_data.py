#!/usr/bin/env python3
"""Fail-fast validation of every numeric shard in a training manifest."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meteo_analysis.deep_learning.data import TensorManifestDataset, load_manifest


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    payload = load_manifest(args.manifest)
    shared = payload.get("sharedStatic")
    if (
        shared and shared.get("sha256")
        and _sha256(payload["_resolvedSharedStatic"]) != shared["sha256"]
    ):
        raise ValueError("checksum dello statico condiviso non valido")
    checked = 0
    for split in ("train", "validation", "test"):
        dataset = TensorManifestDataset(
            args.manifest, task=payload["task"], split=split
        )
        for index in range(len(dataset)):
            dataset[index]
            record = dataset.samples[index]
            expected_hash = record.get("sha256")
            if expected_hash and _sha256(record["_resolvedPath"]) != expected_hash:
                raise ValueError(f"checksum shard non valido: {record['path']}")
            checked += 1
        print(f"{split}: {len(dataset)} campioni validi")
    print(f"OK: {checked} shard {payload['task']} verificati senza pickle.")


if __name__ == "__main__":
    main()
