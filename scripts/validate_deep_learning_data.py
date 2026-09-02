#!/usr/bin/env python3
"""Fail-fast validation of every numeric shard in a training manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meteo_analysis.deep_learning.data import TensorManifestDataset, load_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    payload = load_manifest(args.manifest)
    checked = 0
    for split in ("train", "validation", "test"):
        dataset = TensorManifestDataset(
            args.manifest, task=payload["task"], split=split
        )
        for index in range(len(dataset)):
            dataset[index]
            checked += 1
        print(f"{split}: {len(dataset)} campioni validi")
    print(f"OK: {checked} shard {payload['task']} verificati senza pickle.")


if __name__ == "__main__":
    main()
