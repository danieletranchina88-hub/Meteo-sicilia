#!/usr/bin/env python3
"""Export verified candidate-checkpoint metadata as transparent JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meteo_analysis.deep_learning.training import load_checkpoint


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    state_dict, metadata = load_checkpoint(
        checkpoint, require_accepted=False, map_location="cpu"
    )
    report = {
        "checkpoint": checkpoint.name,
        "checkpointBytes": checkpoint.stat().st_size,
        "checkpointSha256": file_sha256(checkpoint),
        "tensorCount": len(state_dict),
        "metadata": metadata,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    partial.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, output)
    print(output)


if __name__ == "__main__":
    main()
