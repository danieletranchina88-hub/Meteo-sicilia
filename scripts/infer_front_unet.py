#!/usr/bin/env python3
"""Research inference of a validated FrontUNet on run-wide ICON-2I GRIBs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meteo_analysis.deep_learning.front_inputs import build_front_tensor
from meteo_analysis.deep_learning.fronts import FrontUNet
from meteo_analysis.deep_learning.inference import tiled_front_logits
from meteo_analysis.deep_learning.schemas import (
    FRONT_CLASSES,
    FRONT_FEATURES,
    schema_hash,
)
from meteo_analysis.deep_learning.training import load_checkpoint
from meteo_analysis.ml.features import theta_gradient_50_from_store
from meteo_analysis.ml.icon2i import Icon2IStore


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
    parser.add_argument("--run", required=True, help="YYYYMMDDHH")
    parser.add_argument("--paths-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/front_unet")
    parser.add_argument("--hours", nargs="*", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--allow-candidate", action="store_true",
        help="Solo ricerca: consente un checkpoint non promosso.",
    )
    args = parser.parse_args()
    state, metadata = load_checkpoint(
        args.checkpoint, require_accepted=not args.allow_candidate
    )
    if (
        tuple(metadata.get("channels") or ()) != FRONT_FEATURES
        or tuple(metadata.get("classes") or ()) != FRONT_CLASSES
        or metadata.get("schemaHash") != schema_hash(FRONT_FEATURES, FRONT_CLASSES)
    ):
        raise ValueError("checkpoint frontale con schema incompatibile")
    normal = metadata["normalization"]
    model = FrontUNet(
        input_mean=normal["mean"],
        input_standard_deviation=normal["standardDeviation"],
        class_count=len(FRONT_CLASSES),
        base_channels=int(metadata["modelConfig"]["baseChannels"]),
    )
    model.load_state_dict(state, strict=True)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    model.to(device).eval()

    paths = json.loads(Path(args.paths_json).read_text(encoding="utf-8"))
    grid = metadata.get("grid") or {}
    resolution = float(grid.get("resolutionDegrees", 0.20))
    store = Icon2IStore(paths, args.run, resolution=resolution)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        required = (
            "t850", "q850", "u850", "v850", "t700", "q700",
            "u500", "v500", "fi500", "u10", "v10", "pmsl",
        )
        common = set.intersection(*[
            store.available_hours(name) for name in required
        ])
        selected = sorted(common if not args.hours else common & set(args.hours))
        if not selected:
            raise ValueError("nessuna scadenza comune disponibile")
        gradient_cache = {}
        for hour in selected:
            history = []
            for previous in (hour - 2, hour - 1):
                if previous in common:
                    if previous not in gradient_cache:
                        gradient_cache[previous] = theta_gradient_50_from_store(
                            store, previous
                        )
                    history.append(gradient_cache[previous])
            fields = {}
            for name in (
                "t850", "q850", "u850", "v850", "t700", "q700",
                "u500", "v500", "fi500", "u10", "v10", "pmsl",
                "t925", "q925", "u925", "v925", "u700", "v700",
                "omega700",
            ):
                if hour in store.available_hours(name):
                    fields[name] = store.field(name, hour)
            shape = fields["pmsl"].shape
            for name in (
                "wshear_u_0_6km", "wshear_v_0_6km", "hsurf",
                "ruggedness_10km",
            ):
                fields[name] = np.zeros(shape, dtype=np.float32)
            previous_pmsl = (
                store.field("pmsl", hour - 3)
                if hour >= 3 and hour - 3 in store.available_hours("pmsl")
                else None
            )
            valid_time = datetime.strptime(
                args.run, "%Y%m%d%H"
            ).replace(tzinfo=timezone.utc) + timedelta(hours=hour)
            values = build_front_tensor(
                fields, store.target_latitudes, store.target_longitudes,
                valid_time=valid_time, previous_pmsl_3h=previous_pmsl,
                gradient_history=history,
            )[None]
            with torch.inference_mode():
                logits = tiled_front_logits(
                    model, torch.from_numpy(values).to(device),
                    tile_size=256, overlap=64,
                )
                temperature = float(metadata["calibration"]["temperature"])
                classes = torch.softmax(
                    logits["class_logits"] / temperature, dim=1
                )[0]
                # The calibrated multi-class distribution is authoritative;
                # the binary head is an auxiliary training constraint.
                frontness = 1.0 - classes[0]
            _write_npz(
                output_dir / f"front_unet_{hour:03d}.npz",
                class_probability=classes.cpu().numpy().astype(np.float32),
                front_probability=frontness.cpu().numpy().astype(np.float32),
                latitude=store.target_latitudes.astype(np.float32),
                longitude=store.target_longitudes.astype(np.float32),
            )
            print(
                f"+{hour:03d}h: frontness max={float(frontness.max()):.3f}",
                flush=True,
            )
    finally:
        store.close()


if __name__ == "__main__":
    main()
