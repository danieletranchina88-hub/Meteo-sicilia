#!/usr/bin/env python3
"""Run a promoted or explicitly allowed downscaling checkpoint on one NPZ."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meteo_analysis.deep_learning.downscaling import OrographicDownscaler
from meteo_analysis.deep_learning.inference import tiled_downscaling
from meteo_analysis.deep_learning.schemas import (
    DOWNSCALING_OUTPUTS,
    STATIC_DOWNSCALING_FEATURES,
    schema_hash,
)
from meteo_analysis.deep_learning.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="NPZ con array coarse e static")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-candidate", action="store_true")
    parser.add_argument("--tile-size-coarse", type=int, default=128)
    parser.add_argument("--overlap-coarse", type=int, default=16)
    args = parser.parse_args()
    state, metadata = load_checkpoint(
        args.checkpoint, require_accepted=not args.allow_candidate
    )
    coarse_channels = tuple(metadata.get("coarseChannels") or ())
    static_channels = tuple(metadata.get("staticChannels") or ())
    outputs = tuple(metadata.get("outputChannels") or ())
    if (
        not coarse_channels
        or static_channels != STATIC_DOWNSCALING_FEATURES
        or outputs != DOWNSCALING_OUTPUTS
        or metadata.get("schemaHash")
        != schema_hash(coarse_channels, static_channels, outputs)
    ):
        raise ValueError("checkpoint downscaling con schema incompatibile")
    config = metadata["modelConfig"]
    normal = metadata["normalization"]
    model = OrographicDownscaler(
        coarse_mean=normal["coarse"]["mean"],
        coarse_standard_deviation=normal["coarse"]["standardDeviation"],
        static_mean=normal["static"]["mean"],
        static_standard_deviation=normal["static"]["standardDeviation"],
        output_coarse_indices=config["outputCoarseIndices"],
        scale=config["scale"],
        base_channels=config["baseChannels"],
        residual_blocks=config["residualBlocks"],
        bias_correction=config["biasCorrection"],
    )
    model.load_state_dict(state, strict=True)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    model.to(device).eval()
    with np.load(args.input, allow_pickle=False) as archive:
        coarse = np.asarray(archive["coarse"], dtype=np.float32)
        static = np.asarray(archive["static"], dtype=np.float32)
        cell_area = np.asarray(archive["cell_area_m2"], dtype=np.float32)
    if coarse.ndim == 3:
        coarse = coarse[None]
    if static.ndim == 3:
        static = static[None]
    if cell_area.ndim == 2:
        cell_area = cell_area[None]
    with torch.inference_mode():
        result = tiled_downscaling(
            model,
            torch.from_numpy(coarse).to(device),
            torch.from_numpy(static).to(device),
            torch.from_numpy(cell_area).to(device),
            tile_size_coarse=args.tile_size_coarse,
            overlap_coarse=args.overlap_coarse,
        )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with partial.open("wb") as output_file:
            np.savez_compressed(
                output_file,
                prediction=result["prediction"].cpu().numpy().astype(np.float32),
                corrected_coarse=result["corrected_coarse"].cpu().numpy().astype(np.float32),
            )
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
