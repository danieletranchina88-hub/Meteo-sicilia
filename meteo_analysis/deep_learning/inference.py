"""Seam-free tiled inference helpers for full ICON-2I domains."""

from __future__ import annotations

import torch

from .downscaling import enforce_block_mean, enforce_nonnegative_block_mean


def _starts(length: int, tile: int, stride: int):
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def tiled_front_logits(model, inputs, *, tile_size=256, overlap=64):
    """Blend overlapping U-Net tiles with a raised-cosine weight window."""
    if inputs.ndim != 4 or inputs.shape[0] != 1:
        raise ValueError("inferenza tiled prevista per un singolo campo NCHW")
    tile_size, overlap = int(tile_size), int(overlap)
    if tile_size < 32 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_size/overlap non validi")
    height, width = inputs.shape[-2:]
    pad_y, pad_x = max(0, tile_size - height), max(0, tile_size - width)
    padded = torch.nn.functional.pad(inputs, (0, pad_x, 0, pad_y), mode="replicate")
    height_p, width_p = padded.shape[-2:]
    stride = tile_size - overlap
    window_1d = torch.hann_window(
        tile_size, periodic=False, dtype=inputs.dtype, device=inputs.device
    ).clamp_min(0.05)
    window = torch.outer(window_1d, window_1d)[None, None]
    accumulated = None
    frontness = None
    authority = torch.zeros(
        (1, 1, height_p, width_p), dtype=inputs.dtype, device=inputs.device
    )
    for y in _starts(height_p, tile_size, stride):
        for x in _starts(width_p, tile_size, stride):
            output = model(padded[:, :, y:y + tile_size, x:x + tile_size])
            if accumulated is None:
                accumulated = torch.zeros(
                    (1, output["class_logits"].shape[1], height_p, width_p),
                    dtype=output["class_logits"].dtype,
                    device=output["class_logits"].device,
                )
                frontness = torch.zeros_like(authority)
            accumulated[:, :, y:y + tile_size, x:x + tile_size] += (
                output["class_logits"] * window
            )
            frontness[:, :, y:y + tile_size, x:x + tile_size] += (
                output["frontness_logits"][:, None] * window
            )
            authority[:, :, y:y + tile_size, x:x + tile_size] += window
    return {
        "class_logits": (accumulated / authority)[:, :, :height, :width],
        "frontness_logits": (frontness / authority)[:, 0, :height, :width],
    }


def tiled_downscaling(
    model, coarse, static, cell_area_m2, *,
    tile_size_coarse=128, overlap_coarse=16
):
    """Memory-bounded full-domain downscaling with post-blend constraints."""
    if coarse.ndim != 4 or static.ndim != 4 or coarse.shape[0] != 1:
        raise ValueError("inferenza downscaling tiled prevista per batch unitario")
    tile = int(tile_size_coarse)
    overlap = int(overlap_coarse)
    if tile < 4 or overlap < 0 or overlap >= tile:
        raise ValueError("tile/overlap coarse non validi")
    scale = int(model.scale)
    height, width = coarse.shape[-2:]
    if static.shape[-2:] != (height * scale, width * scale):
        raise ValueError("griglia statica non allineata al coarse")
    if cell_area_m2.ndim == 3:
        cell_area_m2 = cell_area_m2[:, None]
    if cell_area_m2.shape != (1, 1, height * scale, width * scale):
        raise ValueError("area celle non allineata alla griglia fine")
    pad_y, pad_x = max(0, tile - height), max(0, tile - width)
    coarse_pad = torch.nn.functional.pad(
        coarse, (0, pad_x, 0, pad_y), mode="replicate"
    )
    static_pad = torch.nn.functional.pad(
        static, (0, pad_x * scale, 0, pad_y * scale), mode="replicate"
    )
    area_pad = torch.nn.functional.pad(
        cell_area_m2, (0, pad_x * scale, 0, pad_y * scale), mode="replicate"
    )
    padded_h, padded_w = coarse_pad.shape[-2:]
    stride = tile - overlap
    window_1d = torch.hann_window(
        tile, periodic=False, dtype=coarse.dtype, device=coarse.device
    ).clamp_min(0.05)
    coarse_window = torch.outer(window_1d, window_1d)[None, None]
    high_window = coarse_window.repeat_interleave(
        scale, -2
    ).repeat_interleave(scale, -1)
    corrected_sum = torch.zeros(
        (1, model.output_channels, padded_h, padded_w),
        dtype=coarse.dtype, device=coarse.device,
    )
    prediction_sum = torch.zeros(
        (1, model.output_channels, padded_h * scale, padded_w * scale),
        dtype=coarse.dtype, device=coarse.device,
    )
    coarse_authority = torch.zeros_like(corrected_sum[:, :1])
    high_authority = torch.zeros_like(prediction_sum[:, :1])
    for y in _starts(padded_h, tile, stride):
        for x in _starts(padded_w, tile, stride):
            hy, hx = y * scale, x * scale
            result = model(
                coarse_pad[:, :, y:y + tile, x:x + tile],
                static_pad[
                    :, :, hy:hy + tile * scale, hx:hx + tile * scale
                ],
                area_pad[
                    :, :, hy:hy + tile * scale, hx:hx + tile * scale
                ],
            )
            corrected_sum[:, :, y:y + tile, x:x + tile] += (
                result["corrected_coarse"] * coarse_window
            )
            coarse_authority[:, :, y:y + tile, x:x + tile] += coarse_window
            prediction_sum[
                :, :, hy:hy + tile * scale, hx:hx + tile * scale
            ] += result["prediction"] * high_window
            high_authority[
                :, :, hy:hy + tile * scale, hx:hx + tile * scale
            ] += high_window
    corrected = (corrected_sum / coarse_authority)[:, :, :height, :width]
    prediction = (prediction_sum / high_authority)[
        :, :, :height * scale, :width * scale
    ]
    area_weights = cell_area_m2[:, :, :height * scale, :width * scale]
    rain_index = int(model.precipitation_index)
    pieces = []
    if rain_index:
        pieces.append(enforce_block_mean(
            prediction[:, :rain_index], corrected[:, :rain_index], scale,
            area_weights,
        ))
    pieces.append(enforce_nonnegative_block_mean(
        prediction[:, rain_index:rain_index + 1],
        corrected[:, rain_index:rain_index + 1], scale, area_weights,
    ))
    if rain_index + 1 < model.output_channels:
        pieces.append(enforce_block_mean(
            prediction[:, rain_index + 1:], corrected[:, rain_index + 1:], scale,
            area_weights,
        ))
    return {
        "prediction": torch.cat(pieces, 1),
        "corrected_coarse": corrected,
        "area_weights": area_weights,
    }
