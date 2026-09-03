"""Shape, gradient, conservation and data-leakage tests for PyTorch models."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import torch
except ImportError:
    print("SKIP: PyTorch non installato; usare requirements-deep-learning.txt")
    raise SystemExit(0)

from meteo_analysis.deep_learning.data import TensorManifestDataset, load_manifest
from meteo_analysis.deep_learning.downscaling import (
    DownscalingLoss,
    OrographicDownscaler,
    block_mean,
)
from meteo_analysis.deep_learning.front_inputs import build_front_tensor
from meteo_analysis.deep_learning.fronts import FrontSegmentationLoss, FrontUNet
from meteo_analysis.deep_learning.inference import (
    tiled_downscaling,
    tiled_front_logits,
)
from meteo_analysis.deep_learning.preparation import build_orographic_sample
from meteo_analysis.deep_learning.training import (
    atomic_torch_checkpoint,
    load_checkpoint,
)

torch.manual_seed(7)

# The front tensor includes the full 925-850-700 hPa structure and preserves
# the versioned order expected by checkpoints.
from meteo_analysis.deep_learning.schemas import (
    DWD_SUPERVISED_FRONT_CLASSES,
    FRONT_CLASSES,
    FRONT_FEATURES,
    validate_front_class_schema,
)

assert DWD_SUPERVISED_FRONT_CLASSES == FRONT_CLASSES[:-1]
assert validate_front_class_schema(DWD_SUPERVISED_FRONT_CLASSES) == (
    "background", "cold", "warm", "occluded"
)
for invalid_classes in (
    ("cold", "warm"),
    ("background", "warm", "cold"),
    ("background", "cold", "unknown"),
):
    try:
        validate_front_class_schema(invalid_classes)
    except ValueError:
        pass
    else:
        raise AssertionError(f"schema classi invalido accettato: {invalid_classes}")

grid_lat = np.linspace(38.0, 42.0, 8)
grid_lon = np.linspace(8.0, 15.0, 12)
grid_shape = (len(grid_lat), len(grid_lon))
constant = lambda value: np.full(grid_shape, value, dtype=float)
front_fields = {
    "t850": constant(283), "q850": constant(0.006),
    "u850": constant(10), "v850": constant(2),
    "t700": constant(273), "q700": constant(0.003),
    "u700": constant(15), "v700": constant(3),
    "t925": constant(288), "q925": constant(0.008),
    "u925": constant(7), "v925": constant(1),
    "u500": constant(20), "v500": constant(5), "fi500": constant(55_000),
    "u10": constant(4), "v10": constant(1), "pmsl": constant(101_300),
    "omega700": constant(-0.1),
    "wshear_u_0_6km": constant(12), "wshear_v_0_6km": constant(3),
    "hsurf": constant(100), "ruggedness_10km": constant(20),
}
front_tensor = build_front_tensor(
    front_fields, grid_lat, grid_lon,
    valid_time="2025-01-01T00:00:00Z",
    previous_pmsl_3h=constant(101_500),
    gradient_history=[constant(0.0), constant(0.0)],
)
assert front_tensor.shape == (len(FRONT_FEATURES), *grid_shape)
assert np.all(np.isfinite(front_tensor))

# Odd dimensions must survive the encoder/decoder without cropping labels.
front = FrontUNet(
    input_mean=[0.0] * 5,
    input_standard_deviation=[1.0] * 5,
    class_count=5,
    base_channels=8,
)
front_input = torch.randn(2, 5, 65, 71)
front_input[0, 2, 4:8, 9:12] = float("nan")
front_target = torch.randint(0, 5, (2, 65, 71))
front_target[:, :2] = -100
front_output = front(front_input)
assert front_output["class_logits"].shape == (2, 5, 65, 71)
front_losses = FrontSegmentationLoss(class_weights=[0.2, 1, 1, 1, 1])(
    front_output,
    front_target,
    label_weight=torch.ones_like(front_target, dtype=torch.float32),
    physics_support=torch.ones_like(front_target, dtype=torch.float32),
)
assert torch.isfinite(front_losses["loss"])
front_losses["loss"].backward()
assert any(parameter.grad is not None for parameter in front.parameters())

# Tiled inference returns a seamless tensor with the exact original extent.
front.eval()
with torch.no_grad():
    tiled = tiled_front_logits(front, front_input[:1], tile_size=48, overlap=16)
assert tiled["class_logits"].shape == (1, 5, 65, 71)
assert torch.isfinite(tiled["class_logits"]).all()

# x4 output must be exactly consistent with corrected coarse means.  Rain is
# non-negative and uses the same conservation check as temperature/wind.
downscaler = OrographicDownscaler(
    coarse_mean=[280.0, 1.0, 0.0, 0.0, 1000.0],
    coarse_standard_deviation=[8.0, 2.0, 8.0, 8.0, 12.0],
    static_mean=[500.0, 0.0, 0.0, 0.6, 30.0],
    static_standard_deviation=[600.0, 0.1, 0.1, 0.45, 40.0],
    output_coarse_indices=(0, 1, 2, 3),
    scale=4,
    base_channels=8,
    residual_blocks=2,
)
coarse = torch.randn(2, 5, 8, 9)
coarse[:, 0] += 280.0
coarse[:, 1] = torch.rand(2, 8, 9) * 5.0
static = torch.randn(2, 5, 32, 36)
cell_area = 1_000_000.0 + 100_000.0 * torch.rand(2, 32, 36)
result = downscaler(coarse, static, cell_area)
prediction = result["prediction"]
assert prediction.shape == (2, 4, 32, 36)
assert torch.min(prediction[:, 1]) >= 0
consistency = torch.max(torch.abs(
    block_mean(prediction, 4, result["area_weights"])
    - result["corrected_coarse"]
)).item()
print(f"errore massimo riaggregazione={consistency:.8g}")
# Around 280 K one float32 ulp is ~3e-5 K; two ulps is the expected ceiling
# after pooling and projection.
assert consistency < 1.0e-4
downscaler.eval()
with torch.no_grad():
    tiled_downscale = tiled_downscaling(
        downscaler, coarse[:1], static[:1], cell_area[:1],
        tile_size_coarse=6, overlap_coarse=2,
    )
tiled_consistency = torch.max(torch.abs(
    block_mean(
        tiled_downscale["prediction"], 4, tiled_downscale["area_weights"]
    )
    - tiled_downscale["corrected_coarse"]
)).item()
assert tiled_downscale["prediction"].shape == (1, 4, 32, 36)
assert tiled_consistency < 1.0e-4
target = prediction.detach() + 0.1 * torch.randn_like(prediction)
target[:, 1].clamp_min_(0.0)
downscale_losses = DownscalingLoss()(result, target, torch.ones_like(target, dtype=torch.bool))
assert torch.isfinite(downscale_losses["loss"])
downscale_losses["loss"].backward()
assert any(parameter.grad is not None for parameter in downscaler.parameters())

# DEM derivatives use metres/metres and the coast distance is geodesic, not a
# pixel count that changes when the target resolution changes.
latitude = np.linspace(37.0, 37.2, 8)
longitude = np.linspace(14.0, 14.3, 12)
lon2d, lat2d = np.meshgrid(longitude, latitude)
elevation = 100.0 + 200.0 * (lon2d - longitude[0])
land = np.zeros_like(elevation)
land[:, 5:] = 1.0
built = build_orographic_sample(
    np.zeros((5, 2, 3), dtype=np.float32),
    np.stack((
        np.full_like(elevation, 285.0), np.zeros_like(elevation),
        np.zeros_like(elevation), np.zeros_like(elevation),
    )),
    elevation_m=elevation, land_fraction=land,
    latitude=latitude, longitude=longitude,
)
assert built["static"].shape == (5, 8, 12)
assert built["scale"] == 4
assert np.all(built["static"][4] >= 0)
assert np.all(built["cell_area_m2"] > 0)

# A whole valid time cannot leak across split boundaries, even if filenames
# differ because the same atmosphere was divided into multiple patches.
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for name in ("first.npz", "second.npz"):
        np.savez(root / name, inputs=np.zeros((1, 2, 2)), target=np.zeros((2, 2)))
    manifest = {
        "schemaVersion": 1,
        "task": "front-segmentation",
        "samples": [
            {"path": "first.npz", "validTime": "2024-01-01T00:00:00Z", "split": "train"},
            {"path": "second.npz", "validTime": "2024-01-01T00:00:00Z", "split": "test"},
        ],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        load_manifest(path)
    except ValueError as error:
        assert "data leakage" in str(error)
    else:
        raise AssertionError("manifest leakage non rilevato")

    checkpoint = root / "candidate.pt"
    atomic_torch_checkpoint(
        checkpoint,
        state_dict={"weight": torch.ones(1)},
        metadata={"formatVersion": 1, "accepted": False, "torchVersion": str(torch.__version__)},
    )
    try:
        load_checkpoint(checkpoint)
    except ValueError as error:
        assert "non promosso" in str(error)
    else:
        raise AssertionError("checkpoint candidato accettato come operativo")
    state, metadata = load_checkpoint(checkpoint, require_accepted=False)
    assert torch.equal(state["weight"], torch.ones(1))
    assert metadata["accepted"] is False

    down_path = root / "down.npz"
    np.savez(
        down_path,
        coarse=np.stack((
            np.full((2, 3), 285.0), np.zeros((2, 3)),
            np.zeros((2, 3)), np.zeros((2, 3)), np.full((2, 3), 101_300.0),
        )).astype(np.float32),
        static=np.zeros((5, 8, 12), dtype=np.float32),
        target=np.stack((
            np.full((8, 12), 285.0), np.zeros((8, 12)),
            np.zeros((8, 12)), np.zeros((8, 12)),
        )).astype(np.float32),
        cell_area_m2=np.full((8, 12), 300_000.0, dtype=np.float32),
    )
    down_manifest = root / "down.json"
    down_manifest.write_text(json.dumps({
        "schemaVersion": 1,
        "task": "orographic-downscaling",
        "coarseChannels": ["t", "rain", "u", "v", "pmsl"],
        "staticChannels": [
            "elevation_m", "slope_east_m_per_m", "slope_north_m_per_m",
            "land_fraction", "distance_to_coast_km",
        ],
        "outputChannels": [
            "temperature_2m_k", "precipitation_rate_mm_h",
            "wind_u10_m_s", "wind_v10_m_s",
        ],
        "samples": [
            {"path": "down.npz", "validTime": "2024-01-01T00:00:00Z", "split": "train"},
            {"path": "down.npz", "validTime": "2024-02-01T00:00:00Z", "split": "validation"},
            {"path": "down.npz", "validTime": "2024-03-01T00:00:00Z", "split": "test"},
        ],
    }), encoding="utf-8")
    down_dataset = TensorManifestDataset(
        down_manifest, task="orographic-downscaling", split="test"
    )
    assert down_dataset[0]["cell_area_m2"].shape == (8, 12)

    # Static terrain is large but time-invariant.  A shared full-domain file
    # plus an explicit fine-grid window must produce the same sample contract.
    shared_path = root / "shared_static.npz"
    np.savez(
        shared_path,
        static=np.zeros((5, 16, 20), dtype=np.float32),
        cell_area_m2=np.full((16, 20), 250_000.0, dtype=np.float32),
    )
    shared_sample = root / "shared_sample.npz"
    np.savez(
        shared_sample,
        coarse=np.zeros((5, 2, 3), dtype=np.float32),
        target=np.stack((
            np.full((8, 12), 285.0), np.zeros((8, 12)),
            np.zeros((8, 12)), np.zeros((8, 12)),
        )).astype(np.float32),
    )
    shared_manifest = root / "shared.json"
    shared_manifest.write_text(json.dumps({
        "schemaVersion": 1,
        "task": "orographic-downscaling",
        "coarseChannels": ["t", "rain", "u", "v", "pmsl"],
        "staticChannels": [
            "elevation_m", "slope_east_m_per_m", "slope_north_m_per_m",
            "land_fraction", "distance_to_coast_km",
        ],
        "outputChannels": [
            "temperature_2m_k", "precipitation_rate_mm_h",
            "wind_u10_m_s", "wind_v10_m_s",
        ],
        "sharedStatic": {"path": "shared_static.npz"},
        "samples": [
            {"path": "shared_sample.npz", "validTime": "2024-01-01T00:00:00Z", "split": "train", "fineWindow": [4, 12, 4, 16]},
            {"path": "shared_sample.npz", "validTime": "2024-02-01T00:00:00Z", "split": "validation", "fineWindow": [4, 12, 4, 16]},
            {"path": "shared_sample.npz", "validTime": "2024-03-01T00:00:00Z", "split": "test", "fineWindow": [4, 12, 4, 16]},
        ],
    }), encoding="utf-8")
    shared_dataset = TensorManifestDataset(
        shared_manifest, task="orographic-downscaling", split="test"
    )
    shared_item = shared_dataset[0]
    assert shared_item["static"].shape == (5, 8, 12)
    assert shared_item["cell_area_m2"].shape == (8, 12)

print("OK: U-Net, downscaling conservativo e split temporali verificati.")
