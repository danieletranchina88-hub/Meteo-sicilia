#!/usr/bin/env python3
"""Train a physically constrained, orography-aware downscaling candidate."""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meteo_analysis.deep_learning.data import (
    TensorManifestDataset,
    estimate_channel_statistics,
)
from meteo_analysis.deep_learning.downscaling import (
    DownscalingLoss,
    OrographicDownscaler,
    block_mean,
)
from meteo_analysis.deep_learning.schemas import (
    DOWNSCALING_OUTPUTS,
    STATIC_DOWNSCALING_FEATURES,
    schema_hash,
)
from meteo_analysis.deep_learning.training import (
    atomic_torch_checkpoint,
    read_json,
    seed_everything,
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    sums = torch.zeros(model.output_channels, dtype=torch.float64, device=device)
    baseline_sums = torch.zeros_like(sums)
    counts = torch.zeros_like(sums)
    total, batches, consistency = 0.0, 0, 0.0
    temperature_bias_sum = 0.0
    wind_vector_sum = 0.0
    baseline_wind_vector_sum = 0.0
    wind_count = 0
    rain_thresholds = (0.1, 1.0, 5.0, 10.0)
    rain_counts = {
        threshold: {"tp": 0, "fp": 0, "fn": 0}
        for threshold in rain_thresholds
    }
    for batch in loader:
        coarse = batch["coarse"].to(device)
        static = batch["static"].to(device)
        cell_area = batch["cell_area_m2"].to(device)
        target = batch["target"].to(device)
        valid = batch["valid_mask"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        enabled = scaler is not None
        with torch.amp.autocast("cuda", enabled=enabled):
            output = model(coarse, static, cell_area)
            losses = criterion(output, target, valid)
        if training:
            if scaler is None:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        error = torch.abs(output["prediction"] - target)
        finite = valid & torch.isfinite(target)
        indices = torch.as_tensor(
            model.output_coarse_indices, device=device, dtype=torch.long
        )
        baseline_coarse = torch.index_select(coarse, 1, indices)
        baseline_coarse[:, model.precipitation_index].clamp_min_(0.0)
        baseline = baseline_coarse.repeat_interleave(
            model.scale, -2
        ).repeat_interleave(model.scale, -1)
        baseline_error = torch.abs(baseline - target)
        sums += torch.where(finite, error, 0.0).sum((0, 2, 3)).double()
        baseline_sums += torch.where(
            finite, baseline_error, 0.0
        ).sum((0, 2, 3)).double()
        counts += finite.sum((0, 2, 3)).double()
        temperature_bias_sum += float(torch.where(
            finite[:, 0], output["prediction"][:, 0] - target[:, 0], 0.0
        ).sum())
        wind_valid = finite[:, 2] & finite[:, 3]
        wind_vector_sum += float(torch.where(
            wind_valid,
            torch.hypot(
                output["prediction"][:, 2] - target[:, 2],
                output["prediction"][:, 3] - target[:, 3],
            ), 0.0,
        ).sum())
        baseline_wind_vector_sum += float(torch.where(
            wind_valid,
            torch.hypot(
                baseline[:, 2] - target[:, 2],
                baseline[:, 3] - target[:, 3],
            ), 0.0,
        ).sum())
        wind_count += int(wind_valid.sum())
        rain_valid = finite[:, model.precipitation_index]
        rain_prediction = output["prediction"][:, model.precipitation_index]
        rain_target = target[:, model.precipitation_index]
        for threshold in rain_thresholds:
            predicted_event = rain_prediction >= threshold
            observed_event = rain_target >= threshold
            rain_counts[threshold]["tp"] += int((
                predicted_event & observed_event & rain_valid
            ).sum())
            rain_counts[threshold]["fp"] += int((
                predicted_event & ~observed_event & rain_valid
            ).sum())
            rain_counts[threshold]["fn"] += int((
                ~predicted_event & observed_event & rain_valid
            ).sum())
        aggregate = block_mean(
            output["prediction"], model.scale, output["area_weights"]
        )
        consistency += float(torch.max(torch.abs(
            aggregate - output["corrected_coarse"]
        )).detach())
        total += float(losses["loss"].detach())
        batches += 1
    mae = sums / counts.clamp_min(1.0)
    baseline_mae = baseline_sums / counts.clamp_min(1.0)
    skill = 1.0 - mae / baseline_mae.clamp_min(1.0e-12)
    csi = {}
    for threshold, values in rain_counts.items():
        denominator = values["tp"] + values["fp"] + values["fn"]
        csi[str(threshold)] = (
            values["tp"] / denominator if denominator else None
        )
    return {
        "loss": total / max(batches, 1),
        "maeByChannel": [float(value) for value in mae.cpu()],
        "baselineMaeByChannel": [float(value) for value in baseline_mae.cpu()],
        "maeSkillByChannel": [float(value) for value in skill.cpu()],
        "temperatureBiasK": temperature_bias_sum / max(int(counts[0]), 1),
        "windVectorMaeMs": wind_vector_sum / max(wind_count, 1),
        "baselineWindVectorMaeMs": baseline_wind_vector_sum / max(wind_count, 1),
        "precipitationCSI": csi,
        "maximumCoarseConsistencyError": consistency / max(batches, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--config", default="configs/orographic_downscaler.json")
    parser.add_argument("--output", default="models/candidates/orographic_downscaler.pt")
    parser.add_argument("--acceptance-config")
    args = parser.parse_args()
    config = read_json(args.config)
    seed = int(config.get("seed", 314159))
    seed_everything(seed, deterministic=bool(config.get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = TensorManifestDataset(
        args.manifest, task="orographic-downscaling", split="train"
    )
    validation = TensorManifestDataset(
        args.manifest, task="orographic-downscaling", split="validation"
    )
    test = TensorManifestDataset(
        args.manifest, task="orographic-downscaling", split="test"
    )
    coarse_channels = tuple(train.manifest.get("coarseChannels") or ())
    static_channels = tuple(train.manifest.get("staticChannels") or ())
    outputs = tuple(train.manifest.get("outputChannels") or ())
    if static_channels != STATIC_DOWNSCALING_FEATURES or outputs != DOWNSCALING_OUTPUTS:
        raise ValueError("schema statico/output del downscaler non compatibile")
    if not coarse_channels:
        raise ValueError("coarseChannels assenti")
    if not isinstance(train.manifest.get("targetAuthority"), dict):
        raise TypeError("provenienza dei target ad alta risoluzione non documentata")
    maximum = config.get("normalizationMaximumSamples")
    coarse_stats = estimate_channel_statistics(
        train, "coarse", coarse_channels, maximum_samples=maximum
    )
    static_stats = estimate_channel_statistics(
        train, "static", static_channels, maximum_samples=maximum
    )
    first = train[0]
    scale = first["target"].shape[-1] // first["coarse"].shape[-1]
    target_indices = tuple(config.get("outputCoarseIndices", (0, 1, 2, 3)))
    model = OrographicDownscaler(
        coarse_mean=coarse_stats.mean,
        coarse_standard_deviation=coarse_stats.standard_deviation,
        static_mean=static_stats.mean,
        static_standard_deviation=static_stats.standard_deviation,
        output_coarse_indices=target_indices,
        scale=scale,
        base_channels=int(config.get("baseChannels", 48)),
        residual_blocks=int(config.get("residualBlocks", 6)),
        bias_correction=bool(config.get("biasCorrection", True)),
    ).to(device)
    criterion = DownscalingLoss(
        channel_weights=config.get("channelWeights", (1, 2, 1, 1)),
        gradient_weight=float(config.get("gradientWeight", 0.15)),
        coarse_bias_weight=float(config.get("coarseBiasWeight", 0.35)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("learningRate", 2e-4)),
        weight_decay=float(config.get("weightDecay", 1e-4)),
    )
    batch_size = int(config.get("batchSize", 2))
    workers = int(config.get("workers", 0))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=workers,
        generator=generator, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    amp = device.type == "cuda" and bool(config.get("automaticMixedPrecision", True))
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp else None
    best_loss, best_state = float("inf"), None
    patience = int(config.get("earlyStoppingPatience", 12))
    remaining = patience
    for epoch in range(1, int(config.get("epochs", 80)) + 1):
        train_metrics = _epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        with torch.no_grad():
            validation_metrics = _epoch(
                model, validation_loader, criterion, device
            )
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.5f} "
            f"validation={validation_metrics['loss']:.5f} "
            f"mae={validation_metrics['maeByChannel']}", flush=True,
        )
        if validation_metrics["loss"] < best_loss - 1.0e-5:
            best_loss = validation_metrics["loss"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            remaining = patience
        else:
            remaining -= 1
            if remaining <= 0:
                break
    if best_state is None:
        raise RuntimeError("addestramento privo di checkpoint valido")
    model.load_state_dict(best_state)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        test_metrics = _epoch(model, test_loader, criterion, device)
    gates = read_json(args.acceptance_config) if args.acceptance_config else None
    accepted = False
    if gates:
        limits = list(map(float, gates["maximumMaeByChannel"]))
        minimum_skill = list(map(float, gates["minimumMaeSkillByChannel"]))
        accepted = (
            len(limits) == len(outputs) == len(minimum_skill)
            and all(value <= limit for value, limit in zip(
                test_metrics["maeByChannel"], limits
            ))
            and all(value >= limit for value, limit in zip(
                test_metrics["maeSkillByChannel"], minimum_skill
            ))
            and test_metrics["windVectorMaeMs"]
            < test_metrics["baselineWindVectorMaeMs"]
            and test_metrics["maximumCoarseConsistencyError"]
            <= float(gates.get("maximumCoarseConsistencyError", 1e-4))
        )
    metadata = {
        "formatVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": "hard-consistent-orographic-residual-downscaler",
        "accepted": accepted,
        "promotionPolicy": (
            "explicit-independent-held-out-gates" if gates else "candidate-only"
        ),
        "coarseChannels": list(coarse_channels),
        "staticChannels": list(static_channels),
        "outputChannels": list(outputs),
        "schemaHash": schema_hash(coarse_channels, static_channels, outputs),
        "normalization": {
            "coarse": coarse_stats.as_dict(), "static": static_stats.as_dict(),
        },
        "modelConfig": {
            "scale": scale,
            "baseChannels": int(config.get("baseChannels", 48)),
            "residualBlocks": int(config.get("residualBlocks", 6)),
            "biasCorrection": bool(config.get("biasCorrection", True)),
            "outputCoarseIndices": list(target_indices),
        },
        "manifestSha256": _sha256(args.manifest),
        "splitPolicy": train.manifest.get("splitPolicy"),
        "targetAuthority": train.manifest.get("targetAuthority"),
        "validationLoss": best_loss,
        "testMetrics": test_metrics,
        "acceptanceCriteria": gates,
        "seed": seed,
        "torchVersion": str(torch.__version__),
    }
    atomic_torch_checkpoint(args.output, state_dict=best_state, metadata=metadata)
    print(f"checkpoint={args.output} accepted={accepted} test={test_metrics}")


if __name__ == "__main__":
    main()
