#!/usr/bin/env python3
"""Train a candidate FrontUNet; promotion requires explicit held-out gates."""

from __future__ import annotations

import argparse
import hashlib
import os
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
from meteo_analysis.deep_learning.fronts import FrontSegmentationLoss, FrontUNet
from meteo_analysis.deep_learning.schemas import (
    FRONT_CLASSES,
    FRONT_FEATURES,
    schema_hash,
    validate_front_class_schema,
)
from meteo_analysis.deep_learning.training import (
    atomic_torch_checkpoint,
    front_confusion_matrix,
    front_metrics,
    read_json,
    seed_everything,
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_weights(dataset, class_count):
    counts = torch.zeros(class_count, dtype=torch.float64)
    for sample in dataset:
        target = sample["target"]
        valid = target >= 0
        counts += torch.bincount(target[valid], minlength=class_count)
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0).flatten().tolist()
        raise ValueError(f"classi assenti dal training set: {missing}")
    frequency = counts / counts.sum().clamp_min(1.0)
    weights = 1.0 / torch.sqrt(frequency.clamp_min(1.0e-6))
    return (weights / weights.mean()).to(torch.float32)


def _epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total, batches = 0.0, 0
    confusion = torch.zeros(
        model.class_count, model.class_count, dtype=torch.int64, device=device
    )
    for batch in loader:
        inputs = batch["inputs"].to(device)
        target = batch["target"].to(device)
        weights = batch["label_weight"].to(device)
        support = batch["physics_support"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        enabled = scaler is not None
        with torch.amp.autocast("cuda", enabled=enabled):
            output = model(inputs)
            losses = criterion(
                output, target, label_weight=weights, physics_support=support
            )
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
        total += float(losses["loss"].detach())
        batches += 1
        confusion += front_confusion_matrix(
            output["class_logits"].detach(), target, model.class_count
        )
    return total / max(batches, 1), front_metrics(confusion)


def _fit_temperature(model, loader, device, *, maximum_pixels=2_500_000):
    """Scalar temperature scaling using validation pixels only."""
    collected_logits, collected_target, collected_weight = [], [], []
    remaining = int(maximum_pixels)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(batch["inputs"].to(device))["class_logits"]
            target = batch["target"].to(device)
            weight = batch["label_weight"].to(device)
            valid = target >= 0
            logits = output.permute(0, 2, 3, 1)[valid]
            target = target[valid]
            weight = weight[valid]
            if not len(target):
                continue
            take = min(remaining, len(target))
            # Uniform deterministic subsampling retains all synoptic cases.
            index = torch.linspace(
                0, len(target) - 1, take, device=device
            ).long()
            collected_logits.append(logits[index].cpu())
            collected_target.append(target[index].cpu())
            collected_weight.append(weight[index].cpu())
            remaining -= take
            if remaining <= 0:
                break
    if not collected_target:
        raise ValueError("validation priva di pixel per la calibrazione")
    logits = torch.cat(collected_logits)
    target = torch.cat(collected_target)
    weight = torch.cat(collected_weight)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        error = torch.nn.functional.cross_entropy(
            logits / temperature, target, reduction="none"
        )
        loss = (error * weight).sum() / weight.sum().clamp_min(1.0)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).clamp(0.05, 20.0))


def _calibration_metrics(model, loader, device, temperature, bins=10):
    squared_error = 0.0
    pixel_count = 0
    bin_count = torch.zeros(bins, dtype=torch.float64)
    bin_confidence = torch.zeros_like(bin_count)
    bin_observed = torch.zeros_like(bin_count)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["inputs"].to(device))["class_logits"]
            target = batch["target"].to(device)
            valid = target >= 0
            probability = 1.0 - torch.softmax(
                logits / float(temperature), dim=1
            )[:, 0]
            observed = (target > 0).to(probability.dtype)
            p = probability[valid].cpu().double()
            y = observed[valid].cpu().double()
            squared_error += float(torch.sum((p - y) ** 2))
            pixel_count += len(p)
            index = torch.clamp((p * bins).long(), max=bins - 1)
            bin_count += torch.bincount(index, minlength=bins)
            bin_confidence.scatter_add_(0, index, p)
            bin_observed.scatter_add_(0, index, y)
    authority = bin_count.clamp_min(1.0)
    gap = torch.abs(bin_confidence / authority - bin_observed / authority)
    ece = torch.sum(gap * bin_count) / max(pixel_count, 1)
    return {
        "frontBrier": squared_error / max(pixel_count, 1),
        "frontExpectedCalibrationError10Bin": float(ece),
        "pixelCount": pixel_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--config", default="configs/front_unet.json")
    parser.add_argument("--output", default="models/candidates/front_unet.pt")
    parser.add_argument("--acceptance-config")
    args = parser.parse_args()
    config = read_json(args.config)
    seed = int(config.get("seed", 271828))
    seed_everything(seed, deterministic=bool(config.get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = TensorManifestDataset(
        args.manifest, task="front-segmentation", split="train"
    )
    validation = TensorManifestDataset(
        args.manifest, task="front-segmentation", split="validation"
    )
    test = TensorManifestDataset(
        args.manifest, task="front-segmentation", split="test"
    )
    channels = tuple(train.manifest.get("channels") or ())
    classes = validate_front_class_schema(train.manifest.get("classes"))
    if channels != FRONT_FEATURES:
        raise ValueError("schema canali non compatibile con FrontUNet")
    if not isinstance(train.manifest.get("targetAuthority"), dict):
        raise TypeError("provenienza delle etichette frontali non documentata")
    stats = estimate_channel_statistics(
        train, "inputs", channels,
        maximum_samples=config.get("normalizationMaximumSamples"),
    )
    class_weights = _class_weights(train, len(classes)).to(device)
    model = FrontUNet(
        input_mean=stats.mean,
        input_standard_deviation=stats.standard_deviation,
        class_count=len(classes),
        base_channels=int(config.get("baseChannels", 32)),
    ).to(device)
    criterion = FrontSegmentationLoss(
        class_weights,
        contradiction_weight=float(config.get("physicsContradictionWeight", 0.05)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("learningRate", 3e-4)),
        weight_decay=float(config.get("weightDecay", 1e-4)),
    )
    batch_size = int(config.get("batchSize", 4))
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
    best_loss = float("inf")
    best_state = None
    history = []
    patience = int(config.get("earlyStoppingPatience", 12))
    remaining = patience
    for epoch in range(1, int(config.get("epochs", 80)) + 1):
        train_loss, _ = _epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        with torch.no_grad():
            validation_loss, validation_metrics = _epoch(
                model, validation_loader, criterion, device
            )
        print(
            f"epoch={epoch:03d} train={train_loss:.5f} "
            f"validation={validation_loss:.5f} "
            f"frontIoU={validation_metrics['meanFrontalIoU']:.4f}", flush=True,
        )
        history.append({
            "epoch": epoch,
            "trainLoss": train_loss,
            "validationLoss": validation_loss,
            "validationMetrics": validation_metrics,
        })
        if validation_loss < best_loss - 1.0e-5:
            best_loss = validation_loss
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
    temperature = _fit_temperature(model, validation_loader, device)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        test_loss, test_metrics = _epoch(model, test_loader, criterion, device)
        test_metrics.update(_calibration_metrics(
            model, test_loader, device, temperature
        ))

    gates = read_json(args.acceptance_config) if args.acceptance_config else None
    accepted = False
    if gates:
        accepted = (
            test_metrics["meanFrontalIoU"] >= float(gates["minimumMeanFrontalIoU"])
            and test_metrics["meanFrontalDice"] >= float(gates["minimumMeanFrontalDice"])
            and all(value is not None for value in test_metrics["perClassIoU"][1:])
            and test_metrics["frontBrier"] <= float(gates["maximumFrontBrier"])
            and test_metrics["frontExpectedCalibrationError10Bin"]
            <= float(gates["maximumFrontCalibrationError"])
            and test_loss <= float(gates["maximumTestLoss"])
        )
    metadata = {
        "formatVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": "physics-aware-residual-front-unet",
        "accepted": accepted,
        "promotionPolicy": (
            "explicit-independent-held-out-gates" if gates else "candidate-only"
        ),
        "classes": list(classes),
        "unsupportedClasses": [
            name for name in FRONT_CLASSES if name not in classes
        ],
        "channels": list(channels),
        "schemaHash": schema_hash(channels, classes),
        "normalization": stats.as_dict(),
        "calibration": {
            "method": "scalar-temperature-scaling",
            "temperature": temperature,
            "fitSplit": "validation",
        },
        "modelConfig": {
            "baseChannels": int(config.get("baseChannels", 32)),
            "classCount": len(classes),
        },
        "manifestSha256": _sha256(args.manifest),
        "splitPolicy": train.manifest.get("splitPolicy"),
        "grid": train.manifest.get("grid"),
        "targetAuthority": train.manifest.get("targetAuthority"),
        "validationLoss": best_loss,
        "testLoss": test_loss,
        "testMetrics": test_metrics,
        "acceptanceCriteria": gates,
        "sampleCounts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "parameterCount": sum(parameter.numel() for parameter in model.parameters()),
        "trainingHistory": history,
        "sourceCommit": os.environ.get("GITHUB_SHA"),
        "trainingDevice": device.type,
        "seed": seed,
        "torchVersion": str(torch.__version__),
    }
    atomic_torch_checkpoint(args.output, state_dict=best_state, metadata=metadata)
    print(f"checkpoint={args.output} accepted={accepted} test={test_metrics}")


if __name__ == "__main__":
    main()
