"""Reproducibility, metrics and atomic PyTorch checkpoint utilities."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

from .schemas import schema_hash


def seed_everything(seed: int, *, deterministic: bool = False):
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True)


def front_confusion_matrix(logits, target, class_count, ignore_index=-100):
    import torch

    prediction = torch.argmax(logits, dim=1)
    valid = target != ignore_index
    encoded = target[valid] * class_count + prediction[valid]
    return torch.bincount(
        encoded, minlength=class_count * class_count
    ).reshape(class_count, class_count)


def front_metrics(confusion) -> dict:
    matrix = confusion.detach().cpu().numpy().astype(np.float64)
    true_positive = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    observed = matrix.sum(axis=1)
    union = predicted + observed - true_positive
    iou = np.divide(
        true_positive, union, out=np.full_like(union, np.nan), where=union > 0
    )
    dice = np.divide(
        2 * true_positive, predicted + observed,
        out=np.full_like(union, np.nan), where=(predicted + observed) > 0,
    )
    return {
        "meanFrontalIoU": float(np.nanmean(iou[1:])),
        "meanFrontalDice": float(np.nanmean(dice[1:])),
        "perClassIoU": [None if np.isnan(value) else float(value) for value in iou],
        "perClassDice": [None if np.isnan(value) else float(value) for value in dice],
    }


def atomic_torch_checkpoint(path, *, state_dict, metadata):
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata["metadataHash"] = schema_hash(metadata)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        torch.save({"state_dict": state_dict, "metadata": metadata}, partial)
        os.replace(partial, target)
    finally:
        if partial.exists():
            partial.unlink()


def load_checkpoint(path, *, require_accepted=True, map_location="cpu"):
    import torch

    payload = torch.load(path, map_location=map_location, weights_only=True)
    metadata = dict(payload.get("metadata") or {})
    stored_hash = metadata.pop("metadataHash", None)
    if stored_hash != schema_hash(metadata):
        raise ValueError("metadati checkpoint corrotti o incompatibili")
    if metadata.get("formatVersion") != 1:
        raise ValueError("versione checkpoint non supportata")
    if require_accepted and metadata.get("accepted") is not True:
        raise ValueError("checkpoint non promosso per uso operativo")
    metadata["metadataHash"] = stored_hash
    return payload["state_dict"], metadata


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
