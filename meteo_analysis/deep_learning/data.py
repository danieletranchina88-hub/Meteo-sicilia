"""Safe manifest datasets and training-only normalisation statistics."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

MANIFEST_VERSION = 1
TASK_KEYS = {
    "front-segmentation": ("inputs", "target"),
    "orographic-downscaling": ("coarse", "target"),
}


def _utc_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("validTime deve includere il fuso orario UTC")
    return parsed


def load_manifest(path, *, expected_task: str | None = None) -> dict:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != MANIFEST_VERSION:
        raise ValueError("versione manifest deep-learning non supportata")
    task = payload.get("task")
    if task not in TASK_KEYS or (expected_task and task != expected_task):
        raise ValueError(f"task del manifest non valido: {task!r}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest privo di campioni")

    by_split: dict[str, set[str]] = {}
    seen_identity = set()
    root = path.parent
    for sample in samples:
        if not isinstance(sample, dict):
            raise TypeError("campione manifest non rappresentato da un oggetto")
        split = str(sample.get("split") or "")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"split non valido: {split!r}")
        valid_time = str(sample.get("validTime") or "")
        _utc_timestamp(valid_time)
        # All patches from one atmosphere/time must stay in the same split.
        by_split.setdefault(split, set()).add(valid_time)
        identity = (valid_time, str(sample.get("path") or ""))
        if identity in seen_identity:
            raise ValueError(f"campione duplicato: {identity}")
        seen_identity.add(identity)

        sample_path = (root / str(sample.get("path") or "")).resolve()
        try:
            sample_path.relative_to(root)
        except ValueError as error:
            raise ValueError("path campione esterno alla directory manifest") from error
        sample["_resolvedPath"] = str(sample_path)

    shared_static = payload.get("sharedStatic")
    if shared_static is not None:
        if not isinstance(shared_static, dict) or not shared_static.get("path"):
            raise TypeError("sharedStatic deve dichiarare un path")
        shared_path = (root / str(shared_static["path"])).resolve()
        try:
            shared_path.relative_to(root)
        except ValueError as error:
            raise ValueError("path statico esterno alla directory manifest") from error
        payload["_resolvedSharedStatic"] = str(shared_path)

    split_names = sorted(by_split)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1:]:
            overlap = by_split[first] & by_split[second]
            if overlap:
                raise ValueError(
                    "data leakage: validTime presenti in split diversi: "
                    + ", ".join(sorted(overlap)[:3])
                )
    if str(payload.get("splitPolicy") or "").startswith("chronological"):
        missing_splits = {"train", "validation", "test"} - set(by_split)
        if missing_splits:
            raise ValueError(f"split cronologici mancanti: {sorted(missing_splits)}")
        ordered = {
            name: sorted(_utc_timestamp(value) for value in values)
            for name, values in by_split.items()
        }
        if not (
            ordered["train"][-1] < ordered["validation"][0]
            and ordered["validation"][-1] < ordered["test"][0]
        ):
            raise ValueError("split dichiarati cronologici ma temporalmente mescolati")
    return payload


class TensorManifestDataset:
    """Lazy NPZ reader compatible with ``torch.utils.data.DataLoader``.

    NPZ is opened with ``allow_pickle=False``.  This is both safer and makes
    every sample a transparent collection of numeric arrays.
    """

    def __init__(self, manifest, *, task: str, split: str):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = load_manifest(self.manifest_path, expected_task=task)
        self.task = task
        self.samples = [
            sample for sample in self.manifest["samples"]
            if sample["split"] == split
        ]
        if not self.samples:
            raise ValueError(f"nessun campione nello split {split!r}")
        self.shared_static = None
        shared_path = self.manifest.get("_resolvedSharedStatic")
        if shared_path:
            with np.load(shared_path, allow_pickle=False) as archive:
                missing = {"static", "cell_area_m2"} - set(archive.files)
                if missing:
                    raise ValueError(
                        f"statico condiviso privo di array {sorted(missing)}"
                    )
                static = np.asarray(archive["static"], dtype=np.float32)
                cell_area = np.asarray(archive["cell_area_m2"], dtype=np.float32)
            if static.ndim != 3 or cell_area.shape != static.shape[1:]:
                raise ValueError("forme dello statico condiviso incoerenti")
            expected = len(self.manifest.get("staticChannels") or ())
            if static.shape[0] != expected:
                raise ValueError("canali dello statico condiviso incoerenti")
            if not (
                np.all(np.isfinite(static))
                and np.all(np.isfinite(cell_area) & (cell_area > 0))
            ):
                raise ValueError("statico condiviso non finito o area non positiva")
            self.shared_static = (static, cell_area)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch

        record = self.samples[index]
        with np.load(record["_resolvedPath"], allow_pickle=False) as archive:
            missing = set(TASK_KEYS[self.task]) - set(archive.files)
            if missing:
                raise ValueError(
                    f"{record['path']}: array mancanti {sorted(missing)}"
                )
            if self.task == "front-segmentation":
                inputs = np.asarray(archive["inputs"], dtype=np.float32)
                target = np.asarray(archive["target"], dtype=np.int64)
                if inputs.ndim != 3 or target.shape != inputs.shape[1:]:
                    raise ValueError(f"{record['path']}: forme frontali incoerenti")
                valid = (
                    np.asarray(archive["valid_mask"], dtype=bool)
                    if "valid_mask" in archive else np.ones_like(target, bool)
                )
                weights = (
                    np.asarray(archive["label_weight"], dtype=np.float32)
                    if "label_weight" in archive else np.ones_like(target, np.float32)
                )
                physics = (
                    np.asarray(archive["physics_support"], dtype=np.float32)
                    if "physics_support" in archive else np.ones_like(target, np.float32)
                )
                if not (valid.shape == weights.shape == physics.shape == target.shape):
                    raise ValueError(f"{record['path']}: maschere frontali incoerenti")
                if not (
                    np.all(np.isfinite(weights)) and np.all(weights >= 0)
                    and np.all(np.isfinite(physics))
                    and np.min(physics) >= 0 and np.max(physics) <= 1
                ):
                    raise ValueError(f"{record['path']}: pesi/supporto non validi")
                class_count = len(self.manifest.get("classes") or ())
                labelled = target[valid]
                if (
                    class_count < 2 or labelled.size == 0
                    or np.min(labelled) < 0 or np.max(labelled) >= class_count
                ):
                    raise ValueError(f"{record['path']}: classi target non valide")
                target = np.where(valid, target, -100)
                return {
                    "inputs": torch.from_numpy(inputs),
                    "target": torch.from_numpy(target),
                    "label_weight": torch.from_numpy(weights),
                    "physics_support": torch.from_numpy(physics),
                }

            coarse = np.asarray(archive["coarse"], dtype=np.float32)
            target = np.asarray(archive["target"], dtype=np.float32)
            has_local_static = {"static", "cell_area_m2"} <= set(archive.files)
            if has_local_static:
                static = np.asarray(archive["static"], dtype=np.float32)
                cell_area = np.asarray(archive["cell_area_m2"], dtype=np.float32)
            elif self.shared_static is not None:
                window = record.get("fineWindow")
                if not (
                    isinstance(window, list) and len(window) == 4
                    and all(isinstance(value, int) for value in window)
                ):
                    raise ValueError(
                        f"{record['path']}: fineWindow mancante per statico condiviso"
                    )
                y0, y1, x0, x1 = window
                shared, shared_area = self.shared_static
                if not (0 <= y0 < y1 <= shared.shape[1] and 0 <= x0 < x1 <= shared.shape[2]):
                    raise ValueError(f"{record['path']}: fineWindow fuori dominio")
                static = shared[:, y0:y1, x0:x1].copy()
                cell_area = shared_area[y0:y1, x0:x1].copy()
            else:
                raise ValueError(
                    f"{record['path']}: static/cell_area_m2 assenti e nessuno statico condiviso"
                )
            if coarse.ndim != 3 or static.ndim != 3 or target.ndim != 3:
                raise ValueError(f"{record['path']}: tensori downscaling non 3-D")
            if not np.all(np.isfinite(coarse)) or not np.all(np.isfinite(static)):
                raise ValueError(f"{record['path']}: coarse/static contengono NaN")
            if static.shape[1:] != target.shape[1:]:
                raise ValueError(f"{record['path']}: statico e target non allineati")
            if cell_area.shape != target.shape[1:] or not np.all(
                np.isfinite(cell_area) & (cell_area > 0)
            ):
                raise ValueError(f"{record['path']}: area celle non valida")
            scale_y = target.shape[1] / coarse.shape[1]
            scale_x = target.shape[2] / coarse.shape[2]
            if scale_y != scale_x or int(scale_y) != scale_y:
                raise ValueError(f"{record['path']}: fattore di scala non intero")
            valid = (
                np.asarray(archive["valid_mask"], dtype=bool)
                if "valid_mask" in archive else np.isfinite(target)
            )
            if valid.ndim == 2:
                valid = np.broadcast_to(valid, target.shape).copy()
            if valid.shape != target.shape:
                raise ValueError(f"{record['path']}: valid_mask non allineata")
            output_names = tuple(self.manifest.get("outputChannels") or ())
            if len(output_names) != target.shape[0]:
                raise ValueError(f"{record['path']}: schema output non coerente")
            for channel, name in enumerate(output_names):
                values = target[channel][valid[channel]]
                if not values.size:
                    raise ValueError(f"{record['path']}: target {name} privo di dati")
                if name == "temperature_2m_k" and not (
                    np.min(values) >= 180.0 and np.max(values) <= 340.0
                ):
                    raise ValueError(f"{record['path']}: temperatura target fuori scala")
                if name == "precipitation_rate_mm_h" and not (
                    np.min(values) >= 0.0 and np.max(values) <= 1000.0
                ):
                    raise ValueError(f"{record['path']}: precipitazione target fuori scala")
                if name in {"wind_u10_m_s", "wind_v10_m_s"} and np.max(
                    np.abs(values)
                ) > 200.0:
                    raise ValueError(f"{record['path']}: vento target fuori scala")
            return {
                "coarse": torch.from_numpy(coarse),
                "static": torch.from_numpy(static),
                "target": torch.from_numpy(target),
                "valid_mask": torch.from_numpy(valid),
                "cell_area_m2": torch.from_numpy(cell_area),
            }


@dataclass(frozen=True)
class ChannelStatistics:
    names: tuple[str, ...]
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]

    def as_dict(self):
        return {
            "names": list(self.names),
            "mean": list(self.mean),
            "standardDeviation": list(self.standard_deviation),
        }


def estimate_channel_statistics(
    dataset: TensorManifestDataset,
    key: str,
    names: Iterable[str],
    *,
    maximum_samples: int | None = None,
) -> ChannelStatistics:
    """Compute statistics only from the caller-provided training dataset."""
    names = tuple(names)
    sums = np.zeros(len(names), dtype=np.float64)
    squares = np.zeros(len(names), dtype=np.float64)
    counts = np.zeros(len(names), dtype=np.int64)
    limit = len(dataset) if maximum_samples is None else min(len(dataset), maximum_samples)
    for index in range(limit):
        values = dataset[index][key].detach().cpu().numpy().astype(np.float64)
        if values.shape[0] != len(names):
            raise ValueError(f"{key}: {values.shape[0]} canali, attesi {len(names)}")
        flat = values.reshape(values.shape[0], -1)
        finite = np.isfinite(flat)
        sums += np.where(finite, flat, 0.0).sum(axis=1)
        squares += np.where(finite, flat * flat, 0.0).sum(axis=1)
        counts += finite.sum(axis=1)
    if np.any(counts == 0):
        absent = [names[i] for i in np.flatnonzero(counts == 0)]
        raise ValueError(f"canali interamente mancanti: {absent}")
    mean = sums / counts
    variance = np.maximum(squares / counts - mean * mean, 1.0e-12)
    return ChannelStatistics(
        names=names,
        mean=tuple(map(float, mean)),
        standard_deviation=tuple(map(float, np.sqrt(variance))),
    )
