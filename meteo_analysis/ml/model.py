"""Versioned XGBoost model bundle with strict feature-schema validation."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import gzip
import json
from pathlib import Path

import numpy as np

from .features import feature_schema_hash


@dataclass
class FrontModel:
    booster: object
    metadata: dict

    @property
    def features(self):
        return list(self.metadata["features"])

    @property
    def threshold(self):
        return float(self.metadata["threshold"])

    @classmethod
    def load(cls, model_path, metadata_path=None):
        import xgboost as xgb

        model_path = Path(model_path)
        if model_path.name.endswith(".json.gz.b64"):
            default_metadata = model_path.with_name(
                model_path.name.removesuffix(".json.gz.b64") + ".metadata.json"
            )
        else:
            default_metadata = model_path.with_suffix(".metadata.json")
        metadata_path = Path(metadata_path or default_metadata)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("formatVersion") != 1 or not metadata.get("accepted"):
            raise ValueError("modello ML non validato per uso operativo")
        expected = feature_schema_hash(metadata["features"])
        if metadata.get("featureSchemaHash") != expected:
            raise ValueError("metadati ML corrotti: hash delle feature non valido")
        booster = xgb.Booster()
        if model_path.name.endswith(".json.gz.b64"):
            encoded = model_path.read_text(encoding="ascii")
            payload = gzip.decompress(base64.b64decode(encoded, validate=True))
            booster.load_model(bytearray(payload))
        else:
            booster.load_model(model_path)
        if booster.feature_names and booster.feature_names != metadata["features"]:
            raise ValueError("ordine delle feature diverso da quello addestrato")
        return cls(booster=booster, metadata=metadata)

    def predict(self, frame):
        import xgboost as xgb

        matrix = frame.reindex(columns=self.features)
        probability = self.booster.predict(
            xgb.DMatrix(matrix, feature_names=self.features)
        )
        probability = np.asarray(probability, float)
        if self.metadata.get("calibration"):
            a = float(self.metadata["calibration"]["a"])
            b = float(self.metadata["calibration"]["b"])
            raw = np.clip(probability, 1e-7, 1 - 1e-7)
            probability = 1.0 / (
                1.0 + np.exp(-(a * np.log(raw / (1.0 - raw)) + b))
            )
        return np.clip(probability, 0.0, 1.0)
