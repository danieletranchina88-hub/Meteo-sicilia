#!/usr/bin/env python3
"""Train and calibrate the binary supervised front recogniser."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, precision_score,
    recall_score, f1_score, jaccard_score, roc_auc_score, brier_score_loss,
)
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meteo_analysis.ml.features import (
    ERA5_TRANSFER_FEATURE_COLUMNS,
    ERA5_TRANSFER_NO_COORDINATES_FEATURE_COLUMNS,
    feature_schema_hash,
)


def temporal_splits(frame):
    times = np.sort(frame.time.drop_duplicates().to_numpy())
    if len(times) < 30:
        raise ValueError("servono almeno 30 analisi distinte")
    train_end = times[int(len(times) * 0.70) - 1]
    valid_end = times[int(len(times) * 0.85) - 1]
    return (
        frame[frame.time <= train_end],
        frame[(frame.time > train_end) & (frame.time <= valid_end)],
        frame[frame.time > valid_end],
    )


def metrics(y, probability, threshold):
    prediction = probability >= threshold
    return {
        "rocAuc": float(roc_auc_score(y, probability)),
        "averagePrecision": float(average_precision_score(y, probability)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "iou": float(jaccard_score(y, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
    }


def position_metrics(distance_km, probability, threshold):
    """Held-out localisation checks that complement cell-wise scores."""
    distance = np.asarray(distance_km, dtype=float)
    probability = np.asarray(probability, dtype=float)
    prediction = probability >= threshold
    core = np.isfinite(distance) & (distance <= 20.0)
    far = (~np.isfinite(distance)) | (distance >= 80.0)
    bins = {}
    for name, mask in (
        ("0-20", core),
        ("20-40", (distance > 20.0) & (distance <= 40.0)),
        ("40-80", (distance > 40.0) & (distance < 80.0)),
        ("80+", far),
    ):
        bins[name] = float(np.mean(probability[mask])) if np.any(mask) else None
    return {
        "coreRecall20Km": float(np.mean(prediction[core])) if np.any(core) else 0.0,
        "farPositiveRate80Km": float(np.mean(prediction[far])) if np.any(far) else 1.0,
        "meanProbabilityByDistanceKm": bins,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--model", default="models/front_model.json")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--feature-profile",
        choices=("era5-transfer", "era5-transfer-no-coordinates"),
        default="era5-transfer-no-coordinates",
        help="Il profilo senza coordinate riduce scorciatoie geografiche.",
    )
    args = parser.parse_args()
    frame = pd.read_parquet(args.dataset)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    features = list(
        ERA5_TRANSFER_NO_COORDINATES_FEATURE_COLUMNS
        if args.feature_profile == "era5-transfer-no-coordinates"
        else ERA5_TRANSFER_FEATURE_COLUMNS
    )
    required = features + ["time", "y", "labelDistanceKm", "labelWeight"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"colonne mancanti: {missing}")

    train, valid, test = temporal_splits(frame)
    # Deterministic row cap preserves the temporal split and class ratio.
    if args.max_rows:
        train = train.sample(
            min(len(train), args.max_rows), random_state=args.seed
        )
    positives = max(1, int(train.y.sum()))
    scale = min(40.0, (len(train) - positives) / positives)
    classifier = xgb.XGBClassifier(
        n_estimators=1400,
        max_depth=7,
        learning_rate=0.035,
        min_child_weight=12,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_alpha=0.15,
        reg_lambda=2.0,
        scale_pos_weight=scale,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=args.seed,
        early_stopping_rounds=70,
    )
    classifier.fit(
        train[features], train.y,
        sample_weight=train.labelWeight,
        eval_set=[(valid[features], valid.y)],
        sample_weight_eval_set=[valid.labelWeight],
        verbose=25,
    )

    raw_valid = classifier.predict_proba(valid[features])[:, 1]
    raw_test = classifier.predict_proba(test[features])[:, 1]
    # Platt calibration corrects probability inflation from scale_pos_weight.
    logits = np.log(np.clip(raw_valid, 1e-7, 1 - 1e-7) /
                    np.clip(1 - raw_valid, 1e-7, 1))
    calibrator = LogisticRegression(C=1e3).fit(
        logits[:, None], valid.y, sample_weight=valid.labelWeight
    )

    def calibrate(raw):
        value = np.log(np.clip(raw, 1e-7, 1 - 1e-7) /
                       np.clip(1 - raw, 1e-7, 1))
        return calibrator.predict_proba(value[:, None])[:, 1]

    calibrated_valid = calibrate(raw_valid)
    calibrated_test = calibrate(raw_test)
    precision, recall, thresholds = precision_recall_curve(
        valid.y, calibrated_valid
    )
    # A balanced F1 threshold; publication still requires physical confirmation.
    score = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    threshold = float(thresholds[int(np.nanargmax(score))])

    test_metrics = metrics(test.y, calibrated_test, threshold)
    localisation = position_metrics(
        test.labelDistanceKm, calibrated_test, threshold
    )
    acceptance = {
        "minimumRocAuc": 0.75,
        "minimumAveragePrecision": 0.07,
        "minimumF1": 0.12,
        "maximumBrier": 0.08,
        "minimumCoreRecall20Km": 0.25,
        "maximumFarPositiveRate80Km": 0.05,
    }
    accepted = bool(
        test_metrics["rocAuc"] >= acceptance["minimumRocAuc"]
        and test_metrics["averagePrecision"]
        >= acceptance["minimumAveragePrecision"]
        and test_metrics["f1"] >= acceptance["minimumF1"]
        and test_metrics["brier"] <= acceptance["maximumBrier"]
        and localisation["coreRecall20Km"]
        >= acceptance["minimumCoreRecall20Km"]
        and localisation["farPositiveRate80Km"]
        <= acceptance["maximumFarPositiveRate80Km"]
    )
    if not accepted:
        raise RuntimeError(
            "modello rifiutato dai criteri fuori campione: "
            + json.dumps({"cell": test_metrics, "position": localisation})
        )
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "formatVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": "XGBoost binary front proximity classifier",
        "label": "DWD manual front within 40 km",
        "trainingGroundTruth": {
            "name": "Front polylines extracted from DWD Maps",
            "doi": "10.5281/zenodo.5785816",
            "license": "CC BY 4.0",
        },
        "trainingPredictors": "ERA5",
        "operationalPredictors": "ICON-2I",
        "featureProfile": args.feature_profile,
        "features": features,
        "featureSchemaHash": feature_schema_hash(features),
        "threshold": threshold,
        "calibration": {
            "method": "Platt-logit",
            "a": float(calibrator.coef_[0, 0]),
            "b": float(calibrator.intercept_[0]),
        },
        "split": {
            "method": "chronological 70/15/15",
            "trainEnd": str(train.time.max()),
            "validationEnd": str(valid.time.max()),
            "testStart": str(test.time.min()),
        },
        "classBalance": {
            "analyses": int(frame.time.nunique()),
            "datasetRows": int(len(frame)),
            "trainingRows": int(len(train)),
            "positiveFraction": float(train.y.mean()),
            "scalePosWeight": float(scale),
        },
        "bestIteration": int(classifier.best_iteration),
        "validationMetrics": metrics(valid.y, calibrated_valid, threshold),
        "testMetrics": test_metrics,
        "testPositionMetrics": localisation,
        "labelUncertainty": {
            "method": "distance-to-40-km-boundary sample weighting",
            "minimumWeight": 0.35,
            "fullWeightCoreKm": 20.0,
            "fullWeightFarKm": 80.0,
        },
        "acceptanceCriteria": acceptance,
        "accepted": accepted,
        "featureImportanceGain": dict(sorted(
            classifier.get_booster().get_score(importance_type="gain").items(),
            key=lambda item: item[1], reverse=True,
        )),
    }
    # Save only after the held-out acceptance gate: a failed retraining cannot
    # overwrite the last known-good operational model.
    temporary_model = model_path.with_name(model_path.stem + ".part.json")
    classifier.get_booster().save_model(temporary_model)
    os.replace(temporary_model, model_path)
    metadata_path = model_path.with_suffix(".metadata.json")
    temporary_metadata = metadata_path.with_name(
        metadata_path.stem + ".part.json"
    )
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata, metadata_path)
    print(json.dumps(metadata["testMetrics"], indent=2))
    print(f"Modello: {model_path}; soglia calibrata: {threshold:.4f}")


if __name__ == "__main__":
    main()
