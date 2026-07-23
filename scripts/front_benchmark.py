"""Evaluate front GeoJSON against independent human-labelled front lines.

The benchmark matches complete polylines, not isolated vertices.  A prediction
and a label are considered the same front only when each line substantially
supports the other inside a configurable spatial corridor.  One-to-one
assignment prevents a fragmented prediction from claiming the same label
multiple times.

This module measures the frozen algorithm; it does not tune thresholds.
Labels must be independently drawn and temporally separated into
train/validation/test cases by the manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

import front_detection as fd

DEFAULT_RADIUS_KM = 100.0
DEFAULT_MIN_OVERLAP = 0.60
MIN_PROBABILITY_CASES = 200
MIN_PROBABILITY_PREDICTIONS = 500
SCORE_BINS = np.linspace(0.0, 1.0, 6)


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _round_metric(value: float | None, digits: int = 4) -> float | None:
    return None if value is None or not math.isfinite(value) else round(value, digits)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson interval for a binomial proportion."""
    if total <= 0:
        return None
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (proportion + z2 / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


def _front_type(properties: dict[str, Any]) -> str | None:
    value = properties.get("frontType", properties.get("type"))
    if value is None:
        return None
    normalised = str(value).strip().lower()
    aliases = {
        "freddo": "cold",
        "caldo": "warm",
        "occluso": "occluded",
        "stazionario": "stationary",
    }
    return aliases.get(normalised, normalised)


def _score(properties: dict[str, Any]) -> float | None:
    value = properties.get("qualityScore", properties.get("confidence"))
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, score)) if math.isfinite(score) else None


def load_fronts(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load valid LineString/MultiLineString features from a GeoJSON file."""
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    features = document.get("features", []) if isinstance(document, dict) else []
    fronts = []
    for feature_index, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        geometry_type = geometry.get("type")
        raw_lines = (
            [geometry.get("coordinates")]
            if geometry_type == "LineString"
            else geometry.get("coordinates", [])
            if geometry_type == "MultiLineString"
            else []
        )
        for part_index, raw_line in enumerate(raw_lines):
            try:
                coordinates = np.asarray(raw_line, dtype=float)
            except (TypeError, ValueError):
                continue
            if (
                coordinates.ndim != 2
                or coordinates.shape[0] < 2
                or coordinates.shape[1] < 2
                or not np.all(np.isfinite(coordinates[:, :2]))
            ):
                continue
            fronts.append(
                {
                    "coordinates": coordinates[:, :2],
                    "frontType": _front_type(properties),
                    "score": _score(properties),
                    "featureIndex": feature_index,
                    "partIndex": part_index,
                }
            )
    return fronts


def line_match_score(
    prediction: np.ndarray,
    label: np.ndarray,
    radius_km: float = DEFAULT_RADIUS_KM,
) -> float:
    """Bidirectional line support; the shorter/easier direction cannot dominate."""
    prediction_support = fd.line_support_fraction(
        prediction, [label], radius_km
    )
    label_support = fd.line_support_fraction(label, [prediction], radius_km)
    return min(prediction_support, label_support)


def match_fronts(
    predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    radius_km: float = DEFAULT_RADIUS_KM,
    minimum_overlap: float = DEFAULT_MIN_OVERLAP,
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    """Return accepted one-to-one matches and matched indexes."""
    if not predictions or not labels:
        return [], set(), set()

    scores = np.zeros((len(predictions), len(labels)), dtype=float)
    for prediction_index, prediction in enumerate(predictions):
        for label_index, label in enumerate(labels):
            scores[prediction_index, label_index] = line_match_score(
                prediction["coordinates"], label["coordinates"], radius_km
            )

    prediction_indexes, label_indexes = linear_sum_assignment(-scores)
    matches = []
    matched_predictions: set[int] = set()
    matched_labels: set[int] = set()
    for prediction_index, label_index in zip(prediction_indexes, label_indexes):
        overlap = float(scores[prediction_index, label_index])
        if overlap < minimum_overlap:
            continue
        prediction = predictions[int(prediction_index)]
        label = labels[int(label_index)]
        predicted_type = prediction["frontType"]
        labelled_type = label["frontType"]
        type_correct = (
            predicted_type == labelled_type
            if predicted_type is not None and labelled_type is not None
            else None
        )
        matches.append(
            {
                "predictionIndex": int(prediction_index),
                "labelIndex": int(label_index),
                "overlap": round(overlap, 4),
                "predictedType": predicted_type,
                "labelledType": labelled_type,
                "typeCorrect": type_correct,
            }
        )
        matched_predictions.add(int(prediction_index))
        matched_labels.add(int(label_index))
    return matches, matched_predictions, matched_labels


def evaluate_case(
    predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    radius_km: float = DEFAULT_RADIUS_KM,
    minimum_overlap: float = DEFAULT_MIN_OVERLAP,
) -> dict[str, Any]:
    matches, matched_predictions, matched_labels = match_fronts(
        predictions, labels, radius_km, minimum_overlap
    )
    true_positives = len(matches)
    false_positives = len(predictions) - len(matched_predictions)
    false_negatives = len(labels) - len(matched_labels)
    typed = [match for match in matches if match["typeCorrect"] is not None]
    correct_types = sum(match["typeCorrect"] is True for match in typed)
    return {
        "predictionCount": len(predictions),
        "labelCount": len(labels),
        "truePositives": true_positives,
        "falsePositives": false_positives,
        "falseNegatives": false_negatives,
        "frontCountError": len(predictions) - len(labels),
        "typedMatches": len(typed),
        "correctTypes": correct_types,
        "matches": matches,
        "predictionOutcomes": [
            {
                "matched": index in matched_predictions,
                "score": prediction["score"],
            }
            for index, prediction in enumerate(predictions)
        ],
    }


def _score_bins(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index in range(len(SCORE_BINS) - 1):
        lower = float(SCORE_BINS[index])
        upper = float(SCORE_BINS[index + 1])
        members = [
            outcome
            for outcome in outcomes
            if outcome["score"] is not None
            and lower <= outcome["score"] <= upper
            and (index == len(SCORE_BINS) - 2 or outcome["score"] < upper)
        ]
        if not members:
            continue
        matched = sum(member["matched"] for member in members)
        result.append(
            {
                "range": [round(lower, 2), round(upper, 2)],
                "count": len(members),
                "meanQualityScore": round(
                    float(np.mean([member["score"] for member in members])), 4
                ),
                "observedPrecision": round(matched / len(members), 4),
                "observedPrecision95CI": _wilson_interval(matched, len(members)),
            }
        )
    return result


def evaluate_manifest(
    manifest_path: str | os.PathLike[str],
    split: str | None = "test",
    radius_km: float = DEFAULT_RADIUS_KM,
    minimum_overlap: float = DEFAULT_MIN_OVERLAP,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    root = manifest_path.parent
    selected = [
        case
        for case in manifest.get("cases", [])
        if split is None or case.get("split") == split
    ]
    case_reports = []
    warnings = []
    for case in selected:
        case_id = str(case.get("id", f"case-{len(case_reports) + 1}"))
        if not case.get("validTime"):
            warnings.append(f"{case_id}: validTime mancante")
        if not case.get("labelSource"):
            warnings.append(f"{case_id}: labelSource mancante")
        prediction_path = (root / case["prediction"]).resolve()
        label_path = (root / case["label"]).resolve()
        predictions = load_fronts(prediction_path)
        labels = load_fronts(label_path)
        metrics = evaluate_case(
            predictions, labels, radius_km, minimum_overlap
        )
        case_reports.append(
            {
                "id": case_id,
                "validTime": case.get("validTime"),
                "labelSource": case.get("labelSource"),
                **metrics,
            }
        )

    totals = {
        key: sum(case[key] for case in case_reports)
        for key in (
            "predictionCount",
            "labelCount",
            "truePositives",
            "falsePositives",
            "falseNegatives",
            "frontCountError",
            "typedMatches",
            "correctTypes",
        )
    }
    precision = _safe_div(
        totals["truePositives"],
        totals["truePositives"] + totals["falsePositives"],
    )
    recall = _safe_div(
        totals["truePositives"],
        totals["truePositives"] + totals["falseNegatives"],
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    type_accuracy = _safe_div(totals["correctTypes"], totals["typedMatches"])
    outcomes = [
        outcome
        for case in case_reports
        for outcome in case.pop("predictionOutcomes")
    ]
    metadata = manifest.get("metadata") or {}
    probability_eligible = (
        metadata.get("scoreCalibratorFrozen") is True
        and split == "test"
        and len(case_reports) >= MIN_PROBABILITY_CASES
        and totals["predictionCount"] >= MIN_PROBABILITY_PREDICTIONS
    )
    return {
        "schemaVersion": 1,
        "manifest": str(manifest_path),
        "split": split if split is not None else "all",
        "spatialToleranceKm": radius_km,
        "minimumBidirectionalOverlap": minimum_overlap,
        "caseCount": len(case_reports),
        "counts": totals,
        "metrics": {
            "precision": _round_metric(precision),
            "precision95CI": _wilson_interval(
                totals["truePositives"],
                totals["truePositives"] + totals["falsePositives"],
            ),
            "recall": _round_metric(recall),
            "recall95CI": _wilson_interval(
                totals["truePositives"],
                totals["truePositives"] + totals["falseNegatives"],
            ),
            "f1": _round_metric(f1),
            "typeAccuracy": _round_metric(type_accuracy),
            "meanAbsoluteFrontCountError": _round_metric(
                float(
                    np.mean(
                        [abs(case["frontCountError"]) for case in case_reports]
                    )
                )
                if case_reports
                else None
            ),
        },
        "qualityScoreDiagnostic": {
            "interpretation": (
                "Empirical match rate by heuristic quality score; not a probability."
            ),
            "bins": _score_bins(outcomes),
            "eligibleForProbabilityInterpretation": probability_eligible,
            "eligibilityRule": (
                f"frozen calibrator + test split + >= {MIN_PROBABILITY_CASES} "
                f"cases + >= {MIN_PROBABILITY_PREDICTIONS} predictions"
            ),
        },
        "warnings": warnings,
        "cases": case_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="manifest JSON dei casi etichettati")
    parser.add_argument("--split", default="test", help="train, validation, test o all")
    parser.add_argument("--radius-km", type=float, default=DEFAULT_RADIUS_KM)
    parser.add_argument("--min-overlap", type=float, default=DEFAULT_MIN_OVERLAP)
    parser.add_argument("--out", help="scrive il report JSON; altrimenti stdout")
    args = parser.parse_args()
    split = None if args.split == "all" else args.split
    report = evaluate_manifest(
        args.manifest, split, args.radius_km, args.min_overlap
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
