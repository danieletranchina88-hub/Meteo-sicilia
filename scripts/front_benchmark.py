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
import re
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


EARTH_KM_PER_DEG = 111.32


def _polyline_length_km(line: np.ndarray) -> float:
    line = np.asarray(line, dtype=float)
    if len(line) < 2:
        return 0.0
    mean_lat = np.deg2rad(float(np.mean(line[:, 1])))
    dx = np.diff(line[:, 0]) * EARTH_KM_PER_DEG * np.cos(mean_lat)
    dy = np.diff(line[:, 1]) * EARTH_KM_PER_DEG
    return float(np.sum(np.hypot(dx, dy)))


def _resample_km(line: np.ndarray, step_km: float = 15.0) -> np.ndarray:
    """Uniform arc-length resampling so per-vertex stats are density-fair."""
    line = np.asarray(line, dtype=float)
    if len(line) < 2:
        return line
    mean_lat = np.deg2rad(float(np.mean(line[:, 1])))
    seg = np.hypot(np.diff(line[:, 0]) * EARTH_KM_PER_DEG * np.cos(mean_lat),
                   np.diff(line[:, 1]) * EARTH_KM_PER_DEG)
    cumulative = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cumulative[-1])
    if total <= 0.0:
        return line
    targets = np.arange(0.0, total + step_km, step_km)
    return np.column_stack((np.interp(targets, cumulative, line[:, 0]),
                            np.interp(targets, cumulative, line[:, 1])))


def _min_distances_km(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Nearest distance (km) from each of ``points`` (as-is) to ``line``."""
    a = np.asarray(points, dtype=float)
    b = _resample_km(line)
    if len(a) == 0 or len(b) == 0:
        return np.array([])
    mean_lat = np.deg2rad(float(np.mean(np.concatenate([a[:, 1], b[:, 1]]))))
    ax = a[:, 0] * EARTH_KM_PER_DEG * np.cos(mean_lat)
    ay = a[:, 1] * EARTH_KM_PER_DEG
    bx = b[:, 0] * EARTH_KM_PER_DEG * np.cos(mean_lat)
    by = b[:, 1] * EARTH_KM_PER_DEG
    d = np.hypot(ax[:, None] - bx[None, :], ay[:, None] - by[None, :])
    return np.min(d, axis=1)


def _symmetric_mean_distance_km(a: np.ndarray, b: np.ndarray) -> float:
    """Mean symmetric nearest-neighbour distance between two lines (km)."""
    da = _min_distances_km(_resample_km(a), b)
    db = _min_distances_km(_resample_km(b), a)
    both = np.concatenate([da, db])
    return float(np.mean(both)) if len(both) else float("nan")


def _covered_length_km(line: np.ndarray, others: list[np.ndarray],
                       radius_km: float) -> float:
    """Length of ``line`` whose points lie within ``radius_km`` of any other line."""
    line = np.asarray(line, dtype=float)
    if len(line) < 2 or not others:
        return 0.0
    dense = _resample_km(line, step_km=5.0)
    covered = np.zeros(len(dense), dtype=bool)
    for other in others:
        d = _min_distances_km(dense, other)
        if len(d) == len(dense):
            covered |= d <= radius_km
    return _polyline_length_km(line) * float(np.mean(covered))


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

    # Maximise the number of eligible matches first, then their overlap.
    # Maximising raw overlap alone can choose one excellent pair plus an
    # ineligible pair instead of two valid, moderately overlapping pairs.
    cardinality_bonus = min(scores.shape) + 1.0
    utility = scores + (scores >= minimum_overlap) * cardinality_bonus
    prediction_indexes, label_indexes = linear_sum_assignment(-utility)
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

    # Symmetric nearest-neighbour distance over the accepted matches.
    match_distances = []
    for match in matches:
        prediction = predictions[match["predictionIndex"]]["coordinates"]
        label = labels[match["labelIndex"]]["coordinates"]
        distance = _symmetric_mean_distance_km(prediction, label)
        if np.isfinite(distance):
            match_distances.append(distance)

    # Length-based accounting: a front matched only over a short stretch is
    # not fully detected. detected/missed reference length and false predicted
    # length capture position quality that TP/FP/FN counts miss.
    prediction_lines = [p["coordinates"] for p in predictions]
    label_lines = [l["coordinates"] for l in labels]
    reference_length = sum(_polyline_length_km(l) for l in label_lines)
    predicted_length = sum(_polyline_length_km(p) for p in prediction_lines)
    detected_length = sum(
        _covered_length_km(l, prediction_lines, radius_km) for l in label_lines
    )
    false_length = sum(
        _polyline_length_km(p) - _covered_length_km(p, label_lines, radius_km)
        for p in prediction_lines
    )

    # Fragmentation / split-merge: how many predictions fall within radius of
    # each label (>1 = the reference front was split) and vice versa.
    preds_per_label = [
        sum(1 for p in prediction_lines
            if _covered_length_km(l, [p], radius_km) > 0.1 * _polyline_length_km(l))
        for l in label_lines
    ]
    labels_per_pred = [
        sum(1 for l in label_lines
            if _covered_length_km(p, [l], radius_km) > 0.1 * _polyline_length_km(p))
        for p in prediction_lines
    ]
    split_events = sum(1 for n in preds_per_label if n >= 2)
    merge_events = sum(1 for n in labels_per_pred if n >= 2)

    # Type confusion pairs (only over matched, typed pairs).
    type_confusion = [
        [match["labelledType"], match["predictedType"]] for match in typed
    ]
    return {
        "predictionCount": len(predictions),
        "labelCount": len(labels),
        "truePositives": true_positives,
        "falsePositives": false_positives,
        "falseNegatives": false_negatives,
        "frontCountError": len(predictions) - len(labels),
        "typedMatches": len(typed),
        "correctTypes": correct_types,
        "referenceLengthKm": round(reference_length, 1),
        "predictedLengthKm": round(predicted_length, 1),
        "detectedReferenceLengthKm": round(detected_length, 1),
        "missedLengthKm": round(max(0.0, reference_length - detected_length), 1),
        "falsePredictedLengthKm": round(max(0.0, false_length), 1),
        "splitEvents": split_events,
        "mergeEvents": merge_events,
        "matchDistancesKm": match_distances,
        "typeConfusionPairs": type_confusion,
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


def _lead_band(lead_hours) -> str:
    try:
        lead = float(lead_hours)
    except (TypeError, ValueError):
        return "unknown"
    if lead <= 12:
        return "0-12h"
    if lead <= 36:
        return "12-36h"
    return "36-72h"


def _stratified(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Presence metrics grouped by case metadata (season, lead band, surface,
    orography, front type, lifecycle). Only groups that carry the field."""
    dimensions = {
        "season": lambda c: c.get("season"),
        "localHour": lambda c: c.get("localHour"),
        "surface": lambda c: c.get("surface"),
        "orography": lambda c: c.get("orography"),
        "lifecycle": lambda c: c.get("lifecycle"),
        "leadBand": lambda c: _lead_band(c.get("leadHours")),
    }
    out: dict[str, Any] = {}
    for name, getter in dimensions.items():
        groups: dict[str, dict[str, int]] = {}
        for case in case_reports:
            key = getter(case)
            if key is None:
                continue
            bucket = groups.setdefault(
                str(key), {"tp": 0, "fp": 0, "fn": 0, "cases": 0}
            )
            bucket["tp"] += case["truePositives"]
            bucket["fp"] += case["falsePositives"]
            bucket["fn"] += case["falseNegatives"]
            bucket["cases"] += 1
        if not groups:
            continue
        out[name] = {
            key: {
                "pod": _round_metric(_safe_div(g["tp"], g["tp"] + g["fn"])),
                "successRatio": _round_metric(_safe_div(g["tp"], g["tp"] + g["fp"])),
                "csi": _round_metric(_safe_div(g["tp"], g["tp"] + g["fp"] + g["fn"])),
                "cases": g["cases"],
            }
            for key, g in groups.items()
        }
    return out


def label_provenance_warnings(case_id: str, case: dict[str, Any]) -> list[str]:
    """Return audit warnings that prevent a case from being a gold label."""
    warnings = []
    source = case.get("labelSource")
    if not isinstance(source, dict):
        return [f"{case_id}: labelSource mancante"]
    for key in ("provider", "url", "accessedAt", "sha256"):
        if not source.get(key):
            warnings.append(f"{case_id}: labelSource.{key} mancante")
    digest = str(source.get("sha256", ""))
    if digest and not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        warnings.append(f"{case_id}: labelSource.sha256 non valido")

    annotation = case.get("annotation")
    if not isinstance(annotation, dict):
        warnings.append(f"{case_id}: annotation mancante")
    else:
        if not annotation.get("analyst"):
            warnings.append(f"{case_id}: annotation.analyst mancante")
        if annotation.get("blindToPrediction") is not True:
            warnings.append(f"{case_id}: etichetta non dichiarata cieca alla previsione")
        if not annotation.get("digitisedAt"):
            warnings.append(f"{case_id}: annotation.digitisedAt mancante")
    if case.get("split") not in {"train", "validation", "test"}:
        warnings.append(f"{case_id}: split non valido")
    return warnings

def evaluate_manifest(
    manifest_path: str | os.PathLike[str],
    split: str | None = "test",
    radius_km: float = DEFAULT_RADIUS_KM,
    minimum_overlap: float = DEFAULT_MIN_OVERLAP,
    mode: str | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    root = manifest_path.parent
    selected = [
        case
        for case in manifest.get("cases", [])
        if (split is None or case.get("split") == split)
        and (mode is None or case.get("mode", "forecast") == mode)
    ]
    case_reports = []
    warnings = []
    for case in selected:
        case_id = str(case.get("id", f"case-{len(case_reports) + 1}"))
        if not case.get("validTime"):
            warnings.append(f"{case_id}: validTime mancante")
        warnings.extend(label_provenance_warnings(case_id, case))
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
                "mode": case.get("mode", "forecast"),
                "season": case.get("season"),
                "localHour": case.get("localHour"),
                "surface": case.get("surface"),
                "orography": case.get("orography"),
                "lifecycle": case.get("lifecycle"),
                "leadHours": case.get("leadHours"),
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
    length_totals = {
        key: round(sum(case[key] for case in case_reports), 1)
        for key in (
            "referenceLengthKm",
            "predictedLengthKm",
            "detectedReferenceLengthKm",
            "missedLengthKm",
            "falsePredictedLengthKm",
        )
    }
    continuity_totals = {
        key: sum(case[key] for case in case_reports)
        for key in ("splitEvents", "mergeEvents")
    }
    all_distances = [
        distance for case in case_reports for distance in case["matchDistancesKm"]
    ]
    length_recall = _safe_div(
        length_totals["detectedReferenceLengthKm"], length_totals["referenceLengthKm"]
    )
    false_length_ratio = _safe_div(
        length_totals["falsePredictedLengthKm"], length_totals["predictedLengthKm"]
    )
    confusion: dict[str, dict[str, int]] = {}
    for case in case_reports:
        for labelled, predicted in case["typeConfusionPairs"]:
            confusion.setdefault(labelled, {})
            confusion[labelled][predicted] = (
                confusion[labelled].get(predicted, 0) + 1
            )
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
    # Standard forecast-verification names (Wilks): POD == recall, success
    # ratio == precision, CSI counts every kind of error in one number.
    csi = _safe_div(
        totals["truePositives"],
        totals["truePositives"] + totals["falsePositives"]
        + totals["falseNegatives"],
    )
    type_accuracy = _safe_div(totals["correctTypes"], totals["typedMatches"])
    outcomes = [
        outcome
        for case in case_reports
        for outcome in case.pop("predictionOutcomes")
    ]
    metadata = manifest.get("metadata") or {}
    if split == "test" and metadata.get("testFrozen") is not True:
        warnings.append("manifest: il test non e' dichiarato congelato")
    if strict and warnings:
        details = "\n - ".join(warnings)
        raise ValueError(f"manifest non idoneo come gold benchmark:\n - {details}")

    probability_eligible = (
        metadata.get("scoreCalibratorFrozen") is True
        and split == "test"
        and len(case_reports) >= MIN_PROBABILITY_CASES
        and totals["predictionCount"] >= MIN_PROBABILITY_PREDICTIONS
    )
    return {
        "schemaVersion": 2,
        "manifest": str(manifest_path),
        "split": split if split is not None else "all",
        "evaluationMode": mode if mode is not None else "all",
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
            "pod": _round_metric(recall),
            "successRatio": _round_metric(precision),
            "csi": _round_metric(csi),
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
            "lengthRecall": _round_metric(length_recall),
            "falseLengthRatio": _round_metric(false_length_ratio),
            "meanSymmetricDistanceKm": _round_metric(
                float(np.mean(all_distances)) if all_distances else None, 1
            ),
            "medianSymmetricDistanceKm": _round_metric(
                float(np.median(all_distances)) if all_distances else None, 1
            ),
            "p95SymmetricDistanceKm": _round_metric(
                float(np.percentile(all_distances, 95)) if all_distances else None, 1
            ),
        },
        "lengthAccountingKm": {
            "reference": length_totals["referenceLengthKm"],
            "predicted": length_totals["predictedLengthKm"],
            "detectedReference": length_totals["detectedReferenceLengthKm"],
            "missed": length_totals["missedLengthKm"],
            "falsePredicted": length_totals["falsePredictedLengthKm"],
        },
        "continuity": continuity_totals,
        "typeConfusionMatrix": confusion,
        "matchTypePolicy": "presence type-independent; typeAccuracy over matched pairs",
        "stratification": _stratified(case_reports),
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


def radius_sensitivity(
    manifest_path: str | os.PathLike[str],
    radii_km: list[float],
    split: str | None = "test",
    minimum_overlap: float = DEFAULT_MIN_OVERLAP,
    mode: str | None = None,
) -> list[dict[str, Any]]:
    """Re-run the benchmark at several spatial tolerances.

    A result that looks good only with a very generous radius is hiding
    position errors: metrics MUST be read together with the radius at which
    they were computed. Returns one summary row per radius, strictest first.
    """
    rows = []
    for radius in sorted(set(float(r) for r in radii_km)):
        report = evaluate_manifest(
            manifest_path, split, radius, minimum_overlap, mode
        )
        rows.append({
            "radiusKm": radius,
            "counts": report["counts"],
            "pod": report["metrics"]["pod"],
            "successRatio": report["metrics"]["successRatio"],
            "csi": report["metrics"]["csi"],
            "f1": report["metrics"]["f1"],
            "lengthRecall": report["metrics"]["lengthRecall"],
            "medianSymmetricDistanceKm": report["metrics"]["medianSymmetricDistanceKm"],
            "typeAccuracy": report["metrics"]["typeAccuracy"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="manifest JSON dei casi etichettati")
    parser.add_argument("--split", default="test", help="train, validation, test o all")
    parser.add_argument("--radius-km", type=float, default=DEFAULT_RADIUS_KM)
    parser.add_argument("--min-overlap", type=float, default=DEFAULT_MIN_OVERLAP)
    parser.add_argument(
        "--radius-grid-km",
        help="lista di raggi (es. 25,50,100,150) per la sezione radiusSensitivity",
    )
    parser.add_argument(
        "--mode", choices=["detector", "forecast"], default=None,
        help="detector = ICON-2I-ASSIM/analisi; forecast = previsione deterministica",
    )
    parser.add_argument("--out", help="scrive il report JSON; altrimenti stdout")
    parser.add_argument(
        "--strict", action="store_true",
        help="rifiuta casi senza fonte SHA-256 e annotazione cieca verificabile",
    )
    args = parser.parse_args()
    split = None if args.split == "all" else args.split
    report = evaluate_manifest(
        args.manifest, split, args.radius_km, args.min_overlap, args.mode,
        strict=args.strict,
    )
    if args.radius_grid_km:
        radii = [float(token) for token in args.radius_grid_km.split(",") if token.strip()]
        report["radiusSensitivity"] = radius_sensitivity(
            args.manifest, radii, split, args.min_overlap, args.mode
        )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
