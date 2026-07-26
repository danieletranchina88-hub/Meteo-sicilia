"""Conservative decision layer between supervised ML and front physics.

The model recognises patterns learned near manual DWD fronts. It does not know
causality and is therefore never allowed to set front type or override a hard
thermodynamic/dynamic contradiction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

HARD_REJECTIONS = {
    "synoptic-structure",
    "divergent-wind",
    "orographic-boundary",
    "cold-pool",
    "sea-breeze",
    "moisture-boundary",
    "pressure-trough-only",
}
SOFT_REJECTIONS = {
    "thermal-core",
    "wind-boundary",
    "dynamic-core",
    "pressure-ridge-warning",
    "lower-level-incoherence-warning",
}
MAX_EVIDENCE_BONUS = 0.05
RESCUE_MIN_EVIDENCE = 0.40
RESCUE_MIN_PROBABILITY = 0.30
RESCUE_MIN_SUPPORT = 0.55


def _finite(value, default=-999.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _near_thermal_core(metrics):
    """Every causal thermal ingredient must reach at least 80% of its gate."""
    checks = (
        ("deltaThetaW", 1.20),
        ("deltaTemperature", 0.60),
        ("deltaThetaV", 0.25),
        ("dryThermalGradient", 0.80),
        ("thermalAlignment", 0.05),
        ("thermalContrastFraction", 0.48),
        ("thermalAlignmentFraction", 0.48),
    )
    return all(_finite(metrics.get(name)) >= threshold * 0.80
               for name, threshold in checks)


def fuse_candidate(metrics, gates, ml_stats, *, model_threshold):
    """Return copied ``metrics``/``gates`` plus an auditable fusion verdict."""
    metrics = dict(metrics)
    gates = dict(gates)
    median = _finite(ml_stats.get("median"), np.nan)
    support = _finite(ml_stats.get("supportFraction"), np.nan)
    q75 = _finite(ml_stats.get("q75"), np.nan)
    available = bool(np.isfinite(median) and np.isfinite(support))
    metrics.update({
        "mlFrontProbability": None if not available else round(median, 4),
        "mlFrontProbabilityQ75": None if not available else round(q75, 4),
        "mlSupportFraction": None if not available else round(support, 3),
        "physicalCandidateEvidence": metrics.get("candidateEvidence"),
    })
    decision = "physics-only"
    bonus = 0.0
    rescued = False
    if available and gates.get("continuationPass"):
        # A line can run along the edge of a 40-km probability corridor:
        # require either median support, or at least the upper quartile plus
        # 20% of sampled vertices. This confirms partial alignment without
        # pretending the whole physical line is a model hit.
        signal = max(median, q75)
        aligned = (
            median >= model_threshold
            or (q75 >= model_threshold and support >= 0.20)
        )
        confidence = 0.0
        if aligned:
            ceiling = max(0.30, 4.0 * model_threshold)
            excess = np.clip(
                (signal - model_threshold)
                / max(ceiling - model_threshold, 1e-6),
                0.0, 1.0,
            )
            confidence = 0.25 + 0.75 * excess
        bonus = MAX_EVIDENCE_BONUS * confidence * np.clip(support, 0.0, 1.0)
        decision = "physics-confirmed-by-ml" if bonus > 0 else "physics-retained"
    elif available and not gates.get("continuationPass"):
        reasons = set(gates.get("rejectionReasons") or [])
        diagnosis = gates.get("diagnosis")
        high_ml = (
            median >= model_threshold
            and q75 >= max(
                RESCUE_MIN_PROBABILITY, model_threshold * 3.0
            )
            and support >= RESCUE_MIN_SUPPORT
        )
        no_hard_conflict = not (reasons & HARD_REJECTIONS)
        only_known_reasons = reasons <= SOFT_REJECTIONS
        physical_floor = (
            _finite(metrics.get("candidateEvidence"), 0.0)
            >= RESCUE_MIN_EVIDENCE
            and _near_thermal_core(metrics)
            and _finite(metrics.get("synopticSupport"), 0.0) >= 0.58
            and diagnosis == "synoptic-front"
        )
        # ML may bridge a marginal thermal threshold, never manufacture the
        # boundary. The result remains continuation-grade and must survive the
        # normal >=3 h tracking and classification checks.
        if (
            high_ml and no_hard_conflict and only_known_reasons
            and physical_floor and "thermal-core" in reasons
        ):
            rescued = True
            bonus = min(
                MAX_EVIDENCE_BONUS,
                0.02 + 0.03 * (median - RESCUE_MIN_PROBABILITY)
                / (1.0 - RESCUE_MIN_PROBABILITY),
            )
            gates["continuationPass"] = True
            gates["strongPass"] = False
            gates["gateStatus"] = "ml-assisted-continuation"
            decision = "ml-assisted-physical-near-pass"
        else:
            decision = "physics-rejected"
    if bonus:
        metrics["candidateEvidence"] = round(min(
            1.0, _finite(metrics.get("candidateEvidence"), 0.0) + bonus
        ), 3)
    metrics["fusionEvidenceBonus"] = round(float(bonus), 4)
    metrics["fusionDecision"] = decision
    metrics["mlAssisted"] = rescued
    metrics.update(gates)
    return metrics, gates


@dataclass
class MLFrontGuidance:
    latitude: np.ndarray
    longitude: np.ndarray
    probabilities: dict[int, np.ndarray]
    threshold: float

    def __post_init__(self):
        self.latitude = np.asarray(self.latitude, float)
        self.longitude = np.asarray(self.longitude, float)
        self._interpolators = {}
        for hour, values in self.probabilities.items():
            values = np.asarray(values, float)
            if values.shape != (len(self.latitude), len(self.longitude)):
                raise ValueError(f"griglia ML +{hour}h con forma errata")
            self._interpolators[int(hour)] = RegularGridInterpolator(
                (self.latitude, self.longitude), values,
                bounds_error=False, fill_value=np.nan,
            )

    def sample(self, hour, coordinates):
        interpolator = self._interpolators.get(int(hour))
        if interpolator is None and self._interpolators:
            available = sorted(self._interpolators)
            before = [value for value in available if value < int(hour)]
            after = [value for value in available if value > int(hour)]
            if before and after and after[0] - before[-1] <= 3:
                first, second = before[-1], after[0]
                fraction = (int(hour) - first) / (second - first)
                field = (
                    (1.0 - fraction) * self.probabilities[first]
                    + fraction * self.probabilities[second]
                )
                interpolator = RegularGridInterpolator(
                    (self.latitude, self.longitude), field,
                    bounds_error=False, fill_value=np.nan,
                )
        if interpolator is None:
            return {"median": np.nan, "q75": np.nan, "supportFraction": np.nan}
        coordinates = np.asarray(coordinates, float)
        values = interpolator(np.column_stack((
            coordinates[:, 1], coordinates[:, 0]
        )))
        finite = values[np.isfinite(values)]
        if not finite.size:
            return {"median": np.nan, "q75": np.nan, "supportFraction": np.nan}
        return {
            "median": float(np.median(finite)),
            "q75": float(np.quantile(finite, 0.75)),
            "supportFraction": float(np.mean(finite >= self.threshold)),
        }

    def evaluate(self, hour, coordinates, metrics, gates):
        return fuse_candidate(
            metrics, gates, self.sample(hour, coordinates),
            model_threshold=self.threshold,
        )
