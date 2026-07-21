"""Geometric front tracking and consensus classification (v10, phase 3).

Tracking is a central part of the algorithm, not a persistence filter.
Candidates from consecutive hours are linked into tracks by a **global**
(Hungarian) assignment on a geometric cost that uses the track's predicted
position, symmetric line-to-line distance, orientation, length change and
warm-side consistency. Births and deaths are represented explicitly; a
split or merge is intentionally treated as the end of one identity and the
birth of another, rather than being claimed as a continuously tracked front.

Classification is decided by **consensus** of three independent signals,
not a fixed blend:

 1. geometric motion  - the real displacement of the line between hours,
    measured locally along the line and projected on the warm-ward normal
    (positive toward warm = cold front).  This is the primary signal and
    needs no wind.
 2. normal advection  - wind component across the front (physical check),
    available when a wind sampler is provided.
 3. OFA frontal speed - Hewson's V . grad|grad theta_w| / |grad|grad|
    (thermodynamic type), also wind-based.

When the available signals agree -> confident cold/warm/stationary; when
they conflict -> ``frontType = "uncertain"`` (better honest than wrong).

The published score is a ``qualityScore`` (a physical-support heuristic,
explicitly NOT a probability) with separate components:
thermalSupport, dynamicSupport, temporalSupport, modelAgreement,
classificationCertainty.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

EARTH_KM_PER_DEG = 111.32
COLD_WARM_THRESHOLD_KMH = 5.0


def _finite_median(values, default=np.nan) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else float(default)


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _project_km(coordinates: np.ndarray, mean_lat_rad: float) -> np.ndarray:
    scale_lon = EARTH_KM_PER_DEG * np.cos(mean_lat_rad)
    return np.column_stack((coordinates[:, 0] * scale_lon,
                            coordinates[:, 1] * EARTH_KM_PER_DEG))


def _resample(coordinates: np.ndarray, count: int = 32) -> np.ndarray:
    if len(coordinates) < 2:
        return np.repeat(coordinates, count, axis=0)[:count]
    deltas = np.diff(coordinates, axis=0)
    seg = np.hypot(deltas[:, 0], deltas[:, 1])
    cumulative = np.concatenate(([0.0], np.cumsum(seg)))
    total = max(cumulative[-1], 1.0e-9)
    targets = np.linspace(0.0, total, count)
    return np.column_stack((
        np.interp(targets, cumulative, coordinates[:, 0]),
        np.interp(targets, cumulative, coordinates[:, 1]),
    ))


def _aligned_resample(reference: np.ndarray, candidate: np.ndarray,
                      count: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Resample two lines and orient the second like the first."""
    first = _resample(reference, count)
    second = _resample(candidate, count)
    forward = float(np.mean(np.hypot(*(first - second).T)))
    backward = float(np.mean(np.hypot(*(first - second[::-1]).T)))
    if backward < forward:
        second = second[::-1]
    return first, second


def _length_km(coordinates: np.ndarray) -> float:
    if len(coordinates) < 2:
        return 0.0
    lon1, lat1 = coordinates[:-1, 0], coordinates[:-1, 1]
    lon2, lat2 = coordinates[1:, 0], coordinates[1:, 1]
    mean_lat = np.deg2rad((lat1 + lat2) * 0.5)
    dx = (lon2 - lon1) * EARTH_KM_PER_DEG * np.cos(mean_lat)
    dy = (lat2 - lat1) * EARTH_KM_PER_DEG
    return float(np.sum(np.hypot(dx, dy)))


def _symmetric_distance_km(line_a: np.ndarray, line_b: np.ndarray) -> float:
    """Mean symmetric nearest-neighbour distance between two lines (km)."""
    mean_lat = math_mean_lat(line_a, line_b)
    a = _project_km(_resample(line_a), mean_lat)
    b = _project_km(_resample(line_b), mean_lat)
    dab = np.min(np.hypot(a[:, None, 0] - b[None, :, 0],
                          a[:, None, 1] - b[None, :, 1]), axis=1)
    dba = np.min(np.hypot(b[:, None, 0] - a[None, :, 0],
                          b[:, None, 1] - a[None, :, 1]), axis=1)
    return 0.5 * (float(np.mean(dab)) + float(np.mean(dba)))


def _point_to_segments_km(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    if len(line) < 2:
        return np.hypot(points[:, 0] - line[0, 0], points[:, 1] - line[0, 1])
    starts, ends = line[:-1], line[1:]
    segments = ends - starts
    length_sq = np.maximum(np.sum(segments * segments, axis=1), 1.0e-9)
    relative = points[:, None, :] - starts[None, :, :]
    fraction = np.clip(
        np.sum(relative * segments[None, :, :], axis=2) / length_sq[None, :],
        0.0,
        1.0,
    )
    projection = starts[None, :, :] + fraction[:, :, None] * segments[None, :, :]
    return np.min(np.hypot(
        points[:, None, 0] - projection[:, :, 0],
        points[:, None, 1] - projection[:, :, 1],
    ), axis=1)


def _matched_fraction(line: np.ndarray, reference: np.ndarray,
                      radius_km: float) -> float:
    """Fraction of ``line`` within radius of reference polyline segments."""
    mean_lat = math_mean_lat(line, reference)
    count = max(2, int(_length_km(line) / 20.0))
    points = _project_km(_resample(line, count), mean_lat)
    reference_km = _project_km(np.asarray(reference, dtype=float), mean_lat)
    return float(np.mean(_point_to_segments_km(points, reference_km) <= radius_km))


def cross_model_diagnostics(
    track: dict,
    reference_by_hour: dict,
    radius_km_fn,
    *,
    min_line_match: float = 0.55,
    min_hour_fraction: float = 0.60,
    min_coverage: float = 0.50,
) -> dict:
    """Confirm a whole track against published fronts of another model."""
    guide_hours = [h for h in track["hours"] if h in reference_by_hour]
    coverage = len(guide_hours) / max(len(track["hours"]), 1)
    if not guide_hours:
        return {
            "available": False,
            "coverage": 0.0,
            "agreement": 0.0,
            "matchedHourFraction": 0.0,
            "confirmed": False,
            "referenceFrontType": None,
        }

    scores = []
    matched_types = []
    for hour in guide_hours:
        features = (reference_by_hour.get(hour) or {}).get("features") or []
        line = np.asarray(track["lines"][hour], dtype=float)
        best_score = 0.0
        best_type = None
        for feature in features:
            reference = np.asarray(
                feature.get("geometry", {}).get("coordinates") or [], dtype=float
            )
            if len(reference) < 2:
                continue
            radius = float(radius_km_fn(hour))
            forward = _matched_fraction(line, reference, radius)
            backward = _matched_fraction(reference, line, radius)
            score = float(np.sqrt(max(forward, 0.0) * max(backward, 0.0)))
            if score > best_score:
                best_score = score
                best_type = (feature.get("properties") or {}).get("frontType")
        scores.append(best_score)
        if best_score >= min_line_match and best_type:
            matched_types.append(best_type)

    matched_hour_fraction = float(np.mean(np.asarray(scores) >= min_line_match))
    agreement = float(np.mean(scores))
    reference_type = None
    if matched_types:
        values, counts = np.unique(matched_types, return_counts=True)
        reference_type = str(values[int(np.argmax(counts))])
    type_conflict = (
        track.get("frontType") in {"cold", "warm"}
        and reference_type in {"cold", "warm"}
        and track.get("frontType") != reference_type
    )
    confirmed = (
        coverage >= min_coverage
        and matched_hour_fraction >= min_hour_fraction
        and float(np.median(scores)) >= min_line_match
        and not type_conflict
    )
    return {
        "available": True,
        "coverage": round(coverage, 2),
        "agreement": round(agreement, 2),
        "matchedHourFraction": round(matched_hour_fraction, 2),
        "confirmed": bool(confirmed),
        "referenceFrontType": reference_type,
    }


def math_mean_lat(*lines: np.ndarray) -> float:
    lats = np.concatenate([line[:, 1] for line in lines])
    return float(np.deg2rad(np.mean(lats)))


def _orientation_deg(coordinates: np.ndarray) -> float:
    tangent = coordinates[-1] - coordinates[0]
    mean_lat = np.deg2rad(float(np.mean(coordinates[:, 1])))
    angle = np.degrees(np.arctan2(tangent[1], tangent[0] * np.cos(mean_lat)))
    return angle % 180.0


def _orientation_diff(a: np.ndarray, b: np.ndarray) -> float:
    diff = abs(_orientation_deg(a) - _orientation_deg(b)) % 180.0
    return min(diff, 180.0 - diff)


def geometric_motion_kmh(
    line_prev: np.ndarray, line_next: np.ndarray,
    warm_normal_prev: np.ndarray, hours: float,
) -> float:
    """Signed displacement of the line along the warm-ward normal, km/h.

    Positive = the line moved toward the warm air (cold front); negative =
    toward the cold air (warm front).  Measured locally: each resampled
    point of the previous line is displaced to its nearest point on the
    next line and projected on the local warm normal.
    """
    mean_lat = math_mean_lat(line_prev, line_next)
    prev_km = _project_km(_resample(line_prev), mean_lat)
    next_km = _project_km(_resample(line_next), mean_lat)
    normal = _resample_vectors(warm_normal_prev, len(prev_km))
    scale_lon = EARTH_KM_PER_DEG * np.cos(mean_lat)
    normal_km = np.column_stack((normal[:, 0] * scale_lon,
                                 normal[:, 1] * EARTH_KM_PER_DEG))
    norm = np.hypot(normal_km[:, 0], normal_km[:, 1])
    normal_km /= np.maximum(norm[:, None], 1.0e-9)
    displacement = np.empty(len(prev_km))
    for i, point in enumerate(prev_km):
        j = np.argmin(np.hypot(next_km[:, 0] - point[0], next_km[:, 1] - point[1]))
        displacement[i] = np.dot(next_km[j] - point, normal_km[i])
    return float(np.median(displacement)) / max(hours, 1.0e-6)


def _resample_vectors(vectors: np.ndarray, count: int) -> np.ndarray:
    idx = np.linspace(0, len(vectors) - 1, count)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(vectors) - 1)
    frac = (idx - lo)[:, None]
    return vectors[lo] * (1 - frac) + vectors[hi] * frac


# --------------------------------------------------------------------------
# Track objects
# --------------------------------------------------------------------------
class Track:
    __slots__ = ("id", "hours", "lines")

    def __init__(self, track_id: int, hour: int, candidate: dict):
        self.id = track_id
        self.hours = [hour]
        self.lines = {hour: candidate}

    def last_hour(self) -> int:
        return self.hours[-1]

    def predicted_line(self, hour: int) -> np.ndarray:
        """Predict geometry at ``hour`` by extrapolating the last motion."""
        last = self.lines[self.hours[-1]]["coordinates"]
        if len(self.hours) < 2:
            return last
        prev = self.lines[self.hours[-2]]["coordinates"]
        dt_hist = self.hours[-1] - self.hours[-2]
        dt_pred = hour - self.hours[-1]
        # rigid translation by the median point displacement
        a, b = _aligned_resample(prev, last)
        shift = np.median(b - a, axis=0) * (dt_pred / max(dt_hist, 1))
        return last + shift


def _assignment_cost(track: Track, candidate: dict, hour: int,
                     gate_km: float) -> float:
    predicted = track.predicted_line(hour)
    line = candidate["coordinates"]
    distance = _symmetric_distance_km(predicted, line)
    gap_hours = max(hour - track.last_hour(), 1)
    # An hourly front cannot jump 250 km. Keep a hard ceiling for sparse
    # 6-hour steps while using a physical gate for ICON's hourly sequence.
    physical_gate = min(gate_km, 45.0 + 35.0 * gap_hours)
    if distance > physical_gate:
        return np.inf
    orient = _orientation_diff(predicted, line)
    if orient > 70.0:
        return np.inf
    length_prev = track.lines[track.hours[-1]]["lengthKm"]
    length_change = abs(candidate["lengthKm"] - length_prev) / max(length_prev, 1.0)
    previous_normal = _resample_vectors(
        track.lines[track.hours[-1]]["warmNormal"], 32
    )
    candidate_normal = _resample_vectors(candidate["warmNormal"], 32)
    predicted_points = _resample(predicted, 32)
    candidate_points = _resample(line, 32)
    forward = float(np.mean(np.hypot(*(predicted_points - candidate_points).T)))
    backward = float(np.mean(np.hypot(*(predicted_points - candidate_points[::-1]).T)))
    if backward < forward:
        candidate_normal = candidate_normal[::-1]
    normal_alignment = float(np.nanmedian(
        np.sum(previous_normal * candidate_normal, axis=1)
    ))
    if not np.isfinite(normal_alignment) or normal_alignment < 0.0:
        return np.inf
    return (
        distance
        + 2.0 * orient
        + 60.0 * min(length_change, 2.0)
        + 55.0 * (1.0 - np.clip(normal_alignment, 0.0, 1.0))
    )


def _global_assign(tracks: list[Track], candidates: list[dict], hour: int,
                   gate_km: float):
    """Hungarian assignment of candidates to active tracks (birth/death)."""
    if not tracks:
        return {}, list(range(len(candidates)))
    if not candidates:
        return {}, []
    cost = np.full((len(tracks), len(candidates)), 1.0e6)
    for i, track in enumerate(tracks):
        for j, candidate in enumerate(candidates):
            c = _assignment_cost(track, candidate, hour, gate_km)
            cost[i, j] = c if np.isfinite(c) else 1.0e6
    rows, cols = linear_sum_assignment(cost)
    matches, matched_candidates = {}, set()
    for r, c in zip(rows, cols):
        if cost[r, c] < 1.0e6:
            matches[r] = c
            matched_candidates.add(c)
    unmatched = [j for j in range(len(candidates)) if j not in matched_candidates]
    return matches, unmatched


# --------------------------------------------------------------------------
# Classification (consensus)
# --------------------------------------------------------------------------
def _type_from_speed(speed_kmh: float) -> str:
    if speed_kmh >= COLD_WARM_THRESHOLD_KMH:
        return "cold"
    if speed_kmh <= -COLD_WARM_THRESHOLD_KMH:
        return "warm"
    return "stationary"


def classify_track(track: Track, window_hours: int, wind_sampler=None) -> dict:
    """Consensus cold/warm/stationary + certainty for a whole track."""
    hours = track.hours
    geo_speeds = []
    for h_prev, h_next in zip(hours[:-1], hours[1:]):
        line_prev = track.lines[h_prev]["coordinates"]
        line_next = track.lines[h_next]["coordinates"]
        warm_prev = track.lines[h_prev]["warmNormal"]
        geo_speeds.append(
            geometric_motion_kmh(line_prev, line_next, warm_prev, h_next - h_prev)
        )
    geo_motion = float(np.median(geo_speeds)) if geo_speeds else 0.0
    geo_type = _type_from_speed(geo_motion)

    votes = [geo_type]
    adv_motion = None
    ofa_motion = None
    if wind_sampler is not None:
        adv_list, ofa_list = [], []
        for hour in hours:
            candidate = track.lines[hour]
            points = candidate["coordinates"]
            u, v = wind_sampler(hour, points)
            warm = candidate["warmNormal"]
            hewson = candidate["hewsonDir"]
            # advection toward warm (km/h): wind . warm_normal.
            # Positive = wind toward warm = cold advection = cold front.
            adv_list.append(np.nanmedian(u * warm[:, 0] + v * warm[:, 1]) * 3.6)
            # OFA (Hewson) frontal-speed projection.  hewson_dir points
            # toward the crest of |grad theta_w|, i.e. toward the COLD side,
            # so the sign is negated to keep the same convention (positive =
            # toward warm = cold front) as the other two signals.
            ofa_list.append(-np.nanmedian(u * hewson[:, 0] + v * hewson[:, 1]) * 3.6)
        adv_motion = float(np.nanmedian(adv_list))
        ofa_motion = float(np.nanmedian(ofa_list))
        votes.append(_type_from_speed(adv_motion))
        votes.append(_type_from_speed(ofa_motion))

    # consensus
    unique = set(votes)
    if len(unique) == 1:
        front_type = votes[0]
        certainty = 1.0 if wind_sampler is not None else 0.6
    else:
        # geometric motion is authoritative on movement; disagreement only
        # about magnitude (stationary vs moving) keeps the moving type at
        # reduced certainty, a genuine cold/warm conflict is 'uncertain'.
        if "cold" in unique and "warm" in unique:
            front_type = "uncertain"
            certainty = 0.2
        else:
            front_type = geo_type if geo_type != "stationary" else next(
                t for t in votes if t != "stationary"
            )
            certainty = 0.5
    return {
        "frontType": front_type,
        "geoMotionKmh": round(geo_motion, 1),
        "advectionKmh": None if adv_motion is None else round(adv_motion, 1),
        "ofaSpeedKmh": None if ofa_motion is None else round(ofa_motion, 1),
        "classificationCertainty": round(certainty, 2),
    }


# --------------------------------------------------------------------------
# Quality score (NOT a probability)
# --------------------------------------------------------------------------
def _quality_score(track: Track, classification: dict, window_hours: int) -> dict:
    hours = track.hours
    grads = [track.lines[h]["medianThetaWGradient"] for h in hours]
    dry_gradients = [track.lines[h].get("dryThermalGradient", 0.0) for h in hours]
    wind_shift = [track.lines[h].get("windShiftMs", 0.0) for h in hours]
    convergence = [max(track.lines[h].get("convergenceMs", 0.0), 0.0) for h in hours]
    support = [track.lines[h].get("synopticSupport", 0.0) for h in hours]

    moist_thermal = float(np.clip(np.median(grads) / 5.0, 0.0, 1.0))
    dry_thermal = float(np.clip(np.median(dry_gradients) / 3.0, 0.0, 1.0))
    thermal = 0.6 * moist_thermal + 0.4 * dry_thermal
    shift_support = float(np.clip(np.median(wind_shift) / 6.0, 0.0, 1.0))
    convergence_support = float(np.clip(np.median(convergence) / 2.0, 0.0, 1.0))
    dynamic = max(shift_support, convergence_support)
    span = hours[-1] - hours[0]
    expected = span / max(window_hours, 1) + 1
    temporal = float(np.clip(len(hours) / max(expected, 1.0), 0.0, 1.0))
    structural = float(np.clip(np.median(support), 0.0, 1.0))
    certainty = classification["classificationCertainty"]

    components = {
        "thermalSupport": round(thermal, 2),
        "dynamicSupport": round(dynamic, 2),
        "temporalSupport": round(temporal, 2),
        "structuralSupport": round(structural, 2),
        # Filled only after comparison with an independent model. Never use
        # the same-model smoothing prior as fake model agreement.
        "modelAgreement": None,
        "classificationCertainty": round(certainty, 2),
    }
    overall = float(np.clip(
        0.3 * thermal + 0.2 * dynamic + 0.25 * temporal
        + 0.15 * structural + 0.1 * certainty, 0.0, 1.0
    ))
    return {"qualityScore": round(overall, 2), "qualityComponents": components}


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def track_fronts(
    hourly_candidates: dict[int, list],
    *,
    window_hours: int = 3,
    gate_km: float = 250.0,
    min_lifetime_hours: int = 6,
    min_coverage: float = 0.5,
    wind_sampler=None,
    require_physical_support: bool = False,
    min_dry_gradient: float = 1.2,
    min_wind_shift_ms: float = 1.5,
    min_convergence_ms: float = 0.15,
    max_terrain_fraction: float = 0.25,
):
    """Link hourly candidates into classified, quality-scored tracks.

    ``hourly_candidates`` maps forecast hour -> list of candidate dicts
    (from front_detection).  ``wind_sampler(hour, points_lonlat)`` -> (u, v)
    in m/s enables the advection and OFA cross-checks (optional).
    Returns a list of accepted track summaries.
    """
    hours = sorted(hourly_candidates)
    tracks: list[Track] = []
    active: list[Track] = []
    next_id = 0
    max_gap = 2 * window_hours

    for hour in hours:
        candidates = list(hourly_candidates[hour])
        active = [t for t in active if hour - t.last_hour() <= max_gap]
        matches, unmatched = _global_assign(active, candidates, hour, gate_km)
        for track_index, cand_index in matches.items():
            track = active[track_index]
            track.hours.append(hour)
            track.lines[hour] = candidates[cand_index]
        for cand_index in unmatched:
            track = Track(next_id, hour, candidates[cand_index])
            next_id += 1
            tracks.append(track)
            active.append(track)

    available = sorted(hourly_candidates)
    results = []
    for track in tracks:
        span = track.hours[-1] - track.hours[0]
        if len(track.hours) < 2 or span < min_lifetime_hours:
            continue
        expected = [h for h in available if track.hours[0] <= h <= track.hours[-1]]
        if len(track.hours) / max(len(expected), 1) < min_coverage:
            continue
        diagnostics = {
            "dryThermalGradient": _finite_median([
                track.lines[h].get("dryThermalGradient", np.nan) for h in track.hours
            ]),
            "windShiftMs": _finite_median([
                track.lines[h].get("windShiftMs", np.nan) for h in track.hours
            ]),
            "convergenceMs": _finite_median([
                track.lines[h].get("convergenceMs", np.nan) for h in track.hours
            ]),
            "pressureTroughHpa": _finite_median([
                track.lines[h].get("pressureTroughHpa", np.nan) for h in track.hours
            ]),
            "terrainFraction": _finite_median([
                track.lines[h].get("terrainFraction", np.nan) for h in track.hours
            ]),
        }
        if require_physical_support:
            dry_gradient = diagnostics["dryThermalGradient"]
            wind_shift_value = diagnostics["windShiftMs"]
            convergence_value = diagnostics["convergenceMs"]
            terrain_fraction = diagnostics["terrainFraction"]
            if not np.isfinite(dry_gradient) or dry_gradient < min_dry_gradient:
                continue
            if (
                (not np.isfinite(wind_shift_value) or wind_shift_value < min_wind_shift_ms)
                and (
                    not np.isfinite(convergence_value)
                    or convergence_value < min_convergence_ms
                )
            ):
                continue
            if np.isfinite(terrain_fraction) and terrain_fraction > max_terrain_fraction:
                continue
        classification = classify_track(track, window_hours, wind_sampler)
        quality = _quality_score(track, classification, window_hours)
        results.append({
            "id": track.id,
            "hours": list(track.hours),
            "lifetimeH": span,
            "lines": {h: track.lines[h]["coordinates"] for h in track.hours},
            "diagnostics": diagnostics,
            **classification,
            **quality,
        })
    results.sort(key=lambda r: r["qualityScore"], reverse=True)
    return results
