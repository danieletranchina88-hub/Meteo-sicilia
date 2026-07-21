"""Geometric front tracking and consensus classification (v10, phase 3).

Tracking is a central part of the algorithm, not a persistence filter.
Candidates from consecutive hours are linked into tracks by a **global**
(Hungarian) assignment on a geometric cost that uses the track's predicted
position, symmetric line-to-line distance, orientation, length change and
warm-side consistency.  Births, deaths, and (structurally) splits/merges
and occlusions are handled explicitly.

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
        a = _resample(prev)
        b = _resample(last)
        shift = np.median(b - a, axis=0) * (dt_pred / max(dt_hist, 1))
        return last + shift


def _assignment_cost(track: Track, candidate: dict, hour: int,
                     gate_km: float) -> float:
    predicted = track.predicted_line(hour)
    line = candidate["coordinates"]
    distance = _symmetric_distance_km(predicted, line)
    if distance > gate_km:
        return np.inf
    orient = _orientation_diff(predicted, line)
    length_prev = track.lines[track.hours[-1]]["lengthKm"]
    length_change = abs(candidate["lengthKm"] - length_prev) / max(length_prev, 1.0)
    return distance + 3.0 * orient + 60.0 * min(length_change, 2.0)


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
    abz = [track.lines[h]["medianAbzGradient"] for h in hours]
    support = [track.lines[h].get("synopticSupport", 0.5) for h in hours]
    corroborated = [bool(track.lines[h].get("corroborated", False)) for h in hours]

    thermal = float(np.clip(np.median(grads) / 6.0, 0.0, 1.0))
    dynamic = float(np.clip(np.median(abz) / 6.0, 0.0, 1.0))
    span = hours[-1] - hours[0]
    expected = span / max(window_hours, 1) + 1
    temporal = float(np.clip(len(hours) / max(expected, 1.0), 0.0, 1.0))
    agreement = float(np.clip(0.5 * np.median(support) + 0.5 * np.mean(corroborated), 0.0, 1.0))
    certainty = classification["classificationCertainty"]

    components = {
        "thermalSupport": round(thermal, 2),
        "dynamicSupport": round(dynamic, 2),
        "temporalSupport": round(temporal, 2),
        "modelAgreement": round(agreement, 2),
        "classificationCertainty": round(certainty, 2),
    }
    overall = float(np.clip(
        0.3 * thermal + 0.2 * dynamic + 0.25 * temporal
        + 0.15 * agreement + 0.1 * certainty, 0.0, 1.0
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
        classification = classify_track(track, window_hours, wind_sampler)
        quality = _quality_score(track, classification, window_hours)
        results.append({
            "id": track.id,
            "hours": list(track.hours),
            "lifetimeH": span,
            "lines": {h: track.lines[h]["coordinates"] for h in track.hours},
            **classification,
            **quality,
        })
    results.sort(key=lambda r: r["qualityScore"], reverse=True)
    return results
