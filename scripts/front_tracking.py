"""Global front tracking, motion classification and uncertainty (v13).

Tracking is a central part of the algorithm, not a persistence filter.
Candidates from consecutive hours are linked into tracks by a **global**
(Hungarian) assignment on a geometric cost that uses the track's predicted
position, symmetric line-to-line distance, orientation, length change and
warm-side consistency. Births and deaths are represented explicitly; a
split or merge is intentionally treated as the end of one identity and the
birth of another, rather than being claimed as a continuously tracked front.

Classification is decided by **consensus** of two independent signals,
not a fixed blend:

 1. geometric motion  - the real displacement of the line between hours,
    measured locally along the line and projected on the warm-ward normal
    (positive toward warm = cold front).  This is the primary signal and
    needs no wind.
 2. OFA frontal speed - Hewson's V . grad|grad theta_w| / |grad|grad|
    (thermodynamic type), precomputed on each line. A wind sampler supplies
    exactly this signal only for backward-compatible synthetic callers.

When the available signals agree -> confident cold/warm/stationary; when
they conflict -> ``frontType = "uncertain"`` (better honest than wrong).

Hewson's 850-hPa front speed is the thermodynamic classification signal;
observed line displacement is an independent temporal check.  A cold/warm
sign conflict is never averaged away: the track becomes ``uncertain`` and
is not published by the operational analyzer.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

EARTH_KM_PER_DEG = 111.32
COLD_WARM_THRESHOLD_KMH = 5.4  # 1.5 m/s, Hewson/Berry/Sansom-Catto


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
    """Classify only when three independent motion families agree.

    Geometry measures where the analysed boundary actually moved; phase
    speed follows the theta-w tendency; the wind family combines Hewson OFA
    speed and flow normal to the air-mass boundary.  Wind is deliberately a
    single family because its two diagnostics are not independent.
    """
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

    ofa_values = [
        track.lines[h].get("ofaSpeedMps", np.nan) for h in hours
    ]
    # Standard OFA speed is negative for cold fronts.  Convert to the common
    # convention used here: positive motion points toward warm air (cold).
    ofa_motion = -_finite_median(ofa_values) * 3.6
    tendency_motion = _finite_median([
        track.lines[h].get("tendencyMotionKmh", np.nan) for h in hours
    ])
    airmass_motion = _finite_median([
        track.lines[h].get("airmassMotionKmh", np.nan) for h in hours
    ])

    # Backward-compatible fallback for synthetic callers that provide winds
    # but not precomputed OFA diagnostics.
    if not np.isfinite(ofa_motion) and wind_sampler is not None:
        fallback = []
        for hour in hours:
            candidate = track.lines[hour]
            u, v = wind_sampler(hour, candidate["coordinates"])
            hewson = candidate["hewsonDir"]
            fallback.append(-np.nanmedian(
                u * hewson[:, 0] + v * hewson[:, 1]
            ) * 3.6)
        ofa_motion = _finite_median(fallback)

    ofa_type = _type_from_speed(ofa_motion) if np.isfinite(ofa_motion) else None
    airmass_type = (
        _type_from_speed(airmass_motion) if np.isfinite(airmass_motion) else None
    )
    wind_parts = [value for value in (ofa_type, airmass_type) if value]
    wind_moving = {value for value in wind_parts if value != "stationary"}
    wind_conflict = len(wind_moving) > 1
    if wind_conflict:
        wind_type = "uncertain"
    elif wind_moving:
        wind_type = next(iter(wind_moving))
    elif wind_parts:
        wind_type = "stationary"
    else:
        wind_type = None

    phase_type = (
        _type_from_speed(tendency_motion)
        if np.isfinite(tendency_motion) else None
    )
    family_votes = {
        "geometry": geo_type,
        "phase": phase_type,
        "wind": wind_type,
    }
    available_votes = [
        value for value in family_votes.values() if value is not None
    ]
    moving = {
        value for value in available_votes
        if value not in ("stationary", "uncertain")
    }
    has_uncertain = "uncertain" in available_votes
    counts = {
        kind: sum(value == kind for value in available_votes)
        for kind in ("cold", "warm", "stationary")
    }

    if has_uncertain or len(moving) > 1:
        front_type, certainty = "uncertain", 0.10
    elif moving:
        moving_type = next(iter(moving))
        opposite = "warm" if moving_type == "cold" else "cold"
        support = counts[moving_type]
        # A moving front needs two independent families. A third stationary
        # family is allowed; an opposite vote is not.
        if support >= 2 and counts[opposite] == 0:
            front_type = moving_type
            certainty = 0.72 + 0.12 * min(support - 2, 1)
        else:
            front_type, certainty = "uncertain", 0.24
    else:
        if counts["stationary"] >= 2:
            front_type = "stationary"
            certainty = 0.76 + 0.08 * min(counts["stationary"] - 2, 1)
        else:
            front_type, certainty = "uncertain", 0.24

    # Penalise an unstable OFA sign over the track without allowing a weak
    # single hour to flip the median classification.
    hourly_motion_types = []
    for h in hours:
        ofa_value = track.lines[h].get("ofaSpeedMps", np.nan)
        air_value = track.lines[h].get("airmassMotionKmh", np.nan)
        if np.isfinite(ofa_value):
            hourly_motion_types.append(_type_from_speed(-float(ofa_value) * 3.6))
        if np.isfinite(air_value):
            hourly_motion_types.append(_type_from_speed(float(air_value)))
    if hourly_motion_types and front_type != "uncertain":
        agreement = float(np.mean([
            value == front_type or value == "stationary"
            for value in hourly_motion_types
        ]))
        contradiction = float(np.mean([
            value not in (front_type, "stationary")
            for value in hourly_motion_types
        ]))
        certainty *= 0.55 + 0.45 * agreement
        if contradiction > 0.20:
            front_type = "uncertain"
            certainty = min(certainty, 0.18)

    motion_mad = _finite_median(np.abs(np.asarray(geo_speeds) - geo_motion), 0.0)
    return {
        "frontType": front_type,
        "geoMotionKmh": round(geo_motion, 1),
        "ofaSpeedKmh": None if not np.isfinite(ofa_motion) else round(ofa_motion, 1),
        "tendencyMotionKmh": None if not np.isfinite(tendency_motion) else round(tendency_motion, 1),
        "airmassMotionKmh": None if not np.isfinite(airmass_motion) else round(airmass_motion, 1),
        "motionVotes": family_votes,
        "windMotionConflict": bool(wind_conflict),
        "motionMadKmh": round(float(motion_mad), 1),
        "classificationCertainty": round(float(certainty), 2),
    }


def classify_track_locally(
    track: Track, window_hours: int, wind_sampler=None
) -> dict[int, dict]:
    """Classify the same identity in a moving temporal window.

    A front may slow down or reverse without ceasing to be the same air-mass
    boundary. One label for a 20-hour track would therefore be physically
    misleading. Endpoints use the nearest three detections when available.
    """
    radius = max(2, int(window_hours))
    local: dict[int, dict] = {}
    for center in track.hours:
        selected = [h for h in track.hours if abs(h - center) <= radius]
        if len(selected) < 3:
            selected = sorted(
                sorted(track.hours, key=lambda h: (abs(h - center), h))[:3]
            )
        if len(selected) < 2:
            local[center] = {
                "frontType": "uncertain",
                "classificationCertainty": 0.0,
            }
            continue
        subset = Track(track.id, selected[0], track.lines[selected[0]])
        for hour in selected[1:]:
            subset.hours.append(hour)
            subset.lines[hour] = track.lines[hour]
        local[center] = classify_track(subset, window_hours, wind_sampler)
    return local


# --------------------------------------------------------------------------
# Quality score (NOT a probability)
# --------------------------------------------------------------------------
def _quality_score(track: Track, classification: dict, window_hours: int) -> dict:
    hours = track.hours
    candidate_evidence = _finite_median([
        track.lines[h].get("candidateEvidence", np.nan) for h in hours
    ], 0.0)
    component_names = ("thermal", "dynamic", "pressure", "vertical", "activity", "structural")
    physical_components = {
        name: _finite_median([
            track.lines[h].get("evidenceComponents", {}).get(name, np.nan)
            for h in hours
        ], 0.0)
        for name in component_names
    }
    span = hours[-1] - hours[0]
    coverage = float(np.clip(len(hours) / max(span + 1, 1), 0.0, 1.0))
    lifetime = float(np.clip((span - 3.0) / 6.0, 0.0, 1.0))
    strong_fraction = float(np.mean([
        track.lines[h].get("gateStatus", "strong") == "strong"
        for h in hours
    ]))
    motion_consistency = float(np.clip(
        1.0 - classification.get("motionMadKmh", 0.0) / 25.0, 0.0, 1.0
    ))
    temporal = (
        0.38 * coverage + 0.20 * lifetime
        + 0.22 * motion_consistency + 0.20 * strong_fraction
    )
    certainty = float(classification["classificationCertainty"])
    structural = physical_components["structural"]
    components = {
        "physicalEvidence": round(candidate_evidence, 2),
        "thermalSupport": round(physical_components["thermal"], 2),
        "dynamicSupport": round(physical_components["dynamic"], 2),
        "pressureSupport": round(physical_components["pressure"], 2),
        "verticalSupport": round(physical_components["vertical"], 2),
        "temporalSupport": round(temporal, 2),
        "structuralSupport": round(structural, 2),
        "classificationCertainty": round(certainty, 2),
        "strongDetectionFraction": round(strong_fraction, 2),
    }
    overall = float(np.clip(
        0.55 * candidate_evidence + 0.25 * temporal
        + 0.12 * structural + 0.08 * certainty,
        0.0, 1.0,
    ))
    # Diagnostic uncertainty also accounts for a weak *existence* dimension:
    # a high temporal score cannot hide a poor thermal contrast or an
    # incoherent structure. Dynamics are deliberately excluded here - a
    # genuine quasi-stationary front legitimately has weak cross-front wind
    # and convergence, and must not be labelled "uncertain" (and dropped)
    # for that. Weak dynamics already lower qualityScore through the dynamic
    # evidence component; they must not additionally erase a thermally and
    # structurally solid boundary.
    core_penalty = max(
        0.0,
        0.50 - min(
            physical_components["thermal"],
            structural,
        ),
    ) * 0.35
    uncertainty = float(np.clip(1.0 - overall + core_penalty, 0.0, 1.0))
    uncertainty_class = "low" if uncertainty <= 0.36 else (
        "moderate" if uncertainty <= 0.52 else "high"
    )
    return {
        "qualityScore": round(overall, 2),
        "uncertaintyIndex": round(uncertainty, 2),
        "uncertaintyClass": uncertainty_class,
        "qualityComponents": components,
    }


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def track_fronts(
    hourly_candidates: dict[int, list],
    *,
    window_hours: int = 3,
    gate_km: float = 250.0,
    min_lifetime_hours: int = 6,
    min_detections: int = 2,
    min_coverage: float = 0.5,
    wind_sampler=None,
    require_physical_support: bool = False,
    min_dry_gradient: float = 1.2,
    min_wind_shift_ms: float = 1.5,
    min_convergence_ms: float = 0.15,
    max_terrain_fraction: float = 0.25,
    min_strong_detections: int = 1,
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
    # A front absent longer than the declared tracking window is a new
    # identity, not the continuation of the old one. This prevents an
    # hourly line from disappearing for 4-6 hours and reappearing with the
    # same track id and an inflated lifetime.
    max_gap = window_hours

    for hour in hours:
        candidates = list(hourly_candidates[hour])
        active = [t for t in active if hour - t.last_hour() <= max_gap]
        matches, unmatched = _global_assign(active, candidates, hour, gate_km)
        for track_index, cand_index in matches.items():
            track = active[track_index]
            track.hours.append(hour)
            track.lines[hour] = candidates[cand_index]
        for cand_index in unmatched:
            # Every unmatched candidate reaching this point already passed the
            # hard, non-compensating physical gates (continuationPass): it is a
            # genuine synoptic-front segment, not mesoscale noise. It may
            # therefore start a track even when it is only "continuation" grade.
            # A real but rarely-"strong" synoptic front (a slow, quasi-
            # stationary boundary is the textbook case) would otherwise never be
            # born and would be lost entirely. Ephemeral or spurious segments
            # are still removed downstream by the lifetime, detection and
            # coverage thresholds.
            track = Track(next_id, hour, candidates[cand_index])
            next_id += 1
            tracks.append(track)
            active.append(track)

    available = sorted(hourly_candidates)
    results = []
    for track in tracks:
        span = track.hours[-1] - track.hours[0]
        if len(track.hours) < max(2, int(min_detections)) or span < min_lifetime_hours:
            continue
        # A single "strong"-grade hour anchors the track's identity, but a
        # coherent front that stays at "continuation" grade for many hours
        # (a slow synoptic boundary passing every hard physical gate) must NOT
        # be discarded for lacking repeated "strong" hours. Track quality and
        # coverage below, plus the qualityScore, weigh a continuation-heavy
        # track down without erasing it. The former "no long marginal run"
        # rule erased exactly these real long-lived boundaries and is gone.
        strong_hours = [
            h for h in track.hours
            if track.lines[h].get("gateStatus", "strong") == "strong"
        ]
        if len(strong_hours) < max(1, int(min_strong_detections)):
            continue
        expected = [h for h in available if track.hours[0] <= h <= track.hours[-1]]
        if len(track.hours) / max(len(expected), 1) < min_coverage:
            continue
        diagnostics = {
            key: _finite_median([
                track.lines[h].get(key, np.nan) for h in track.hours
            ])
            for key in (
                "medianTfpStrength", "medianAbzGradient", "deltaThetaW",
                "deltaThetaE", "deltaTemperature", "deltaThetaV", "dryThermalGradient",
                "thermalAlignment", "windShiftMs", "convergenceMs",
                "thermalContrastFraction", "thermalAlignmentFraction",
                "crossDistanceThermalSupport",
                "windShiftAngleDeg", "windBoundaryFraction",
                "convergenceFraction", "airmassMotionKmh",
                "vorticity1e5", "frontogenesis", "pressureTroughHpa",
                "pressureTroughFraction", "linePressureTendencyHpa3h",
                "coldPressureTendencyHpa3h", "warmPressureTendencyHpa3h",
                "lowerLevelSupport", "deltaThetaW925", "omega700PaS",
                "terrainFraction", "candidateEvidence",
            )
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
        local_classifications = classify_track_locally(
            track, window_hours, wind_sampler
        )
        local_certainties = [
            value.get("classificationCertainty", 0.0)
            for value in local_classifications.values()
            if value.get("frontType") != "uncertain"
        ]
        quality_classification = dict(classification)
        if local_certainties:
            quality_classification["classificationCertainty"] = float(
                np.median(local_certainties)
            )
        quality = _quality_score(track, quality_classification, window_hours)
        diagnoses = [
            str(track.lines[h].get("diagnosis", "synoptic-front"))
            for h in track.hours
        ]
        dominant_diagnosis = max(set(diagnoses), key=diagnoses.count)
        results.append({
            "id": track.id,
            "hours": list(track.hours),
            "lifetimeH": span,
            "lines": {h: track.lines[h]["coordinates"] for h in track.hours},
            "diagnostics": diagnostics,
            "diagnosis": dominant_diagnosis,
            "localClassifications": local_classifications,
            **classification,
            **quality,
        })
    results.sort(key=lambda r: r["qualityScore"], reverse=True)
    return results
