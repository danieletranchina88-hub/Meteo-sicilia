"""Continuous geometric penalties for a candidate front line.

A front is not forbidden from bending, hooking or nearly closing -- a real
occlusion wraps around a low and does exactly that. This module therefore
never vetoes a shape; it produces continuous [0, 1] penalty diagnostics and a
few flags (self-intersection, closed loop, enclosed area) that the physics
layer combines with dynamical support. A closed loop keeps high confidence
only when it is backed by a cyclonic signature; otherwise it is flagged so
the reason codes can mark it CLOSED_LOOP_UNSUPPORTED.

Pure geometry, no meteorology: works on lon/lat polylines, vectorised.
"""

from __future__ import annotations

import numpy as np

EARTH_KM_PER_DEG = 111.32


def _to_km(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    mean_lat = np.deg2rad(float(np.mean(coordinates[:, 1])))
    return np.column_stack((
        coordinates[:, 0] * EARTH_KM_PER_DEG * np.cos(mean_lat),
        coordinates[:, 1] * EARTH_KM_PER_DEG,
    ))


def _turning_angles_deg(points_km: np.ndarray) -> np.ndarray:
    if len(points_km) < 3:
        return np.zeros(0)
    v = np.diff(points_km, axis=0)
    norm = np.hypot(v[:, 0], v[:, 1])
    ok = norm > 1.0e-9
    v = v[ok]
    if len(v) < 2:
        return np.zeros(0)
    u = v / np.hypot(v[:, 0], v[:, 1])[:, None]
    dots = np.clip(np.sum(u[:-1] * u[1:], axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def _segments_intersect(p1, p2, p3, p4) -> bool:
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)
            and ccw(p1, p2, p3) != ccw(p1, p2, p4))


def self_intersection_count(points_km: np.ndarray) -> int:
    """Number of non-adjacent segment pairs that cross.

    Bounded to ~90 vertices before the O(n^2) pair scan: a genuine loop
    survives the subsampling, and the cost stays small on long ICON lines.
    """
    points_km = np.asarray(points_km, dtype=float)
    if len(points_km) > 90:
        idx = np.linspace(0, len(points_km) - 1, 90).astype(int)
        points_km = points_km[idx]
    n = len(points_km) - 1
    if n < 4:
        return 0
    count = 0
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # endpoints touching is not a crossing
            if _segments_intersect(points_km[i], points_km[i + 1],
                                   points_km[j], points_km[j + 1]):
                count += 1
    return count


def _signed_area_km2(points_km: np.ndarray) -> float:
    x, y = points_km[:, 0], points_km[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def geometry_metrics(coordinates: np.ndarray) -> dict:
    """Continuous geometric penalties and shape flags for one line.

    Returns:
      lengthKm, meanTurningDegPerStep, maxTurningDeg, tangentReversals,
      selfIntersections, endpointGapKm, closedLoop (bool),
      enclosedAreaKm2, isoperimetricPenalty, curvaturePenalty in [0,1].
    """
    coordinates = np.asarray(coordinates, dtype=float)
    empty = {
        "lengthKm": 0.0, "meanTurningDegPerStep": 0.0, "maxTurningDeg": 0.0,
        "tangentReversals": 0, "selfIntersections": 0, "endpointGapKm": np.nan,
        "closedLoop": False, "enclosedAreaKm2": 0.0,
        "isoperimetricPenalty": 0.0, "curvaturePenalty": 0.0,
    }
    if coordinates.ndim != 2 or len(coordinates) < 3:
        return empty
    pts = _to_km(coordinates)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    length = float(np.sum(seg))
    angles = _turning_angles_deg(pts)
    mean_turn = float(np.mean(angles)) if len(angles) else 0.0
    max_turn = float(np.max(angles)) if len(angles) else 0.0
    # tangent reversals: sign changes of the cross product of consecutive segs
    v = np.diff(pts, axis=0)
    cross = v[:-1, 0] * v[1:, 1] - v[:-1, 1] * v[1:, 0]
    signs = np.sign(cross[np.abs(cross) > 1.0e-6])
    reversals = int(np.sum(signs[:-1] != signs[1:])) if len(signs) > 1 else 0

    endpoint_gap = float(np.hypot(*(pts[0] - pts[-1])))
    closed = endpoint_gap < 0.25 * length and length > 0
    area = abs(_signed_area_km2(np.vstack((pts, pts[0]))))
    # isoperimetric: a compact loop (area large for its perimeter) is more
    # "blob-like" than "front-like". 4*pi*A/P^2 -> 1 for a circle, ~0 for a line
    iso = (4.0 * np.pi * area / (length * length)) if length > 0 else 0.0
    iso_penalty = float(np.clip(iso, 0.0, 1.0)) if closed else 0.0
    # curvature penalty: grows with mean turning and reversals per unit length
    turn_density = mean_turn / 25.0
    reversal_density = reversals / max(len(v), 1) / 0.5
    curvature_penalty = float(np.clip(
        0.6 * turn_density + 0.4 * reversal_density, 0.0, 1.0
    ))
    # Self-intersection is the only O(n^2) step: skip it for lines that
    # cannot plausibly cross themselves (open and low-curvature). A straight
    # or gently-curved open front never self-intersects.
    could_cross = closed or curvature_penalty > 0.30 or reversals >= 3
    intersections = self_intersection_count(pts) if could_cross else 0
    return {
        "lengthKm": round(length, 1),
        "meanTurningDegPerStep": round(mean_turn, 2),
        "maxTurningDeg": round(max_turn, 2),
        "tangentReversals": reversals,
        "selfIntersections": intersections,
        "endpointGapKm": round(endpoint_gap, 1),
        "closedLoop": bool(closed),
        "enclosedAreaKm2": round(area, 1),
        "isoperimetricPenalty": round(iso_penalty, 3),
        "curvaturePenalty": round(curvature_penalty, 3),
    }
