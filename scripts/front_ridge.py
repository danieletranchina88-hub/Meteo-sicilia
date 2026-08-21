"""Support-guided line extraction: least-cost ridge path (Fase C).

Biard & Kunkel (2019) extract a front line from the ridge of a continuous
support field. This module does the same **subordinated to the physics**: it
does not invent a line anywhere the support is high, it refines an existing
TFL-TFP-ABZ candidate by finding, *inside a corridor around that candidate*,
the path that best follows the thermodynamic crest of ``geometry_support``
(with ``any_front_support`` as a backward-compatible fallback).

The path is the minimum-cost route on a metric grid graph whose per-cell cost
is ``-log(support + eps)`` plus explicit penalties (terrain, domain edge,
missing data) and a soft fidelity term to the original TFL contour. Staying on
high support is cheap; crossing a support gap, terrain, or the domain edge is
expensive. Step costs use real kilometres, not pixel geometry. Endpoints are
locally snapped to supported cells so an off-ridge endpoint cannot create a
long artificial spur.

The published analyzer uses the refined line only after the regression guard
confirms physical support, bounded displacement, plausible length and clean
topology; otherwise it keeps the original contour. Uses only numpy + scipy
(already dependencies); no scikit-image, no machine learning.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

EARTH_KM_PER_DEG = 111.32
_EPS = 1.0e-3
_NEIGHBOURS = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)


def _resample_km(line: np.ndarray, step_km: float = 15.0) -> np.ndarray:
    line = np.asarray(line, dtype=float)
    if len(line) < 2:
        return line
    mean_lat = np.deg2rad(float(np.mean(line[:, 1])))
    seg = np.hypot(
        np.diff(line[:, 0]) * EARTH_KM_PER_DEG * np.cos(mean_lat),
        np.diff(line[:, 1]) * EARTH_KM_PER_DEG,
    )
    cumulative = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cumulative[-1])
    if total <= 0.0:
        return line
    targets = np.arange(0.0, total + step_km, step_km)
    return np.column_stack((
        np.interp(targets, cumulative, line[:, 0]),
        np.interp(targets, cumulative, line[:, 1]),
    ))


def corridor_mask(longitudes, latitudes, line, corridor_km, metrics=None):
    """Boolean grid mask of cells within ``corridor_km`` of a lon/lat line."""
    lon = np.asarray(longitudes, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    mean_lat = np.deg2rad(float(np.mean(lat)))
    dense = _resample_km(line, step_km=max(5.0, corridor_km * 0.25))
    # Nearest distance from each cell to the densified line. A KD-tree avoids
    # the old (ny, nx, n_line) temporary arrays, which could exceed 500 MB on
    # the complete 761x761 ICON grid for a long front.
    scale_lon = EARTH_KM_PER_DEG * np.cos(mean_lat)
    line_km = np.column_stack((
        dense[:, 0] * scale_lon, dense[:, 1] * EARTH_KM_PER_DEG,
    ))
    grid_km = np.column_stack((
        lon_grid.ravel() * scale_lon,
        lat_grid.ravel() * EARTH_KM_PER_DEG,
    ))
    distance = cKDTree(line_km).query(grid_km, k=1, workers=1)[0]
    distance = distance.reshape(lon_grid.shape)
    return distance <= corridor_km, distance


def _nearest_masked(mask: np.ndarray, target_rc: tuple[int, int]) -> tuple[int, int]:
    cells = np.argwhere(mask)
    if not len(cells):
        return target_rc
    d = np.hypot(cells[:, 0] - target_rc[0], cells[:, 1] - target_rc[1])
    r, c = cells[int(np.argmin(d))]
    return int(r), int(c)


def _snap_endpoint(
    cost: np.ndarray,
    mask: np.ndarray,
    target_rc: tuple[int, int],
    dx_km_col: np.ndarray,
    dy_km: float,
    radius_km: float,
) -> tuple[int, int]:
    """Choose a nearby supported endpoint without changing frontal extent."""
    cells = np.argwhere(mask)
    if not len(cells):
        return target_rc
    row = int(np.clip(target_rc[0], 0, mask.shape[0] - 1))
    dx = float(np.asarray(dx_km_col).ravel()[row])
    distance = np.hypot(
        (cells[:, 1] - target_rc[1]) * dx,
        (cells[:, 0] - target_rc[0]) * float(dy_km),
    )
    local = distance <= max(float(radius_km), max(dx, float(dy_km)))
    if not np.any(local):
        return _nearest_masked(mask, target_rc)
    choices = cells[local]
    local_distance = distance[local]
    # A one-radius displacement costs 0.75: enough to keep the original
    # extent, while a terrain/edge penalty can still move an endpoint away.
    score = cost[choices[:, 0], choices[:, 1]] + 0.75 * (
        local_distance / max(float(radius_km), 1.0)
    ) ** 2
    r, c = choices[int(np.argmin(score))]
    return int(r), int(c)


def least_cost_path(
    cost, mask, start_rc, end_rc, *, dx_km_col=None, dy_km=None
):
    """Minimum-cost 8-connected path between two cells through ``mask``.

    ``cost`` is a per-cell non-negative cost; edges are the average of the two
    cell costs times their physical step length. Returns an (r, c) array, or
    None. Unit pixel steps remain available for direct/legacy callers.
    """
    ny, nx = cost.shape
    index = -np.ones((ny, nx), dtype=int)
    cells = np.argwhere(mask)
    if len(cells) < 2:
        return None
    index[cells[:, 0], cells[:, 1]] = np.arange(len(cells))
    start_rc = start_rc if mask[start_rc] else _nearest_masked(mask, start_rc)
    end_rc = end_rc if mask[end_rc] else _nearest_masked(mask, end_rc)

    rows, cols, data = [], [], []
    dx_rows = (
        np.asarray(dx_km_col, dtype=float).ravel()
        if dx_km_col is not None else None
    )
    for k, (r, c) in enumerate(cells):
        base = cost[r, c]
        for dr, dc in _NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < ny and 0 <= cc < nx and mask[rr, cc]:
                if dx_km_col is None or dy_km is None:
                    step = np.hypot(dr, dc)
                else:
                    local_dx = 0.5 * (dx_rows[r] + dx_rows[rr])
                    step = np.hypot(dc * local_dx, dr * float(dy_km))
                rows.append(k)
                cols.append(index[rr, cc])
                data.append(0.5 * (base + cost[rr, cc]) * step)
    graph = csr_matrix((data, (rows, cols)), shape=(len(cells), len(cells)))
    source = int(index[start_rc])
    target = int(index[end_rc])
    _, predecessors = dijkstra(graph, indices=source, return_predecessors=True)
    path = []
    current = target
    guard = 0
    while current != source and current >= 0 and guard < len(cells) + 1:
        path.append(cells[current])
        current = predecessors[current]
        guard += 1
    if current != source:
        return None
    path.append(cells[source])
    path.reverse()
    return np.asarray(path)


def refine_line(
    support_result: dict,
    longitudes,
    latitudes,
    line,
    *,
    corridor_km: float = 120.0,
    terrain_weight: float = 2.5,
    edge_weight: float = 3.0,
    off_corridor_cost: float = 25.0,
    reference_weight: float = 1.35,
    endpoint_radius_km: float = 45.0,
):
    """Refine one candidate line to the crest of any_front_support.

    Returns the refined lon/lat polyline, or the input unchanged when routing
    fails or the result does not pass the geometry regression guard.
    """
    lon = np.asarray(longitudes, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    line = np.asarray(line, dtype=float)
    if len(line) < 2:
        return line

    support = np.asarray(
        support_result.get(
            "geometry_support", support_result["any_front_support"]
        ),
        dtype=float,
    )
    penalties = support_result.get("penalties", {})
    terrain = np.asarray(penalties.get("terrain", np.zeros_like(support)), float)
    edge = np.asarray(penalties.get("edge", np.zeros_like(support)), float)

    mask, distance_to_reference = corridor_mask(
        lon, lat, line, corridor_km
    )
    if np.count_nonzero(mask) < 4:
        return line

    # Non-negative cost: 0 on full support (S=1), growing as support drops.
    # Dijkstra requires non-negative edge weights.
    cost = np.log((1.0 + _EPS) / (np.clip(support, 0.0, 1.0) + _EPS))
    cost = cost + terrain_weight * terrain + edge_weight * edge
    cost = cost + reference_weight * np.square(
        distance_to_reference / max(float(corridor_km), 1.0)
    )
    cost = np.where(np.isfinite(cost), cost, off_corridor_cost)

    dx_km_col = (
        np.abs(lon[1] - lon[0]) * EARTH_KM_PER_DEG
        * np.cos(np.deg2rad(lat))
    )[:, None]
    dy_km = float(np.abs(lat[1] - lat[0]) * EARTH_KM_PER_DEG)

    def to_rc(point):
        c = int(np.argmin(np.abs(lon - point[0])))
        r = int(np.argmin(np.abs(lat - point[1])))
        return r, c

    start = _snap_endpoint(
        cost, mask, to_rc(line[0]), dx_km_col, dy_km, endpoint_radius_km
    )
    end = _snap_endpoint(
        cost, mask, to_rc(line[-1]), dx_km_col, dy_km, endpoint_radius_km
    )
    path = least_cost_path(
        cost, mask, start, end, dx_km_col=dx_km_col, dy_km=dy_km
    )
    if path is None or len(path) < 2:
        return line
    refined = np.column_stack((lon[path[:, 1]], lat[path[:, 0]]))
    # The 8-connected least-cost path is optimal but stair-stepped: it can only
    # turn in 45/90-degree steps, so the raw polyline is jagged even when it
    # tracks the crest. Chaikin corner-cutting removes the grid staircase
    # (orientation-independent, endpoints preserved) before decluttering.
    refined = _chaikin(refined, iterations=3)
    refined = fl_rdp(refined)
    return refined if _geometry_is_safe(
        line, refined, support, lon, lat, corridor_km,
        reference_distance=distance_to_reference,
    ) else line


def _sample_nearest(field, longitudes, latitudes, line) -> np.ndarray:
    """Cheap support sampling used only by the geometry regression guard."""
    line = _resample_km(np.asarray(line, dtype=float), step_km=15.0)
    cols = np.abs(longitudes[None, :] - line[:, 0, None]).argmin(axis=1)
    rows = np.abs(latitudes[None, :] - line[:, 1, None]).argmin(axis=1)
    return np.asarray(field, dtype=float)[rows, cols]


def _has_self_intersection(line: np.ndarray) -> bool:
    """Return True for any non-adjacent crossing or collinear overlap."""
    points = np.asarray(line, dtype=float)
    if len(points) < 4:
        return False

    def orientation(a, b, c):
        ab, ac = b - a, c - a
        return float(ab[0] * ac[1] - ab[1] * ac[0])

    def on_segment(a, b, p, tolerance=1.0e-10):
        return (
            min(a[0], b[0]) - tolerance <= p[0]
            <= max(a[0], b[0]) + tolerance
            and min(a[1], b[1]) - tolerance <= p[1]
            <= max(a[1], b[1]) + tolerance
        )

    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        for j in range(i + 2, len(points) - 1):
            c, d = points[j], points[j + 1]
            o1, o2 = orientation(a, b, c), orientation(a, b, d)
            o3, o4 = orientation(c, d, a), orientation(c, d, b)
            if o1 * o2 < 0.0 and o3 * o4 < 0.0:
                return True
            tolerance = 1.0e-10
            if (
                abs(o1) <= tolerance and on_segment(a, b, c)
                or abs(o2) <= tolerance and on_segment(a, b, d)
                or abs(o3) <= tolerance and on_segment(c, d, a)
                or abs(o4) <= tolerance and on_segment(c, d, b)
            ):
                return True
    return False


def _maximum_turn_deg(line: np.ndarray) -> float:
    """Largest local heading change after metric, uniform resampling."""
    points = _resample_km(np.asarray(line, dtype=float), step_km=20.0)
    if len(points) < 3:
        return 0.0
    mean_lat = np.deg2rad(float(np.mean(points[:, 1])))
    metric = np.column_stack((
        points[:, 0] * EARTH_KM_PER_DEG * np.cos(mean_lat),
        points[:, 1] * EARTH_KM_PER_DEG,
    ))
    vectors = np.diff(metric, axis=0)
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    valid = lengths > 1.0e-6
    vectors = vectors[valid] / lengths[valid, None]
    if len(vectors) < 2:
        return 0.0
    dot = np.sum(vectors[:-1] * vectors[1:], axis=1)
    return float(np.max(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))))


def _geometry_is_safe(
    original, refined, support, longitudes, latitudes, corridor_km,
    *, reference_distance=None,
) -> bool:
    """Reject shortcuts, loops and refinements that lose physical support."""
    original = np.asarray(original, dtype=float)
    refined = np.asarray(refined, dtype=float)
    if (
        len(refined) < 2
        or _has_self_intersection(refined)
        or _maximum_turn_deg(refined) > 135.0
    ):
        return False
    original_length = _line_length_km(original)
    refined_length = _line_length_km(refined)
    ratio = refined_length / max(original_length, 1.0)
    if not 0.62 <= ratio <= 1.45:
        return False
    if reference_distance is None:
        _, displacement = corridor_mask(
            longitudes, latitudes, original, corridor_km
        )
    else:
        displacement = np.asarray(reference_distance, dtype=float)
    cols = np.abs(longitudes[None, :] - refined[:, 0, None]).argmin(axis=1)
    rows = np.abs(latitudes[None, :] - refined[:, 1, None]).argmin(axis=1)
    if float(np.max(displacement[rows, cols])) > corridor_km * 1.02:
        return False
    before = _sample_nearest(support, longitudes, latitudes, original)
    after = _sample_nearest(support, longitudes, latitudes, refined)
    before_med = (
        float(np.nanmedian(before)) if np.any(np.isfinite(before)) else 0.0
    )
    after_med = (
        float(np.nanmedian(after)) if np.any(np.isfinite(after)) else 0.0
    )
    return after_med + 0.04 >= before_med


def _line_length_km(line: np.ndarray) -> float:
    """Geodesic-enough length for a limited-area lon/lat domain."""
    line = np.asarray(line, dtype=float)
    if len(line) < 2:
        return 0.0
    mean_lat = np.deg2rad(0.5 * (line[:-1, 1] + line[1:, 1]))
    dx = np.diff(line[:, 0]) * EARTH_KM_PER_DEG * np.cos(mean_lat)
    dy = np.diff(line[:, 1]) * EARTH_KM_PER_DEG
    return float(np.sum(np.hypot(dx, dy)))


def _chaikin(points: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Chaikin corner-cutting smoothing, keeping the two endpoints fixed."""
    points = np.asarray(points, dtype=float)
    for _ in range(max(0, iterations)):
        if len(points) < 3:
            break
        p0, p1 = points[:-1], points[1:]
        q = 0.75 * p0 + 0.25 * p1
        r = 0.25 * p0 + 0.75 * p1
        cut = np.empty((2 * len(q), 2), dtype=float)
        cut[0::2], cut[1::2] = q, r
        points = np.vstack((points[0], cut, points[-1]))
    return points


def fl_rdp(points: np.ndarray, tolerance_deg: float = 0.05) -> np.ndarray:
    """Small Ramer-Douglas-Peucker to declutter the stair-stepped path."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    line = end - start
    length = np.hypot(*line)
    if length < 1.0e-9:
        return np.vstack((start, end))
    rel = points - start
    proj = np.clip((rel @ line) / (length * length), 0.0, 1.0)
    perp = np.hypot(*(rel - np.outer(proj, line)).T)
    index = int(np.argmax(perp))
    if perp[index] > tolerance_deg:
        left = fl_rdp(points[:index + 1], tolerance_deg)
        right = fl_rdp(points[index:], tolerance_deg)
        return np.vstack((left[:-1], right))
    return np.vstack((start, end))
