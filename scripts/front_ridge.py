"""Support-guided line extraction: least-cost ridge path (Fase C).

Biard & Kunkel (2019) extract a front line from the ridge of a continuous
support field. This module does the same **subordinated to the physics**: it
does not invent a line anywhere the support is high, it refines an existing
TFL-TFP-ABZ candidate by finding, *inside a corridor around that candidate*,
the path that best follows the crest of ``any_front_support`` (Fase B).

The path is the minimum-cost route on a grid graph whose per-cell cost is
``-log(support + eps)`` plus explicit penalties (terrain, domain edge, missing
data). Staying on high support is cheap; crossing a support gap, terrain, or
the domain edge is expensive. Diagonal steps cost sqrt(2). Endpoints are the
candidate's own ends, snapped into the corridor.

This is built **side by side** with the current contour geometry: it returns a
refined polyline for comparison and is NOT wired into the published analyzer
until a benchmark shows it is at least as good. Uses only numpy + scipy
(already dependencies); no scikit-image, no machine learning.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

EARTH_KM_PER_DEG = 111.32
_EPS = 1.0e-3
_NEIGHBOURS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, np.sqrt(2.0)), (-1, 1, np.sqrt(2.0)),
    (1, -1, np.sqrt(2.0)), (1, 1, np.sqrt(2.0)),
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
    # nearest distance from each cell to any densified line point (km)
    dx = (lon_grid[..., None] - dense[:, 0]) * EARTH_KM_PER_DEG * np.cos(mean_lat)
    dy = (lat_grid[..., None] - dense[:, 1]) * EARTH_KM_PER_DEG
    distance = np.sqrt(np.min(dx * dx + dy * dy, axis=-1))
    return distance <= corridor_km, distance


def _nearest_masked(mask: np.ndarray, target_rc: tuple[int, int]) -> tuple[int, int]:
    cells = np.argwhere(mask)
    if not len(cells):
        return target_rc
    d = np.hypot(cells[:, 0] - target_rc[0], cells[:, 1] - target_rc[1])
    r, c = cells[int(np.argmin(d))]
    return int(r), int(c)


def least_cost_path(cost, mask, start_rc, end_rc):
    """Minimum-cost 8-connected path between two cells through ``mask``.

    ``cost`` is a per-cell non-negative cost; edges are the average of the two
    cell costs times the step length. Returns an (r, c) array, or None.
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
    for k, (r, c) in enumerate(cells):
        base = cost[r, c]
        for dr, dc, step in _NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < ny and 0 <= cc < nx and mask[rr, cc]:
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
):
    """Refine one candidate line to the crest of any_front_support.

    Returns the refined lon/lat polyline (or the input line unchanged when the
    corridor is too small to route). The published geometry is NOT replaced by
    this yet; it is produced for side-by-side comparison.
    """
    lon = np.asarray(longitudes, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    line = np.asarray(line, dtype=float)
    if len(line) < 2:
        return line

    support = np.asarray(support_result["any_front_support"], dtype=float)
    penalties = support_result.get("penalties", {})
    terrain = np.asarray(penalties.get("terrain", np.zeros_like(support)), float)
    edge = np.asarray(penalties.get("edge", np.zeros_like(support)), float)

    mask, _ = corridor_mask(lon, lat, line, corridor_km)
    if np.count_nonzero(mask) < 4:
        return line

    # Non-negative cost: 0 on full support (S=1), growing as support drops.
    # Dijkstra requires non-negative edge weights.
    cost = np.log((1.0 + _EPS) / (np.clip(support, 0.0, 1.0) + _EPS))
    cost = cost + terrain_weight * terrain + edge_weight * edge
    cost = np.where(np.isfinite(cost), cost, off_corridor_cost)

    def to_rc(point):
        c = int(np.clip(np.searchsorted(lon, point[0]), 0, len(lon) - 1))
        r = int(np.clip(np.searchsorted(lat, point[1]), 0, len(lat) - 1))
        return r, c

    path = least_cost_path(cost, mask, to_rc(line[0]), to_rc(line[-1]))
    if path is None or len(path) < 2:
        return line
    refined = np.column_stack((lon[path[:, 1]], lat[path[:, 0]]))
    # The 8-connected least-cost path is optimal but stair-stepped: it can only
    # turn in 45/90-degree steps, so the raw polyline is jagged even when it
    # tracks the crest. Chaikin corner-cutting removes the grid staircase
    # (orientation-independent, endpoints preserved) before decluttering.
    refined = _chaikin(refined, iterations=3)
    return fl_rdp(refined)


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
