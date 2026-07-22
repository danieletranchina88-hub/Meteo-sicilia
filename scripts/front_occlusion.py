"""Occlusion detection for tracked fronts (documento sez. 14).

An occlusion is not decided from a single-hour theta-w gradient. It is the
topological situation in which a **cold front catches up to the warm front
around a common surface low**: the two boundaries meet at a triple point and
the segment that wraps from the triple point toward the low centre is the
occluded front.

This module works on the *already published* fronts of one hour plus the
mean-sea-level pressure field, so it uses spatial context (a real pressure
minimum, two fronts anchored to it, a narrow warm sector) rather than a local
gradient trick. It is deliberately conservative: when the configuration is not
clearly an occluding wave, nothing is relabelled.
"""

from __future__ import annotations

import numpy as np

EARTH_KM_PER_DEG = 111.32


def _project_km(coordinates: np.ndarray, mean_lat_rad: float) -> np.ndarray:
    scale_lon = EARTH_KM_PER_DEG * np.cos(mean_lat_rad)
    return np.column_stack((coordinates[:, 0] * scale_lon,
                            coordinates[:, 1] * EARTH_KM_PER_DEG))


def _min_pair_distance_km(line_a: np.ndarray, line_b: np.ndarray):
    """Closest approach between two lon/lat lines: (km, index_a, index_b)."""
    mean_lat = np.deg2rad(float(np.mean(
        np.concatenate((line_a[:, 1], line_b[:, 1]))
    )))
    a = _project_km(line_a, mean_lat)
    b = _project_km(line_b, mean_lat)
    dist = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
    return float(dist[i, j]), int(i), int(j)


def _haversine_km(lon0, lat0, lon_grid, lat_grid):
    mean_lat = np.deg2rad(lat0)
    dx = (lon_grid - lon0) * EARTH_KM_PER_DEG * np.cos(mean_lat)
    dy = (lat_grid - lat0) * EARTH_KM_PER_DEG
    return np.hypot(dx, dy)


def find_low_centers(
    pressure_hpa: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    *,
    min_prominence_hpa: float = 2.0,
    neighbourhood_km: float = 300.0,
) -> list[dict]:
    """Local MSLP minima that are genuine lows (prominence over the ring)."""
    if pressure_hpa is None:
        return []
    p = pressure_hpa
    ny, nx = p.shape
    if ny < 3 or nx < 3:
        return []
    # Vectorised local minima against the 8 neighbours.
    core = p[1:-1, 1:-1]
    is_min = np.isfinite(core)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            is_min &= core <= p[1 + dy:ny - 1 + dy, 1 + dx:nx - 1 + dx]
    # Box radius in grid cells that approximates the physical neighbourhood.
    mean_lat = float(np.mean(latitudes))
    dlat_km = abs(latitudes[1] - latitudes[0]) * EARTH_KM_PER_DEG
    dlon_km = abs(longitudes[1] - longitudes[0]) * EARTH_KM_PER_DEG * np.cos(
        np.deg2rad(mean_lat)
    )
    ry = max(1, int(neighbourhood_km / max(dlat_km, 1.0e-6)))
    rx = max(1, int(neighbourhood_km / max(dlon_km, 1.0e-6)))
    lows: list[dict] = []
    ys, xs = np.where(is_min)
    for iy, ix in zip(ys + 1, xs + 1):
        value = float(p[iy, ix])
        y0, y1 = max(0, iy - ry), min(ny, iy + ry + 1)
        x0, x1 = max(0, ix - rx), min(nx, ix + rx + 1)
        ring = p[y0:y1, x0:x1]
        prominence = float(np.nanmedian(ring) - value)
        if prominence >= min_prominence_hpa:
            lows.append({
                "lon": float(longitudes[ix]), "lat": float(latitudes[iy]),
                "pressure": value, "prominence": round(prominence, 2),
            })
    # Deduplicate nearby minima, keep the deepest.
    lows.sort(key=lambda low: low["pressure"])
    unique: list[dict] = []
    for low in lows:
        if all(
            _haversine_km(low["lon"], low["lat"],
                          np.array([o["lon"]]), np.array([o["lat"]]))[0]
            > neighbourhood_km * 0.6
            for o in unique
        ):
            unique.append(low)
    return unique


def detect_occlusion(
    features: list[dict],
    pressure_hpa: np.ndarray | None,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    *,
    anchor_km: float = 220.0,
    junction_km: float = 170.0,
    min_wrap_km: float = 70.0,
) -> list[dict]:
    """Relabel the wrapped segment of an occluding wave as ``occluded``.

    ``features`` is a list of dicts with keys ``coordinates`` (Nx2 lon/lat)
    and ``frontType``. Returns a list of occlusion descriptors; each has the
    index of the feature that carries the occluded segment, the point index
    range of that segment, and the anchoring low. Nothing is returned unless a
    cold and a warm front clearly meet at a triple point near a real low.
    """
    if pressure_hpa is None or len(features) < 2:
        return []
    lows = find_low_centers(pressure_hpa, longitudes, latitudes)
    if not lows:
        return []

    cold = [(i, np.asarray(f["coordinates"], dtype=float))
            for i, f in enumerate(features) if f.get("frontType") == "cold"]
    warm = [(i, np.asarray(f["coordinates"], dtype=float))
            for i, f in enumerate(features) if f.get("frontType") == "warm"]
    if not cold or not warm:
        return []

    occlusions: list[dict] = []
    used_features: set[int] = set()
    for low in lows:
        lon0, lat0 = low["lon"], low["lat"]
        # A front is anchored to the low if any of its points is within
        # anchor_km of the centre.
        def anchored(line):
            return np.min(_haversine_km(lon0, lat0, line[:, 0], line[:, 1])) <= anchor_km

        cold_here = [(i, ln) for i, ln in cold if anchored(ln) and i not in used_features]
        warm_here = [(i, ln) for i, ln in warm if anchored(ln) and i not in used_features]
        if not cold_here or not warm_here:
            continue

        for ci, cold_line in cold_here:
            for wi, warm_line in warm_here:
                gap_km, ic, iw = _min_pair_distance_km(cold_line, warm_line)
                if gap_km > junction_km:
                    continue
                triple = 0.5 * (cold_line[ic] + warm_line[iw])
                # Occlusion, not an open wave: the triple point must be
                # displaced OUT from the low, with the cold front wrapping from
                # the triple point back to the low centre. In an open wave both
                # fronts start at the low and there is no wrapped branch.
                triple_low_km = float(_haversine_km(
                    lon0, lat0, np.array([triple[0]]), np.array([triple[1]])
                )[0])
                if not (min_wrap_km <= triple_low_km <= anchor_km):
                    continue
                d_low = _haversine_km(lon0, lat0, cold_line[:, 0], cold_line[:, 1])
                near_low_idx = int(np.argmin(d_low))
                if float(d_low[near_low_idx]) > junction_km:
                    continue  # the cold front does not actually reach the low
                lo, hi = sorted((ic, near_low_idx))
                if hi - lo < 2:
                    continue  # wrapped branch too short to draw
                wrap_km = float(np.sum(np.hypot(*np.diff(
                    _project_km(cold_line[lo:hi + 1],
                                np.deg2rad(lat0)), axis=0).T)))
                if wrap_km < min_wrap_km:
                    continue
                occlusions.append({
                    "featureIndex": ci,
                    "segment": (lo, hi),
                    "tripleIndex": ic,
                    "lowIndex": near_low_idx,
                    "low": low,
                    "triplePoint": [round(float(triple[0]), 3),
                                    round(float(triple[1]), 3)],
                    "wrapKm": round(wrap_km, 0),
                    "junctionKm": round(gap_km, 0),
                })
                used_features.add(ci)
                break
            if ci in used_features:
                break
    return occlusions
