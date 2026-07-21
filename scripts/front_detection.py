"""Two-scale synoptic front detection (v10 core, phase 2).

The synoptic scale decides *whether and where* a frontal structure exists;
ICON-2I decides *the exact geometry* inside that structure.  The synoptic
scale is used as a **structural prior**, not as an a-posteriori geometric
filter:

    theta_w
      -> STRONG smoothing  (~150 km) -> synoptic locator -> corridors
      -> LIGHT smoothing   (~50 km)  -> refined locator   -> candidates
      -> keep the refined candidates that lie inside a synoptic corridor
         (the refined geometry replaces the coarse one - ICON refines it)

Refined candidates with no synoptic support are dropped.  A very strong
gradient is not, by itself, evidence of a synoptic front: sea-breeze lines,
convective outflows and the rim of an orographic air pool can all be very
sharp.  Persistence and intensity therefore never bypass the synoptic-scale
prior.

The corridor comes from a strongly smoothed version of the SAME model: it
is a scale prior, not independent evidence and not a comparison with ECMWF.
"""

from __future__ import annotations

import numpy as np

import front_locator as fl

EARTH_KM_PER_DEG = fl.EARTH_KM_PER_DEG


def _resample_km(coordinates: np.ndarray, step_km: float = 20.0) -> np.ndarray:
    """Resample a lon/lat polyline to roughly ``step_km`` spacing."""
    length = fl._line_length_km(coordinates)
    count = max(2, int(length / step_km))
    deltas = np.diff(coordinates, axis=0)
    seg = np.hypot(deltas[:, 0], deltas[:, 1])
    cumulative = np.concatenate(([0.0], np.cumsum(seg)))
    total = max(cumulative[-1], 1.0e-9)
    targets = np.linspace(0.0, total, count)
    return np.column_stack((
        np.interp(targets, cumulative, coordinates[:, 0]),
        np.interp(targets, cumulative, coordinates[:, 1]),
    ))


def _points_to_segments_km(points_km: np.ndarray, line_km: np.ndarray) -> np.ndarray:
    """Min distance (km) from each point to the segments of a polyline."""
    if len(line_km) == 1:
        return np.hypot(points_km[:, 0] - line_km[0, 0],
                        points_km[:, 1] - line_km[0, 1])
    starts, ends = line_km[:-1], line_km[1:]
    seg = ends - starts
    seg_len_sq = np.maximum((seg ** 2).sum(axis=1), 1.0e-9)
    rel = points_km[:, None, :] - starts[None, :, :]
    t = np.clip((rel * seg[None, :, :]).sum(axis=2) / seg_len_sq[None, :], 0.0, 1.0)
    proj = starts[None, :, :] + t[:, :, None] * seg[None, :, :]
    dist = np.hypot(points_km[:, None, 0] - proj[:, :, 0],
                    points_km[:, None, 1] - proj[:, :, 1])
    return dist.min(axis=1)


def _synoptic_support(
    refined: np.ndarray, synoptic_lines: list[np.ndarray], corridor_km: float
) -> float:
    """Fraction of a refined line lying within ``corridor_km`` of any synoptic line."""
    if not synoptic_lines:
        return 0.0
    mean_lat = np.deg2rad(float(np.mean(refined[:, 1])))
    scale_lon = EARTH_KM_PER_DEG * np.cos(mean_lat)
    dense = _resample_km(refined, 20.0)
    points_km = np.column_stack((dense[:, 0] * scale_lon, dense[:, 1] * EARTH_KM_PER_DEG))
    best = np.full(len(points_km), np.inf)
    for line in synoptic_lines:
        line_km = np.column_stack((line[:, 0] * scale_lon, line[:, 1] * EARTH_KM_PER_DEG))
        best = np.minimum(best, _points_to_segments_km(points_km, line_km))
    return float(np.mean(best <= corridor_km))


def _shape_metrics(coordinates: np.ndarray) -> tuple[float, float]:
    """Return sinuosity and total turning of an open synoptic line."""
    points = _resample_km(np.asarray(coordinates, dtype=float), 30.0)
    if len(points) < 3:
        return 1.0, 0.0
    mean_lat = np.deg2rad(float(np.mean(points[:, 1])))
    projected = np.column_stack((
        points[:, 0] * EARTH_KM_PER_DEG * np.cos(mean_lat),
        points[:, 1] * EARTH_KM_PER_DEG,
    ))
    segments = np.diff(projected, axis=0)
    lengths = np.hypot(segments[:, 0], segments[:, 1])
    path_length = float(np.sum(lengths))
    endpoint = float(np.hypot(*(projected[-1] - projected[0])))
    sinuosity = path_length / max(endpoint, 1.0)
    headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    net_turn = abs(float(np.degrees(headings[-1] - headings[0])))
    net_turn = min(net_turn % 360.0, 360.0 - (net_turn % 360.0))
    return sinuosity, net_turn


def detect_fronts_two_scale(
    theta_w: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    *,
    synoptic_sigma_km: float = 150.0,
    refine_sigma_km: float = 50.0,
    derivative_sigma_km: float = 20.0,
    corridor_km: float = 180.0,
    min_synoptic_support: float = 0.5,
    synoptic_min_length_km: float = 400.0,
    refine_min_length_km: float = 250.0,
    abz_gradient_threshold: float = 1.5,
    return_synoptic: bool = False,
):
    """Two-scale detection: refined candidates constrained by a synoptic prior.

    Returns the list of final candidates (refined geometry, each with a
    ``synopticSupport`` fraction and a ``corroborated`` flag).  With
    ``return_synoptic`` also returns the synoptic corridor lines.
    """
    synoptic = fl.locate_fronts(
        theta_w, longitudes, latitudes,
        synoptic_sigma_km=synoptic_sigma_km,
        derivative_sigma_km=max(derivative_sigma_km, synoptic_sigma_km * 0.3),
        abz_gradient_threshold=abz_gradient_threshold * 0.7,
        min_length_km=synoptic_min_length_km,
    )
    refined = fl.locate_fronts(
        theta_w, longitudes, latitudes,
        synoptic_sigma_km=refine_sigma_km,
        derivative_sigma_km=derivative_sigma_km,
        abz_gradient_threshold=abz_gradient_threshold,
        min_length_km=refine_min_length_km,
    )
    synoptic_lines = [c["coordinates"] for c in synoptic]

    final = []
    for candidate in refined:
        sinuosity, net_turn = _shape_metrics(candidate["coordinates"])
        if sinuosity > 1.8 or net_turn > 145.0:
            continue
        candidate["sinuosity"] = round(sinuosity, 2)
        candidate["netTurnDeg"] = round(net_turn, 1)
        support = _synoptic_support(candidate["coordinates"], synoptic_lines, corridor_km)
        candidate["synopticSupport"] = round(support, 2)
        if support >= min_synoptic_support:
            candidate["corroborated"] = True
            final.append(candidate)
        # Nessuna eccezione basata sulla sola intensita': un confine locale
        # puo' essere piu' netto di un fronte sinottico autentico.

    final.sort(key=lambda c: c["lengthKm"], reverse=True)
    if return_synoptic:
        return final, synoptic
    return final
