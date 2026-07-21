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

Refined candidates with no synoptic support are dropped, EXCEPT
exceptionally strong ones, which are kept but flagged
``synopticSupport = 0`` and ``corroborated = False`` (persistence, which
would fully justify such a candidate, is added in phase 3).

Note on independence: the corridor here comes from a strongly smoothed
version of the SAME model, so it is a scale prior, not an independent
model.  Cross-model agreement (ECMWF) remains a separate, weaker
corroboration handled elsewhere.
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
    exceptional_gradient_factor: float = 2.5,
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

    strong = abz_gradient_threshold * exceptional_gradient_factor
    final = []
    for candidate in refined:
        support = _synoptic_support(candidate["coordinates"], synoptic_lines, corridor_km)
        candidate["synopticSupport"] = round(support, 2)
        if support >= min_synoptic_support:
            candidate["corroborated"] = True
            final.append(candidate)
        elif candidate["medianThetaWGradient"] >= strong:
            # Eccezionalmente forte ma senza struttura sinottica: tenuto ma
            # marcato non corroborato (la persistenza, in fase 3, decidera'
            # se e' un fronte reale o un artefatto mesoscalare).
            candidate["corroborated"] = False
            final.append(candidate)
        # altrimenti scartato: nessuna struttura sinottica, non eccezionale.

    final.sort(key=lambda c: c["lengthKm"], reverse=True)
    if return_synoptic:
        return final, synoptic
    return final
