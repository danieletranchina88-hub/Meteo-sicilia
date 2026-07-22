"""Multi-scale structural filter for kilometre-scale ICON-2I fields.

The synoptic scale decides *whether and where* a frontal structure exists;
ICON-2I decides *the exact geometry* inside that structure.  The synoptic
scale is used as a **structural prior**, not as an a-posteriori geometric
filter:

    theta_w
      -> STRONG smoothing  (~100 km) -> synoptic locator -> corridors
      -> LIGHT smoothing   (~45 km)  -> refined locator   -> candidates
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


def line_support_fraction(
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


def _shape_metrics(coordinates: np.ndarray) -> tuple[float, float, float, float]:
    """Return sinuosity, net turn, closure ratio and accumulated turn."""
    points = _resample_km(np.asarray(coordinates, dtype=float), 30.0)
    if len(points) < 3:
        return 1.0, 0.0, 1.0, 0.0
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
    closure_ratio = endpoint / max(path_length, 1.0)
    headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    net_turn = abs(float(np.degrees(headings[-1] - headings[0])))
    net_turn = min(net_turn % 360.0, 360.0 - (net_turn % 360.0))
    turn_steps = np.diff(headings)
    total_turn = float(np.degrees(np.sum(np.abs(turn_steps))))
    return sinuosity, net_turn, closure_ratio, total_turn


def _overlap_fraction(first: np.ndarray, second: np.ndarray, radius_km: float) -> float:
    """Symmetric fraction of two lines that represents the same boundary."""
    a = line_support_fraction(first, [second], radius_km)
    b = line_support_fraction(second, [first], radius_km)
    return min(a, b)


def detect_fronts_two_scale(
    theta_w: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    *,
    synoptic_sigma_km: float = 100.0,
    refine_sigma_km: float = 45.0,
    derivative_sigma_km: float = 15.0,
    corridor_km: float = 110.0,
    min_synoptic_support: float = 0.60,
    synoptic_min_length_km: float = 350.0,
    refine_min_length_km: float = 220.0,
    boundary_margin_km: float = 70.0,
    return_synoptic: bool = False,
    synoptic_tfp_weak: float = -1.5e-5,
    synoptic_tfp_full: float = -4.0e-5,
    synoptic_abz_weak: float = 0.65,
    synoptic_abz_full: float = 1.10,
    refined_tfp_weak: float = -2.5e-5,
    refined_tfp_full: float = -9.0e-5,
    refined_abz_weak: float = 0.90,
    refined_abz_full: float = 1.70,
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
        tfp_threshold=synoptic_tfp_weak,
        tfp_full_strength=synoptic_tfp_full,
        abz_gradient_threshold=synoptic_abz_weak,
        abz_gradient_full_strength=synoptic_abz_full,
        min_length_km=synoptic_min_length_km,
        boundary_margin_km=boundary_margin_km,
    )
    refined = fl.locate_fronts(
        theta_w, longitudes, latitudes,
        synoptic_sigma_km=refine_sigma_km,
        derivative_sigma_km=derivative_sigma_km,
        # Beckert et al. (2023) show that less-smoothed kilometre-scale
        # fields need stricter TFP/gradient intervals than synoptic fields.
        tfp_threshold=refined_tfp_weak,
        tfp_full_strength=refined_tfp_full,
        abz_gradient_threshold=refined_abz_weak,
        abz_gradient_full_strength=refined_abz_full,
        min_length_km=refine_min_length_km,
        boundary_margin_km=boundary_margin_km,
    )
    synoptic_lines = [c["coordinates"] for c in synoptic]

    final = []
    for candidate in refined:
        sinuosity, net_turn, closure_ratio, total_turn = _shape_metrics(
            candidate["coordinates"]
        )
        # Closed/near-closed thermal anomalies and hairpins are local air
        # pools or convective outflows, not interfaces between two extended
        # air masses.  Accumulated turning is retained as a soft diagnostic;
        # closure and sinuosity provide the robust hard rejection.
        if sinuosity > 2.35 or closure_ratio < 0.42 or net_turn > 165.0:
            continue
        candidate["sinuosity"] = round(sinuosity, 2)
        candidate["netTurnDeg"] = round(net_turn, 1)
        candidate["closureRatio"] = round(closure_ratio, 2)
        candidate["totalTurnDeg"] = round(total_turn, 1)
        support = line_support_fraction(
            candidate["coordinates"], synoptic_lines, corridor_km
        )
        candidate["synopticSupport"] = round(support, 2)
        if support >= min_synoptic_support:
            candidate["corroborated"] = True
            final.append(candidate)
        # Nessuna eccezione basata sulla sola intensita': un confine locale
        # puo' essere piu' netto di un fronte sinottico autentico.

    # TFL zero contours occasionally yield nearly coincident fragments.
    # Keep the physically strongest representative rather than publishing
    # parallel duplicate fronts.
    final.sort(
        key=lambda c: (c.get("locatorConfidence", 0.0), c["lengthKm"]),
        reverse=True,
    )
    unique = []
    for candidate in final:
        if any(
            _overlap_fraction(candidate["coordinates"], kept["coordinates"], 55.0)
            >= 0.72
            for kept in unique
        ):
            continue
        unique.append(candidate)
    final = unique
    final.sort(key=lambda c: c["lengthKm"], reverse=True)
    if return_synoptic:
        return final, synoptic
    return final
