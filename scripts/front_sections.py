"""Cross-front sections: metric-offset theta_w profiles and their diagnostics.

A real front is not a line but a transition zone. This module samples the
wet-bulb potential temperature along the cold->warm normal of a candidate
line at fixed metric offsets and turns the profiles into physical
diagnostics: how much of the line shows a coherent thermal contrast, how
sharp the transition is, where the strongest gradient sits relative to the
drawn line, how wide the frontal zone is, and how homogeneous the two air
masses are. With 925 hPa data the same profile is compared across height
into a vertical-coherence score.

All quantities are DIAGNOSTIC physics, not calibrated thresholds: the
frontal width in particular must not be used as a hard gate until it has
been calibrated against an independent archive. Missing data (NaN samples,
absent 925 hPa level) are treated as neutral evidence, never as
counter-evidence.
"""

from __future__ import annotations

import numpy as np

EARTH_KM_PER_DEG = 111.32
OFFSETS_KM = (-85.0, -55.0, -30.0, 0.0, 30.0, 55.0, 85.0)
# A per-point theta_w contrast smaller than this is indistinguishable from
# analysis noise at 850 hPa (same floor used by the detector's air-mass test).
MIN_CONTRAST_K = 0.5


def sample_bilinear(field, longitudes, latitudes, points) -> np.ndarray:
    """Bilinear sample of ``field`` at lon/lat ``points``; NaN outside."""
    field = np.asarray(field, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    pts = np.asarray(points, dtype=float)
    x = (pts[:, 0] - lon[0]) / max(lon[1] - lon[0], 1.0e-12)
    y = (pts[:, 1] - lat[0]) / max(lat[1] - lat[0], 1.0e-12)
    inside = (
        (x >= 0.0) & (x <= len(lon) - 1.001)
        & (y >= 0.0) & (y <= len(lat) - 1.001)
    )
    xc = np.clip(x, 0.0, len(lon) - 1.001)
    yc = np.clip(y, 0.0, len(lat) - 1.001)
    x0 = np.floor(xc).astype(int)
    y0 = np.floor(yc).astype(int)
    fx = xc - x0
    fy = yc - y0
    values = (
        field[y0, x0] * (1 - fx) * (1 - fy)
        + field[y0, x0 + 1] * fx * (1 - fy)
        + field[y0 + 1, x0] * (1 - fx) * fy
        + field[y0 + 1, x0 + 1] * fx * fy
    )
    return np.where(inside, values, np.nan)


def offset_points(coordinates, warm_normal, distance_km) -> np.ndarray:
    """Displace each line point ``distance_km`` along its warm-ward normal."""
    coordinates = np.asarray(coordinates, dtype=float)
    normal = np.asarray(warm_normal, dtype=float)
    latitude = coordinates[:, 1]
    return coordinates + np.column_stack((
        normal[:, 0] * distance_km
        / (EARTH_KM_PER_DEG * np.cos(np.deg2rad(latitude))),
        normal[:, 1] * distance_km / EARTH_KM_PER_DEG,
    ))


def cross_profiles(
    field, longitudes, latitudes, coordinates, warm_normal,
    offsets_km=OFFSETS_KM,
) -> np.ndarray:
    """theta_w profile across the front: (n_line_points, n_offsets)."""
    columns = [
        sample_bilinear(
            field, longitudes, latitudes,
            offset_points(coordinates, warm_normal, offset),
        )
        for offset in offsets_km
    ]
    return np.column_stack(columns)


def profile_diagnostics(profiles, offsets_km=OFFSETS_KM) -> dict:
    """Physical diagnostics of the cross-front profiles.

    Offsets are signed along the warm-ward normal, so a genuine front has
    theta_w increasing with the offset. All statistics are per-line-point
    first, then reduced by the median, so a locally corrupted stretch cannot
    dominate a long line.
    """
    profiles = np.asarray(profiles, dtype=float)
    offsets = np.asarray(offsets_km, dtype=float)
    n_points = len(profiles)
    empty = {
        "profileValidFraction": 0.0,
        "profileThermalSupport": 0.0,
        "profilePeakGradient": np.nan,
        "frontWidthKm": np.nan,
        "frontOffsetKm": np.nan,
        "airMassHomogeneity": np.nan,
    }
    if n_points == 0 or profiles.ndim != 2:
        return empty
    valid_rows = np.all(np.isfinite(profiles), axis=1)
    valid_fraction = float(np.mean(valid_rows))
    if not np.any(valid_rows):
        return {**empty, "profileValidFraction": valid_fraction}
    rows = profiles[valid_rows]

    warm_side = offsets > 0.0
    cold_side = offsets < 0.0
    contrast = rows[:, warm_side].mean(axis=1) - rows[:, cold_side].mean(axis=1)
    thermal_support = float(np.mean(contrast > MIN_CONTRAST_K))

    # slopes between adjacent offsets, in K/100km, evaluated at midpoints
    d_offset = np.diff(offsets)
    midpoints = 0.5 * (offsets[:-1] + offsets[1:])
    slopes = np.diff(rows, axis=1) / d_offset * 100.0
    peak_index = np.argmax(slopes, axis=1)
    peak_gradient = slopes[np.arange(len(rows)), peak_index]
    front_offset = midpoints[peak_index]

    widths = np.full(len(rows), np.nan)
    for i, (slope_row, peak) in enumerate(zip(slopes, peak_gradient)):
        if not np.isfinite(peak) or peak <= 0.0:
            continue
        strong = slope_row >= 0.5 * peak
        # contiguous strong run containing the peak
        j = peak_index[i]
        lo = j
        while lo > 0 and strong[lo - 1]:
            lo -= 1
        hi = j
        while hi < len(strong) - 1 and strong[hi + 1]:
            hi += 1
        widths[i] = midpoints[hi] - midpoints[lo] + float(np.mean(d_offset))

    homogeneity_scale_k = 2.0
    side_std = 0.5 * (rows[:, warm_side].std(axis=1)
                      + rows[:, cold_side].std(axis=1))
    homogeneity = np.clip(1.0 - side_std / homogeneity_scale_k, 0.0, 1.0)

    positive = peak_gradient > 0.0
    return {
        "profileValidFraction": round(valid_fraction, 3),
        "profileThermalSupport": round(thermal_support, 3),
        "profilePeakGradient": _round(np.median(peak_gradient[positive])
                                      if np.any(positive) else np.nan),
        "frontWidthKm": _round(np.nanmedian(widths)
                               if np.any(np.isfinite(widths)) else np.nan, 1),
        "frontOffsetKm": _round(np.median(front_offset[positive])
                                if np.any(positive) else np.nan, 1),
        "airMassHomogeneity": _round(np.median(homogeneity), 3),
    }


def vertical_coherence(diagnostics_850: dict, diagnostics_925: dict | None):
    """Coherence of the frontal structure between 850 and 925 hPa, in [0, 1].

    Returns ``None`` (neutral, NOT counter-evidence) when the 925 hPa
    profile is unavailable or almost entirely invalid, e.g. where the
    pressure surface intersects terrain. The score combines:

    - same-sign thermal contrast at both levels;
    - ratio of the peak cross-front gradients;
    - horizontal distance between the two gradient maxima (a real front
      tilts, so a modest shift is normal; ~60 km is the tolerance scale);
    - similarity of the frontal widths;
    - fraction of the line valid at both levels.
    """
    if not diagnostics_925:
        return None
    valid_925 = float(diagnostics_925.get("profileValidFraction") or 0.0)
    if valid_925 < 0.20:
        return None
    support_850 = float(diagnostics_850.get("profileThermalSupport") or 0.0)
    support_925 = float(diagnostics_925.get("profileThermalSupport") or 0.0)
    sign_score = min(support_850, support_925)

    peak_850 = diagnostics_850.get("profilePeakGradient")
    peak_925 = diagnostics_925.get("profilePeakGradient")
    if _finite(peak_850) and _finite(peak_925) and max(peak_850, peak_925) > 0:
        ratio_score = float(np.clip(
            min(peak_850, peak_925) / max(peak_850, peak_925), 0.0, 1.0
        ))
    else:
        ratio_score = 0.0

    offset_850 = diagnostics_850.get("frontOffsetKm")
    offset_925 = diagnostics_925.get("frontOffsetKm")
    if _finite(offset_850) and _finite(offset_925):
        offset_score = float(np.clip(
            1.0 - abs(offset_850 - offset_925) / 60.0, 0.0, 1.0
        ))
    else:
        offset_score = 0.0

    width_850 = diagnostics_850.get("frontWidthKm")
    width_925 = diagnostics_925.get("frontWidthKm")
    if (_finite(width_850) and _finite(width_925)
            and max(width_850, width_925) > 0):
        width_score = float(np.clip(
            min(width_850, width_925) / max(width_850, width_925), 0.0, 1.0
        ))
    else:
        width_score = 0.0

    both_valid = min(
        float(diagnostics_850.get("profileValidFraction") or 0.0), valid_925
    )
    coherence = (
        0.35 * sign_score + 0.25 * ratio_score
        + 0.20 * offset_score + 0.20 * width_score
    ) * float(np.clip(both_valid / 0.60, 0.0, 1.0))
    return round(float(np.clip(coherence, 0.0, 1.0)), 3)


def _pair_coherence(diagnostics_a: dict, diagnostics_b: dict, min_valid=0.20):
    """Coherence of two cross-front profiles (same metric-offset diagnostics).

    Returns None (neutral) when the second profile is unavailable or almost
    entirely invalid, else a [0, 1] score from same-sign contrast, gradient
    ratio, distance between the gradient maxima and width similarity.
    """
    if not diagnostics_b:
        return None
    valid_b = float(diagnostics_b.get("profileValidFraction") or 0.0)
    if valid_b < min_valid:
        return None
    sign = min(
        float(diagnostics_a.get("profileThermalSupport") or 0.0),
        float(diagnostics_b.get("profileThermalSupport") or 0.0),
    )
    pa = diagnostics_a.get("profilePeakGradient")
    pb = diagnostics_b.get("profilePeakGradient")
    ratio = (
        float(np.clip(min(pa, pb) / max(pa, pb), 0.0, 1.0))
        if _finite(pa) and _finite(pb) and max(pa, pb) > 0 else 0.0
    )
    oa = diagnostics_a.get("frontOffsetKm")
    ob = diagnostics_b.get("frontOffsetKm")
    offset = (
        float(np.clip(1.0 - abs(oa - ob) / 60.0, 0.0, 1.0))
        if _finite(oa) and _finite(ob) else 0.0
    )
    wa = diagnostics_a.get("frontWidthKm")
    wb = diagnostics_b.get("frontWidthKm")
    width = (
        float(np.clip(min(wa, wb) / max(wa, wb), 0.0, 1.0))
        if _finite(wa) and _finite(wb) and max(wa, wb) > 0 else 0.0
    )
    both_valid = min(
        float(diagnostics_a.get("profileValidFraction") or 0.0), valid_b
    )
    score = (0.35 * sign + 0.25 * ratio + 0.20 * offset + 0.20 * width) \
        * float(np.clip(both_valid / 0.60, 0.0, 1.0))
    return round(float(np.clip(score, 0.0, 1.0)), 3)


def multilevel_coherence(diagnostics_850, diagnostics_925, diagnostics_700=None,
                         terrain_fraction=0.0):
    """Vertical coherence across 925/850/700 hPa (sez. 5).

    850 hPa is the reference. 925 hPa checks the link to the low levels;
    700 hPa corroborates aloft and becomes the FALLBACK reference when 850 is
    close to the ground (high ``terrain_fraction``), where the 850 signal is
    unreliable. Missing levels are neutral (contribute nothing), never
    counter-evidence. Returns a dict with the combined score and the
    per-level breakdown plus which levels supported the object.
    """
    coherence_925 = _pair_coherence(diagnostics_850, diagnostics_925)
    coherence_700 = _pair_coherence(diagnostics_850, diagnostics_700)
    parts = []
    levels = []
    # near the ground the 850 anchor is weak: lean on 700 more heavily
    orographic = float(np.clip(terrain_fraction, 0.0, 1.0))
    if coherence_925 is not None:
        parts.append((coherence_925, 1.0))
        levels.append("925")
    if coherence_700 is not None:
        parts.append((coherence_700, 1.0 + 1.5 * orographic))
        levels.append("700")
    if not parts:
        combined = None
    else:
        weight_sum = sum(w for _, w in parts)
        combined = round(sum(c * w for c, w in parts) / weight_sum, 3)
    return {
        "verticalCoherence": combined,
        "coherence925": coherence_925,
        "coherence700": coherence_700,
        "supportedLevels": levels,
    }


def _finite(value) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _round(value, digits: int = 2):
    return float(np.round(value, digits)) if np.isfinite(value) else np.nan
