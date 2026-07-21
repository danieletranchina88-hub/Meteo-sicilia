"""Synoptic front locator (v10 core) - Sansom & Catto formulation.

Contour-then-mask objective front location on the wet-bulb potential
temperature field theta_w, following the modern, portable version of the
Hewson (1998) method used by Sansom & Catto (2020):

    theta_w (raw)
      -> physical km smoothing (NaN-aware)
      -> metric gradient  ->  |grad theta_w|
      -> TFL = laplacian(|grad theta_w|)
      -> zero contour of TFL           (contour, not mask, first)
      -> sample TFP and ABZ on the contour points
      -> mask (TFP > 0 warm side, ABZ gradient strong) and split
      -> minimum length filter

This module produces only thermodynamic *candidates* with diagnostics.
It does NOT classify cold/warm/stationary, does not assign a final
confidence, and does not track in time - those belong to later modules.

References
----------
Hewson (1998), Met. Apps 5, 37-65.
Sansom & Catto (2020), GMD - contour-then-mask, TFL zero contour + TFP/ABZ.
"""

from __future__ import annotations

import math

import contourpy
import numpy as np

EARTH_KM_PER_DEG = 111.32
LOCATOR_LAPLACIAN = "laplacian_gradient"   # default (Sansom-Catto)
LOCATOR_HEWSON = "hewson_directional"      # reserved for future comparison


# --------------------------------------------------------------------------
# Grid geometry (metric-aware)
# --------------------------------------------------------------------------
def grid_metrics(longitudes: np.ndarray, latitudes: np.ndarray) -> dict:
    """Cell sizes in km for a regular lon/lat grid (latitudes ascending)."""
    lon = np.asarray(longitudes, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    dlon = float(abs(lon[1] - lon[0]))
    dlat = float(abs(lat[1] - lat[0]))
    dy_km = dlat * EARTH_KM_PER_DEG
    dx_km_col = (dlon * EARTH_KM_PER_DEG * np.cos(np.deg2rad(lat)))[:, None]
    return {"dx_km_col": np.maximum(dx_km_col, 1.0e-3), "dy_km": dy_km,
            "dlon": dlon, "dlat": dlat}


# --------------------------------------------------------------------------
# NaN-aware physical (km) Gaussian smoothing
# --------------------------------------------------------------------------
def _gaussian_kernel(sigma_points: float) -> np.ndarray:
    sigma = max(float(sigma_points), 1.0e-3)
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _conv_axis(data: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pad = [(radius, radius) if a == axis else (0, 0) for a in range(data.ndim)]
    padded = np.pad(data, pad, mode="reflect")
    out = np.zeros_like(data, dtype=float)
    for i, weight in enumerate(kernel):
        index = [slice(None)] * data.ndim
        index[axis] = slice(i, i + data.shape[axis])
        out += weight * padded[tuple(index)]
    return out


def _conv_x_perrow(data: np.ndarray, sigma_points_per_row: np.ndarray) -> np.ndarray:
    """1-D Gaussian along x with a sigma that varies per latitude row."""
    out = np.empty_like(data, dtype=float)
    # Group rows with near-equal sigma to reuse kernels (cheap and exact
    # enough: sigma rounded to 0.05 points).
    rounded = np.round(sigma_points_per_row / 0.05) * 0.05
    for sigma in np.unique(rounded):
        rows = np.where(rounded == sigma)[0]
        kernel = _gaussian_kernel(sigma)
        block = data[rows, :]
        radius = len(kernel) // 2
        padded = np.pad(block, ((0, 0), (radius, radius)), mode="reflect")
        acc = np.zeros_like(block, dtype=float)
        for i, weight in enumerate(kernel):
            acc += weight * padded[:, i:i + block.shape[1]]
        out[rows, :] = acc
    return out


def smooth_km(field: np.ndarray, sigma_km: float, metrics: dict) -> np.ndarray:
    """NaN-aware Gaussian smoothing with a physical (km) scale.

    Independent of downsampling (sigma is in km, converted to points via
    the grid metrics).  y uses a constant sigma; x uses a per-row sigma so
    the km scale is honoured at every latitude (dx = R cos(phi) dlon).
    """
    if sigma_km <= 0.0:
        return np.asarray(field, dtype=float)
    values = np.asarray(field, dtype=float)
    valid = np.isfinite(values).astype(float)
    filled = np.where(np.isfinite(values), values, 0.0)

    dy_km = metrics["dy_km"]
    sigma_y = sigma_km / dy_km
    sigma_x_row = sigma_km / metrics["dx_km_col"].ravel()

    def blur(array: np.ndarray) -> np.ndarray:
        array = _conv_axis(array, _gaussian_kernel(sigma_y), axis=0)
        array = _conv_x_perrow(array, sigma_x_row)
        return array

    numerator = blur(filled)
    denominator = blur(valid)
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = numerator / np.where(denominator > 1.0e-9, denominator, 1.0)
    return np.where(denominator >= 0.2, smoothed, np.nan)


# --------------------------------------------------------------------------
# Metric-aware differential operators (local flat-plane approximation)
# --------------------------------------------------------------------------
def gradient(field: np.ndarray, metrics: dict) -> tuple[np.ndarray, np.ndarray]:
    """East (d/dx) and north (d/dy) derivatives, per km."""
    east = np.gradient(field, axis=1) / metrics["dx_km_col"]
    north = np.gradient(field, axis=0) / metrics["dy_km"]
    return east, north


def laplacian(field: np.ndarray, metrics: dict) -> np.ndarray:
    east, north = gradient(field, metrics)
    east_x, _ = gradient(east, metrics)
    _, north_y = gradient(north, metrics)
    return east_x + north_y


# --------------------------------------------------------------------------
# Bilinear sampling on the grid (NaN outside)
# --------------------------------------------------------------------------
def _sample(field, coordinates, longitudes, latitudes, dlon, dlat):
    x = (coordinates[:, 0] - longitudes[0]) / dlon
    y = (coordinates[:, 1] - latitudes[0]) / dlat
    inside = (x >= 0) & (x <= len(longitudes) - 1.001) & (y >= 0) & (y <= len(latitudes) - 1.001)
    # Coordinate NaN (es. sondaggi ABZ dove la direzione e' indefinita) ->
    # indice sicuro 0, mascherate poi da 'inside'.
    xc = np.where(np.isfinite(x), np.clip(x, 0.0, len(longitudes) - 1.001), 0.0)
    yc = np.where(np.isfinite(y), np.clip(y, 0.0, len(latitudes) - 1.001), 0.0)
    x0, y0 = np.floor(xc).astype(int), np.floor(yc).astype(int)
    x1 = np.minimum(x0 + 1, field.shape[1] - 1)
    y1 = np.minimum(y0 + 1, field.shape[0] - 1)
    fx, fy = xc - x0, yc - y0
    value = (
        field[y0, x0] * (1 - fx) * (1 - fy)
        + field[y0, x1] * fx * (1 - fy)
        + field[y1, x0] * (1 - fx) * fy
        + field[y1, x1] * fx * fy
    )
    return np.where(inside, value, np.nan)


def _line_length_km(coordinates: np.ndarray) -> float:
    if len(coordinates) < 2:
        return 0.0
    lon1, lat1 = coordinates[:-1, 0], coordinates[:-1, 1]
    lon2, lat2 = coordinates[1:, 0], coordinates[1:, 1]
    mean_lat = np.deg2rad((lat1 + lat2) * 0.5)
    dx = (lon2 - lon1) * EARTH_KM_PER_DEG * np.cos(mean_lat)
    dy = (lat2 - lat1) * EARTH_KM_PER_DEG
    return float(np.sum(np.hypot(dx, dy)))


def _split_where(coordinates: np.ndarray, keep: np.ndarray, min_points: int = 4) -> list:
    pieces, start = [], None
    keep = np.asarray(keep, dtype=bool)
    for i, ok in enumerate(keep):
        if ok and start is None:
            start = i
        if start is not None and (not ok or i == len(keep) - 1):
            stop = i + 1 if ok and i == len(keep) - 1 else i
            if stop - start >= min_points:
                pieces.append(coordinates[start:stop])
            start = None
    return pieces


# --------------------------------------------------------------------------
# Locator
# --------------------------------------------------------------------------
def locate_fronts(
    theta_w: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    *,
    synoptic_sigma_km: float = 50.0,
    derivative_sigma_km: float = 20.0,
    tfp_threshold: float = 0.0,
    abz_gradient_threshold: float = 1.5,
    abz_distance_km: float = 120.0,
    min_length_km: float = 250.0,
    locator_method: str = LOCATOR_LAPLACIAN,
    return_fields: bool = False,
):
    """Locate synoptic front candidates on theta_w (single time step).

    theta_w in K.  Thresholds: ``abz_gradient_threshold`` in K/100 km,
    ``abz_distance_km`` the physical step into the adjacent baroclinic zone.
    Returns a list of candidate dicts (geometry + diagnostics, no
    classification).  With ``return_fields`` also returns the diagnostic
    fields for inspection/plotting.
    """
    if locator_method != LOCATOR_LAPLACIAN:
        raise NotImplementedError(
            f"locator '{locator_method}' non ancora implementato "
            f"(default: {LOCATOR_LAPLACIAN})"
        )
    lon = np.asarray(longitudes, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    grid = np.asarray(theta_w, dtype=float)
    # Normalizza a coordinate crescenti: contourpy e le metriche assumono
    # lon/lat monotone crescenti.  Il risultato geometrico e' identico
    # qualunque sia l'orientamento in ingresso.
    if lat[1] < lat[0]:
        lat = lat[::-1]
        grid = grid[::-1, :]
    if lon[1] < lon[0]:
        lon = lon[::-1]
        grid = grid[:, ::-1]
    theta_w = grid
    metrics = grid_metrics(lon, lat)
    dlon, dlat = metrics["dlon"], metrics["dlat"]

    # 1) physical smoothing BEFORE any derivative
    field = smooth_km(np.asarray(theta_w, dtype=float), synoptic_sigma_km, metrics)

    # 2) metric gradient and its magnitude (K/km)
    grad_e, grad_n = gradient(field, metrics)
    grad_mag = np.hypot(grad_e, grad_n)
    grad_mag = smooth_km(grad_mag, derivative_sigma_km, metrics)

    # 3) TFL = laplacian(|grad theta_w|), zero contour locates the ridge
    tfl = laplacian(grad_mag, metrics)

    # 4) TFP = -grad(|grad theta_w|) . grad(theta_w)/|grad theta_w|
    #    Positive on the WARM side of the baroclinic zone (verified by test).
    gm_e, gm_n = gradient(grad_mag, metrics)
    safe_mag = np.maximum(grad_mag, 1.0e-9)
    tfp = -(gm_e * grad_e + gm_n * grad_n) / safe_mag

    # gradient magnitude expressed in K/100 km for thresholds/diagnostics
    grad_mag_100 = grad_mag * 100.0

    # 5) contour TFL = 0 FIRST, then sample/mask (contour-then-mask)
    generator = contourpy.contour_generator(
        x=lon, y=lat, z=np.where(np.isfinite(tfl), tfl, np.nan),
        line_type="Separate",
    )
    def abz_gradient_at(points: np.ndarray) -> np.ndarray:
        """Sample |grad theta_w| one ABZ step into the baroclinic zone.

        The step follows the direction of grad(|grad theta_w|) - toward the
        crest of the gradient, i.e. into the zone behind the warm edge - by
        a physical distance ``abz_distance_km``.
        """
        step_e = _sample(gm_e, points, lon, lat, dlon, dlat)
        step_n = _sample(gm_n, points, lon, lat, dlon, dlat)
        step_mag = np.maximum(np.hypot(step_e, step_n), 1.0e-12)
        dir_e, dir_n = step_e / step_mag, step_n / step_mag
        mean_lat = np.deg2rad(points[:, 1])
        lon_off = dir_e * abz_distance_km / np.maximum(
            EARTH_KM_PER_DEG * np.cos(mean_lat), 25.0
        )
        lat_off = dir_n * abz_distance_km / EARTH_KM_PER_DEG
        abz_points = points + np.column_stack((lon_off, lat_off))
        return _sample(grad_mag_100, abz_points, lon, lat, dlon, dlat)

    candidates = []
    for segment in generator.lines(0.0):
        coordinates = np.asarray(segment, dtype=float)
        if len(coordinates) < 4:
            continue

        line_tfp = _sample(tfp, coordinates, lon, lat, dlon, dlat)
        abz_grad = abz_gradient_at(coordinates)

        # contour-then-mask: keep the points on the warm side (TFP>threshold)
        # with a genuine baroclinic zone behind them.
        keep = (
            np.isfinite(line_tfp)
            & np.isfinite(abz_grad)
            & (line_tfp > tfp_threshold)
            & (abz_grad > abz_gradient_threshold)
        )
        for piece in _split_where(coordinates, keep):
            if _line_length_km(piece) < min_length_km:
                continue
            piece_tfp = _sample(tfp, piece, lon, lat, dlon, dlat)
            piece_grad = _sample(grad_mag_100, piece, lon, lat, dlon, dlat)
            piece_abz = abz_gradient_at(piece)
            candidates.append({
                "coordinates": piece,
                "medianTfp": float(np.nanmedian(piece_tfp)),
                "medianThetaWGradient": float(np.nanmedian(piece_grad)),
                "medianAbzGradient": float(np.nanmedian(piece_abz)),
                "peakAbzGradient": float(np.nanmax(piece_abz)),
                "lengthKm": _line_length_km(piece),
                "locatorMethod": locator_method,
            })

    candidates.sort(key=lambda c: c["lengthKm"], reverse=True)
    if return_fields:
        return candidates, {
            "theta_w_smooth": field,
            "grad_mag_100": grad_mag_100,
            "tfl": tfl,
            "tfp": tfp,
        }
    return candidates
