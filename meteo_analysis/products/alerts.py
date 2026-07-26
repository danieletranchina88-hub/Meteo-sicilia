"""Location-aware frontal-transit estimates."""

from __future__ import annotations

import math

import numpy as np


def _line_distance(point_xy: np.ndarray, line_xy: np.ndarray) -> float:
    starts = line_xy[:-1]
    vectors = line_xy[1:] - starts
    lengths2 = np.sum(vectors * vectors, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.sum((point_xy - starts) * vectors, axis=1) / lengths2
    fraction = np.clip(np.nan_to_num(fraction), 0.0, 1.0)
    closest = starts + fraction[:, None] * vectors
    return float(np.min(np.hypot(*(closest - point_xy).T)))


def _normalise_front(front: dict):
    if front.get("type") == "Feature":
        geometry = front.get("geometry") or {}
        properties = front.get("properties") or {}
        coordinates = geometry.get("coordinates")
    else:
        properties = front.get("properties") or front
        coordinates = front.get("coordinates")
    array = np.asarray(coordinates, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2 or len(array) < 2:
        return None
    return array[:, :2], properties


def calculate_front_eta(
    user_lat: float,
    user_lon: float,
    front_lines: list,
    front_speed_kmh: float | None = None,
    radius_km: float = 20.0,
    front_direction_deg: float | None = None,
    max_eta_hours: float = 72.0,
):
    """Estimate when a translated front line intersects a 20-km user radius.

    Direction is mandatory because distance/speed alone cannot distinguish an
    approaching front from one moving away.  Bearings are degrees clockwise
    from north.  A per-feature ``motionBearingDeg``/``motionKmh`` takes
    precedence over the scalar fallback arguments.
    """
    reference_lat = float(user_lat)
    km_per_lon = 111.195 * math.cos(math.radians(reference_lat))
    user_xy = np.array([user_lon * km_per_lon, user_lat * 111.195])
    best = None
    missing_motion = False

    for front in front_lines:
        normalised = _normalise_front(front)
        if normalised is None:
            continue
        coordinates, properties = normalised
        line_xy = np.column_stack(
            (coordinates[:, 0] * km_per_lon, coordinates[:, 1] * 111.195)
        )
        current_distance = _line_distance(user_xy, line_xy)
        if current_distance <= radius_km:
            eta = 0.0
        else:
            speed = properties.get("motionKmh", front_speed_kmh)
            bearing = properties.get("motionBearingDeg", front_direction_deg)
            try:
                speed = float(speed)
                bearing = float(bearing)
            except (TypeError, ValueError):
                missing_motion = True
                continue
            if not np.isfinite(speed) or not np.isfinite(bearing) or speed <= 0.0:
                missing_motion = True
                continue
            radians = math.radians(bearing)
            velocity = np.array(
                [speed * math.sin(radians), speed * math.cos(radians)]
            )
            if _line_distance(user_xy, line_xy + velocity) >= current_distance:
                continue
            eta = None
            for candidate in np.arange(0.25, max_eta_hours + 0.25, 0.25):
                if _line_distance(user_xy, line_xy + velocity * candidate) <= radius_km:
                    eta = float(candidate)
                    break
            if eta is None:
                continue

        candidate_result = {
            "status": "imminent" if eta == 0.0 else "approaching",
            "eta_hours": round(eta, 2),
            "distance_km": round(current_distance, 1),
            "front_type": properties.get("frontType", "unknown"),
            "effects": {
                "temperature_change_c": properties.get("deltaTemperature"),
                "motion_kmh": properties.get("motionKmh", front_speed_kmh),
                "rain": "possibile in prossimità del passaggio",
                "gusts": "possibili, soprattutto con fronte freddo",
            },
        }
        if best is None or candidate_result["eta_hours"] < best["eta_hours"]:
            best = candidate_result

    if best is not None:
        return best
    if missing_motion:
        return {
            "status": "unknown",
            "message": (
                "Fronte individuato, ma direzione di movimento insufficiente "
                "per calcolare un orario di transito affidabile."
            ),
        }
    return {"status": "clear", "message": "Nessun fronte in avvicinamento."}
