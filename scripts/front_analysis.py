"""Objective, deliberately conservative synoptic-front estimation.

The detector works on equivalent potential temperature and wind at 850 hPa.
It is intended for a mobile synoptic viewer: the output is a small GeoJSON
collection, not a replacement for a forecaster's analysed fronts.
"""

from __future__ import annotations

from collections import OrderedDict
import math

import contourpy
import numpy as np
import xarray as xr


FRONT_METHOD = "theta-e-850-tfp-wind-v4-ptrough"


def _box_smooth(field: np.ndarray, radius: int) -> np.ndarray:
    """Fast edge-preserving box mean without a scipy dependency."""
    if radius <= 0:
        return np.asarray(field, dtype=float)
    values = np.asarray(field, dtype=float)
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    width = radius * 2 + 1
    return (
        integral[width:, width:]
        - integral[:-width, width:]
        - integral[width:, :-width]
        + integral[:-width, :-width]
    ) / float(width * width)


def _equivalent_potential_temperature(temp_k: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Bolton-style equivalent potential temperature at 850 hPa."""
    pressure_pa = 85_000.0
    temperature = np.clip(np.asarray(temp_k, dtype=float), 185.0, 330.0)
    specific_humidity = np.clip(np.asarray(q, dtype=float), 1.0e-6, 0.04)
    mixing_ratio = specific_humidity / np.maximum(1.0 - specific_humidity, 1.0e-6)
    vapor_pressure = pressure_pa * mixing_ratio / (0.622 + mixing_ratio)
    vapor_pressure_hpa = np.clip(vapor_pressure / 100.0, 0.05, 80.0)
    logarithm = np.log(vapor_pressure_hpa / 6.112)
    dewpoint_c = 243.5 * logarithm / (17.67 - logarithm)
    dewpoint_k = np.clip(dewpoint_c + 273.15, 180.0, temperature)
    lcl_temperature = 1.0 / (
        1.0 / np.maximum(dewpoint_k - 56.0, 1.0)
        + np.log(temperature / dewpoint_k) / 800.0
    ) + 56.0
    dry_theta = temperature * (100_000.0 / pressure_pa) ** (
        0.2854 * (1.0 - 0.28 * mixing_ratio)
    )
    exponent = (3376.0 / lcl_temperature - 2.54) * mixing_ratio * (
        1.0 + 0.81 * mixing_ratio
    )
    return dry_theta * np.exp(np.clip(exponent, -2.0, 2.0))


def _line_length_km(coordinates: np.ndarray) -> float:
    if len(coordinates) < 2:
        return 0.0
    lon1, lat1 = coordinates[:-1, 0], coordinates[:-1, 1]
    lon2, lat2 = coordinates[1:, 0], coordinates[1:, 1]
    mean_latitude = np.deg2rad((lat1 + lat2) * 0.5)
    dx = (lon2 - lon1) * 111.32 * np.cos(mean_latitude)
    dy = (lat2 - lat1) * 111.32
    return float(np.sum(np.hypot(dx, dy)))


def _project_km(coordinates: np.ndarray, mean_latitude_rad: float) -> np.ndarray:
    """Project lon/lat degrees onto a local flat plane measured in km."""
    scale_lon = 111.32 * math.cos(mean_latitude_rad)
    return np.column_stack(
        (coordinates[:, 0] * scale_lon, coordinates[:, 1] * 111.32)
    )


def _line_shape_metrics(coordinates: np.ndarray) -> tuple[float, float]:
    """Sinuosity and net heading change (degrees) of a polyline.

    A synoptic front separates two air masses, so it is quasi-linear or
    gently arced.  A line that folds back on itself (hairpin) or nearly
    closes (ring) is by construction the rim of a local anomaly - a warm
    sea pool, a convective cold pool - and can never be a front.

    The turning measure is the NET change between the initial and final
    heading, not the accumulated one: a long quasi-straight front with
    small zig-zags accumulates hundreds of degrees without ever folding,
    while a hairpin nets ~180 regardless of how it is sampled.
    """
    if len(coordinates) < 3:
        return 1.0, 0.0
    mean_latitude = math.radians(float(np.mean(coordinates[:, 1])))
    simplified = _rdp(coordinates, 0.05)
    points_km = _project_km(simplified, mean_latitude)
    segments = np.diff(points_km, axis=0)
    lengths = np.hypot(segments[:, 0], segments[:, 1])
    keep = lengths > 1.0
    segments, lengths = segments[keep], lengths[keep]
    path_length = float(np.sum(lengths))
    endpoint_distance = float(
        np.hypot(*(points_km[-1] - points_km[0]))
    )
    sinuosity = path_length / max(endpoint_distance, 1.0)
    if len(segments) < 2:
        return sinuosity, 0.0
    headings = np.arctan2(segments[:, 1], segments[:, 0])
    net_turn = (headings[-1] - headings[0] + math.pi) % (2.0 * math.pi) - math.pi
    return sinuosity, abs(float(np.degrees(net_turn)))


def _matched_fraction(
    coordinates: np.ndarray,
    reference_lines: list[np.ndarray],
    max_distance_km: float,
) -> float:
    """Fraction of a candidate line lying within reach of any reference line."""
    mean_latitude = math.radians(float(np.mean(coordinates[:, 1])))
    candidate_km = _project_km(coordinates, mean_latitude)
    best = np.full(len(candidate_km), np.inf)
    for reference in reference_lines:
        reference_km = _project_km(reference, mean_latitude)
        dx = candidate_km[:, 0][:, None] - reference_km[:, 0][None, :]
        dy = candidate_km[:, 1][:, None] - reference_km[:, 1][None, :]
        best = np.minimum(best, np.min(np.hypot(dx, dy), axis=1))
    return float(np.mean(best <= max_distance_km))


def corroborate_with_reference(
    fronts: dict,
    reference_fronts: dict | None,
    max_distance_km: float = 220.0,
    min_fraction: float = 0.5,
) -> dict:
    """Discard candidate fronts with no counterpart in a coarser, independent analysis.

    A 2.2 km model such as ICON-2I resolves mesoscale boundaries (sea-breeze
    convergence, orographic channelling, convective outflows) that a
    synoptic-front detector can mistake for real fronts, no matter how the
    thermal/wind thresholds above are tuned. A genuine synoptic front is a
    large-scale feature, so it is also present - smoothed and displaced by at
    most a few tens of km - in an independent, coarser-resolution model run.

    The match requires at least ``min_fraction`` of the candidate line to lie
    within ``max_distance_km`` of a reference front: a single nearby point is
    not corroboration, otherwise an artefact is "confirmed" by a real front
    that merely passes in the neighbourhood.
    """
    reference_features = (reference_fronts or {}).get("features") or []
    if not reference_features:
        return fronts

    reference_lines = [
        np.asarray(feature["geometry"]["coordinates"], dtype=float)
        for feature in reference_features
    ]
    kept = []
    for feature in fronts.get("features", []):
        coordinates = np.asarray(feature["geometry"]["coordinates"], dtype=float)
        fraction = _matched_fraction(coordinates, reference_lines, max_distance_km)
        if fraction >= min_fraction:
            kept.append(feature)

    result = dict(fronts)
    result["features"] = kept
    return result


def _rdp(points: np.ndarray, tolerance_degrees: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification for compact GeoJSON output."""
    if len(points) <= 2:
        return points
    start, end = points[0], points[-1]
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1.0e-12:
        distances = np.hypot(points[:, 0] - start[0], points[:, 1] - start[1])
    else:
        fractions = np.clip(((points - start) @ segment) / denominator, 0.0, 1.0)
        projections = start + fractions[:, None] * segment
        distances = np.hypot(points[:, 0] - projections[:, 0], points[:, 1] - projections[:, 1])
    index = int(np.argmax(distances))
    if distances[index] <= tolerance_degrees:
        return np.vstack((start, end))
    left = _rdp(points[: index + 1], tolerance_degrees)
    right = _rdp(points[index:], tolerance_degrees)
    return np.vstack((left[:-1], right))


class SynopticFrontAnalyzer:
    """Extract a few high-confidence, large-scale fronts for each forecast step.

    The four pressure-level fields may live in separate files (ICON-2I) or in
    one multi-field GRIB (ECMWF Open Data).  All calculations are cropped to
    the requested synoptic domain before entering memory.
    """

    def __init__(
        self,
        temperature_path: str,
        humidity_path: str,
        u_wind_path: str,
        v_wind_path: str,
        orography_path: str | None = None,
        pressure_path: str | None = None,
        downsample: int = 4,
        bounds: tuple[float, float, float, float] | None = None,
        filters: dict[str, dict] | None = None,
        method: str = FRONT_METHOD,
        source: str = "NWP",
        tendency_window_hours: int = 3,
    ) -> None:
        self.bounds = bounds
        self.method = method
        self.source = source
        self.tendency_window_hours = max(1, int(tendency_window_hours))
        filters = filters or {}

        def open_field(path: str, name: str) -> xr.Dataset:
            backend_kwargs: dict = {"indexpath": ""}
            if filters.get(name):
                backend_kwargs["filter_by_keys"] = filters[name]
            return xr.open_dataset(
                path,
                engine="cfgrib",
                backend_kwargs=backend_kwargs,
            )

        self.datasets = {
            "t": open_field(temperature_path, "t"),
            "q": open_field(humidity_path, "q"),
            "u": open_field(u_wind_path, "u"),
            "v": open_field(v_wind_path, "v"),
        }
        if orography_path:
            self.datasets["h"] = open_field(orography_path, "h")
        if pressure_path:
            # La pressione e' una firma in piu', non un requisito: se manca
            # l'analisi prosegue con le firme termiche e dinamiche.
            try:
                self.datasets["p"] = open_field(pressure_path, "p")
            except Exception:
                pass
        if pressure_path:
            # La pressione e' una firma in piu', non un requisito: se manca
            # l'analisi prosegue con le firme termiche e dinamiche.
            try:
                self.datasets["p"] = open_field(pressure_path, "p")
            except Exception:
                pass
        self.keys = {
            key: next(iter(dataset.data_vars))
            for key, dataset in self.datasets.items()
        }
        factor = max(1, int(downsample))
        self.factor = factor
        source_dataset = self.datasets["t"]
        source_latitudes = np.sort(
            np.asarray(source_dataset.latitude.values, dtype=float).ravel()
        )
        source_longitudes = np.sort(
            np.asarray(source_dataset.longitude.values, dtype=float).ravel()
        )
        if bounds is not None:
            lon_min, lon_max, lat_min, lat_max = bounds
            source_latitudes = source_latitudes[
                (source_latitudes >= lat_min) & (source_latitudes <= lat_max)
            ]
            source_longitudes = source_longitudes[
                (source_longitudes >= lon_min) & (source_longitudes <= lon_max)
            ]
        self.latitudes = source_latitudes[::factor]
        self.longitudes = source_longitudes[::factor]
        if len(self.latitudes) < 3 or len(self.longitudes) < 3:
            raise ValueError("Dominio insufficiente per l'analisi dei fronti")
        self.delta_latitude = float(abs(self.latitudes[1] - self.latitudes[0]))
        self.delta_longitude = float(abs(self.longitudes[1] - self.longitudes[0]))
        self.dy_km = self.delta_latitude * 111.32
        self.dx_km = (
            self.delta_longitude
            * 111.32
            * np.cos(np.deg2rad(self.latitudes))[:, None]
        )
        raw_steps = np.atleast_1d(np.asarray(source_dataset.step.values))
        self.available_hours = [
            int(value / np.timedelta64(1, "h")) for value in raw_steps
        ]
        self.hour_to_index = {
            hour: index for index, hour in enumerate(self.available_hours)
        }
        self.theta_smooth_radius = max(
            1,
            min(4, int(round(0.40 / self.delta_latitude))),
        )
        self.detail_smooth_radius = max(
            1,
            min(2, int(round(0.20 / self.delta_latitude))),
        )
        if "h" in self.datasets:
            self.terrain = self._field("h", self.available_hours[0])
        else:
            self.terrain = np.zeros(
                (len(self.latitudes), len(self.longitudes)),
                dtype=float,
            )
        self.theta_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def close(self) -> None:
        for dataset in self.datasets.values():
            dataset.close()

    def _field(self, name: str, hour: int) -> np.ndarray:
        data = self.datasets[name][self.keys[name]]
        if "step" in data.dims:
            index = self.hour_to_index.get(hour)
            if index is None:
                index = int(np.argmin(np.abs(np.asarray(self.available_hours) - hour)))
            data = data.isel(step=index)
        data = data.squeeze(drop=True)
        data = data.sortby("latitude", ascending=True).sortby(
            "longitude", ascending=True
        )
        if self.bounds is not None:
            lon_min, lon_max, lat_min, lat_max = self.bounds
            data = data.sel(
                latitude=slice(lat_min, lat_max),
                longitude=slice(lon_min, lon_max),
            )
        data = data.transpose("latitude", "longitude")
        values = np.asarray(data.values, dtype=float)
        return values[:: self.factor, :: self.factor]

    def _theta_e(self, hour: int) -> np.ndarray:
        if hour in self.theta_cache:
            result = self.theta_cache.pop(hour)
            self.theta_cache[hour] = result
            return result
        theta = _equivalent_potential_temperature(
            self._field("t", hour), self._field("q", hour)
        )
        # Two passes suppress grid noise while preserving synoptic-scale boundaries.
        theta = _box_smooth(
            _box_smooth(theta, self.theta_smooth_radius),
            self.theta_smooth_radius,
        )
        self.theta_cache[hour] = theta
        while len(self.theta_cache) > 4:
            self.theta_cache.popitem(last=False)
        return theta

    def _gradients(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        north = np.gradient(field, axis=0) / self.dy_km
        east = np.gradient(field, axis=1) / self.dx_km
        return east, north

    def _sample(self, field: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        x = (coordinates[:, 0] - self.longitudes[0]) / self.delta_longitude
        y = (coordinates[:, 1] - self.latitudes[0]) / self.delta_latitude
        x = np.clip(x, 0.0, len(self.longitudes) - 1.001)
        y = np.clip(y, 0.0, len(self.latitudes) - 1.001)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, field.shape[1] - 1)
        y1 = np.minimum(y0 + 1, field.shape[0] - 1)
        fx, fy = x - x0, y - y0
        return (
            field[y0, x0] * (1.0 - fx) * (1.0 - fy)
            + field[y0, x1] * fx * (1.0 - fy)
            + field[y1, x0] * (1.0 - fx) * fy
            + field[y1, x1] * fx * fy
        )

    @staticmethod
    def _split_valid(segment: np.ndarray, valid: np.ndarray) -> list[np.ndarray]:
        # Bridge isolated one-point gaps to avoid chopping otherwise coherent fronts.
        keep = np.asarray(valid, dtype=bool).copy()
        if len(keep) > 2:
            keep[1:-1] |= keep[:-2] & keep[2:]
        pieces = []
        start = None
        for index, accepted in enumerate(keep):
            if accepted and start is None:
                start = index
            if start is not None and (not accepted or index == len(keep) - 1):
                stop = index + 1 if accepted and index == len(keep) - 1 else index
                if stop - start >= 4:
                    pieces.append(segment[start:stop])
                start = None
        return pieces

    def _orient_warm_side_left(
        self, coordinates: np.ndarray, gradient_east: np.ndarray, gradient_north: np.ndarray
    ) -> np.ndarray:
        if len(coordinates) < 2:
            return coordinates
        tangent = coordinates[-1] - coordinates[0]
        mean_latitude = math.radians(float(np.mean(coordinates[:, 1])))
        tangent_east = tangent[0] * math.cos(mean_latitude)
        tangent_north = tangent[1]
        norm = math.hypot(tangent_east, tangent_north)
        if norm <= 1.0e-8:
            return coordinates
        left_east, left_north = -tangent_north / norm, tangent_east / norm
        gx = float(np.nanmedian(self._sample(gradient_east, coordinates)))
        gy = float(np.nanmedian(self._sample(gradient_north, coordinates)))
        if left_east * gx + left_north * gy < 0.0:
            return coordinates[::-1].copy()
        return coordinates

    def _candidate_metrics(
        self,
        coordinates: np.ndarray,
        strength: np.ndarray,
        gradient_east: np.ndarray,
        gradient_north: np.ndarray,
        tendency: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        threshold: float,
        dry_gradient: np.ndarray,
        pressure: np.ndarray | None,
    ) -> dict | None:
        line_strength = self._sample(strength, coordinates)
        gx = self._sample(gradient_east, coordinates)
        gy = self._sample(gradient_north, coordinates)
        magnitude = np.maximum(np.hypot(gx, gy), 1.0e-5)
        normal_east, normal_north = gx / magnitude, gy / magnitude
        line_tendency = self._sample(tendency, coordinates)
        center_u = self._sample(u_wind, coordinates)
        center_v = self._sample(v_wind, coordinates)

        # Sample winds roughly 45 km to either side of the analysed boundary.
        mean_latitude = np.deg2rad(coordinates[:, 1])
        lon_offset = normal_east * 45.0 / np.maximum(111.32 * np.cos(mean_latitude), 25.0)
        lat_offset = normal_north * 45.0 / 111.32
        cold_points = coordinates - np.column_stack((lon_offset, lat_offset))
        warm_points = coordinates + np.column_stack((lon_offset, lat_offset))
        cold_u = self._sample(u_wind, cold_points)
        cold_v = self._sample(v_wind, cold_points)
        warm_u = self._sample(u_wind, warm_points)
        warm_v = self._sample(v_wind, warm_points)
        wind_shift = np.hypot(cold_u - warm_u, cold_v - warm_v)
        convergence = (
            (cold_u - warm_u) * normal_east
            + (cold_v - warm_v) * normal_north
        )

        terrain_fraction = float(np.mean(self._sample(self.terrain, coordinates) > 900.0))
        if terrain_fraction > 0.55:
            return None

        # Firma barica: un fronte giace lungo una saccatura, quindi la
        # pressione ai lati della linea (~75 km) deve essere piu' alta che
        # sulla linea stessa.  Il segno opposto (linea su un promontorio
        # anticiclonico) e' fisicamente incompatibile con un fronte.
        pressure_trough = None
        if pressure is not None:
            lon_offset_p = normal_east * 75.0 / np.maximum(
                111.32 * np.cos(mean_latitude), 25.0
            )
            lat_offset_p = normal_north * 75.0 / 111.32
            offsets_p = np.column_stack((lon_offset_p, lat_offset_p))
            p_center = self._sample(pressure, coordinates)
            p_cold = self._sample(pressure, coordinates - offsets_p)
            p_warm = self._sample(pressure, coordinates + offsets_p)
            pressure_trough = float(
                np.nanmedian((p_cold + p_warm) / 2.0 - p_center)
            )

        median_strength = float(np.nanmedian(line_strength))
        median_shift = float(np.nanmedian(wind_shift))
        median_convergence = float(np.nanmedian(convergence))
        median_tendency = float(np.nanmedian(line_tendency))
        median_gradient = float(np.nanmedian(magnitude))
        median_dry = float(np.nanmedian(self._sample(dry_gradient, coordinates)))
        normal_flow_kmh = float(
            np.nanmedian(center_u * normal_east + center_v * normal_north) * 3.6
        )
        propagation_kmh = -median_tendency / max(median_gradient, 0.01)
        motion_kmh = 0.7 * np.clip(propagation_kmh, -80.0, 80.0) + 0.3 * np.clip(
            normal_flow_kmh, -80.0, 80.0
        )

        if motion_kmh >= 5.0:
            front_type = "cold"
        elif motion_kmh <= -5.0:
            front_type = "warm"
        else:
            front_type = "stationary"

        # Gate 1 - contrasto termico reale in quota.  theta-e mescola
        # temperatura e umidita': un bordo visibile in theta-e ma privo di
        # gradiente di T a 850 hPa e' un confine di umidita' (brezza, sacca
        # d'aria marina, outflow), il cui contrasto vive tutto nel boundary
        # layer sottostante.  Un fronte sinottico ha sempre baroclinicita'
        # anche a 850 hPa.
        if median_dry < 2.0:
            return None

        # Gate 2 - firma dinamica.  Un fronte giace in una saccatura di
        # pressione: attraversandolo il vento ruota e converge.  Un bordo
        # termico senza alcuna risposta del vento non e' un fronte.
        if median_shift < 2.0 and median_convergence < 0.2:
            return None

        # Gate 3 - firma barica.  Una linea adagiata su un massimo di
        # pressione non puo' essere un fronte, qualunque cosa dicano gli
        # altri campi.  (Neutrale se la pressione non e' disponibile.)
        if pressure_trough is not None and pressure_trough < -0.3:
            return None

        length_km = _line_length_km(coordinates)
        thermal_score = np.clip((median_strength - threshold) / 7.0, 0.0, 1.0)
        dry_score = np.clip((median_dry - 2.0) / 6.0, 0.0, 1.0)
        shift_score = np.clip((median_shift - 1.0) / 4.5, 0.0, 1.0)
        convergence_score = np.clip((median_convergence + 0.4) / 4.5, 0.0, 1.0)
        if pressure_trough is None:
            trough_score = 0.5  # neutro: assenza di dato, non di saccatura
        else:
            trough_score = float(np.clip((pressure_trough + 0.1) / 1.2, 0.0, 1.0))
        length_score = np.clip((math.sqrt(length_km) - math.sqrt(200.0)) / 18.0, 0.0, 1.0)
        confidence = (
            0.40
            + 0.16 * thermal_score
            + 0.10 * dry_score
            + 0.10 * shift_score
            + 0.09 * convergence_score
            + 0.08 * trough_score
            + 0.07 * length_score
            + 0.05 * np.clip(abs(motion_kmh) / 25.0, 0.0, 1.0)
            - 0.38 * terrain_fraction
        )

        if confidence < 0.55:
            return None

        metrics = {
            "frontType": front_type,
            "confidence": float(np.clip(confidence, 0.0, 0.99)),
            "strength": median_strength,
            "tempGradient": median_dry,
            "windShift": median_shift,
            "convergence": median_convergence,
            "motion": motion_kmh,
            "length": length_km,
            "terrainFraction": terrain_fraction,
            "score": confidence + min(length_km, 900.0) / 5000.0,
        }
        if pressure_trough is not None:
            metrics["pressureTrough"] = pressure_trough
        return metrics

    def analyze(self, hour: int) -> dict:
        if hour not in self.hour_to_index:
            return {"type": "FeatureCollection", "features": []}

        theta = self._theta_e(hour)
        gradient_east, gradient_north = self._gradients(theta)
        gradient_magnitude = np.hypot(gradient_east, gradient_north)
        strength = gradient_magnitude * 100.0  # K / 100 km
        strength = _box_smooth(strength, self.detail_smooth_radius)
        strength_threshold = max(6.0, float(np.nanpercentile(strength, 88.0)))

        # Gradiente della temperatura secca a 850 hPa (K / 100 km): serve a
        # distinguere la vera baroclinicita' dai confini di sola umidita'.
        dry_temperature = _box_smooth(
            _box_smooth(self._field("t", hour), self.theta_smooth_radius),
            self.theta_smooth_radius,
        )
        dry_east, dry_north = self._gradients(dry_temperature)
        dry_gradient = _box_smooth(
            np.hypot(dry_east, dry_north) * 100.0,
            self.detail_smooth_radius,
        )

        # Pressione al livello del mare (hPa) per la firma della saccatura.
        pressure = None
        if "p" in self.datasets:
            try:
                raw_pressure = self._field("p", hour)
                if float(np.nanmedian(raw_pressure)) > 20000.0:
                    raw_pressure = raw_pressure / 100.0
                pressure = _box_smooth(raw_pressure, self.theta_smooth_radius)
            except Exception:
                pressure = None

        magnitude_east, magnitude_north = self._gradients(gradient_magnitude)
        tfp = -(
            magnitude_east * gradient_east + magnitude_north * gradient_north
        ) / np.maximum(gradient_magnitude, 1.0e-6)
        tfp = _box_smooth(tfp, 1)
        tfp = np.where(self.terrain > 1_500.0, np.nan, tfp)

        available = np.asarray(self.available_hours)
        previous_hour = int(
            available[
                np.argmin(
                    np.abs(available - (hour - self.tendency_window_hours))
                )
            ]
        )
        next_hour = int(
            available[
                np.argmin(
                    np.abs(available - (hour + self.tendency_window_hours))
                )
            ]
        )
        elapsed = max(next_hour - previous_hour, 1)
        tendency = (self._theta_e(next_hour) - self._theta_e(previous_hour)) / elapsed
        u_wind = _box_smooth(
            self._field("u", hour),
            self.detail_smooth_radius,
        )
        v_wind = _box_smooth(
            self._field("v", hour),
            self.detail_smooth_radius,
        )

        generator = contourpy.contour_generator(
            x=self.longitudes, y=self.latitudes, z=tfp, line_type="Separate"
        )
        candidates = []
        for segment in generator.lines(0.0):
            coordinates = np.asarray(segment, dtype=float)
            if len(coordinates) < 4:
                continue
            valid = self._sample(strength, coordinates) >= strength_threshold
            for piece in self._split_valid(coordinates, valid):
                if _line_length_km(piece) < 200.0:
                    continue
                # Gate 0 - geometria sinottica.  Un fronte separa due masse
                # d'aria: e' quasi-lineare o dolcemente arcuato.  Una linea
                # che si ripiega (forcina) o quasi si chiude (anello) e' il
                # bordo di un'anomalia locale, non un fronte.
                sinuosity, net_turn_degrees = _line_shape_metrics(piece)
                if sinuosity > 1.8 or net_turn_degrees > 150.0:
                    continue
                piece = self._orient_warm_side_left(piece, gradient_east, gradient_north)
                metrics = self._candidate_metrics(
                    piece,
                    strength,
                    gradient_east,
                    gradient_north,
                    tendency,
                    u_wind,
                    v_wind,
                    strength_threshold,
                    dry_gradient,
                    pressure,
                )
                if metrics is not None:
                    candidates.append((piece, metrics))

        candidates.sort(key=lambda item: item[1]["score"], reverse=True)
        accepted = []
        for coordinates, metrics in candidates:
            centroid = np.mean(coordinates, axis=0)
            duplicate = False
            for previous_coordinates, previous_metrics in accepted:
                previous_centroid = np.mean(previous_coordinates, axis=0)
                midpoint = np.vstack((centroid, previous_centroid))
                if (
                    _line_length_km(midpoint) < 150.0
                    and metrics["frontType"] == previous_metrics["frontType"]
                ):
                    duplicate = True
                    break
            if not duplicate:
                accepted.append((coordinates, metrics))
            if len(accepted) >= 4:
                break

        features = []
        for coordinates, metrics in accepted:
            simplified = _rdp(coordinates, 0.035)
            rounded = [
                [round(float(lon), 3), round(float(lat), 3)]
                for lon, lat in simplified
            ]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": rounded},
                    "properties": {
                        "frontType": metrics["frontType"],
                        "confidence": round(metrics["confidence"], 2),
                        "strength": round(metrics["strength"], 1),
                        "tempGradient": round(metrics["tempGradient"], 1),
                        "windShift": round(metrics["windShift"], 1),
                        "convergence": round(metrics["convergence"], 1),
                        "motionKmh": float(round(metrics["motion"], 0)),
                        "lengthKm": round(metrics["length"], 0),
                        **(
                            {"pressureTrough": round(metrics["pressureTrough"], 2)}
                            if "pressureTrough" in metrics
                            else {}
                        ),
                        "method": self.method,
                        "source": self.source,
                    },
                }
            )

        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "method": self.method,
                "source": self.source,
                "level": "850 hPa",
                "estimated": True,
                "thresholdKPer100Km": round(strength_threshold, 1),
            },
        }


    def upper_air(self, hour: int, stride: int = 2) -> dict | None:
        """Compact 850-hPa fields for the map's inspection layer.

        Exports theta-e, temperature and wind on a coarsened grid (default
        ~19 km for ICON-2I) so users can visually verify the analysed fronts
        against the raw ingredients.  Rows are ordered north to south to
        match the surface data convention of the site.
        """
        if hour not in self.hour_to_index:
            return None

        def prepare(field: np.ndarray, decimals: int) -> list:
            coarse = np.flipud(field[::stride, ::stride])
            rounded = np.round(coarse.astype(float), decimals).ravel()
            return [
                None if not np.isfinite(value) else float(value)
                for value in rounded
            ]

        theta = self._theta_e(hour)
        temperature = self._field("t", hour) - 273.15
        u_wind = self._field("u", hour)
        v_wind = self._field("v", hour)
        latitudes = self.latitudes[::stride]
        longitudes = self.longitudes[::stride]
        return {
            "level": "850 hPa",
            "nx": int(len(longitudes)),
            "ny": int(len(latitudes)),
            "lo1": float(longitudes[0]),
            "la1": float(latitudes[-1]),
            "lo2": float(longitudes[-1]),
            "la2": float(latitudes[0]),
            "dx": float(self.delta_longitude * stride),
            "dy": float(self.delta_latitude * stride),
            "thetaE": prepare(theta, 1),
            "t": prepare(temperature, 1),
            "u": prepare(u_wind, 1),
            "v": prepare(v_wind, 1),
        }


# Backwards-compatible name for older workflow revisions.
IconFrontAnalyzer = SynopticFrontAnalyzer
