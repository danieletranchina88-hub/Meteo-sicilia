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


FRONT_METHOD = "theta-e-850-ofa-v9"


def _box_sum(padded: np.ndarray, width: int) -> np.ndarray:
    """Sliding-window sum via integral image (summed-area table)."""
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    return (
        integral[width:, width:]
        - integral[:-width, width:]
        - integral[width:, :-width]
        + integral[:-width, :-width]
    )


def _box_smooth(field: np.ndarray, radius: int) -> np.ndarray:
    """Fast edge-preserving box mean, NaN-aware.

    La media di finestra e' calcolata solo sui punti finiti: un singolo
    NaN in ingresso NON contamina piu' l'intero quadrante (il vecchio
    integral image con cumsum propagava il NaN a valle).  Una cella resta
    NaN solo se meno di un quarto della finestra e' valido.  Sui campi
    senza NaN il risultato coincide con la media di box classica.
    """
    if radius <= 0:
        return np.asarray(field, dtype=float)
    values = np.asarray(field, dtype=float)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    width = radius * 2 + 1
    padded_values = np.pad(filled, ((radius, radius), (radius, radius)), mode="edge")
    padded_valid = np.pad(
        valid.astype(float), ((radius, radius), (radius, radius)), mode="edge"
    )
    window_sum = _box_sum(padded_values, width)
    window_count = _box_sum(padded_valid, width)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = window_sum / np.maximum(window_count, 1.0)
    return np.where(window_count >= 0.25 * width * width, mean, np.nan)


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


def _points_to_polyline_km(points_km: np.ndarray, line_km: np.ndarray) -> np.ndarray:
    """Minimum distance (km) from each point to a polyline's SEGMENTS.

    La distanza va misurata dai segmenti, non dai vertici: se la linea di
    riferimento e' molto semplificata (pochi vertici, come i candidati
    ECMWF dopo l'RDP), il confronto punto-punto sovrastima enormemente la
    distanza e respinge fronti in realta' coincidenti.
    """
    if len(line_km) == 1:
        return np.hypot(
            points_km[:, 0] - line_km[0, 0], points_km[:, 1] - line_km[0, 1]
        )
    starts = line_km[:-1]
    ends = line_km[1:]
    seg = ends - starts
    seg_len_sq = np.maximum((seg**2).sum(axis=1), 1.0e-9)
    rel = points_km[:, None, :] - starts[None, :, :]
    t = np.clip((rel * seg[None, :, :]).sum(axis=2) / seg_len_sq[None, :], 0.0, 1.0)
    proj = starts[None, :, :] + t[:, :, None] * seg[None, :, :]
    dist = np.hypot(
        points_km[:, None, 0] - proj[:, :, 0],
        points_km[:, None, 1] - proj[:, :, 1],
    )
    return dist.min(axis=1)


def _matched_fraction(
    coordinates: np.ndarray,
    reference_lines: list[np.ndarray],
    max_distance_km: float,
) -> float:
    """Fraction of a candidate line lying within reach of any reference line.

    Il candidato viene ricampionato a passo fisso (~20 km) e la distanza e'
    presa dai segmenti della linea di riferimento: robusto anche quando il
    riferimento ha pochi vertici.
    """
    mean_latitude = math.radians(float(np.mean(coordinates[:, 1])))
    dense = _resample_line(coordinates, max(2, int(_line_length_km(coordinates) / 20.0)))
    candidate_km = _project_km(dense, mean_latitude)
    best = np.full(len(candidate_km), np.inf)
    for reference in reference_lines:
        reference_km = _project_km(reference, mean_latitude)
        best = np.minimum(best, _points_to_polyline_km(candidate_km, reference_km))
    return float(np.mean(best <= max_distance_km))


def _resample_line(coordinates: np.ndarray, count: int = 24) -> np.ndarray:
    """Resample a polyline to a fixed number of arc-length-spaced points."""
    if len(coordinates) < 2:
        return np.repeat(coordinates, count, axis=0)[:count]
    deltas = np.diff(coordinates, axis=0)
    lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = max(cumulative[-1], 1.0e-9)
    targets = np.linspace(0.0, total, count)
    return np.column_stack(
        (
            np.interp(targets, cumulative, coordinates[:, 0]),
            np.interp(targets, cumulative, coordinates[:, 1]),
        )
    )


def _blend_lines(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    """Linear interpolation between two polylines (used to fill track gaps)."""
    a = _resample_line(first)
    b = _resample_line(second)
    forward = float(np.sum(np.hypot(*(a - b).T)))
    backward = float(np.sum(np.hypot(*(a - b[::-1]).T)))
    if backward < forward:
        b = b[::-1]
    return (1.0 - weight) * a + weight * b


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

        self._open_field = open_field

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
        self.keys = {
            key: next(iter(dataset.data_vars))
            for key, dataset in self.datasets.items()
        }
        self._validate_inputs()
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
        self.detect_cache: dict[int, dict] = {}
        self.reference_by_hour: dict[int, dict] = {}
        self.reference_radius_fn = lambda hour: 220.0
        self.tracks: list[dict] | None = None

    def _validate_inputs(self) -> None:
        """Validazione esplicita dei GRIB: livello, unita', scadenze.

        Meglio un errore chiaro all'ingresso che un risultato apparentemente
        valido ma corrotto da un'unita' sbagliata (q in g/kg mascherata dal
        clipping, geopotenziale usato come quota, ecc.).
        """
        def level_of(dataset: xr.Dataset):
            for candidate in ("isobaricInhPa", "level"):
                if candidate in dataset.coords:
                    values = np.atleast_1d(dataset.coords[candidate].values)
                    if values.size == 1:
                        return float(values.item())
            return None

        # Livello 850 hPa per i campi in quota.
        for key in ("t", "q", "u", "v"):
            level = level_of(self.datasets[key])
            if level is not None and abs(level - 850.0) > 1.0:
                raise ValueError(
                    f"Campo '{key}' a {level:.0f} hPa invece di 850 hPa"
                )

        # Temperatura in kelvin (non gia' in Celsius).
        t = self.datasets["t"][self.keys["t"]]
        t_med = float(np.nanmedian(np.asarray(t.values, dtype=float)))
        if not (180.0 < t_med < 330.0):
            raise ValueError(
                f"Temperatura 850 hPa con mediana {t_med:.1f}: attesa in kelvin"
            )

        # Umidita' specifica in kg/kg: in g/kg la mediana sarebbe ~1-15,
        # e il clipping a 0.04 la maschererebbe producendo theta-e sbagliata.
        q = self.datasets["q"][self.keys["q"]]
        q_med = float(np.nanmedian(np.asarray(q.values, dtype=float)))
        if q_med > 0.1:
            raise ValueError(
                f"Umidita' specifica con mediana {q_med:.3f}: attesa in kg/kg "
                "(sembra g/kg)"
            )

        # Orografia in metri: il geopotenziale (m^2/s^2) sarebbe ~10x e
        # spingerebbe l'intero dominio sopra le soglie orografiche.
        if "h" in self.datasets:
            h = self.datasets["h"][self.keys["h"]]
            h_max = float(np.nanmax(np.asarray(h.values, dtype=float)))
            if h_max > 9000.0:
                raise ValueError(
                    f"Orografia con massimo {h_max:.0f}: attesa in metri "
                    "(sembra geopotenziale m^2/s^2)"
                )

        # Le scadenze di T/Q/U/V devono coincidere: indici condivisi
        # altrimenti leggerebbero l'ora sbagliata per umidita' o vento.
        def steps_of(dataset: xr.Dataset):
            if "step" not in dataset.coords and "step" not in dataset.dims:
                return None
            return np.atleast_1d(np.asarray(dataset["step"].values))

        reference_steps = steps_of(self.datasets["t"])
        if reference_steps is not None:
            for key in ("q", "u", "v"):
                other = steps_of(self.datasets[key])
                if other is None or other.shape != reference_steps.shape or not np.array_equal(
                    other, reference_steps
                ):
                    raise ValueError(
                        f"Scadenze del campo '{key}' non coincidono con T"
                    )

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
        """Bilinear sample; NaN for points outside the grid.

        I punti fuori dominio devono restituire NaN, non il valore della
        cella di bordo: un sondaggio del vento/pressione a +/-45-75 km che
        cade fuori mappa, se "incollato" al bordo, inventa un salto o una
        convergenza inesistenti e genera falsi fronti ai margini.  Le
        mediane a valle usano nanmedian, e _candidate_metrics scarta le
        linee con troppi campioni non validi.
        """
        x = (coordinates[:, 0] - self.longitudes[0]) / self.delta_longitude
        y = (coordinates[:, 1] - self.latitudes[0]) / self.delta_latitude
        inside = (
            (x >= 0.0)
            & (x <= len(self.longitudes) - 1.001)
            & (y >= 0.0)
            & (y <= len(self.latitudes) - 1.001)
        )
        xc = np.clip(x, 0.0, len(self.longitudes) - 1.001)
        yc = np.clip(y, 0.0, len(self.latitudes) - 1.001)
        x0, y0 = np.floor(xc).astype(int), np.floor(yc).astype(int)
        x1 = np.minimum(x0 + 1, field.shape[1] - 1)
        y1 = np.minimum(y0 + 1, field.shape[0] - 1)
        fx, fy = xc - x0, yc - y0
        result = (
            field[y0, x0] * (1.0 - fx) * (1.0 - fy)
            + field[y0, x1] * fx * (1.0 - fy)
            + field[y1, x0] * (1.0 - fx) * fy
            + field[y1, x1] * fx * fy
        )
        return np.where(inside, result, np.nan)

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
        vorticity: np.ndarray,
        frontogenesis: np.ndarray,
        virtual_temperature: np.ndarray,
        pressure_tendency: np.ndarray | None,
    ) -> dict | None:
        line_strength = self._sample(strength, coordinates)
        # Scarta le linee troppo a ridosso del bordo: se meno del 75% dei
        # punti e' interno al dominio, le firme laterali (vento, saccatura)
        # sarebbero dominate da NaN o da estrapolazioni inaffidabili.
        if float(np.mean(np.isfinite(line_strength))) < 0.75:
            return None
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
        isallobaric_raw = None
        if pressure is not None:
            lon_offset_p = normal_east * 75.0 / np.maximum(
                111.32 * np.cos(mean_latitude), 25.0
            )
            lat_offset_p = normal_north * 75.0 / 111.32
            offsets_p = np.column_stack((lon_offset_p, lat_offset_p))
            p_center = self._sample(pressure, coordinates)
            p_cold = self._sample(pressure, coordinates - offsets_p)
            p_warm = self._sample(pressure, coordinates + offsets_p)
            trough_value = float(
                np.nanmedian((p_cold + p_warm) / 2.0 - p_center)
            )
            # NaN (tutti i sondaggi laterali fuori dominio) -> None, non un
            # valore che sfuggirebbe ai confronti dei gate.
            pressure_trough = trough_value if np.isfinite(trough_value) else None
            if pressure_tendency is not None:
                # Coppia isallobarica del passaggio frontale: la pressione
                # "decresce prima, tocca il minimo durante, aumenta dopo".
                # Quindi a ogni istante la tendenza barica sul lato che il
                # fronte sta lasciando supera quella sul lato verso cui si
                # muove.  Differenza lato freddo - lato caldo, in hPa/h.
                tendency_cold = self._sample(pressure_tendency, coordinates - offsets_p)
                tendency_warm = self._sample(pressure_tendency, coordinates + offsets_p)
                iso_value = float(np.nanmedian(tendency_cold - tendency_warm))
                isallobaric_raw = iso_value if np.isfinite(iso_value) else None

        median_strength = float(np.nanmedian(line_strength))
        median_shift = float(np.nanmedian(wind_shift))
        median_convergence = float(np.nanmedian(convergence))
        median_tendency = float(np.nanmedian(line_tendency))
        median_gradient = float(np.nanmedian(magnitude))
        median_dry = float(np.nanmedian(self._sample(dry_gradient, coordinates)))
        median_vorticity = float(np.nanmedian(self._sample(vorticity, coordinates)))
        median_frontogenesis = float(
            np.nanmedian(self._sample(frontogenesis, coordinates))
        )
        # Se una qualsiasi firma sempre-presente risulta non finita (troppi
        # campioni fuori dominio) il candidato non e' valutabile: scartare,
        # non lasciare che i confronti con NaN lo facciano passare.  I
        # valori opzionali (pressione, isallobarica) sono gia' normalizzati
        # a None a monte; signed_delta_t e' controllato al Gate 1b.
        if not all(
            np.isfinite(value)
            for value in (
                median_strength,
                median_shift,
                median_convergence,
                median_tendency,
                median_gradient,
                median_dry,
                median_vorticity,
                median_frontogenesis,
            )
        ):
            return None
        # Moto del fronte lungo la sua normale (che punta verso l'aria
        # calda).  Due misure indipendenti, con lo stesso segno per un
        # fronte reale:
        #  - PROPAGAZIONE: velocita' di traslazione dell'isolinea theta-e,
        #    dalla tendenza temporale (-d(theta-e)/dt / |grad|).  E' il moto
        #    "vero" della linea, valido anche quando il vento a 850 hPa
        #    scorre parallelo al fronte.
        #  - AVVEZIONE: componente del vento lungo la normale (metodo OFA):
        #    vento verso il caldo -> avvezione fredda -> fronte freddo.
        # Segno positivo (verso il caldo) = FREDDO; negativo = CALDO.
        propagation_kmh = float(-median_tendency / max(median_gradient, 0.01))
        advection_kmh = float(
            np.nanmedian(center_u * normal_east + center_v * normal_north) * 3.6
        )
        # La propagazione e' la misura principale del movimento (la review
        # ha ragione: il vento locale da solo classifica male i fronti che
        # avanzano non paralleli al flusso); l'avvezione la stabilizza.
        if not (np.isfinite(propagation_kmh) and np.isfinite(advection_kmh)):
            return None
        motion_kmh = 0.6 * np.clip(propagation_kmh, -80.0, 80.0) + 0.4 * np.clip(
            advection_kmh, -80.0, 80.0
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
        # anche a 850 hPa.  (Pavimento fisico per-ora: la soglia piena di
        # 2 K/100 km e' richiesta come MEDIANA sulla vita della traccia,
        # cosi' un fronte debole ma reale non viene spezzato dalle ore in
        # cui oscilla appena sotto soglia.)
        if median_dry < 1.3:
            return None

        # Gate 1b - coerenza termodinamica firmata.  Un fronte e' un
        # contrasto di DENSITA', e la densita' a pressione fissata dipende
        # dalla temperatura VIRTUALE (l'aria umida e' piu' leggera): il
        # lato caldo secondo theta-e deve essere piu' leggero anche in Tv.
        # Se i segni si oppongono nettamente (aria secca e calda contro
        # aria umida e fresca a parita' di densita') il bordo e' una
        # dryline, non una superficie frontale.
        signed_delta_t = float(
            np.nanmedian(
                self._sample(virtual_temperature, warm_points)
                - self._sample(virtual_temperature, cold_points)
            )
        )
        # Non finito (sondaggi laterali Tv fuori dominio) -> non valutabile.
        if not np.isfinite(signed_delta_t) or signed_delta_t < -0.3:
            return None

        # Gate 2 - firma dinamica.  Un fronte giace in una saccatura di
        # pressione: attraversandolo il vento ruota e converge.  Un bordo
        # termico senza alcuna risposta del vento non e' un fronte.
        # (Anche qui: pavimento per-ora, soglia piena a livello di traccia.)
        if median_shift < 1.5 and median_convergence < 0.15:
            return None

        # Gate 3 - firma barica.  Una linea adagiata su un massimo di
        # pressione non puo' essere un fronte, qualunque cosa dicano gli
        # altri campi.  (Neutrale se la pressione non e' disponibile.)
        if pressure_trough is not None and pressure_trough < -0.3:
            return None

        # Gate 4 - vorticita'.  Un fronte e' una striscia di shear
        # ciclonico: vorticita' relativa mediamente negativa (anticiclonica)
        # lungo la linea e' incompatibile con un fronte reale.
        if median_vorticity < -2.0e-5:
            return None

        length_km = _line_length_km(coordinates)
        thermal_score = np.clip((median_strength - threshold) / 7.0, 0.0, 1.0)
        dry_score = np.clip((median_dry - 2.0) / 6.0, 0.0, 1.0)
        shift_score = np.clip((median_shift - 1.0) / 4.5, 0.0, 1.0)
        convergence_score = np.clip((median_convergence + 0.4) / 4.5, 0.0, 1.0)
        vorticity_score = np.clip((median_vorticity * 1.0e5 + 1.0) / 6.0, 0.0, 1.0)
        # Frontogenesi positiva = fronte in intensificazione (attivo);
        # negativa = frontolisi, tipica dei bordi in dissoluzione.
        frontogenesis_score = np.clip((median_frontogenesis + 1.0) / 4.0, 0.0, 1.0)
        if pressure_trough is None:
            trough_score = 0.5  # neutro: assenza di dato, non di saccatura
        else:
            trough_score = float(np.clip((pressure_trough + 0.1) / 1.2, 0.0, 1.0))
        # Coppia isallobarica: per un fronte in moto la tendenza barica sul
        # lato lasciato deve superare quella sul lato d'avanzamento (in
        # hPa/3h).  Neutra per i fronti stazionari o senza dato.
        if isallobaric_raw is None or abs(motion_kmh) < 5.0:
            isallobaric_score = 0.5
        else:
            signed_couplet = math.copysign(1.0, motion_kmh) * isallobaric_raw * 3.0
            isallobaric_score = float(np.clip((signed_couplet + 0.2) / 1.2, 0.0, 1.0))
        length_score = np.clip((math.sqrt(length_km) - math.sqrt(200.0)) / 18.0, 0.0, 1.0)
        confidence = (
            0.34
            + 0.12 * thermal_score
            + 0.08 * dry_score
            + 0.10 * shift_score
            + 0.07 * convergence_score
            + 0.07 * vorticity_score
            + 0.06 * frontogenesis_score
            + 0.07 * trough_score
            + 0.05 * isallobaric_score
            + 0.06 * length_score
            + 0.03 * np.clip(abs(motion_kmh) / 25.0, 0.0, 1.0)
            - 0.38 * terrain_fraction
        )

        # Pavimento per-ora: la soglia di pubblicazione (0.55) e' richiesta
        # come mediana sulla vita della traccia.
        if confidence < 0.50:
            return None

        metrics = {
            "frontType": front_type,
            "confidence": float(np.clip(confidence, 0.0, 0.99)),
            "strength": median_strength,
            "tempGradient": median_dry,
            "deltaT": signed_delta_t,
            "windShift": median_shift,
            "convergence": median_convergence,
            "vorticity": median_vorticity,
            "frontogenesis": median_frontogenesis,
            "motion": motion_kmh,
            "length": length_km,
            "terrainFraction": terrain_fraction,
            "score": confidence + min(length_km, 900.0) / 5000.0,
        }
        if pressure_trough is not None:
            metrics["pressureTrough"] = pressure_trough
        if isallobaric_raw is not None:
            metrics["isallobaric3h"] = isallobaric_raw * 3.0
        return metrics

    def _detect(self, hour: int) -> dict:
        """Rilevamento grezzo per una singola ora (tutti i gate locali).

        Il risultato e' in cache: la verifica di coerenza temporale in
        analyze() riusa i candidati delle ore adiacenti senza ricalcolarli.
        """
        if hour in self.detect_cache:
            return self.detect_cache[hour]

        theta = self._theta_e(hour)
        gradient_east, gradient_north = self._gradients(theta)
        gradient_magnitude = np.hypot(gradient_east, gradient_north)
        strength = gradient_magnitude * 100.0  # K / 100 km
        strength = _box_smooth(strength, self.detail_smooth_radius)
        # Soglia del gradiente OFA standard: 4 K/100 km e' il valore usato
        # dai centri operativi per separare la zona frontale dalle masse
        # d'aria omogenee.  Resta adattiva verso l'alto (percentile 85) per
        # non tracciare troppi rami nelle scene molto barocline.
        strength_threshold = max(4.0, float(np.nanpercentile(strength, 85.0)))

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

        # Temperatura virtuale (densita' a pressione fissata): l'aria umida
        # e' piu' leggera, quindi il confronto di densita' tra i due lati
        # del fronte deve tenere conto dell'umidita'.
        virtual_temperature = _box_smooth(
            _box_smooth(
                self._field("t", hour)
                * (1.0 + 0.61 * np.clip(self._field("q", hour), 0.0, 0.04)),
                self.theta_smooth_radius,
            ),
            self.theta_smooth_radius,
        )

        # Tendenza barica (hPa/h) per la coppia isallobarica del passaggio.
        pressure_tendency = None
        if pressure is not None and next_hour != previous_hour:
            try:
                p_prev = self._field("p", previous_hour)
                p_next = self._field("p", next_hour)
                if float(np.nanmedian(p_prev)) > 20000.0:
                    p_prev = p_prev / 100.0
                if float(np.nanmedian(p_next)) > 20000.0:
                    p_next = p_next / 100.0
                pressure_tendency = _box_smooth(
                    (p_next - p_prev) / elapsed,
                    self.theta_smooth_radius,
                )
            except Exception:
                pressure_tendency = None

        u_wind = _box_smooth(
            self._field("u", hour),
            self.detail_smooth_radius,
        )
        v_wind = _box_smooth(
            self._field("v", hour),
            self.detail_smooth_radius,
        )

        # Vorticita' relativa (s^-1): un fronte e' una striscia di shear
        # ciclonico.  Frontogenesi cinematica di Petterssen su theta-e
        # (K/100 km/3 h): positiva se il flusso sta stringendo il gradiente
        # (fronte attivo), negativa in frontolisi.
        u_east, u_north = self._gradients(u_wind)
        v_east, v_north = self._gradients(v_wind)
        vorticity = _box_smooth(
            (v_east - u_north) / 1000.0,
            self.detail_smooth_radius,
        )
        safe_magnitude = np.maximum(gradient_magnitude, 1.0e-6)
        frontogenesis = -(
            u_east * gradient_east**2
            + v_north * gradient_north**2
            + (v_east + u_north) * gradient_east * gradient_north
        ) / (safe_magnitude * 1000.0)
        frontogenesis = _box_smooth(
            frontogenesis * 100.0 * 10800.0,
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
                if _line_length_km(piece) < 150.0:
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
                    vorticity,
                    frontogenesis,
                    virtual_temperature,
                    pressure_tendency,
                )
                if metrics is not None:
                    candidates.append((piece, metrics))

        candidates.sort(key=lambda item: item[1]["score"], reverse=True)
        result = {"candidates": candidates, "threshold": strength_threshold}
        self.detect_cache[hour] = result
        return result

    def set_reference(self, reference_by_hour: dict, radius_km_fn=None) -> None:
        """Guida di un modello indipendente, valutata a livello di traccia.

        ``reference_by_hour`` mappa l'ora di previsione sulla
        FeatureCollection dei fronti di riferimento alla stessa validita'.
        La conferma per singola ora creerebbe sfarfallio quando la guida
        (a passo piu' rado) cambia scadenza: per questo il consenso e'
        richiesto sull'insieme della vita del fronte, non ora per ora.
        """
        self.reference_by_hour = reference_by_hour or {}
        if radius_km_fn is not None:
            self.reference_radius_fn = radius_km_fn
        self.tracks = None

    def _build_tracks(self) -> list[dict]:
        """Collega i candidati orari in tracce e accetta a livello di traccia.

        Un fronte e' un oggetto che vive nel tempo, non 73 decisioni
        indipendenti: l'accettazione richiede vita minima, fiducia mediana
        sull'intera vita e (se disponibile) consenso complessivo del
        modello guida.  E' questo che garantisce un output coerente:
        niente fronti che compaiono e scompaiono di ora in ora.
        """
        if self.tracks is not None:
            return self.tracks

        window = self.tendency_window_hours
        # Raggio di collegamento: velocita' frontale massima plausibile
        # (~70 km/h) per l'intervallo, ma con un TETTO fisso a 200 km.  Un
        # raggio che cresce senza limite (90 km/h x gap = 540 km a 6 h)
        # fonderebbe fronti diversi dello stesso ciclone; il tetto lo
        # impedisce.  Serve inoltre che almeno META' della linea coincida
        # (non il 35%), e la sovrapposizione e' verificata da entrambe le
        # parti (candidato->traccia e traccia->candidato).
        def link_radius(gap_hours: int) -> float:
            return min(200.0, 60.0 + 70.0 * gap_hours)

        tracks: list[dict] = []
        for hour in self.available_hours:
            for coordinates, metrics in self._detect(hour)["candidates"]:
                best_track, best_fraction = None, 0.0
                for track in tracks:
                    last_hour = track["hours"][-1]
                    gap = hour - last_hour
                    if gap <= 0 or gap > 2 * window:
                        continue
                    last_coords = track["lines"][last_hour][0]
                    radius = link_radius(gap)
                    forward = _matched_fraction(coordinates, [last_coords], radius)
                    backward = _matched_fraction(last_coords, [coordinates], radius)
                    overlap = min(forward, backward)
                    if overlap > best_fraction:
                        best_track, best_fraction = track, overlap
                if (
                    best_track is not None
                    and best_fraction >= 0.5
                    and hour not in best_track["lines"]
                ):
                    best_track["lines"][hour] = (coordinates, metrics)
                    best_track["hours"].append(hour)
                else:
                    tracks.append({"hours": [hour], "lines": {hour: (coordinates, metrics)}})

        available_sorted = sorted(self.available_hours)
        accepted = []
        for track in tracks:
            hours = track["hours"]
            span = hours[-1] - hours[0]
            # Vita minima: due rilevamenti distinti su un arco di almeno
            # due finestre (6 h per ICON-2I, 12 h per ECMWF).
            if len(hours) < 2 or span < 2 * window:
                continue
            # Copertura: un fronte vero e' rilevato quasi sempre durante la
            # sua vita; un bordo diurno (convezione pomeridiana, brezze)
            # riappare a grappoli con lunghi vuoti notturni.  Richiediamo
            # che i rilevamenti coprano almeno meta' delle scadenze REALI
            # nell'arco della traccia (contate sulle scadenze disponibili,
            # non su un passo assunto: robusto a scadenze irregolari).
            expected_hours = [
                h for h in available_sorted if hours[0] <= h <= hours[-1]
            ]
            if len(hours) / max(len(expected_hours), 1) < 0.5:
                continue
            confidences = [track["lines"][h][1]["confidence"] for h in hours]
            if float(np.median(confidences)) < 0.55:
                continue
            # Le soglie piene di baroclinicita' e risposta del vento valgono
            # come mediane sulla vita della traccia: severe sul fronte nel
            # suo complesso, tolleranti verso l'oscillazione della singola
            # ora (che altrimenti spezzerebbe la traccia).
            if float(np.median(
                [track["lines"][h][1]["tempGradient"] for h in hours]
            )) < 2.0:
                continue
            median_shift_track = float(np.median(
                [track["lines"][h][1]["windShift"] for h in hours]
            ))
            median_conv_track = float(np.median(
                [track["lines"][h][1]["convergence"] for h in hours]
            ))
            if median_shift_track < 2.0 and median_conv_track < 0.2:
                continue
            # Conferma del modello guida.  Se il riferimento e' disponibile
            # per l'arco della traccia, richiede consenso; se non lo e'
            # (fail-open), la traccia passa ma resta marcata come NON
            # confermata e con doppia severita' sulle firme fisiche, cosi'
            # l'assenza della guida non spalanca le porte agli artefatti.
            track["corroborated"] = None
            if self.reference_by_hour:
                reference_hours = [
                    h
                    for h in hours
                    if (self.reference_by_hour.get(h) or {}).get("features")
                ]
                if reference_hours:
                    corroborated = 0
                    for h in reference_hours:
                        reference_lines = [
                            np.asarray(f["geometry"]["coordinates"], dtype=float)
                            for f in self.reference_by_hour[h]["features"]
                        ]
                        if _matched_fraction(
                            track["lines"][h][0],
                            reference_lines,
                            float(self.reference_radius_fn(h)),
                        ) >= 0.5:
                            corroborated += 1
                    track["corroborated"] = corroborated / len(reference_hours) >= 0.5
                    if not track["corroborated"]:
                        continue
            # Se la guida e' del tutto assente (corroborated resta None) la
            # traccia passa sulle sole firme fisiche, ma viene marcata come
            # non confermata e la sua confidenza e' penalizzata in output:
            # l'assenza della guida e' resa esplicita, non nascosta.
            # Moto smussato sulla vita della traccia: il tipo del fronte
            # deriva da questo, quindi non puo' sfarfallare tra un'ora e
            # l'altra per rumore.
            track["smoothed_motion"] = {}
            for h in hours:
                neighbourhood = [
                    track["lines"][k][1]["motion"]
                    for k in hours
                    if abs(k - h) <= 2 * window
                ]
                track["smoothed_motion"][h] = float(np.median(neighbourhood))
            track["score"] = float(
                np.median([track["lines"][h][1]["score"] for h in hours])
            )
            track["span"] = span
            accepted.append(track)

        accepted.sort(key=lambda t: t["score"], reverse=True)
        self.tracks = accepted
        return accepted

    def _track_state_at(self, track: dict, hour: int):
        """Line and metrics of a track at an hour, interpolating short gaps."""
        hours = track["hours"]
        if hour in track["lines"]:
            coordinates, metrics = track["lines"][hour]
            metrics = dict(metrics)
            metrics["interpolated"] = False
        else:
            if hour < hours[0] or hour > hours[-1]:
                return None
            previous_hour = max(h for h in hours if h < hour)
            next_hour = min(h for h in hours if h > hour)
            if next_hour - previous_hour > self.tendency_window_hours:
                return None
            weight = (hour - previous_hour) / (next_hour - previous_hour)
            before, before_metrics = track["lines"][previous_hour]
            after, after_metrics = track["lines"][next_hour]
            coordinates = _blend_lines(before, after, weight)
            metrics = dict(before_metrics if weight < 0.5 else after_metrics)
            metrics["confidence"] = float(
                min(before_metrics["confidence"], after_metrics["confidence"]) * 0.92
            )
            metrics["interpolated"] = True

        motion_hours = sorted(track["smoothed_motion"])
        nearest = min(motion_hours, key=lambda h: abs(h - hour))
        smoothed = track["smoothed_motion"][nearest]
        metrics["motion"] = smoothed
        if smoothed >= 5.0:
            metrics["frontType"] = "cold"
        elif smoothed <= -5.0:
            metrics["frontType"] = "warm"
        else:
            metrics["frontType"] = "stationary"
        metrics["lifetimeH"] = track["span"]
        # Stato di conferma del modello guida (None = guida assente per
        # questa traccia).  Se non confermata, penalizza la confidenza.
        corroborated = track.get("corroborated")
        metrics["corroborated"] = corroborated
        if corroborated is None:
            metrics["confidence"] = float(metrics["confidence"] * 0.9)
        return coordinates, metrics

    def analyze(self, hour: int) -> dict:
        if hour not in self.hour_to_index:
            return {"type": "FeatureCollection", "features": []}

        strength_threshold = self._detect(hour)["threshold"]
        entries = []
        for track in self._build_tracks():
            state = self._track_state_at(track, hour)
            if state is not None:
                entries.append(state)

        accepted = []
        for coordinates, metrics in entries:
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
                        "vorticity1e5": round(metrics["vorticity"] * 1.0e5, 1),
                        "frontogenesis": round(metrics["frontogenesis"], 1),
                        "motionKmh": float(round(metrics["motion"], 0)),
                        "lengthKm": round(metrics["length"], 0),
                        "lifetimeH": int(metrics.get("lifetimeH", 0)),
                        "interpolated": bool(metrics.get("interpolated", False)),
                        "corroborated": (
                            None
                            if metrics.get("corroborated") is None
                            else bool(metrics.get("corroborated"))
                        ),
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


    def candidate_lines(self, hour: int) -> dict:
        """Candidati di rilevamento (pre-tracciamento) come FeatureCollection.

        Servono da riferimento per la conferma incrociata di un altro
        modello: i fronti PUBBLICATI sono volutamente pochi (tracciamento
        severo), ma come guida serve l'insieme piu' ricco di cio' che il
        modello vede a quella scadenza, altrimenti la conferma diventa
        quasi sempre non applicabile.
        """
        if hour not in self.hour_to_index:
            return {"type": "FeatureCollection", "features": []}
        features = []
        for coordinates, metrics in self._detect(hour)["candidates"]:
            simplified = _rdp(coordinates, 0.05)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [round(float(lon), 2), round(float(lat), 2)]
                            for lon, lat in simplified
                        ],
                    },
                    "properties": {"frontType": metrics["frontType"]},
                }
            )
        return {"type": "FeatureCollection", "features": features}

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
