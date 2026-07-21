"""ICON-2I objective synoptic-front analysis, physical rebuild (v12).

The detector is intentionally conservative.  It identifies the warm-air
edge of baroclinic zones in wet-bulb potential temperature at 850 hPa, then
requires independent dry-temperature, density and flow evidence.  Optional
925 hPa fields test low-level vertical coherence and optional 700 hPa omega
describes frontal ascent; neither can erase a well-supported front merely
because a pressure surface intersects complex terrain.

Published confidence is an internal evidence score, not a calibrated
probability and not a substitute for a forecaster's surface analysis.
"""

from __future__ import annotations

import numpy as np

import front_detection as fd
import front_locator as fl
import front_physics as fp
import front_tracking as ftk
import thermodynamics as thermo
from front_analysis import SynopticFrontAnalyzer, _blend_lines, _line_length_km, _rdp


FRONT_METHOD = "icon2i-ofa-multilevel-physical-v12"
ANALYSIS_PRESSURE_PA = 85_000.0
LOWER_PRESSURE_PA = 92_500.0

SYNOPTIC_SIGMA_KM = 100.0
REFINE_SIGMA_KM = 45.0
DERIVATIVE_SIGMA_KM = 15.0
CROSS_FRONT_KM = 55.0
PRESSURE_CROSS_KM = 100.0

TRACK_WINDOW_HOURS = 2
TRACK_GATE_KM = 170.0
TRACK_MIN_LIFETIME_HOURS = 3
TRACK_MIN_DETECTIONS = 4
TRACK_MIN_COVERAGE = 0.72
MIN_PUBLISH_QUALITY = 0.64
MAX_PUBLISH_UNCERTAINTY = 0.36
MAX_FRONTS_PER_HOUR = 4


def _finite_median(values, default=np.nan) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else float(default)


class IconSynopticFrontAnalyzer(SynopticFrontAnalyzer):
    """OFA front detector assembled around the validated ICON GRIB loader."""

    def __init__(
        self,
        *args,
        lower_temperature_path: str | None = None,
        lower_humidity_path: str | None = None,
        omega_700_path: str | None = None,
        **kwargs,
    ) -> None:
        requested_method = kwargs.get("method")
        super().__init__(*args, **kwargs)
        self.method = str(requested_method or FRONT_METHOD)
        self._theta_w_cache: dict[tuple[int, int], np.ndarray] = {}
        self._pressure_cache: dict[int, np.ndarray] = {}
        self._tracks: list[dict] | None = None
        self._by_hour: dict[int, list[tuple[np.ndarray, dict]]] | None = None
        self.analysis_summary: dict | None = None

        if lower_temperature_path and lower_humidity_path:
            added = []
            opened = None
            try:
                for key, path in (
                    ("t925", lower_temperature_path),
                    ("q925", lower_humidity_path),
                ):
                    opened = self._open_field(path, key)
                    self._validate_optional_dataset(opened, key, 925.0)
                    variable = next(iter(opened.data_vars))
                    values = np.asarray(opened[variable].values, dtype=float)
                    median = _finite_median(values)
                    if key == "t925" and not 180.0 < median < 330.0:
                        raise ValueError("T925 non espressa in kelvin")
                    if key == "q925" and not 0.0 <= median < 0.1:
                        raise ValueError("QV925 non espressa in kg/kg")
                    self.datasets[key] = opened
                    self.keys[key] = variable
                    added.append(key)
                    opened = None
            except Exception:
                if opened is not None:
                    opened.close()
                for key in added:
                    self.datasets.pop(key).close()
                    self.keys.pop(key, None)

        if omega_700_path:
            try:
                dataset = self._open_field(omega_700_path, "omega700")
                self._validate_optional_dataset(dataset, "omega700", 700.0)
                variable = next(iter(dataset.data_vars))
                values = np.asarray(dataset[variable].values, dtype=float)
                if np.nanpercentile(np.abs(values), 99.9) > 20.0:
                    raise ValueError("omega 700 hPa fuori scala Pa/s")
                self.datasets["omega700"] = dataset
                self.keys["omega700"] = variable
            except Exception:
                try:
                    dataset.close()
                except Exception:
                    pass

    def _validate_optional_dataset(
        self, dataset, name: str, expected_level_hpa: float
    ) -> None:
        level = None
        for coordinate in ("isobaricInhPa", "level"):
            if coordinate in dataset.coords:
                values = np.atleast_1d(dataset.coords[coordinate].values)
                if values.size == 1:
                    level = float(values.item())
        if level is not None and abs(level - expected_level_hpa) > 1.0:
            raise ValueError(
                f"{name} a {level:.0f} hPa invece di {expected_level_hpa:.0f} hPa"
            )
        if "step" in dataset.coords or "step" in dataset.dims:
            steps = np.atleast_1d(np.asarray(dataset["step"].values))
            reference = np.atleast_1d(np.asarray(self.datasets["t"]["step"].values))
            if steps.shape != reference.shape or not np.array_equal(steps, reference):
                raise ValueError(f"scadenze di {name} non coerenti con T850")
        for coordinate in ("latitude", "longitude"):
            optional = np.asarray(dataset[coordinate].values, dtype=float)
            reference = np.asarray(
                self.datasets["t"][coordinate].values, dtype=float
            )
            if optional.shape != reference.shape or not np.allclose(
                optional, reference, atol=1.0e-7, rtol=0.0
            ):
                raise ValueError(f"griglia di {name} non coerente con T850")

    @property
    def has_lower_level(self) -> bool:
        return "t925" in self.datasets and "q925" in self.datasets

    @staticmethod
    def _offset_points(
        coordinates: np.ndarray, normals: np.ndarray, distance_km: float
    ) -> np.ndarray:
        latitude = coordinates[:, 1]
        lon_scale = 111.32 * np.maximum(np.cos(np.deg2rad(latitude)), 0.25)
        return coordinates + np.column_stack((
            normals[:, 0] * distance_km / lon_scale,
            normals[:, 1] * distance_km / 111.32,
        ))

    @staticmethod
    def _orient_warm_left(candidate: dict) -> dict:
        """Orient every output line so the warm air is consistently left."""
        coordinates = np.asarray(candidate["coordinates"], dtype=float)
        normals = np.asarray(candidate["warmNormal"], dtype=float)
        if len(coordinates) < 2:
            return candidate
        projected = np.column_stack((
            coordinates[:, 0] * 111.32 * np.cos(np.deg2rad(coordinates[:, 1])),
            coordinates[:, 1] * 111.32,
        ))
        tangent = np.gradient(projected, axis=0)
        tangent /= np.maximum(np.hypot(tangent[:, 0], tangent[:, 1])[:, None], 1.0e-9)
        left = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        alignment = _finite_median(np.sum(left * normals, axis=1), 0.0)
        if alignment < 0.0:
            candidate = dict(candidate)
            candidate["coordinates"] = coordinates[::-1].copy()
            candidate["warmNormal"] = normals[::-1].copy()
            candidate["hewsonDir"] = np.asarray(
                candidate["hewsonDir"], dtype=float
            )[::-1].copy()
        return candidate

    def _theta_w(self, hour: int, level_hpa: int = 850) -> np.ndarray:
        key = (level_hpa, hour)
        cached = self._theta_w_cache.get(key)
        if cached is not None:
            return cached
        if level_hpa == 925:
            if not self.has_lower_level:
                raise KeyError("campi 925 hPa non disponibili")
            temperature = self._field("t925", hour)
            humidity = self._field("q925", hour)
            pressure = LOWER_PRESSURE_PA
        else:
            temperature = self._field("t", hour)
            humidity = self._field("q", hour)
            pressure = ANALYSIS_PRESSURE_PA
        fields = thermo.thermodynamic_fields(
            np.full_like(temperature, pressure),
            temperature,
            humidity,
            method="davies_jones",
        )
        result = np.where(fields["out_of_domain"], np.nan, fields["theta_w"])
        self._theta_w_cache[key] = result
        return result

    def _pressure_hpa(self, hour: int) -> np.ndarray | None:
        if "p" not in self.datasets:
            return None
        cached = self._pressure_cache.get(hour)
        if cached is not None:
            return cached
        pressure = np.asarray(self._field("p", hour), dtype=float)
        if _finite_median(pressure) > 2_000.0:
            pressure = pressure / 100.0
        self._pressure_cache[hour] = pressure
        return pressure

    def _central_tendency(self, field_getter, hour: int) -> np.ndarray | None:
        window = max(1, int(self.tendency_window_hours))
        previous = [h for h in self.available_hours if h < hour and hour - h <= window]
        following = [h for h in self.available_hours if h > hour and h - hour <= window]
        try:
            if previous and following:
                h0, h1 = max(previous), min(following)
                return (field_getter(h1) - field_getter(h0)) / float(h1 - h0)
            if following:
                h1 = min(following)
                return (field_getter(h1) - field_getter(hour)) / float(h1 - hour)
            if previous:
                h0 = max(previous)
                return (field_getter(hour) - field_getter(h0)) / float(hour - h0)
        except Exception:
            return None
        return None

    def _lower_candidates(self, hour: int) -> list[dict]:
        if not self.has_lower_level:
            return []
        lower = self._theta_w(hour, 925).copy()
        # 925 hPa intersects terrain around 750 m.  Mask model values where
        # that pressure surface is not a trustworthy low-level air-mass
        # sample; 850 hPa remains the primary geometry across mountains.
        lower[(self.terrain > 650.0) | ~np.isfinite(lower)] = np.nan
        return fd.detect_fronts_two_scale(
            lower,
            self.longitudes,
            self.latitudes,
            synoptic_sigma_km=90.0,
            refine_sigma_km=40.0,
            derivative_sigma_km=15.0,
            corridor_km=120.0,
            min_synoptic_support=0.52,
            synoptic_min_length_km=300.0,
            refine_min_length_km=180.0,
            boundary_margin_km=70.0,
        )

    def _detect_hour(self, hour: int) -> list[dict]:
        theta_w = self._theta_w(hour)
        candidates = fd.detect_fronts_two_scale(
            theta_w,
            self.longitudes,
            self.latitudes,
            synoptic_sigma_km=SYNOPTIC_SIGMA_KM,
            refine_sigma_km=REFINE_SIGMA_KM,
            derivative_sigma_km=DERIVATIVE_SIGMA_KM,
            corridor_km=110.0,
            min_synoptic_support=0.60,
            synoptic_min_length_km=350.0,
            refine_min_length_km=220.0,
            boundary_margin_km=70.0,
        )
        candidates = [self._orient_warm_left(item) for item in candidates]
        lower_candidates = self._lower_candidates(hour)
        lower_lines = [np.asarray(item["coordinates"], dtype=float) for item in lower_candidates]

        grid_metrics = fl.grid_metrics(self.longitudes, self.latitudes)
        temperature = fl.smooth_km(self._field("t", hour), REFINE_SIGMA_KM, grid_metrics)
        humidity = fl.smooth_km(self._field("q", hour), REFINE_SIGMA_KM, grid_metrics)
        theta_v = temperature * (1.0 + 0.61 * humidity) * (
            100_000.0 / ANALYSIS_PRESSURE_PA
        ) ** thermo.KAPPA
        temp_east, temp_north = fl.gradient(temperature, grid_metrics)
        dry_gradient = np.hypot(temp_east, temp_north) * 100.0

        raw_u, raw_v = self._field("u", hour), self._field("v", hour)
        u_wind = fl.smooth_km(raw_u, REFINE_SIGMA_KM, grid_metrics)
        v_wind = fl.smooth_km(raw_v, REFINE_SIGMA_KM, grid_metrics)
        kinematics = fp.kinematic_fields(
            theta_w, raw_u, raw_v, grid_metrics, smoothing_km=REFINE_SIGMA_KM
        )
        pressure = self._pressure_hpa(hour)
        if pressure is not None:
            pressure = fl.smooth_km(pressure, 80.0, grid_metrics)
        pressure_tendency = (
            self._central_tendency(
                lambda h: fl.smooth_km(
                    self._pressure_hpa(h), 80.0, grid_metrics
                ),
                hour,
            )
            if pressure is not None else None
        )
        theta_tendency = self._central_tendency(
            lambda h: fl.smooth_km(
                self._theta_w(h), REFINE_SIGMA_KM, grid_metrics
            ),
            hour,
        )
        omega = None
        if "omega700" in self.datasets:
            omega = fl.smooth_km(
                self._field("omega700", hour), 80.0, grid_metrics
            )
        theta_w_925 = None
        if self.has_lower_level:
            theta_w_925 = self._theta_w(hour, 925).copy()
            theta_w_925[self.terrain > 650.0] = np.nan

        accepted = []
        for source in candidates:
            candidate = dict(source)
            coordinates = np.asarray(candidate["coordinates"], dtype=float)
            normal = np.asarray(candidate["warmNormal"], dtype=float)
            hewson = np.asarray(candidate["hewsonDir"], dtype=float)
            if len(coordinates) < 4 or normal.shape != coordinates.shape:
                continue

            warm = self._offset_points(coordinates, normal, CROSS_FRONT_KM)
            cold = self._offset_points(coordinates, -normal, CROSS_FRONT_KM)
            validity = (
                np.isfinite(self._sample(theta_w, warm))
                & np.isfinite(self._sample(theta_w, cold))
                & np.isfinite(self._sample(temperature, warm))
                & np.isfinite(self._sample(temperature, cold))
            )
            if float(np.mean(validity)) < 0.75:
                continue

            warm_temp = self._sample(temperature, warm)
            cold_temp = self._sample(temperature, cold)
            warm_theta_v = self._sample(theta_v, warm)
            cold_theta_v = self._sample(theta_v, cold)
            warm_theta_w = self._sample(theta_w, warm)
            cold_theta_w = self._sample(theta_w, cold)

            gx = self._sample(temp_east, coordinates)
            gy = self._sample(temp_north, coordinates)
            gmag = np.maximum(np.hypot(gx, gy), 1.0e-9)
            alignment = (gx * normal[:, 0] + gy * normal[:, 1]) / gmag

            warm_u, warm_v = self._sample(u_wind, warm), self._sample(v_wind, warm)
            cold_u, cold_v = self._sample(u_wind, cold), self._sample(v_wind, cold)
            wind_valid = (
                np.isfinite(warm_u) & np.isfinite(warm_v)
                & np.isfinite(cold_u) & np.isfinite(cold_v)
            )
            if float(np.mean(wind_valid)) < 0.70:
                continue
            wind_shift = np.hypot(warm_u - cold_u, warm_v - cold_v)
            convergence = (
                (cold_u - warm_u) * normal[:, 0]
                + (cold_v - warm_v) * normal[:, 1]
            )

            center_u = self._sample(u_wind, coordinates)
            center_v = self._sample(v_wind, coordinates)
            ofa_speed = center_u * hewson[:, 0] + center_v * hewson[:, 1]

            metrics = {
                **candidate,
                "deltaThetaW": _finite_median(warm_theta_w - cold_theta_w),
                "deltaTemperature": _finite_median(warm_temp - cold_temp),
                "deltaThetaV": _finite_median(warm_theta_v - cold_theta_v),
                "dryThermalGradient": _finite_median(
                    self._sample(dry_gradient, coordinates)
                ),
                "thermalAlignment": _finite_median(alignment),
                "windShiftMs": _finite_median(wind_shift[wind_valid]),
                "convergenceMs": _finite_median(convergence[wind_valid]),
                "vorticity1e5": _finite_median(
                    self._sample(kinematics["vorticity1e5"], coordinates)
                ),
                "kinematicConvergence1e5": _finite_median(
                    self._sample(kinematics["convergence1e5"], coordinates)
                ),
                "frontogenesis": _finite_median(
                    self._sample(kinematics["frontogenesis"], coordinates)
                ),
                "thermalAdvection3h": _finite_median(
                    self._sample(kinematics["thermalAdvection3h"], coordinates)
                ),
                # Standard Hewson speed: negative cold, positive warm.
                "ofaSpeedMps": _finite_median(ofa_speed),
                "terrainFraction": float(np.mean(
                    self._sample(self.terrain, coordinates) > 1_200.0
                )),
            }

            if theta_tendency is not None:
                gradient_at_line = np.maximum(
                    self._sample(
                        np.hypot(
                            kinematics["thetaWGradientEast"],
                            kinematics["thetaWGradientNorth"],
                        ),
                        coordinates,
                    ),
                    0.005,
                )
                metrics["tendencyMotionKmh"] = _finite_median(
                    -self._sample(theta_tendency, coordinates) / gradient_at_line
                )
            else:
                metrics["tendencyMotionKmh"] = np.nan

            if pressure is not None:
                far_warm = self._offset_points(
                    coordinates, normal, PRESSURE_CROSS_KM
                )
                far_cold = self._offset_points(
                    coordinates, -normal, PRESSURE_CROSS_KM
                )
                center_p = self._sample(pressure, coordinates)
                side_p = 0.5 * (
                    self._sample(pressure, far_warm)
                    + self._sample(pressure, far_cold)
                )
                metrics["pressureTroughHpa"] = _finite_median(side_p - center_p)
                if pressure_tendency is not None:
                    cold_pt = self._sample(pressure_tendency, far_cold)
                    warm_pt = self._sample(pressure_tendency, far_warm)
                    metrics["isallobaricSupportHpa3h"] = abs(
                        _finite_median((cold_pt - warm_pt) * 3.0)
                    )
            else:
                metrics["pressureTroughHpa"] = np.nan

            if omega is not None:
                line_omega = self._sample(omega, coordinates)
                warm_omega = self._sample(omega, warm)
                metrics["omega700PaS"] = _finite_median(
                    np.fmin(line_omega, warm_omega)
                )

            if theta_w_925 is not None:
                metrics["lowerLevelSupport"] = fd.line_support_fraction(
                    coordinates, lower_lines, 130.0
                ) if lower_lines else 0.0
                metrics["deltaThetaW925"] = _finite_median(
                    self._sample(theta_w_925, warm)
                    - self._sample(theta_w_925, cold)
                )

            evidence = fp.candidate_evidence(metrics)
            metrics.update(evidence)
            if not fp.candidate_is_plausible(metrics, evidence):
                continue
            accepted.append(metrics)
        return accepted

    def _ensure_tracks(self) -> None:
        if self._by_hour is not None:
            return
        hourly: dict[int, list[dict]] = {}
        detection_errors: dict[int, str] = {}
        for hour in self.available_hours:
            try:
                hourly[hour] = self._detect_hour(hour)
            except Exception as error:
                print(f"   front diagnostics +{hour:02d}h: {error}", flush=True)
                hourly[hour] = []
                detection_errors[hour] = str(error)

        total_candidates = sum(len(items) for items in hourly.values())
        allowed_errors = max(2, int(np.ceil(len(self.available_hours) * 0.05)))
        if len(detection_errors) > allowed_errors:
            raise RuntimeError(
                "analisi frontale incompleta: "
                f"{len(detection_errors)}/{len(self.available_hours)} ore in errore"
            )
        if total_candidates == 0:
            raise RuntimeError(
                "nessun candidato OFA in tutte le scadenze: output non pubblicabile"
            )

        raw_tracks = ftk.track_fronts(
            hourly,
            window_hours=TRACK_WINDOW_HOURS,
            gate_km=TRACK_GATE_KM,
            min_lifetime_hours=TRACK_MIN_LIFETIME_HOURS,
            min_detections=TRACK_MIN_DETECTIONS,
            min_coverage=TRACK_MIN_COVERAGE,
        )
        tracks = [
            track for track in raw_tracks
            if track.get("frontType") != "uncertain"
            and track.get("qualityScore", 0.0) >= MIN_PUBLISH_QUALITY
            and track.get("uncertaintyIndex", 1.0) <= MAX_PUBLISH_UNCERTAINTY
        ]
        self.analysis_summary = {
            "hours": len(self.available_hours),
            "hoursWithCandidates": sum(bool(items) for items in hourly.values()),
            "candidateLines": total_candidates,
            "trackingTracks": len(raw_tracks),
            "publishedTracks": len(tracks),
            "detectionErrors": len(detection_errors),
            "lowerLevel925": self.has_lower_level,
            "omega700": "omega700" in self.datasets,
            "pressure": "p" in self.datasets,
        }
        print(
            "   Front QC: "
            f"{self.analysis_summary['candidateLines']} candidati, "
            f"{self.analysis_summary['trackingTracks']} tracce, "
            f"{self.analysis_summary['publishedTracks']} pubblicate, "
            f"{self.analysis_summary['detectionErrors']} errori.",
            flush=True,
        )
        self._tracks = tracks

        by_hour: dict[int, list[tuple[np.ndarray, dict]]] = {
            hour: [] for hour in self.available_hours
        }
        for track in tracks:
            properties = self._track_properties(track)
            expanded = dict(track["lines"])
            detected = sorted(track["lines"])
            for first, second in zip(detected[:-1], detected[1:]):
                gap = second - first
                if gap != 2:
                    continue
                missing = first + 1
                if missing in self.hour_to_index:
                    expanded[missing] = _blend_lines(
                        np.asarray(track["lines"][first], dtype=float),
                        np.asarray(track["lines"][second], dtype=float),
                        0.5,
                    )
            for hour, coordinates in expanded.items():
                hour_properties = dict(properties)
                hour_properties["interpolated"] = hour not in track["lines"]
                if hour_properties["interpolated"]:
                    hour_properties["uncertaintyIndex"] = round(
                        min(1.0, hour_properties["uncertaintyIndex"] + 0.06), 2
                    )
                by_hour.setdefault(hour, []).append((
                    np.asarray(coordinates, dtype=float), hour_properties
                ))
        self._by_hour = by_hour

    def _track_properties(self, track: dict) -> dict:
        return {
            "frontType": track["frontType"],
            "confidence": round(float(track["qualityScore"]), 2),
            "qualityScore": round(float(track["qualityScore"]), 2),
            "uncertaintyIndex": round(float(track["uncertaintyIndex"]), 2),
            "uncertaintyClass": track.get("uncertaintyClass", "low"),
            "motionKmh": round(float(track.get("geoMotionKmh", 0.0)), 1),
            "geoMotionKmh": track.get("geoMotionKmh"),
            "ofaSpeedKmh": track.get("ofaSpeedKmh"),
            "tendencyMotionKmh": track.get("tendencyMotionKmh"),
            "classificationCertainty": track.get("classificationCertainty"),
            "qualityComponents": track.get("qualityComponents", {}),
            "diagnostics": track.get("diagnostics", {}),
            "lifetimeH": int(track.get("lifetimeH", 0)),
            "trackId": int(track.get("id", -1)),
            "method": self.method,
            "source": self.source,
        }

    def analyze(self, hour: int) -> dict:
        base_properties = {
            "method": self.method,
            "source": self.source,
            "level": "850 hPa",
            "lowerLevelSupport": "925 hPa" if self.has_lower_level else None,
            "estimated": True,
            "uncertainty": "diagnostic-not-probabilistic",
        }
        if hour not in self.hour_to_index:
            return {"type": "FeatureCollection", "features": [], "properties": base_properties}
        self._ensure_tracks()
        entries = list(self._by_hour.get(hour, []))
        entries.sort(key=lambda item: item[1]["qualityScore"], reverse=True)
        features = []
        for coordinates, properties in entries[:MAX_FRONTS_PER_HOUR]:
            simplified = _rdp(coordinates, 0.025)
            if len(simplified) < 2 or _line_length_km(simplified) < 100.0:
                continue
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [round(float(lon), 3), round(float(lat), 3)]
                        for lon, lat in simplified
                    ],
                },
                "properties": dict(properties),
            })
        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": base_properties,
        }

    def candidate_lines(self, hour: int) -> dict:
        if hour not in self.hour_to_index:
            return {"type": "FeatureCollection", "features": []}
        features = []
        for candidate in self._detect_hour(hour):
            coordinates = _rdp(np.asarray(candidate["coordinates"]), 0.04)
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": np.round(coordinates, 2).tolist(),
                },
                "properties": {
                    "candidateEvidence": candidate.get("candidateEvidence"),
                    "deltaTemperature": round(candidate.get("deltaTemperature", 0.0), 2),
                    "windShiftMs": round(candidate.get("windShiftMs", 0.0), 2),
                },
            })
        return {"type": "FeatureCollection", "features": features}

    def upper_air(self, hour: int, stride: int = 2) -> dict | None:
        result = super().upper_air(hour, stride=stride)
        if result is None:
            return None
        coarse = np.flipud(self._theta_w(hour)[::stride, ::stride])
        result["thetaW"] = [
            None if not np.isfinite(value) else float(value)
            for value in np.round(coarse.astype(float), 1).ravel()
        ]
        result["thetaE"] = result["thetaW"]
        result["primaryThermalField"] = "thetaW"
        return result


# Explicit alias used by process_data.py.
FrontalAnalysisV12 = IconSynopticFrontAnalyzer
