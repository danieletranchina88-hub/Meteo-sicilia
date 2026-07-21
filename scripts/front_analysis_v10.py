"""Objective Frontal Analysis v10 - drop-in analyzer for the pipeline.

This is the v10 assembly.  It deliberately reuses the *validated v9
infrastructure* (GRIB reading, grid setup, input validation, bilinear
sampling, the compact 850-hPa ``upper_air`` export and ``close``) by
subclassing :class:`SynopticFrontAnalyzer`, and replaces the scientific
core with the v10 modules:

  * :mod:`thermodynamics`  - wet-bulb potential temperature theta_w
    (Davies-Jones 2008), computed at the analysis level (850 hPa);
  * :mod:`front_detection` - two-scale detection (a strongly smoothed
    synoptic field as a structural prior + a lightly smoothed field for
    the refined geometry), so isolated ICON-2I mesoscale gradients are
    not published as synoptic fronts;
  * :mod:`front_tracking`  - global (Hungarian) association across the
    whole forecast, geometric motion as the primary classification
    signal, wind advection / OFA speed as cross-checks, consensus
    classification (``uncertain`` on real conflict), and a
    ``qualityScore`` (a physical-support heuristic, NOT a probability).

The public interface matches v9 exactly - ``available_hours``,
``set_reference``, ``analyze(hour)``, ``upper_air(hour)``, ``close()`` -
so ``process_data.py`` can switch to it without any other change.

Product objective (site wording): "una rappresentazione automatica,
fisicamente coerente e temporalmente stabile delle principali strutture
frontali sinottiche previste dai modelli" - fronts are *estimated*, not
the official subjective analysis.
"""

from __future__ import annotations

import numpy as np

import front_detection as fd
import front_locator as fl
import front_tracking as ftk
import thermodynamics as thermo
from front_analysis import (
    SynopticFrontAnalyzer,
    _blend_lines,
    _line_length_km,
    _line_shape_metrics,
    _rdp,
)

# Analysis level in Pa (850 hPa): theta_w is computed on the constant
# pressure surface the fields live on, not on PMSL.
ANALYSIS_PRESSURE_PA = 85000.0

V10_METHOD = "OFA v10 (theta_w Hewson, due scale, tracking)"

# Detection / tracking scales.  Tuned so the synoptic prior filters ICON-2I
# mesoscale noise while the refined scale keeps a realistic frontal shape.
SYNOPTIC_SIGMA_KM = 150.0
REFINE_SIGMA_KM = 50.0
DERIVATIVE_SIGMA_KM = 20.0
CORRIDOR_KM = 140.0
SYNOPTIC_MIN_LENGTH_KM = 450.0
REFINE_MIN_LENGTH_KM = 280.0
ABZ_GRADIENT_THRESHOLD = 1.8

# Independent physical signatures. theta_w identifies the air-mass
# boundary; these fields reject humidity-only, orographic and dynamically
# inactive boundaries.
DRY_SMOOTH_KM = 50.0
CROSS_FRONT_DISTANCE_KM = 45.0
PRESSURE_DISTANCE_KM = 90.0
TERRAIN_LIMIT_METERS = 900.0
MAX_TERRAIN_FRACTION = 0.25

# Track-level cross-model requirements.
REFERENCE_MIN_LINE_MATCH = 0.55
REFERENCE_MIN_HOUR_FRACTION = 0.60
REFERENCE_MIN_COVERAGE = 0.50

TRACK_WINDOW_HOURS = 3
TRACK_GATE_KM = 250.0
TRACK_MIN_LIFETIME_HOURS = 6
TRACK_MIN_COVERAGE = 0.5

# Published fronts per hour: keep the map readable, drop the weakest.
MAX_FRONTS_PER_HOUR = 4
# A track must clear this quality floor to be published at all.
MIN_PUBLISH_QUALITY = 0.50
MIN_CROSS_MODEL_QUALITY = 0.58


class FrontalAnalysisV10(SynopticFrontAnalyzer):
    """v10 analyzer: v9 data infrastructure, Hewson/OFA scientific core."""

    def __init__(self, *args, **kwargs) -> None:
        requested_method = kwargs.get("method")
        self.require_reference = bool(kwargs.pop("require_reference", False))
        self.reference_source = str(kwargs.pop("reference_source", "ECMWF IFS"))
        # Reuse the whole v9 loader (open GRIB, validate units/levels,
        # build the grid, dx/dy, available_hours, terrain, upper_air).
        super().__init__(*args, **kwargs)
        self.method = str(requested_method or V10_METHOD)
        self._theta_w_cache: dict[int, np.ndarray] = {}
        self._wind_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._v10_tracks: list[dict] | None = None
        # per hour -> list of (coordinates, properties) ready to publish
        self._v10_by_hour: dict[int, list[tuple[np.ndarray, dict]]] | None = None

    @staticmethod
    def _offset_points(
        coordinates: np.ndarray, normal: np.ndarray, distance_km: float
    ) -> np.ndarray:
        """Move lon/lat points by a physical east/north offset."""
        latitude = coordinates[:, 1]
        lon_scale = 111.32 * np.maximum(np.cos(np.deg2rad(latitude)), 0.25)
        return coordinates + np.column_stack((
            normal[:, 0] * distance_km / lon_scale,
            normal[:, 1] * distance_km / 111.32,
        ))

    # -- scientific fields --------------------------------------------------
    def _theta_w(self, hour: int) -> np.ndarray:
        """Wet-bulb potential temperature (K) at the analysis level."""
        cached = self._theta_w_cache.get(hour)
        if cached is not None:
            return cached
        temperature_k = self._field("t", hour)
        specific_humidity = self._field("q", hour)
        pressure = np.full_like(temperature_k, ANALYSIS_PRESSURE_PA)
        fields = thermo.thermodynamic_fields(
            pressure, temperature_k, specific_humidity, method="davies_jones"
        )
        theta_w = fields["theta_w"]
        theta_w = np.where(fields["out_of_domain"], np.nan, theta_w)
        self._theta_w_cache[hour] = theta_w
        return theta_w

    def _winds(self, hour: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._wind_cache.get(hour)
        if cached is not None:
            return cached
        result = (self._field("u", hour), self._field("v", hour))
        self._wind_cache[hour] = result
        return result

    def _wind_sampler(self, hour: int, points_lonlat: np.ndarray):
        """(u, v) in m/s at the given lon/lat points; NaN outside the grid."""
        u_field, v_field = self._winds(hour)
        u = self._sample(u_field, points_lonlat)
        v = self._sample(v_field, points_lonlat)
        return u, v

    # -- detection + tracking (computed once) ------------------------------
    def _detect_hour(self, hour: int) -> list[dict]:
        theta_w = self._theta_w(hour)
        candidates = fd.detect_fronts_two_scale(
            theta_w,
            self.longitudes,
            self.latitudes,
            synoptic_sigma_km=SYNOPTIC_SIGMA_KM,
            refine_sigma_km=REFINE_SIGMA_KM,
            derivative_sigma_km=DERIVATIVE_SIGMA_KM,
            corridor_km=CORRIDOR_KM,
            synoptic_min_length_km=SYNOPTIC_MIN_LENGTH_KM,
            refine_min_length_km=REFINE_MIN_LENGTH_KM,
            abz_gradient_threshold=ABZ_GRADIENT_THRESHOLD,
        )

        # A theta_w edge without a dry temperature contrast is commonly a
        # humidity boundary. Add independent T850 and wind diagnostics.
        metrics = fl.grid_metrics(self.longitudes, self.latitudes)
        temperature = fl.smooth_km(
            self._field("t", hour), DRY_SMOOTH_KM, metrics
        )
        temp_east, temp_north = fl.gradient(temperature, metrics)
        dry_gradient = np.hypot(temp_east, temp_north) * 100.0
        u_field, v_field = self._winds(hour)
        pressure_field = self._field("p", hour) if "p" in self.datasets else None

        accepted = []
        for candidate in candidates:
            coordinates = np.asarray(candidate["coordinates"], dtype=float)
            normal = np.asarray(candidate["warmNormal"], dtype=float)
            if len(coordinates) < 4 or normal.shape != coordinates.shape:
                continue

            sinuosity, net_turn = _line_shape_metrics(coordinates)
            if sinuosity > 1.8 or net_turn > 145.0:
                continue

            terrain = self._sample(self.terrain, coordinates)
            terrain_valid = np.isfinite(terrain)
            if float(np.mean(terrain_valid)) < 0.80:
                continue
            terrain_fraction = float(np.mean(
                terrain_valid & (terrain > TERRAIN_LIMIT_METERS)
            ))
            if terrain_fraction > MAX_TERRAIN_FRACTION:
                continue

            dry_support = self._sample(dry_gradient, coordinates)
            if float(np.mean(np.isfinite(dry_support))) < 0.80:
                continue
            candidate["dryThermalGradient"] = float(np.nanmedian(dry_support))
            candidate["terrainFraction"] = terrain_fraction
            candidate["sinuosity"] = float(sinuosity)
            candidate["netTurnDeg"] = float(net_turn)

            warm_points = self._offset_points(
                coordinates, normal, CROSS_FRONT_DISTANCE_KM
            )
            cold_points = self._offset_points(
                coordinates, -normal, CROSS_FRONT_DISTANCE_KM
            )
            warm_u = self._sample(u_field, warm_points)
            warm_v = self._sample(v_field, warm_points)
            cold_u = self._sample(u_field, cold_points)
            cold_v = self._sample(v_field, cold_points)
            wind_valid = (
                np.isfinite(warm_u) & np.isfinite(warm_v)
                & np.isfinite(cold_u) & np.isfinite(cold_v)
            )
            if float(np.mean(wind_valid)) < 0.70:
                continue
            wind_shift = np.hypot(warm_u - cold_u, warm_v - cold_v)
            cold_normal_flow = cold_u * normal[:, 0] + cold_v * normal[:, 1]
            warm_normal_flow = warm_u * normal[:, 0] + warm_v * normal[:, 1]
            candidate["windShiftMs"] = float(np.nanmedian(wind_shift[wind_valid]))
            candidate["convergenceMs"] = float(np.nanmedian(
                (cold_normal_flow - warm_normal_flow)[wind_valid]
            ))

            if pressure_field is not None:
                far_warm = self._offset_points(
                    coordinates, normal, PRESSURE_DISTANCE_KM
                )
                far_cold = self._offset_points(
                    coordinates, -normal, PRESSURE_DISTANCE_KM
                )
                pressure_line = self._sample(pressure_field, coordinates)
                pressure_sides = 0.5 * (
                    self._sample(pressure_field, far_warm)
                    + self._sample(pressure_field, far_cold)
                )
                valid_pressure = np.isfinite(pressure_line) & np.isfinite(pressure_sides)
                if np.any(valid_pressure):
                    median_pressure = float(np.median(pressure_line[valid_pressure]))
                    pressure_scale = 100.0 if median_pressure > 2_000 else 1.0
                    candidate["pressureTroughHpa"] = float(np.median(
                        (pressure_sides[valid_pressure] - pressure_line[valid_pressure])
                        / pressure_scale
                    ))
                else:
                    candidate["pressureTroughHpa"] = np.nan
            else:
                candidate["pressureTroughHpa"] = np.nan

            accepted.append(candidate)
        return accepted

    def _reference_diagnostics(self, track: dict) -> dict:
        """Compare a whole track with independent-model published fronts."""
        return ftk.cross_model_diagnostics(
            track,
            self.reference_by_hour,
            self.reference_radius_fn,
            min_line_match=REFERENCE_MIN_LINE_MATCH,
            min_hour_fraction=REFERENCE_MIN_HOUR_FRACTION,
            min_coverage=REFERENCE_MIN_COVERAGE,
        )

    @staticmethod
    def _apply_reference_quality(track: dict, reference: dict) -> None:
        components = dict(track.get("qualityComponents") or {})
        components["modelAgreement"] = reference["agreement"]
        track["qualityComponents"] = components
        track["modelAgreement"] = reference["agreement"]
        track["corroborated"] = reference["confirmed"]
        track["referenceCoverage"] = reference["coverage"]
        track["referenceMatchedHours"] = reference["matchedHourFraction"]
        track["referenceFrontType"] = reference["referenceFrontType"]
        track["qualityScore"] = round(float(np.clip(
            0.25 * float(components.get("thermalSupport") or 0.0)
            + 0.20 * float(components.get("dynamicSupport") or 0.0)
            + 0.20 * float(components.get("temporalSupport") or 0.0)
            + 0.10 * float(components.get("structuralSupport") or 0.0)
            + 0.15 * float(reference["agreement"])
            + 0.10 * float(components.get("classificationCertainty") or 0.0),
            0.0,
            1.0,
        )), 2)

    def _ensure_tracks(self) -> None:
        if self._v10_by_hour is not None:
            return
        hourly: dict[int, list[dict]] = {}
        for hour in self.available_hours:
            try:
                hourly[hour] = self._detect_hour(hour)
            except Exception:
                hourly[hour] = []

        tracks = ftk.track_fronts(
            hourly,
            window_hours=TRACK_WINDOW_HOURS,
            gate_km=TRACK_GATE_KM,
            min_lifetime_hours=TRACK_MIN_LIFETIME_HOURS,
            min_coverage=TRACK_MIN_COVERAGE,
            wind_sampler=self._wind_sampler,
            require_physical_support=True,
            max_terrain_fraction=MAX_TERRAIN_FRACTION,
        )
        filtered_tracks = []
        for track in tracks:
            reference = self._reference_diagnostics(track)
            if reference["available"]:
                self._apply_reference_quality(track, reference)
            elif self.require_reference:
                continue
            if track.get("frontType") == "uncertain":
                continue
            if self.require_reference and not track.get("corroborated", False):
                continue
            quality_floor = (
                MIN_CROSS_MODEL_QUALITY
                if self.require_reference
                else MIN_PUBLISH_QUALITY
            )
            if track["qualityScore"] < quality_floor:
                continue
            filtered_tracks.append(track)
        tracks = filtered_tracks
        self._v10_tracks = tracks

        by_hour: dict[int, list[tuple[np.ndarray, dict]]] = {
            hour: [] for hour in self.available_hours
        }
        for track in tracks:
            properties = self._track_properties(track)
            expanded_lines = dict(track["lines"])
            detected_hours = sorted(track["lines"])
            for first_hour, second_hour in zip(detected_hours[:-1], detected_hours[1:]):
                gap = second_hour - first_hour
                if gap <= 1 or gap > TRACK_WINDOW_HOURS:
                    continue
                for missing_hour in range(first_hour + 1, second_hour):
                    if missing_hour not in self.hour_to_index:
                        continue
                    weight = (missing_hour - first_hour) / gap
                    expanded_lines[missing_hour] = _blend_lines(
                        np.asarray(track["lines"][first_hour], dtype=float),
                        np.asarray(track["lines"][second_hour], dtype=float),
                        weight,
                    )
            for hour, coordinates in expanded_lines.items():
                coordinates = np.asarray(coordinates, dtype=float)
                if len(coordinates) < 2:
                    continue
                hour_properties = dict(properties)
                hour_properties["interpolated"] = hour not in track["lines"]
                by_hour.setdefault(hour, []).append((coordinates, hour_properties))
        self._v10_by_hour = by_hour

    def _track_properties(self, track: dict) -> dict:
        """Map a v10 track to the front feature properties the site reads.

        ``confidence`` is fed by ``qualityScore`` so the map's opacity keeps
        working; ``motionKmh`` is the geometric line speed (positive = toward
        the warm air).  v10-specific fields are added for transparency.
        """
        components = track.get("qualityComponents", {})
        return {
            "frontType": track["frontType"],
            "confidence": round(float(track["qualityScore"]), 2),
            "qualityScore": round(float(track["qualityScore"]), 2),
            "motionKmh": float(round(track.get("geoMotionKmh", 0.0), 0)),
            "geoMotionKmh": track.get("geoMotionKmh"),
            "advectionKmh": track.get("advectionKmh"),
            "ofaSpeedKmh": track.get("ofaSpeedKmh"),
            "classificationCertainty": track.get("classificationCertainty"),
            "qualityComponents": components,
            "corroborated": track.get("corroborated"),
            "modelAgreement": track.get("modelAgreement"),
            "referenceCoverage": track.get("referenceCoverage"),
            "referenceMatchedHours": track.get("referenceMatchedHours"),
            "referenceFrontType": track.get("referenceFrontType"),
            "referenceSource": (
                self.reference_source if track.get("corroborated") else None
            ),
            "diagnostics": track.get("diagnostics", {}),
            "lifetimeH": int(track.get("lifetimeH", 0)),
            "trackId": int(track.get("id", -1)),
            "method": self.method,
            "source": self.source,
        }

    # -- public API (same shape as v9) -------------------------------------
    def analyze(self, hour: int) -> dict:
        empty = {
            "type": "FeatureCollection",
            "features": [],
            "properties": {
                "method": self.method,
                "source": self.source,
                "level": "850 hPa",
                "estimated": True,
            },
        }
        if hour not in self.hour_to_index:
            return empty
        self._ensure_tracks()
        entries = list(self._v10_by_hour.get(hour, []))
        # strongest first, cap the count
        entries.sort(key=lambda item: item[1]["qualityScore"], reverse=True)
        entries = entries[:MAX_FRONTS_PER_HOUR]

        features = []
        for coordinates, properties in entries:
            simplified = _rdp(coordinates, 0.035)
            if len(simplified) < 2 or _line_length_km(simplified) < 50.0:
                continue
            rounded = [
                [round(float(lon), 3), round(float(lat), 3)]
                for lon, lat in simplified
            ]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": rounded},
                    "properties": dict(properties),
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
            },
        }

    def set_reference(self, reference_by_hour: dict, radius_km_fn=None) -> None:
        """Set the independent-model guide used as a strict track filter."""
        self.reference_by_hour = reference_by_hour or {}
        if radius_km_fn is not None:
            self.reference_radius_fn = radius_km_fn
        self._v10_tracks = None
        self._v10_by_hour = None

    def candidate_lines(self, hour: int) -> dict:
        """Return pre-tracking candidates for diagnostics only."""
        if hour not in self.hour_to_index:
            return {"type": "FeatureCollection", "features": []}
        features = []
        for candidate in self._detect_hour(hour):
            coordinates = _rdp(np.asarray(candidate["coordinates"]), 0.05)
            if len(coordinates) < 2:
                continue
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [round(float(lon), 2), round(float(lat), 2)]
                        for lon, lat in coordinates
                    ],
                },
                "properties": {
                    "medianThetaWGradient": round(
                        float(candidate.get("medianThetaWGradient", 0.0)), 2
                    ),
                    "dryThermalGradient": round(
                        float(candidate.get("dryThermalGradient", 0.0)), 2
                    ),
                },
            })
        return {"type": "FeatureCollection", "features": features}

    def upper_air(self, hour: int, stride: int = 2) -> dict | None:
        """Export theta_w (the actual detector field), T and wind at 850 hPa."""
        result = super().upper_air(hour, stride=stride)
        if result is None:
            return None
        coarse = np.flipud(self._theta_w(hour)[::stride, ::stride])
        rounded = np.round(coarse.astype(float), 1).ravel()
        theta_w = [
            None if not np.isfinite(value) else float(value)
            for value in rounded
        ]
        result["thetaW"] = theta_w
        # Backward-compatible alias for already deployed front-end versions.
        result["thetaE"] = theta_w
        result["primaryThermalField"] = "thetaW"
        return result


# Backwards-compatible alias, mirroring front_analysis.py.
IconFrontAnalyzerV10 = FrontalAnalysisV10
