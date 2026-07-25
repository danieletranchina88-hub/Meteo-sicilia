"""ICON-2I objective synoptic-front analysis, physics-guided engine (v16).

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

import json
import os

import numpy as np

import front_detection as fd
import front_locator as fl
import front_occlusion as focc
import front_physics as fp
import front_ridge as fridge
import front_sections as fsec
import front_support as fsup
import front_tracking as ftk
import thermodynamics as thermo
from front_analysis import SynopticFrontAnalyzer, _blend_lines, _line_length_km, _rdp


FRONT_METHOD = "icon2i-ofa-physics-guided-v16"
ANALYSIS_PRESSURE_PA = 85_000.0
LOWER_PRESSURE_PA = 92_500.0

# Climatological calibration of the detection thresholds (documento sez. 17).
# Built offline by calibrate_thresholds.py from many ICON-2I runs; picked by
# month, whole-domain band. Absent/empty -> the detector uses its fixed
# fuzzy defaults, so this is an enhancement that never blocks the analysis.
CLIMATOLOGY_PATH = os.path.join(
    os.path.dirname(__file__), "climatology_thresholds.json"
)


MIN_CLIMATOLOGY_RUNS = 30
MIN_CLIMATOLOGY_DAYS = 15


def threshold_climatology_status(month: str) -> tuple[dict | None, str]:
    """Return validated monthly thresholds and an explicit QC status.

    Distributional quantiles are not automatically a calibration.  A month
    is eligible only when it contains enough independent runs/days and the
    benchmark process has explicitly promoted the file for operational use.
    Thresholds from another season are never used as a fallback.
    """
    try:
        with open(CLIMATOLOGY_PATH, encoding="utf-8") as handle:
            clim = json.load(handle)
    except OSError:
        return None, "file-missing"
    except ValueError:
        return None, "invalid-json"

    node = clim.get(month)
    if not isinstance(node, dict):
        return None, "month-not-covered"

    meta = clim.get("_meta") or {}
    run_tags = [
        str(tag) for tag in meta.get("runs", [])
        if len(str(tag)) >= 10 and str(tag)[4:6] == month
    ]
    distinct_days = {tag[:8] for tag in run_tags}
    if len(set(run_tags)) < MIN_CLIMATOLOGY_RUNS:
        return None, "insufficient-independent-runs"
    if len(distinct_days) < MIN_CLIMATOLOGY_DAYS:
        return None, "insufficient-distinct-days"
    if meta.get("operationalValidated") is not True:
        return None, "not-operationally-validated"

    syn = (node.get("synoptic") or {}).get("all")
    ref = (node.get("refined") or {}).get("all")
    if not isinstance(syn, dict) or not isinstance(ref, dict):
        return None, "missing-whole-domain-band"

    try:
        ordered = (
            syn["tfp_full"] < syn["tfp_weak"] < 0.0
            and ref["tfp_full"] < ref["tfp_weak"] < 0.0
            and 0.0 < syn["abz_weak"] < syn["abz_full"]
            and 0.0 < ref["abz_weak"] < ref["abz_full"]
        )
    except (KeyError, TypeError):
        return None, "incomplete-threshold-set"
    if not ordered:
        return None, "invalid-threshold-order"

    return {
        "synoptic_tfp_weak": syn["tfp_weak"],
        "synoptic_tfp_full": syn["tfp_full"],
        "synoptic_abz_weak": syn["abz_weak"],
        "synoptic_abz_full": syn["abz_full"],
        "refined_tfp_weak": ref["tfp_weak"],
        "refined_tfp_full": ref["tfp_full"],
        "refined_abz_weak": ref["abz_weak"],
        "refined_abz_full": ref["abz_full"],
    }, "validated"


def load_threshold_climatology(month: str) -> dict | None:
    """Backward-compatible thresholds-only view of the QC-aware loader."""
    return threshold_climatology_status(month)[0]

SYNOPTIC_SIGMA_KM = 100.0
REFINE_SIGMA_KM = 45.0
DERIVATIVE_SIGMA_KM = 15.0
CROSS_FRONT_KM = 55.0
CROSS_FRONT_DISTANCES_KM = (30.0, 55.0, 85.0)
PRESSURE_CROSS_KM = 100.0

TRACK_WINDOW_HOURS = 2
TRACK_GATE_KM = 170.0
TRACK_MIN_LIFETIME_HOURS = 3
TRACK_MIN_DETECTIONS = 4
TRACK_MIN_COVERAGE = 0.72
MIN_PUBLISH_QUALITY = 0.61
MAX_PUBLISH_UNCERTAINTY = 0.39
# A track just below the standard gate is not automatically noise: dropping
# it silently makes a real but modest front indistinguishable from "nothing
# there". Tracks in this lower band are still published, but explicitly
# tagged confidenceTier="low" so the map/API can show them as provisional
# instead of hiding a plausible front (documento sez. 5, "come un analista").
# 0.50 keeps symmetry with the standard uncertaintyIndex complement
# (quality/uncertainty are anti-correlated by construction, see
# qualityComponents) and sits roughly halfway between the standard gate and
# the noise/no-signal region (qualityScore ~0 for a candidate with no
# corroborated evidence): it separates "still evidence-backed but weaker"
# from "essentially unsupported", without duplicating the standard gate.
LOW_TIER_THRESHOLD = 0.50
MIN_PUBLISH_QUALITY_LOW = LOW_TIER_THRESHOLD
MAX_PUBLISH_UNCERTAINTY_LOW = LOW_TIER_THRESHOLD
MAX_FRONTS_PER_HOUR = 4

# Fase C/E: la geometria pubblicata segue la cresta di any_front_support
# (least-cost path in front_ridge), non piu' il solo contorno TFL. Attivato
# dopo il benchmark Fase E (supporto medio 0.43->0.50, tortuosita' non
# peggiore). Un solo interruttore, reversibile: False torna ai contorni.
REFINE_PUBLISHED_GEOMETRY = True
GEOMETRY_CORRIDOR_KM = 120.0


def _finite_median(values, default=np.nan) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else float(default)


def _json_number(value, digits: int | None = None):
    """Return a finite builtin float or JSON null, never NaN/Infinity."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits) if digits is not None else number


def _json_mapping(values: dict, digits: int | None = None) -> dict:
    return {
        key: _json_number(value, digits)
        for key, value in values.items()
    }


class IconSynopticFrontAnalyzer(SynopticFrontAnalyzer):
    """OFA front detector assembled around the validated ICON GRIB loader."""

    def __init__(
        self,
        *args,
        lower_temperature_path: str | None = None,
        lower_humidity_path: str | None = None,
        lower_u_wind_path: str | None = None,
        lower_v_wind_path: str | None = None,
        omega_700_path: str | None = None,
        **kwargs,
    ) -> None:
        requested_method = kwargs.get("method")
        super().__init__(*args, **kwargs)
        self.method = str(requested_method or FRONT_METHOD)
        self._thermodynamic_cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        self._pressure_cache: dict[int, np.ndarray] = {}
        self._tracks: list[dict] | None = None
        self._by_hour: dict[int, list[tuple[np.ndarray, dict]]] | None = None
        self.analysis_summary: dict | None = None
        # Fase B diagnostics: computed on demand / recorded as a side effect,
        # never changing the published fronts.
        self._support_cache: dict[int, dict] = {}
        self._pipeline_diag: dict[int, dict] = {}
        self._rejected: dict[int, list[dict]] = {}
        self._detection_errors: dict[int, str] = {}
        # Gate-rejected candidates whose differential diagnosis still says
        # synoptic-front: the weak phase of a real boundary. Kept in full so
        # tracking can reclaim these hours for an established track.
        self._weak: dict[int, list[dict]] = {}
        # Climatological detection thresholds for the run's month (falls back
        # to the detector's fixed fuzzy defaults when the climatology is absent
        # or does not cover this month).
        run_month = "00"
        try:
            reference = np.atleast_1d(
                np.asarray(self.datasets["t"].time.values)
            ).ravel()[0]
            run_month = str(np.datetime64(reference, "M")).split("-")[1]
        except Exception:
            pass
        self.run_month = run_month
        (
            self._threshold_climatology,
            self._threshold_climatology_status,
        ) = threshold_climatology_status(run_month)

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

        if lower_u_wind_path and lower_v_wind_path:
            added = []
            opened = None
            try:
                for key, path in (
                    ("u925", lower_u_wind_path),
                    ("v925", lower_v_wind_path),
                ):
                    opened = self._open_field(path, key)
                    self._validate_optional_dataset(opened, key, 925.0)
                    variable = next(iter(opened.data_vars))
                    values = np.asarray(opened[variable].values, dtype=float)
                    if np.nanpercentile(np.abs(values), 99.9) > 150.0:
                        raise ValueError(f"{key} fuori scala m/s")
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

    @property
    def has_lower_wind(self) -> bool:
        return "u925" in self.datasets and "v925" in self.datasets

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

    def _thermodynamics(self, hour: int, level_hpa: int = 850) -> dict[str, np.ndarray]:
        key = (level_hpa, hour)
        cached = self._thermodynamic_cache.get(key)
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
        invalid = fields["out_of_domain"]
        result = {
            name: np.where(invalid, np.nan, np.asarray(fields[name], dtype=float))
            for name in ("theta_w", "theta", "theta_e")
        }
        self._thermodynamic_cache[key] = result
        return result

    def _theta_w(self, hour: int, level_hpa: int = 850) -> np.ndarray:
        return self._thermodynamics(hour, level_hpa)["theta_w"]

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
        thermodynamics = self._thermodynamics(hour)
        theta_w = thermodynamics["theta_w"]
        wet_candidates = fd.detect_fronts_two_scale(
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
            **(self._threshold_climatology or {}),
        )
        lower_candidates = self._lower_candidates(hour)
        lower_lines = [np.asarray(item["coordinates"], dtype=float) for item in lower_candidates]

        grid_metrics = fl.grid_metrics(self.longitudes, self.latitudes)
        raw_temperature = self._field("t", hour)
        temperature = fl.smooth_km(raw_temperature, REFINE_SIGMA_KM, grid_metrics)
        humidity = fl.smooth_km(self._field("q", hour), REFINE_SIGMA_KM, grid_metrics)
        dry_theta = thermodynamics["theta"]
        theta_e = thermodynamics["theta_e"]
        dry_candidates = fd.detect_fronts_two_scale(
            dry_theta,
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
            **(self._threshold_climatology or {}),
        )

        # A theta-w locator is sensitive to the physically useful moisture
        # contrast; a dry-theta locator recovers a genuine dry cold-air
        # intrusion that humidity can partly cancel.  Neither source can
        # publish alone: the strict cross-front gates below still require
        # independent dry, moist and density contrasts.
        wet_lines = [np.asarray(c["coordinates"], dtype=float) for c in wet_candidates]
        dry_lines = [np.asarray(c["coordinates"], dtype=float) for c in dry_candidates]
        combined = []
        for source_name, source_items, other_lines in (
            ("thetaW", wet_candidates, dry_lines),
            ("dryTheta", dry_candidates, wet_lines),
        ):
            for item in source_items:
                candidate = dict(item)
                candidate["thermalLocator"] = source_name
                candidate["crossThermalSupport"] = round(
                    fd.line_support_fraction(
                        np.asarray(candidate["coordinates"], dtype=float),
                        other_lines,
                        80.0,
                    ) if other_lines else 0.0,
                    2,
                )
                combined.append(self._orient_warm_left(candidate))
        combined.sort(
            key=lambda c: (
                0.65 * float(c.get("locatorConfidence", 0.0))
                + 0.35 * float(c.get("crossThermalSupport", 0.0)),
                float(c.get("lengthKm", 0.0)),
            ),
            reverse=True,
        )
        candidates = []
        for candidate in combined:
            line = np.asarray(candidate["coordinates"], dtype=float)
            duplicate = False
            for kept in candidates:
                kept_line = np.asarray(kept["coordinates"], dtype=float)
                overlap = min(
                    fd.line_support_fraction(line, [kept_line], 55.0),
                    fd.line_support_fraction(kept_line, [line], 55.0),
                )
                if overlap >= 0.68:
                    duplicate = True
                    break
            if not duplicate:
                candidates.append(candidate)

        theta_v = temperature * (1.0 + 0.61 * humidity) * (
            100_000.0 / ANALYSIS_PRESSURE_PA
        ) ** thermo.KAPPA
        temp_east, temp_north = fl.gradient(temperature, grid_metrics)
        dry_gradient = np.hypot(temp_east, temp_north) * 100.0

        raw_u, raw_v = self._field("u", hour), self._field("v", hour)
        u_wind = fl.smooth_km(raw_u, REFINE_SIGMA_KM, grid_metrics)
        v_wind = fl.smooth_km(raw_v, REFINE_SIGMA_KM, grid_metrics)
        lower_u_wind = lower_v_wind = None
        if self.has_lower_wind:
            lower_u_wind = fl.smooth_km(
                self._field("u925", hour), REFINE_SIGMA_KM, grid_metrics
            )
            lower_v_wind = fl.smooth_km(
                self._field("v925", hour), REFINE_SIGMA_KM, grid_metrics
            )
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
        rejected: list[dict] = []
        reject_counts: dict[str, int] = {}
        for source in candidates:
            candidate = dict(source)
            coordinates = np.asarray(candidate["coordinates"], dtype=float)
            normal = np.asarray(candidate["warmNormal"], dtype=float)
            hewson = np.asarray(candidate["hewsonDir"], dtype=float)
            if len(coordinates) < 4 or normal.shape != coordinates.shape:
                continue

            # Sample the air masses at three metric distances.  A single
            # 55-km pair can accidentally cross a local perturbation; the
            # median of 30/55/85 km is much more stable while retaining the
            # 55-km pair for flow, pressure and motion diagnostics.
            samples: dict[float, tuple[np.ndarray, np.ndarray]] = {}
            temperature_deltas = []
            theta_v_deltas = []
            theta_w_deltas = []
            theta_e_deltas = []
            distance_supports = []
            central_validity = None
            for distance in CROSS_FRONT_DISTANCES_KM:
                warm_at_distance = self._offset_points(coordinates, normal, distance)
                cold_at_distance = self._offset_points(coordinates, -normal, distance)
                samples[distance] = (warm_at_distance, cold_at_distance)
                warm_temp_d = self._sample(temperature, warm_at_distance)
                cold_temp_d = self._sample(temperature, cold_at_distance)
                warm_theta_v_d = self._sample(theta_v, warm_at_distance)
                cold_theta_v_d = self._sample(theta_v, cold_at_distance)
                warm_theta_w_d = self._sample(theta_w, warm_at_distance)
                cold_theta_w_d = self._sample(theta_w, cold_at_distance)
                warm_theta_e_d = self._sample(theta_e, warm_at_distance)
                cold_theta_e_d = self._sample(theta_e, cold_at_distance)
                valid_d = (
                    np.isfinite(warm_theta_w_d) & np.isfinite(cold_theta_w_d)
                    & np.isfinite(warm_temp_d) & np.isfinite(cold_temp_d)
                    & np.isfinite(warm_theta_v_d) & np.isfinite(cold_theta_v_d)
                )
                delta_temp_d = warm_temp_d - cold_temp_d
                delta_theta_v_d = warm_theta_v_d - cold_theta_v_d
                delta_theta_w_d = warm_theta_w_d - cold_theta_w_d
                delta_theta_e_d = warm_theta_e_d - cold_theta_e_d
                support_d = float(np.mean(
                    valid_d
                    & (delta_temp_d >= 0.60)
                    & (delta_theta_v_d >= 0.25)
                    & (delta_theta_w_d >= 1.20)
                ))
                distance_supports.append(support_d)
                temperature_deltas.append(delta_temp_d)
                theta_v_deltas.append(delta_theta_v_d)
                theta_w_deltas.append(delta_theta_w_d)
                theta_e_deltas.append(delta_theta_e_d)
                if distance == CROSS_FRONT_KM:
                    central_validity = valid_d

            validity = central_validity
            if validity is None or float(np.mean(validity)) < 0.75:
                continue
            warm, cold = samples[CROSS_FRONT_KM]
            delta_temperature_points = np.nanmedian(np.stack(temperature_deltas), axis=0)
            delta_theta_v_points = np.nanmedian(np.stack(theta_v_deltas), axis=0)
            delta_theta_w_points = np.nanmedian(np.stack(theta_w_deltas), axis=0)
            delta_theta_e_points = np.nanmedian(np.stack(theta_e_deltas), axis=0)
            cross_distance_support = float(np.mean(distance_supports))

            gx = self._sample(temp_east, coordinates)
            gy = self._sample(temp_north, coordinates)
            gmag = np.maximum(np.hypot(gx, gy), 1.0e-9)
            alignment = (gx * normal[:, 0] + gy * normal[:, 1]) / gmag

            thermal_point_valid = (
                validity
                & np.isfinite(delta_temperature_points)
                & np.isfinite(delta_theta_v_points)
                & np.isfinite(delta_theta_w_points)
            )
            thermal_contrast_fraction = float(np.mean(
                thermal_point_valid
                & (delta_temperature_points >= 0.60)
                & (delta_theta_v_points >= 0.25)
                & (delta_theta_w_points >= 1.20)
            ))
            thermal_alignment_fraction = float(np.mean(
                np.isfinite(alignment) & (alignment >= 0.10)
            ))

            warm_u, warm_v = self._sample(u_wind, warm), self._sample(v_wind, warm)
            cold_u, cold_v = self._sample(u_wind, cold), self._sample(v_wind, cold)
            center_u = self._sample(u_wind, coordinates)
            center_v = self._sample(v_wind, coordinates)
            lower_wind_fraction = 0.0
            if lower_u_wind is not None and lower_v_wind is not None:
                terrain_line = self._sample(self.terrain, coordinates)
                terrain_warm = self._sample(self.terrain, warm)
                terrain_cold = self._sample(self.terrain, cold)
                lower_center_u = self._sample(lower_u_wind, coordinates)
                lower_center_v = self._sample(lower_v_wind, coordinates)
                lower_warm_u = self._sample(lower_u_wind, warm)
                lower_warm_v = self._sample(lower_v_wind, warm)
                lower_cold_u = self._sample(lower_u_wind, cold)
                lower_cold_v = self._sample(lower_v_wind, cold)
                usable_center = (
                    (terrain_line < 650.0)
                    & np.isfinite(lower_center_u) & np.isfinite(lower_center_v)
                )
                usable_warm = (
                    (terrain_warm < 650.0)
                    & np.isfinite(lower_warm_u) & np.isfinite(lower_warm_v)
                )
                usable_cold = (
                    (terrain_cold < 650.0)
                    & np.isfinite(lower_cold_u) & np.isfinite(lower_cold_v)
                )
                lower_wind_fraction = float(np.mean(
                    usable_center & usable_warm & usable_cold
                ))
                center_u = np.where(usable_center, lower_center_u, center_u)
                center_v = np.where(usable_center, lower_center_v, center_v)
                warm_u = np.where(usable_warm, lower_warm_u, warm_u)
                warm_v = np.where(usable_warm, lower_warm_v, warm_v)
                cold_u = np.where(usable_cold, lower_cold_u, cold_u)
                cold_v = np.where(usable_cold, lower_cold_v, cold_v)
            wind_valid = (
                np.isfinite(warm_u) & np.isfinite(warm_v)
                & np.isfinite(cold_u) & np.isfinite(cold_v)
            )
            if float(np.mean(wind_valid)) < 0.70:
                continue
            wind_shift = np.hypot(warm_u - cold_u, warm_v - cold_v)
            warm_speed = np.hypot(warm_u, warm_v)
            cold_speed = np.hypot(cold_u, cold_v)
            directional_valid = wind_valid & (warm_speed >= 1.5) & (cold_speed >= 1.5)
            wind_dot = warm_u * cold_u + warm_v * cold_v
            wind_cross = warm_u * cold_v - warm_v * cold_u
            wind_angle = np.degrees(np.arctan2(np.abs(wind_cross), wind_dot))
            convergence = (
                (cold_u - warm_u) * normal[:, 0]
                + (cold_v - warm_v) * normal[:, 1]
            )
            wind_boundary_fraction = float(np.mean(
                wind_valid & (
                    (wind_shift >= 1.8)
                    | (directional_valid & (wind_angle >= 10.0))
                )
            ))
            convergence_fraction = float(np.mean(
                wind_valid & (convergence >= 0.10)
            ))
            normal_airmass_flow = center_u * normal[:, 0] + center_v * normal[:, 1]
            ofa_speed = center_u * hewson[:, 0] + center_v * hewson[:, 1]

            metrics = {
                **candidate,
                "deltaThetaW": _finite_median(delta_theta_w_points),
                "deltaThetaE": _finite_median(delta_theta_e_points),
                "deltaTemperature": _finite_median(delta_temperature_points),
                "deltaThetaV": _finite_median(delta_theta_v_points),
                "thermalContrastFraction": thermal_contrast_fraction,
                "thermalAlignmentFraction": thermal_alignment_fraction,
                "crossDistanceThermalSupport": cross_distance_support,
                "crossDistanceKm": list(CROSS_FRONT_DISTANCES_KM),
                "dryThermalGradient": _finite_median(
                    self._sample(dry_gradient, coordinates)
                ),
                "thermalAlignment": _finite_median(alignment),
                "windShiftMs": _finite_median(wind_shift[wind_valid]),
                "windShiftAngleDeg": _finite_median(
                    wind_angle[directional_valid]
                ),
                "windBoundaryFraction": wind_boundary_fraction,
                "convergenceMs": _finite_median(convergence[wind_valid]),
                "convergenceFraction": convergence_fraction,
                # Positive points from cold toward warm: cold-air advance.
                "airmassMotionKmh": _finite_median(normal_airmass_flow) * 3.6,
                "lowerWindFraction": lower_wind_fraction,
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
                warm_p = self._sample(pressure, far_warm)
                cold_p = self._sample(pressure, far_cold)
                side_p = 0.5 * (warm_p + cold_p)
                trough_points = side_p - center_p
                pressure_valid = (
                    np.isfinite(trough_points) & np.isfinite(center_p)
                )
                metrics["pressureTroughHpa"] = _finite_median(
                    trough_points[pressure_valid]
                )
                metrics["pressureTroughFraction"] = float(np.mean(
                    pressure_valid & (trough_points >= 0.0)
                ))
                if pressure_tendency is not None:
                    line_pt = self._sample(pressure_tendency, coordinates)
                    cold_pt = self._sample(pressure_tendency, far_cold)
                    warm_pt = self._sample(pressure_tendency, far_warm)
                    metrics["linePressureTendencyHpa3h"] = _finite_median(
                        line_pt * 3.0
                    )
                    metrics["coldPressureTendencyHpa3h"] = _finite_median(
                        cold_pt * 3.0
                    )
                    metrics["warmPressureTendencyHpa3h"] = _finite_median(
                        warm_pt * 3.0
                    )
                    metrics["isallobaricSupportHpa3h"] = _finite_median(
                        (cold_pt - warm_pt) * 3.0
                    )
            else:
                metrics["pressureTroughHpa"] = np.nan
                metrics["pressureTroughFraction"] = np.nan

            if omega is not None:
                line_omega = self._sample(omega, coordinates)
                warm_omega = self._sample(omega, warm)
                metrics["omega700PaS"] = _finite_median(
                    np.fmin(line_omega, warm_omega)
                )

            if theta_w_925 is not None:
                lower_warm = self._sample(theta_w_925, warm)
                lower_cold = self._sample(theta_w_925, cold)
                lower_valid = np.isfinite(lower_warm) & np.isfinite(lower_cold)
                metrics["lowerValidFraction"] = float(np.mean(lower_valid))
                metrics["lowerLevelSupport"] = fd.line_support_fraction(
                    coordinates, lower_lines, 130.0
                ) if lower_lines else 0.0
                metrics["deltaThetaW925"] = _finite_median(
                    (lower_warm - lower_cold)[lower_valid]
                )
            else:
                metrics["lowerValidFraction"] = 0.0

            # Cross-front sections (front_sections): the transition-zone
            # profile at 850 hPa, and its 925 hPa vertical coherence when the
            # lower level exists. Width/offset stay diagnostics (not gates);
            # a missing 925 hPa profile is neutral, never counter-evidence.
            section_850 = fsec.profile_diagnostics(fsec.cross_profiles(
                theta_w, self.longitudes, self.latitudes, coordinates, normal,
            ))
            metrics.update(section_850)
            if theta_w_925 is not None:
                section_925 = fsec.profile_diagnostics(fsec.cross_profiles(
                    theta_w_925, self.longitudes, self.latitudes,
                    coordinates, normal,
                ))
                metrics["frontWidth925Km"] = section_925.get("frontWidthKm")
                coherence = fsec.vertical_coherence(section_850, section_925)
                if coherence is not None:
                    metrics["verticalCoherence"] = coherence

            evidence = fp.candidate_evidence(metrics)
            metrics.update(evidence)
            gates = fp.candidate_gate_report(metrics, evidence)
            metrics.update(gates)
            if not gates["continuationPass"]:
                # A rejected candidate is recorded, not silently dropped, so
                # the reason is inspectable (diagnostic mode / QC).
                rejected.append({
                    "coordinates": np.asarray(candidate["coordinates"], dtype=float),
                    "rejectedAs": gates.get("diagnosis", "unknown"),
                    "reasons": list(gates.get("rejectionReasons", [])),
                    "candidateEvidence": metrics.get("candidateEvidence"),
                })
                if gates.get("diagnosis") == "synoptic-front":
                    # Weak phase of a possibly real boundary: keep the full
                    # candidate so an established track can reclaim this hour.
                    self._weak.setdefault(hour, []).append(metrics)
                for reason in gates.get("rejectionReasons", []) or ["unspecified"]:
                    reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            if gates["gateStatus"] == "continuation":
                metrics["candidateEvidence"] = round(
                    float(metrics["candidateEvidence"]) * 0.88, 3
                )
            accepted.append(metrics)

        self._rejected[hour] = rejected
        self._pipeline_diag[hour] = {
            "candidateLines": len(candidates),
            "finalPolylines": len(accepted),
            "rejectedLines": len(rejected),
            "rejected": reject_counts,
        }
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
        self._detection_errors = detection_errors
        if len(detection_errors) > allowed_errors:
            raise RuntimeError(
                "analisi frontale incompleta: "
                f"{len(detection_errors)}/{len(self.available_hours)} ore in errore"
            )
        # Zero physically robust candidates is a valid meteorological result.
        # It must publish a fresh empty analysis instead of leaving stale fronts
        # from the previous run online.  Excessive processing errors above still
        # fail closed and remain distinct from a genuine no-front situation.

        raw_tracks = ftk.track_fronts(
            hourly,
            window_hours=TRACK_WINDOW_HOURS,
            gate_km=TRACK_GATE_KM,
            min_lifetime_hours=TRACK_MIN_LIFETIME_HOURS,
            min_detections=TRACK_MIN_DETECTIONS,
            min_coverage=TRACK_MIN_COVERAGE,
            weak_candidates=getattr(self, "_weak", None),
        )
        # Publication is judged on the CORE hours (strong detections): the
        # weak-phase hours recovered by the tracker extend where the boundary
        # is drawn but must not dilute -- nor artificially satisfy -- the
        # confidence requirements.
        def _consensus_count(track: dict) -> int:
            return sum(
                track.get("localClassifications", {}).get(h, {}).get("frontType")
                != "uncertain"
                for h in track.get("coreHours", track.get("hours", []))
            )

        def _consensus_required(track: dict) -> int:
            core = track.get("coreHours", track.get("hours", []))
            return max(TRACK_MIN_DETECTIONS, int(np.ceil(0.55 * len(core))))

        tracks = []
        provisional_tracks = []
        for track in raw_tracks:
            quality = track.get("qualityScore", 0.0)
            uncertainty = track.get("uncertaintyIndex", 1.0)
            consensus_ok = _consensus_count(track) >= _consensus_required(track)
            if not consensus_ok:
                continue
            if (
                quality >= MIN_PUBLISH_QUALITY
                and uncertainty <= MAX_PUBLISH_UNCERTAINTY
            ):
                track["confidenceTier"] = "standard"
                tracks.append(track)
            elif (
                quality >= MIN_PUBLISH_QUALITY_LOW
                and uncertainty <= MAX_PUBLISH_UNCERTAINTY_LOW
            ):
                # Just below the standard gate: a candidate this coherent is
                # more likely a modest real front than noise (documento sez.
                # 4, "un fronte reale ma modesto non viene scartato"). It is
                # published as a distinct, clearly-flagged provisional tier
                # instead of being silently dropped like plain noise.
                track["confidenceTier"] = "low"
                provisional_tracks.append(track)
        published_tracks = tracks + provisional_tracks
        run_status = (
            "partially-unavailable"
            if detection_errors
            else "fronts-detected"
            if published_tracks
            else "no-robust-fronts"
        )
        self.analysis_summary = {
            "analysisStatus": run_status,
            "analysisMessage": (
                f"{len(detection_errors)} ore frontali non disponibili nel run."
                if detection_errors
                else "Fronti sinottici robusti rilevati nel run."
                if published_tracks
                else "Nessun fronte sinottico robusto nel run."
            ),
            "hours": len(self.available_hours),
            "hoursWithCandidates": sum(bool(items) for items in hourly.values()),
            "candidateLines": total_candidates,
            "trackingTracks": len(raw_tracks),
            "publishedTracks": len(published_tracks),
            "publishedTracksStandard": len(tracks),
            "publishedTracksLowConfidence": len(provisional_tracks),
            "detectionErrors": len(detection_errors),
            "lowerLevel925": self.has_lower_level,
            "lowerWind925": self.has_lower_wind,
            "omega700": "omega700" in self.datasets,
            "pressure": "p" in self.datasets,
            "thresholdClimatology": (
                self.run_month if self._threshold_climatology else None
            ),
            "thresholdClimatologyStatus": self._threshold_climatology_status,
            "rejectedByReason": self._aggregate_rejections(),
        }
        print(
            "   Front QC: "
            f"{self.analysis_summary['candidateLines']} candidati, "
            f"{self.analysis_summary['trackingTracks']} tracce, "
            f"{self.analysis_summary['publishedTracks']} pubblicate "
            f"({self.analysis_summary['publishedTracksLowConfidence']} a bassa fiducia), "
            f"{self.analysis_summary['detectionErrors']} errori.",
            flush=True,
        )
        self._tracks = published_tracks

        by_hour: dict[int, list[tuple[np.ndarray, dict]]] = {
            hour: [] for hour in self.available_hours
        }
        for track in published_tracks:
            expanded = dict(track["lines"])
            local = dict(track.get("localClassifications", {}))
            detected = sorted(track["lines"])
            for first, second in zip(detected[:-1], detected[1:]):
                gap = second - first
                if gap != 2:
                    continue
                missing = first + 1
                first_type = local.get(first, {}).get("frontType")
                second_type = local.get(second, {}).get("frontType")
                if (
                    missing in self.hour_to_index
                    and first_type == second_type
                    and first_type != "uncertain"
                ):
                    expanded[missing] = _blend_lines(
                        np.asarray(track["lines"][first], dtype=float),
                        np.asarray(track["lines"][second], dtype=float),
                        0.5,
                    )
                    local[missing] = dict(local[first])
                    local[missing]["classificationCertainty"] = round(
                        min(
                            float(local[first].get("classificationCertainty", 0.0)),
                            float(local[second].get("classificationCertainty", 0.0)),
                        ) * 0.92,
                        2,
                    )
            # A published track is a single continuous boundary. An isolated
            # hour whose local motion is ambiguous ("uncertain") must NOT punch
            # a hole in the middle of the track: that is the "front disappears
            # for one hour" artefact. Such hours are displayed with the track's
            # dominant published type, so the line stays continuous while its
            # reduced certainty is still recorded.
            local_types = [
                value.get("frontType")
                for value in local.values()
                if value.get("frontType") not in (None, "uncertain")
            ]
            dominant_type = track.get("frontType")
            if dominant_type in (None, "uncertain"):
                dominant_type = (
                    max(set(local_types), key=local_types.count)
                    if local_types else None
                )
            for hour, coordinates in expanded.items():
                classification = dict(local.get(hour, {}))
                if classification.get("frontType") in (None, "uncertain"):
                    if dominant_type is None:
                        continue
                    classification["frontType"] = dominant_type
                    classification["classificationCertainty"] = round(
                        float(classification.get("classificationCertainty", 0.0)),
                        2,
                    )
                    classification["typeInferredFromTrack"] = True
                hour_properties = self._track_properties(
                    track, classification, hour=hour
                )
                # Per-segment character along the line (existence stays one
                # continuous track; the type may vary west->east). Additive:
                # the single frontType remains for backward compatibility.
                segments = track.get("segmentTypes", {}).get(hour)
                if segments:
                    hour_properties["segmentTypes"] = segments
                hour_properties["interpolated"] = hour not in track["lines"]
                if hour_properties["interpolated"]:
                    hour_properties["uncertaintyIndex"] = round(
                        min(1.0, hour_properties["uncertaintyIndex"] + 0.06), 2
                    )
                geometry = self._refine_geometry(
                    hour, np.asarray(coordinates, dtype=float)
                )
                by_hour.setdefault(hour, []).append((geometry, hour_properties))
        self._occlusion_hours = self._apply_occlusions(by_hour)
        self._by_hour = by_hour

    def _apply_occlusions(self, by_hour: dict) -> int:
        """Relabel the wrapped cold-front branch of an occluding wave.

        Spatial+temporal context (a real MSLP low with a cold and a warm front
        meeting at a triple point) decides the occlusion, per documento sez.
        14. The cold front is trimmed at the triple point and the branch that
        wraps toward the low centre becomes a separate ``occluded`` feature.
        """
        occluded_hours = 0
        for hour, entries in by_hour.items():
            if len(entries) < 2:
                continue
            features = [{"coordinates": coords, "frontType": props["frontType"]}
                        for coords, props in entries]
            pressure = self._pressure_hpa(hour)
            occlusions = focc.detect_occlusion(
                features, pressure, self.longitudes, self.latitudes
            )
            if not occlusions:
                continue
            for occ in occlusions:
                fi = occ["featureIndex"]
                coords, props = entries[fi]
                coords = np.asarray(coords, dtype=float)
                triple, low_idx = occ["tripleIndex"], occ["lowIndex"]
                if triple <= low_idx:
                    occluded_part = coords[triple:low_idx + 1]
                    cold_part = coords[:triple + 1]
                else:
                    occluded_part = coords[low_idx:triple + 1]
                    cold_part = coords[triple:]
                if len(occluded_part) < 3:
                    continue
                occ_props = dict(props)
                occ_props["frontType"] = "occluded"
                occ_props["occlusion"] = {
                    "lowPressureHpa": round(occ["low"]["pressure"], 1),
                    "triplePoint": occ["triplePoint"],
                    "wrapKm": occ["wrapKm"],
                }
                occ_props["explanation"] = [
                    "fronte freddo che raggiunge il fronte caldo attorno al "
                    f"minimo barico ({round(occ['low']['pressure'])} hPa)",
                    "settore caldo ristretto al punto di tripla giunzione",
                ] + [
                    reason for reason in props.get("explanation", [])
                    if "persistente" in reason
                ]
                # The trailing cold front keeps its identity only if a real
                # segment survives past the triple point.
                if len(cold_part) >= 3:
                    entries[fi] = (np.asarray(cold_part, dtype=float), props)
                    entries.append((occluded_part, occ_props))
                else:
                    entries[fi] = (occluded_part, occ_props)
                occluded_hours += 1
        return occluded_hours

    @staticmethod
    def _front_reasoning(track: dict) -> dict:
        """Compact, human-facing view of the multi-hypothesis reasoning.

        Feeds the reasoning engine both the spatial diagnostics AND the
        temporal evolution of the track (lifetime, coverage, motion
        consistency), so the verdict weighs how the feature behaves over time
        — a front persists and moves coherently — exactly as a forecaster
        would. Exposes the verdict, by how much it beat the field, the best
        competing hypothesis and why.
        """
        diagnostics = dict(track.get("diagnostics", {}))
        hours = track.get("hours", [])
        span = (max(hours) - min(hours)) if hours else 0
        diagnostics["lifetimeH"] = track.get("lifetimeH", span)
        diagnostics["temporalCoverage"] = (
            len(hours) / max(span + 1, 1) if hours else 0.0
        )
        diagnostics["motionMadKmh"] = track.get("motionMadKmh", np.nan)
        report = fp.differential_diagnosis(diagnostics)
        top_two = report["ranking"][:2]
        return {
            "verdict": report["verdict"],
            "margin": report["margin"],
            "alternatives": [
                {"hypothesis": name, "support": report["supports"][name]}
                for name in top_two
            ],
            "reasons": report["reasons"],
        }

    def _track_properties(
        self, track: dict, classification: dict | None = None,
        hour: int | None = None,
    ) -> dict:
        """Feature properties for one display hour.

        Track-level values (persistence, publication) are always exposed as
        ``trackQualityScore`` / ``trackUncertaintyIndex`` / ``trackDiagnostics``.
        When ``hour`` is given and directly observed, ``qualityScore``,
        ``uncertaintyIndex``, ``diagnostics`` and the explanation describe
        THAT hour, so a front weakening at one lead time shows it instead of
        repeating the track median. An interpolated hour never pretends to be
        a detection: ``detectionQuality`` is null and the score carries an
        explicit penalty.
        """
        classification = classification or track
        properties = {
            "frontType": classification["frontType"],
            "confidence": round(float(track["qualityScore"]), 2),
            "qualityScore": round(float(track["qualityScore"]), 2),
            "uncertaintyIndex": round(float(track["uncertaintyIndex"]), 2),
            "uncertaintyClass": track.get("uncertaintyClass", "low"),
            # "standard" meets the full publish gate; "low" is a track that
            # cleared the lower provisional band (MIN_PUBLISH_QUALITY_LOW /
            # MAX_PUBLISH_UNCERTAINTY_LOW) and consensus classification but
            # not the standard one. Shown, never hidden, always labelled.
            "confidenceTier": track.get("confidenceTier", "standard"),
            "trackQualityScore": round(
                float(track.get("trackQualityScore", track["qualityScore"])), 2
            ),
            "trackUncertaintyIndex": round(
                float(track.get("trackUncertaintyIndex",
                                track["uncertaintyIndex"])), 2
            ),
            "trackDiagnostics": _json_mapping(track.get("diagnostics", {}), 4),
            "motionKmh": round(float(classification.get("geoMotionKmh", 0.0)), 1),
            "geoMotionKmh": _json_number(classification.get("geoMotionKmh"), 1),
            "ofaSpeedKmh": _json_number(classification.get("ofaSpeedKmh"), 1),
            "tendencyMotionKmh": _json_number(
                classification.get("tendencyMotionKmh"), 1
            ),
            "airmassMotionKmh": _json_number(
                classification.get("airmassMotionKmh"), 1
            ),
            "motionVotes": dict(classification.get("motionVotes", {})),
            "classificationCertainty": _json_number(
                classification.get("classificationCertainty"), 2
            ),
            "qualityComponents": _json_mapping(
                track.get("qualityComponents", {}), 2
            ),
            "diagnostics": _json_mapping(track.get("diagnostics", {}), 4),
            "diagnosis": track.get("diagnosis", "synoptic-front"),
            "reasoning": self._front_reasoning(track),
            "explanation": fp.frontal_explanation(
                track.get("diagnostics", {}),
                classification,
                track.get("diagnosis", "synoptic-front"),
                track.get("lifetimeH", 0),
            ),
            "lifetimeH": int(track.get("lifetimeH", 0)),
            "trackId": int(track.get("id", -1)),
            "method": self.method,
            "source": self.source,
        }
        if hour is None:
            return properties
        hourly_quality = (track.get("hourlyQuality") or {}).get(hour)
        if hourly_quality is not None:
            # Directly observed hour: instantaneous quality view.
            properties["qualityScore"] = round(float(hourly_quality), 2)
            properties["confidence"] = properties["qualityScore"]
            properties["uncertaintyIndex"] = _json_number(
                (track.get("hourlyUncertainty") or {}).get(hour), 2
            ) or properties["uncertaintyIndex"]
            properties["detectionQuality"] = _json_number(
                (track.get("detectionQuality") or {}).get(hour), 3
            )
            properties["trackingConfidence"] = _json_number(
                (track.get("trackingConfidence") or {}).get(hour), 3
            )
            properties["classificationConfidence"] = _json_number(
                (track.get("classificationConfidence") or {}).get(hour), 2
            )
            observation = (track.get("observations") or {}).get(hour) or {}
            properties["gateStatus"] = observation.get("gateStatus")
            properties["recovered"] = hour in (
                track.get("recoveredHours") or []
            )
            hour_diagnostics = observation.get("diagnostics") or {}
            if hour_diagnostics:
                properties["diagnostics"] = _json_mapping(hour_diagnostics, 4)
                # The meteorological explanation describes THIS hour; the
                # track's persistence enters separately via lifetimeH.
                properties["explanation"] = fp.frontal_explanation(
                    hour_diagnostics,
                    classification,
                    observation.get("diagnosis", "synoptic-front"),
                    track.get("lifetimeH", 0),
                )
        else:
            # Interpolated display hour: no direct detection to describe.
            detected = sorted(track.get("hourlyQuality") or {})
            before = [h for h in detected if h < hour]
            after = [h for h in detected if h > hour]
            neighbour_scores = [
                float((track.get("hourlyQuality") or {})[h])
                for h in ([before[-1]] if before else []) + ([after[0]] if after else [])
            ]
            if neighbour_scores:
                # explicit penalty: the hour is a bridge, not an observation
                bridged = float(np.mean(neighbour_scores)) * 0.85
                properties["qualityScore"] = round(bridged, 2)
                properties["confidence"] = properties["qualityScore"]
                properties["uncertaintyIndex"] = round(
                    float(np.clip(1.0 - bridged + 0.10, 0.0, 1.0)), 2
                )
                neighbour_diagnostics = [
                    ((track.get("observations") or {}).get(h) or {})
                    .get("diagnostics") or {}
                    for h in ([before[-1]] if before else [])
                    + ([after[0]] if after else [])
                ]
                merged = {}
                for key in ftk.DIAGNOSTIC_KEYS:
                    values = [
                        diag.get(key) for diag in neighbour_diagnostics
                        if diag.get(key) is not None
                    ]
                    finite = [
                        float(v) for v in values
                        if isinstance(v, (int, float)) and np.isfinite(float(v))
                    ]
                    if finite:
                        merged[key] = float(np.mean(finite))
                if merged:
                    properties["diagnostics"] = _json_mapping(merged, 4)
            properties["detectionQuality"] = None
            properties["trackingConfidence"] = None
            properties["classificationConfidence"] = _json_number(
                classification.get("classificationCertainty"), 2
            )
            properties["gateStatus"] = None
            properties["recovered"] = False
        return properties

    def analyze(self, hour: int) -> dict:
        base_properties = {
            "method": self.method,
            "source": self.source,
            "level": "850 hPa",
            "lowerLevelSupport": "925 hPa" if self.has_lower_level else None,
            "classificationWind": "925/850 hPa" if self.has_lower_wind else "850 hPa",
            "estimated": True,
            "uncertainty": "diagnostic-not-probabilistic",
            "analysisStatus": "unavailable",
            "analysisMessage": "Scadenza non disponibile per l'analisi frontale.",
        }
        if hour not in self.hour_to_index:
            return {
                "type": "FeatureCollection",
                "features": [],
                "properties": base_properties,
            }

        self._ensure_tracks()
        if hour in self._detection_errors:
            base_properties["analysisStatus"] = "unavailable"
            base_properties["analysisMessage"] = (
                "Analisi frontale non disponibile per un errore diagnostico: "
                + self._detection_errors[hour]
            )
            return {
                "type": "FeatureCollection",
                "features": [],
                "properties": base_properties,
            }
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

        if features:
            base_properties["analysisStatus"] = "fronts-detected"
            base_properties["analysisMessage"] = (
                f"{len(features)} strutture frontali sinottiche robuste."
            )
        else:
            base_properties["analysisStatus"] = "no-robust-fronts"
            base_properties["analysisMessage"] = (
                "Nessun fronte sinottico robusto in questa ora."
            )
        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": base_properties,
        }

    def _aggregate_rejections(self) -> dict:
        """Total rejected candidates per reason across all forecast hours."""
        totals: dict[str, int] = {}
        for diag in self._pipeline_diag.values():
            for reason, count in diag.get("rejected", {}).items():
                totals[reason] = totals.get(reason, 0) + count
        return totals

    def _support_config(self) -> dict | None:
        """Feed the climatological refined ABZ/TFP into the support field."""
        clim = self._threshold_climatology
        if not clim:
            return None
        return {
            "abz_weak": clim["refined_abz_weak"],
            "abz_full": clim["refined_abz_full"],
            "tfp_weak": clim["refined_tfp_weak"],
            "tfp_full": clim["refined_tfp_full"],
        }

    def _refine_geometry(self, hour: int, coordinates: np.ndarray) -> np.ndarray:
        """Snap a published front line onto the crest of any_front_support.

        Fase C/E: instead of drawing the raw TFL contour, follow the least-cost
        path along the continuous support field inside a corridor around the
        contour. Benchmarked (Fase E) as better-or-equal to the contour, so it
        is on by default; ``REFINE_PUBLISHED_GEOMETRY = False`` reverts. The
        step is conservative by construction (bounded corridor) and fully
        guarded: any failure or a degenerate result falls back to the contour,
        so publication can never regress to an empty or broken line.
        """
        if not REFINE_PUBLISHED_GEOMETRY or len(coordinates) < 2:
            return coordinates
        try:
            support = self.support_field(hour)
            refined = fridge.refine_line(
                support, self.longitudes, self.latitudes, coordinates,
                corridor_km=GEOMETRY_CORRIDOR_KM,
            )
        except Exception:
            return coordinates
        refined = np.asarray(refined, dtype=float)
        if refined.ndim != 2 or len(refined) < 2 or not np.all(np.isfinite(refined)):
            return coordinates
        return refined

    def support_field(self, hour: int) -> dict:
        """Continuous any_front_support field for one hour (Fase B).

        Computed on demand and cached once per hour. With
        ``REFINE_PUBLISHED_GEOMETRY`` on (Fase C/E) it drives the least-cost
        line extraction that shapes the published geometry; it does not add or
        remove fronts, only refines where each existing line is drawn.
        """
        cached = self._support_cache.get(hour)
        if cached is not None:
            return cached
        thermodynamics = self._thermodynamics(hour)
        theta_w_925 = None
        if self.has_lower_level:
            try:
                theta_w_925 = self._theta_w(hour, 925).copy()
                theta_w_925[self.terrain > 650.0] = np.nan
            except Exception:
                theta_w_925 = None
        field = fsup.physical_support_field(
            thermodynamics["theta_w"], thermodynamics["theta"],
            thermodynamics["theta_e"],
            self._field("u", hour), self._field("v", hour),
            self.longitudes, self.latitudes,
            terrain=self.terrain,
            pressure_hpa=self._pressure_hpa(hour),
            theta_w_925=theta_w_925,
            synoptic_sigma_km=SYNOPTIC_SIGMA_KM,
            refine_sigma_km=REFINE_SIGMA_KM,
            derivative_sigma_km=DERIVATIVE_SIGMA_KM,
            config=self._support_config(),
        )
        self._support_cache[hour] = field
        return field

    def rejected_candidates(self, hour: int) -> dict:
        """Diagnostic export of candidates rejected by the gates, with reasons.

        No candidate disappears silently: each rejected line carries the
        winning alternative hypothesis (``rejectedAs``) and the failed gates.
        """
        self._ensure_tracks()
        features = []
        for item in self._rejected.get(hour, []):
            coordinates = _rdp(np.asarray(item["coordinates"], dtype=float), 0.05)
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": np.round(coordinates, 2).tolist(),
                },
                "properties": {
                    "rejectedAs": item["rejectedAs"],
                    "reasons": item["reasons"],
                    "candidateEvidence": item.get("candidateEvidence"),
                },
            })
        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {"pipeline": self._pipeline_diag.get(hour, {})},
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
        """Export map-ready 850/925-hPa fields used by the level selector.

        The browser receives a compact, coarsened inspection grid rather than
        the full ICON mesh.  Relative humidity is derived from the native
        specific humidity with the same Bolton-consistent vapour-pressure
        relation used by the thermodynamic diagnostic module.
        """
        result = super().upper_air(hour, stride=stride)
        if result is None:
            return None

        def prepare(field: np.ndarray, decimals: int) -> list:
            coarse = np.flipud(field[::stride, ::stride])
            values = np.round(coarse.astype(float), decimals).ravel()
            return [None if not np.isfinite(value) else float(value) for value in values]

        def relative_humidity(temperature: np.ndarray, humidity: np.ndarray, pressure_pa: float) -> np.ndarray:
            q = np.asarray(humidity, dtype=float)
            mixing_ratio = q / np.maximum(1.0 - q, 1.0e-9)
            vapour_pressure = pressure_pa * mixing_ratio / (thermo.EPSILON + mixing_ratio)
            saturation = thermo.saturation_vapour_pressure_pa(temperature)
            with np.errstate(divide="ignore", invalid="ignore"):
                rh = 100.0 * vapour_pressure / saturation
            return np.where(np.isfinite(rh), np.clip(rh, 0.0, 100.0), np.nan)

        def level_payload(level_hpa: int) -> dict | None:
            if level_hpa == 850:
                temperature = self._field("t", hour)
                humidity = self._field("q", hour)
                u_wind = self._field("u", hour)
                v_wind = self._field("v", hour)
                thermodynamic = self._thermodynamics(hour)
                pressure = ANALYSIS_PRESSURE_PA
            elif not (self.has_lower_level and self.has_lower_wind):
                return None
            else:
                temperature = self._field("t925", hour)
                humidity = self._field("q925", hour)
                u_wind = self._field("u925", hour)
                v_wind = self._field("v925", hour)
                thermodynamic = self._thermodynamics(hour, 925)
                pressure = LOWER_PRESSURE_PA

                # 925 hPa is close to 700–800 m: over higher terrain this
                # pressure surface is underground or extrapolated and must
                # not be displayed as an atmospheric observation.
                below_ground = self.terrain > 650.0
                temperature = np.where(below_ground, np.nan, temperature)
                humidity = np.where(below_ground, np.nan, humidity)
                u_wind = np.where(below_ground, np.nan, u_wind)
                v_wind = np.where(below_ground, np.nan, v_wind)
                thermodynamic = {
                    name: np.where(below_ground, np.nan, values)
                    for name, values in thermodynamic.items()
                }

            payload = {
                "level": f"{level_hpa} hPa",
                "nx": int(len(self.longitudes[::stride])),
                "ny": int(len(self.latitudes[::stride])),
                "lo1": float(self.longitudes[0]),
                "la1": float(self.latitudes[-1]),
                "lo2": float(self.longitudes[-1]),
                "la2": float(self.latitudes[0]),
                "dx": float(self.delta_longitude * stride),
                "dy": float(self.delta_latitude * stride),
                "t": prepare(temperature - thermo.CELSIUS0, 1),
                "rh": prepare(relative_humidity(temperature, humidity, pressure), 0),
                "u": prepare(u_wind, 1),
                "v": prepare(v_wind, 1),
                "thetaW": prepare(thermodynamic["theta_w"], 1),
                "thetaE": prepare(thermodynamic["theta_e"], 1),
            }
            return payload

        levels = {"850": level_payload(850)}
        lower = level_payload(925)
        if lower is not None:
            levels["925"] = lower

        # Preserve the former top-level 850-hPa schema so old cached pages
        # and the dedicated theta-w front layer remain compatible.
        primary = levels["850"]
        result.update(primary)
        result["levels"] = levels
        result["primaryThermalField"] = "thetaW"
        return result


# Explicit alias used by process_data.py.
FrontalAnalysisV12 = IconSynopticFrontAnalyzer
