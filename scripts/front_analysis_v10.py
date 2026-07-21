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
import front_tracking as ftk
import thermodynamics as thermo
from front_analysis import SynopticFrontAnalyzer, _rdp, _line_length_km

# Analysis level in Pa (850 hPa): theta_w is computed on the constant
# pressure surface the fields live on, not on PMSL.
ANALYSIS_PRESSURE_PA = 85000.0

V10_METHOD = "OFA v10 (theta_w Hewson, due scale, tracking)"

# Detection / tracking scales.  Tuned so the synoptic prior filters ICON-2I
# mesoscale noise while the refined scale keeps a realistic frontal shape.
SYNOPTIC_SIGMA_KM = 150.0
REFINE_SIGMA_KM = 50.0
DERIVATIVE_SIGMA_KM = 20.0
CORRIDOR_KM = 180.0
SYNOPTIC_MIN_LENGTH_KM = 400.0
REFINE_MIN_LENGTH_KM = 250.0
ABZ_GRADIENT_THRESHOLD = 1.5

TRACK_WINDOW_HOURS = 3
TRACK_GATE_KM = 250.0
TRACK_MIN_LIFETIME_HOURS = 6
TRACK_MIN_COVERAGE = 0.5

# Published fronts per hour: keep the map readable, drop the weakest.
MAX_FRONTS_PER_HOUR = 4
# A track must clear this quality floor to be published at all.
MIN_PUBLISH_QUALITY = 0.30


class FrontalAnalysisV10(SynopticFrontAnalyzer):
    """v10 analyzer: v9 data infrastructure, Hewson/OFA scientific core."""

    def __init__(self, *args, **kwargs) -> None:
        # Reuse the whole v9 loader (open GRIB, validate units/levels,
        # build the grid, dx/dy, available_hours, terrain, upper_air).
        super().__init__(*args, **kwargs)
        self.method = V10_METHOD
        self._theta_w_cache: dict[int, np.ndarray] = {}
        self._wind_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._v10_tracks: list[dict] | None = None
        # per hour -> list of (coordinates, properties) ready to publish
        self._v10_by_hour: dict[int, list[tuple[np.ndarray, dict]]] | None = None

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
        return fd.detect_fronts_two_scale(
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
        )
        self._v10_tracks = tracks

        by_hour: dict[int, list[tuple[np.ndarray, dict]]] = {
            hour: [] for hour in self.available_hours
        }
        for track in tracks:
            if track["qualityScore"] < MIN_PUBLISH_QUALITY:
                continue
            properties = self._track_properties(track)
            for hour, coordinates in track["lines"].items():
                coordinates = np.asarray(coordinates, dtype=float)
                if len(coordinates) < 2:
                    continue
                by_hour.setdefault(hour, []).append((coordinates, properties))
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
        """Accept the ECMWF guide for interface compatibility.

        v10 corroborates internally via its own synoptic-scale prior
        (two-scale detection), so the external guide is not required.  It
        is stored so a future revision can fold it into ``modelAgreement``.
        """
        self.reference_by_hour = reference_by_hour or {}
        if radius_km_fn is not None:
            self.reference_radius_fn = radius_km_fn


# Backwards-compatible alias, mirroring front_analysis.py.
IconFrontAnalyzerV10 = FrontalAnalysisV10
