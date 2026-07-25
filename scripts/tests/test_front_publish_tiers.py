"""Regression: a modest-but-coherent track is published, not silently dropped.

A track just under the standard publish gate (MIN_PUBLISH_QUALITY /
MAX_PUBLISH_UNCERTAINTY) is not automatically noise. It must still be
published, explicitly tagged ``confidenceTier == "low"``, instead of
disappearing as if nothing had been detected at all.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

import front_analysis_v12 as v12


def analyzer_with(raw_tracks):
    analyzer = object.__new__(v12.IconSynopticFrontAnalyzer)
    analyzer.available_hours = [0, 1, 2, 3]
    analyzer.hour_to_index = {h: h for h in analyzer.available_hours}
    analyzer.datasets = {}
    analyzer.method = v12.FRONT_METHOD
    analyzer.source = "synthetic"
    analyzer._by_hour = None
    analyzer._tracks = None
    analyzer._pipeline_diag = {}
    analyzer._rejected = {}
    analyzer._support_cache = {}
    analyzer._detection_errors = {}
    analyzer._threshold_climatology = None
    analyzer._threshold_climatology_status = "test-defaults"
    analyzer.run_month = "01"
    analyzer.analysis_summary = None
    analyzer.longitudes = np.linspace(10.0, 16.0, 5)
    analyzer.latitudes = np.linspace(36.0, 42.0, 5)
    analyzer._weak = {}
    analyzer._detect_hour = lambda hour: []
    v12.ftk.track_fronts = lambda *args, **kwargs: raw_tracks
    return analyzer


def _line(lat0=40.0):
    return [[12.0, lat0], [13.0, lat0 + 0.3], [14.0, lat0 + 0.6]]


def _base_track(track_id, quality, uncertainty):
    hours = [0, 1, 2, 3]
    return {
        "id": track_id,
        "lines": {h: _line() for h in hours},
        "hours": hours,
        "coreHours": hours,
        "frontType": "cold",
        "qualityScore": quality,
        "uncertaintyIndex": uncertainty,
        "uncertaintyClass": "low",
        "lifetimeH": 4,
        "diagnostics": {"deltaThetaW": 2.0},
        "localClassifications": {
            h: {"frontType": "cold", "classificationCertainty": 0.8}
            for h in hours
        },
        "hourlyQuality": {h: quality for h in hours},
        "hourlyUncertainty": {h: uncertainty for h in hours},
        "detectionQuality": {h: quality for h in hours},
        "trackingConfidence": {h: 0.8 for h in hours},
        "classificationConfidence": {h: 0.8 for h in hours},
        "observations": {
            h: {"gateStatus": "strong", "diagnosis": "synoptic-front",
                "diagnostics": {"deltaThetaW": 2.0}}
            for h in hours
        },
        "recoveredHours": [],
        "motionVotes": {},
        "classificationCertainty": 0.8,
        "qualityComponents": {},
    }


ok = True

standard = _base_track(1, v12.MIN_PUBLISH_QUALITY + 0.05,
                        v12.MAX_PUBLISH_UNCERTAINTY - 0.05)
low = _base_track(2, v12.MIN_PUBLISH_QUALITY_LOW + 0.02,
                   v12.MAX_PUBLISH_UNCERTAINTY_LOW - 0.02)
dropped = _base_track(3, v12.MIN_PUBLISH_QUALITY_LOW - 0.10,
                       v12.MAX_PUBLISH_UNCERTAINTY_LOW + 0.10)

analyzer = analyzer_with([standard, low, dropped])
result = analyzer.analyze(0)
tiers = {f["properties"]["trackId"]: f["properties"]["confidenceTier"]
         for f in result["features"]}
print("livelli pubblicati:", tiers)
print("riepilogo:", {
    k: analyzer.analysis_summary[k]
    for k in ("publishedTracks", "publishedTracksStandard",
              "publishedTracksLowConfidence")
})

if tiers.get(1) != "standard":
    print("FAIL: la traccia sopra soglia standard deve avere confidenceTier=standard")
    ok = False
if tiers.get(2) != "low":
    print("FAIL: la traccia nella fascia provvisoria deve essere pubblicata come 'low'")
    ok = False
if 3 in tiers:
    print("FAIL: una traccia sotto la soglia provvisoria non deve essere pubblicata")
    ok = False
if analyzer.analysis_summary["publishedTracks"] != 2:
    print("FAIL: publishedTracks deve contare standard+low")
    ok = False
if analyzer.analysis_summary["publishedTracksStandard"] != 1:
    print("FAIL: publishedTracksStandard deve contare solo la traccia standard")
    ok = False
if analyzer.analysis_summary["publishedTracksLowConfidence"] != 1:
    print("FAIL: publishedTracksLowConfidence deve contare solo la traccia provvisoria")
    ok = False

print("ESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
