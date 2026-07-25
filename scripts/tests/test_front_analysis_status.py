"""Regression: an empty analysis is valid, a diagnostic error is unavailable."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import front_analysis_v12 as v12


def analyzer_with(detector):
    analyzer = object.__new__(v12.IconSynopticFrontAnalyzer)
    analyzer.available_hours = [0]
    analyzer.hour_to_index = {0: 0}
    analyzer.datasets = {}
    analyzer.method = v12.FRONT_METHOD
    analyzer.source = "synthetic"
    analyzer._by_hour = None
    analyzer._tracks = None
    analyzer._pipeline_diag = {}
    analyzer._rejected = {}
    analyzer._detection_errors = {}
    analyzer._threshold_climatology = None
    analyzer._threshold_climatology_status = "test-defaults"
    analyzer.run_month = "01"
    analyzer.analysis_summary = None
    analyzer._detect_hour = detector
    return analyzer


ok = True

empty = analyzer_with(lambda hour: [])
empty_result = empty.analyze(0)
print("vuoto:", empty_result["properties"]["analysisStatus"])
if empty_result["features"] != []:
    print("FAIL: l'analisi vuota contiene feature")
    ok = False
if empty_result["properties"]["analysisStatus"] != "no-robust-fronts":
    print("FAIL: zero fronti non è stato trattato come risultato valido")
    ok = False
if empty.analysis_summary["publishedTracks"] != 0:
    print("FAIL: riepilogo vuoto incoerente")
    ok = False


def broken_detector(hour):
    raise ValueError("errore sintetico")


broken = analyzer_with(broken_detector)
broken_result = broken.analyze(0)
print("errore:", broken_result["properties"]["analysisStatus"])
if broken_result["properties"]["analysisStatus"] != "unavailable":
    print("FAIL: errore diagnostico confuso con assenza di fronti")
    ok = False
if broken.analysis_summary["analysisStatus"] != "partially-unavailable":
    print("FAIL: riepilogo run non segnala l'errore diagnostico")
    ok = False

omega = np.zeros(1000, dtype=float)
omega[0] = 80.0
omega[1:6] = 10.0
clean_omega = v12._mask_omega_spikes(omega)
print("omega QC:", v12._omega_bulk_is_plausible(omega),
      "spike masked:", np.isnan(clean_omega[0]))
if not v12._omega_bulk_is_plausible(omega) or not np.isnan(clean_omega[0]):
    print("FAIL: un outlier isolato non deve eliminare la diagnostica omega")
    ok = False

# --- v15: per-hour properties vs track properties --------------------------
stub = analyzer_with(lambda hour: [])
stub.method = v12.FRONT_METHOD
fake_track = {
    "frontType": "cold",
    "qualityScore": 0.70, "uncertaintyIndex": 0.30,
    "trackQualityScore": 0.70, "trackUncertaintyIndex": 0.30,
    "uncertaintyClass": "low",
    "geoMotionKmh": 12.0, "diagnostics": {"deltaThetaW": 3.0},
    "diagnosis": "synoptic-front", "lifetimeH": 8, "id": 1,
    "hours": [5, 6, 8], "recoveredHours": [],
    "hourlyQuality": {5: 0.80, 6: 0.44, 8: 0.78},
    "hourlyUncertainty": {5: 0.20, 6: 0.60, 8: 0.22},
    "detectionQuality": {5: 0.82, 6: 0.40, 8: 0.80},
    "trackingConfidence": {5: 0.9, 6: 0.85, 8: 0.9},
    "classificationConfidence": {5: 0.8, 6: 0.5, 8: 0.8},
    "observations": {
        5: {"gateStatus": "strong", "diagnosis": "synoptic-front",
            "diagnostics": {"deltaThetaW": 3.4}},
        6: {"gateStatus": "continuation", "diagnosis": "synoptic-front",
            "diagnostics": {"deltaThetaW": 1.1}},
        8: {"gateStatus": "strong", "diagnosis": "synoptic-front",
            "diagnostics": {"deltaThetaW": 3.2}},
    },
    "motionVotes": {}, "classificationCertainty": 0.7,
    "qualityComponents": {},
}
observed = stub._track_properties(fake_track, hour=6)
print("ora osservata debole:", observed["qualityScore"],
      "track:", observed["trackQualityScore"])
if not (observed["qualityScore"] == 0.44
        and observed["trackQualityScore"] == 0.70
        and observed["detectionQuality"] == 0.40
        and observed["uncertaintyIndex"] == 0.60
        and observed["diagnostics"].get("deltaThetaW") == 1.1
        and observed["trackDiagnostics"].get("deltaThetaW") == 3.0):
    print("FAIL: l'ora osservata deve esporre la vista oraria + quella di traccia")
    ok = False
if observed["reasoning"] == stub._front_reasoning(fake_track):
    print("FAIL: il ragionamento dell'ora debole non deve ripetere la mediana")
    ok = False

interpolated = stub._track_properties(fake_track, hour=7)
print("ora interpolata:", interpolated["qualityScore"],
      "detectionQuality:", interpolated["detectionQuality"])
neighbour_mean = 0.5 * (0.44 + 0.78)
if not (interpolated["detectionQuality"] is None
        and interpolated["trackingConfidence"] is None
        and interpolated["qualityScore"] < neighbour_mean
        and interpolated["qualityScore"] < interpolated["trackQualityScore"]):
    print("FAIL: un'ora interpolata deve avere detectionQuality nulla e penalita'")
    ok = False

print("ESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
