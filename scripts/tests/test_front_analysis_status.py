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
            "diagnostics": {
                "deltaThetaW": 1.1, "positionUncertaintyKm": 28.0,
                "methodAgreementCount": 2, "methodAvailability": 3,
            }},
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
        and observed["trackDiagnostics"].get("deltaThetaW") == 3.0
        and observed["existenceConfidence"] == 0.44
        and observed["typeConfidence"] == 0.5
        and observed["positionUncertaintyKm"] == 28.0
        and observed["methodAgreement"] == {"count": 2.0, "available": 3.0}
        and observed["confidenceSemantics"]
        == "heuristic-not-calibrated-probability"):
    print("FAIL: l'ora osservata deve esporre la vista oraria + quella di traccia")
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

# --- shared-ridge topology: one trunk, independent branch retained ----------
owner = np.array([[7.0, 44.0], [17.0, 44.0]])
branching = np.array([
    [4.0, 42.0], [8.0, 44.0], [15.0, 44.0], [20.0, 46.0]
])
topology, changed = v12.deconflict_shared_front_trunks([
    (owner, {"trackId": 1, "qualityScore": 0.80, "typeConfidence": 0.8}),
    (branching, {
        "trackId": 2, "qualityScore": 0.60, "typeConfidence": 0.7,
        "segmentTypes": [
            {"start": 0.0, "end": 0.55, "type": "cold"},
            {"start": 0.55, "end": 1.0, "type": "stationary"},
        ],
    }),
])
print("topologia condivisa:", changed, "feature:", len(topology))
branch = next(item for item in topology if item[1].get("trackId") == 2)
branch_support = v12.fd.line_support_fraction(branch[0], [owner], 25.0)
if not (
    changed == 1
    and len(topology) == 2
    and branch[1].get("topologyDeconflicted") is True
    and branch[1].get("suppressedOverlapWithTrackIds") == [1]
    and branch_support < 0.25
    and branch[1]["segmentTypes"][0]["start"] == 0.0
    and branch[1]["segmentTypes"][-1]["end"] == 1.0
):
    print("FAIL: il tronco comune non e' stato assegnato a una sola traccia")
    ok = False

# A front crossing another at one point is a real junction, not a duplicate.
crossing = np.array([[12.0, 40.0], [12.0, 48.0]])
crossed, crossing_changes = v12.deconflict_shared_front_trunks([
    (owner, {"trackId": 1, "qualityScore": 0.80}),
    (crossing, {"trackId": 3, "qualityScore": 0.70}),
])
if crossing_changes != 0 or len(crossed) != 2:
    print("FAIL: una giunzione puntuale e' stata confusa con un tronco duplicato")
    ok = False

print("ESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
