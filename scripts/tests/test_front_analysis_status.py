"""Regression: an empty analysis is valid, a diagnostic error is unavailable."""

import os
import sys

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

print("ESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
