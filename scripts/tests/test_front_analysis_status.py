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

# --- v16: l'aria calda deve stare a sinistra della linea PUBBLICATA --------
# E' la convenzione da cui dipende il lato dei simboli: se salta, un fronte
# freddo viene disegnato come se avanzasse all'indietro. Va garantita sulla
# geometria finale, non ereditata dal candidato.
import numpy as np  # noqa: E402

side = object.__new__(v12.IconSynopticFrontAnalyzer)
side.longitudes = np.arange(3.0, 22.01, 0.20)
side.latitudes = np.arange(34.0, 49.01, 0.20)
_, lat_grid = np.meshgrid(side.longitudes, side.latitudes)
# theta_w cala verso nord: l'aria calda sta a sud.
warm_south = 300.0 - 12.0 / (1.0 + np.exp(-(lat_grid - 42.0) / 1.2))
side._theta_w = lambda hour, level_hpa=850: warm_south
side._sample = lambda field, coordinates: v12.fl._sample(
    field, np.asarray(coordinates, dtype=float), side.longitudes, side.latitudes,
    float(side.longitudes[1] - side.longitudes[0]),
    float(side.latitudes[1] - side.latitudes[0]),
)

# Linea ovest->est lungo il fronte: la sinistra del cammino guarda a nord,
# cioe' verso il FREDDO. Va invertita.
west_to_east = np.column_stack((np.linspace(6.0, 16.0, 40), np.full(40, 42.0)))
props = {"segmentTypes": [{"type": "cold", "start": 0.0, "end": 0.25},
                          {"type": "stationary", "start": 0.25, "end": 1.0}]}
fixed = side._orient_published_line(0, west_to_east, props)
print("linea con il caldo a destra -> invertita:", props.get("warmSideReoriented"))
if not props.get("warmSideReoriented") or fixed[0, 0] <= fixed[-1, 0]:
    print("FAIL: la linea con l'aria calda a destra non e' stata riorientata")
    ok = False
# I segmenti devono seguire il tratto di linea che descrivono.
if [(round(s["start"], 3), round(s["end"], 3)) for s in props["segmentTypes"]] != [
        (0.0, 0.75), (0.75, 1.0)]:
    print("FAIL: i segmenti non sono stati specchiati con la linea")
    ok = False
if props["segmentTypes"][-1]["type"] != "cold":
    print("FAIL: lo specchiamento ha perso il tipo dei segmenti")
    ok = False

# Linea est->ovest: la sinistra guarda a sud, cioe' verso il caldo. Intatta.
east_to_west = west_to_east[::-1].copy()
untouched_props = {}
kept = side._orient_published_line(0, east_to_west, untouched_props)
print("linea gia' corretta -> lasciata:", not untouched_props.get("warmSideReoriented"))
if untouched_props.get("warmSideReoriented") or not np.allclose(kept, east_to_west):
    print("FAIL: una linea gia' orientata bene e' stata invertita")
    ok = False

# --- v16: un confine, una linea ------------------------------------------
# I candidati vengono deduplicati dentro l'ora, ma la pubblicazione e' per
# TRACCIA: due tracce possono consegnare due copie dello stesso fronte senza
# che nulla le confronti. Sul run reale due "cold" si sovrapponevano per il
# 72% della lunghezza entro 60 km.
def line(lon0, lon1, lat, count=40, shift=0.0):
    return np.column_stack((np.linspace(lon0, lon1, count),
                            np.full(count, lat + shift)))


def published_with(entries):
    stub = object.__new__(v12.IconSynopticFrontAnalyzer)
    stub.available_hours = [0]
    stub.hour_to_index = {0: 0}
    stub.datasets = {}
    stub.method = v12.FRONT_METHOD
    stub.source = "synthetic"
    stub._detection_errors = {}
    stub._by_hour = {0: entries}
    stub._tracks = []
    stub.analysis_summary = {}
    return stub.analyze(0)


def props(quality, front_type="cold"):
    return {"frontType": front_type, "qualityScore": quality,
            "confidence": quality, "uncertaintyIndex": 1.0 - quality}


# Due copie quasi coincidenti (spostate di 0,2 gradi, ~22 km): resta la
# migliore.
twins = published_with([
    (line(6.0, 16.0, 44.0), props(0.75)),
    (line(6.0, 16.0, 44.0, shift=0.20), props(0.61)),
])
print("due copie dello stesso fronte -> pubblicate:", len(twins["features"]))
if len(twins["features"]) != 1:
    print("FAIL: lo stesso confine viene disegnato due volte")
    ok = False
elif twins["features"][0]["properties"]["qualityScore"] != 0.75:
    print("FAIL: e' sopravvissuta la copia peggiore")
    ok = False

# Due fronti realmente distinti (4 gradi di latitudine, ~440 km): restano due.
distinct = published_with([
    (line(6.0, 16.0, 44.0), props(0.75)),
    (line(6.0, 16.0, 40.0), props(0.70, "warm")),
])
print("due fronti distinti -> pubblicati:", len(distinct["features"]))
if len(distinct["features"]) != 2:
    print("FAIL: due fronti distinti sono stati fusi")
    ok = False

print("ESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
