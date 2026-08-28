"""Regression tests for v18 published-front topology.

These checks target map artefacts that physical candidate tests cannot see:
contradictory symbols after branch trimming, occluded arcs inheriting cold
symbols, and a weak branch alternating between opposite ends of a shared
front from one hour to the next.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import front_topology as top


def line(lon0, lon1, lat=42.0, count=30):
    return np.column_stack((np.linspace(lon0, lon1, count), np.full(count, lat)))


ok = True

# A) A cold arc may weaken to stationary, never become warm on one end.
cold = {
    "frontType": "cold",
    "segmentTypes": [
        {"start": 0.0, "end": 0.45, "type": "cold", "certainty": 0.8},
        {"start": 0.45, "end": 1.0, "type": "warm", "certainty": 0.7},
    ],
}
top.cohere_feature_type(cold)
cold_labels = {item["type"] for item in cold["segmentTypes"]}
print(f"A) tipo esclusivo cold: {cold_labels}")
if cold_labels != {"cold", "stationary"} or cold["frontType"] != "cold":
    print("  FAIL: il segno opposto deve diventare stazionario")
    ok = False

# B) If trimming retained only a stationary stretch, metadata follows it.
trimmed = {
    "frontType": "cold",
    "segmentTypes": [
        {"start": 0.0, "end": 1.0, "type": "stationary", "certainty": 0.9}
    ],
}
top.cohere_feature_type(trimmed)
print(f"B) tipo dopo trim: {trimmed['frontType']}")
if trimmed["frontType"] != "stationary":
    print("  FAIL: frontType contraddice il solo segmento rimasto")
    ok = False

# C) An occlusion must render as one occluded arc, not inherit cold symbols.
occluded = {
    "frontType": "occluded",
    "segmentTypes": [
        {"start": 0.0, "end": 1.0, "type": "cold", "certainty": 0.8}
    ],
}
top.cohere_feature_type(occluded)
print(f"C) simbolo occluso: {occluded['segmentTypes'][0]['type']}")
if occluded["segmentTypes"] != [{
    "start": 0.0, "end": 1.0, "type": "occluded", "certainty": 1.0,
}]:
    print("  FAIL: l'occlusione deve essere un arco esclusivo")
    ok = False

# D) Shared-trunk trimming must not alternate between west/east branches.
common = {
    "trackId": 12,
    "frontType": "cold",
    "qualityScore": 0.75,
    "segmentTypes": [
        {"start": 0.0, "end": 1.0, "type": "cold", "certainty": 0.8}
    ],
}
by_hour = {
    0: [(line(4.0, 9.0), dict(common))],
    1: [(line(4.1, 9.1), {
        **common, "topologyDeconflicted": True,
        "originalExtentKm": 900.0, "publishedBranchKm": 420.0,
    })],
    2: [(line(14.0, 19.0), {
        **common, "topologyDeconflicted": True,
        "originalExtentKm": 900.0, "publishedBranchKm": 160.0,
    })],
    3: [(line(4.3, 9.3), dict(common))],
}
suppressed, unresolved = top.stabilize_deconflicted_branches(by_hour)
remaining = [hour for hour, entries in by_hour.items() if entries]
print(f"D) branch teleport: soppresse={suppressed}, ore={remaining}")
if suppressed != 1 or remaining != [0, 1, 3] or unresolved != 0:
    print("  FAIL: va rimossa solo la diramazione che salta a est")
    ok = False

# E) Large along-front growth is legal when the shorter core still overlaps.
growth_ok = top.published_transition_is_coherent(
    line(4.0, 10.0), line(4.1, 16.0), 1
)
print(f"E) crescita di estensione: coerente={growth_ok}")
if not growth_ok:
    print("  FAIL: la crescita lungo lo stesso asse non e' un teletrasporto")
    ok = False

motion = top.published_motion_statistics(by_hour)
print(f"F) QC geometria pubblicata: {motion}")
if motion["implausibleTransitions"] != 0:
    print("  FAIL: resta un salto impossibile dopo la stabilizzazione")
    ok = False

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
