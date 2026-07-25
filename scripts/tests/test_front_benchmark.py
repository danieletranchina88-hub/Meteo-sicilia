"""Synthetic checks for the independent front-line benchmark."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import front_benchmark as benchmark


def feature(coordinates, front_type, score=None):
    properties = {"frontType": front_type}
    if score is not None:
        properties["qualityScore"] = score
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coordinates},
        "properties": properties,
    }


def write_geojson(path, features):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"type": "FeatureCollection", "features": features}, handle)


ok = True

with tempfile.TemporaryDirectory() as directory:
    labels = [
        feature([[6.0, 40.0], [10.0, 40.0], [14.0, 40.0]], "cold"),
        feature([[7.0, 43.0], [11.0, 43.0], [15.0, 43.0]], "warm"),
    ]
    predictions = [
        feature([[6.0, 40.2], [10.0, 40.2], [14.0, 40.2]], "cold", 0.9),
        feature([[7.0, 42.8], [11.0, 42.8], [15.0, 42.8]], "cold", 0.7),
        feature([[5.0, 47.0], [9.0, 47.0], [13.0, 47.0]], "warm", 0.3),
    ]
    write_geojson(os.path.join(directory, "labels.geojson"), labels)
    write_geojson(os.path.join(directory, "predictions.geojson"), predictions)
    manifest = {
        "metadata": {"scoreCalibratorFrozen": False},
        "cases": [
            {
                "id": "synthetic",
                "validTime": "2026-01-01T00:00:00Z",
                "split": "test",
                "prediction": "predictions.geojson",
                "label": "labels.geojson",
                "labelSource": "synthetic-test",
            }
        ],
    }
    manifest_path = os.path.join(directory, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    report = benchmark.evaluate_manifest(
        manifest_path, split="test", radius_km=80.0, minimum_overlap=0.60
    )
    # Radius-sensitivity re-run while the temp files still exist. The
    # predictions sit ~22 km from the labels, so a 15 km radius must fail
    # them and every metric must worsen versus the 80 km run.
    sensitivity = benchmark.radius_sensitivity(
        manifest_path, [15.0, 80.0], split="test", minimum_overlap=0.60
    )

counts = report["counts"]
metrics = report["metrics"]
print("conteggi:", counts)
print("metriche:", metrics)

expected_counts = {
    "predictionCount": 3,
    "labelCount": 2,
    "truePositives": 2,
    "falsePositives": 1,
    "falseNegatives": 0,
    "frontCountError": 1,
    "typedMatches": 2,
    "correctTypes": 1,
}
if counts != expected_counts:
    print(f"FAIL: conteggi attesi {expected_counts}")
    ok = False
if metrics["precision"] != 0.6667 or metrics["recall"] != 1.0:
    print("FAIL: precision/recall errate")
    ok = False
if metrics["f1"] != 0.8 or metrics["typeAccuracy"] != 0.5:
    print("FAIL: F1 o accuratezza del tipo errata")
    ok = False
# v15: POD == recall, success ratio == precision, CSI = TP/(TP+FP+FN)
if not (metrics["pod"] == metrics["recall"] == 1.0
        and metrics["successRatio"] == metrics["precision"] == 0.6667
        and metrics["csi"] == 0.6667):
    print("FAIL: POD/success ratio/CSI errati")
    ok = False

# v15: a stricter spatial radius must WORSEN the benchmark (position errors
# cannot hide behind a generous tolerance)
strict, generous = sensitivity[0], sensitivity[1]
print("sensibilita' al raggio:",
      [(row["radiusKm"], row["pod"], row["csi"]) for row in sensitivity])
if not (strict["radiusKm"] == 15.0 and generous["radiusKm"] == 80.0):
    print("FAIL: righe radiusSensitivity non ordinate per raggio")
    ok = False
pod_strict = strict["pod"] if strict["pod"] is not None else 0.0
csi_strict = strict["csi"] if strict["csi"] is not None else 0.0
if not (pod_strict < generous["pod"] and csi_strict < generous["csi"]):
    print("FAIL: un raggio piu' severo deve peggiorare POD e CSI")
    ok = False

if report["qualityScoreDiagnostic"]["eligibleForProbabilityInterpretation"]:
    print("FAIL: il piccolo test sintetico è stato interpretato come probabilistico")
    ok = False
if len(report["qualityScoreDiagnostic"]["bins"]) != 3:
    print("FAIL: bin diagnostici del quality score inattesi")
    ok = False

# v15+: length metrics, distance percentiles, confusion matrix, splits ------
la = report["lengthAccountingKm"]
m = report["metrics"]
print("length accounting:", la)
print("distanze km:", m["meanSymmetricDistanceKm"], m["medianSymmetricDistanceKm"],
      m["p95SymmetricDistanceKm"], " lengthRecall:", m["lengthRecall"])
if not (la["reference"] > 0 and la["detectedReference"] > 0
        and la["detectedReference"] <= la["reference"] + 1e-6):
    print("FAIL: contabilita' delle lunghezze incoerente")
    ok = False
if not (0.0 <= m["lengthRecall"] <= 1.0 and 0.0 <= m["falseLengthRatio"] <= 1.0):
    print("FAIL: lengthRecall/falseLengthRatio fuori [0,1]")
    ok = False
# the two predictions matched ~22 km from labels: median distance must be finite & small
if not (m["medianSymmetricDistanceKm"] is not None
        and m["medianSymmetricDistanceKm"] < 60.0):
    print("FAIL: distanza mediana simmetrica non plausibile")
    ok = False
# type confusion: one label is warm, predicted cold -> confusion["warm"]["cold"] >= 1
conf = report["typeConfusionMatrix"]
print("matrice confusione tipi:", conf)
if conf.get("warm", {}).get("cold", 0) < 1:
    print("FAIL: la confusione warm->cold deve comparire in matrice")
    ok = False
if "stratification" not in report or "continuity" not in report:
    print("FAIL: sezioni stratification/continuity assenti")
    ok = False

# stricter radius must also worsen lengthRecall (reuse the in-block run)
lr_strict = sensitivity[0]["lengthRecall"] or 0.0
lr_gen = sensitivity[1]["lengthRecall"] or 0.0
if not lr_strict < lr_gen:
    print("FAIL: un raggio piu' severo deve peggiorare anche lengthRecall")
    ok = False

# Gold provenance must be machine-auditable, not a free-text source note.
gold_case = {
    "split": "test",
    "labelSource": {
        "provider": "DWD",
        "url": "https://opendata.dwd.de/example.png",
        "accessedAt": "2026-07-25T08:00:00Z",
        "sha256": "a" * 64,
    },
    "annotation": {
        "analyst": "A1",
        "blindToPrediction": True,
        "digitisedAt": "2026-07-25T09:00:00Z",
    },
}
if benchmark.label_provenance_warnings("gold", gold_case):
    print("FAIL: un caso gold completo viene rifiutato")
    ok = False
not_blind = dict(gold_case)
not_blind["annotation"] = {**gold_case["annotation"], "blindToPrediction": False}
if not any("cieca" in warning for warning in
           benchmark.label_provenance_warnings("bad", not_blind)):
    print("FAIL: un'etichetta non cieca non viene segnalata")
    ok = False

print("ESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
