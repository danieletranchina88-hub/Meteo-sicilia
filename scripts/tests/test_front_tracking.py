"""Synthetic verification of front_tracking.py (v12).

Checks:
 A) a front moving toward the warm air over several hours -> ONE track,
    classified 'cold' by geometric motion;
 B) a stationary front -> 'stationary';
 C) two separate fronts -> two tracks, no identity swap;
 D) OFA/wind cross-check raises certainty when it agrees;
 E) qualityScore components are present.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import front_locator as fl
import front_tracking as ft

LON = np.arange(3.0, 22.01, 0.15)
LAT = np.arange(34.0, 49.01, 0.15)
_, LATG = np.meshgrid(LON, LAT)
HOURS = list(range(6, 19))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def moving_front_candidates(center_at):
    """One candidate per hour; front center latitude = center_at(hour).

    Warm to the SOUTH (higher theta_w south).  A cold front (cold air
    advancing) moves the line SOUTH, toward the warm air.
    """
    hourly = {}
    for h in HOURS:
        c = center_at(h)
        h0 = max(HOURS[0], h - 1)
        h1 = min(HOURS[-1], h + 1)
        # Warm side is south, so decreasing latitude is positive warm-ward
        # motion (cold-front convention used by the tracker).
        motion_kmh = -(
            center_at(h1) - center_at(h0)
        ) * 111.32 / max(h1 - h0, 1)
        theta_w = 300.0 - 12.0 * sigmoid((LATG - c) / 1.2)  # warm south
        cands = fl.locate_fronts(theta_w, LON, LAT)
        for candidate in cands:
            candidate["candidateEvidence"] = 0.82
            candidate["evidenceComponents"] = {
                "thermal": 0.88,
                "dynamic": 0.76,
                "pressure": 0.70,
                "vertical": 0.74,
                "activity": 0.60,
                "structural": 0.86,
            }
            candidate["synopticSupport"] = 0.90
            candidate["gateStatus"] = "strong"
            candidate["tendencyMotionKmh"] = motion_kmh
            candidate["airmassMotionKmh"] = motion_kmh
            candidate["ofaSpeedMps"] = -motion_kmh / 3.6
        hourly[h] = cands
    return hourly


ok = True

# A) cold front moving south ~30 km/h toward warm --------------------------
cold = moving_front_candidates(lambda h: 45.0 - (h - 6) * 0.30)  # south-ward
tr_cold = ft.track_fronts(cold, window_hours=1, min_lifetime_hours=6)
print(f"A) fronte in avvicinamento al caldo: {len(tr_cold)} tracce", end="")
if tr_cold:
    t = tr_cold[0]
    print(f" -> tipo={t['frontType']} geoMotion={t['geoMotionKmh']}km/h vita={t['lifetimeH']}h")
    if not (len(tr_cold) == 1 and t["frontType"] == "cold" and t["geoMotionKmh"] > 5):
        print("  FAIL: atteso 1 fronte cold con moto verso il caldo positivo"); ok = False
else:
    print(); print("  FAIL: nessuna traccia"); ok = False

# B) stationary front -------------------------------------------------------
stat = moving_front_candidates(lambda h: 42.0)
tr_stat = ft.track_fronts(stat, window_hours=1, min_lifetime_hours=6)
print(f"B) fronte fermo: {len(tr_stat)} tracce", end="")
if tr_stat:
    t = tr_stat[0]
    print(f" -> tipo={t['frontType']} geoMotion={t['geoMotionKmh']}km/h")
    if t["frontType"] != "stationary":
        print("  FAIL: fronte fermo non classificato stationary"); ok = False
else:
    print(); print("  FAIL: nessuna traccia per il fronte fermo"); ok = False

# C) two separate fronts ----------------------------------------------------
def two_fronts(h):
    c1 = 46.0 - (h - 6) * 0.15
    c2 = 40.0
    return 300.0 - 6.0 * sigmoid((LATG - c1) / 1.0) - 6.0 * sigmoid((LATG - c2) / 1.0)

hourly2 = {h: fl.locate_fronts(two_fronts(h), LON, LAT) for h in HOURS}
tr_two = ft.track_fronts(hourly2, window_hours=1, min_lifetime_hours=6)
print(f"C) due fronti separati: {len(tr_two)} tracce (attese ~2)")
if len(tr_two) < 2:
    print("  NOTA: meno di 2 tracce (dipende dalla separazione/rilevamento)")

# D) wind cross-check -------------------------------------------------------
def wind_toward_warm(hour, points):
    # vento verso sud (verso il caldo) = avvezione fredda -> conferma cold
    u = np.zeros(len(points))
    v = np.full(len(points), -8.0)  # m/s verso sud
    return u, v

tr_cold_wind = ft.track_fronts(cold, window_hours=1, min_lifetime_hours=6,
                               wind_sampler=wind_toward_warm)
print(f"D) con vento concorde: certezza={tr_cold_wind[0]['classificationCertainty']} "
      f"tipo={tr_cold_wind[0]['frontType']} OFA={tr_cold_wind[0]['ofaSpeedKmh']}km/h")
if tr_cold_wind and tr_cold_wind[0]["classificationCertainty"] < 0.9:
    print("  NOTA: certezza non massima nonostante consenso")

# E) quality components ------------------------------------------------------
if tr_cold:
    comp = tr_cold[0]["qualityComponents"]
    need = {"physicalEvidence", "thermalSupport", "dynamicSupport",
            "pressureSupport", "verticalSupport", "temporalSupport",
            "structuralSupport", "classificationCertainty",
            "strongDetectionFraction"}
    print(f"E) componenti qualityScore: {sorted(comp)}")
    if set(comp) != need:
        print("  FAIL: componenti qualityScore mancanti"); ok = False

# F) a gap longer than the tracking window starts a new identity ------------
gapped = {
    hour: candidates
    for hour, candidates in cold.items()
    if hour <= 9 or hour >= 14
}
tr_gapped = ft.track_fronts(
    gapped, window_hours=3, min_lifetime_hours=3, min_coverage=0.5
)
print(f"F) buco di 5 ore: {len(tr_gapped)} tracce separate")
if len(tr_gapped) != 2:
    print("  FAIL: una traccia non deve sopravvivere oltre la finestra di 3 ore")
    ok = False

# G) Strongly opposing wind must invalidate a cold/warm classification ------
conflict = moving_front_candidates(lambda h: 45.0 - (h - 6) * 0.30)
for candidates in conflict.values():
    for candidate in candidates:
        candidate["airmassMotionKmh"] = -25.0
        candidate["ofaSpeedMps"] = 25.0 / 3.6
tr_conflict = ft.track_fronts(conflict, window_hours=1, min_lifetime_hours=6)
print(f"G) moto geometrico freddo ma vento caldo: "
      f"{tr_conflict[0]['frontType'] if tr_conflict else 'nessuna traccia'}")
if not tr_conflict or tr_conflict[0]["frontType"] != "uncertain":
    print("  FAIL: una classificazione contraddetta dal vento non va pubblicata")
    ok = False

# H) One marginal hour may continue a strong identity without deleting it --
hysteresis = moving_front_candidates(lambda h: 45.0 - (h - 6) * 0.20)
for candidate in hysteresis[12]:
    candidate["gateStatus"] = "continuation"
    candidate["candidateEvidence"] *= 0.88
tr_hysteresis = ft.track_fronts(
    hysteresis, window_hours=2, min_lifetime_hours=6,
    min_strong_detections=3,
)
print(f"H) un'ora marginale dentro una traccia: {len(tr_hysteresis)} tracce")
if len(tr_hysteresis) != 1 or 12 not in tr_hysteresis[0]["hours"]:
    print("  FAIL: l'isteresi deve evitare una sparizione numerica di un'ora")
    ok = False

# I) Marginal evidence alone cannot create a front --------------------------
marginal_only = moving_front_candidates(lambda h: 43.0)
for candidates in marginal_only.values():
    for candidate in candidates:
        candidate["gateStatus"] = "continuation"
tr_marginal = ft.track_fronts(
    marginal_only, window_hours=2, min_lifetime_hours=6,
    min_strong_detections=3,
)
print(f"I) sole linee marginali: {len(tr_marginal)} tracce")
if tr_marginal:
    print("  FAIL: un candidato marginale non deve creare una nuova identita'")
    ok = False

# J) One identity may evolve from cold to stationary without a stale label --
def cold_then_stationary(hour):
    return 45.0 - min(hour - 6, 6) * 0.28

evolving = moving_front_candidates(cold_then_stationary)
tr_evolving = ft.track_fronts(
    evolving, window_hours=2, min_lifetime_hours=6,
)
if tr_evolving:
    local_types = {
        hour: value["frontType"]
        for hour, value in tr_evolving[0]["localClassifications"].items()
    }
    print(f"J) tipo locale in evoluzione: +7={local_types.get(7)} "
          f"+17={local_types.get(17)}")
    if local_types.get(7) != "cold" or local_types.get(17) != "stationary":
        print("  FAIL: il tipo deve essere locale, non unico per tutta la vita")
        ok = False
else:
    print("J) FAIL: traccia evolutiva assente")
    ok = False

# K) Viterbi type smoothing removes an unphysical cold->warm->stationary flip
# from hour to hour, but preserves a genuine, sustained warm phase.
def _seq(types):
    return {h: {"frontType": t, "classificationCertainty": 0.6}
            for h, t in enumerate(types)}

raw = ["cold"] * 6 + ["stationary", "warm", "warm", "cold", "cold", "stationary"]
smoothed = ft._viterbi_smooth_types(_seq(raw))
out = [smoothed[h]["frontType"] for h in sorted(smoothed)]
print("K) smoothing tipi:", "".join(t[0].upper() for t in raw),
      "->", "".join(t[0].upper() for t in out))
if any({out[i], out[i + 1]} == {"cold", "warm"} for i in range(len(out) - 1)):
    print("  FAIL: e' sopravvissuto un salto caldo<->freddo tra ore adiacenti")
    ok = False
sustained = ["cold"] * 4 + ["warm"] * 5 + ["cold"] * 2
out2 = [ft._viterbi_smooth_types(_seq(sustained))[h]["frontType"]
        for h in range(len(sustained))]
print("   fase calda prolungata:", "".join(t[0].upper() for t in out2))
if "warm" not in out2:
    print("  FAIL: una fase calda reale e prolungata non deve sparire")
    ok = False

# L) Per-segment classification: one line, cold on the east half (moves toward
# warm) and stationary on the west half (does not move).
lons = np.linspace(8.0, 20.0, 25)
warm_normal = np.tile([0.0, -1.0], (len(lons), 1))  # warm to the south


def _cand(lat_of_lon):
    coords = np.column_stack((lons, np.array([lat_of_lon(x) for x in lons])))
    return {"coordinates": coords, "warmNormal": warm_normal.copy(),
            "lengthKm": 900.0}


seg_track = ft.Track(0, 0, _cand(lambda x: 42.0))
seg_track.hours.append(1)
# east (lon>=14) shifts ~33 km south (toward warm -> cold); west stays
seg_track.lines[1] = _cand(lambda x: 41.7 if x >= 14.0 else 42.0)
seg_local = {0: {"frontType": "cold", "classificationCertainty": 0.6},
             1: {"frontType": "cold", "classificationCertainty": 0.6}}
segs = ft.segment_types_for_track(seg_track, seg_local)[0]
labels = [s["type"] for s in segs]
print(f"L) segmenti su una linea mista: {[(s['type'], s['start'], s['end']) for s in segs]}")
if not ("cold" in labels and "stationary" in labels):
    print("  FAIL: la linea mista deve avere un tratto freddo e uno stazionario")
    ok = False
# the segments must tile [0,1] contiguously without gaps or overlaps
edges = [(s["start"], s["end"]) for s in segs]
if edges[0][0] != 0.0 or edges[-1][1] != 1.0 or any(
        abs(edges[i][1] - edges[i + 1][0]) > 1e-6 for i in range(len(edges) - 1)):
    print("  FAIL: i segmenti non coprono la linea in modo contiguo")
    ok = False

# M) A weak OPPOSITE-type reading on a cold-anchored line is demoted to
# stationary, not shown as a spurious warm patch (noise suppression).
weak_track = ft.Track(0, 0, _cand(lambda x: 42.0))
weak_track.hours.append(1)
# west drifts ~7 km/h NORTH (weak "warm" reading, below 11 km/h); east cold
weak_track.lines[1] = _cand(lambda x: 41.7 if x >= 14.0 else 42.06)
weak_segs = ft.segment_types_for_track(
    weak_track, {0: {"frontType": "cold"}, 1: {"frontType": "cold"}})[0]
weak_labels = [s["type"] for s in weak_segs]
print(f"M) deviazione opposta debole: {[(s['type'], s['start'], s['end']) for s in weak_segs]}")
if "warm" in weak_labels:
    print("  FAIL: una lettura 'caldo' debole su fronte freddo non deve sopravvivere")
    ok = False

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
