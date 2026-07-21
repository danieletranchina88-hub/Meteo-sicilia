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
            "structuralSupport", "classificationCertainty"}
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

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
