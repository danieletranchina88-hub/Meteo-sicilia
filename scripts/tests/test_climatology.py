"""QC tests for candidate climatological thresholds.

The committed July quantiles are intentionally exploratory.  They must remain
disabled until enough independent cases exist and a labelled benchmark has
explicitly validated them.
"""

import json
import os
import sys

import numpy as np

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SCRIPTS)
import front_analysis_v12 as v12
import front_detection as fd

ok = True
KEYS = {
    "synoptic_tfp_weak",
    "synoptic_tfp_full",
    "synoptic_abz_weak",
    "synoptic_abz_full",
    "refined_tfp_weak",
    "refined_tfp_full",
    "refined_abz_weak",
    "refined_abz_full",
}


def fail(message):
    global ok
    print(f"  FAIL: {message}")
    ok = False


# 1) Six July runs are insufficient and cannot silently affect production.
july, july_status = v12.threshold_climatology_status("07")
print(f"1) luglio operativo: {july is not None}; stato={july_status}")
if july is not None:
    fail("la piccola climatologia esplorativa è stata attivata")
if july_status != "insufficient-independent-runs":
    fail(f"stato QC inatteso: {july_status}")

# 2) Another month never borrows thresholds from the wrong season.
january, january_status = v12.threshold_climatology_status("01")
print(f"2) gennaio operativo: {january is not None}; stato={january_status}")
if january is not None or january_status != "month-not-covered":
    fail("è stato applicato un fallback stagionale non valido")

# 3) The candidate file remains complete and can be passed explicitly to the
# detector for offline benchmark experiments.
with open(v12.CLIMATOLOGY_PATH, encoding="utf-8") as handle:
    raw = json.load(handle)
syn = raw["07"]["synoptic"]["all"]
ref = raw["07"]["refined"]["all"]
candidate = {
    "synoptic_tfp_weak": syn["tfp_weak"],
    "synoptic_tfp_full": syn["tfp_full"],
    "synoptic_abz_weak": syn["abz_weak"],
    "synoptic_abz_full": syn["abz_full"],
    "refined_tfp_weak": ref["tfp_weak"],
    "refined_tfp_full": ref["tfp_full"],
    "refined_abz_weak": ref["abz_weak"],
    "refined_abz_full": ref["abz_full"],
}
print(f"3) soglie candidate complete: {sorted(candidate)}")
if set(candidate) != KEYS:
    fail("il file candidato non espone le otto soglie")
if not (
    candidate["synoptic_tfp_full"] < candidate["synoptic_tfp_weak"] < 0
    and candidate["refined_tfp_full"] < candidate["refined_tfp_weak"] < 0
    and 0 < candidate["synoptic_abz_weak"] < candidate["synoptic_abz_full"]
    and 0 < candidate["refined_abz_weak"] < candidate["refined_abz_full"]
):
    fail("le soglie candidate non rispettano l'ordine fisico")

lon = np.arange(3.0, 22.01, 0.15)
lat = np.arange(34.0, 49.01, 0.15)
_, lat_grid = np.meshgrid(lon, lat)
theta_w = 300.0 - 12.0 / (1.0 + np.exp(-(lat_grid - 42.0) / 1.2))
tuned = fd.detect_fronts_two_scale(theta_w, lon, lat, **candidate)
print(f"   detector offline: {len(tuned)} candidati")
if not isinstance(tuned, list):
    fail("il detector non accetta le soglie candidate esplicite")

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
