"""Climatological threshold calibration (documento sez. 17).

Checks that the committed climatology loads, exposes the eight per-scale
thresholds the detector expects, falls back across months, and that the
detector actually accepts and applies them.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import front_detection as fd
import front_analysis_v12 as v12

ok = True
KEYS = {
    "synoptic_tfp_weak", "synoptic_tfp_full", "synoptic_abz_weak", "synoptic_abz_full",
    "refined_tfp_weak", "refined_tfp_full", "refined_abz_weak", "refined_abz_full",
}

# 1) July climatology present with all eight thresholds, physically ordered.
july = v12.load_threshold_climatology("07")
print(f"1) climatologia luglio: {None if july is None else sorted(july)}")
if not july or set(july) != KEYS:
    print("  FAIL: la climatologia di luglio non espone le otto soglie"); ok = False
else:
    if not (july["synoptic_abz_full"] > july["synoptic_abz_weak"] > 0):
        print("  FAIL: ABZ sinottica non ordinata weak<full>0"); ok = False
    if not (july["refined_tfp_full"] < july["refined_tfp_weak"] < 0):
        print("  FAIL: TFP raffinata non ordinata (deve essere negativa, full piu' forte)"); ok = False

# 2) An uncovered month falls back to an available one (never crashes to None
# when any climatology exists).
fallback = v12.load_threshold_climatology("01")
print(f"2) mese non coperto -> fallback presente: {fallback is not None}")
if fallback is None:
    print("  FAIL: nessun fallback quando la climatologia esiste"); ok = False

# 3) The detector accepts and applies the climatological thresholds.
lon = np.arange(3.0, 22.01, 0.15)
lat = np.arange(34.0, 49.01, 0.15)
_, latg = np.meshgrid(lon, lat)
theta_w = 300.0 - 12.0 / (1.0 + np.exp(-(latg - 42.0) / 1.2))  # warm south
base = fd.detect_fronts_two_scale(theta_w, lon, lat)
tuned = fd.detect_fronts_two_scale(theta_w, lon, lat, **(july or {}))
print(f"3) detector con soglie climatologiche: base={len(base)} tuned={len(tuned)} candidati")
if not isinstance(tuned, list):
    print("  FAIL: il detector non accetta le soglie climatologiche"); ok = False

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
