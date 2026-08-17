"""Synthetic verification of front_sections.py (cross-front profiles).

 1) a coherent synthetic front has high thermal support, a plausible width
    and a gradient maximum on the detected line;
 2) an identical (slightly shifted) 925 hPa profile gives HIGH vertical
    coherence;
 3) a displaced or opposite-sign 925 hPa profile REDUCES the coherence;
 4) fully missing profiles are handled: neutral (None) coherence, zero
    valid fraction, no NaN crashes.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import front_sections as fs

LON = np.arange(3.0, 22.01, 0.15)
LAT = np.arange(34.0, 49.01, 0.15)
LONG, LATG = np.meshgrid(LON, LAT)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def front_field(center_lat, width_deg=0.55, amplitude_k=8.0):
    """theta_w with warm air SOUTH of ``center_lat`` (transition ~width)."""
    return 290.0 + amplitude_k * sigmoid(-(LATG - center_lat) / width_deg)


line = np.column_stack((np.linspace(6.0, 19.0, 40), np.full(40, 42.0)))
warm_normal = np.tile([0.0, -1.0], (40, 1))   # warm side is south

ok = True

# 1) coherent front ---------------------------------------------------------
theta_850 = front_field(42.0)
prof_850 = fs.cross_profiles(theta_850, LON, LAT, line, warm_normal)
diag_850 = fs.profile_diagnostics(prof_850)
print("1) fronte coerente:", diag_850)
if not (diag_850["profileValidFraction"] > 0.95
        and diag_850["profileThermalSupport"] > 0.9
        and diag_850["profilePeakGradient"] > 2.0
        and 25.0 <= diag_850["frontWidthKm"] <= 300.0
        and abs(diag_850["frontOffsetKm"]) <= 45.0
        and diag_850["airMassHomogeneity"] > 0.3):
    print("  FAIL: diagnostica del profilo non plausibile"); ok = False

# 2) coherent 925 profile -> high vertical coherence ------------------------
theta_925 = front_field(42.15, width_deg=0.6, amplitude_k=7.0)  # slight tilt
diag_925 = fs.profile_diagnostics(
    fs.cross_profiles(theta_925, LON, LAT, line, warm_normal))
coh_good = fs.vertical_coherence(diag_850, diag_925)
print(f"2) coerenza verticale con 925 concorde: {coh_good}")
if coh_good is None or coh_good < 0.60:
    print("  FAIL: profili coerenti devono dare coerenza alta"); ok = False

# 3) misaligned / opposite-sign 925 reduces coherence -----------------------
theta_925_far = front_field(44.5)             # gradient max ~280 km away
diag_925_far = fs.profile_diagnostics(
    fs.cross_profiles(theta_925_far, LON, LAT, line, warm_normal))
coh_far = fs.vertical_coherence(diag_850, diag_925_far)
theta_925_opp = 290.0 + 8.0 * sigmoid((LATG - 42.0) / 0.55)  # warm NORTH
diag_925_opp = fs.profile_diagnostics(
    fs.cross_profiles(theta_925_opp, LON, LAT, line, warm_normal))
coh_opp = fs.vertical_coherence(diag_850, diag_925_opp)
print(f"3) 925 disallineato: {coh_far}   925 segno opposto: {coh_opp}")
if coh_far is None or coh_good is None or not coh_far < coh_good:
    print("  FAIL: un massimo 925 lontano deve ridurre la coerenza"); ok = False
if coh_opp is None or not (coh_opp < 0.25 and coh_opp < coh_good):
    print("  FAIL: contrasto di segno opposto deve dare coerenza bassa"); ok = False

# 4) fully missing profiles -------------------------------------------------
nan_field = np.full_like(theta_850, np.nan)
diag_nan = fs.profile_diagnostics(
    fs.cross_profiles(nan_field, LON, LAT, line, warm_normal))
coh_nan = fs.vertical_coherence(diag_850, diag_nan)
coh_none = fs.vertical_coherence(diag_850, None)
print(f"4) profili mancanti: validFraction={diag_nan['profileValidFraction']} "
      f"coerenza(NaN)={coh_nan} coerenza(None)={coh_none}")
if diag_nan["profileValidFraction"] != 0.0 or coh_nan is not None or coh_none is not None:
    print("  FAIL: dati 925 mancanti devono essere neutri (None), non contrari")
    ok = False
empty = fs.profile_diagnostics(np.empty((0, 7)))
if empty["profileValidFraction"] != 0.0:
    print("  FAIL: linea vuota non gestita"); ok = False

# 5) complete 925-850-700 structure ----------------------------------------
theta_700 = front_field(41.75, width_deg=0.75, amplitude_k=5.5)
diag_700 = fs.profile_diagnostics(
    fs.cross_profiles(theta_700, LON, LAT, line, warm_normal)
)
multi = fs.multilevel_vertical_coherence(diag_925, diag_850, diag_700)
missing_upper = fs.multilevel_vertical_coherence(diag_925, diag_850, None)
print("5) struttura verticale 925-850-700:", multi)
if (
    multi["verticalCoherence3Level"] is None
    or multi["verticalCoherence3Level"] < 0.55
    or multi["frontalTiltKm"] is None
):
    print("  FAIL: struttura coerente a tre livelli non riconosciuta")
    ok = False
if missing_upper["verticalCoherence3Level"] != coh_good:
    print("  FAIL: 700 mancante non resta neutrale")
    ok = False

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
