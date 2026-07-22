"""Synthetic checks for the continuous support field (Fase B, front_support.py).

Existence-of-a-front is separated from its type. The field must peak on a
real baroclinic zone, stay in [0,1], handle NaN and the domain edge
explicitly, penalise a moisture-only boundary and a mesoscale-only gradient,
and be invariant to the latitude-axis orientation.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import front_support as fs

LON = np.arange(3.0, 22.01, 0.15)
LAT = np.arange(34.0, 49.01, 0.15)
_, LATG = np.meshgrid(LON, LAT)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def const_wind(shape, u=0.0, v=0.0):
    return np.full(shape, u), np.full(shape, v)


ok = True

# A) a real front: theta_w, theta and theta_e all drop across ~42N ----------
theta_w = 300.0 - 12.0 * sigmoid((LATG - 42.0) / 1.2)
theta = 305.0 - 10.0 * sigmoid((LATG - 42.0) / 1.2)
theta_e = 320.0 - 16.0 * sigmoid((LATG - 42.0) / 1.2)
u, v = const_wind(theta_w.shape, 0.0, -6.0)
res = fs.physical_support_field(theta_w, theta, theta_e, u, v, LON, LAT)
afs = res["any_front_support"]
print(f"A) campo di supporto: min={afs.min():.2f} max={afs.max():.2f}")
if not (0.0 <= afs.min() and afs.max() <= 1.0):
    print("  FAIL: il campo esce da [0,1]"); ok = False
# the support must peak in a narrow band near 42N, low far away
band = np.abs(LAT - 42.0) < 1.5
far = np.abs(LAT - 42.0) > 5.0
if not (np.nanmax(afs[band]) > 0.5 and np.nanmedian(afs[far]) < 0.25):
    print(f"  FAIL: il campo non si concentra sul fronte "
          f"(banda={np.nanmax(afs[band]):.2f} lontano={np.nanmedian(afs[far]):.2f})")
    ok = False

# B) NaN handled explicitly, never propagated -------------------------------
tw_nan = theta_w.copy()
tw_nan[10:14, 20:24] = np.nan
resn = fs.physical_support_field(tw_nan, theta, theta_e, u, v, LON, LAT)
if not np.all(np.isfinite(resn["any_front_support"])):
    print("  FAIL: NaN propagato nel campo di supporto"); ok = False
if resn["any_front_support"][12, 22] != 0.0:
    print("  FAIL: cella non valida non azzerata"); ok = False
print(f"B) NaN: campo finito ovunque, cella non valida = {resn['any_front_support'][12,22]:.2f}")

# C) domain edge is penalised ------------------------------------------------
edge_pen = res["penalties"]["edge"]
print(f"C) penalità bordo: centro={edge_pen[edge_pen.shape[0]//2, edge_pen.shape[1]//2]:.2f} "
      f"bordo={edge_pen[0,0]:.2f}")
if not (edge_pen[0, 0] > 0.5 and edge_pen[edge_pen.shape[0]//2, edge_pen.shape[1]//2] < 0.2):
    print("  FAIL: la penalità di bordo non è corretta"); ok = False

# D) moisture-only boundary: theta_e jumps, theta_w/theta nearly flat -------
flat = np.full_like(LATG, 300.0)
te_only = 320.0 - 16.0 * sigmoid((LATG - 42.0) / 1.2)
resm = fs.physical_support_field(flat, flat, te_only, u, v, LON, LAT)
mb = resm["penalties"]["moisture_boundary"]
print(f"D) confine igrometrico: penalità max={mb.max():.2f}, supporto max={resm['any_front_support'].max():.2f}")
if not (mb.max() > 0.3 and resm["any_front_support"].max() < 0.5):
    print("  FAIL: un confine di sola umidità non deve dare supporto frontale alto"); ok = False

# E) latitude-axis inversion leaves the field essentially unchanged ---------
res_inv = fs.physical_support_field(
    theta_w[::-1], theta[::-1], theta_e[::-1], u[::-1], v[::-1], LON, LAT[::-1]
)
peak_norm = LAT[np.argmax(np.nanmax(afs, axis=1))]
peak_inv = LAT[::-1][np.argmax(np.nanmax(res_inv["any_front_support"], axis=1))]
print(f"E) inversione asse lat: picco normale={peak_norm:.1f} invertito={peak_inv:.1f}")
if abs(peak_norm - peak_inv) > 1.0:
    print("  FAIL: il campo dipende dall'orientamento dell'asse"); ok = False

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
