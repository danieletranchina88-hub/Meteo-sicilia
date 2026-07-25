"""Synthetic verification of the least-cost ridge extraction (front_ridge.py).

A) the refined line hugs the crest of a curved support field, pulled off a
   deliberately crooked initial guess;
B) a single WIDE support band yields ONE ridge line, not several parallel
   lines (contour-then-mask can produce three lines in one broad gradient;
   the crest path produces one);
C) terrain and the domain edge are avoided when an equal-support detour exists.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import front_ridge as fr

LON = np.arange(3.0, 22.01, 0.15)
LAT = np.arange(34.0, 49.01, 0.15)
LONG, LATG = np.meshgrid(LON, LAT)


def support_ridge(centre_lat_of_lon, width_deg=0.8):
    """Gaussian support band whose crest latitude depends on longitude."""
    crest = centre_lat_of_lon(LONG)
    return np.exp(-((LATG - crest) / width_deg) ** 2)


ok = True

# A) crest-following on a curved ridge -------------------------------------
crest_fn = lambda lon: 42.0 + 1.5 * np.sin((lon - 3.0) / 6.0)
support = support_ridge(crest_fn)
result = {"any_front_support": support, "penalties": {}}
# a crooked initial guess straddling the crest
guess = np.column_stack((
    np.linspace(6.0, 19.0, 8),
    42.0 + 0.9 * np.array([1, -1, 1, -1, 1, -1, 1, -1]),
))
refined = fr.refine_line(result, LON, LAT, guess, corridor_km=200.0)
# error of the refined line vs the true crest
crest_true = crest_fn(refined[:, 0])
err = float(np.mean(np.abs(refined[:, 1] - crest_true)))
guess_err = float(np.mean(np.abs(guess[:, 1] - crest_fn(guess[:, 0]))))
print(f"A) crest-following: errore rifinito={err:.2f}° (guess={guess_err:.2f}°)")
if not (err < 0.3 and err < guess_err):
    print("  FAIL: la linea rifinita non segue la cresta del supporto"); ok = False

# B) one wide band -> one line ---------------------------------------------
wide = support_ridge(lambda lon: 42.0, width_deg=2.2)
res_wide = {"any_front_support": wide, "penalties": {}}
guess_wide = np.column_stack((np.linspace(6.0, 19.0, 6), np.full(6, 42.3)))
refined_wide = fr.refine_line(res_wide, LON, LAT, guess_wide, corridor_km=250.0)
spread = float(np.std(refined_wide[:, 1]))
print(f"B) banda larga: una sola linea, dispersione lat={spread:.2f}°")
if spread > 0.6:
    print("  FAIL: una banda larga non deve produrre una linea frastagliata/multipla"); ok = False

# C) avoid a terrain penalty patch when support is equal -------------------
flat = np.ones_like(support) * 0.7
terrain = np.zeros_like(flat)
terrain[(np.abs(LATG - 42.0) < 0.2)] = 1.0   # a terrain wall along 42N
res_t = {"any_front_support": flat, "penalties": {"terrain": terrain}}
guess_t = np.column_stack((np.linspace(8.0, 18.0, 6), np.full(6, 42.0)))
refined_t = fr.refine_line(res_t, LON, LAT, guess_t, corridor_km=120.0,
                           terrain_weight=6.0)
on_wall = float(np.mean(np.abs(refined_t[:, 1] - 42.0) < 0.15))
print(f"C) evitamento terreno: frazione sul muro={on_wall:.2f}")
if on_wall > 0.6:
    print("  FAIL: il percorso non evita la penalità di terreno"); ok = False

original = np.column_stack((np.linspace(8.0, 18.0, 15), np.full(15, 42.0)))
mild = np.column_stack((
    np.linspace(8.0, 18.0, 30),
    42.0 + 0.08 * np.sin(np.linspace(0.0, np.pi, 30)),
))
detour = np.array([
    [8.0, 42.0], [11.0, 42.0], [11.0, 43.5], [9.0, 43.5],
    [9.0, 41.0], [14.0, 41.0], [14.0, 43.0], [18.0, 42.0],
])
safe_mild = fr.refinement_is_safe(original, mild)
safe_detour = fr.refinement_is_safe(original, detour)
print(f"D) guardia geometria: lieve={safe_mild}, tornanti={safe_detour}")
if not safe_mild or safe_detour:
    print("  FAIL: la rifinitura deve conservare identità e topologia")
    ok = False

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
