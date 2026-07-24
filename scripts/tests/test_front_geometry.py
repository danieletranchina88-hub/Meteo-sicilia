"""Synthetic verification of front_geometry.py continuous penalties."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import front_geometry as fg

ok = True

# 1) a straight, gently sloped front: low penalties, not a loop -------------
straight = np.column_stack((np.linspace(8.0, 18.0, 20), np.linspace(44.0, 41.0, 20)))
gm = fg.geometry_metrics(straight)
print("1) linea dritta:", {k: gm[k] for k in
      ("curvaturePenalty", "closedLoop", "selfIntersections", "tangentReversals")})
if not (gm["curvaturePenalty"] < 0.2 and not gm["closedLoop"]
        and gm["selfIntersections"] == 0 and gm["lengthKm"] > 500):
    print("  FAIL: una linea dritta non deve avere penalita' o anelli"); ok = False

# 2) a nearly closed loop -> closedLoop True, isoperimetric penalty > 0 ------
theta = np.linspace(0, 2 * np.pi * 0.97, 40)
loop = np.column_stack((14.0 + 1.2 * np.cos(theta), 43.0 + 1.2 * np.sin(theta)))
gl = fg.geometry_metrics(loop)
print("2) anello quasi chiuso:", {k: gl[k] for k in
      ("closedLoop", "endpointGapKm", "isoperimetricPenalty", "enclosedAreaKm2")})
if not (gl["closedLoop"] and gl["isoperimetricPenalty"] > 0.4
        and gl["enclosedAreaKm2"] > 1000):
    print("  FAIL: un anello quasi chiuso deve essere riconosciuto"); ok = False

# 3) a self-crossing figure-eight -> selfIntersections >= 1 ------------------
t = np.linspace(0, 2 * np.pi, 60)
eight = np.column_stack((14.0 + 1.5 * np.sin(t), 43.0 + 1.2 * np.sin(t) * np.cos(t)))
ge = fg.geometry_metrics(eight)
print("3) figura a otto: selfIntersections =", ge["selfIntersections"])
if ge["selfIntersections"] < 1:
    print("  FAIL: un'auto-intersezione deve essere rilevata"); ok = False

# 4) a jagged zig-zag -> high curvature penalty and many reversals ----------
xs = np.linspace(8.0, 18.0, 21)
ys = 42.0 + 0.6 * (np.arange(21) % 2)
zig = np.column_stack((xs, ys))
gz = fg.geometry_metrics(zig)
print("4) zig-zag: curvaturePenalty =", gz["curvaturePenalty"],
      "reversals =", gz["tangentReversals"])
if not (gz["curvaturePenalty"] > 0.5 and gz["tangentReversals"] >= 5):
    print("  FAIL: uno zig-zag deve avere penalita' di curvatura alta"); ok = False

# 5) degenerate inputs are safe ---------------------------------------------
for bad in (np.zeros((0, 2)), np.array([[8.0, 42.0]]), np.array([[8.0, 42.0], [8.0, 42.0]])):
    g = fg.geometry_metrics(bad)
    if g["closedLoop"] or g["curvaturePenalty"] != 0.0:
        print("  FAIL: input degenere non gestito"); ok = False

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
