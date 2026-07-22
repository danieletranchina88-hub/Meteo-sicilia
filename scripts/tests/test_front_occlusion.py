"""Synthetic verification of occlusion detection (front_occlusion.py).

A) an idealised occluding wave (a real MSLP low with a cold and a warm front
   meeting at a triple point displaced from the low, the cold front wrapping
   back to the centre) -> the wrapped branch is flagged as occluded;
B) an isolated cold front with no low and no warm front -> nothing occluded;
C) an open wave (cold + warm fronts both starting AT the low, no wrapped
   branch) -> not occluded.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import front_occlusion as focc

LON = np.arange(6.0, 20.01, 0.2)
LAT = np.arange(38.0, 48.01, 0.2)
LONG, LATG = np.meshgrid(LON, LAT)

ok = True


def low_field(lon0, lat0, depth=8.0, radius_deg=3.0):
    """A gaussian MSLP low centred at (lon0, lat0)."""
    r2 = ((LONG - lon0) / radius_deg) ** 2 + ((LATG - lat0) / radius_deg) ** 2
    return 1015.0 - depth * np.exp(-r2)


def polyline(points, n=40):
    points = np.asarray(points, dtype=float)
    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
    t = np.linspace(0, seg[-1], n)
    return np.column_stack((np.interp(t, seg, points[:, 0]),
                            np.interp(t, seg, points[:, 1])))


# A) occluding wave --------------------------------------------------------
low_lon, low_lat = 12.0, 44.0
pressure = low_field(low_lon, low_lat)
# cold front: SW far end -> triple point (12.5, 43) -> wraps to the low
cold = polyline([(8.5, 40.5), (10.5, 42.0), (12.5, 43.0),
                 (12.3, 43.5), (12.0, 44.0)])
# warm front: triple point -> east
warm = polyline([(12.5, 43.0), (14.5, 43.2), (17.0, 43.4)])
features = [{"coordinates": cold, "frontType": "cold"},
            {"coordinates": warm, "frontType": "warm"}]
occ = focc.detect_occlusion(features, pressure, LON, LAT)
print(f"A) onda in occlusione: {len(occ)} occlusioni")
if occ:
    o = occ[0]
    print(f"   wrapKm={o['wrapKm']} triple={o['triplePoint']} "
          f"low={o['low']['pressure']:.0f}hPa junction={o['junctionKm']}km")
if not occ:
    print("   FAIL: l'onda in occlusione non e' riconosciuta"); ok = False

# B) isolated cold front, flat pressure ------------------------------------
flat = np.full_like(pressure, 1013.0)
occ_b = focc.detect_occlusion(
    [{"coordinates": cold, "frontType": "cold"}], flat, LON, LAT)
print(f"B) fronte freddo isolato: {len(occ_b)} occlusioni (atteso 0)")
if occ_b:
    print("   FAIL: falsa occlusione senza minimo/fronte caldo"); ok = False

# C) open wave: both fronts start AT the low, no wrapped branch ------------
cold_open = polyline([(8.5, 40.5), (10.2, 42.2), (12.0, 44.0)])   # ends at low
warm_open = polyline([(12.0, 44.0), (14.5, 44.0), (17.0, 44.0)])  # starts at low
occ_c = focc.detect_occlusion(
    [{"coordinates": cold_open, "frontType": "cold"},
     {"coordinates": warm_open, "frontType": "warm"}],
    low_field(low_lon, low_lat), LON, LAT)
print(f"C) onda aperta (non occlusa): {len(occ_c)} occlusioni (atteso 0)")
if occ_c:
    print("   FAIL: onda aperta scambiata per occlusione"); ok = False

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
