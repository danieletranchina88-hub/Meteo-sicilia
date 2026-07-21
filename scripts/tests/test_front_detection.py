"""Synthetic verification of the two-scale detection (front_detection.py).

Checks the structural prior: a refined candidate that coincides with a
synoptic-scale front is kept; an isolated mesoscale gradient far from any
synoptic structure is removed (unless exceptionally strong, in which case
kept but flagged not corroborated).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import front_detection as fd

LON = np.arange(3.0, 22.01, 0.15)
LAT = np.arange(34.0, 49.01, 0.15)
LONG, LATG = np.meshgrid(LON, LAT)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


ok = True

# --- 1) synoptic front present -> kept, well supported ---------------------
synoptic_front = 300.0 - 12.0 * sigmoid((LATG - 42.0) / 1.2)
final = fd.detect_fronts_two_scale(synoptic_front, LON, LAT)
print(f"1) fronte sinottico solo: {len(final)} finali, "
      f"support={[c['synopticSupport'] for c in final]}")
if not (len(final) >= 1 and all(c["corroborated"] for c in final)):
    print("  FAIL: il fronte sinottico non e' tenuto/corroborato"); ok = False

# --- 2) synoptic front + short isolated sub-synoptic front far away --------
# a short E-W front near (19, 37), localised in longitude so it is well
# under the synoptic minimum length: detected by the refined scale but not
# a synoptic structure -> must not be corroborated.
big_front = 300.0 - 12.0 * sigmoid((LATG - 46.0) / 1.2)
short_step = -6.0 * sigmoid((LATG - 37.0) / 0.7)
window = np.exp(-((LONG - 19.0) / 1.4) ** 2)
field2 = big_front + short_step * window
final2, syn = fd.detect_fronts_two_scale(field2, LON, LAT, return_synoptic=True)
kept_near_patch = [
    c for c in final2
    if abs(np.mean(c["coordinates"][:, 0]) - 19.0) < 2.0
    and abs(np.mean(c["coordinates"][:, 1]) - 37.0) < 2.0
]
print(f"2) sinottico + patch mesoscalare isolato: {len(syn)} corridoi, "
      f"{len(final2)} finali; candidati vicino al patch corroborati="
      f"{[c['corroborated'] for c in kept_near_patch]}")
# il patch isolato non deve produrre un fronte corroborato
if any(c.get("corroborated") for c in kept_near_patch):
    print("  FAIL: il patch isolato e' stato corroborato come sinottico"); ok = False
else:
    print("  OK: il patch isolato non e' corroborato (rimosso o flaggato)")

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
