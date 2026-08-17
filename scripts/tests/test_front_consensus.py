"""Synthetic checks for independent frontal consensus diagnostics."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import front_consensus as fc

lon = np.linspace(5.0, 20.0, 121)
lat = np.linspace(35.0, 48.0, 105)
lon2d, lat2d = np.meshgrid(lon, lat)
ok = True

# Cyclonic shear and a genuine temperature transition should exceed F=1;
# uniform flow must not manufacture Parfitt support.
temperature = 286.0 + 7.0 / (1.0 + np.exp((lat2d - 42.0) / 0.45))
u = 12.0 + (lat2d - 42.0) * 0.001
v = 5.0 * np.tanh((lon2d - 12.0) / 3.0)
strong = fc.parfitt_f_field(temperature, u, v, lon, lat)
uniform = fc.parfitt_f_field(
    temperature, np.full_like(u, 10.0), np.full_like(v, 2.0), lon, lat
)
strong_peak = float(np.nanpercentile(strong["parfittF"], 99))
uniform_peak = float(np.nanmax(np.abs(uniform["parfittF"])))
print(f"1) Parfitt-F ciclonico={strong_peak:.2f}, uniforme={uniform_peak:.3g}")
if strong_peak <= 1.0 or uniform_peak > 1.0e-6:
    print("  FAIL: diagnostica Parfitt-F non selettiva")
    ok = False

# Classic SW->NW passage in component space: u remains eastward, v changes
# northward to southward. A steady wind is a strict null case.
u0 = np.full_like(temperature, 6.0)
v0 = np.full_like(temperature, 4.0)
u1 = np.full_like(temperature, 6.0)
v1 = np.full_like(temperature, -4.0)
wnd = fc.temporal_wind_shift_field(
    u0, v0, u1, v1, lon, lat, elapsed_hours=6
)
steady = fc.temporal_wind_shift_field(
    u0, v0, u0, v0, lon, lat, elapsed_hours=3
)
print(
    "2) WND passaggio freddo=%.2f, vento costante=%.2f"
    % (np.nanmedian(wnd["wndColdSupport"]),
       np.nanmedian(steady["wndGeneralSupport"]))
)
if np.nanmedian(wnd["wndColdSupport"]) < 0.8:
    print("  FAIL: passaggio SW->NW non riconosciuto")
    ok = False
if np.nanmax(steady["wndGeneralSupport"]) > 1.0e-8:
    print("  FAIL: vento invariato produce supporto temporale")
    ok = False

# Geometry uncertainty must reflect the real parallel-line separation.
line_a = np.column_stack((np.linspace(7.0, 18.0, 40), np.full(40, 42.0)))
line_b = line_a.copy()
line_b[:, 1] += 0.45
distance = fc.symmetric_line_distance_km(line_a, line_b)
print(f"3) distanza simmetrica={distance:.1f} km")
if not 45.0 <= distance <= 55.0:
    print("  FAIL: incertezza geometrica non metrica")
    ok = False

summary = fc.line_consensus_metrics(
    line_a, lon, lat, parfitt=strong, wnd=wnd,
    locator_sources=["thetaW-laplacian", "thetaW-directional"],
)
print("4) consenso:", summary)
if summary["methodAvailability"] != 3 or summary["locatorAgreementCount"] != 2:
    print("  FAIL: conteggio dei metodi errato")
    ok = False

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
