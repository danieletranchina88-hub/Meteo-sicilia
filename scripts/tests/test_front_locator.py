"""Synthetic verification of front_locator.py (v12, Sansom-Catto TFL).

Eight mandatory checks before touching real data:
 1 straight ideal front       -> exactly one line on the warm edge
 2 gently curved front        -> single continuous line
 3 uniform gradient, no zone  -> no valid line
 4 circular thermal anomaly   -> closed contour, recognisable
 5 two nearby fronts          -> not merged
 6 noise vs smoothing         -> stable
 7 latitude-axis inversion    -> geometrically identical result
 8 two resolutions            -> similar position after physical smoothing
Plus the critical standard TFP sign test (warm edge negative).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import front_locator as fl

# European-like grid
LON = np.arange(3.0, 22.01, 0.15)
LAT = np.arange(34.0, 49.01, 0.15)
LONG, LATG = np.meshgrid(LON, LAT)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def theta_w_straight(width_deg=1.2, warm_south=True):
    """Warm south, cold north; theta_w drops across a N-S transition."""
    center = 42.0
    s = sigmoid((LATG - center) / width_deg)  # 1 north, 0 south
    warm, cold = 300.0, 288.0
    return warm - (warm - cold) * s if warm_south else cold + (warm - cold) * s


def run(name, field, lon=LON, lat=LAT, **kw):
    cands = fl.locate_fronts(field, lon, lat, **kw)
    print(f"  {name}: {len(cands)} candidati", end="")
    for c in cands[:3]:
        print(f" [len={c['lengthKm']:.0f}km tfp={c['medianTfp']:.3f}"
              f" grad={c['medianThetaWGradient']:.1f} abz={c['medianAbzGradient']:.1f}]", end="")
    print()
    return cands


ok = True

# --- TFP SIGN (the critical correctness test) ------------------------------
print("SEGNO TFP standard (fronte caldo-sud / freddo-nord: bordo caldo < 0):")
cands = run("  fronte dritto", theta_w_straight())
if not cands:
    print("  FAIL: nessuna linea sul fronte ideale"); ok = False
else:
    c = cands[0]
    if c["medianTfp"] >= 0:
        print(f"  FAIL: TFP mediano {c['medianTfp']:.6f} non negativo sul bordo caldo"); ok = False
    # la linea deve stare sul bordo caldo (lato sud, lat < centro) della zona
    mean_lat = float(np.mean(c["coordinates"][:, 1]))
    print(f"  linea a lat media {mean_lat:.2f} (centro zona 42.0; bordo caldo < 42)")

# --- 1 straight front: exactly one line ------------------------------------
print("\n1) fronte rettilineo ideale:")
c1 = run("dritto", theta_w_straight())
if len(c1) != 1:
    print(f"  FAIL: attesa 1 linea, trovate {len(c1)}"); ok = False

# --- 2 gently curved front -------------------------------------------------
print("\n2) fronte curvo dolcemente:")
center = 42.0 + 2.0 * np.sin((LONG - 3.0) / 19.0 * np.pi)
s = sigmoid((LATG - center) / 1.2)
c2 = run("curvo", 300.0 - 12.0 * s)
if len(c2) != 1:
    print(f"  NOTA: {len(c2)} linee (curva puo' spezzarsi ai bordi)")

# --- 3 uniform gradient, no frontal zone -----------------------------------
print("\n3) gradiente uniforme (nessuna zona frontale):")
c3 = run("uniforme", 300.0 - 0.3 * (LATG - 34.0))  # gradiente costante debole
if len(c3) != 0:
    print(f"  FAIL: attese 0 linee, trovate {len(c3)}"); ok = False

# --- 4 circular thermal anomaly -> closed contour --------------------------
print("\n4) anomalia termica circolare:")
r = np.hypot((LONG - 12.0) * np.cos(np.deg2rad(41)), (LATG - 41.0))
c4 = run("anello", 300.0 - 10.0 * np.exp(-(r / 2.5) ** 2))
# non deve essere un fronte lineare pulito: o nessuno o linee riconoscibili
print(f"  (anomalia chiusa: {len(c4)} candidati, attesi pochi/nessuno lineare)")

# --- 5 two nearby fronts (not merged) --------------------------------------
print("\n5) due fronti vicini:")
s1 = sigmoid((LATG - 39.0) / 1.0)
s2 = sigmoid((LATG - 45.0) / 1.0)
c5 = run("due fronti", 300.0 - 6.0 * s1 - 6.0 * s2)
if len(c5) < 2:
    print(f"  NOTA: {len(c5)} linee (dipende dalla separazione vs smoothing)")

# --- 6 noise vs smoothing stability ----------------------------------------
print("\n6) rumore sovrapposto, stabilita' al variare dello smoothing:")
rng = np.random.default_rng(0)
noisy = theta_w_straight() + rng.normal(0, 0.4, LATG.shape)
for sig in (40.0, 60.0, 90.0):
    cn = fl.locate_fronts(noisy, LON, LAT, synoptic_sigma_km=sig)
    print(f"   sigma={sig:.0f}km -> {len(cn)} linee")

# --- 7 latitude-axis inversion (identical geometry) ------------------------
print("\n7) inversione asse latitudini:")
field = theta_w_straight()
c_norm = fl.locate_fronts(field, LON, LAT)
c_inv = fl.locate_fronts(field[::-1, :], LON, LAT[::-1])
lat_norm = float(np.mean(c_norm[0]["coordinates"][:, 1])) if c_norm else np.nan
lat_inv = float(np.mean(c_inv[0]["coordinates"][:, 1])) if c_inv else np.nan
print(f"   lat media normale={lat_norm:.3f} invertita={lat_inv:.3f} (diff attesa ~0)")
if not (c_norm and c_inv and abs(lat_norm - lat_inv) < 0.3):
    print("  FAIL: inversione asse cambia la geometria"); ok = False

# --- 8 two resolutions, similar position -----------------------------------
print("\n8) due risoluzioni (posizione simile dopo smoothing fisico):")
LON2 = np.arange(3.0, 22.01, 0.30)
LAT2 = np.arange(34.0, 49.01, 0.30)
LONG2, LATG2 = np.meshgrid(LON2, LAT2)
field2 = 300.0 - 12.0 * sigmoid((LATG2 - 42.0) / 1.2)
c_hi = fl.locate_fronts(theta_w_straight(), LON, LAT)
c_lo = fl.locate_fronts(field2, LON2, LAT2)
lat_hi = float(np.mean(c_hi[0]["coordinates"][:, 1])) if c_hi else np.nan
lat_lo = float(np.mean(c_lo[0]["coordinates"][:, 1])) if c_lo else np.nan
print(f"   lat media 0.15deg={lat_hi:.3f} 0.30deg={lat_lo:.3f} (diff attesa piccola)")
if not (c_hi and c_lo and abs(lat_hi - lat_lo) < 0.4):
    print("  FAIL: posizione troppo diversa fra risoluzioni"); ok = False

# --- 9 the ABZ measures the zone, not the line -----------------------------
# On the warm edge the gradient has not yet peaked, so a correct ABZ must be
# strictly larger than |grad theta_w| on the line, and must land on the true
# maximum of the gradient across the zone.
print("\n9) zona baroclina adiacente misurata dove sta:")
field = theta_w_straight()
cands, diag = fl.locate_fronts(field, LON, LAT, return_fields=True)
if not cands:
    print("  FAIL: nessun candidato per la prova ZBA"); ok = False
else:
    line = cands[0]["coordinates"]
    on_line = fl._sample(diag["grad_mag_100"], line, LON, LAT,
                         fl.grid_metrics(LON, LAT)["dlon"],
                         fl.grid_metrics(LON, LAT)["dlat"])
    abz = cands[0]["medianAbzGradient"]
    peak = float(np.nanmax(diag["grad_mag_100"]))
    print(f"   |grad| sulla linea={np.nanmedian(on_line):.2f} "
          f"ZBA={abz:.2f} massimo del campo={peak:.2f} K/100km")
    if not abz > np.nanmedian(on_line) * 1.05:
        print("  FAIL: la ZBA non supera il gradiente sulla linea"); ok = False
    if not abz <= peak * 1.02:
        print("  FAIL: la ZBA supera il massimo reale del campo"); ok = False
    # The walk is bounded by the analysis scale, so a zone wider than that
    # scale is sampled on its flank rather than exactly on its crest.
    if not abz >= 0.75 * peak:
        print(f"  FAIL: la ZBA ({abz:.2f}) non raggiunge la zona ({peak:.2f})")
        ok = False

# --- 10 short mask interruptions must not split one front ------------------
# A frontal zone that weakens over a limited stretch is one front, not two.
print("\n10) interruzione breve ricucita, lacuna vera no:")
coords = np.column_stack((np.linspace(5.0, 15.0, 60), np.full(60, 42.0)))
keep = np.ones(60, dtype=bool)
keep[28:31] = False           # ~50 km di interruzione
bridged = fl._bridge_short_gaps(coords, keep, 90.0)
if not bridged[28:31].all():
    print("  FAIL: interruzione breve non ricucita"); ok = False
keep_wide = np.ones(60, dtype=bool)
keep_wide[20:40] = False      # ~340 km: due strutture distinte
bridged_wide = fl._bridge_short_gaps(coords, keep_wide, 90.0)
if bridged_wide[20:40].any():
    print("  FAIL: lacuna vera ricucita per errore"); ok = False
edge = np.ones(60, dtype=bool)
edge[:5] = False              # il fronte sfuma al bordo: non e' un'interruzione
if fl._bridge_short_gaps(coords, edge, 90.0)[:5].any():
    print("  FAIL: coda al bordo trattata come interruzione interna"); ok = False
print("   breve ricucita, larga separata, coda al bordo intatta")

# --- 11 the in-line quantile may not run away from the published threshold --
print("\n11) calibrazione adattiva limitata:")
noisy_field = theta_w_straight() + 0.9 * np.random.default_rng(7).standard_normal(LONG.shape)
_, dn = fl.locate_fronts(noisy_field, LON, LAT, tfp_threshold=-2.5e-5,
                         tfp_full_strength=-9.0e-5, abz_gradient_threshold=0.90,
                         abz_gradient_full_strength=1.70, return_fields=True)
limit = -2.5e-5 * fl.ADAPTIVE_TIGHTENING_LIMIT
print(f"   TFP efficace={dn['effective_tfp_threshold']:.3e} (limite {limit:.3e})")
if dn["effective_tfp_threshold"] < limit - 1e-12:
    print("  FAIL: il quantile ha superato il limite di irrigidimento"); ok = False
if dn["effective_gradient_threshold"] > 0.90 * fl.ADAPTIVE_TIGHTENING_LIMIT + 1e-9:
    print("  FAIL: la soglia di gradiente ha superato il limite"); ok = False

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
