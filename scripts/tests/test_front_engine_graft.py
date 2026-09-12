"""End-to-end check of the engine grafted into the v12 hour detector.

The unit tests in ``test_front_engine.py`` prove the engine is right on its
own.  This one proves the graft: that ``_detect_hour`` still reads the fields
it needs before it needs them, that the candidates it produces survive the
cross-front air-mass gates unchanged, and that the two cases which motivated
the rewrite come out the right way round -- a genuine baroclinic boundary with
real dynamics is published with a smooth axis, and a thermal contrast welded
to a mountain slope with no dynamics is not published at all.

The analyzer is built with ``object.__new__`` and stubbed fields, the pattern
already used by ``test_front_analysis_status.py``: no GRIB, no network, and
the real detection path runs.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import front_analysis_v12 as v12
import front_engine as fe
import front_locator as fl

erf = np.vectorize(math.erf)

LON = np.arange(6.0, 19.001, 0.08)
LAT = np.arange(36.0, 47.001, 0.08)
LONG, LATG = np.meshgrid(LON, LAT)
LON0, LAT0 = float(np.mean(LON)), float(np.mean(LAT))
EAST_KM = (LONG - LON0) * fl.EARTH_KM_PER_DEG * np.cos(np.deg2rad(LATG))
NORTH_KM = (LATG - LAT0) * fl.EARTH_KM_PER_DEG
# Una catena montuosa zonale a nord, come il bordo alpino visto a scala
# sinottica: e' l'oggetto che oggi viene pubblicato come fronte stazionario.
TERRAIN = 1600.0 * np.exp(-((LATG - 46.2) / 0.9) ** 2)

ok = True


def report(passed, message, detail=""):
    global ok
    if not passed:
        ok = False
        print("  FAIL: " + message + (("  " + detail) if detail else ""))
    elif detail:
        print("    " + detail)


def build_analyzer(fields, thermo_by_level):
    analyzer = object.__new__(v12.IconSynopticFrontAnalyzer)
    analyzer.available_hours = [0]
    analyzer.hour_to_index = {0: 0}
    analyzer.datasets = {name: object() for name in fields}
    analyzer.longitudes = LON
    analyzer.latitudes = LAT
    analyzer.terrain = TERRAIN
    analyzer.delta_longitude = float(LON[1] - LON[0])
    analyzer.delta_latitude = float(LAT[1] - LAT[0])
    analyzer.dy_km = analyzer.delta_latitude * 111.32
    analyzer.dx_km = (analyzer.delta_longitude * 111.32
                      * np.cos(np.deg2rad(LAT))[:, None])
    analyzer.tendency_window_hours = 3
    analyzer._threshold_climatology = None
    analyzer._rejected = {}
    analyzer._detection_errors = {}
    analyzer._pipeline_diag = {}
    analyzer.ml_guidance = None
    analyzer._thermodynamic_cache = {}
    analyzer._pressure_cache = {}
    analyzer._surface_pressure_cache = {}
    analyzer._height_500_cache = {}
    analyzer._field = lambda name, hour, _f=fields: _f[name]
    analyzer._thermodynamics = (
        lambda hour, level_hpa=850, _t=thermo_by_level: _t[level_hpa]
    )
    return analyzer


def air_mass(theta_w):
    """Thermodynamic triple consistent enough for the cross-front gates."""
    return {
        "theta_w": theta_w,
        # A 850 hPa theta_w contrast of this size comes with a dry-theta and
        # theta_e contrast of the same sign; the gates test all three.
        "theta": theta_w + 12.0,
        "theta_e": theta_w * 1.0 + 18.0,
    }


# --------------------------------------------------------------------------
print("1) Un fronte vero: deve uscire, e uscire liscio")
# --------------------------------------------------------------------------
ACROSS = 0.6 * EAST_KM + 0.8 * NORTH_KM
front_theta_w = 286.0 + 3.0 * erf(ACROSS / (np.sqrt(2.0) * 80.0))
# Confluenza attraverso il fronte -- deformazione piu' convergenza, quindi
# frontogenetica -- e un getto parallelo al fronte, che e' la parte del vento
# che un fronte vero ha sempre e un contrasto orografico no: da' il taglio
# ciclonico e il salto di direzione attraverso la linea.
front_shape = erf(ACROSS / (np.sqrt(2.0) * 80.0))
front_u = 9.0 - 1.0e-5 * (0.6 * ACROSS) * 1000.0 - 14.0 * 0.8 * front_shape
front_v = 2.0 - 1.0e-5 * (0.8 * ACROSS) * 1000.0 + 14.0 * 0.6 * front_shape
front_fields = {
    "t": 276.0 + 3.0 * erf(ACROSS / (np.sqrt(2.0) * 80.0)),
    "q": 0.0045 + 0.0012 * erf(ACROSS / (np.sqrt(2.0) * 80.0)),
    "u": front_u,
    "v": front_v,
    "t700": 268.0 + 2.2 * erf(ACROSS / (np.sqrt(2.0) * 80.0)),
    "q700": np.full_like(LONG, 0.0020),
    "p": 1014.0 - 7.0 * np.exp(-(ACROSS / 260.0) ** 2),
}
front_thermo = {
    850: air_mass(front_theta_w),
    700: air_mass(284.0 + 2.4 * erf(ACROSS / (np.sqrt(2.0) * 80.0))),
}
analyzer = build_analyzer(front_fields, front_thermo)
detected = analyzer._detect_hour(0)
report(bool(detected), "un fronte baroclino con dinamica reale non viene rilevato",
       "candidati accettati: %d" % len(detected))
if detected:
    best = max(detected, key=lambda item: item.get("lengthKm", 0.0))
    line = np.asarray(best["coordinates"], dtype=float)
    turn = fe.mean_turn_deg_per_km(line)
    # Metro del progetto (docs/algoritmo_fronti.md): il prodotto in linea
    # prima di questa riscrittura stava a 12,5 gradi ogni 20 km.
    report(turn < 6.0, "la geometria pubblicata serpeggia ancora",
           "L=%.0f km, curva %.2f gr/20km, vertici %d, confidenza %.3f"
           % (best["lengthKm"], turn, len(line),
              float(best.get("locatorConfidence", 0.0))))
    report(np.asarray(best["warmNormal"]).shape == line.shape
           and np.asarray(best["hewsonDir"]).shape == line.shape,
           "warmNormal/hewsonDir non hanno la lunghezza dei vertici finali",
           "forme: coordinate %s, normale %s, hewson %s"
           % (line.shape, np.asarray(best["warmNormal"]).shape,
              np.asarray(best["hewsonDir"]).shape))
    for key in ("synopticSupport", "sinuosity", "locatorConfidence",
                "medianTfpStrength", "medianAbzGradient"):
        report(key in best and np.isfinite(float(best[key])),
               "manca la chiave di contratto %s (a valle vale 0 e fa fallire "
               "ogni cancello)" % key)
    report(float(best["lengthKm"]) >= v12.ENGINE_MIN_LENGTH_KM,
           "pubblicata una linea sotto la lunghezza minima dichiarata",
           "lunghezza minima dichiarata %.0f km" % v12.ENGINE_MIN_LENGTH_KM)

# --------------------------------------------------------------------------
print("\n2) Il bordo montuoso: stesso contrasto termico, nessuna dinamica")
# --------------------------------------------------------------------------
SLOPE = (LATG - 46.2) * fl.EARTH_KM_PER_DEG
locked_theta_w = 286.0 - 3.0 * erf(SLOPE / (np.sqrt(2.0) * 80.0))
locked_fields = {
    "t": 276.0 - 3.0 * erf(SLOPE / (np.sqrt(2.0) * 80.0)),
    "q": 0.0045 - 0.0012 * erf(SLOPE / (np.sqrt(2.0) * 80.0)),
    # Vento debole e uniforme: niente deformazione, niente convergenza,
    # niente movimento.  E' il caso della linea alpina di 1730 km.
    "u": np.full_like(LONG, 3.0),
    "v": np.full_like(LONG, 1.0),
    "t700": np.full_like(LONG, 268.0),
    "q700": np.full_like(LONG, 0.0020),
    "p": np.full_like(LONG, 1016.0),
}
locked_thermo = {850: air_mass(locked_theta_w), 700: air_mass(np.full_like(LONG, 284.0))}
locked_analyzer = build_analyzer(locked_fields, locked_thermo)
locked_detected = locked_analyzer._detect_hour(0)
report(not locked_detected,
       "il contrasto termico orografico viene ancora pubblicato",
       "candidati accettati: %d%s" % (
           len(locked_detected),
           (" (il piu' lungo %.0f km)" % max(
               item.get("lengthKm", 0.0) for item in locked_detected))
           if locked_detected else ""))

# Il ripiego deve essersi acceso davvero, e deve essere stato respinto
# dall'evidenza: se non si accendesse, questo test passerebbe per il motivo
# sbagliato e non direbbe niente sul ripiego.
diagnostics = locked_analyzer._pipeline_diag.get(0, {})
funnel = diagnostics.get("engineFunnel") or {}
report(diagnostics.get("candidateSource") == "two-scale-fallback"
       and funnel.get("fallbackOffered", 0) > 0
       and funnel.get("fallbackAccepted", 1) == 0,
       "il ripiego non si e' acceso, oppure ha pubblicato senza il vaglio "
       "dell'evidenza",
       "sorgente %s: il rilevatore a due scale ha offerto %s linee, "
       "l'evidenza ne ha accettate %s"
       % (diagnostics.get("candidateSource"),
          funnel.get("fallbackOffered"), funnel.get("fallbackAccepted")))

# E sul fronte vero il ripiego non deve essere servito affatto.
front_diagnostics = analyzer._pipeline_diag.get(0, {})
report(front_diagnostics.get("candidateSource") == "engine",
       "il fronte vero e' stato trovato dal ripiego invece che dal motore",
       "sorgente sul fronte vero: %s, candidati dal motore: %s"
       % (front_diagnostics.get("candidateSource"),
          front_diagnostics.get("engineCandidates")))

# --------------------------------------------------------------------------
print("\n3) Il cancello di supporto non deve mangiarsi l'Italia")
# --------------------------------------------------------------------------
# La geometria e' vietata dove il nucleo di lisciamento pesca troppo fuori
# dominio, perche' li' le derivate sono sbilanciate.  A 150 km quel margine
# vale circa 200 km, che e' quasi esattamente il cuscinetto fra il bordo di
# ICON-2I e la costa: la verifica serve perche' un domani chi alzasse il sigma
# o la soglia spingerebbe fuori le Alpi senza accorgersene.
ICON_LON = np.arange(3.0, 22.001, 0.04)
ICON_LAT = np.arange(33.7, 48.901, 0.04)
ICON_GRID = fl.grid_metrics(ICON_LON, ICON_LAT)
support = fe.smoothing_support(
    np.ones((len(ICON_LAT), len(ICON_LON))), fe.SYNOPTIC_SIGMA_KM, ICON_GRID
)
outside = []
for name, longitude, latitude in (
    ("Lampedusa", 12.60, 35.50), ("Palermo", 13.36, 38.12),
    ("Cagliari", 9.12, 39.22), ("Roma", 12.50, 41.90),
    ("Milano", 9.19, 45.46), ("Aosta", 7.32, 45.74),
    ("Trieste", 13.77, 45.65), ("Bolzano", 11.35, 46.50),
    ("Tarvisio", 13.58, 46.50),
):
    row = int(np.argmin(np.abs(ICON_LAT - latitude)))
    column = int(np.argmin(np.abs(ICON_LON - longitude)))
    if support[row, column] < 0.90:
        outside.append("%s %.3f" % (name, support[row, column]))
report(not outside, "il cancello di supporto esclude parti d'Italia",
       "supporto minimo sulle nove localita' controllate: %.3f"
       % min(
           support[int(np.argmin(np.abs(ICON_LAT - la))),
                   int(np.argmin(np.abs(ICON_LON - lo)))]
           for lo, la in ((12.60, 35.50), (13.36, 38.12), (9.12, 39.22),
                          (12.50, 41.90), (9.19, 45.46), (7.32, 45.74),
                          (13.77, 45.65), (11.35, 46.50), (13.58, 46.50))
       ))

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
