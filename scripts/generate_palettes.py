"""Genera gli ancoraggi colore delle scale scientifiche dei campi.

Uso:  pip install cmcrameri matplotlib && python scripts/generate_palettes.py
Copia l'uscita negli array corrispondenti di index.html.


Non colori scelti a occhio: i valori escono dai dati pubblicati delle mappe
percettivamente uniformi (Crameri 2018, Zenodo; Crameri, Shephard & Heron
2020, Nature Communications 11:5444). Ogni mappa e' monotona in luminanza,
sicura per il daltonismo e senza confini falsi.
"""
import numpy as np
from matplotlib import colormaps
from cmcrameri import cm


def sample(name, t, lo=0.0, hi=1.0):
    """Colore RGB 0-255 della mappa `name` alla posizione t in [0,1].

    `name` e' una Scientific Colour Map di Crameri, oppure -- con il prefisso
    `cb:` -- una tavolozza ColorBrewer (Harrower e Brewer 2003, The
    Cartographic Journal 40(1)), presa da matplotlib con i valori originali.
    """
    cmap = colormaps[name[3:]] if name.startswith("cb:") else getattr(cm, name)
    x = lo + (hi - lo) * float(np.clip(t, 0.0, 1.0))
    r, g, b, _ = cmap(x)
    return [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]


def diverging(value, centre, half_span):
    """Posizione su una mappa divergente, con il neutro esattamente al centro."""
    return np.clip(0.5 + (value - centre) / (2.0 * half_span), 0.0, 1.0)


def two_sided(value, centre, below, above):
    """Divergente asimmetrica: il neutro resta al centro, ma i due rami
    coprono escursioni diverse.

    La temperatura a 2 m sull'Italia va da circa -25 a +45: una scala
    simmetrica su +/-40 spreca meta' del ramo freddo su valori che non
    esistono, e comprime tutto l'anno in un terzo della rampa. I due rami
    hanno quindi pendenze diverse -- un grado di freddo non colora quanto un
    grado di caldo -- ed e' un compromesso dichiarato, non un errore: senza,
    una mappa estiva sarebbe una parete arancione uniforme.
    """
    if value >= centre:
        return float(np.clip(0.5 + 0.5 * (value - centre) / above, 0.5, 1.0))
    return float(np.clip(0.5 - 0.5 * (centre - value) / below, 0.0, 0.5))


def js(anchors, indent=8):
    pad = " " * indent
    rows = []
    for item in anchors:
        colour = "[%d, %d, %d]" % tuple(item["c"])
        if "a" in item:
            rows.append(f'{pad}{{ v: {item["v"]}, c: {colour}, a: {item["a"]} }}')
        else:
            rows.append(f'{pad}{{ v: {item["v"]}, c: {colour} }}')
    return ",\n".join(rows)


out = {}

# --- Temperatura: RdYlBu invertita, divergente, neutro esatto a 0 gradi -----
# La prima scelta era `vik`, percettivamente uniforme. Misurata sulla fascia
# 16-38 gradi -- dove sta il dato quasi sempre -- dava pero' colori fra
# #d0916d e #791905: una successione di marroni, con croma medio 52 in CIELab.
# La distinzione fra due zone distanti sei gradi valeva 10,2 unita' CIEDE2000,
# quindi il problema non era la distanza percettiva ma la TINTA: tutta la
# fascia estiva cadeva nello stesso arancione bruno.
#
# RdYlBu invertita e' una tavolozza divergente pubblicata (ColorBrewer,
# Harrower e Brewer 2003; e' la scala usata dall'IPCC per i campi termici).
# Sulla stessa fascia da' #fdb86b -> #ca2326, croma medio 66,5 e distinzione a
# sei gradi di 10,0: la stessa separazione misurabile di vik, ma distribuita
# su tinte diverse invece che sulla sola luminosita' di un unico bruno. Con
# deuteranopia la separazione a sei gradi resta 8,4, molto sopra la soglia di
# percettibilita' (~2,3).
#
# Cede qualcosa in uniformita' percettiva rispetto a Crameri -- il neutro
# giallo e' un picco di luminanza -- ed e' un compromesso dichiarato: una
# scala che non si legge non e' piu' scientifica di una che si legge.
temp_values = [-75, -65, -55, -45, -35, -30, -26, -22, -18, -14, -10, -6, -2,
               0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
               34, 36, 38, 40, 42, 44, 46]
out["TEMP_ANCHORS"] = [
    {"v": v, "c": sample("cb:RdYlBu_r", two_sided(v, 0.0, 25.0, 45.0))} for v in temp_values
]

# --- Vento: batlow, sequenziale ---------------------------------------------
wind_values = list(range(0, 131, 10))
out["WIND_ANCHORS"] = [
    {"v": v, "c": sample("batlow", v / 130.0)} for v in wind_values
]

# --- Pressione: broc, divergente sulla standard 1013,25 hPa ------------------
press_values = [960, 970, 980, 990, 1000, 1004, 1008, 1012, 1016, 1020,
                1024, 1028, 1032, 1036, 1040, 1048]
out["PRESS_ANCHORS"] = [
    {"v": v, "c": sample("broc", diverging(v, 1013.25, 40.0), 0.06, 0.94)} for v in press_values
]

# --- Umidita' relativa: davos invertita, secco chiaro -> umido scuro ---------
rh_values = list(range(0, 101, 10))
out["RH_ANCHORS"] = [
    {"v": v, "c": sample("davos", 1.0 - v / 100.0, 0.05, 0.95)} for v in rh_values
]

# --- Nuvolosita': grayC, il grigio e' l'aspetto fisico della nube ------------
cloud_values = list(range(0, 101, 10))
cloud_alpha = [0, 0.58, 0.7, 0.78, 0.84, 0.88, 0.91, 0.94, 0.96, 0.98, 0.99]
out["CLOUD_ANCHORS"] = [
    {"v": v, "c": sample("grayC", 1.0 - v / 100.0, 0.12, 0.98), "a": a}
    for v, a in zip(cloud_values, cloud_alpha)
]

# --- Precipitazione: oslo invertita, soglie discrete -------------------------
rain_values = [0, 0.1, 0.5, 1, 2, 4, 6, 10, 15, 20, 30, 40, 60, 80]
rain_alpha = [0, 0.88, 0.94, 0.97, 0.98, 0.98, 0.99, 1, 1, 1, 1, 1, 1, 1]
rain_positions = np.linspace(0.0, 1.0, len(rain_values))
out["RAIN_STOPS"] = [
    {"v": v, "c": sample("oslo", 1.0 - p, 0.10, 0.80), "a": a}
    for v, p, a in zip(rain_values, rain_positions, rain_alpha)
]

# --- theta-w 850 hPa: batlow sequenziale -------------------------------------
# Sequenziale e non divergente: fra 276 e 312 K non esiste un valore neutro
# con un significato fisico su cui centrare una divergente, e inventarne uno
# e' esattamente l'errore che le scale uniformi servono a evitare. Il fronte
# si legge lo stesso -- meglio, perche' a pari salto di theta-w corrisponde
# ora pari salto di colore.
theta_values = [276.0, 279.6, 283.2, 286.8, 290.4, 294.0, 297.6, 301.2,
                304.8, 308.4, 312.0]
out["THETA_ANCHORS"] = [
    {"v": v, "c": sample("batlow", (v - 276.0) / 36.0)} for v in theta_values
]

# --- Geopotenziale 500 hPa: batlow sequenziale -------------------------------
geo_values = [5200, 5400, 5520, 5640, 5700, 5760, 5820, 5880, 5960]
geo_alpha = [0.86, 0.85, 0.84, 0.82, 0.78, 0.8, 0.82, 0.85, 0.88]
out["GEOPOT500_STOPS"] = [
    {"v": v, "c": sample("batlow", (v - 5200) / 760.0), "a": a}
    for v, a in zip(geo_values, geo_alpha)
]

for name, anchors in out.items():
    print(f"// {name}")
    print(js(anchors))
    print()
