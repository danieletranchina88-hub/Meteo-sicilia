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

# --- Temperatura: scala fornita, in kelvin ----------------------------------
# Questa non viene da una libreria: e' una scala fornita, espressa in kelvin.
# Qui si limita a essere convertita in gradi Celsius, senza ricampionarla ne'
# rimappare i valori, perche' le posizioni degli ancoraggi fanno parte della
# scala tanto quanto i colori.
#
# Il salto fra 273,15 K e 274 K -- blu [93,133,198] e verde [68,125,99] in
# meno di un kelvin -- e' la linea del gelo e va conservato. Perche' cada
# davvero a zero, le fasce discrete del sito vanno campionate al centro e non
# al bordo inferiore: vedi ``sampleAtMidpoint`` in index.html.
TEMPERATURE_KELVIN = [
    (203.0, [115, 70, 105]),
    (218.0, [202, 172, 195]),
    (233.0, [162, 70, 145]),
    (248.0, [143, 89, 169]),
    (258.0, [157, 219, 217]),
    (265.0, [106, 191, 181]),
    (269.0, [100, 166, 189]),
    (273.15, [93, 133, 198]),
    (274.0, [68, 125, 99]),
    (283.0, [128, 147, 24]),
    (294.0, [243, 183, 4]),
    (303.0, [232, 83, 25]),
    (320.0, [71, 14, 0]),
]
out["TEMP_ANCHORS"] = [
    {"v": round(kelvin - 273.15, 2), "c": colour}
    for kelvin, colour in TEMPERATURE_KELVIN
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
