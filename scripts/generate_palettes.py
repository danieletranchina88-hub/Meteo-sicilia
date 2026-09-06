"""Stampa la sorgente canonica delle palette operative della mappa.

Uso: ``python scripts/generate_palettes.py``. L'output e' JavaScript pronto
da confrontare o copiare negli array omonimi di ``index.html``.

Le rampe sono esplicite e dipendono dal significato del campo: neutro e
trasparente quando il fenomeno non esiste, sequenziale per intensita',
divergente solo dove esiste una soglia fisica o sinottica utile. Non servono
dipendenze grafiche esterne, quindi lo stesso RGB viene rigenerato in ogni
ambiente senza cambiamenti fra versioni di libreria.
"""


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

# --- Campi di superficie ----------------------------------------------------
out["WIND_STOPS"] = [
    {"v": 0, "c": [250, 250, 249]},
    {"v": 5, "c": [174, 229, 252]},
    {"v": 10, "c": [101, 201, 251]},
    {"v": 15, "c": [63, 154, 245]},
    {"v": 20, "c": [113, 248, 163]},
    {"v": 25, "c": [101, 203, 112]},
    {"v": 30, "c": [126, 203, 80]},
    {"v": 35, "c": [144, 249, 76]},
    {"v": 40, "c": [184, 238, 88]},
    {"v": 45, "c": [190, 207, 69]},
    {"v": 50, "c": [220, 239, 80]},
    {"v": 55, "c": [201, 158, 50]},
    {"v": 60, "c": [239, 159, 58]},
    {"v": 65, "c": [242, 159, 112]},
    {"v": 70, "c": [196, 155, 154]},
    {"v": 75, "c": [194, 105, 64]},
    {"v": 80, "c": [191, 62, 58]},
    {"v": 85, "c": [232, 54, 42]},
    {"v": 90, "c": [184, 36, 25]},
    {"v": 95, "c": [118, 20, 13]},
    {"v": 100, "c": [118, 20, 80]},
    {"v": 105, "c": [145, 29, 113]},
    {"v": 110, "c": [188, 38, 198]},
    {"v": 115, "c": [234, 52, 243]},
    {"v": 120, "c": [236, 80, 251]},
    {"v": 125, "c": [238, 135, 252]},
    {"v": 130, "c": [242, 164, 249]},
    {"v": 135, "c": [246, 194, 247]},
]

# Raffiche. Il vento medio si legge su una scala regolare, la raffica no: cio'
# che conta e' il grado Beaufort raggiunto, perche' e' quello che descrive il
# danno. Percio' i bordi delle fasce non sono numeri tondi ma le soglie della
# scala Beaufort convertite in km/h -- 12, 20, 29, 39, 50, 62, 75, 89, 103,
# 118 -- e ogni cambio di colore e' un cambio di grado. Le tre soglie che si
# devono riconoscere senza leggere la legenda (50 km/h vento forte, 75
# burrasca forte, 103 tempesta violenta) cadono su salti cromatici
# volutamente ampi.
#
# Misurato in CAM02-UCS fra fasce adiacenti: distanza minima 7,6, e 4,9
# simulando deuteranomalia, protanomalia e tritanomalia -- sopra la soglia di
# distinguibilita'. Il salto piu' grande, 26,4, cade a 50 km/h.
out["GUST_STOPS"] = [
    {"v": 0, "c": [236, 242, 244]},
    {"v": 12, "c": [206, 232, 226]},
    {"v": 20, "c": [160, 214, 200]},
    {"v": 29, "c": [108, 190, 176]},
    {"v": 39, "c": [96, 176, 118]},
    {"v": 50, "c": [214, 199, 74]},
    {"v": 62, "c": [232, 155, 53]},
    {"v": 75, "c": [222, 106, 47]},
    {"v": 89, "c": [199, 56, 56]},
    {"v": 103, "c": [160, 28, 74]},
    {"v": 118, "c": [110, 30, 110]},
    {"v": 140, "c": [70, 22, 92]},
]

# Probabilita'. Una grandezza sola, quindi una scala sola per tutti i livelli
# probabilistici: il 60% ha lo stesso colore che parli di pioggia o di raffiche,
# e i due campi restano confrontabili a colpo d'occhio. Sequenziale e con
# luminosita' monotona, percio' resta ordinata anche stampata in bianco e nero;
# sale verso il viola per non confondersi con pioggia e raffiche, che percorrono
# gia' verde-giallo-rosso. Trasparente a zero: dove non succede niente si deve
# vedere la carta.
#
# Misurata in CAM02-UCS fra fasce adiacenti: distanza minima 7,8 e mediana 11,6,
# con 5,1 simulando deuteranomalia, protanomalia e tritanomalia. I salti sono
# volutamente uniformi (7,8-13,6) perche' incrementi uguali di probabilita'
# devono sembrare ugualmente diversi: su una scala divergente o a salti
# irregolari il passaggio dal 40 al 50% sembrerebbe piu' grande di quello dal
# 70 all'80, che sarebbe falso.
out["PROB_STOPS"] = [
    {"v": 0, "c": [236, 244, 248], "a": 0},
    {"v": 10, "c": [206, 231, 243], "a": 0.62},
    {"v": 20, "c": [166, 211, 235], "a": 0.78},
    {"v": 30, "c": [122, 186, 224], "a": 0.86},
    {"v": 40, "c": [84, 157, 210], "a": 0.9},
    {"v": 50, "c": [62, 124, 192], "a": 0.92},
    {"v": 60, "c": [66, 92, 172], "a": 0.93},
    {"v": 70, "c": [86, 62, 148], "a": 0.94},
    {"v": 80, "c": [104, 38, 124], "a": 0.95},
    {"v": 90, "c": [114, 20, 92], "a": 0.96},
    {"v": 100, "c": [104, 10, 58], "a": 0.96},
]

# Quote in metri: zero termico e quota neve. Il verso e' quello intuitivo -- in
# basso fa freddo, quindi il blu sta alle quote basse: una quota neve a 300 m e'
# notizia, a 3500 non lo e'.
#
# La luminosita' sale fino a 2100 m e poi riscende, e non e' una svista: non e'
# una scala di intensita' ma di quota, e in Italia i 2100 m sono il valore che
# non dice niente, mentre entrambi gli estremi -- neve sul mare, oppure zero
# termico altissimo -- meritano di risaltare.
#
# Misurata in CAM02-UCS: distanza minima fra fasce 10,4, mediana 14,4, e 5,8
# simulando il daltonismo piu' penalizzante. La prima stesura si fermava a 2,3
# fra 2100 e 2400 m, cioe' due gialli quasi uguali per chi ha deuteranomalia.
out["ALTITUDE_STOPS"] = [
    {"v": 0, "c": [74, 30, 104]},
    {"v": 300, "c": [62, 62, 152]},
    {"v": 600, "c": [52, 104, 184]},
    {"v": 900, "c": [58, 146, 196]},
    {"v": 1200, "c": [86, 180, 186]},
    {"v": 1500, "c": [128, 202, 150]},
    {"v": 1800, "c": [186, 216, 124]},
    {"v": 2100, "c": [236, 230, 138]},
    {"v": 2400, "c": [224, 176, 84]},
    {"v": 2700, "c": [212, 138, 72]},
    {"v": 3000, "c": [196, 104, 78]},
    {"v": 3500, "c": [172, 80, 96]},
    {"v": 4000, "c": [146, 74, 116]},
    {"v": 4500, "c": [122, 72, 130]},
]

out["RAIN_STOPS"] = [
    {"v": 0, "c": [0, 0, 0], "a": 0},
    {"v": 0.1, "c": [213, 240, 255], "a": 0.78},
    {"v": 0.5, "c": [134, 210, 255], "a": 0.9},
    {"v": 1, "c": [57, 168, 242], "a": 0.95},
    {"v": 2, "c": [37, 102, 212], "a": 0.98},
    {"v": 4, "c": [32, 180, 134], "a": 0.98},
    {"v": 10, "c": [242, 211, 79], "a": 1},
    {"v": 20, "c": [242, 142, 43], "a": 1},
    {"v": 40, "c": [214, 69, 69], "a": 1},
    {"v": 80, "c": [142, 47, 143], "a": 1},
]

out["CLOUD_ANCHORS"] = [
    {"v": 0, "c": [247, 250, 250], "a": 0},
    {"v": 10, "c": [241, 245, 246], "a": 0.28},
    {"v": 30, "c": [221, 230, 233], "a": 0.62},
    {"v": 50, "c": [187, 200, 205], "a": 0.78},
    {"v": 70, "c": [139, 154, 162], "a": 0.88},
    {"v": 90, "c": [91, 105, 113], "a": 0.95},
    {"v": 100, "c": [57, 70, 77], "a": 0.99},
]

out["RH_ANCHORS"] = [
    {"v": 0, "c": [112, 64, 37]},
    {"v": 20, "c": [185, 120, 62]},
    {"v": 40, "c": [217, 185, 110]},
    {"v": 60, "c": [121, 185, 158]},
    {"v": 80, "c": [42, 145, 163]},
    {"v": 100, "c": [39, 75, 143]},
]

# Pressione. I valori non vengono campionati qui a ogni esecuzione: questo file
# dichiara di non dipendere da librerie grafiche, cosi' lo stesso RGB esce in
# ogni ambiente. Sono il risultato, riportato una volta sola, di questa
# costruzione:
#
#   tavolozza RdBu di ColorBrewer, divergente e percettivamente uniforme,
#   invertita perche' il blu vada in basso; il legame valore->colore e'
#   deformato in modo che l'80% del percorso cromatico cada fra 1000 e
#   1032 hPa, dove la pressione al livello del mare sull'Italia sta quasi
#   sempre, e il resto sulle code fino a 976 e 1048.
#
# Serve perche' una scala distesa uniformemente su 960-1048 spendeva meta'
# della gamma su valori che non capitano mai: una giornata normale, da 1011 a
# 1025, ricadeva in tre o quattro fasce quasi identiche, tutte fra il crema e
# l'ambra. Misurato in CAM02-UCS su fasce da 2 hPa fra 1000 e 1032: distanza
# mediana fra fasce adiacenti 8,9 contro 4,7 di prima, minima 5,5. Simulando
# deuteranomalia, protanomalia e tritanomalia la minima resta 4,0, ben sopra
# la soglia di distinguibilita'.
out["PRESS_ANCHORS"] = [
    {"v": 976, "c": [5, 48, 97]},
    {"v": 988, "c": [21, 80, 141]},
    {"v": 1000, "c": [39, 110, 176]},
    {"v": 1004, "c": [82, 157, 200]},
    {"v": 1008, "c": [157, 203, 225]},
    {"v": 1012, "c": [216, 233, 241]},
    {"v": 1016, "c": [248, 241, 237]},
    {"v": 1020, "c": [251, 208, 185]},
    {"v": 1024, "c": [238, 150, 119]},
    {"v": 1028, "c": [207, 82, 70]},
    {"v": 1032, "c": [162, 19, 40]},
    {"v": 1040, "c": [132, 9, 36]},
    {"v": 1048, "c": [103, 0, 31]},
]

# --- Campi sinottici e in quota --------------------------------------------
out["THETA_ANCHORS"] = [
    {"v": 276, "c": [37, 52, 148]},
    {"v": 282, "c": [44, 127, 184]},
    {"v": 288, "c": [127, 205, 187]},
    {"v": 294, "c": [243, 241, 218]},
    {"v": 300, "c": [253, 187, 92]},
    {"v": 306, "c": [227, 74, 51]},
    {"v": 312, "c": [140, 29, 64]},
]

out["GEOPOT500_STOPS"] = [
    {"v": 5200, "c": [59, 15, 112], "a": 0.88},
    {"v": 5400, "c": [44, 77, 155], "a": 0.86},
    {"v": 5520, "c": [38, 136, 184], "a": 0.84},
    {"v": 5640, "c": [183, 216, 216], "a": 0.8},
    {"v": 5700, "c": [242, 230, 177], "a": 0.78},
    {"v": 5820, "c": [231, 163, 75], "a": 0.83},
    {"v": 5960, "c": [182, 64, 59], "a": 0.88},
]

out["T850_ANCHORS"] = [
    {"v": -30, "c": [22, 11, 53]},
    {"v": -20, "c": [36, 22, 79]},
    {"v": -10, "c": [46, 94, 170]},
    {"v": -5, "c": [111, 195, 213]},
    {"v": 0, "c": [237, 242, 242]},
    {"v": 10, "c": [246, 211, 101]},
    {"v": 20, "c": [233, 122, 52]},
    {"v": 30, "c": [181, 54, 74]},
    {"v": 35, "c": [111, 27, 58]},
]

# --- Diagnostica convettiva e fenomeni a impatto ---------------------------
out["CONVECTION_STOPS"] = [
    {"v": 0, "c": [255, 255, 255], "a": 0},
    {"v": 20, "c": [178, 210, 223], "a": 0.45},
    {"v": 40, "c": [242, 212, 92], "a": 0.75},
    {"v": 70, "c": [229, 111, 55], "a": 0.9},
    {"v": 95, "c": [143, 40, 92], "a": 1.0},
]

out["VISIBILITY_STOPS"] = [
    {"v": 0, "c": [63, 45, 86], "a": 1.0},
    {"v": 200, "c": [102, 90, 120], "a": 0.9},
    {"v": 1000, "c": [154, 167, 184], "a": 0.72},
    {"v": 5000, "c": [214, 227, 232], "a": 0.35},
    {"v": 10000, "c": [247, 250, 250], "a": 0},
]

out["FREEZING_STOPS"] = [
    {"v": 0, "c": [255, 255, 255], "a": 0},
    {"v": 1, "c": [139, 197, 227], "a": 0.9},
    {"v": 2, "c": [135, 57, 166], "a": 1.0},
]

out["FOEHN_STOPS"] = [
    {"v": 0, "c": [255, 255, 255], "a": 0},
    {"v": 1, "c": [47, 128, 237], "a": 0.88},
    {"v": 2, "c": [231, 111, 81], "a": 0.88},
]

out["STORM_STOPS"] = [
    {"v": 0, "c": [221, 234, 242], "a": 0},
    {"v": 10, "c": [201, 223, 233], "a": 0.25},
    {"v": 20, "c": [178, 210, 223], "a": 0.45},
    {"v": 40, "c": [242, 212, 92], "a": 0.78},
    {"v": 60, "c": [240, 147, 53], "a": 0.9},
    {"v": 80, "c": [209, 73, 91], "a": 0.97},
    {"v": 100, "c": [118, 30, 89], "a": 1.0},
]

out["UPDRAFT_STOPS"] = [
    {"v": -5, "c": [35, 78, 155]},
    {"v": -1, "c": [131, 189, 227]},
    {"v": 0, "c": [238, 243, 243]},
    {"v": 1, "c": [168, 216, 163]},
    {"v": 3, "c": [243, 212, 91]},
    {"v": 8, "c": [242, 142, 43]},
    {"v": 15, "c": [214, 69, 80]},
    {"v": 30, "c": [123, 30, 99]},
]

out["CAPE_STOPS"] = [
    {"v": 0, "c": [238, 244, 240], "a": 0},
    {"v": 300, "c": [201, 233, 212], "a": 0.5},
    {"v": 800, "c": [231, 222, 84], "a": 0.82},
    {"v": 1500, "c": [242, 180, 71], "a": 0.9},
    {"v": 2500, "c": [230, 91, 58], "a": 0.97},
    {"v": 4000, "c": [133, 30, 90], "a": 1.0},
]

out["TRIGGER_STOPS"] = [
    {"v": 0, "c": [234, 240, 239], "a": 0},
    {"v": 15, "c": [214, 232, 229], "a": 0.35},
    {"v": 35, "c": [143, 203, 179], "a": 0.72},
    {"v": 55, "c": [241, 211, 107], "a": 0.84},
    {"v": 75, "c": [238, 145, 64], "a": 0.93},
    {"v": 100, "c": [184, 50, 74], "a": 1.0},
]

out["BOWEN_STOPS"] = [
    {"v": 0, "c": [31, 111, 120]},
    {"v": 0.3, "c": [76, 165, 139]},
    {"v": 0.8, "c": [168, 198, 134]},
    {"v": 1.5, "c": [217, 181, 109]},
    {"v": 3, "c": [185, 120, 62]},
    {"v": 6, "c": [108, 58, 36]},
]

for name, anchors in out.items():
    print(f"// {name}")
    print(js(anchors))
    print()
