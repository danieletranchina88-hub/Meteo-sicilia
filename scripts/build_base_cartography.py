"""Costruisce il fondo cartografico del sito dal Natural Earth, ritagliato al
dominio ICON-2I. Solo cio' che una carta meteo usa: costa, confini, laghi,
fiumi principali, citta' diradate. Niente strade, niente uso del suolo.

Il risultato finisce in ``data_base/`` e sta nel repository: e' cartografia
statica, non cambia a ogni corsa del modello, e va servita insieme alla pagina.
Natural Earth e' di dominio pubblico -- nessuna attribuzione dovuta, nessuna
chiave, nessun limite di traffico.

Uso: scaricare in questa cartella i file ``ne_10m_*.geojson`` da
https://github.com/nvkelso/natural-earth-vector/tree/master/geojson e lanciare
``python scripts/build_base_cartography.py``. Va rifatto solo se cambia il
dominio del modello o se si vuole aggiungere un tema.
"""
import json, gzip, os, sys

sys.setrecursionlimit(100000)

DOM = (3.0, 33.7, 22.0, 48.9)          # il dominio del modello
PAD = 0.6                               # un margine, per non tagliare sul bordo
BOX = (DOM[0] - PAD, DOM[1] - PAD, DOM[2] + PAD, DOM[3] + PAD)


def dentro_punto(x, y):
    return BOX[0] <= x <= BOX[2] and BOX[1] <= y <= BOX[3]


def tocca(coords):
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return not (max(xs) < BOX[0] or min(xs) > BOX[2]
                or max(ys) < BOX[1] or min(ys) > BOX[3])


def semplifica(punti, tolleranza):
    """Douglas-Peucker. La costa a scala sinottica non ha bisogno di un
    vertice ogni cento metri: pesa e non si vede."""
    if len(punti) < 3:
        return punti
    dmax, idx = 0.0, 0
    ax, ay = punti[0]
    bx, by = punti[-1]
    dx, dy = bx - ax, by - ay
    norm = (dx * dx + dy * dy) ** 0.5
    for i in range(1, len(punti) - 1):
        px, py = punti[i]
        if norm == 0:
            d = ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
        else:
            d = abs(dy * px - dx * py + bx * ay - by * ax) / norm
        if d > dmax:
            dmax, idx = d, i
    if dmax <= tolleranza:
        return [punti[0], punti[-1]]
    return (semplifica(punti[:idx + 1], tolleranza)[:-1]
            + semplifica(punti[idx:], tolleranza))


def tonda(punti, cifre=4):
    return [[round(x, cifre), round(y, cifre)] for x, y in punti]


def linee(path, tolleranza, tieni=None):
    """Estrae le linee che toccano il dominio, semplificate e arrotondate."""
    dati = json.load(open(path))
    out = []
    for f in dati["features"]:
        prop = f.get("properties") or {}
        if tieni and not tieni(prop):
            continue
        g = f["geometry"]
        if g is None:
            continue
        parti = ([g["coordinates"]] if g["type"] == "LineString"
                 else g["coordinates"] if g["type"] == "MultiLineString" else [])
        for parte in parti:
            if len(parte) < 2 or not tocca(parte):
                continue
            out.append(tonda(semplifica([list(p[:2]) for p in parte], tolleranza)))
    return out


def poligoni(path, tolleranza, tieni=None):
    dati = json.load(open(path))
    out = []
    for f in dati["features"]:
        prop = f.get("properties") or {}
        if tieni and not tieni(prop):
            continue
        g = f["geometry"]
        if g is None:
            continue
        gruppi = ([g["coordinates"]] if g["type"] == "Polygon"
                  else g["coordinates"] if g["type"] == "MultiPolygon" else [])
        for anelli in gruppi:
            fuori = anelli[0]
            if len(fuori) < 4 or not tocca(fuori):
                continue
            semplificati = []
            for anello in anelli:
                punti = [list(p[:2]) for p in anello]
                if tolleranza > 0:
                    punti = semplifica(punti, tolleranza)
                punti = ritaglia(punti, BOX)
                if len(punti) >= 4:
                    semplificati.append(tonda(punti + [punti[0]]))
            if semplificati:
                out.append(semplificati)
    return out



def ritaglia(anello, box):
    """Sutherland-Hodgman: ritaglia un anello al riquadro. Senza questo, il
    poligono dell'oceano entra intero -- otto megabyte di coste del mondo per
    campire il Mediterraneo."""
    x0, y0, x1, y1 = box
    def taglio(punti, dentro, incrocia):
        if not punti:
            return []
        out = []
        prec = punti[-1]
        for p in punti:
            if dentro(p):
                if not dentro(prec):
                    out.append(incrocia(prec, p))
                out.append(p)
            elif dentro(prec):
                out.append(incrocia(prec, p))
            prec = p
        return out
    def interp(a, b, t):
        return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
    r = anello
    r = taglio(r, lambda p: p[0] >= x0,
               lambda a, b: interp(a, b, (x0 - a[0]) / (b[0] - a[0])))
    r = taglio(r, lambda p: p[0] <= x1,
               lambda a, b: interp(a, b, (x1 - a[0]) / (b[0] - a[0])))
    r = taglio(r, lambda p: p[1] >= y0,
               lambda a, b: interp(a, b, (y0 - a[1]) / (b[1] - a[1])))
    r = taglio(r, lambda p: p[1] <= y1,
               lambda a, b: interp(a, b, (y1 - a[1]) / (b[1] - a[1])))
    return r

def fc(geoms, tipo, props=None):
    feats = []
    for i, g in enumerate(geoms):
        feats.append({"type": "Feature",
                      "properties": (props[i] if props else {}),
                      "geometry": {"type": tipo, "coordinates": g}})
    return {"type": "FeatureCollection", "features": feats}


def scrivi(nome, obj):
    testo = json.dumps(obj, separators=(",", ":"))
    open(f"data_base/{nome}.json", "w").write(testo)
    gz = len(gzip.compress(testo.encode()))
    print(f"  {nome:<14} {len(obj['features']):>5} elementi"
          f"  {len(testo)/1024:8.0f} KB  compresso {gz/1024:7.0f} KB")
    return len(testo), gz


os.makedirs("data_base", exist_ok=True)
print("fondo vettoriale, dominio ICON-2I:")
tot = gztot = 0

# La costa non si semplifica. Misurato: semplificarla a 445 m risparmia 45 KB
# compressi su 107 e in cambio squadra il promontorio di Portofino. Su una
# carta il costo si vede e il risparmio no.
n, g = scrivi("coast", fc(linee("ne_10m_coastline.geojson", 0), "LineString"))
tot += n; gztot += g

# La campitura della terra, dallo stesso dato a 10m della linea di costa: cosi'
# il bordo della campitura E' la linea. Prima veniva da un dato a 50m e lungo
# tutto il litorale si vedeva una frangia chiara sbordare oltre il tratto nero.
# Ritagliata al riquadro, altrimenti entrerebbero i continenti interi.
n, g = scrivi("land", fc(poligoni("ne_10m_land.geojson", 0), "Polygon"))
tot += n; gztot += g

# Il mare come poligono, ritagliato: serve a rimettere l'azzurro pulito sopra
# il rilievo, che altrimenti ombreggia anche il mare e lo sporca di grigio.
n, g = scrivi("sea", fc(poligoni("ne_10m_ocean.geojson", 0), "Polygon"))
tot += n; gztot += g

# Confini nazionali e regionali.
n, g = scrivi("borders", fc(linee("ne_10m_admin_0_boundary_lines_land.geojson", 0),
                            "LineString"))
tot += n; gztot += g
n, g = scrivi("regions", fc(linee("ne_10m_admin_1_states_provinces_lines.geojson", 0,
                                  lambda p: p.get("ADM0_A3") == "ITA"), "LineString"))
tot += n; gztot += g

# Laghi: solo quelli che si vedono a scala sinottica.
n, g = scrivi("lakes", fc(poligoni("ne_10m_lakes.geojson", 0.003,
                                   lambda p: (p.get("scalerank") if p.get("scalerank") is not None else 99) <= 7), "Polygon"))
tot += n; gztot += g

# Fiumi principali.
n, g = scrivi("rivers", fc(linee("ne_10m_rivers_lake_centerlines.geojson", 0.004,
                                 lambda p: (p.get("scalerank") or 12) <= 6), "LineString"))
tot += n; gztot += g

# Citta': tenute con il loro rango, cosi' il disegno le dirada da solo.
citta = json.load(open("ne_10m_populated_places.geojson"))
punti, props = [], []
for f in citta["features"]:
    p = f.get("properties") or {}
    x, y = f["geometry"]["coordinates"][:2]
    if not dentro_punto(x, y):
        continue
    rango = p.get("SCALERANK", p.get("scalerank", 10))
    if rango is None or rango > 7:
        continue
    punti.append([round(x, 4), round(y, 4)])
    # Il nome in italiano quando c'e': il sito e' italiano, e "Milan" su una
    # carta dell'Italia e' semplicemente sbagliato.
    nome = (p.get("NAME_IT") or p.get("name_it")
            or p.get("NAME") or p.get("name") or "")
    props.append({"n": nome,
                  "r": int(rango),
                  "pop": int(p.get("POP_MAX") or p.get("pop_max") or 0)})
n, g = scrivi("places", fc(punti, "Point", props))
tot += n; gztot += g

print(f"  {'TOTALE':<14} {'':>5}          {tot/1024:8.0f} KB  compresso {gztot/1024:7.0f} KB")
