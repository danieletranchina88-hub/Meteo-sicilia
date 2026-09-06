"""Probabilita' di superamento sul vicinato, su griglia geografica anisotropa.

E' il modo standard di leggere un modello che risolve la convezione (Theis,
Hense e Damrath 2005; Schwartz e Sobash 2017).  Un modello a 2,2 km sa che ci
sara' un rovescio, non su quale paese cadra': la posizione esatta di una cella
non e' prevedibile, la probabilita' che una cella interessi una zona lo e'.
Chiedere al campo puntuale "quanta pioggia cade qui" e' quindi una domanda mal
posta; chiedere "che probabilita' c'e' che entro R chilometri cadano almeno T
millimetri" e' una domanda a cui il modello puo' davvero rispondere, ed e'
anche l'unica delle due che si possa verificare contro le osservazioni.

La definizione e' una frazione, non un modello statistico:

    P(x) = |{ y : dist(x, y) <= R  e  F(y) >= T }| / |{ y : dist(x, y) <= R }|

Niente addestramento, niente parametri liberi, niente archivio: solo il
conteggio dei punti del vicinato che superano la soglia.  I punti mancanti non
contano ne' a favore ne' contro, cioe' escono da entrambi gli insiemi.

Il vicinato e' un cerchio **sul terreno**, e qui sta la differenza con una
convoluzione qualsiasi.  La griglia ICON-2I e' regolare in gradi, non in
chilometri: il passo in latitudine resta 2,226 km ovunque, quello in
longitudine passa da 2,315 km a 33,7 gradi N a 1,829 km a 48,9 gradi N, perche'
i meridiani si stringono verso il polo.  Un disco con lo stesso numero di celle
in x e in y sarebbe quindi un'ellisse sul terreno, schiacciata in longitudine
del 18% al bordo nord del dominio.  Per questo il raggio in celle lungo x
dipende dalla riga, e viene ricavato dalla latitudine di quella riga.

Due convenzioni, dichiarate perche' non sono le uniche possibili.

La distanza est-ovest e' misurata nel piano tangente al punto CENTRALE, cioe'
con il passo in longitudine della sua latitudine, non di quella del vicino.
Su un vicinato di 25 km la latitudine cambia al massimo di 0,22 gradi e il
coseno con essa dello 0,45%: trascurabile accanto al 18% che questo modulo
corregge, e in cambio il vicinato resta un insieme ben definito attorno a x.

Al bordo del dominio il vicinato si restringe invece di replicare l'ultima
riga.  Replicare significherebbe contare piu' volte un dato che non esiste e
farlo pesare su una probabilita'; qui il mezzo disco disponibile sta sia al
numeratore sia al denominatore, e la frazione resta quella dei punti veri.
"""

from __future__ import annotations

import numpy as np

EARTH_DEGREE_KM = 111.32


def cell_sizes_km(meta: dict) -> tuple[np.ndarray, float]:
    """Passo della griglia in chilometri: per riga lungo x, costante lungo y.

    ``meta`` e' l'intestazione pubblicata con ogni campo (``la1``, ``dy``,
    ``dx``, ``ny``).  Restituisce ``(cell_x_km, cell_y_km)`` dove il primo e'
    un vettore lungo ``ny``: e' la larghezza reale di una cella a quella
    latitudine, e serve perche' il vicinato resti circolare sul terreno.
    """
    latitudes = float(meta["la1"]) - np.arange(int(meta["ny"])) * float(meta["dy"])
    cell_x_km = float(meta["dx"]) * EARTH_DEGREE_KM * np.cos(np.radians(latitudes))
    cell_y_km = float(meta["dy"]) * EARTH_DEGREE_KM
    return cell_x_km, cell_y_km


def _row_half_widths(
    radius_km: float, cell_x_km: np.ndarray, cell_y_km: float, radius_y: int
) -> np.ndarray:
    """Semi-larghezza in celle del disco, per ogni riga e ogni scarto in y.

    Per uno scarto ``dy`` di righe la distanza percorsa in latitudine e'
    ``dy * cell_y_km``; quel che resta del raggio va speso in longitudine, dove
    una cella vale ``cell_x_km`` di quella riga.  Da qui il teorema di
    Pitagora, e non un raggio costante in celle.
    """
    offsets = np.arange(-radius_y, radius_y + 1, dtype=float)
    along_y = np.abs(offsets) * float(cell_y_km)
    # Fuori dal cerchio la radice sarebbe immaginaria: quella riga non
    # partecipa, e la semi-larghezza resta negativa per segnalarlo.
    remaining = np.square(float(radius_km)) - np.square(along_y)
    span_km = np.where(remaining >= 0.0, np.sqrt(np.maximum(remaining, 0.0)), -1.0)
    widths = np.floor(span_km[:, None] / np.asarray(cell_x_km)[None, :])
    return np.where(span_km[:, None] < 0.0, -1.0, widths).astype(int)


def _neighbourhood_sums(
    values: np.ndarray,
    radius_km: float,
    cell_x_km: np.ndarray,
    cell_y_km: float,
) -> np.ndarray:
    """Somma di ``values`` sul disco geografico centrato in ogni punto.

    Il disco e' scomposto in segmenti orizzontali, uno per riga, e la somma di
    un segmento si legge in tempo costante da una somma cumulata lungo x.  Il
    costo cresce con il raggio in righe, non in celle: con raggio 25 km il
    disco contiene circa 400 punti ma bastano 23 letture per punto.

    Le righe hanno semi-larghezze diverse, perche' il disco e' geografico:
    quelle che condividono lo stesso numero intero di celle vengono trattate
    insieme, e i valori distinti sono pochissimi (tre o quattro sull'intero
    dominio), quindi il ciclo resta corto e il lavoro tutto dentro numpy.
    """
    height, width = values.shape
    radius_y = int(np.floor(float(radius_km) / float(cell_y_km)))
    widths = _row_half_widths(radius_km, cell_x_km, cell_y_km, radius_y)

    prefix = np.zeros((height, width + 1), dtype=float)
    np.cumsum(values, axis=1, out=prefix[:, 1:])

    rows = np.arange(height)
    columns = np.arange(width)
    total = np.zeros((height, width), dtype=float)
    for offset_index, offset in enumerate(range(-radius_y, radius_y + 1)):
        # Fuori griglia non si replica: la riga semplicemente non c'e'. Il
        # vicinato di un punto di bordo e' un mezzo disco, e la frazione viene
        # calcolata su quel mezzo disco perche' anche il denominatore lo usa.
        inside = (rows + offset >= 0) & (rows + offset < height)
        row_widths = widths[offset_index]
        for half in np.unique(row_widths):
            if half < 0:
                continue
            selected = rows[inside & (row_widths == half)]
            if not selected.size:
                continue
            left = np.maximum(columns - half, 0)
            right = np.minimum(columns + half + 1, width)
            block = prefix[selected + offset]
            total[selected] += (
                block[:, right] - block[:, left]
            )
    return total


def neighbourhood_probability(
    field,
    threshold: float,
    radius_km: float,
    *,
    cell_x_km,
    cell_y_km: float,
) -> np.ndarray:
    """Frazione di punti entro ``radius_km`` che raggiunge ``threshold`` (0-1).

    ``cell_x_km`` puo' essere uno scalare (griglia gia' metrica) oppure un
    vettore lungo quanto le righe (griglia in gradi): nel secondo caso il
    vicinato resta circolare sul terreno a ogni latitudine.

    La soglia e' inclusiva: "almeno T", non "piu' di T".  Su un campo continuo
    la differenza e' di misura nulla, ma la legenda dice "almeno 20 mm" e il
    conto deve dire la stessa cosa.
    """
    values = np.asarray(field, dtype=float)
    if values.ndim != 2:
        raise ValueError("il campo deve essere una griglia 2D")
    if float(radius_km) <= 0.0 or float(cell_y_km) <= 0.0:
        raise ValueError("raggio e passo della griglia devono essere positivi")

    height = values.shape[0]
    cells_x = np.asarray(cell_x_km, dtype=float)
    if cells_x.ndim == 0:
        cells_x = np.full(height, float(cells_x))
    if cells_x.shape != (height,):
        raise ValueError("cell_x_km deve essere scalare o lungo quanto le righe")
    if not np.all(cells_x > 0.0):
        raise ValueError("il passo in longitudine deve essere positivo")

    valid = np.isfinite(values)
    reaches = np.where(valid, values >= float(threshold), False)

    hits = _neighbourhood_sums(
        reaches.astype(float), radius_km, cells_x, float(cell_y_km)
    )
    counts = _neighbourhood_sums(
        valid.astype(float), radius_km, cells_x, float(cell_y_km)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        probability = hits / counts
    return np.where(counts > 0.0, probability, np.nan)


def event_probability(
    field,
    threshold: float,
    *,
    cell_x_km,
    cell_y_km: float,
    event_radius_km: float = 10.0,
    spread_radius_km: float = 25.0,
) -> np.ndarray:
    """Probabilita' (0-1) che l'evento "almeno T entro ``event_radius_km``" accada.

    Due stadi, e la differenza fra i due non e' un dettaglio di forma: e' la
    differenza fra due domande diverse.

    ``neighbourhood_probability`` da' la frazione di area che supera la soglia.
    E' una quantita' corretta, ma piccola per definizione quando il fenomeno e'
    a chiazze: misurato sul run del 6 settembre, la pioggia oraria sopra 1 mm
    copriva lo 0,013% dei punti, e la frazione su un disco da 15 km non saliva
    sopra 0,33 nemmeno dentro il rovescio.  Letta come "probabilita' che piova"
    sarebbe fuorviante verso il basso.

    La domanda che interessa non e' "quanta parte del disco si bagna" ma "piove
    da qualche parte vicino a me".  Quindi:

    1. l'evento diventa areale -- ``E(x) = 1`` se il campo raggiunge la soglia
       in almeno un punto entro ``event_radius_km`` da ``x``.  Ancora nessuna
       probabilita': e' solo la definizione di cosa conta come "e' successo";
    2. la probabilita' e' la frazione di punti entro ``spread_radius_km`` dove
       ``E`` vale 1.  Qui entra l'incertezza: un modello a 2,2 km sbaglia la
       posizione della cella, non la sua esistenza, e ``spread_radius_km`` e'
       la scala su cui quella posizione e' incerta.

    Il secondo raggio non e' tarato su un archivio, perche' un archivio non
    c'e': e' una scelta dichiarata, e vale come tale.  Cambiarlo cambia quanto
    la probabilita' si spalma, non dove sta il fenomeno.
    """
    values = np.asarray(field, dtype=float)
    if values.ndim != 2:
        raise ValueError("il campo deve essere una griglia 2D")
    if float(event_radius_km) <= 0.0 or float(spread_radius_km) <= 0.0:
        raise ValueError("entrambi i raggi devono essere positivi")

    height = values.shape[0]
    cells_x = np.asarray(cell_x_km, dtype=float)
    if cells_x.ndim == 0:
        cells_x = np.full(height, float(cells_x))

    return event_probabilities(
        values,
        [threshold],
        cell_x_km=cells_x,
        cell_y_km=cell_y_km,
        event_radius_km=event_radius_km,
        spread_radius_km=spread_radius_km,
    )[0]


def event_probabilities(
    field,
    thresholds,
    *,
    cell_x_km,
    cell_y_km: float,
    event_radius_km: float = 10.0,
    spread_radius_km: float = 25.0,
) -> list[np.ndarray]:
    """Come ``event_probability``, ma per piu' soglie in una passata sola.

    Delle quattro somme sul vicinato che servono, due non dipendono dalla
    soglia: il conteggio dei punti noti entro il raggio dell'evento e quello
    entro il raggio di dispersione.  Dipendono solo da dove il campo esiste,
    che e' lo stesso per tutte le soglie.  Calcolarle una volta dimezza il
    lavoro, e con cinque soglie per scadenza la differenza si sente.
    """
    values = np.asarray(field, dtype=float)
    if values.ndim != 2:
        raise ValueError("il campo deve essere una griglia 2D")
    if float(event_radius_km) <= 0.0 or float(spread_radius_km) <= 0.0:
        raise ValueError("entrambi i raggi devono essere positivi")

    height = values.shape[0]
    cells_x = np.asarray(cell_x_km, dtype=float)
    if cells_x.ndim == 0:
        cells_x = np.full(height, float(cells_x))

    valid = np.isfinite(values)
    valid_float = valid.astype(float)

    # Parte comune: dove il campo e' noto. Non dipende dalla soglia.
    known = _neighbourhood_sums(
        valid_float, event_radius_km, cells_x, float(cell_y_km)
    )
    reachable = known > 0.0
    counts = _neighbourhood_sums(
        reachable.astype(float), spread_radius_km, cells_x, float(cell_y_km)
    )

    results = []
    for threshold in thresholds:
        reaches = np.where(valid, values >= float(threshold), False).astype(float)
        # Stadio 1: l'evento entro event_radius_km. Su un indicatore binario il
        # massimo sul disco e' "la somma e' maggiore di zero", quindi la
        # dilatazione riusa la stessa somma senza un secondo macchinario.
        nearby = _neighbourhood_sums(
            reaches, event_radius_km, cells_x, float(cell_y_km)
        )
        happened = np.where(reachable, nearby > 0.0, False).astype(float)
        # Stadio 2: quanto e' incerta la posizione.
        hits = _neighbourhood_sums(
            happened, spread_radius_km, cells_x, float(cell_y_km)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            probability = hits / counts
        results.append(np.where(counts > 0.0, probability, np.nan))
    return results
