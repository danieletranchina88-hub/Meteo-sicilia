"""Diagnostica dei temporali sui campi nativi ICON-2I.

Il modello gira a 2,2 km e risolve esplicitamente le celle convettive: qui non
si ricostruisce nulla che il modello sappia gia' dire, si traduce.  Ogni
funzione lavora su un solo anello della catena che porta dal riscaldamento del
suolo al fulmine, e restituisce NaN dove manca un ingrediente invece di
sostituirlo con una costante climatologica.
"""

from __future__ import annotations

import numpy as np

# Accelerazione di gravita' e densita' tipica dell'aria ai livelli isobarici
# usati.  Servono solo per convertire omega (Pa/s) in velocita' verticale
# (m/s): l'errore sulla densita' e' di pochi punti percentuali e non cambia la
# lettura, mentre usare una densita' unica falserebbe il confronto fra livelli.
GRAVITY = 9.80665
AIR_DENSITY_BY_LEVEL = {
    1000.0: 1.20,
    925.0: 1.13,
    850.0: 1.06,
    700.0: 0.91,
    500.0: 0.69,
    250.0: 0.40,
}

# Frazione del limite di particella effettivamente raggiunta da una corrente
# ascensionale reale.  Misurata sul run ICON-2I del 12 agosto 2026 00 UTC:
# nei 4233 punti con w > 5 m/s a 500 hPa il rapporto fra velocita' risolta dal
# modello e sqrt(2*CAPE) ha mediana 0,26 e novantesimo percentile 0,48.  La
# differenza e' l'entrainment di aria secca e il peso dell'acqua trasportata.
PARCEL_EFFICIENCY = 0.45


def _array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def updraft_from_omega(omega_pa_s, level_hpa: float) -> np.ndarray:
    """Velocita' verticale (m/s, positiva verso l'alto) da omega (Pa/s).

    ``omega`` e' la derivata della pressione seguendo la particella, quindi e'
    negativa quando l'aria sale.  La relazione idrostatica da w = -omega/(rho*g).
    """
    omega = _array(omega_pa_s)
    density = AIR_DENSITY_BY_LEVEL.get(float(level_hpa))
    if density is None:
        raise ValueError(f"densita' non nota per il livello {level_hpa} hPa")
    with np.errstate(invalid="ignore"):
        return -omega / (density * GRAVITY)


def strongest_updraft(omega_by_level: dict[float, np.ndarray]) -> np.ndarray:
    """La corrente ascensionale piu' forte fra i livelli disponibili.

    Una cella convettiva accelera salendo finche' resta piu' calda
    dell'ambiente, quindi il suo massimo non sta sempre allo stesso livello: a
    850 hPa una cella giovane e' gia' veloce mentre una matura ha il nucleo
    molto piu' in alto.  Prendere il massimo sulla colonna disponibile evita di
    perdere la cella solo perche' si e' guardato il livello sbagliato.
    """
    fields = [
        updraft_from_omega(omega, level)
        for level, omega in omega_by_level.items()
        if omega is not None
    ]
    if not fields:
        return np.array([], dtype=float)
    stack = np.stack(np.broadcast_arrays(*fields), axis=0)
    missing = np.all(~np.isfinite(stack), axis=0)
    with np.errstate(invalid="ignore"):
        strongest = np.nanmax(np.where(np.isfinite(stack), stack, -np.inf), axis=0)
    return np.where(missing, np.nan, strongest)


def potential_updraft(cape, efficiency: float = PARCEL_EFFICIENCY) -> np.ndarray:
    """Velocita' verticale possibile (m/s) se la convezione si innesca.

    La teoria della particella da w = sqrt(2*CAPE) come limite superiore.  Il
    valore restituito e' gia' ridotto della frazione misurata sul modello, per
    non promettere velocita' che nessun temporale reale raggiunge.

    E' un numero diverso dalla corrente risolta: questo dice quanto potrebbe
    essere violento un temporale *se* nasce, quella dice dove il modello lo sta
    gia' mettendo.
    """
    values = _array(cape)
    with np.errstate(invalid="ignore"):
        return np.where(
            np.isfinite(values) & (values > 0.0),
            float(efficiency) * np.sqrt(2.0 * np.maximum(values, 0.0)),
            np.where(np.isfinite(values), 0.0, np.nan),
        )


def bulk_shear(shear_u, shear_v) -> np.ndarray:
    """Modulo dello shear di massa (m/s) dalle sue componenti.

    ICON-2I pubblica WSHEAR_U e WSHEAR_V gia' calcolati sullo strato 0-6 km
    usando i livelli veri del modello.  E' molto meglio della differenza fra il
    vento a 500 hPa e quello a 10 m, che approssima lo stesso strato ma ignora
    tutto quello che c'e' in mezzo.
    """
    u = _array(shear_u)
    v = _array(shear_v)
    u, v = np.broadcast_arrays(u, v)
    valid = np.isfinite(u) & np.isfinite(v)
    with np.errstate(invalid="ignore"):
        return np.where(valid, np.hypot(u, v), np.nan)


def lifting_condensation_level(temperature_2m_c, dewpoint_2m_c) -> np.ndarray:
    """Quota della base delle nubi (m sul suolo) con la formula di Lawrence.

    ``z = 125 * (T - Td)`` approssima l'LCL entro poche decine di metri per
    scarti fino a 20 K, che coprono ogni situazione convettiva reale.

    La quota conta piu' di quanto sembri: una base bassa produce temporali
    efficienti a fare pioggia, una base alta lascia evaporare gran parte della
    precipitazione durante la caduta, e quell'evaporazione raffredda l'aria
    che poi arriva al suolo come raffica.
    """
    temperature = _array(temperature_2m_c)
    dewpoint = _array(dewpoint_2m_c)
    temperature, dewpoint = np.broadcast_arrays(temperature, dewpoint)
    spread = temperature - dewpoint
    valid = np.isfinite(spread)
    return np.where(valid, np.maximum(125.0 * spread, 0.0), np.nan)


def neighbourhood_probability(
    field,
    threshold: float,
    radius_km: float,
    cell_km: float,
) -> np.ndarray:
    """Frazione di punti entro ``radius_km`` che supera ``threshold`` (0-1).

    E' il metodo standard per i modelli che risolvono la convezione (Theis et
    al. 2005; Schwartz e Sobash 2017), e serve a rispondere alla domanda giusta.
    Un modello a 2,2 km sa che ci sara' un temporale, non su quale paese: la
    posizione esatta di una cella non e' prevedibile, la probabilita' che una
    cella interessi una zona lo e'.

    Senza questo passaggio il campo sarebbe anche illeggibile: sul run reale
    solo lo 0,027% dei punti ha LPI sopra 1 J/kg, cioe' poche decine di celle su
    mezzo milione.

    I punti mancanti non contano ne' a favore ne' contro: la frazione e'
    calcolata sui soli punti validi del vicinato.
    """
    values = _array(field)
    if values.ndim != 2:
        raise ValueError("il campo deve essere una griglia 2D")
    if cell_km <= 0 or radius_km <= 0:
        raise ValueError("raggio e passo della griglia devono essere positivi")

    valid = np.isfinite(values)
    exceeds = np.where(valid, values > float(threshold), False)

    radius_cells = int(round(float(radius_km) / float(cell_km)))
    if radius_cells < 1:
        return np.where(valid, exceeds.astype(float), np.nan)

    kernel = _disc_kernel(radius_cells)
    hits = _convolve_same(exceeds.astype(float), kernel)
    counts = _convolve_same(valid.astype(float), kernel)
    with np.errstate(divide="ignore", invalid="ignore"):
        probability = hits / counts
    return np.where(counts > 0.0, probability, np.nan)


def storm_probability(
    lightning_potential,
    threshold: float = 1.0,
    event_radius_km: float = 10.0,
    smoothing_radius_km: float = 25.0,
    cell_km: float = 2.2,
) -> np.ndarray:
    """Probabilita' (0-1) che un temporale interessi la zona.

    Due passaggi, e la differenza fra i due non e' un dettaglio.

    Primo: l'evento non e' "fulmini in questo punto" ma "un temporale entro
    ``event_radius_km``".  Una cella convettiva e' larga pochi chilometri: se
    l'evento restasse puntuale, la frazione di punti colpiti resterebbe bassa
    anche in piena attivita' -- misurato sul run reale, il massimo si fermava
    al 26% in un'ora con 1236 celle temporalesche accese.  Quel numero non e'
    sbagliato, risponde a un'altra domanda ("che probabilita' ho di prendere
    il fulmine proprio qui"), che non e' quella che si fa chi guarda la mappa.

    Secondo: la frazione di punti dell'intorno in cui il primo passaggio e'
    vero.  Serve perche' il modello sa che ci sara' un temporale ma non su
    quale paese: sfumare su ``smoothing_radius_km`` e' il modo di dire
    "da queste parti", che e' l'unica cosa che i dati permettono di dire.

    Attenzione a cosa significa il numero: e' la frequenza spaziale con cui il
    modello mette temporali attorno al punto, non una probabilita' verificata
    contro le fulminazioni osservate.  Per calibrarla servirebbe un archivio
    di osservazioni che non abbiamo.
    """
    presence = neighbourhood_probability(
        lightning_potential, threshold, event_radius_km, cell_km
    )
    affected = np.where(np.isfinite(presence), presence > 0.0, np.nan)
    return neighbourhood_probability(affected, 0.5, smoothing_radius_km, cell_km)


def _disc_kernel(radius_cells: int) -> np.ndarray:
    """Maschera circolare: il vicinato e' un cerchio, non un quadrato.

    Un quadrato darebbe piu' peso alle diagonali e produrrebbe macchie di
    probabilita' con gli angoli, un artefatto che si nota subito sulla mappa.
    """
    span = np.arange(-radius_cells, radius_cells + 1, dtype=float)
    dy, dx = np.meshgrid(span, span, indexing="ij")
    return (dx * dx + dy * dy <= radius_cells * radius_cells).astype(float)


def _convolve_same(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Somma sul vicinato circolare, bordi replicati, via immagine integrale.

    Il disco e' scomposto in righe: ogni riga e' un segmento contiguo, e la
    somma di un segmento si legge in tempo costante da un'immagine integrale.
    Il costo cresce con il raggio in righe, non in celle: con raggio 25 km su
    griglia da 2,2 km il disco contiene circa 400 punti ma bastano 23 letture
    per punto invece di 400.
    """
    radius = (kernel.shape[0] - 1) // 2
    height, width = values.shape
    padded = np.pad(values, radius, mode="edge")

    # integral[i, j] = somma di padded[:i, :j]; la riga e la colonna di zeri in
    # testa evitano di trattare a parte i bordi.
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=float)
    integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)

    result = np.zeros((height, width), dtype=float)
    for row_index in range(kernel.shape[0]):
        columns = np.nonzero(kernel[row_index])[0]
        if not columns.size:
            continue
        # Il pixel di uscita (y, x) sta in padded a (y + radius, x + radius);
        # la riga del disco lo sposta di (row_index - radius).
        top = row_index
        first = int(columns[0])
        last = int(columns[-1]) + 1
        result += (
            integral[top + 1:top + 1 + height, last:last + width]
            - integral[top + 1:top + 1 + height, first:first + width]
            - integral[top:top + height, last:last + width]
            + integral[top:top + height, first:first + width]
        )
    return result


def storm_mode(cape, shear) -> np.ndarray:
    """Modo del temporale atteso, come codice intero.

    ``0`` nessuna convezione profonda, ``1`` cella singola, ``2`` multicella o
    sistema organizzato, ``3`` supercella possibile.

    E' la lettura classica del piano CAPE-shear.  Senza shear la pioggia cade
    dentro la propria corrente ascendente e la spegne in una ventina di minuti;
    con vento che cambia con la quota ascendente e discendente si separano e la
    cella si autoalimenta per ore.  Lo shear da solo non fa nulla: serve
    energia, ed e' per questo che ogni soglia di shear e' condizionata a una
    soglia di CAPE.
    """
    energy = _array(cape)
    wind = _array(shear)
    energy, wind = np.broadcast_arrays(energy, wind)
    valid = np.isfinite(energy) & np.isfinite(wind)

    mode = np.zeros(energy.shape, dtype=float)
    deep = energy >= 300.0
    mode = np.where(deep, 1.0, mode)
    mode = np.where(deep & (energy >= 500.0) & (wind >= 10.0), 2.0, mode)
    mode = np.where(deep & (energy >= 1000.0) & (wind >= 20.0), 3.0, mode)
    return np.where(valid, mode, np.nan)


def hail_potential(graupel_mm, freezing_level_m, cape, shear) -> np.ndarray:
    """Indice di grandine 0-1 da graupel, zero termico, energia e shear.

    Il graupel prodotto dal modello e' il segnale diretto: e' il precursore
    della grandine nella microfisica.  Ma quello che arriva al suolo dipende da
    quanto fonde scendendo, quindi uno zero termico alto riduce il rischio, e
    da quanto a lungo il chicco resta sostenuto dalla corrente, che dipende da
    energia e organizzazione.
    """
    graupel = _array(graupel_mm)
    freezing = _array(freezing_level_m)
    energy = _array(cape)
    wind = _array(shear)
    graupel, freezing, energy, wind = np.broadcast_arrays(
        graupel, freezing, energy, wind
    )
    valid = np.isfinite(graupel) & np.isfinite(energy)

    graupel_score = np.clip(graupel / 2.0, 0.0, 1.0)
    energy_score = np.clip((energy - 500.0) / 2_000.0, 0.0, 1.0)
    shear_score = np.where(
        np.isfinite(wind), 0.6 + 0.4 * np.clip((wind - 10.0) / 15.0, 0.0, 1.0), 0.6
    )
    # Sopra i 4000 m di zero termico il chicco attraversa uno strato caldo
    # abbastanza spesso da fondere quasi del tutto.
    melting_score = np.where(
        np.isfinite(freezing), np.clip((4_200.0 - freezing) / 1_400.0, 0.15, 1.0), 1.0
    )
    score = graupel_score * (0.45 + 0.55 * energy_score) * shear_score * melting_score
    return np.where(valid, np.clip(score, 0.0, 1.0), np.nan)


def downburst_potential(cloud_base_m, gust_ms, cape) -> np.ndarray:
    """Indice di raffica discendente 0-1.

    Una base delle nubi alta significa uno strato profondo di aria sotto la
    nube in cui la pioggia evapora.  L'evaporazione raffredda, l'aria fredda e'
    piu' densa e precipita: e' il downburst.  La raffica massima prevista dal
    modello e' il secondo segnale, indipendente dal primo.
    """
    base = _array(cloud_base_m)
    gust = _array(gust_ms)
    energy = _array(cape)
    base, gust, energy = np.broadcast_arrays(base, gust, energy)
    valid = np.isfinite(gust) | (np.isfinite(base) & np.isfinite(energy))

    dry_layer = np.where(
        np.isfinite(base), np.clip((base - 1_500.0) / 2_000.0, 0.0, 1.0), 0.0
    )
    energy_score = np.where(
        np.isfinite(energy), np.clip((energy - 300.0) / 1_500.0, 0.0, 1.0), 0.0
    )
    gust_score = np.where(
        np.isfinite(gust), np.clip((gust - 15.0) / 20.0, 0.0, 1.0), 0.0
    )
    score = np.maximum(gust_score, dry_layer * energy_score)
    return np.where(valid, np.clip(score, 0.0, 1.0), np.nan)


def bowen_ratio(sensible_heat_flux, latent_heat_flux) -> np.ndarray:
    """Rapporto di Bowen dal flusso sensibile e latente al suolo.

    E' la ripartizione dell'energia solare fra scaldare l'aria e evaporare
    acqua, e dipende da quanta acqua c'e' nel terreno.  Suolo secco: rapporto
    alto, strato limite profondo, base delle nubi alta, celle isolate e
    violente.  Suolo umido: rapporto basso, piu' vapore, base bassa, temporali
    piu' diffusi e piovosi.

    ICON pubblica i flussi con il verso positivo verso il basso, quindi di
    giorno sono negativi: qui si lavora sui moduli, e il rapporto e' definito
    solo dove il flusso latente e' abbastanza grande da non far esplodere la
    divisione.
    """
    sensible = np.abs(_array(sensible_heat_flux))
    latent = np.abs(_array(latent_heat_flux))
    sensible, latent = np.broadcast_arrays(sensible, latent)
    valid = np.isfinite(sensible) & np.isfinite(latent) & (latent > 5.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sensible / latent
    return np.where(valid, np.clip(ratio, 0.0, 20.0), np.nan)


def coarsen(values, factor: int = 2, how: str = "mean") -> np.ndarray:
    """Riduce la griglia aggregando blocchi ``factor x factor``.

    Serve al peso dei file: a piena risoluzione la sola sezione temporali
    pesa 3,6 MB per scadenza, cioe' 260 MB per corsa, e chi la apre li
    scarica.  Dimezzando il passo si scende a un quarto.

    Il modo di aggregare non e' un dettaglio.  ``mean`` va bene per i campi
    gia' lisci -- una probabilita' smussata su 25 km non perde nulla a essere
    campionata ogni 4 km.  ``max`` serve ai campi sparsi e appuntiti: il
    nucleo di una cella convettiva occupa pochi punti, e mediarlo con i vicini
    sereni lo farebbe sparire proprio dove conta.  ``nearest`` e' per i campi
    categorici, dove una media non ha significato.
    """
    array = _array(values)
    if array.ndim != 2:
        raise ValueError("l'aggregazione richiede una griglia 2D")
    step = int(factor)
    if step < 2:
        return array
    height, width = array.shape
    pad_y = (-height) % step
    pad_x = (-width) % step
    if pad_y or pad_x:
        array = np.pad(array, ((0, pad_y), (0, pad_x)), mode="edge")
    blocks = array.reshape(
        array.shape[0] // step, step, array.shape[1] // step, step
    )
    if how == "nearest":
        return blocks[:, 0, :, 0]
    with np.errstate(invalid="ignore"):
        if how == "max":
            allnan = np.all(~np.isfinite(blocks), axis=(1, 3))
            filled = np.where(np.isfinite(blocks), blocks, -np.inf)
            result = np.max(filled, axis=(1, 3))
            return np.where(allnan, np.nan, result)
        if how == "mean":
            allnan = np.all(~np.isfinite(blocks), axis=(1, 3))
            summed = np.nansum(np.where(np.isfinite(blocks), blocks, 0.0), axis=(1, 3))
            counts = np.sum(np.isfinite(blocks), axis=(1, 3))
            result = np.divide(
                summed,
                counts,
                out=np.full(summed.shape, np.nan),
                where=counts > 0,
            )
            return np.where(allnan, np.nan, result)
    raise ValueError(f"modo di aggregazione sconosciuto: {how}")


def summarize_storms(probability_percent, updraft_ms=None) -> dict:
    """Statistiche robuste per il controllo qualita' e i prodotti testuali."""
    values = _array(probability_percent)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            "status": "unavailable",
            "maximum": None,
            "areaAbove20Pct": None,
            "areaAbove50Pct": None,
            "maxUpdraft": None,
        }
    summary = {
        "status": "available",
        "maximum": round(float(np.max(finite)), 1),
        "areaAbove20Pct": round(float(np.mean(finite >= 20.0) * 100.0), 3),
        "areaAbove50Pct": round(float(np.mean(finite >= 50.0) * 100.0), 3),
        "maxUpdraft": None,
    }
    if updraft_ms is not None:
        updraft = _array(updraft_ms)
        finite_updraft = updraft[np.isfinite(updraft)]
        if finite_updraft.size:
            summary["maxUpdraft"] = round(float(np.max(finite_updraft)), 1)
    return summary
