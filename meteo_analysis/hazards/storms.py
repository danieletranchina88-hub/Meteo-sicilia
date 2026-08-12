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


def _normalise_cin(cin) -> np.ndarray:
    """CIN nella convenzione firmata negativa, preservando i dati mancanti."""
    values = _array(cin)
    valid = np.isfinite(values) & (values > -900.0)
    return np.where(valid, -np.abs(values), np.nan)


def _ramp(values, lower: float, upper: float) -> np.ndarray:
    """Rampa continua 0-1; le soglie non diventano cancelli binari."""
    array = _array(values)
    if upper <= lower:
        raise ValueError("il limite superiore deve essere maggiore dell'inferiore")
    return np.where(
        np.isfinite(array),
        np.clip((array - float(lower)) / (float(upper) - float(lower)), 0.0, 1.0),
        np.nan,
    )


def _available_mean(parts: list[tuple[np.ndarray | None, float]], shape) -> np.ndarray:
    """Media pesata dei soli indizi disponibili, NaN se non ce n'e' nessuno."""
    numerator = np.zeros(shape, dtype=float)
    denominator = np.zeros(shape, dtype=float)
    for values, weight in parts:
        if values is None or weight <= 0.0:
            continue
        array = np.broadcast_to(_array(values), shape)
        valid = np.isfinite(array)
        numerator += np.where(valid, array * float(weight), 0.0)
        denominator += valid.astype(float) * float(weight)
    return np.divide(
        numerator,
        denominator,
        out=np.full(shape, np.nan, dtype=float),
        where=denominator > 0.0,
    )


def _available_max(parts: list[np.ndarray | None], shape) -> np.ndarray:
    """Massimo fra meccanismi alternativi senza trasformare NaN in zero."""
    available = []
    for values in parts:
        if values is not None:
            available.append(np.broadcast_to(_array(values), shape))
    if not available:
        return np.full(shape, np.nan, dtype=float)
    stack = np.stack(available, axis=0)
    missing = np.all(~np.isfinite(stack), axis=0)
    maximum = np.max(np.where(np.isfinite(stack), stack, -np.inf), axis=0)
    return np.where(missing, np.nan, maximum)


def intelligent_storm_probability(
    lightning_potential,
    cape_ml,
    cin,
    trigger,
    cell_km: float,
    *,
    updraft_ms=None,
    cape_mu=None,
    surface_rh=None,
    mid_level_rh=None,
    cloud_base_m=None,
    omega_700=None,
    front_distance_km=None,
    shear=None,
    helicity=None,
    previous_lightning_potential=None,
    next_lightning_potential=None,
    lpi_threshold: float = 1.0,
    event_radius_km: float = 10.0,
    smoothing_radius_km: float = 25.0,
) -> dict[str, np.ndarray]:
    """Fonde prove fisiche indipendenti per l'ipotesi "temporale entro 10 km".

    La probabilita' di vicinato dell'LPI resta l'osservazione modellistica
    primaria, ma non decide piu' da sola. L'algoritmo cerca:

    * un segnale diretto: LPI e corrente ascensionale convettiva risolta;
    * un ambiente capace di sostenerlo: CAPE, CIN e umidita' della colonna;
    * un meccanismo capace di innescarlo: terreno/brezza/convergenza, salita
      sinottica o fronte;
    * corroborazione indipendente: persistenza a +/-1 ora e organizzazione.

    Gli indizi alternativi usano il massimo (un fronte non si somma a una
    brezza), mentre gli ingredienti necessari si moltiplicano. Un nucleo gia'
    risolto da LPI e updraft puo' sopravvivere a CAPE locale bassa: dentro un
    temporale maturo l'energia e' gia' stata consumata. Al contrario, CAPE o
    shear da soli non possono produrre un'alta probabilita'.

    Il risultato e' una diagnostica deterministica fisicamente vincolata, non
    una probabilita' calibrata contro una rete di fulminazioni. Oltre alla
    stima restituisce tutti i gruppi di evidenza per rendere la decisione
    verificabile sulla mappa e nei controlli qualita'.
    """
    lpi = _array(lightning_potential)
    if lpi.ndim != 2:
        raise ValueError("LPI deve essere una griglia 2D")
    shape = lpi.shape

    lpi_probability = storm_probability(
        lpi,
        threshold=lpi_threshold,
        event_radius_km=event_radius_km,
        smoothing_radius_km=smoothing_radius_km,
        cell_km=cell_km,
    )

    updraft_probability = None
    if updraft_ms is not None:
        updraft = np.broadcast_to(_array(updraft_ms), shape)
        updraft_probability = storm_probability(
            updraft,
            threshold=1.0,
            event_radius_km=event_radius_km,
            smoothing_radius_km=smoothing_radius_km,
            cell_km=cell_km,
        )

    # LPI e updraft sono due osservazioni modellistiche diverse dello stesso
    # processo. Il noisy-OR evita la somma lineare e limita il contributo
    # dell'updraft: salita forte senza elettrificazione puo' essere solo nube.
    direct = np.clip(lpi_probability, 0.0, 1.0)
    if updraft_probability is not None:
        direct = 1.0 - (1.0 - direct) * (
            1.0 - 0.72 * np.nan_to_num(updraft_probability, nan=0.0)
        )

    cape_ml_array = (
        np.broadcast_to(_array(cape_ml), shape)
        if cape_ml is not None else np.full(shape, np.nan)
    )
    cape_mu_array = (
        np.broadcast_to(_array(cape_mu), shape)
        if cape_mu is not None else np.full(shape, np.nan)
    )
    # La particella piu' instabile permette di riconoscere la convezione
    # elevata. Il fattore 0,9 impedisce a un singolo massimo molto sottile di
    # dominare completamente lo strato mescolato.
    cape_candidates = np.stack((cape_ml_array, 0.9 * cape_mu_array), axis=0)
    cape_missing = np.all(~np.isfinite(cape_candidates), axis=0)
    effective_cape = np.max(
        np.where(np.isfinite(cape_candidates), cape_candidates, -np.inf), axis=0
    )
    effective_cape = np.where(cape_missing, np.nan, effective_cape)
    cape_score = _ramp(effective_cape, 100.0, 1_600.0)

    inhibition = (
        _normalise_cin(np.broadcast_to(_array(cin), shape))
        if cin is not None else np.full(shape, np.nan)
    )
    cin_score = _ramp(inhibition, -250.0, -25.0)
    # Se MU-CAPE supera nettamente ML-CAPE, la particella utile puo' partire
    # sopra il coperchio superficiale. Non si ignora il CIN: se ne riduce solo
    # la capacita' di veto nella quota in cui non descrive piu' la particella.
    elevated = _ramp(cape_mu_array - cape_ml_array, 250.0, 1_000.0)
    effective_cin_score = np.where(
        np.isfinite(cin_score),
        np.maximum(cin_score, 0.45 * np.nan_to_num(elevated, nan=0.0)),
        np.where(np.isfinite(elevated), 0.45 * elevated, np.nan),
    )
    instability = np.where(
        np.isfinite(cape_score) & np.isfinite(effective_cin_score),
        cape_score * effective_cin_score,
        np.nan,
    )

    surface_moisture = (
        _ramp(np.broadcast_to(_array(surface_rh), shape), 40.0, 80.0)
        if surface_rh is not None else None
    )
    middle_moisture = (
        _ramp(np.broadcast_to(_array(mid_level_rh), shape), 25.0, 70.0)
        if mid_level_rh is not None else None
    )
    cloud_moisture = None
    if cloud_base_m is not None:
        base = np.broadcast_to(_array(cloud_base_m), shape)
        cloud_moisture = 1.0 - _ramp(base, 800.0, 3_500.0)
    moisture = _available_mean(
        [
            (surface_moisture, 0.25),
            (middle_moisture, 0.55),
            (cloud_moisture, 0.20),
        ],
        shape,
    )

    trigger_score = (
        np.clip(np.broadcast_to(_array(trigger), shape), 0.0, 1.0)
        if trigger is not None else None
    )
    ascent_score = None
    if omega_700 is not None:
        omega = np.broadcast_to(_array(omega_700), shape)
        ascent_score = _ramp(-omega, 0.02, 0.30) * 0.85
    front_score = None
    if front_distance_km is not None:
        distance = np.broadcast_to(_array(front_distance_km), shape)
        front_score = np.where(
            np.isfinite(distance),
            0.70 * np.exp(-np.maximum(distance, 0.0) / 70.0),
            np.nan,
        )
    lift = _available_max([trigger_score, ascent_score, front_score], shape)

    shear_score = (
        _ramp(np.broadcast_to(_array(shear), shape), 8.0, 25.0)
        if shear is not None else None
    )
    helicity_score = (
        _ramp(np.abs(np.broadcast_to(_array(helicity), shape)), 25.0, 150.0)
        if helicity is not None else None
    )
    organisation = _available_mean(
        [(shear_score, 0.65), (helicity_score, 0.35)], shape
    )

    temporal_parts = []
    for neighbouring_lpi in (
        previous_lightning_potential,
        next_lightning_potential,
    ):
        if neighbouring_lpi is None:
            continue
        temporal_parts.append(
            storm_probability(
                neighbouring_lpi,
                threshold=lpi_threshold,
                event_radius_km=event_radius_km,
                smoothing_radius_km=smoothing_radius_km,
                cell_km=cell_km,
            )
        )
    temporal = _available_max(temporal_parts, shape)

    # Gli ingredienti necessari interagiscono, non si accumulano. Umidita' e
    # lift mancanti sono neutrali ma riducono la confidenza separatamente.
    moisture_for_environment = np.where(np.isfinite(moisture), moisture, 0.55)
    lift_for_environment = np.where(np.isfinite(lift), lift, 0.20)
    organisation_for_environment = np.where(
        np.isfinite(organisation), organisation, 0.0
    )
    environment = (
        np.nan_to_num(instability, nan=0.0)
        * (0.50 + 0.50 * moisture_for_environment)
        * (0.12 + 0.88 * lift_for_environment)
        * (0.90 + 0.10 * organisation_for_environment)
    )
    environment = np.clip(environment, 0.0, 1.0)

    # Corroborazione: ogni voce e' una prova indipendente, non un doppio
    # conteggio dello stesso indice. La base 0,68 preserva un LPI esplicito
    # anche se il temporale maturo ha gia' consumato la CAPE locale.
    corroboration = _available_mean(
        [
            (environment, 0.35),
            (updraft_probability, 0.30),
            (temporal, 0.20),
            (lift, 0.15),
        ],
        shape,
    )
    corroboration = np.nan_to_num(corroboration, nan=0.0)
    direct_component = direct * (0.68 + 0.32 * corroboration)

    # Senza un temporale esplicitamente risolto, l'ambiente puo' esprimere
    # soltanto possibilita' d'innesco: non supera da solo il 42%.
    environment_component = 0.42 * environment
    probability = 1.0 - (1.0 - direct_component) * (1.0 - environment_component)
    probability += (
        (1.0 - probability)
        * 0.06
        * organisation_for_environment
        * np.maximum(direct, environment)
    )

    # Spiegazioni concorrenti per un falso segnale: LPI isolato nel tempo,
    # nessuna corrente risolta e ambiente ostile. La penalita' e' continua e
    # non spegne un nucleo che abbia almeno una corroborazione forte.
    no_core = (
        1.0 - np.nan_to_num(updraft_probability, nan=0.0)
        if updraft_probability is not None else np.full(shape, 0.5)
    )
    no_time = (
        1.0 - np.nan_to_num(temporal, nan=0.0)
        if temporal_parts else np.full(shape, 0.5)
    )
    hostile_environment = 1.0 - environment
    contradiction = np.clip(
        direct * (0.40 * no_core + 0.25 * no_time + 0.35 * hostile_environment),
        0.0,
        1.0,
    )
    probability *= 1.0 - 0.22 * contradiction

    # La fascia alta richiede supporto indipendente, ma senza un cancello
    # tutto-o-niente: il tetto cresce continuamente con la prova migliore.
    independent_support = _available_max(
        [environment, updraft_probability, temporal], shape
    )
    independent_support = np.nan_to_num(independent_support, nan=0.0)
    probability = np.minimum(probability, 0.70 + 0.25 * independent_support)

    direct_valid = np.isfinite(lpi_probability)
    thermodynamic_valid = np.isfinite(instability)
    moisture_valid = np.isfinite(moisture)
    lift_valid = np.isfinite(lift)
    temporal_valid = np.isfinite(temporal)
    organisation_valid = np.isfinite(organisation)
    core_valid = (
        np.isfinite(updraft_probability)
        if updraft_probability is not None else np.zeros(shape, dtype=bool)
    )
    coverage = (
        0.24 * direct_valid
        + 0.22 * thermodynamic_valid
        + 0.14 * moisture_valid
        + 0.16 * lift_valid
        + 0.10 * temporal_valid
        + 0.06 * organisation_valid
        + 0.08 * core_valid
    )
    agreement = np.maximum(
        1.0 - np.abs(direct - environment),
        np.maximum(
            np.nan_to_num(temporal, nan=0.0),
            np.nan_to_num(updraft_probability, nan=0.0)
            if updraft_probability is not None else 0.0,
        ),
    )
    confidence = np.clip(0.58 * coverage + 0.42 * agreement - 0.18 * contradiction, 0.0, 1.0)

    valid = direct_valid | thermodynamic_valid
    probability = np.where(valid, np.clip(probability, 0.0, 0.95), np.nan)
    return {
        "probability": probability,
        "direct": np.where(valid, np.clip(direct, 0.0, 1.0), np.nan),
        "environment": np.where(valid, environment, np.nan),
        "instability": np.where(valid, instability, np.nan),
        "moisture": np.where(valid, moisture, np.nan),
        "lift": np.where(valid, lift, np.nan),
        "temporal": np.where(valid, temporal, np.nan),
        "organisation": np.where(valid, organisation, np.nan),
        "contradiction": np.where(valid, contradiction, np.nan),
        "confidence": np.where(valid, confidence, np.nan),
    }


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


def upslope_flow(u_wind, v_wind, orography, latitudes, longitudes) -> np.ndarray:
    """Velocita' di risalita forzata dal rilievo (m/s), positiva in salita.

    Il vento che incontra un pendio non puo' attraversarlo: sale.  La
    componente verticale e' la proiezione del vento orizzontale sul gradiente
    del terreno, ``w = V . grad(h)``.

    E' l'innesco dominante delle giornate estive senza fronti, ed e' il motivo
    per cui alle 13-14 i cumuli si accendono prima sull'Appennino e sulle Alpi
    e solo dopo scendono in pianura: bastano pochi decimi di metro al secondo
    per far superare a una particella il livello di libera convezione, se la
    CAPE c'e' gia'.
    """
    u = _array(u_wind)
    v = _array(v_wind)
    height = _array(orography)
    latitude = _array(latitudes).squeeze()
    longitude = _array(longitudes).squeeze()
    if height.ndim != 2:
        raise ValueError("l'orografia deve essere una griglia 2D")
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("coordinate 1D attese")
    if height.shape != (latitude.size, longitude.size):
        raise ValueError("orografia e coordinate non compatibili")

    # Metri veri sulla griglia: il passo in longitudine si accorcia salendo
    # di latitudine, e ignorarlo darebbe pendenze sbagliate del 25% sulle Alpi.
    metres_per_degree = 111_320.0
    x_metres = (
        np.deg2rad(longitude - longitude[0])[None, :]
        * 6_371_000.0
        * np.cos(np.deg2rad(latitude))[:, None]
    )
    y_metres = np.deg2rad(latitude - latitude[0]) * 6_371_000.0
    del metres_per_degree

    dh_dj = np.gradient(height, axis=1)
    dx_dj = np.gradient(x_metres, axis=1)
    dh_dx = np.divide(
        dh_dj, dx_dj, out=np.full_like(dh_dj, np.nan), where=np.abs(dx_dj) > 1.0
    )
    dh_dy = np.gradient(height, y_metres, axis=0)

    u, v = np.broadcast_arrays(u, v)
    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(dh_dx) & np.isfinite(dh_dy)
    with np.errstate(invalid="ignore"):
        vertical = u * dh_dx + v * dh_dy
    return np.where(valid, vertical, np.nan)


def coast_distance_km(land_fraction, latitudes, longitudes, limit_km: float = 120.0):
    """Distanza dalla costa (km), positiva sulla terra e negativa sul mare.

    Serve a isolare la brezza: la sua linea di convergenza vive in una fascia
    costiera, mentre una convergenza sinottica puo' stare ovunque.  Il segno
    distingue i due lati senza dover portare in giro una seconda maschera.

    Il calcolo e' una dilatazione progressiva della costa, fermata a
    ``limit_km``: oltre quella distanza il valore esatto non serve a nessuno e
    continuare costerebbe soltanto.
    """
    land = _array(land_fraction)
    latitude = _array(latitudes).squeeze()
    longitude = _array(longitudes).squeeze()
    if land.ndim != 2 or land.shape != (latitude.size, longitude.size):
        raise ValueError("maschera terra-mare non compatibile con le coordinate")

    is_land = land >= 0.5
    if is_land.all() or not is_land.any():
        # Nessuna costa nel riquadro: non c'e' brezza da isolare.
        return np.full(land.shape, np.nan)

    mean_latitude = float(np.nanmean(latitude))
    cell_y_km = abs(float(latitude[1] - latitude[0])) * 111.32 if latitude.size > 1 else 1.0
    cell_x_km = (
        abs(float(longitude[1] - longitude[0]))
        * 111.32
        * np.cos(np.deg2rad(mean_latitude))
        if longitude.size > 1 else 1.0
    )
    cell_km = float(np.mean([cell_x_km, cell_y_km]))
    steps = max(1, int(round(limit_km / max(cell_km, 0.1))))

    distance = np.full(land.shape, np.inf)
    # La costa e' il confine: punti che hanno un vicino dell'altro tipo.
    neighbour_land = np.zeros(land.shape, dtype=bool)
    neighbour_sea = np.zeros(land.shape, dtype=bool)
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        rolled = np.roll(is_land, shift, axis=axis)
        neighbour_land |= rolled
        neighbour_sea |= ~rolled
    coast = (is_land & neighbour_sea) | (~is_land & neighbour_land)
    distance[coast] = 0.0

    frontier = coast.copy()
    for step in range(1, steps + 1):
        grown = np.zeros(land.shape, dtype=bool)
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            grown |= np.roll(frontier, shift, axis=axis)
        grown &= ~np.isfinite(distance)
        if not grown.any():
            break
        distance[grown] = step * cell_km
        frontier = grown
    distance[~np.isfinite(distance)] = steps * cell_km
    return np.where(is_land, distance, -distance)


def sea_breeze_lift(
    convergence,
    u_wind,
    v_wind,
    land_fraction,
    latitudes,
    longitudes,
    inland_reach_km: float = 60.0,
) -> np.ndarray:
    """Parte della convergenza attribuibile alla brezza di mare (s-1).

    Tre condizioni insieme, e servono tutte e tre.  Il punto deve stare sulla
    terra entro la portata della brezza; il vento deve venire dal mare, non
    andarci; e la convergenza deve esserci davvero.  Con una sola delle tre
    si scambierebbe per brezza una qualunque linea di convergenza costiera,
    per esempio quella di un fronte che attraversa la costa.

    In Sicilia e lungo l'Adriatico e' l'innesco dominante delle giornate
    estive: il fronte di brezza avanza per decine di chilometri e la sua linea
    di convergenza e' netta.
    """
    values = _array(convergence)
    distance = coast_distance_km(land_fraction, latitudes, longitudes)
    if not np.isfinite(distance).any():
        return np.full(values.shape, np.nan)

    # Verso il mare: gradiente della distanza con segno, che sulla terra punta
    # verso l'interno. Il vento e' "dal mare" se ha componente concorde.
    latitude = _array(latitudes).squeeze()
    longitude = _array(longitudes).squeeze()
    y_metres = np.deg2rad(latitude - latitude[0]) * 6_371_000.0
    x_metres = (
        np.deg2rad(longitude - longitude[0])[None, :]
        * 6_371_000.0
        * np.cos(np.deg2rad(latitude))[:, None]
    )
    dd_dj = np.gradient(distance, axis=1)
    dx_dj = np.gradient(x_metres, axis=1)
    inland_x = np.divide(
        dd_dj, dx_dj, out=np.full_like(dd_dj, np.nan), where=np.abs(dx_dj) > 1.0
    )
    inland_y = np.gradient(distance, y_metres, axis=0)

    u = _array(u_wind)
    v = _array(v_wind)
    u, v = np.broadcast_arrays(u, v)
    with np.errstate(invalid="ignore"):
        onshore = u * inland_x + v * inland_y

    # La terra si riconosce dalla maschera, non dal segno della distanza: sui
    # punti di mare esattamente sulla linea di costa la distanza vale -0.0, e
    # in virgola mobile -0.0 >= 0 e' vero. Con quel confronto una fascia di
    # celle marine risultava terra e prendeva brezza.
    on_land = _array(land_fraction) >= 0.5
    within_reach = on_land & (np.abs(distance) <= float(inland_reach_km))
    from_sea = np.isfinite(onshore) & (onshore > 0.0)
    valid = np.isfinite(values)
    breeze = np.where(
        valid & within_reach & from_sea, np.maximum(values, 0.0), 0.0
    )
    return np.where(valid, breeze, np.nan)


def trigger_index(upslope_ms=None, sea_breeze=None, convergence=None) -> np.ndarray:
    """Indice 0-1 di quanto forte e' il sollevamento che puo' rompere il CIN.

    I tre meccanismi non si sommano: in un punto ne agisce uno, e conta il
    piu' forte.  Sommandoli, una convergenza debole diffusa su una collina
    diventerebbe un innesco inesistente.
    """
    parts = []
    if upslope_ms is not None:
        values = _array(upslope_ms)
        parts.append(np.clip(values / 0.6, 0.0, 1.0))
    if sea_breeze is not None:
        values = _array(sea_breeze)
        parts.append(np.clip(values / 2.0e-4, 0.0, 1.0))
    if convergence is not None:
        values = _array(convergence)
        parts.append(np.clip(values / 3.0e-4, 0.0, 1.0))
    if not parts:
        return np.array([], dtype=float)
    stack = np.stack(np.broadcast_arrays(*parts), axis=0)
    missing = np.all(~np.isfinite(stack), axis=0)
    with np.errstate(invalid="ignore"):
        strongest = np.max(np.where(np.isfinite(stack), stack, -np.inf), axis=0)
    return np.where(missing, np.nan, np.clip(strongest, 0.0, 1.0))


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


def summarize_storms(
    probability_percent,
    updraft_ms=None,
    confidence_percent=None,
    contradiction_percent=None,
) -> dict:
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
            "meanConfidence": None,
            "areaHighProbabilityLowConfidencePct": None,
            "meanContradiction": None,
        }
    summary = {
        "status": "available",
        "maximum": round(float(np.max(finite)), 1),
        "areaAbove20Pct": round(float(np.mean(finite >= 20.0) * 100.0), 3),
        "areaAbove50Pct": round(float(np.mean(finite >= 50.0) * 100.0), 3),
        "maxUpdraft": None,
        "meanConfidence": None,
        "areaHighProbabilityLowConfidencePct": None,
        "meanContradiction": None,
    }
    if updraft_ms is not None:
        updraft = _array(updraft_ms)
        finite_updraft = updraft[np.isfinite(updraft)]
        if finite_updraft.size:
            summary["maxUpdraft"] = round(float(np.max(finite_updraft)), 1)
    if confidence_percent is not None:
        confidence = _array(confidence_percent)
        confidence, aligned_probability = np.broadcast_arrays(confidence, values)
        valid_confidence = np.isfinite(confidence) & np.isfinite(aligned_probability)
        if valid_confidence.any():
            summary["meanConfidence"] = round(
                float(np.mean(confidence[valid_confidence])), 1
            )
            summary["areaHighProbabilityLowConfidencePct"] = round(
                float(
                    np.mean(
                        (aligned_probability[valid_confidence] >= 50.0)
                        & (confidence[valid_confidence] < 45.0)
                    )
                    * 100.0
                ),
                3,
            )
    if contradiction_percent is not None:
        contradiction = _array(contradiction_percent)
        finite_contradiction = contradiction[np.isfinite(contradiction)]
        if finite_contradiction.size:
            summary["meanContradiction"] = round(
                float(np.mean(finite_contradiction)), 1
            )
    return summary
