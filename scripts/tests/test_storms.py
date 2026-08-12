"""Test sintetici della diagnostica temporalesca."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from meteo_analysis.hazards.storms import (  # noqa: E402
    AIR_DENSITY_BY_LEVEL,
    bowen_ratio,
    coarsen,
    coast_distance_km,
    bulk_shear,
    downburst_potential,
    hail_potential,
    intelligent_storm_probability,
    lifting_condensation_level,
    neighbourhood_probability,
    potential_updraft,
    storm_mode,
    sea_breeze_lift,
    storm_probability,
    strongest_updraft,
    trigger_index,
    upslope_flow,
    summarize_storms,
    updraft_from_omega,
)


def test_omega_conversion_sign_and_magnitude():
    """Omega negativo = aria che sale, e il modulo torna con la statica."""
    w = updraft_from_omega(np.array([-100.0, 0.0, 50.0]), 500.0)
    assert w[0] > 0, "omega negativo deve dare salita"
    assert np.isclose(w[1], 0.0)
    assert w[2] < 0, "omega positivo deve dare discesa"
    # -100 Pa/s a 500 hPa: 100 / (0.69 * 9.80665)
    assert np.isclose(w[0], 100.0 / (AIR_DENSITY_BY_LEVEL[500.0] * 9.80665))
    # Lo stesso omega da una velocita' maggiore in alto, dove l'aria e' rarefatta.
    assert updraft_from_omega(np.array([-50.0]), 500.0)[0] > \
        updraft_from_omega(np.array([-50.0]), 850.0)[0]


def test_omega_unknown_level_is_refused():
    """Meglio un errore che una densita' inventata."""
    try:
        updraft_from_omega(np.zeros(3), 600.0)
    except ValueError:
        return
    raise AssertionError("un livello senza densita' nota deve dare errore")


def test_strongest_updraft_takes_the_column_maximum():
    """Il nucleo di una cella non sta sempre allo stesso livello."""
    omega = {
        850.0: np.array([[-10.0, 0.0]]),
        500.0: np.array([[-5.0, -60.0]]),
    }
    w = strongest_updraft(omega)
    # Primo punto: piu' forte in basso. Secondo: molto piu' forte in alto.
    assert w[0, 0] == updraft_from_omega(np.array([-10.0]), 850.0)[0]
    assert w[0, 1] == updraft_from_omega(np.array([-60.0]), 500.0)[0]


def test_strongest_updraft_keeps_missing_as_missing():
    omega = {850.0: np.array([np.nan]), 500.0: np.array([np.nan])}
    assert np.isnan(strongest_updraft(omega)[0])
    # Un livello valido basta a dare un risultato.
    omega = {850.0: np.array([np.nan]), 500.0: np.array([-20.0])}
    assert np.isfinite(strongest_updraft(omega)[0])


def test_potential_updraft_follows_parcel_theory():
    """w = k * sqrt(2 CAPE), gia' ridotta del rendimento reale."""
    w = potential_updraft(np.array([0.0, 1_000.0, 2_000.0]), efficiency=1.0)
    assert np.isclose(w[1], np.sqrt(2_000.0))
    assert np.isclose(w[2], np.sqrt(4_000.0))
    assert w[0] == 0.0
    # Il rendimento predefinito non deve promettere il limite teorico.
    ridotta = potential_updraft(np.array([1_000.0]))
    assert ridotta[0] < np.sqrt(2_000.0)
    assert 0.2 < ridotta[0] / np.sqrt(2_000.0) < 0.6
    assert np.isnan(potential_updraft(np.array([np.nan]))[0])


def test_bulk_shear_modulus():
    assert np.isclose(bulk_shear(np.array([3.0]), np.array([4.0]))[0], 5.0)
    assert np.isnan(bulk_shear(np.array([np.nan]), np.array([4.0]))[0])


def test_lcl_lawrence():
    """125 m per grado di scarto, mai negativa."""
    z = lifting_condensation_level(np.array([30.0, 20.0, 15.0]), np.array([20.0, 20.0, 18.0]))
    assert np.isclose(z[0], 1_250.0)
    assert np.isclose(z[1], 0.0)
    assert z[2] == 0.0, "uno scarto negativo non puo' dare una base sotto terra"


def test_neighbourhood_matches_brute_force():
    """La versione veloce deve dare esattamente la definizione."""
    rng = np.random.default_rng(20260812)
    field = rng.normal(size=(37, 41))
    radius_km, cell_km, threshold = 6.0, 2.0, 0.8
    fast = neighbourhood_probability(field, threshold, radius_km, cell_km)

    radius_cells = int(round(radius_km / cell_km))
    height, width = field.shape
    slow = np.zeros_like(field)
    for y in range(height):
        for x in range(width):
            hits = 0
            total = 0
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy > radius_cells * radius_cells:
                        continue
                    # bordi replicati, come nella versione veloce
                    yy = min(max(y + dy, 0), height - 1)
                    xx = min(max(x + dx, 0), width - 1)
                    total += 1
                    if field[yy, xx] > threshold:
                        hits += 1
            slow[y, x] = hits / total
    assert np.allclose(fast, slow), "la convoluzione veloce non e' la definizione"


def test_neighbourhood_spreads_a_single_cell():
    """Un solo punto acceso deve diventare una macchia leggibile."""
    field = np.zeros((41, 41))
    field[20, 20] = 10.0
    probability = neighbourhood_probability(field, 1.0, 10.0, 2.0)
    assert probability[20, 20] > 0.0
    assert probability[20, 24] > 0.0, "a 8 km deve esserci ancora segnale"
    assert probability[20, 34] == 0.0, "a 28 km il segnale deve essere finito"
    # Il massimo sta sul punto acceso e la probabilita' cala allontanandosi.
    assert probability[20, 20] >= probability[20, 22] >= probability[20, 24]


def test_neighbourhood_circular_not_square():
    """Un vicinato quadrato lascerebbe gli angoli, e si vedono sulla mappa."""
    field = np.zeros((41, 41))
    field[20, 20] = 10.0
    probability = neighbourhood_probability(field, 1.0, 10.0, 2.0)
    radius_cells = 5
    # Sull'angolo del quadrato la distanza e' 5*sqrt(2) > 5 celle: fuori dal disco.
    assert probability[20 - radius_cells, 20 - radius_cells] == 0.0
    assert probability[20 - radius_cells, 20] > 0.0


def test_neighbourhood_ignores_missing_points():
    """I punti mancanti non contano ne' a favore ne' contro."""
    field = np.full((21, 21), np.nan)
    field[10, 10] = 5.0
    field[10, 11] = 0.0
    probability = neighbourhood_probability(field, 1.0, 6.0, 2.0)
    # Nel vicinato ci sono due punti validi, uno solo sopra soglia.
    assert np.isclose(probability[10, 10], 0.5)
    # Dove non c'e' nessun punto valido nel raggio, il risultato e' mancante.
    assert np.isnan(probability[0, 0])


def test_neighbourhood_rejects_bad_geometry():
    for bad in ((0.0, 2.0), (10.0, 0.0), (-5.0, 2.0)):
        try:
            neighbourhood_probability(np.zeros((5, 5)), 1.0, bad[0], bad[1])
        except ValueError:
            continue
        raise AssertionError(f"raggio/passo non validi accettati: {bad}")
    try:
        neighbourhood_probability(np.zeros(5), 1.0, 10.0, 2.0)
    except ValueError:
        return
    raise AssertionError("un campo 1D deve essere rifiutato")


def test_storm_probability_is_readable_where_the_fraction_is_not():
    """Il doppio passaggio esiste perche' la frazione semplice era muta.

    Una cella convettiva larga pochi chilometri riempie una frazione minuscola
    di un intorno da 25 km: la frazione puntuale resta bassa anche quando il
    temporale c'e' di sicuro.  Definendo l'evento come "temporale entro 10 km"
    il numero risponde alla domanda giusta e diventa leggibile.
    """
    field = np.zeros((121, 121))
    field[60, 58:63] = 20.0  # una cella di cinque punti, circa 11 km

    fraction = neighbourhood_probability(field, 1.0, 25.0, 2.2)
    probability = storm_probability(field, 1.0, 10.0, 25.0, 2.2)

    assert fraction[60, 60] < 0.05, "la frazione puntuale resta muta: 1% circa"
    assert probability[60, 60] > 0.2, "la stessa cella deve dare un numero leggibile"
    assert probability[60, 60] > fraction[60, 60] * 10


def test_storm_probability_grows_with_the_size_of_the_system():
    """Il numero deve dire quanto e' probabile essere presi, non quanto piove.

    Una cella isolata puo' mancarti anche se il modello la mette proprio li':
    spostata di venti chilometri passa altrove.  Un sistema esteso ti prende
    comunque.  La scala fra i due casi e' l'informazione utile.
    """
    def probabilita(campo):
        return storm_probability(campo, 1.0, 10.0, 25.0, 2.2)[60, 60]

    isolata = np.zeros((121, 121))
    isolata[60, 58:63] = 20.0
    gruppo = np.zeros((121, 121))
    gruppo[55:66, 55:66] = 20.0
    esteso = np.zeros((121, 121))
    esteso[45:76, 45:76] = 20.0

    assert 0.2 < probabilita(isolata) < 0.5
    assert 0.9 < probabilita(gruppo) <= 1.0
    assert probabilita(esteso) == 1.0
    assert probabilita(isolata) < probabilita(gruppo) <= probabilita(esteso)


def test_storm_probability_fades_with_distance_and_ends():
    field = np.zeros((121, 121))
    field[60, 60] = 20.0
    probability = storm_probability(field, 1.0, 10.0, 25.0, 2.2)
    vicino = probability[60, 60]
    medio = probability[60, 70]     # 22 km
    lontano = probability[60, 90]   # 66 km
    assert vicino > medio > 0.0
    assert lontano == 0.0, "oltre evento + lisciatura non deve restare segnale"


def test_storm_probability_silent_without_lightning():
    field = np.zeros((41, 41))
    probability = storm_probability(field, 1.0, 10.0, 25.0, 2.2)
    assert np.nanmax(probability) == 0.0


def test_storm_mode_needs_energy_before_shear():
    """Lo shear da solo non fa temporali."""
    # Molto shear, nessuna energia: niente.
    assert storm_mode(np.array([50.0]), np.array([30.0]))[0] == 0.0
    # Energia senza shear: cella singola.
    assert storm_mode(np.array([1_500.0]), np.array([2.0]))[0] == 1.0
    # Energia e shear moderato: organizzata.
    assert storm_mode(np.array([1_500.0]), np.array([14.0]))[0] == 2.0
    # Molta energia e molto shear: supercella.
    assert storm_mode(np.array([1_500.0]), np.array([25.0]))[0] == 3.0
    # Poca energia anche con molto shear non arriva a supercella.
    assert storm_mode(np.array([600.0]), np.array([25.0]))[0] == 2.0
    assert np.isnan(storm_mode(np.array([np.nan]), np.array([25.0]))[0])


def test_hail_melts_with_a_high_freezing_level():
    """Lo stesso graupel pesa meno se deve attraversare uno strato caldo."""
    freddo = hail_potential(np.array([3.0]), np.array([2_500.0]), np.array([2_000.0]), np.array([20.0]))
    caldo = hail_potential(np.array([3.0]), np.array([4_800.0]), np.array([2_000.0]), np.array([20.0]))
    assert freddo[0] > caldo[0]
    # Senza graupel non c'e' grandine, per quanta energia ci sia.
    assert hail_potential(np.array([0.0]), np.array([2_500.0]), np.array([4_000.0]), np.array([30.0]))[0] == 0.0


def test_downburst_grows_with_a_high_cloud_base():
    """Base alta = strato secco profondo sotto la nube."""
    bassa = downburst_potential(np.array([600.0]), np.array([np.nan]), np.array([1_500.0]))
    alta = downburst_potential(np.array([3_000.0]), np.array([np.nan]), np.array([1_500.0]))
    assert alta[0] > bassa[0]
    # La raffica prevista dal modello e' un secondo segnale indipendente.
    assert downburst_potential(np.array([500.0]), np.array([32.0]), np.array([100.0]))[0] > 0.5


def test_bowen_ratio_and_its_undefined_zone():
    """Suolo secco = rapporto alto; senza evaporazione il rapporto non esiste."""
    secco = bowen_ratio(np.array([-200.0]), np.array([-40.0]))
    umido = bowen_ratio(np.array([-60.0]), np.array([-180.0]))
    assert secco[0] > 1.0 > umido[0]
    # Il verso del flusso non deve cambiare il risultato: ICON li pubblica
    # positivi verso il basso, quindi di giorno sono negativi.
    assert np.isclose(bowen_ratio(np.array([200.0]), np.array([40.0]))[0], secco[0])
    assert np.isnan(bowen_ratio(np.array([-200.0]), np.array([-1.0]))[0])


def test_coarsen_keeps_the_peaks_it_must_keep():
    """Un nucleo convettivo occupa pochi punti: mediarlo lo cancella."""
    field = np.zeros((8, 8))
    field[3, 3] = 20.0
    assert coarsen(field, 2, "max").max() == 20.0
    assert coarsen(field, 2, "mean").max() == 5.0, "la media diluisce, come deve"
    assert coarsen(field, 2, "max").shape == (4, 4)


def test_coarsen_handles_odd_sizes_and_gaps():
    field = np.arange(35.0).reshape(5, 7)
    assert coarsen(field, 2, "mean").shape == (3, 4)
    holed = np.full((4, 4), np.nan)
    holed[0, 0] = 4.0
    # Un solo punto valido nel blocco: la media e' quel punto, non NaN.
    assert coarsen(holed, 2, "mean")[0, 0] == 4.0
    # Blocco tutto mancante: resta mancante, non diventa zero.
    assert np.isnan(coarsen(holed, 2, "mean")[1, 1])
    assert np.isnan(coarsen(holed, 2, "max")[1, 1])


def test_coarsen_nearest_for_categories():
    """Il modo del temporale e' un codice: la media fra 1 e 3 non e' 2."""
    field = np.array([[3.0, 1.0], [1.0, 1.0]])
    assert coarsen(field, 2, "nearest")[0, 0] == 3.0
    for bad in ("median", ""):
        try:
            coarsen(np.zeros((4, 4)), 2, bad)
        except ValueError:
            continue
        raise AssertionError(f"modo sconosciuto accettato: {bad!r}")


def test_summary_reports_area_and_peak():
    probability = np.array([[0.0, 25.0], [60.0, np.nan]])
    updraft = np.array([[0.0, 3.0], [18.0, np.nan]])
    confidence = np.array([[90.0, 70.0], [30.0, np.nan]])
    contradiction = np.array([[0.0, 10.0], [40.0, np.nan]])
    summary = summarize_storms(probability, updraft, confidence, contradiction)
    assert summary["status"] == "available"
    assert summary["maximum"] == 60.0
    assert summary["maxUpdraft"] == 18.0
    assert summary["meanConfidence"] == 63.3
    assert np.isclose(summary["areaHighProbabilityLowConfidencePct"], 100.0 / 3.0, atol=0.01)
    assert summary["meanContradiction"] == 16.7
    assert np.isclose(summary["areaAbove20Pct"], 200.0 / 3.0, atol=0.01)
    vuoto = summarize_storms(np.full((3, 3), np.nan))
    assert vuoto["status"] == "unavailable" and vuoto["maximum"] is None


def test_upslope_rises_on_the_windward_side():
    """Il vento che incontra un pendio sale; sul lato sottovento scende."""
    latitudes = np.linspace(45.0, 44.0, 21)
    longitudes = np.linspace(10.0, 11.0, 21)
    # Una collina a forma di cono al centro del riquadro.
    gy, gx = np.meshgrid(np.arange(21.0), np.arange(21.0), indexing="ij")
    height = 1_000.0 * np.exp(-(((gx - 10) ** 2 + (gy - 10) ** 2) / 24.0))

    vento_da_ovest_u = np.full(height.shape, 10.0)
    zero = np.zeros(height.shape)
    w = upslope_flow(vento_da_ovest_u, zero, height, latitudes, longitudes)
    # Con vento da ovest il versante occidentale (indici piu' bassi) sale.
    assert w[10, 7] > 0.1, "il sopravvento deve salire"
    assert w[10, 13] < -0.1, "il sottovento deve scendere"
    # Vento nullo: nessuna risalita, per quanto ripido sia il pendio.
    assert np.allclose(upslope_flow(zero, zero, height, latitudes, longitudes), 0.0)
    # Pianura: nessuna risalita, per quanto forte sia il vento.
    piatto = np.zeros(height.shape)
    assert np.allclose(
        upslope_flow(vento_da_ovest_u, zero, piatto, latitudes, longitudes), 0.0
    )


def test_upslope_scales_with_wind_and_slope():
    latitudes = np.linspace(45.0, 44.0, 21)
    longitudes = np.linspace(10.0, 11.0, 21)
    gy, gx = np.meshgrid(np.arange(21.0), np.arange(21.0), indexing="ij")
    dolce = 300.0 * np.exp(-(((gx - 10) ** 2 + (gy - 10) ** 2) / 24.0))
    ripida = 1_200.0 * np.exp(-(((gx - 10) ** 2 + (gy - 10) ** 2) / 24.0))
    zero = np.zeros(dolce.shape)
    debole = np.full(dolce.shape, 3.0)
    forte = np.full(dolce.shape, 12.0)
    assert (
        upslope_flow(forte, zero, ripida, latitudes, longitudes)[10, 7]
        > upslope_flow(forte, zero, dolce, latitudes, longitudes)[10, 7]
    )
    assert (
        upslope_flow(forte, zero, dolce, latitudes, longitudes)[10, 7]
        > upslope_flow(debole, zero, dolce, latitudes, longitudes)[10, 7]
    )


def _riquadro_costiero(n=41):
    """Meta' mare a ovest, meta' terra a est, costa dritta al centro."""
    latitudes = np.linspace(40.0, 39.0, n)
    longitudes = np.linspace(12.0, 13.0, n)
    land = np.zeros((n, n))
    land[:, n // 2:] = 1.0
    return land, latitudes, longitudes


def test_coast_distance_signs_and_growth():
    land, latitudes, longitudes = _riquadro_costiero()
    distance = coast_distance_km(land, latitudes, longitudes)
    n = land.shape[0]
    assert distance[20, n // 2] == 0.0, "la costa e' a distanza zero"
    assert distance[20, n // 2 + 5] > 0, "l'entroterra e' positivo"
    assert distance[20, n // 2 - 5] < 0, "il mare e' negativo"
    # Allontanandosi dalla costa la distanza cresce, da entrambi i lati.
    assert distance[20, n // 2 + 8] > distance[20, n // 2 + 3]
    assert distance[20, n // 2 - 8] < distance[20, n // 2 - 3]


def test_coast_distance_without_a_coast():
    """Tutto mare o tutta terra: non c'e' brezza da isolare."""
    latitudes = np.linspace(40.0, 39.0, 11)
    longitudes = np.linspace(12.0, 13.0, 11)
    assert np.all(np.isnan(coast_distance_km(np.ones((11, 11)), latitudes, longitudes)))
    assert np.all(np.isnan(coast_distance_km(np.zeros((11, 11)), latitudes, longitudes)))


def test_sea_breeze_needs_land_onshore_wind_and_convergence():
    land, latitudes, longitudes = _riquadro_costiero()
    n = land.shape[0]
    convergence = np.full((n, n), 2.0e-4)
    dal_mare = np.full((n, n), 6.0)     # vento da ovest, cioe' dal mare
    verso_il_mare = np.full((n, n), -6.0)
    zero = np.zeros((n, n))

    breeze = sea_breeze_lift(convergence, dal_mare, zero, land, latitudes, longitudes)
    assert breeze[20, n // 2 + 3] > 0, "terra, vento dal mare, convergenza: brezza"
    assert breeze[20, n // 2 - 3] == 0, "sul mare non si attribuisce brezza"

    # Stesso quadro ma vento verso il mare: non e' brezza.
    contraria = sea_breeze_lift(
        convergence, verso_il_mare, zero, land, latitudes, longitudes
    )
    assert contraria[20, n // 2 + 3] == 0

    # Divergenza invece di convergenza: niente.
    divergenza = sea_breeze_lift(
        -convergence, dal_mare, zero, land, latitudes, longitudes
    )
    assert divergenza[20, n // 2 + 3] == 0


def test_sea_breeze_never_lands_on_the_sea():
    """Sui punti di mare in riva la distanza vale -0.0, e -0.0 >= 0 e' vero.

    Con il confronto sul segno una fascia di celle marine risultava terra e
    prendeva brezza: trovato sui dati veri, non sui sintetici.
    """
    land, latitudes, longitudes = _riquadro_costiero(41)
    n = land.shape[0]
    convergence = np.full((n, n), 2.0e-4)
    dal_mare = np.full((n, n), 6.0)
    zero = np.zeros((n, n))
    breeze = sea_breeze_lift(convergence, dal_mare, zero, land, latitudes, longitudes)
    sul_mare = land < 0.5
    assert np.nanmax(breeze[sul_mare]) == 0.0, "nessuna brezza puo' stare sul mare"


def test_sea_breeze_stops_inland():
    land, latitudes, longitudes = _riquadro_costiero(81)
    n = land.shape[0]
    convergence = np.full((n, n), 2.0e-4)
    dal_mare = np.full((n, n), 6.0)
    zero = np.zeros((n, n))
    breeze = sea_breeze_lift(
        convergence, dal_mare, zero, land, latitudes, longitudes,
        inland_reach_km=20.0,
    )
    vicino = breeze[40, n // 2 + 2]
    lontano = breeze[40, n - 3]
    assert vicino > 0
    assert lontano == 0, "oltre la portata la brezza non arriva"


def test_trigger_takes_the_strongest_not_the_sum():
    """In un punto agisce un meccanismo: sommarli inventerebbe inneschi."""
    forte = trigger_index(upslope_ms=np.array([0.6]), sea_breeze=None, convergence=None)
    assert np.isclose(forte[0], 1.0)
    debole = trigger_index(
        upslope_ms=np.array([0.15]),
        sea_breeze=np.array([0.4e-4]),
        convergence=np.array([0.5e-4]),
    )
    # Tre contributi deboli non devono sommarsi a uno forte.
    assert debole[0] < 0.35
    # Il massimo e' quello che conta.
    misto = trigger_index(
        upslope_ms=np.array([0.05]), sea_breeze=np.array([2.0e-4]), convergence=None
    )
    assert np.isclose(misto[0], 1.0)
    assert np.isnan(trigger_index(upslope_ms=np.array([np.nan]))[0])


def _intelligent_case(
    *,
    lpi_active=False,
    updraft_active=False,
    temporal_active=False,
    favourable=True,
    shear=18.0,
):
    """Caso sintetico centrato, con gruppi di evidenza controllabili."""
    n = 61
    centre = n // 2
    shape = (n, n)
    lpi = np.zeros(shape)
    updraft = np.zeros(shape)
    previous = np.zeros(shape)
    next_lpi = np.zeros(shape)
    if lpi_active:
        lpi[centre - 1:centre + 2, centre - 1:centre + 2] = 4.0
    if updraft_active:
        updraft[centre - 1:centre + 2, centre - 1:centre + 2] = 7.0
    if temporal_active:
        previous[centre - 2:centre + 1, centre - 1:centre + 2] = 4.0
        next_lpi[centre:centre + 3, centre - 1:centre + 2] = 4.0

    if favourable:
        cape = np.full(shape, 1_600.0)
        cape_mu = np.full(shape, 1_900.0)
        cin = np.full(shape, -20.0)
        trigger = np.full(shape, 0.9)
        surface_rh = np.full(shape, 78.0)
        middle_rh = np.full(shape, 68.0)
        cloud_base = np.full(shape, 900.0)
        omega = np.full(shape, -0.25)
        front_distance = np.full(shape, 25.0)
    else:
        cape = np.zeros(shape)
        cape_mu = np.zeros(shape)
        cin = np.full(shape, -350.0)
        trigger = np.zeros(shape)
        surface_rh = np.full(shape, 25.0)
        middle_rh = np.full(shape, 12.0)
        cloud_base = np.full(shape, 4_000.0)
        omega = np.full(shape, 0.08)
        front_distance = np.full(shape, 500.0)

    result = intelligent_storm_probability(
        lpi,
        cape,
        cin,
        trigger,
        2.0,
        updraft_ms=updraft,
        cape_mu=cape_mu,
        surface_rh=surface_rh,
        mid_level_rh=middle_rh,
        cloud_base_m=cloud_base,
        omega_700=omega,
        front_distance_km=front_distance,
        shear=np.full(shape, shear),
        helicity=np.full(shape, 80.0 if shear > 5.0 else 0.0),
        previous_lightning_potential=previous if temporal_active else None,
        next_lightning_potential=next_lpi if temporal_active else None,
        event_radius_km=10.0,
        smoothing_radius_km=20.0,
    )
    return result, centre


def test_intelligent_probability_needs_more_than_cape_or_shear():
    """L'ambiente favorevole segnala possibilita', ma non inventa una cella."""
    favourable, centre = _intelligent_case(favourable=True, shear=25.0)
    value = favourable["probability"][centre, centre]
    assert 0.20 < value < 0.50

    hostile, centre = _intelligent_case(favourable=False, shear=35.0)
    assert hostile["probability"][centre, centre] < 0.03


def test_intelligent_probability_rewards_independent_corroboration():
    """LPI, updraft, ambiente e persistenza devono rafforzarsi a vicenda."""
    corroborated, centre = _intelligent_case(
        lpi_active=True,
        updraft_active=True,
        temporal_active=True,
        favourable=True,
    )
    contradicted, _ = _intelligent_case(
        lpi_active=True,
        updraft_active=False,
        temporal_active=False,
        favourable=False,
    )
    strong = corroborated["probability"][centre, centre]
    weak = contradicted["probability"][centre, centre]
    assert strong > weak + 0.25
    assert corroborated["confidence"][centre, centre] > contradicted["confidence"][centre, centre]
    assert corroborated["contradiction"][centre, centre] < contradicted["contradiction"][centre, centre]


def test_mature_resolved_core_is_not_rejected_for_consumed_cape():
    """LPI + updraft restano validi dentro la cold pool a CAPE locale bassa."""
    mature, centre = _intelligent_case(
        lpi_active=True,
        updraft_active=True,
        temporal_active=True,
        favourable=False,
    )
    spurious, _ = _intelligent_case(
        lpi_active=True,
        updraft_active=False,
        temporal_active=False,
        favourable=False,
    )
    assert mature["probability"][centre, centre] > spurious["probability"][centre, centre]
    assert mature["probability"][centre, centre] > 0.20


def test_intelligent_probability_components_are_bounded_and_explainable():
    result, _ = _intelligent_case(
        lpi_active=True,
        updraft_active=True,
        temporal_active=True,
        favourable=True,
    )
    expected = {
        "probability", "direct", "environment", "instability", "moisture",
        "lift", "temporal", "organisation", "contradiction", "confidence",
    }
    assert set(result) == expected
    for name, field in result.items():
        finite = field[np.isfinite(field)]
        assert finite.size, name
        assert np.min(finite) >= 0.0, name
        assert np.max(finite) <= 1.0, name


if __name__ == "__main__":
    for name, function in sorted(list(globals().items())):
        if name.startswith("test_") and callable(function):
            function()
    print("Storm diagnostics tests passed")
