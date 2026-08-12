"""Test sintetici della diagnostica temporalesca."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from meteo_analysis.hazards.storms import (  # noqa: E402
    AIR_DENSITY_BY_LEVEL,
    bowen_ratio,
    coarsen,
    bulk_shear,
    downburst_potential,
    hail_potential,
    lifting_condensation_level,
    neighbourhood_probability,
    potential_updraft,
    storm_mode,
    storm_probability,
    strongest_updraft,
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
    summary = summarize_storms(probability, updraft)
    assert summary["status"] == "available"
    assert summary["maximum"] == 60.0
    assert summary["maxUpdraft"] == 18.0
    assert np.isclose(summary["areaAbove20Pct"], 200.0 / 3.0, atol=0.01)
    vuoto = summarize_storms(np.full((3, 3), np.nan))
    assert vuoto["status"] == "unavailable" and vuoto["maximum"] is None


if __name__ == "__main__":
    for name, function in sorted(list(globals().items())):
        if name.startswith("test_") and callable(function):
            function()
    print("Storm diagnostics tests passed")
