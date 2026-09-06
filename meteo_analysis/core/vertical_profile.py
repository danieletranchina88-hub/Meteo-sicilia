"""Profilo termico verticale del modello, per correggere la quota del terreno.

Un modello a 2,2 km non sa che le montagne esistono: le smussa. Misurato sul
run del 6 settembre 2026, l'orografia ICON-2I sta 834 m sotto il Gran Sasso,
808 sotto il Monte Bianco, 727 sotto la Marmolada, 491 sotto l'Etna. La
temperatura a 2 m che il modello scrive in quei punti e' la temperatura di una
montagna piu' bassa, e sbaglia di conseguenza: con un gradiente ordinario, da
1,6 a 5,4 gradi.

La correzione e' fisica elementare: se si conosce come varia la temperatura con
la quota, si sposta il valore dalla quota che il modello crede a quella vera.

    T(z_vero) = T(z_modello) + [ Tprofilo(z_vero) - Tprofilo(z_modello) ]

Il punto delicato e' da dove viene il profilo. Assumere il gradiente
dell'atmosfera standard, -6,5 K/km, e' comodo e in inverno e' **sbagliato di
segno**: sotto un'inversione di fondovalle la temperatura CRESCE con la quota, e
una cima sopra l'inversione e' piu' calda del paese sotto, non piu' fredda.
Percio' qui il profilo non si assume: si legge dal modello stesso, che i livelli
a 925, 850 e 700 hPa li ha calcolati.

Le quote di quei livelli non sono costanti -- l'aria calda e' piu' spessa -- e
si ricavano dall'equazione ipsometrica, che e' esatta a meno dell'umidita':

    dz = (R_d / g) * T_media * ln(p_basso / p_alto)

con R_d/g = 29,27 m/K. Trascurare l'umidita' significa usare la temperatura
invece di quella virtuale, e sottostimare lo spessore di meno dell'1%.
"""

from __future__ import annotations

import numpy as np

# R_d / g, in metri per kelvin: la costante dell'equazione ipsometrica.
HYPSOMETRIC_M_PER_K = 287.05 / 9.80665

# Riduzione al livello del mare della temperatura a 2 m, per chiudere il
# profilo verso il basso. E' un'assunzione, e resta tale: serve solo ad
# ancorare la quota del primo livello, non a produrre la correzione.
SEA_LEVEL_LAPSE_K_PER_M = 0.0065

LEVELS_HPA = (925.0, 850.0, 700.0)


def _to_celsius(values):
    """Accetta kelvin o gradi: sopra 150 e' kelvin, e nessuna aria e' a 150 C."""
    array = np.asarray(values, dtype=float)
    return np.where(array > 150.0, array - 273.15, array)


def layer_thickness_m(lower_hpa: float, upper_hpa: float, mean_temperature_c):
    """Spessore geopotenziale fra due superfici isobariche, in metri.

    L'aria calda e' piu' spessa: e' il motivo per cui le quote dei livelli
    barici non sono costanti e vanno calcolate invece che tabulate.
    """
    if not (upper_hpa > 0.0 and lower_hpa > upper_hpa):
        raise ValueError("il livello inferiore deve avere pressione maggiore")
    mean_kelvin = _to_celsius(mean_temperature_c) + 273.15
    return HYPSOMETRIC_M_PER_K * mean_kelvin * np.log(lower_hpa / upper_hpa)


def temperature_profile(mslp_hpa, t2m, terrain_m, t925, t850, t700):
    """Quote e temperature dei livelli barici, punto per punto.

    Restituisce ``(heights, temperatures)``: due liste di tre griglie, dal
    basso verso l'alto. Le quote sono sul livello del mare, in metri; le
    temperature in gradi.

    Il profilo si ancora al livello del mare tramite la pressione ridotta, non
    alla quota del modello: cosi' la stessa colonna d'aria descrive sia il
    punto che il modello crede a 2078 m sia la vetta vera a 2912, che e'
    esattamente il confronto che serve.
    """
    pressure = np.asarray(mslp_hpa, dtype=float)
    surface_c = _to_celsius(t2m)
    ground = np.asarray(terrain_m, dtype=float)
    levels_c = [_to_celsius(t925), _to_celsius(t850), _to_celsius(t700)]

    # Temperatura riportata al livello del mare: chiude il profilo in basso.
    sea_level_c = surface_c + SEA_LEVEL_LAPSE_K_PER_M * np.where(
        np.isfinite(ground), ground, 0.0
    )

    heights = []
    previous_height = np.zeros_like(sea_level_c)
    previous_pressure = pressure
    previous_c = sea_level_c
    for level_hpa, level_c in zip(LEVELS_HPA, levels_c):
        mean_c = 0.5 * (previous_c + level_c)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_kelvin = mean_c + 273.15
            thickness = HYPSOMETRIC_M_PER_K * mean_kelvin * np.log(
                np.where(previous_pressure > 0.0, previous_pressure, np.nan)
                / level_hpa
            )
        height = previous_height + thickness
        heights.append(height)
        previous_height = height
        previous_pressure = np.full_like(pressure, level_hpa)
        previous_c = level_c
    return heights, levels_c


def elevation_correction_c(heights, temperatures, model_height_m, true_height_m):
    """Differenza di temperatura fra la quota vera e quella del modello.

    Il profilo viene interpolato linearmente in quota fra i livelli noti;
    sopra e sotto si prosegue con il gradiente del livello piu' vicino, perche'
    inventare un gradiente diverso dove non c'e' dato sarebbe peggio che
    prolungare quello misurato.

    Solo la DIFFERENZA conta: lo scarto fra la temperatura a 2 m e quella
    dell'aria libera alla stessa quota si elide, e non serve assumere nulla su
    quanto valga.
    """
    model = np.asarray(model_height_m, dtype=float)
    true = np.asarray(true_height_m, dtype=float)
    return _sample_profile(heights, temperatures, true) - _sample_profile(
        heights, temperatures, model
    )


def _sample_profile(heights, temperatures, target):
    """Temperatura del profilo alla quota richiesta, con estrapolazione lineare."""
    z = [np.asarray(h, dtype=float) for h in heights]
    t = [np.asarray(v, dtype=float) for v in temperatures]
    target = np.asarray(target, dtype=float)

    # Sotto il primo livello e sopra l'ultimo si prolunga il gradiente del
    # segmento estremo; in mezzo si interpola fra i due livelli che lo
    # contengono. Un ciclo su tre livelli, non su mezzo milione di punti.
    result = _segment(z[0], t[0], z[1], t[1], target)
    inside = target >= z[1]
    if np.any(inside):
        upper = _segment(z[1], t[1], z[2], t[2], target)
        result = np.where(inside, upper, result)
    return result


def _segment(z_low, t_low, z_high, t_high, target):
    span = z_high - z_low
    with np.errstate(divide="ignore", invalid="ignore"):
        gradient = np.where(np.abs(span) > 1.0, (t_high - t_low) / span, 0.0)
    return t_low + gradient * (target - z_low)


# --- Quota neve -------------------------------------------------------------
#
# Il confine fra pioggia e neve non e' lo zero del bulbo secco. La neve che
# cade attraverso aria sopra zero fonde, e fondendo raffredda l'aria che
# attraversa: se quell'aria e' secca, il raffreddamento la porta sotto zero e i
# fiocchi arrivano al suolo anche con due o tre gradi di temperatura misurata.
# La grandezza che governa il passaggio e' quindi il bulbo bagnato, ed e' per
# questo che la quota neve sta sotto lo zero termico, di piu' quanto piu'
# l'aria e' secca.
#
# Il bulbo bagnato si ricava dalla relazione psicrometrica
#
#     e = es(Tw) - gamma * p * (T - Tw)
#
# che non ha soluzione esplicita ma si risolve per bisezione in un intervallo
# noto a priori: Tw sta sempre fra il punto di rugiada e la temperatura.

PSYCHROMETRIC_PER_K = 6.53e-4  # costante psicrometrica, 1/K
SNOW_LINE_WET_BULB_C = 0.0


def saturation_vapour_pressure_pa(temperature_c):
    """Tensione di vapore saturo secondo Bolton (1980), in pascal."""
    t = np.asarray(temperature_c, dtype=float)
    return 611.2 * np.exp(17.67 * t / (t + 243.5))


def vapour_pressure_pa(specific_humidity, pressure_hpa):
    """Tensione di vapore da umidita' specifica (kg/kg) e pressione."""
    q = np.clip(np.asarray(specific_humidity, dtype=float), 1e-9, 0.05)
    p = np.asarray(pressure_hpa, dtype=float) * 100.0
    # e = q * p / (eps + (1 - eps) * q), con eps = R_d/R_v = 0,622.
    return q * p / (0.622 + 0.378 * q)


def wet_bulb_c(temperature_c, specific_humidity, pressure_hpa, iterations: int = 40):
    """Temperatura di bulbo bagnato, in gradi.

    Bisezione fra punto di rugiada e temperatura: l'intervallo contiene sempre
    la soluzione, quindi il metodo converge senza bisogno di un valore di
    partenza fortunato. Quaranta passi portano l'incertezza sotto il millesimo
    di grado su un intervallo di 60 gradi.
    """
    t = np.asarray(temperature_c, dtype=float)
    e = vapour_pressure_pa(specific_humidity, pressure_hpa)
    p_pa = np.asarray(pressure_hpa, dtype=float) * 100.0

    # Punto di rugiada, invertendo Bolton: e' il limite inferiore.
    ratio = np.log(np.clip(e, 1e-6, None) / 611.2)
    dewpoint = 243.5 * ratio / np.clip(17.67 - ratio, 1e-6, None)
    low = np.minimum(dewpoint, t)
    high = t
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        # Residuo positivo: il bulbo bagnato ipotizzato e' troppo alto.
        residual = (
            saturation_vapour_pressure_pa(middle)
            - PSYCHROMETRIC_PER_K * p_pa * (t - middle)
            - e
        )
        high = np.where(residual > 0.0, middle, high)
        low = np.where(residual > 0.0, low, middle)
    return 0.5 * (low + high)


def snow_line_m(heights, wet_bulbs, surface_height_m, surface_wet_bulb_c,
                threshold_c: float = SNOW_LINE_WET_BULB_C):
    """Quota a cui il bulbo bagnato attraversa la soglia, in metri.

    Si scende dall'alto e si prende il primo attraversamento: sopra la quota
    neve fa piu' freddo della soglia, sotto piu' caldo. Dove l'aria e' sotto
    soglia fino al suolo la quota vale quella del terreno, cioe' nevica fino a
    terra; dove e' sopra soglia anche al livello piu' alto disponibile il
    valore resta ignoto, perche' affermare una quota fuori dai dati sarebbe
    inventarla.
    """
    levels = [np.asarray(surface_height_m, dtype=float)] + [
        np.asarray(h, dtype=float) for h in heights
    ]
    values = [np.asarray(surface_wet_bulb_c, dtype=float)] + [
        np.asarray(v, dtype=float) for v in wet_bulbs
    ]
    result = np.full(np.broadcast(levels[0], values[0]).shape, np.nan)
    # Dal basso verso l'alto: l'ultimo attraversamento trovato risalendo e' il
    # primo che si incontra scendendo.
    for index in range(len(levels) - 1):
        z_low, z_high = levels[index], levels[index + 1]
        t_low, t_high = values[index], values[index + 1]
        crosses = (t_low - threshold_c) * (t_high - threshold_c) < 0.0
        span = t_high - t_low
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(np.abs(span) > 1e-9,
                                (threshold_c - t_low) / span, 0.0)
        crossing = z_low + fraction * (z_high - z_low)
        result = np.where(crosses, crossing, result)
    # Gia' sotto soglia al suolo: nevica fino a terra.
    return np.where(values[0] <= threshold_c, levels[0], result)
