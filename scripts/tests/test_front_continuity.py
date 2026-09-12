"""Temporal continuity: a boundary seen in some hours must not vanish in all.

The published run of 2026-09-12 00Z is the case this exists for.  Its own
quality file says the detector accepted candidates in 52 of 73 hours --
including +38h with four and +39h with three -- and the map showed no front at
all from +30h to +72h.  Nothing was missing from the physics; an all-or-nothing
survival rule threw away hours that had been detected, because other hours in
the same span had not been.

What is checked here: interior gaps are carried, the carried position states a
Brownian-bridge uncertainty that is zero at the observations and widest in the
middle, a gap wider than the declared limit is *not* bridged, and the coverage
rule is a floor rather than a second judgement on evidence that already counts
once inside the quality score.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import front_analysis_v12 as v12

ok = True


def report(passed, message, detail=""):
    global ok
    if not passed:
        ok = False
        print("  FAIL: " + message + (("  " + detail) if detail else ""))
    elif detail:
        print("    " + detail)


def straight_line(longitude_shift):
    """A north-south segment displaced east by ``longitude_shift`` degrees."""
    return np.column_stack((
        np.full(9, 10.0 + longitude_shift),
        np.linspace(41.0, 45.0, 9),
    ))


def track_with(hours, motion_mad_kmh=18.0, front_type="cold"):
    return {
        "lines": {hour: straight_line(0.25 * hour) for hour in hours},
        "localClassifications": {
            hour: {"frontType": front_type, "classificationCertainty": 0.80}
            for hour in hours
        },
        "motionMadKmh": motion_mad_kmh,
        "frontType": front_type,
    }


AVAILABLE = set(range(0, 73))

# --------------------------------------------------------------------------
print("1) Il profilo orario vero del run misurato viene reso continuo")
# --------------------------------------------------------------------------
# Ore in cui il run 2026-09-12 00Z aveva almeno una polilinea accettata, da
# +32h in poi, cosi' come le riporta il suo stesso front_qc.json.
OBSERVED = [32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 48, 49, 50,
            51, 52, 63, 65, 66, 67, 68, 70, 71]
expanded, local, bridge = v12.fill_track_gaps(track_with(OBSERVED), AVAILABLE)
span = range(min(OBSERVED), max(OBSERVED) + 1)
covered = sorted(expanded)
holes = [hour for hour in span if hour not in expanded]
report(set(OBSERVED) <= set(covered),
       "il riempimento ha perso ore che erano state osservate")
report(len(covered) > len(OBSERVED),
       "nessun buco e' stato colmato",
       "osservate %d ore, pubblicabili %d su uno span di %d"
       % (len(OBSERVED), len(covered), len(list(span))))
# Il buco +53h..+62h e' di dieci ore: oltre il limite dichiarato, e va lasciato.
report(all(53 <= hour <= 62 for hour in holes),
       "restano buchi che il limite dichiarato avrebbe dovuto colmare",
       "buchi residui: %s (il salto +53..+62 e' di %d ore, limite %d)"
       % (holes, 62 - 53 + 1, v12.MAX_INFERRED_GAP_HOURS))

# --------------------------------------------------------------------------
print("\n2) L'incertezza inferita e' un ponte browniano, non una costante")
# --------------------------------------------------------------------------
track = track_with([10, 16], motion_mad_kmh=20.0)
expanded, local, bridge = v12.fill_track_gaps(track, AVAILABLE)
report(sorted(bridge) == [11, 12, 13, 14, 15],
       "le ore inferite non sono quelle attese",
       "ore inferite: %s" % sorted(bridge))
middle = bridge.get(13, 0.0)
edge = bridge.get(11, 0.0)
analytic = 20.0 * np.sqrt(3.0 * 3.0 / 6.0)   # s sqrt(t (T-t) / T), t=3, T=6
report(abs(middle - analytic) < 1.0e-6 and middle > edge,
       "l'incertezza inferita non segue il ponte browniano",
       "a meta' del buco %.1f km (analitico %.1f), a un'ora dall'osservazione "
       "%.1f km" % (middle, analytic, edge))
report(all(value > 0.0 for value in bridge.values()),
       "un'ora inferita dichiara incertezza nulla")

# La geometria inferita deve stare fra le due osservazioni, in proporzione.
line_11 = np.asarray(expanded[11], dtype=float)
expected_longitude = 10.0 + 0.25 * (10 + (16 - 10) * (1.0 / 6.0))
report(abs(float(np.mean(line_11[:, 0])) - expected_longitude) < 0.02,
       "la geometria inferita non e' l'interpolazione in proporzione al tempo",
       "a +11h longitudine media %.3f, attesa %.3f"
       % (float(np.mean(line_11[:, 0])), expected_longitude))

# --------------------------------------------------------------------------
print("\n3) Un salto troppo lungo resta un salto")
# --------------------------------------------------------------------------
long_gap = track_with([0, 0 + v12.MAX_INFERRED_GAP_HOURS + 2])
expanded, local, bridge = v12.fill_track_gaps(long_gap, AVAILABLE)
report(not bridge and sorted(expanded) == sorted(long_gap["lines"]),
       "un buco oltre il limite dichiarato viene colmato lo stesso",
       "buco di %d ore, limite %d: ore inferite %d"
       % (v12.MAX_INFERRED_GAP_HOURS + 2 - 1, v12.MAX_INFERRED_GAP_HOURS,
          len(bridge)))

# Due estremi entrambi ambigui non hanno un tipo da trasportare.
ambiguous = track_with([4, 7], front_type="uncertain")
expanded, local, bridge = v12.fill_track_gaps(ambiguous, AVAILABLE)
report(not bridge,
       "un buco fra due ore entrambe ambigue viene colmato senza un tipo")

# Un solo estremo ambiguo: l'altro decide, il buco si colma.
half_known = track_with([4, 7])
half_known["localClassifications"][4] = {
    "frontType": "uncertain", "classificationCertainty": 0.3
}
expanded, local, bridge = v12.fill_track_gaps(half_known, AVAILABLE)
report(sorted(bridge) == [5, 6],
       "un estremo ambiguo su due impedisce ancora di colmare il buco",
       "ore inferite: %s, tipo trasportato: %s"
       % (sorted(bridge), local.get(6, {}).get("frontType")))

# --------------------------------------------------------------------------
print("\n4) La copertura e' un pavimento, non un secondo giudizio")
# --------------------------------------------------------------------------
# La copertura entra gia' in qualityScore, con peso 0.38 dentro la componente
# temporale: vietarla di nuovo conta due volte la stessa prova.
report(v12.TRACK_MIN_COVERAGE <= 0.50,
       "la copertura e' ancora usata come veto oltre che come penalita'",
       "TRACK_MIN_COVERAGE = %.2f" % v12.TRACK_MIN_COVERAGE)
observed_share = len(OBSERVED) / float(len(list(span)))
report(observed_share >= v12.TRACK_MIN_COVERAGE,
       "il profilo orario misurato nel run vero non passerebbe nemmeno ora",
       "copertura del caso reale: %.2f contro un pavimento di %.2f"
       % (observed_share, v12.TRACK_MIN_COVERAGE))

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
