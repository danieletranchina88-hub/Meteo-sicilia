"""Analytic checks for the synoptic frontal engine.

Everything here is measured against a field whose answer is known in closed
form, not against a weather case that happened to look right.  The synthetic
front is built in **conformal (Mercator) coordinates**, which is the only way
a straight oblique boundary has a constant normal direction at every latitude
on a lon/lat grid: built the naive way, the front's own normal drifts by
several degrees across the domain and the test ends up measuring the
generator's error instead of the engine's.

What is verified:

* the analysis scale is the one derived from the published climatologies;
* the reference front's gradient and zone curvature match the closed form;
* Petterssen frontogenesis matches 0.5 |grad theta| (E + convergence)
  pointwise, at four orientations, and is untouched by a pure shear jet --
  which is a tensor identity, so any deviation is an implementation error;
* the recovered axis is straight (well under the 6 degrees per 20 km the
  project's own documentation sets) and lies on the true axis;
* the locating field stays low order: its measured ridge curvature must stay
  near the analytic one, which is what fails the moment a second-derivative
  diagnostic is allowed back into the field the Hessian reads;
* an orographic thermal boundary -- fixed, not frontogenetic, welded to a
  slope -- is rejected, and a real front crossing the same barrier is not.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import front_locator as fl
import front_engine as fe

erf = np.vectorize(math.erf)

R_KM = fl.EARTH_KM_PER_DEG * 180.0 / math.pi
LON = np.arange(4.0, 20.001, 0.10)
LAT = np.arange(36.0, 48.001, 0.08)
LONG, LATG = np.meshgrid(LON, LAT)
GRID = fl.grid_metrics(LON, LAT)
LON0, LAT0 = float(np.mean(LON)), float(np.mean(LAT))
COS = np.cos(np.deg2rad(LATG))
COS0 = math.cos(math.radians(LAT0))

# Mercator is conformal: an angle measured on it is the angle on the ground,
# so a straight line here is a straight line for the engine too.
MERC_X = R_KM * np.deg2rad(LONG - LON0)
MERC_Y = R_KM * np.log(
    np.tan(np.pi / 4.0 + np.deg2rad(LATG) / 2.0)
    / math.tan(math.pi / 4.0 + math.radians(LAT0) / 2.0)
)

WIDTH = fe.REFERENCE_HALF_WIDTH_KM
DELTA = fe.REFERENCE_DELTA_K
DEFORMATION = 1.0e-5      # s-1, total stretching deformation
CONVERGENCE = 2.0e-5      # s-1
JET_MS = 12.0             # front-parallel shear, frontogenetically inert
CORE = (slice(40, -40), slice(40, -40))

ok = True


def report(passed, message, detail=""):
    global ok
    if not passed:
        ok = False
        print("  FAIL: " + message + (("  " + detail) if detail else ""))
    elif detail:
        print("    " + detail)


def straight_front(angle_deg, *, jet_ms=JET_MS, delta_k=DELTA,
                   deformation=DEFORMATION, convergence=CONVERGENCE):
    """An exact straight front with its normal at ``angle_deg`` from east.

    theta_w is an error function across the front, so its gradient is Gaussian
    and every derivative used below is available in closed form.  The wind is
    a pure deformation with its axes aligned to the front, a uniform
    convergence, and a front-parallel jet that is pure shear.
    """
    angle = math.radians(angle_deg)
    nx, ny = math.cos(angle), math.sin(angle)
    across = (nx * MERC_X + ny * MERC_Y) * COS0        # km at the mean latitude
    along = (-ny * MERC_X + nx * MERC_Y) * COS0
    theta = 288.0 + 0.5 * delta_k * erf(across / (np.sqrt(2.0) * WIDTH))
    upper = 288.0 + 0.5 * (0.8 * delta_k) * erf(across / (np.sqrt(2.0) * WIDTH))
    normal_flow = -(0.5 * deformation + 0.5 * convergence) * across * 1000.0
    along_flow = +(0.5 * deformation - 0.5 * convergence) * along * 1000.0
    along_flow = along_flow + jet_ms * erf(across / (np.sqrt(2.0) * WIDTH))
    u = normal_flow * nx - along_flow * ny
    v = normal_flow * ny + along_flow * nx
    return {"thetaW": theta, "upper": upper, "u": u, "v": v,
            "across": across, "normal": (nx, ny)}


def signed_distance(lon, lat, normal):
    nx, ny = normal
    return ((lon - LON0) * fl.EARTH_KM_PER_DEG * np.cos(np.deg2rad(lat)) * nx
            + (lat - LAT0) * fl.EARTH_KM_PER_DEG * ny)


def analyse(case, **kwargs):
    evidence = fe.frontal_evidence(
        case["thetaW"], case["u"], case["v"], LON, LAT,
        metrics=GRID, theta_w_upper=case["upper"], **kwargs
    )
    field = fe.locating_field(evidence)
    points = fe.ridge_points(
        field, LON, LAT, metrics=GRID,
        mask=fe.admissible_mask(evidence),
        min_curvature=0.25 * fe.ridge_curvature_scale(),
    )
    lines = sorted(fe.merge_fragments(fe.link_ridge_points(points)),
                   key=fe.line_length_km, reverse=True)
    return evidence, field, points, lines


# --------------------------------------------------------------------------
print("1) La scala di analisi viene dalla letteratura, non dall'occhio")
# --------------------------------------------------------------------------
era_interim = fe.five_point_pass_sigma_km(8, 79.0)
era5 = fe.five_point_pass_sigma_km(96, 28.0)
report(abs(era_interim - 141.3) < 1.0 and abs(era5 - 173.5) < 1.0,
       "l'equivalenza di varianza dei filtri non torna",
       "ERA-Interim 8 passaggi a 79 km -> sigma %.1f km; "
       "ERA5 96 a 28 km -> sigma %.1f km" % (era_interim, era5))
report(min(era_interim, era5) - 15.0 <= fe.SYNOPTIC_SIGMA_KM <= max(era_interim, era5),
       "SYNOPTIC_SIGMA_KM non sta fra le due scale climatologiche",
       "SYNOPTIC_SIGMA_KM = %.0f km" % fe.SYNOPTIC_SIGMA_KM)

# --------------------------------------------------------------------------
print("\n2) Il fronte di riferimento e' quello che dice di essere")
# --------------------------------------------------------------------------
reference = fe.reference_front_scales(fe.SYNOPTIC_SIGMA_KM)
case = straight_front(0.0)
evidence, field, points, lines = analyse(case)
# La costruzione conforme misura le distanze alla latitudine di riferimento,
# quindi e' solo sulla riga di quella latitudine che il fronte sintetico e'
# esattamente il fronte di riferimento: e' li' che va fatto il confronto con
# la forma chiusa, non sul massimo di tutto il dominio.
reference_row = int(np.argmin(np.abs(LAT - LAT0)))
measured_gradient = float(np.nanmax(
    evidence["gradientMagnitude"][reference_row, 40:-40]))
report(abs(measured_gradient / reference["gradientKPerKm"] - 1.0) < 0.01,
       "il gradiente di picco non corrisponde alla forma analitica",
       "|grad theta| picco misurato %.6f K/km, analitico %.6f (scarto %+.2f%%)"
       % (measured_gradient, reference["gradientKPerKm"],
          100.0 * (measured_gradient / reference["gradientKPerKm"] - 1.0)))
report(abs(reference["widthKm"] - math.hypot(WIDTH, fe.SYNOPTIC_SIGMA_KM)) < 1e-6,
       "la larghezza non si compone in quadratura",
       "larghezza della zona dopo il lisciamento %.1f km" % reference["widthKm"])

# --------------------------------------------------------------------------
print("\n3) Frontogenesi di Petterssen contro la forma chiusa")
# --------------------------------------------------------------------------
# F = -0.5 |grad theta| (D + E cos2beta + Esh sin2beta).  Nel sistema del
# fronte beta = 0, quindi F = 0.5 |grad theta| (E + convergenza), ovunque.
for angle in (0.0, 30.0, 60.0, 90.0):
    case = straight_front(angle)
    evidence, field, points, lines = analyse(case)
    gradient_magnitude = evidence["gradientMagnitude"]
    # La costruzione conforme dilata il fronte come COS0/COS: l'attesa e'
    # quindi punto per punto, non un unico numero.
    expected = (0.5 * gradient_magnitude * (DEFORMATION + CONVERGENCE)
                * (COS0 / COS) * 100.0 * 10_800.0)
    on_axis = (np.abs(case["across"]) < 60.0)[CORE]
    ratio = (evidence["frontogenesis"] / np.where(expected > 1e-12, expected, np.nan))[CORE]
    median = float(np.nanmedian(ratio[on_axis]))
    spread = float(np.nanstd(ratio[on_axis]))
    report(abs(median - 1.0) < 0.03 and spread < 0.06,
           "frontogenesi sbagliata con normale a %.0f gradi" % angle,
           "normale %4.0f gr: F misurata / F analitica = %.4f (dispersione %.4f)"
           % (angle, median, spread))

# Un getto parallelo al fronte e' taglio puro: la deformazione di taglio entra
# con sin(2 beta), che nel sistema del fronte e' zero, quindi in forma
# tensoriale non tocca la frontogenesi.  La cancellazione non e' pero' esatta
# nel discreto, e la ragione e' misurabile: front_locator.smooth_km liscia le
# righe con un sigma in punti che cambia con la latitudine (per tenere
# costante la scala in km), e questo introduce in un getto puramente zonale un
# dv/dy che prima non c'era -- misurato, l'1,6% di dv/dx.  Il metro giusto non
# e' quindi zero ma "piccolo rispetto al termine di taglio che dovrebbe
# annullarsi": se la cancellazione fallisse davvero, il residuo sarebbe
# dell'ordine del termine stesso.
for angle in (0.0, 30.0, 60.0):
    without_jet = straight_front(angle, jet_ms=0.0)
    with_jet = straight_front(angle, jet_ms=25.0)
    f_without = fe.frontal_evidence(without_jet["thetaW"], without_jet["u"],
                                    without_jet["v"], LON, LAT,
                                    metrics=GRID)["frontogenesis"]
    f_with = fe.frontal_evidence(with_jet["thetaW"], with_jet["u"], with_jet["v"],
                                 LON, LAT, metrics=GRID)["frontogenesis"]
    axis = (np.abs(without_jet["across"]) < 60.0)[CORE]
    # Termine di taglio che deve annullarsi: 0.5 |grad theta| Esh.
    jet_shear = 25.0 * math.sqrt(2.0 / math.pi) / reference["widthKm"] / 1000.0
    peak_gradient = float(np.nanmax(fe.frontal_evidence(
        without_jet["thetaW"], without_jet["u"], without_jet["v"], LON, LAT,
        metrics=GRID)["gradientMagnitude"][CORE]))
    shear_term = 0.5 * peak_gradient * jet_shear * 100.0 * 10_800.0
    residual = float(np.nanmax(np.abs((f_with - f_without)[CORE][axis])))
    report(residual < 0.10 * shear_term,
           "un getto di puro taglio entra nella frontogenesi (assi mescolati) "
           "con normale a %.0f gradi" % angle,
           "normale %4.0f gr: residuo %.4f, cioe' il %.1f%% del termine di "
           "taglio che si deve cancellare" % (angle, residual,
                                              100.0 * residual / shear_term))

# Lo stesso getto deve pero' comparire tutto nella vorticita'.
jet_case = straight_front(0.0, jet_ms=25.0, deformation=0.0, convergence=0.0)
jet_evidence = fe.frontal_evidence(jet_case["thetaW"], jet_case["u"],
                                   jet_case["v"], LON, LAT, metrics=GRID)
expected_vorticity = 25.0 * math.sqrt(2.0 / math.pi) / reference["widthKm"] / 1000.0 * 1e5
measured_vorticity = float(np.nanmax(jet_evidence["vorticity1e5"][CORE]))
report(abs(measured_vorticity / expected_vorticity - 1.0) < 0.05,
       "la vorticita' del getto non corrisponde al taglio analitico",
       "vorticita' misurata %.2f contro %.2f (unita' 1e-5 s-1)"
       % (measured_vorticity, expected_vorticity))

# --------------------------------------------------------------------------
print("\n4) Geometria: l'asse ritrovato e' dritto e sta dove deve stare")
# --------------------------------------------------------------------------
# Metro del progetto, docs/algoritmo_fronti.md: gradi di curva ogni 20 km.
# Il prodotto misurato in linea prima di questo motore stava a 12,5.
MAX_TURN_DEG_PER_20KM = 6.0
MAX_OFFSET_KM = 40.0
for angle in (0.0, 30.0, 60.0, 90.0):
    case = straight_front(angle)
    evidence, field, points, lines = analyse(case)
    if not lines:
        report(False, "nessuna linea con normale a %.0f gradi" % angle)
        continue
    polished = fe.polish_line(fe.resample_km(lines[0], 8.0), field, LON, LAT,
                              metrics=GRID)
    turn = fe.mean_turn_deg_per_km(polished)
    offset = float(np.max(np.abs(signed_distance(
        polished[:, 0], polished[:, 1], case["normal"]))))
    on_axis = float(np.mean(np.abs(signed_distance(
        points["lon"], points["lat"], case["normal"])) < 25.0))
    report(turn < MAX_TURN_DEG_PER_20KM and offset < MAX_OFFSET_KM and on_axis > 0.85,
           "geometria fuori tolleranza con normale a %.0f gradi" % angle,
           "normale %4.0f gr: L=%5.0f km, curva %.2f gr/20km, scarto max %.1f km, "
           "punti di cresta sull'asse %.0f%%"
           % (angle, fe.line_length_km(polished), turn, offset, 100.0 * on_axis))

# Garanzia strutturale: nessuna linea sopra il limite di curvatura dichiarato
# puo' arrivare alla mappa, qualunque cosa facciano le regole a monte.  Serve
# perche' e' successo: una ricucitura ha saldato gomiti in linee da 500 km e
# ha pubblicato 27 gradi ogni 20 km, il doppio del prodotto che questo lavoro
# sostituisce.  Il controllo si fa su una linea costruita apposta ruvida, non
# su un campo, cosi' non dipende da quale regola a monte l'abbia prodotta.
elbow = np.array([[10.0, 41.0], [10.0, 42.0], [11.0, 42.0], [11.0, 43.0],
                  [12.0, 43.0], [12.0, 44.0], [13.0, 44.0], [13.0, 45.0]])
report(fe.mean_turn_deg_per_km(elbow) > fe.MAX_PUBLISHED_TURN_DEG_PER_20KM,
       "la linea di prova a gomiti non e' abbastanza ruvida per il test",
       "linea a gomiti: %.1f gr/20km contro un limite di %.1f"
       % (fe.mean_turn_deg_per_km(elbow), fe.MAX_PUBLISHED_TURN_DEG_PER_20KM))
rough_case = straight_front(0.0)
rough_evidence = fe.frontal_evidence(
    rough_case["thetaW"], rough_case["u"], rough_case["v"], LON, LAT,
    metrics=GRID, theta_w_upper=rough_case["upper"],
)
rough_fields = {
    "evidence": rough_evidence,
    "abzGradient": fl.adjacent_baroclinic_zone(
        rough_evidence["gradientMagnitude"], rough_evidence["gradientEast"],
        rough_evidence["gradientNorth"], LON, LAT, search_km=fe.SYNOPTIC_SIGMA_KM,
    ),
    "longitudes": LON, "latitudes": LAT,
}
report(not fe.score_lines([elbow], rough_fields, min_length_km=100.0),
       "una linea a gomiti supera la garanzia di curvatura",
       "linee accettate da score_lines: %d"
       % len(fe.score_lines([elbow], rough_fields, min_length_km=100.0)))

# Un solo confine deve dare una sola linea: senza soppressione dei non massimi
# e senza rivendicare il vicinato della catena ne usciva una gemella.
case = straight_front(0.0)
evidence, field, points, lines = analyse(case)
long_lines = [line for line in lines if fe.line_length_km(line) > 250.0]
report(len(long_lines) == 1,
       "un fronte rettilineo produce piu' di una linea pubblicabile",
       "linee sopra i 250 km: %d (la piu' lunga %.0f km)"
       % (len(long_lines), fe.line_length_km(lines[0])))

# La rifinitura non deve scivolare via dalla cresta.
raw = fe.resample_km(lines[0], 8.0)
polished = fe.polish_line(raw, field, LON, LAT, metrics=GRID)
raw_value = float(np.nanmean(fl._sample(field, raw, LON, LAT,
                                        GRID["dlon"], GRID["dlat"])))
polished_value = float(np.nanmean(fl._sample(field, polished, LON, LAT,
                                             GRID["dlon"], GRID["dlat"])))
report(polished_value >= raw_value - 1.0e-3,
       "la rifinitura variazionale allontana la linea dalla cresta",
       "valore medio del campo: grezza %.4f, rifinita %.4f"
       % (raw_value, polished_value))

# --------------------------------------------------------------------------
print("\n5) Il condizionamento: rampe C2 e campo di localizzazione di ordine basso")
# --------------------------------------------------------------------------
# La rampa cubica ha la derivata seconda discontinua agli estremi, e la
# geometria e' fatta di derivate seconde.  Serve la quintica.
probe = np.linspace(-0.5, 1.5, 4001)
ramp = fe._smoothstep(probe, 0.0, 1.0)
step = probe[1] - probe[0]
second = np.gradient(np.gradient(ramp, step), step)
edge = int(0.02 / step)
inner = int(0.5 / step)
report(abs(second[inner - edge]) < 0.02 * np.max(np.abs(second)),
       "la rampa non e' C2 agli estremi (derivata seconda non nulla)",
       "derivata seconda subito dentro la rampa %.4f, massimo %.2f"
       % (second[inner - edge], np.max(np.abs(second))))

# La curvatura misurata sulla cresta deve restare vicino a quella analitica:
# e' l'indicatore che nel campo derivato non e' rientrata una derivata alta.
case = straight_front(0.0)
evidence, field, points, lines = analyse(case)
analytic_curvature = fe.ridge_curvature_scale() * float(np.nanmax(field[CORE]))
measured_curvature = float(np.median(np.abs(points["curvature"])))
report(0.5 < measured_curvature / analytic_curvature < 2.0,
       "la curvatura della cresta e' dominata dal rumore di griglia",
       "curvatura mediana %.2e contro analitica %.2e (rapporto %.2f)"
       % (measured_curvature, analytic_curvature,
          measured_curvature / analytic_curvature))

# --------------------------------------------------------------------------
print("\n6) Il confine termico orografico deve essere respinto")
# --------------------------------------------------------------------------
north_km = (LATG - 45.0) * fl.EARTH_KM_PER_DEG
barrier = 1000.0 * (1.0 + erf(north_km / (np.sqrt(2.0) * WIDTH)))
locked = {
    # Aria fredda intrappolata sul rilievo, calda in pianura: il gradiente
    # termico e' per costruzione allineato con la pendenza.
    "thetaW": 288.0 - 0.5 * DELTA * erf(north_km / (np.sqrt(2.0) * WIDTH)),
    # Poco profondo: a 700 hPa non resta niente.
    "upper": np.full_like(LONG, 288.0),
    # Vento debole e uniforme: niente deformazione, niente convergenza.
    "u": np.full_like(LONG, 3.0),
    "v": np.full_like(LONG, 1.0),
}
locked_evidence, locked_field, locked_points, locked_lines = analyse(
    locked, terrain=barrier)
report(float(np.nanmax(locked_evidence["probability"][CORE])) < 0.10,
       "il contrasto termico orografico raccoglie troppa evidenza",
       "probabilita' massima %.4f, frontogenesi massima %.5f"
       % (np.nanmax(locked_evidence["probability"][CORE]),
          np.nanmax(locked_evidence["frontogenesis"][CORE])))
report(not locked_lines,
       "il contrasto termico orografico viene comunque tracciato",
       "linee prodotte: %d" % len(locked_lines))

# Il testimone del terreno deve essere decisivo su un caso marginale, non
# decorativo: stessa termica, un filo di dinamica, ma saldato alla pendenza.
# "Marginale" qui vuol dire frontogenesi debole -- circa un terzo di quella
# del fronte di riferimento -- perche' e' esattamente la' che la domanda "di
# chi e' questo contrasto, dell'atmosfera o della montagna?" ha un senso.
marginal = {
    "thetaW": locked["thetaW"],
    "upper": 288.0 - 0.5 * (0.55 * DELTA) * erf(north_km / (np.sqrt(2.0) * WIDTH)),
    "u": np.full_like(LONG, 4.0) - 0.10e-5 * ((LONG - LON0)
                                              * fl.EARTH_KM_PER_DEG * COS) * 1000.0,
    "v": -0.175e-5 * north_km * 1000.0,
}
free = fe.frontal_evidence(marginal["thetaW"], marginal["u"], marginal["v"],
                           LON, LAT, metrics=GRID,
                           theta_w_upper=marginal["upper"])
welded = fe.frontal_evidence(marginal["thetaW"], marginal["u"], marginal["v"],
                             LON, LAT, metrics=GRID,
                             theta_w_upper=marginal["upper"], terrain=barrier)
# Il confronto giusto e' in log-odds, non in probabilita': la probabilita'
# satura e nasconde quanto pesa davvero il testimone.
peak = np.unravel_index(np.nanargmax(free["logit"][CORE]), free["logit"][CORE].shape)
free_logit = float(free["logit"][CORE][peak])
welded_logit = float(welded["logit"][CORE][peak])
engaged = (free_logit - welded_logit) / fe._WEIGHTS["terrain"]
report(engaged > 0.70,
       "sul caso marginale il testimone del terreno resta quasi spento",
       "stesso confine: log-odds %.3f senza orografia, %.3f ancorato alla "
       "pendenza; testimone impegnato al %.0f%% del suo peso"
       % (free_logit, welded_logit, 100.0 * engaged))

# E il contrario: un fronte vero che attraversa la barriera non va punito.
crossing = straight_front(0.0)   # fronte meridiano, barriera zonale
free_front = fe.frontal_evidence(crossing["thetaW"], crossing["u"], crossing["v"],
                                 LON, LAT, metrics=GRID,
                                 theta_w_upper=crossing["upper"])
over_barrier = fe.frontal_evidence(crossing["thetaW"], crossing["u"], crossing["v"],
                                   LON, LAT, metrics=GRID,
                                   theta_w_upper=crossing["upper"], terrain=barrier)
loss = (float(np.nanmax(free_front["probability"][CORE]))
        - float(np.nanmax(over_barrier["probability"][CORE])))
report(loss < 0.02,
       "un fronte che attraversa il rilievo viene penalizzato come un artefatto",
       "perdita di probabilita' attraversando la barriera: %.4f" % loss)

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
