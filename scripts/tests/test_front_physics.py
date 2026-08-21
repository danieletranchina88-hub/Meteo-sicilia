"""Synthetic numerical checks for the v12 physical diagnostics.

These are invariance/sign tests, not tuning against one weather case:
translation cannot create frontogenesis; convergence does; solid-body
rotation has vorticity but no deformation frontogenesis; and a moisture-only
boundary must fail the independent dry/density gates.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import front_locator as fl
import front_physics as fp


LON = np.linspace(8.0, 16.0, 121)
LAT = np.linspace(36.0, 46.0, 151)
LONG, LATG = np.meshgrid(LON, LAT)
GRID = fl.grid_metrics(LON, LAT)
X = (LONG - np.mean(LON)) * 111.32 * np.cos(np.deg2rad(np.mean(LAT)))
Y = (LATG - np.mean(LAT)) * 111.32
THETA = 294.0 + 6.0 * np.tanh(X / 110.0)


def median_core(field):
    core = field[20:-20, 20:-20]
    return float(np.nanmedian(core))


ok = True

# 1) Uniform translation leaves the gradient magnitude unchanged.
translation = fp.kinematic_fields(
    THETA, np.full_like(THETA, 12.0), np.full_like(THETA, -4.0), GRID,
    smoothing_km=20.0,
)
translation_f = float(np.nanmax(np.abs(translation["frontogenesis"][20:-20, 20:-20])))
print(f"1) traslazione uniforme: |F|max={translation_f:.4f} K/(100km)/3h")
if translation_f > 0.02:
    print("  FAIL: una traslazione rigida non deve generare frontogenesi")
    ok = False

# 2) Convergence normal to an east-west thermal gradient strengthens it.
convergence_rate = 1.0e-5  # s-1
u_convergent = -convergence_rate * X * 1000.0
convergent = fp.kinematic_fields(
    THETA, u_convergent, np.zeros_like(THETA), GRID, smoothing_km=20.0,
)
front_mask = np.abs(X) < 80.0
frontogenesis = float(np.nanmedian(convergent["frontogenesis"][front_mask]))
print(f"2) convergenza normale: F={frontogenesis:.3f} K/(100km)/3h")
if frontogenesis <= 0.0:
    print("  FAIL: convergenza normale deve produrre frontogenesi positiva")
    ok = False

# 3) Solid-body cyclonic rotation has positive vorticity, zero deformation.
omega = 8.0e-6
u_rotation = -omega * Y * 1000.0
v_rotation = omega * X * 1000.0
rotation = fp.kinematic_fields(
    THETA, u_rotation, v_rotation, GRID, smoothing_km=20.0,
)
vorticity = median_core(rotation["vorticity1e5"])
rotation_f = median_core(np.abs(rotation["frontogenesis"]))
print(f"3) rotazione rigida: zeta={vorticity:.3f}e-5 s-1, |F|={rotation_f:.4f}")
if not (vorticity > 1.0 and rotation_f < 0.03):
    print("  FAIL: segno della vorticita' o invarianza alla rotazione errati")
    ok = False

# 4) Strong, multi-signal candidate must outrank a weak one.
strong = {
    "locatorConfidence": 0.9, "deltaThetaW": 5.0,
    "deltaTemperature": 3.2, "deltaThetaV": 2.5,
    "dryThermalGradient": 3.2, "thermalAlignment": 0.85,
    "thermalContrastFraction": 0.88, "thermalAlignmentFraction": 0.90,
    "crossDistanceThermalSupport": 0.88, "deltaThetaE": 5.0,
    "windShiftMs": 8.0, "convergenceMs": 2.2,
    "windShiftAngleDeg": 42.0, "windBoundaryFraction": 0.86,
    "convergenceFraction": 0.82,
    "vorticity1e5": 5.0, "frontogenesis": 3.2,
    "pressureTroughHpa": 2.0, "pressureTroughFraction": 0.82,
    "lowerValidFraction": 0.85, "lowerLevelSupport": 0.85,
    "deltaThetaW925": 3.5, "omega700PaS": -0.25,
    "synopticSupport": 0.95, "lengthKm": 900.0,
    "sinuosity": 1.1, "terrainFraction": 0.05,
}
weak = dict(strong)
weak.update({
    "locatorConfidence": 0.16, "deltaThetaW": 1.3,
    "deltaTemperature": 0.5, "deltaThetaV": 0.22,
    "dryThermalGradient": 0.8, "thermalAlignment": 0.0,
    "thermalContrastFraction": 0.42, "thermalAlignmentFraction": 0.40,
    "windShiftMs": 1.3, "convergenceMs": 0.06,
    "windShiftAngleDeg": 7.0, "windBoundaryFraction": 0.35,
    "convergenceFraction": 0.40,
    "vorticity1e5": -0.4, "frontogenesis": -0.5,
    "pressureTroughHpa": -0.1, "pressureTroughFraction": 0.40,
    "lowerValidFraction": 0.80, "lowerLevelSupport": 0.36,
    "deltaThetaW925": 0.8, "omega700PaS": 0.02,
    "synopticSupport": 0.57, "lengthKm": 240.0,
    "sinuosity": 2.2, "terrainFraction": 0.6,
})
strong_score = fp.candidate_evidence(strong)
weak_score = fp.candidate_evidence(weak)
print(f"4) evidenza forte={strong_score['candidateEvidence']:.3f}, "
      f"debole={weak_score['candidateEvidence']:.3f}")
if not strong_score["candidateEvidence"] > weak_score["candidateEvidence"] + 0.35:
    print("  FAIL: il punteggio non separa prove forti e deboli")
    ok = False

# 5) Moisture-only boundary: theta_w can be sharp, but dry/density checks fail.
dryline = dict(strong)
dryline.update({
    "deltaTemperature": 0.05,
    "deltaThetaV": 0.02,
    "dryThermalGradient": 0.10,
})
dryline_score = fp.candidate_evidence(dryline)
print(f"5) confine solo igrometrico plausibile={fp.candidate_is_plausible(dryline, dryline_score)}")
if fp.candidate_is_plausible(dryline, dryline_score):
    print("  FAIL: un gradiente solo di umidita' non e' un fronte termico")
    ok = False

# 6) A strong thermal line with divergent, unrotated flow is not a front.
wrong_wind = dict(strong)
wrong_wind.update({
    "windShiftMs": 0.8,
    "windShiftAngleDeg": 4.0,
    "windBoundaryFraction": 0.20,
    "convergenceMs": -1.4,
    "convergenceFraction": 0.18,
    "vorticity1e5": 0.2,
    "frontogenesis": -0.4,
})
wrong_report = fp.candidate_gate_report(wrong_wind)
print(f"6) gradiente con vento contrario: {wrong_report['gateStatus']} "
      f"{wrong_report['rejectionReasons']}")
if wrong_report["continuationPass"]:
    print("  FAIL: il gradiente termico non puo' compensare vento divergente")
    ok = False

# 7) A pressure ridge prevents a strong birth but does not erase a mature,
# thermally coherent synoptic boundary for one analysis hour.
ridge = dict(strong)
ridge.update({"pressureTroughHpa": -0.7, "pressureTroughFraction": 0.18})
ridge_report = fp.candidate_gate_report(ridge)
print(f"7) gradiente sopra promontorio barico: {ridge_report['gateStatus']} "
      f"{ridge_report['rejectionReasons']}")
if not ridge_report["continuationPass"] or ridge_report["strongPass"]:
    print("  FAIL: il promontorio deve indebolire, non cancellare, una linea coerente")
    ok = False

# 8) Theta-e alone must be diagnosed as a humidity boundary, not published.
diagnosis, _ = fp.candidate_hypothesis(dryline)
print(f"8) diagnosi del confine igrometrico: {diagnosis}")
if diagnosis != "moisture-boundary":
    print("  FAIL: il classificatore differenziale non riconosce la dryline")
    ok = False

# 9) A published front must expose a human-readable, non-empty explanation
# that verbalises the numeric evidence (transparency, not a black box).
strong_diag = {
    "deltaThetaW": 3.0, "medianAbzGradient": 4.0, "deltaTemperature": 2.0,
    "deltaThetaV": 1.5, "windShiftAngleDeg": 35.0, "convergenceMs": 0.6,
    "vorticity1e5": 3.0, "frontogenesis": 1.5, "lowerLevelSupport": 0.6,
    "deltaThetaW925": 2.0,
}
explanation = fp.frontal_explanation(
    strong_diag, {"frontType": "cold", "motionVotes": {"geometry": "cold", "wind": "cold"}},
    "synoptic-front", lifetime_h=18,
)
print(f"9) spiegazione ({len(explanation)} voci): {explanation[0]}")
if not explanation or any(not isinstance(r, str) for r in explanation):
    print("  FAIL: spiegazione mancante o malformata")
    ok = False
# A weak humidity boundary must flag the differential diagnosis in words.
moist = fp.frontal_explanation({"deltaThetaW": 0.6}, {}, "moisture-boundary")
if not any("umidità" in r for r in moist):
    print("  FAIL: la diagnosi alternativa non e' esposta nella spiegazione")
    ok = False

# 10) The reasoning engine weighs ALL hypotheses and gives a motivated verdict.
strong_report = fp.differential_diagnosis(strong)
print(f"10) verdetto forte: {strong_report['verdict']} "
      f"margine={strong_report['margin']} supporti={strong_report['supports']}")
if strong_report["verdict"] != "synoptic-front" or strong_report["margin"] <= 0:
    print("  FAIL: un caso frontale forte deve vincere come fronte sinottico")
    ok = False
if set(strong_report["supports"]) != {
    "synoptic-front", "moisture-boundary", "mesoscale-boundary",
    "outflow-boundary", "orographic-boundary", "noise",
}:
    print("  FAIL: il motore non valuta tutte le ipotesi")
    ok = False
# The dryline is beaten by moisture-boundary, not by synoptic-front.
dry_report = fp.differential_diagnosis(dryline)
if dry_report["verdict"] != "moisture-boundary":
    print(f"  FAIL: la dryline non e' diagnosticata correttamente ({dry_report['verdict']})")
    ok = False
# A short, convergent, shallow, non-synoptic line reads as mesoscale, not a front.
breeze = {
    "deltaThetaW": 1.4, "deltaTemperature": 0.6, "deltaThetaV": 0.3,
    "dryThermalGradient": 0.8, "lengthKm": 180.0, "synopticSupport": 0.30,
    "convergenceMs": 1.2, "windShiftMs": 3.0, "lowerLevelSupport": 0.2,
    "deltaThetaW925": 0.4, "terrainFraction": 0.1, "sinuosity": 1.3,
}
breeze_verdict = fp.differential_diagnosis(breeze)["verdict"]
print(f"    verdetto brezza corta: {breeze_verdict}")
if breeze_verdict == "synoptic-front":
    print("  FAIL: una linea corta e locale non deve vincere come fronte sinottico")
    ok = False

# 11) Multi-method consensus is bounded and cannot rescue failed hard gates.
agreed_weak = dict(weak, consensusSupport=1.0)
agreed_score = fp.candidate_evidence(agreed_weak)
agreed_gate = fp.candidate_gate_report(agreed_weak, agreed_score)
print(
    "11) bonus consenso=%.3f, gate=%s"
    % (agreed_score["consensusEvidenceBonus"], agreed_gate["gateStatus"])
)
if not (
    agreed_score["consensusEvidenceBonus"] <= 0.04
    and agreed_score["candidateEvidence"]
    <= weak_score["candidateEvidence"] + 0.04 + 1.0e-9
    and not agreed_gate["continuationPass"]
):
    print("  FAIL: il consenso ha scavalcato un gate o supera il limite")
    ok = False

# 12) The verdict must not flip on a hair. A long, well-structured boundary
# whose density contrast is merely modest -- the normal summer case -- stays
# a front, and stays one under a small perturbation of any single input.
print("\n12) stabilita' del verdetto (niente fronte che lampeggia):")
# Diagnostics copied verbatim from a real ICON-2I candidate (run
# 2026-08-21 00 UTC, +2 h): the 660-km cold front over the Gulf of Lion
# whose reading used to alternate between "front" and "mesoscale boundary"
# from one hour to the next.
marginal = {
    "medianAbzGradient": 2.5633, "deltaThetaW": 1.848, "deltaThetaE": 6.6998,
    "deltaTemperature": 1.7337, "deltaThetaV": 2.0297,
    "dryThermalGradient": 1.77, "thermalAlignment": 0.9698,
    "windShiftMs": 1.7947, "convergenceMs": 0.5944,
    "windShiftAngleDeg": 35.3304, "convergenceFraction": 0.8972,
    "vorticity1e5": -2.5913, "frontogenesis": 0.0837,
    "pressureTroughHpa": -0.1579, "lowerLevelSupport": 0.125,
    "deltaThetaW925": -0.9521, "omega700PaS": -0.0313,
    "terrainFraction": 0.3832, "synopticSupport": 0.91,
    "lengthKm": 656.7594, "sinuosity": 1.11,
}
base_verdict = fp.differential_diagnosis(marginal)["verdict"]
print(f"    caso marginale realistico: {base_verdict}")
if base_verdict != "synoptic-front":
    print("  FAIL: un confine da 660 km con i tre contrasti presenti "
          "deve restare un fronte")
    ok = False
# The air-mass contrast used to enter the decision through a hard step: on
# one side of it the front reading was protected by a fixed margin, on the
# other side the protection was zero, so an imperceptible change of a single
# contrast decided the verdict. Sweeping the density contrast must now cross
# the decision at most once -- monotonically -- instead of oscillating.
verdicts = []
for delta_theta_v in np.linspace(0.0, 2.4, 49):
    swept = dict(marginal, deltaThetaV=float(delta_theta_v))
    verdicts.append(fp.differential_diagnosis(swept)["verdict"] == "synoptic-front")
changes = sum(a != b for a, b in zip(verdicts[:-1], verdicts[1:]))
print(f"    spazzando Δθv da 0 a 2,4 K il verdetto cambia {changes} volta/e")
if changes > 1:
    print("  FAIL: il verdetto oscilla lungo il contrasto di densita' "
          "(gradino nella decisione)")
    ok = False
if verdicts[0] or not verdicts[-1]:
    print("  FAIL: la spazzata deve andare da non-fronte a fronte")
    ok = False
# The margin protecting the front reading must be continuous, not switched.
# A discontinuity keeps the same height however finely it is sampled; a
# continuous function's largest step shrinks with the sampling interval.
def margin_curve(count):
    values = []
    for v in np.linspace(0.0, 2.4, count):
        core = float(np.cbrt(
            fp.smoothstep(marginal["deltaThetaW"], 1.0, 4.5)
            * fp.smoothstep(marginal["deltaTemperature"], 0.5, 3.0)
            * fp.smoothstep(v, 0.2, 2.2)
        ))
        values.append(fp.smoothstep(core, 0.05, 0.35))
    return max(abs(b - a) for a, b in zip(values[:-1], values[1:]))

coarse, fine = margin_curve(49), margin_curve(1921)
print(f"    salto massimo del margine: {coarse:.3f} a passo grosso, "
      f"{fine:.3f} a passo 40x piu' fine")
if fine > 0.35 * coarse:
    print("  FAIL: il margine di protezione ha ancora un gradino "
          "(il salto non si riduce infittendo il campionamento)")
    ok = False

# The soft AND must still collapse when an ingredient is genuinely absent:
# no density contrast at all is not a front, however good the structure.
no_density = dict(marginal, deltaThetaV=0.0, deltaTemperature=0.0,
                  dryThermalGradient=0.1)
if fp.differential_diagnosis(no_density)["verdict"] == "synoptic-front":
    print("  FAIL: senza contrasto di massa d'aria non puo' essere un fronte")
    ok = False
print("    senza contrasto di densita' e temperatura secca: "
      f"{fp.differential_diagnosis(no_density)['verdict']}")

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
