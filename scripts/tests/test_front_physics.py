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

print("\nESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
