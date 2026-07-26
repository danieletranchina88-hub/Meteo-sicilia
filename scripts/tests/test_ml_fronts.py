"""Numerical and policy checks for the supervised/physical fusion."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from meteo_analysis.ml import features
from meteo_analysis.ml.fusion import fuse_candidate

ok = True

# Bolton theta-e implementation must agree with MetPy within numerical/formula
# differences, and never use Celsius where Kelvin is expected.
from metpy.calc import equivalent_potential_temperature
from metpy.units import units

t = np.array([[283.15, 288.15]])
q = np.array([[0.004, 0.008]])
ours = features._thermodynamics(850.0, t, q)
reference = equivalent_potential_temperature(
    850 * units.hPa,
    t * units.kelvin,
    ours["dewpoint"] * units.kelvin,
).magnitude
theta_e_error = float(np.max(np.abs(ours["theta_e"] - reference)))
print(f"theta-e: errore massimo rispetto a MetPy={theta_e_error:.3f} K")
if theta_e_error > 1.5:
    print("FAIL: theta-e operativa non coerente con MetPy")
    ok = False

# Uniform translation must not manufacture Petterssen frontogenesis.
lat = np.linspace(36, 46, 80)
lon = np.linspace(5, 20, 90)
lon2d, _ = np.meshgrid(lon, lat)
theta = 290 + 5 * np.tanh((lon2d - 12) / 1.2)
fgen = features._frontogenesis(
    theta, np.full_like(theta, 12.0), np.full_like(theta, -3.0), lat, lon
)
translation_error = float(np.nanmax(np.abs(fgen[5:-5, 5:-5])))
print(f"frontogenesi in traslazione uniforme={translation_error:.6f}")
if translation_error > 0.02:
    print("FAIL: la traslazione rigida crea frontogenesi")
    ok = False

# Contraction normal to the theta gradient must strengthen that gradient.
earth_radius = features.EARTH_RADIUS_M
x_m = (
    np.deg2rad(lon - lon.mean())[None, :]
    * earth_radius * np.cos(np.deg2rad(lat))[:, None]
)
contraction = features._frontogenesis(
    theta, -1.0e-5 * x_m, np.zeros_like(theta), lat, lon
)
front_mask = np.abs(lon2d - 12.0) < 1.0
contraction_value = float(np.nanmedian(contraction[front_mask]))
print(f"frontogenesi in contrazione normale={contraction_value:.4f}")
if contraction_value <= 0:
    print("FAIL: la contrazione normale deve essere frontogenetica")
    ok = False

strong = {
    "candidateEvidence": 0.64,
    "deltaThetaW": 2.0, "deltaTemperature": 1.2, "deltaThetaV": 0.7,
    "dryThermalGradient": 1.4, "thermalAlignment": 0.4,
    "thermalContrastFraction": 0.7, "thermalAlignmentFraction": 0.7,
    "synopticSupport": 0.8,
}
passed = {
    "continuationPass": True, "strongPass": True, "gateStatus": "strong",
    "rejectionReasons": [], "diagnosis": "synoptic-front",
}
fused, fused_gates = fuse_candidate(
    strong, passed, {"median": 0.90, "q75": 0.95, "supportFraction": 0.9},
    model_threshold=0.30,
)
print("conferma ML, bonus:", fused["fusionEvidenceBonus"])
if not (
    fused_gates["strongPass"]
    and 0 < fused["fusionEvidenceBonus"] <= 0.05
    and fused["candidateEvidence"] <= 0.69
):
    print("FAIL: la conferma ML non è limitata o altera i gate fisici")
    ok = False

# High ML can never override a pressure-only/orographic/mesoscale diagnosis.
rejected = {
    "continuationPass": False, "strongPass": False, "gateStatus": "rejected",
    "rejectionReasons": ["thermal-core", "synoptic-structure", "cold-pool"],
    "diagnosis": "cold-pool",
}
blocked, blocked_gates = fuse_candidate(
    strong, rejected, {"median": 0.99, "q75": 1.0, "supportFraction": 1.0},
    model_threshold=0.30,
)
print("contraddizione fisica:", blocked["fusionDecision"])
if blocked_gates["continuationPass"] or blocked["mlAssisted"]:
    print("FAIL: il ML ha scavalcato una contraddizione causale")
    ok = False

# A marginal thermal near-pass may only become continuation-grade; normal
# physical tracking still requires a real strong anchor and >=3 h persistence.
near = dict(strong)
near.update({
    "candidateEvidence": 0.44, "deltaThetaW": 1.05,
    "deltaTemperature": 0.52, "deltaThetaV": 0.22,
    "dryThermalGradient": 0.70, "thermalAlignment": 0.045,
    "thermalContrastFraction": 0.43, "thermalAlignmentFraction": 0.43,
})
near_gate = {
    "continuationPass": False, "strongPass": False, "gateStatus": "rejected",
    "rejectionReasons": ["thermal-core", "wind-boundary"],
    "diagnosis": "synoptic-front",
}
assisted, assisted_gate = fuse_candidate(
    near, near_gate, {"median": 0.82, "q75": 0.9, "supportFraction": 0.7},
    model_threshold=0.30,
)
print("near-pass:", assisted["fusionDecision"], assisted_gate["gateStatus"])
if not (
    assisted_gate["continuationPass"]
    and not assisted_gate["strongPass"]
    and assisted["mlAssisted"]
):
    print("FAIL: near-pass ML conservativo non applicato")
    ok = False

if not ok:
    raise SystemExit(1)
print("OK: termodinamica ML e guardrail di fusione verificati.")
