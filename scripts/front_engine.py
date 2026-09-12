"""Synoptic frontal engine: declared scale, evidence fusion, variational geometry.

Three layers, in order.

**Scale.**  A front is a synoptic object.  Diagnosing it on a mesoscale field
produces a line that follows mesoscale wobbles, which is the measured cause of
the serpentine geometry this module replaces.  The analysis scale here is a
Gaussian low-pass at ``SYNOPTIC_SIGMA_KM``, derived in ``_SIGMA_DERIVATION``
from the filters used by peer-reviewed frontal climatologies rather than
chosen by eye.

**Evidence.**  A single continuous field combines quasi-independent physical
witnesses in log-odds.  Frontogenesis, vertical coherence and the absence of
terrain locking enter as *necessary* conditions -- near zero they contribute
large negative log-odds -- so that a long, persistent, terrain-anchored
thermal boundary cannot accumulate enough positive evidence to be published.
Length and persistence deliberately carry **no** positive weight here: they
are the defining properties of an orographic artefact, not of a front.

**Geometry.**  The axis is the ridge of that field, extracted with the
sub-pixel Hessian method of Steger (1998, IEEE TPAMI 20(2):113-125), linked
along its own orientation with a hard turn cap, and finished by alternating
elastic smoothing with a re-projection onto the local evidence maximum.  No
zero contour of a third derivative, and no 8-connected path: both quantise
direction and were the second measured cause of the old geometry.

The score is an evidence score in probabilistic form.  It is **not** a
calibrated probability: no labelled frontal archive exists in this repository
against which it could be calibrated.

References
----------
Renard & Clarke (1965), Mon. Wea. Rev. 93, 547-556 -- thermal front parameter.
Hewson (1998), Met. Apps 5, 37-65 -- objective fronts, masking, K3 speed rule.
Petterssen (1936) -- frontogenesis function in deformation form.
Sansom & Catto (2024), Geosci. Model Dev. 17, 6137-6153 -- ERA-Interim/ERA5
    objective front climatology; source of the smoothing equivalence below.
Steger (1998), IEEE TPAMI 20(2), 113-125 -- unbiased sub-pixel line detector.
"""

from __future__ import annotations

import numpy as np

import front_locator as fl


# --------------------------------------------------------------------------
# Layer 1 -- declared analysis scale
# --------------------------------------------------------------------------
#
# Sansom & Catto (2024) smooth theta_w with n passes of a five-point mean
# before diagnosing fronts: n = 8 on ERA-Interim (~79 km) and n = 96 on ERA5
# (~28 km).  One pass of that stencil has variance 2*d^2/5 per axis, so n
# passes give sigma = sqrt(2*n/5)*d:
#
#     ERA-Interim: sqrt(2*8/5)  * 79 km = 141 km
#     ERA5:        sqrt(2*96/5) * 28 km = 173 km
#
# The two independent choices agree on the same physical scale, which is the
# scale of the phenomenon rather than of either grid.  150 km sits between
# them.  This is the single number that decides whether the published line is
# a front or a mesoscale boundary, so it is derived here and not tuned.
_SIGMA_DERIVATION = "sqrt(2n/5)*d; Sansom & Catto 2024: 8 passes @79 km, 96 @28 km"
SYNOPTIC_SIGMA_KM = 150.0

# Positional refinement only: the axis may be sharpened inside a corridor
# around the synoptic ridge, never relocated by it.
REFINE_SIGMA_KM = 60.0
CORRIDOR_KM = 120.0


def five_point_pass_sigma_km(passes: int, spacing_km: float) -> float:
    """Sigma of ``passes`` applications of a five-point mean on a grid.

    Exposed because it is the derivation behind ``SYNOPTIC_SIGMA_KM`` and a
    claim of that weight should be executable, not only written in a comment.
    """
    return float(np.sqrt(2.0 * max(int(passes), 0) / 5.0) * float(spacing_km))


# --------------------------------------------------------------------------
# Reference front: where the evidence ramps come from
# --------------------------------------------------------------------------
#
# Thresholds cannot be inherited from the previous implementation because they
# were set at sigma = 45 km and every derivative shrinks when the analysis
# scale widens.  Instead a *declared reference front* fixes them: an
# error-function transition of DELTA_K kelvin across a zone of half-width
# REFERENCE_HALF_WIDTH_KM.  Smoothing at sigma widens it in quadrature, so the
# peak gradient of the smoothed front is analytic and the ramps follow from
# it.  These are prudential design assumptions, not calibrated parameters.
# Ancorato alle soglie che questo progetto ha gia' tarato sul proprio dominio,
# non a un fronte da manuale.  ``front_locator`` maschera con la zona baroclina
# adiacente fra 0,65 e 1,10 K/100 km a sigma 100 km, ed e' la calibrazione che
# sul run del 12 settembre produceva 81 candidati.  Il picco di gradiente di un
# fronte erf scala come 1/w', quindi passando da sigma 100 a sigma 150 quelle
# soglie diventano 1,10 * hypot(80,100)/hypot(80,150) = 0,83 K/100 km, e il
# DELTA_K che le riproduce e' 0,0083 * sqrt(2 pi) * 170 = 3,5 K.
#
# Misurato sul campo ICON-2I pubblicato: a sigma 150 km il gradiente termico
# piu' forte di tutto il dominio vale 2,4 K/100 km e la mediana 0,67.  Con
# DELTA_K = 8 il fronte di riferimento aveva un picco di 1,88 K/100 km, cioe'
# piu' forte del 99esimo percentile del dominio: ogni rampa saturava solo
# nell'1% piu' estremo e il motore non pubblicava nulla.
REFERENCE_DELTA_K = 3.5
REFERENCE_HALF_WIDTH_KM = 80.0
# Relief of a reference mountain barrier (Alps, Apennines, Dinarides seen at
# synoptic scale).  Like the thermal reference it fixes a ramp instead of a
# magic number: smoothing spreads the rise over the same width, so the slope
# a barrier presents to a 150 km analysis is REFERENCE_RELIEF_M / (sqrt(2 pi)
# w'), about 3.5 m/km -- an order smaller than the raw model slope, which is
# why thresholds taken from the native grid do not transfer.
REFERENCE_RELIEF_M = 1500.0


def reference_front_scales(sigma_km: float) -> dict:
    """Analytic signature of the declared reference front at a given scale.

    For theta(x) = (dT/2) erf(x / (sqrt(2) w)) the gradient is Gaussian with
    peak dT / (sqrt(2 pi) w); smoothing at sigma gives w' = hypot(w, sigma).
    The cross-front second derivative of |grad theta| reaches its extremum at
    x = +/- w' with magnitude peak / (w' sqrt(e)), which is the scale of the
    thermal front parameter.
    """
    width = float(np.hypot(REFERENCE_HALF_WIDTH_KM, max(float(sigma_km), 0.0)))
    peak_gradient = REFERENCE_DELTA_K / (np.sqrt(2.0 * np.pi) * width)  # K/km
    tfp_scale = peak_gradient / (width * np.sqrt(np.e))                 # K/km^2
    return {
        "widthKm": width,
        "gradientKPerKm": float(peak_gradient),
        "gradientK100Km": float(peak_gradient * 100.0),
        "tfpKPerKm2": float(tfp_scale),
        # Curvature of |grad theta| across the zone at its own axis: for a
        # Gaussian of width w' it is exactly peak / w'^2.  This is the
        # strength of Hewson's locating variable, and the scale against which
        # the "zone" witness below is measured.
        "zoneCurvatureKPerKm3": float(peak_gradient / (width * width)),
        "reliefSlopeMPerKm": float(
            REFERENCE_RELIEF_M / (np.sqrt(2.0 * np.pi) * width)
        ),
    }


# --------------------------------------------------------------------------
# Layer 2 -- evidence fusion
# --------------------------------------------------------------------------
#
# Weights are declared, not fitted.  What matters is their structure, not
# their third decimal: the three "necessary" witnesses carry enough negative
# weight that no combination of the merely supportive ones can publish a
# boundary which is not being sharpened, is not deep, and is welded to the
# terrain.  The prior is negative because most of the domain is not a front.
_PRIOR_LOGIT = -1.35
_WEIGHTS = {
    "thermal": 1.55,      # necessary: no baroclinic zone, no front
    "frontogenesis": 1.50,  # necessary: a front is actively maintained
    "vertical": 1.25,     # necessary: a front is deep, an inversion is not
    "terrain": 1.40,      # veto: penalty only, never positive
    "zone": 0.85,
    "vorticity": 0.70,
    "convergence": 0.70,
    "windShift": 0.60,
    "pressure": 0.45,
    "ascent": 0.40,
}


def _smoothstep(values: np.ndarray, weak: float, strong: float) -> np.ndarray:
    """Quintic ramp from 0 at ``weak`` to 1 at ``strong`` (either direction).

    The quintic, not the usual cubic Hermite, and the reason is measured.  The
    cubic ``3t^2 - 2t^3`` has a *discontinuous second derivative* at both ends
    of the ramp.  The geometry layer diagonalises the Hessian of the fused
    field, so every one of those knots becomes a line of enormous spurious
    curvature: on an analytic straight front the detector returned twice as
    many phantom ridge points -- lying along the ramp knots, roughly 95 km off
    the axis -- as real ones, with curvature fifty times the physical scale.
    ``6t^5 - 15t^4 + 10t^3`` has zero first *and* second derivative at both
    ends, so the fused field is C2 and its Hessian means what it says.
    """
    weak = float(weak)
    strong = float(strong)
    if abs(strong - weak) < 1.0e-12:
        return np.where(np.asarray(values, dtype=float) >= strong, 1.0, 0.0)
    t = (np.asarray(values, dtype=float) - weak) / (strong - weak)
    t = np.clip(np.where(np.isfinite(t), t, 0.0), 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def _signed(values: np.ndarray, weak: float, strong: float) -> np.ndarray:
    """Evidence in [-1, +1]: -1 is positive evidence *against*, not absence.

    Only for the **necessary** witnesses.  A front without a baroclinic zone,
    without frontogenesis or without depth is not a front, so for those three
    a value below the ramp really is an argument against.
    """
    return 2.0 * _smoothstep(values, weak, strong) - 1.0


def _support(values: np.ndarray, weak: float, strong: float) -> np.ndarray:
    """Evidence in [0, +1] for the **supportive** witnesses: help, never veto.

    The design always said some witnesses are necessary and the rest merely
    supportive; the first implementation then put every one of them through
    the signed ramp, which let a supportive witness veto.  The consequence was
    measured on the real field: with cyclonic vorticity, convergence and the
    cross-front wind shift each contributing a full negative simply for being
    below a synoptic-scale threshold, the strongest baroclinic zone in the
    whole domain reached a probability of 0.13 against a gate of 0.5, and the
    published run produced zero candidates in all 73 hours.  A quiet front is
    still a front; only the necessary conditions may argue against one.
    """
    return _smoothstep(values, weak, strong)


def frontal_evidence(
    theta_w: np.ndarray,
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    *,
    metrics: dict | None = None,
    sigma_km: float = SYNOPTIC_SIGMA_KM,
    theta_w_lower: np.ndarray | None = None,
    theta_w_upper: np.ndarray | None = None,
    pressure: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    terrain: np.ndarray | None = None,
) -> dict:
    """Fuse physical witnesses into one continuous frontal evidence field.

    Returns the log-odds field, its probability transform, the fields the
    geometry layer needs (gradient, tfp) and every individual witness, so a
    published line can always be audited term by term.
    """
    metrics = metrics or fl.grid_metrics(longitudes, latitudes)
    reference = reference_front_scales(sigma_km)

    theta = fl.smooth_km(np.asarray(theta_w, dtype=float), sigma_km, metrics)
    grad_e, grad_n = fl.gradient(theta, metrics)
    grad_mag = np.hypot(grad_e, grad_n)

    # Renard & Clarke thermal front parameter: second derivative of theta_w
    # along its own gradient.  Negative means the gradient weakens towards the
    # warm air, i.e. we are on the warm flank of the baroclinic zone, which is
    # where an analyst draws the front.
    gm_e, gm_n = fl.gradient(grad_mag, metrics)
    safe_mag = np.maximum(grad_mag, 1.0e-9)
    tfp = (gm_e * grad_e + gm_n * grad_n) / safe_mag

    import front_physics as fp

    kinematics = fp.kinematic_fields(
        theta_w, u_wind, v_wind, metrics, smoothing_km=sigma_km
    )
    frontogenesis = kinematics["frontogenesis"]
    vorticity = kinematics["vorticity1e5"]
    convergence = kinematics["convergence1e5"]

    witnesses: dict[str, np.ndarray] = {}

    # Baroclinicity, against the smoothed reference front.
    full_gradient = reference["gradientKPerKm"]
    witnesses["thermal"] = _signed(grad_mag, 0.35 * full_gradient, full_gradient)

    # Zone sharpness -- Hewson's locating variable, as a witness.
    #
    # A first attempt used the warm-side branch of the TFP itself.  Measured
    # on an analytic front it does the opposite of what is wanted: the TFP
    # vanishes *on* the frontal axis by construction and peaks a full zone
    # half-width out into the warm air, so that witness argued against the
    # very line it was meant to support.  What distinguishes a front from a
    # broad baroclinic slope is instead that |grad theta| has a pronounced
    # ridge -- a large negative second derivative across the zone -- which is
    # exactly Hewson's locating variable and peaks on the axis.
    zone_curvature = -fl.directional_curvature(grad_mag, grad_e, grad_n, metrics)
    witnesses["zone"] = _support(
        zone_curvature,
        0.25 * reference["zoneCurvatureKPerKm3"],
        reference["zoneCurvatureKPerKm3"],
    )

    # Petterssen frontogenesis.  The reference value is what a front of the
    # declared strength experiences in a synoptic deformation of 1e-5 s-1,
    # converted to the module's K/(100 km)/(3 h): 0.5 * |grad| * E.
    reference_frontogenesis = 0.5 * full_gradient * 1.0e-5 * 100.0 * 10_800.0
    witnesses["frontogenesis"] = _signed(
        frontogenesis, 0.15 * reference_frontogenesis, reference_frontogenesis
    )

    witnesses["vorticity"] = _support(vorticity, 0.5, 4.0)
    witnesses["convergence"] = _support(convergence, 0.3, 3.0)

    # Cross-front wind shift: the wind turns through a front.  Measured as the
    # component of the wind shear along the thermal gradient direction.
    u_smooth = fl.smooth_km(np.asarray(u_wind, dtype=float), sigma_km, metrics)
    v_smooth = fl.smooth_km(np.asarray(v_wind, dtype=float), sigma_km, metrics)
    u_e, u_n = fl.gradient(u_smooth, metrics)
    v_e, v_n = fl.gradient(v_smooth, metrics)
    normal_e = grad_e / safe_mag
    normal_n = grad_n / safe_mag
    # d(wind)/d(cross-front distance), in (m/s) per 100 km.
    shear_u = (u_e * normal_e + u_n * normal_n) * 100.0
    shear_v = (v_e * normal_e + v_n * normal_n) * 100.0
    wind_shift = np.hypot(shear_u, shear_v)
    witnesses["windShift"] = _support(wind_shift, 1.5, 8.0)

    # Vertical coherence.  An orographic thermal boundary is a shallow pool of
    # air trapped against a slope and fades upward; a front does not.  With no
    # level above, the witness stays neutral instead of inventing support.
    if theta_w_upper is not None:
        upper = fl.smooth_km(np.asarray(theta_w_upper, dtype=float), sigma_km, metrics)
        upper_e, upper_n = fl.gradient(upper, metrics)
        upper_mag = np.hypot(upper_e, upper_n)
        ratio = upper_mag / np.maximum(grad_mag, 1.0e-9)
        witnesses["vertical"] = _signed(ratio, 0.25, 0.70)
    else:
        witnesses["vertical"] = np.zeros_like(grad_mag)

    if theta_w_lower is not None:
        lower = fl.smooth_km(np.asarray(theta_w_lower, dtype=float), sigma_km, metrics)
        lower_e, lower_n = fl.gradient(lower, metrics)
        lower_mag = np.hypot(lower_e, lower_n)
        agreement = _signed(
            lower_mag, 0.25 * full_gradient, 0.80 * full_gradient
        )
        witnesses["vertical"] = 0.5 * (witnesses["vertical"] + agreement)

    if pressure is not None:
        trough = fl.laplacian(
            fl.smooth_km(np.asarray(pressure, dtype=float), sigma_km, metrics),
            metrics,
        ) * 1.0e4  # hPa per (100 km)^2
        witnesses["pressure"] = _support(trough, 0.05, 0.60)
    else:
        witnesses["pressure"] = np.zeros_like(grad_mag)

    if omega is not None:
        ascent = -fl.smooth_km(np.asarray(omega, dtype=float), sigma_km, metrics)
        witnesses["ascent"] = _support(ascent, 0.0, 0.35)  # Pa/s upward
    else:
        witnesses["ascent"] = np.zeros_like(grad_mag)

    # Terrain locking.  A thermal contrast created by the orography has its
    # gradient aligned with the slope: warm air on the plain, cool air up the
    # mountain.  Pure penalty -- it can only ever argue against.
    #
    # Alignment alone will not do, and the reason is geometric: the Alpine
    # barrier runs east-west, so its slope gradient points north, and a real
    # cold front lying east-west across the Po valley has a north-pointing
    # thermal gradient too.  Measured on a synthetic front crossing such a
    # barrier, an alignment-only penalty cost a genuine boundary 1.1 of
    # log-odds and kept it off the map.  What actually distinguishes the
    # mountain's own contrast is that nothing is sharpening it: the penalty is
    # therefore conditioned on the frontogenesis witness, so it can only bite
    # where the boundary is aligned with the slope *and* is not being
    # maintained.  That conjunction is the statement "this contrast belongs to
    # the terrain, not to the flow".
    if terrain is not None:
        height = fl.smooth_km(np.asarray(terrain, dtype=float), sigma_km, metrics)
        terrain_e, terrain_n = fl.gradient(height, metrics)
        terrain_mag = np.hypot(terrain_e, terrain_n)  # m/km
        alignment = np.abs(
            (terrain_e * normal_e + terrain_n * normal_n)
            / np.maximum(terrain_mag, 1.0e-9)
        )
        relief = reference["reliefSlopeMPerKm"]
        maintained = 0.5 * (witnesses["frontogenesis"] + 1.0)   # back to [0, 1]
        locked = (
            _smoothstep(terrain_mag, 0.40 * relief, relief)
            * _smoothstep(alignment, 0.55, 0.90)
            * (1.0 - maintained)
        )
        witnesses["terrain"] = -locked
    else:
        witnesses["terrain"] = np.zeros_like(grad_mag)

    logit = np.full_like(grad_mag, _PRIOR_LOGIT, dtype=float)
    for name, weight in _WEIGHTS.items():
        logit = logit + weight * np.nan_to_num(witnesses[name], nan=0.0)

    invalid = ~np.isfinite(theta) | ~np.isfinite(grad_mag)
    logit = np.where(invalid, np.nan, logit)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))

    return {
        "logit": logit,
        "probability": np.where(invalid, np.nan, probability),
        "thetaW": theta,
        "gradientEast": grad_e,
        "gradientNorth": grad_n,
        "gradientMagnitude": grad_mag,
        "tfp": tfp,
        "zoneCurvature": zone_curvature,
        # grad|grad theta_w|: the Hewson frontal-speed direction, which the
        # published schema carries per vertex.
        "hewsonEast": gm_e,
        "hewsonNorth": gm_n,
        "frontogenesis": frontogenesis,
        "vorticity1e5": vorticity,
        "convergence1e5": convergence,
        "windShift": wind_shift,
        "witnesses": witnesses,
        "reference": reference,
        "sigmaKm": float(sigma_km),
        "metrics": metrics,
    }


def locating_field(
    evidence: dict, *, cap: float = 2.0, gate_sigma_km: float | None = None
) -> np.ndarray:
    """The field whose ridge is the frontal axis: low order by construction.

    Two things must not be confused.  *Whether* a boundary is a front is the
    fused evidence, which draws on vorticity, convergence, a Laplacian of
    pressure and the curvature of the thermal gradient -- second derivatives,
    some of them.  *Where* the axis lies is a ridge, and finding a ridge means
    taking a Hessian.  Differentiating the fused field directly would make the
    geometry a fourth derivative of the temperature, which is precisely the
    ill-conditioning this engine exists to remove: measured on a straight
    analytic front, the ridge curvature came out twelve times its own analytic
    value, dominated by grid-scale noise, and two thirds of the detected ridge
    points were phantoms lying 90 km off the axis.

    So the locating field carries only ``|grad theta|`` -- one derivative of an
    already synoptically smoothed field, peaking on the axis by definition --
    multiplied by the evidence re-smoothed at the analysis scale, which turns
    it into a slowly varying gate that can switch the ridge off where the
    physics fails but cannot bend it.  The saturation is smooth because a hard
    clip is only C0 and would crease the field the Hessian reads.
    """
    reference = evidence["reference"]["gradientKPerKm"]
    ratio = np.maximum(evidence["gradientMagnitude"], 0.0) / max(reference, 1.0e-12)
    strength = float(cap) * np.tanh(ratio / float(cap))
    gate = fl.smooth_km(
        np.nan_to_num(evidence["probability"], nan=0.0),
        float(gate_sigma_km if gate_sigma_km is not None else evidence["sigmaKm"]),
        evidence["metrics"],
    )
    return strength * gate


def warm_side_normal(evidence: dict) -> tuple[np.ndarray, np.ndarray]:
    """Unit normal pointing into the warm air, as the published schema wants."""
    magnitude = np.maximum(evidence["gradientMagnitude"], 1.0e-9)
    return (evidence["gradientEast"] / magnitude,
            evidence["gradientNorth"] / magnitude)


# --------------------------------------------------------------------------
# Layer 3a -- sub-pixel ridge points (Steger)
# --------------------------------------------------------------------------
def smoothing_support(
    validity: np.ndarray, sigma_km: float, metrics: dict
) -> np.ndarray:
    """Fraction of the smoothing kernel's mass that lies on real data.

    This is exactly the denominator ``front_locator.smooth_km`` divides by,
    recomputed with the same machinery.  Where it is well below one, the
    smoothed field is still a correct weighted mean but its kernel is
    lopsided, so its *derivatives* are biased -- and the geometry layer is all
    derivatives.  Measured on analytic fronts, every phantom ridge point left
    after non-maximum suppression sat in that halo, none outside it.
    """
    values = np.asarray(validity, dtype=float)
    if sigma_km <= 0.0:
        return np.ones_like(values)
    support = fl._conv_axis(values, float(sigma_km) / metrics["dy_km"], axis=0)
    return fl._conv_x_perrow(
        support, float(sigma_km) / metrics["dx_km_col"].ravel()
    )


def admissible_mask(
    evidence: dict,
    *,
    min_strength: float = 0.5,
    min_probability: float = 0.5,
    min_support: float = 0.90,
) -> np.ndarray:
    """Where a ridge is allowed to be a front at all.

    Three independent conditions, each meaning something on its own rather
    than a threshold on some product: at least half the thermal gradient of
    the reference front (below that there is no baroclinic zone to draw);
    evidence at even odds or better (below that the fusion says it is more
    likely not a front); and a smoothing kernel at least ``min_support`` on
    real data, which keeps the geometry out of the lopsided-kernel halo along
    the limited-area boundary where every remaining phantom ridge was
    measured to live.
    """
    reference = evidence["reference"]["gradientKPerKm"]
    strength = evidence["gradientMagnitude"] / max(reference, 1.0e-12)
    # The *raw* probability, deliberately.  Re-smoothing belongs in the
    # locating field, whose Hessian is taken and which therefore must stay low
    # order; here the question is only "what does the evidence say at this
    # point", and smoothing it just spreads a front's evidence over the calm
    # air around it.  Measured on the real field, the smoothed version peaked
    # at 0.033 where the raw one peaked at 0.132: a factor of four thrown away
    # for no reason.
    gate = np.nan_to_num(evidence["probability"], nan=0.0)
    support = smoothing_support(
        np.isfinite(evidence["thetaW"]).astype(float),
        evidence["sigmaKm"], evidence["metrics"],
    )
    return (
        np.isfinite(evidence["gradientMagnitude"])
        & (strength >= float(min_strength))
        & (gate >= float(min_probability))
        & (support >= float(min_support))
    )


def ridge_curvature_scale(sigma_km: float = SYNOPTIC_SIGMA_KM) -> float:
    """Across-ridge curvature of the locating field at the reference front.

    The locating field peaks at 1 on a reference front and falls off with the
    zone width, so its second derivative across the axis is -1/w'^2.  Ridge
    detection needs this scale: without a magnitude threshold the second-order
    test fires on any numerically flat area where round-off happens to make
    the curvature negative, which on an oblique analytic front produced twice
    as many phantom points as real ones.
    """
    width = reference_front_scales(sigma_km)["widthKm"]
    return 1.0 / (width * width)


def ridge_points(
    field: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    *,
    metrics: dict | None = None,
    floor: float = 0.0,
    min_curvature: float = 0.0,
    suppression_km: float = CORRIDOR_KM,
    mask: np.ndarray | None = None,
) -> dict:
    """Sub-pixel ridge points of a 2-D field, with orientation.

    At every cell the Hessian is diagonalised analytically; the eigenvector of
    the most negative eigenvalue is the across-ridge normal, and the field is
    expanded to second order along it.  A cell holds a ridge point when that
    expansion puts the maximum inside the cell, which is what removes the grid
    staircase: the position is continuous even though the search is not.
    """
    metrics = metrics or fl.grid_metrics(longitudes, latitudes)
    values = np.asarray(field, dtype=float)

    f_e, f_n = fl.gradient(values, metrics)
    f_ee = fl.second_derivative(values, metrics["dx_km_col"], axis=1)
    f_nn = fl.second_derivative(values, metrics["dy_km"], axis=0)
    _, f_en = fl.gradient(f_e, metrics)  # mixed derivative, per km^2

    # Analytic eigen-decomposition of [[f_ee, f_en], [f_en, f_nn]].
    half_trace = 0.5 * (f_ee + f_nn)
    half_diff = 0.5 * (f_ee - f_nn)
    root = np.sqrt(np.maximum(half_diff * half_diff + f_en * f_en, 0.0))
    lambda_min = half_trace - root
    angle = 0.5 * np.arctan2(2.0 * f_en, f_ee - f_nn)
    # Eigenvector of the *minor* eigenvalue: the across-ridge normal.
    normal_e = -np.sin(angle)
    normal_n = np.cos(angle)

    curvature = (
        f_ee * normal_e * normal_e
        + 2.0 * f_en * normal_e * normal_n
        + f_nn * normal_n * normal_n
    )
    slope = f_e * normal_e + f_n * normal_n
    with np.errstate(divide="ignore", invalid="ignore"):
        offset_km = -slope / curvature  # distance to the maximum, in km

    dx_km = np.broadcast_to(metrics["dx_km_col"], values.shape)
    dy_km = float(metrics["dy_km"])
    offset_col = offset_km * normal_e / np.maximum(dx_km, 1.0e-9)
    offset_row = offset_km * normal_n / max(dy_km, 1.0e-9)


    # Non-maximum suppression across the ridge.  Without it the second-order
    # test accepts two adjacent cells on the same crest, and the linker, which
    # consumes points by strength, turns the leftovers into a phantom line
    # running parallel to the real one -- measured on an analytic front, a
    # single straight boundary produced a 471 km duplicate.  Steger's detector
    # includes this step; it is not an optional tidy-up.
    lon_grid = np.asarray(longitudes, dtype=float)
    lat_grid = np.asarray(latitudes, dtype=float)
    base = np.column_stack((
        np.broadcast_to(lon_grid, values.shape).ravel(),
        np.repeat(lat_grid, values.shape[1]),
    ))
    cell_km = np.hypot(normal_e * dx_km, normal_n * dy_km)
    half_corridor = 0.5 * max(float(suppression_km), 0.0)
    # One cell, then out to half a corridor.  The corridor is the declared
    # resolution of the analysis: two fronts closer than CORRIDOR_KM are not
    # separable at a 150 km smoothing scale, so within it only the dominant
    # crest may survive.  Comparing against the adjacent cell alone leaves the
    # weak secondary maxima on the flanks of a front, 50-90 km out, and the
    # linker then strings them into excursions off the axis.
    distances = [cell_km]
    if half_corridor > 0.0:
        distances += [np.full_like(cell_km, half_corridor * fraction)
                      for fraction in (0.25, 0.5, 0.75, 1.0)]
    is_local_max = np.ones_like(values, dtype=bool)
    for probe_km in distances:
        step_lon = (probe_km * normal_e / np.maximum(dx_km, 1.0e-9)
                    * float(metrics["dlon"])).ravel()
        step_lat = (probe_km * normal_n / max(dy_km, 1.0e-9)
                    * float(metrics["dlat"])).ravel()
        for sign in (-1.0, 1.0):
            probe = np.column_stack((base[:, 0] + sign * step_lon,
                                     base[:, 1] + sign * step_lat))
            sampled = fl._sample(values, probe, lon_grid, lat_grid,
                                 float(metrics["dlon"]),
                                 float(metrics["dlat"])).reshape(values.shape)
            # A NaN probe lies outside the domain: it cannot veto a crest.
            is_local_max &= ~(np.isfinite(sampled) & (sampled > values))

    is_ridge = (
        np.isfinite(offset_km)
        & (curvature <= -abs(float(min_curvature)))
        & (curvature < 0.0)
        & (lambda_min < 0.0)
        & (np.abs(offset_col) <= 0.5)
        & (np.abs(offset_row) <= 0.5)
        & is_local_max
        & (values >= floor)
        & np.isfinite(values)
    )
    if mask is not None:
        is_ridge &= np.asarray(mask, dtype=bool)
    # The domain rim has one-sided derivatives; a ridge claimed there is an
    # artefact of the stencil, not a structure in the field.
    is_ridge[:2, :] = False
    is_ridge[-2:, :] = False
    is_ridge[:, :2] = False
    is_ridge[:, -2:] = False

    rows, cols = np.nonzero(is_ridge)
    lon = np.asarray(longitudes, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    point_lon = lon[cols] + offset_col[rows, cols] * float(metrics["dlon"])
    point_lat = lat[rows] + offset_row[rows, cols] * float(metrics["dlat"])

    return {
        "rows": rows,
        "cols": cols,
        "lon": point_lon,
        "lat": point_lat,
        # Along-ridge tangent: perpendicular to the across-ridge normal.
        "tangentEast": -normal_n[rows, cols],
        "tangentNorth": normal_e[rows, cols],
        "normalEast": normal_e[rows, cols],
        "normalNorth": normal_n[rows, cols],
        "strength": values[rows, cols],
        "curvature": curvature[rows, cols],
    }


# --------------------------------------------------------------------------
# Layer 3b -- orientation-guided linking
# --------------------------------------------------------------------------
_EARTH_KM_PER_DEG = 111.195


def _km_per_degree(latitude: float) -> tuple[float, float]:
    return (
        _EARTH_KM_PER_DEG * float(np.cos(np.deg2rad(latitude))),
        _EARTH_KM_PER_DEG,
    )


def _separation_km(lon_a, lat_a, lon_b, lat_b):
    kx, ky = _km_per_degree(0.5 * (lat_a + lat_b))
    return np.hypot((lon_b - lon_a) * kx, (lat_b - lat_a) * ky)


def link_ridge_points(
    points: dict,
    *,
    search_cells: int = 2,
    max_curvature_deg_per_km: float = 0.5,
    max_turn_deg: float = 35.0,
    step_jitter_deg: float = 25.0,
    max_orientation_deg: float = 25.0,
    curvature_penalty: float = 0.6,
) -> list[np.ndarray]:
    """Grow polylines along the ridge orientation, capping curvature per km.

    Curvature is part of the choice at every step instead of being repaired
    afterwards: a candidate that would bend the line more than the cap is not
    a candidate at all.  Growth is bidirectional from the strongest unused
    point, as in Steger's linking stage.

    The cap is expressed **per kilometre**, not per step.  A cap per step is
    meaningless because the step is one grid cell: at 9 km a cap of 40 degrees
    per step licenses 89 degrees per 20 km, and measured on an oblique
    analytic front it produced exactly that -- a 1214 km line wandering 118 km
    off a perfectly straight axis.  ``max_turn_deg`` remains only as an
    absolute ceiling for long steps.

    Curvature is enforced on the **ridge tangent**, not on the raw step.  The
    tangent is an eigenvector of the Hessian of a field already smoothed at
    the synoptic scale, so it cannot turn quickly; the step between two
    sub-pixel points one grid cell apart can, because a few hundred metres of
    positional jitter across a 9 km step is several degrees of angle.  Holding
    the step to the same tolerance therefore rejects legitimate continuations
    and shatters oblique ridges -- measured, a straight 1300 km boundary came
    out as a 220 km fragment.  ``step_jitter_deg`` is the angle below which
    the step direction carries no information and is not tested.
    """
    rows = points["rows"]
    cols = points["cols"]
    if rows.size == 0:
        return []

    index_of = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    lon = points["lon"]
    lat = points["lat"]
    tangent_e = points["tangentEast"]
    tangent_n = points["tangentNorth"]
    strength = points["strength"]
    used = np.zeros(rows.size, dtype=bool)

    max_orientation = np.cos(np.deg2rad(max_orientation_deg))

    def grow(seed: int, direction: float) -> list[int]:
        chain: list[int] = []
        current = seed
        # Ridge orientation is undirected; the sign fixes the travel sense.
        heading = np.array([tangent_e[seed], tangent_n[seed]]) * direction
        while True:
            best = None
            best_score = -np.inf
            row0, col0 = int(rows[current]), int(cols[current])
            for drow in range(-search_cells, search_cells + 1):
                for dcol in range(-search_cells, search_cells + 1):
                    if drow == 0 and dcol == 0:
                        continue
                    candidate = index_of.get((row0 + drow, col0 + dcol))
                    if candidate is None or used[candidate]:
                        continue
                    kx, ky = _km_per_degree(lat[current])
                    step = np.array([
                        (lon[candidate] - lon[current]) * kx,
                        (lat[candidate] - lat[current]) * ky,
                    ])
                    distance = float(np.hypot(step[0], step[1]))
                    if distance < 1.0e-6:
                        continue
                    step = step / distance
                    turn = float(np.dot(step, heading))
                    allowed = min(
                        max(max_curvature_deg_per_km * distance,
                            step_jitter_deg),
                        max_turn_deg,
                    )
                    if turn < np.cos(np.deg2rad(allowed)):
                        continue
                    candidate_tangent = np.array(
                        [tangent_e[candidate], tangent_n[candidate]]
                    )
                    # Undirected orientation: |cos| compares axes, not arrows.
                    if abs(float(np.dot(candidate_tangent, heading))) < max_orientation:
                        continue
                    score = float(strength[candidate]) - curvature_penalty * (1.0 - turn)
                    if score > best_score:
                        best_score = score
                        best = (candidate, step, candidate_tangent)
            if best is None:
                return chain
            candidate, step, candidate_tangent = best
            used[candidate] = True
            chain.append(candidate)
            # Keep the tangent pointing the way we travel, then blend with the
            # step so the heading cannot lock onto a stale orientation.
            if float(np.dot(candidate_tangent, heading)) < 0.0:
                candidate_tangent = -candidate_tangent
            heading = candidate_tangent + step
            heading = heading / max(float(np.hypot(heading[0], heading[1])), 1.0e-9)
            current = candidate

    lines: list[np.ndarray] = []
    for seed in np.argsort(-strength):
        if used[seed]:
            continue
        used[seed] = True
        forward = grow(seed, +1.0)
        backward = grow(seed, -1.0)
        order = list(reversed(backward)) + [int(seed)] + forward
        # Growth may step over a cell (``search_cells`` > 1).  A ridge point
        # left unused one cell from an accepted crest is the same crest, and
        # if it survives it seeds a phantom line lying on top of the real one
        # -- measured on an analytic front, a 471 km twin of a 1299 km
        # boundary.  Claim the neighbourhood of the finished chain.
        for member in order:
            row0, col0 = int(rows[member]), int(cols[member])
            for drow in (-1, 0, 1):
                for dcol in (-1, 0, 1):
                    near = index_of.get((row0 + drow, col0 + dcol))
                    if near is not None:
                        used[near] = True
        if len(order) < 3:
            continue
        lines.append(np.column_stack((lon[order], lat[order])))
    return lines


# --------------------------------------------------------------------------
# Layer 3c -- fragment merging
# --------------------------------------------------------------------------
def merge_fragments(
    lines: list[np.ndarray],
    *,
    join_km: float = 180.0,
    max_angle_deg: float = 45.0,
) -> list[np.ndarray]:
    """Rejoin collinear pieces of the same boundary.

    Masking and gaps in the evidence cut one physical boundary into several
    polylines.  Nothing in the previous implementation put them back together,
    so a single front reached the map as fragments.  Two ends merge when they
    are close, their end tangents agree, and the bridge between them runs the
    same way.
    """
    remaining = [np.asarray(line, dtype=float) for line in lines if len(line) >= 2]
    limit = np.cos(np.deg2rad(max_angle_deg))

    def end_tangent(line: np.ndarray, at_end: bool) -> np.ndarray:
        span = min(4, len(line) - 1)
        if at_end:
            a, b = line[-1 - span], line[-1]
        else:
            a, b = line[span], line[0]
        kx, ky = _km_per_degree(0.5 * (a[1] + b[1]))
        vector = np.array([(b[0] - a[0]) * kx, (b[1] - a[1]) * ky])
        return vector / max(float(np.hypot(vector[0], vector[1])), 1.0e-9)

    merged = True
    while merged and len(remaining) > 1:
        merged = False
        for i in range(len(remaining)):
            for j in range(len(remaining)):
                if i == j:
                    continue
                first, second = remaining[i], remaining[j]
                tail = first[-1]
                head = second[0]
                gap = _separation_km(tail[0], tail[1], head[0], head[1])
                if gap > join_km:
                    continue
                out_tangent = end_tangent(first, True)
                in_tangent = end_tangent(second, False)
                kx, ky = _km_per_degree(0.5 * (tail[1] + head[1]))
                bridge = np.array([(head[0] - tail[0]) * kx, (head[1] - tail[1]) * ky])
                bridge_length = float(np.hypot(bridge[0], bridge[1]))
                if bridge_length < 1.0e-6:
                    bridge = out_tangent
                else:
                    bridge = bridge / bridge_length
                if float(np.dot(out_tangent, in_tangent)) < limit:
                    continue
                if float(np.dot(out_tangent, bridge)) < limit:
                    continue
                if float(np.dot(bridge, in_tangent)) < limit:
                    continue
                remaining[i] = np.vstack((first, second))
                remaining.pop(j)
                merged = True
                break
            if merged:
                break
    return remaining


# --------------------------------------------------------------------------
# Layer 3d -- variational finishing
# --------------------------------------------------------------------------
def resample_km(line: np.ndarray, spacing_km: float) -> np.ndarray:
    """Uniform arc-length resampling; the honest basis for every shape metric."""
    points = np.asarray(line, dtype=float)
    if len(points) < 2:
        return points
    kx, ky = _km_per_degree(float(np.mean(points[:, 1])))
    xy = np.column_stack((points[:, 0] * kx, points[:, 1] * ky))
    steps = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    distance = np.concatenate(([0.0], np.cumsum(steps)))
    total = float(distance[-1])
    if total < spacing_km:
        return points
    count = max(int(round(total / spacing_km)) + 1, 2)
    targets = np.linspace(0.0, total, count)
    return np.column_stack((
        np.interp(targets, distance, points[:, 0]),
        np.interp(targets, distance, points[:, 1]),
    ))


def polish_line(
    line: np.ndarray,
    field: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    *,
    metrics: dict | None = None,
    iterations: int = 40,
    elasticity: float = 0.35,
    projection_gain: float = 0.40,
    search_km: float = 45.0,
    max_step_km: float = 12.0,
) -> np.ndarray:
    """Alternate elastic smoothing with re-projection onto the evidence crest.

    This is the active-contour half of the method: the elastic pass removes
    what is left of the linking lattice, and the projection pass puts every
    vertex back on the local maximum of the evidence across the line, so
    smoothing can never quietly slide the front off the baroclinic zone.

    ``projection_gain`` is what makes it a balance rather than a sequence.  At
    gain 1 the projection simply undoes the elastic pass: each vertex snaps
    onto its own local crest sample, jitter included, and the line ends up as
    rough as it started -- measured, 9.4 degrees per 20 km on a perfectly
    straight analytic front.  Below 1 the two forces reach an equilibrium, and
    the curve settles where the pull towards the crest equals the pull towards
    smoothness, which is the stationary point of the energy this is solving.
    """
    metrics = metrics or fl.grid_metrics(longitudes, latitudes)
    points = np.array(line, dtype=float)
    if len(points) < 5:
        return points
    dlon = float(metrics["dlon"])
    dlat = float(metrics["dlat"])
    offsets = np.linspace(-search_km, search_km, 9)

    for _ in range(max(int(iterations), 0)):
        # 1. elastic pass, endpoints fixed
        smoothed = points.copy()
        smoothed[1:-1] += elasticity * (
            points[:-2] + points[2:] - 2.0 * points[1:-1]
        )
        points = smoothed

        # 2. projection pass along the local normal
        tangent = np.gradient(points, axis=0)
        latitude = points[:, 1]
        kx = _EARTH_KM_PER_DEG * np.cos(np.deg2rad(latitude))
        ky = np.full_like(kx, _EARTH_KM_PER_DEG)
        tx = tangent[:, 0] * kx
        ty = tangent[:, 1] * ky
        norm = np.maximum(np.hypot(tx, ty), 1.0e-9)
        normal_x = -ty / norm
        normal_y = tx / norm

        probes = np.empty((len(points), offsets.size))
        for index, distance in enumerate(offsets):
            probe_lon = points[:, 0] + distance * normal_x / np.maximum(kx, 1.0e-9)
            probe_lat = points[:, 1] + distance * normal_y / ky
            probes[:, index] = fl._sample(
                field,
                np.column_stack((probe_lon, probe_lat)),
                longitudes,
                latitudes,
                dlon,
                dlat,
            )

        best = np.nanargmax(np.where(np.isfinite(probes), probes, -np.inf), axis=1)
        shift = offsets[best]
        # Parabolic interpolation around the sampled maximum: the crest is
        # continuous, so the vertex should not snap to a probe position.
        interior = (best > 0) & (best < offsets.size - 1)
        if np.any(interior):
            rows = np.nonzero(interior)[0]
            left = probes[rows, best[rows] - 1]
            middle = probes[rows, best[rows]]
            right = probes[rows, best[rows] + 1]
            denominator = left - 2.0 * middle + right
            valid = np.isfinite(denominator) & (np.abs(denominator) > 1.0e-12)
            step = np.zeros(rows.size)
            step[valid] = 0.5 * (left[valid] - right[valid]) / denominator[valid]
            step = np.clip(step, -1.0, 1.0) * (offsets[1] - offsets[0])
            shift[rows] = shift[rows] + step

        shift = float(projection_gain) * np.clip(
            np.nan_to_num(shift), -max_step_km, max_step_km
        )
        shift[0] = 0.0
        shift[-1] = 0.0
        points[:, 0] += shift * normal_x / np.maximum(kx, 1.0e-9)
        points[:, 1] += shift * normal_y / ky

    return points


def mean_turn_deg_per_km(line: np.ndarray, spacing_km: float = 20.0) -> float:
    """Mean heading change per ``spacing_km``: the tortuosity metric in use."""
    points = resample_km(np.asarray(line, dtype=float), spacing_km)
    if len(points) < 3:
        return 0.0
    kx, ky = _km_per_degree(float(np.mean(points[:, 1])))
    xy = np.column_stack((points[:, 0] * kx, points[:, 1] * ky))
    deltas = np.diff(xy, axis=0)
    headings = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0]))
    turns = np.abs((np.diff(headings) + 180.0) % 360.0 - 180.0)
    return float(np.mean(turns)) if turns.size else 0.0


def line_length_km(line: np.ndarray) -> float:
    points = np.asarray(line, dtype=float)
    if len(points) < 2:
        return 0.0
    kx, ky = _km_per_degree(float(np.mean(points[:, 1])))
    return float(np.sum(np.hypot(
        np.diff(points[:, 0]) * kx, np.diff(points[:, 1]) * ky
    )))


# --------------------------------------------------------------------------
# Entry point: the same contract as front_detection.detect_fronts_two_scale
# --------------------------------------------------------------------------
#
# The keys below are not decoration.  ``coordinates``, ``warmNormal`` and
# ``hewsonDir`` must be three arrays of the same length or the caller drops
# the candidate in silence, and they are recomputed on the final vertices
# rather than carried along from the ridge points, because the variational
# pass moves those vertices.  ``synopticSupport``, ``sinuosity``,
# ``locatorConfidence``, ``medianTfpStrength`` and ``medianAbzGradient``
# default to zero downstream when absent, and zero fails every gate.
MIN_LENGTH_KM = 300.0
VERTEX_SPACING_KM = 8.0
LOCATOR_NAME = "thetaW-evidence-ridge"


def _line_pieces(line: np.ndarray, keep: np.ndarray) -> list[np.ndarray]:
    """Split a polyline where the mask fails, dropping one-point pieces."""
    pieces: list[np.ndarray] = []
    start = None
    for index, good in enumerate(keep):
        if good and start is None:
            start = index
        elif not good and start is not None:
            if index - start >= 2:
                pieces.append(line[start:index])
            start = None
    if start is not None and len(keep) - start >= 2:
        pieces.append(line[start:])
    return pieces


def _build_candidate(
    line: np.ndarray, evidence: dict, abz_gradient: np.ndarray,
    longitudes, latitudes, metrics: dict, *,
    min_probability: float, source: str,
) -> dict:
    """Assemble one candidate in the contract shape, scored on the evidence.

    ``warmNormal`` and ``hewsonDir`` are recomputed on the vertices that are
    actually published -- never carried over from wherever the line came from,
    because the variational pass moves them and the caller silently drops a
    candidate whose three arrays disagree in length.
    """
    dlon, dlat = float(metrics["dlon"]), float(metrics["dlat"])

    def sample(values):
        return fl._sample(values, line, longitudes, latitudes, dlon, dlat)

    reference = evidence["reference"]
    east = sample(evidence["gradientEast"])
    north = sample(evidence["gradientNorth"])
    magnitude = np.maximum(np.hypot(east, north), 1.0e-12)
    hewson_e = sample(evidence["hewsonEast"])
    hewson_n = sample(evidence["hewsonNorth"])
    hewson_mag = np.maximum(np.hypot(hewson_e, hewson_n), 1.0e-12)
    line_tfp = sample(evidence["tfp"])
    line_abz = sample(abz_gradient)
    line_probability = sample(evidence["probability"])
    sinuosity, net_turn, closure, total_turn = _shape_metrics(line)

    return {
        "coordinates": line,
        "warmNormal": np.column_stack((east / magnitude, north / magnitude)),
        "hewsonDir": np.column_stack((hewson_e / hewson_mag,
                                      hewson_n / hewson_mag)),
        "medianTfp": float(np.nanmedian(line_tfp)),
        "medianTfpStrength": float(np.nanmedian(-line_tfp * 10_000.0)),
        "medianThetaWGradient": float(
            np.nanmedian(sample(evidence["gradientMagnitude"])) * 100.0
        ),
        "medianAbzGradient": float(np.nanmedian(line_abz)),
        "peakAbzGradient": float(np.nanmax(line_abz)),
        "lengthKm": line_length_km(line),
        # Evidence in probabilistic form along the line, not a calibrated
        # probability: no labelled archive exists here.
        "locatorConfidence": float(np.nanmedian(line_probability)),
        # The share of the line the physics actually supports.  In the old
        # two-scale detector this was the fraction lying inside a separate
        # synoptic corridor; here the line *is* the synoptic ridge, so the
        # honest analogue is the share of it where the fused evidence stands
        # at even odds or better.
        "synopticSupport": round(float(np.nanmean(
            np.where(np.isfinite(line_probability), line_probability, 0.0)
            >= min_probability
        )), 2),
        "corroborated": True,
        "sinuosity": round(sinuosity, 2),
        "netTurnDeg": round(net_turn, 1),
        "closureRatio": round(closure, 2),
        "totalTurnDeg": round(total_turn, 1),
        "meanTurnDegPer20Km": round(mean_turn_deg_per_km(line), 2),
        "effectiveTfpThreshold": -reference["tfpKPerKm2"],
        "effectiveGradientThreshold": reference["gradientK100Km"],
        "locatorMethod": source,
        "analysisSigmaKm": float(evidence["sigmaKm"]),
    }


def _shape_metrics(line: np.ndarray):
    import front_detection as fd
    return fd._shape_metrics(line)


def detect_fronts(
    theta_w: np.ndarray,
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    *,
    metrics: dict | None = None,
    sigma_km: float = SYNOPTIC_SIGMA_KM,
    theta_w_lower: np.ndarray | None = None,
    theta_w_upper: np.ndarray | None = None,
    pressure: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    terrain: np.ndarray | None = None,
    min_length_km: float = MIN_LENGTH_KM,
    vertex_spacing_km: float = VERTEX_SPACING_KM,
    min_probability: float = 0.5,
    max_sinuosity: float = 2.35,
    min_closure_ratio: float = 0.42,
    max_net_turn_deg: float = 165.0,
    return_fields: bool = False,
):
    """Locate fronts and return candidates in the established contract shape.

    The geometry is the ridge of the locating field; the fused evidence
    decides which parts of that ridge may be published and supplies the
    per-candidate scores.  Length and persistence play no part in either.
    """
    import front_detection as fd

    metrics = metrics or fl.grid_metrics(longitudes, latitudes)
    evidence = frontal_evidence(
        theta_w, u_wind, v_wind, longitudes, latitudes,
        metrics=metrics, sigma_km=sigma_km,
        theta_w_lower=theta_w_lower, theta_w_upper=theta_w_upper,
        pressure=pressure, omega=omega, terrain=terrain,
    )
    field = locating_field(evidence)
    mask = admissible_mask(evidence, min_probability=min_probability)
    points = ridge_points(
        field, longitudes, latitudes, metrics=metrics, mask=mask,
        min_curvature=0.25 * ridge_curvature_scale(sigma_km),
    )
    raw_lines = merge_fragments(link_ridge_points(points))

    reference = evidence["reference"]
    # Hewson tests the baroclinic zone *behind* the front, not the value on
    # the line.  The walk takes K/km in and returns K/100 km; the first-order
    # extrapolation stays as a floor, exactly as front_locator combines them.
    grad_magnitude = evidence["gradientMagnitude"]
    abz_gradient = fl.adjacent_baroclinic_zone(
        grad_magnitude, evidence["gradientEast"], evidence["gradientNorth"],
        longitudes, latitudes, search_km=float(sigma_km),
    )
    local_grid_km = np.sqrt(metrics["dx_km_col"] * metrics["dy_km"])
    abz_gradient = np.fmax(abz_gradient, (
        grad_magnitude + (local_grid_km / np.sqrt(2.0))
        * np.hypot(evidence["hewsonEast"], evidence["hewsonNorth"])
    ) * 100.0)
    dlon, dlat = float(metrics["dlon"]), float(metrics["dlat"])

    def sample(values, line):
        return fl._sample(values, line, longitudes, latitudes, dlon, dlat)

    # Un imbuto che si chiude in silenzio non e' diagnosticabile: il run del 12
    # settembre ha pubblicato "0 candidati" per 73 ore di fila senza lasciare
    # un solo numero che dicesse quale passaggio li avesse fermati.  Ogni
    # stadio conta cio' che gli entra e cio' che ne esce.
    funnel = {
        "admissibleCells": int(np.count_nonzero(mask)),
        "gridCells": int(mask.size),
        "peakProbability": round(float(np.nanmax(evidence["probability"]))
                                 if np.any(np.isfinite(evidence["probability"]))
                                 else 0.0, 3),
        "peakGradientK100Km": round(float(
            np.nanmax(evidence["gradientMagnitude"]) * 100.0
        ), 2), 
        "referenceGradientK100Km": round(
            evidence["reference"]["gradientK100Km"], 2
        ),
        "ridgePoints": int(points["rows"].size),
        "rawLines": len(raw_lines),
        "longestRawKm": round(max((line_length_km(line) for line in raw_lines),
                                  default=0.0), 1),
        "droppedTooShort": 0,
        "droppedShape": 0,
    }

    candidates: list[dict] = []
    for raw in raw_lines:
        polished = polish_line(
            resample_km(raw, vertex_spacing_km), field,
            longitudes, latitudes, metrics=metrics,
        )
        # The variational pass can walk a vertex out of the admissible area;
        # split there rather than publishing a tail with no evidence behind
        # it, and re-check the length afterwards -- checking it before the
        # cut is how a 300 km rule ends up publishing a 120 km stub.
        probability = sample(evidence["probability"], polished)
        for piece in _line_pieces(polished, np.isfinite(probability)
                                  & (probability >= min_probability)):
            line = resample_km(piece, vertex_spacing_km)
            length = line_length_km(line)
            if length < min_length_km or len(line) < 4:
                funnel["droppedTooShort"] += 1
                continue
            sinuosity, net_turn, closure, total_turn = fd._shape_metrics(line)
            # A closed or hairpin thermal anomaly is a pool of air, not an
            # interface between two extended air masses.
            if (sinuosity > max_sinuosity or closure < min_closure_ratio
                    or net_turn > max_net_turn_deg):
                funnel["droppedShape"] += 1
                continue

            candidates.append(_build_candidate(
                line, evidence, abz_gradient, longitudes, latitudes, metrics,
                min_probability=min_probability, source=LOCATOR_NAME,
            ))

    candidates.sort(
        key=lambda item: (item["locatorConfidence"], item["lengthKm"]),
        reverse=True,
    )
    unique: list[dict] = []
    for candidate in candidates:
        if any(
            fd._overlap_fraction(candidate["coordinates"],
                                 kept["coordinates"], 55.0) >= 0.72
            for kept in unique
        ):
            continue
        unique.append(candidate)
    unique.sort(key=lambda item: item["lengthKm"], reverse=True)
    funnel["published"] = len(unique)
    for candidate in unique:
        candidate["engineFunnel"] = funnel

    if return_fields:
        return unique, {"evidence": evidence, "locating": field, "mask": mask,
                        "ridgePoints": points, "abzGradient": abz_gradient,
                        "funnel": funnel, "longitudes": longitudes,
                        "latitudes": latitudes}
    if not unique:
        return _EmptyWithFunnel(funnel)
    return unique


class _EmptyWithFunnel(list):
    """An empty candidate list that still carries why it is empty."""

    def __init__(self, funnel: dict):
        super().__init__()
        self.funnel = funnel


def score_lines(
    lines,
    fields: dict,
    *,
    min_length_km: float = MIN_LENGTH_KM,
    vertex_spacing_km: float = VERTEX_SPACING_KM,
    min_probability: float = 0.5,
    min_supported_fraction: float = 0.6,
    source: str = "external-line",
) -> list[dict]:
    """Judge lines produced elsewhere with this engine's own evidence.

    The point of separating *where* from *whether* is that the two can come
    from different places.  A fallback detector may supply the geometry when
    the ridge search finds nothing, but the verdict must stay here, or the
    fallback quietly reinstates what the engine exists to remove: measured,
    the two-scale detector handed back the 1730 km Alpine thermal boundary the
    moment it was allowed to publish unchecked.

    A supplied line is kept only where the fused evidence stands at even odds
    or better, is cut where it does not, and must still clear the length it
    was asked for *after* the cut.
    """
    evidence = fields["evidence"]
    abz_gradient = fields["abzGradient"]
    metrics = evidence["metrics"]
    longitudes = fields["longitudes"]
    latitudes = fields["latitudes"]
    dlon, dlat = float(metrics["dlon"]), float(metrics["dlat"])

    kept: list[dict] = []
    for raw in lines:
        line = resample_km(np.asarray(raw, dtype=float), vertex_spacing_km)
        if len(line) < 4:
            continue
        probability = fl._sample(evidence["probability"], line,
                                 longitudes, latitudes, dlon, dlat)
        supported = np.isfinite(probability) & (probability >= min_probability)
        if float(np.mean(supported)) < min_supported_fraction:
            continue
        for piece in _line_pieces(line, supported):
            piece = resample_km(piece, vertex_spacing_km)
            if line_length_km(piece) < min_length_km or len(piece) < 4:
                continue
            kept.append(_build_candidate(
                piece, evidence, abz_gradient, longitudes, latitudes, metrics,
                min_probability=min_probability, source=source,
            ))
    kept.sort(key=lambda item: item["lengthKm"], reverse=True)
    return kept


def last_funnel(candidates: list[dict], fallback: dict | None = None) -> dict:
    """The funnel counters, readable even when nothing was published."""
    if isinstance(candidates, _EmptyWithFunnel):
        return candidates.funnel
    for candidate in candidates:
        if "engineFunnel" in candidate:
            return candidate["engineFunnel"]
    return fallback or {}
