"""Physical diagnostics and transparent evidence scores for front analysis.

This module contains no GRIB or tracking code.  It turns gridded ICON fields
into kinematic diagnostics, then maps independent pieces of evidence to
bounded scores.  The resulting ``evidenceScore`` is deliberately *not* a
probability: it measures internal physical consistency of one deterministic
forecast.  Keeping this distinction explicit avoids pretending that a
single ICON run can quantify forecast probability.

The frontogenesis equation is the horizontal, adiabatic Petterssen/Keyser
form for the material change in |grad(theta_w)|.  Positive values mean
frontogenesis and negative values frontolysis.
"""

from __future__ import annotations

import numpy as np

import front_locator as fl


def _finite(value, default=np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def smoothstep(value, weak: float, strong: float) -> float:
    """Linear fuzzy membership in [0, 1] with a safe finite fallback."""
    value = _finite(value)
    if not np.isfinite(value):
        return 0.0
    if strong == weak:
        return float(value >= strong)
    fraction = (value - weak) / (strong - weak)
    return float(np.clip(fraction, 0.0, 1.0))


def kinematic_fields(
    theta_w: np.ndarray,
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    metrics: dict,
    *,
    smoothing_km: float = 45.0,
) -> dict[str, np.ndarray]:
    """Return metric-aware wind and frontogenesis diagnostics.

    Output units:
      * vorticity / convergence: 1e-5 s-1
      * frontogenesis: K (100 km)-1 (3 h)-1
      * thermal advection: K (3 h)-1
    """
    theta = fl.smooth_km(theta_w, smoothing_km, metrics)
    u = fl.smooth_km(u_wind, smoothing_km, metrics)
    v = fl.smooth_km(v_wind, smoothing_km, metrics)

    theta_x, theta_y = fl.gradient(theta, metrics)  # K/km
    u_x_km, u_y_km = fl.gradient(u, metrics)        # (m/s)/km
    v_x_km, v_y_km = fl.gradient(v, metrics)
    # Velocity derivatives must be per metre to obtain s-1.
    u_x, u_y = u_x_km / 1000.0, u_y_km / 1000.0
    v_x, v_y = v_x_km / 1000.0, v_y_km / 1000.0

    magnitude = np.hypot(theta_x, theta_y)
    safe = np.maximum(magnitude * magnitude, 1.0e-12)
    divergence = u_x + v_y
    stretching = u_x - v_y
    shearing = v_x + u_y
    cos_two_beta = (theta_x * theta_x - theta_y * theta_y) / safe
    sin_two_beta = 2.0 * theta_x * theta_y / safe

    # D|grad(theta)|/Dt from horizontal deformation and divergence.
    frontogenesis_si = -0.5 * magnitude * (
        divergence + stretching * cos_two_beta + shearing * sin_two_beta
    )
    frontogenesis = frontogenesis_si * 100.0 * 10_800.0
    vorticity = (v_x - u_y) * 1.0e5
    convergence = -divergence * 1.0e5
    thermal_advection = -(
        u * theta_x + v * theta_y
    ) / 1000.0 * 10_800.0

    invalid = ~(
        np.isfinite(theta) & np.isfinite(u) & np.isfinite(v)
    )
    return {
        "thetaWSmooth": theta,
        "thetaWGradientEast": theta_x,
        "thetaWGradientNorth": theta_y,
        "vorticity1e5": np.where(invalid, np.nan, vorticity),
        "convergence1e5": np.where(invalid, np.nan, convergence),
        "frontogenesis": np.where(invalid, np.nan, frontogenesis),
        "thermalAdvection3h": np.where(invalid, np.nan, thermal_advection),
    }


def candidate_evidence(metrics: dict) -> dict:
    """Score independent thermodynamic, dynamic and structural evidence."""
    locator = smoothstep(metrics.get("locatorConfidence"), 0.15, 0.85)
    theta_w_contrast = smoothstep(metrics.get("deltaThetaW"), 1.2, 4.5)
    dry_contrast = smoothstep(metrics.get("deltaTemperature"), 0.45, 3.0)
    virtual_contrast = smoothstep(metrics.get("deltaThetaV"), 0.20, 2.2)
    dry_gradient = smoothstep(metrics.get("dryThermalGradient"), 0.75, 3.0)
    alignment = smoothstep(metrics.get("thermalAlignment"), 0.05, 0.75)
    # A genuine air-mass boundary should survive sampling on both sides at
    # several physical distances.  Missing legacy data is deliberately
    # neutral rather than negative: this module is also used by unit tests
    # and by older archived analysis files.
    cross_distance_value = _finite(metrics.get("crossDistanceThermalSupport"))
    cross_distance = (
        smoothstep(cross_distance_value, 0.36, 0.84)
        if np.isfinite(cross_distance_value) else 0.50
    )
    thermal = float(np.average(
        [locator, theta_w_contrast, dry_contrast, virtual_contrast,
         dry_gradient, alignment, cross_distance],
        weights=[1.1, 1.2, 1.3, 1.2, 1.0, 0.8, 1.0],
    ))

    wind_shift = max(
        smoothstep(metrics.get("windShiftMs"), 1.5, 7.0),
        smoothstep(metrics.get("windShiftAngleDeg"), 8.0, 42.0),
    )
    convergence = max(
        smoothstep(metrics.get("convergenceMs"), 0.05, 2.0),
        smoothstep(metrics.get("convergenceFraction"), 0.42, 0.78),
    )
    vorticity = smoothstep(metrics.get("vorticity1e5"), -0.5, 4.0)
    frontogenesis = smoothstep(metrics.get("frontogenesis"), -0.6, 2.8)
    # Do not let vorticity/frontogenesis hide a wind field that does not
    # contain a boundary.  Wind rotation/shear and mass convergence form one
    # pillar; deformation/vorticity form the second.
    wind_pillar = 0.58 * wind_shift + 0.42 * convergence
    deformation_pillar = 0.55 * frontogenesis + 0.45 * vorticity
    dynamic = float(0.62 * wind_pillar + 0.38 * deformation_pillar)

    pressure_value = _finite(metrics.get("pressureTroughHpa"))
    pressure = (
        smoothstep(pressure_value, -0.15, 1.8)
        if np.isfinite(pressure_value) else 0.45
    )
    isallobaric = _finite(metrics.get("isallobaricSupportHpa3h"))
    if np.isfinite(isallobaric):
        pressure = 0.72 * pressure + 0.28 * smoothstep(isallobaric, -0.3, 1.2)

    lower_support = _finite(metrics.get("lowerLevelSupport"))
    lower_contrast = _finite(metrics.get("deltaThetaW925"))
    if np.isfinite(lower_support) and np.isfinite(lower_contrast):
        vertical = 0.58 * smoothstep(lower_support, 0.35, 0.80) + 0.42 * smoothstep(
            lower_contrast, 0.7, 3.2
        )
    else:
        vertical = 0.45
    upper_contrast = _finite(metrics.get("deltaThetaW700"))
    if np.isfinite(upper_contrast):
        vertical = 0.85 * vertical + 0.15 * smoothstep(
            upper_contrast, 0.4, 2.8
        )
    # Cross-section 925/850 coherence (front_sections): prudent weight until
    # calibrated on an independent archive; missing stays strictly neutral.
    coherence = _finite(metrics.get("verticalCoherence3Level"))
    if not np.isfinite(coherence):
        coherence = _finite(metrics.get("verticalCoherence"))
    if np.isfinite(coherence):
        vertical = 0.78 * vertical + 0.22 * float(np.clip(coherence, 0.0, 1.0))

    omega = _finite(metrics.get("omega700PaS"))
    activity = smoothstep(-omega, -0.03, 0.20) if np.isfinite(omega) else 0.50

    synoptic = smoothstep(metrics.get("synopticSupport"), 0.55, 0.95)
    length = smoothstep(metrics.get("lengthKm"), 220.0, 750.0)
    sinuosity = _finite(metrics.get("sinuosity"), 1.0)
    shape = smoothstep(2.30 - sinuosity, 0.0, 1.15)
    terrain = 1.0 - smoothstep(metrics.get("terrainFraction"), 0.20, 0.70)
    structural = float(np.average(
        [synoptic, length, shape, terrain], weights=[1.4, 0.8, 0.8, 0.6]
    ))

    base_evidence = float(np.clip(
        0.38 * thermal
        + 0.24 * dynamic
        + 0.10 * pressure
        + 0.10 * vertical
        + 0.04 * activity
        + 0.14 * structural,
        0.0,
        1.0,
    ))
    # Multi-method agreement is confirmation, not a substitute for any
    # non-compensating gate. Missing evidence is neutral and disagreement
    # never erases a physically sound front. The maximum bonus is 0.04.
    consensus = _finite(metrics.get("consensusSupport"))
    consensus_bonus = (
        0.08 * max(0.0, min(consensus, 1.0) - 0.50)
        if np.isfinite(consensus) else 0.0
    )
    evidence = float(np.clip(base_evidence + consensus_bonus, 0.0, 1.0))
    components = {
        "thermal": round(thermal, 3),
        "dynamic": round(dynamic, 3),
        "pressure": round(float(pressure), 3),
        "vertical": round(float(vertical), 3),
        "activity": round(float(activity), 3),
        "structural": round(structural, 3),
        "consensus": round(float(consensus), 3) if np.isfinite(consensus) else 0.5,
    }
    return {
        "candidateEvidence": round(evidence, 3),
        "physicalCandidateEvidence": round(evidence, 3),
        "preConsensusEvidence": round(base_evidence, 3),
        "consensusEvidenceBonus": round(consensus_bonus, 3),
        "evidenceComponents": components,
    }


_HYPOTHESIS_LABEL = {
    "synoptic-front": "fronte sinottico",
    "moisture-boundary": "confine di umidità / dryline",
    "mesoscale-boundary": "confine mesoscalare (brezza)",
    "outflow-boundary": "outflow convettivo / cold pool",
    "orographic-boundary": "segnale orografico",
    "noise": "gradiente debole / rumore",
}


def _neutral(value, weak, strong, fallback=0.5):
    """Fuzzy membership, but a genuinely missing field stays neutral."""
    v = _finite(value, np.nan)
    return smoothstep(v, weak, strong) if np.isfinite(v) else float(fallback)


def differential_diagnosis(metrics: dict) -> dict:
    """Weigh competing physical hypotheses like a human forecaster.

    Instead of a chain of thresholds, this observes *all* the characteristics
    of the case and scores each archetype the way a meteorologist compares
    what he sees against what each phenomenon should look like: a synoptic
    front, a moisture boundary/dryline, a sea-breeze/mesoscale line, a
    convective outflow, an orographic signal, or plain weak noise. The best
    supported hypothesis is the verdict; the full ranking and the reasons the
    winner beat the runner-up are returned so the decision is transparent.

    Supports are internal physical-consistency scores in [0, 1], NOT
    probabilities: they say how well the case matches each archetype.
    """
    # --- observed characteristics (soft memberships) --------------------
    theta_w = smoothstep(metrics.get("deltaThetaW"), 1.0, 4.5)
    dry = smoothstep(metrics.get("deltaTemperature"), 0.5, 3.0)
    density = smoothstep(metrics.get("deltaThetaV"), 0.2, 2.2)
    dry_gradient = smoothstep(metrics.get("dryThermalGradient"), 0.7, 3.0)
    theta_e = _neutral(metrics.get("deltaThetaE"), 1.5, 6.0, 0.0)
    synoptic = smoothstep(metrics.get("synopticSupport"), 0.55, 0.95)
    length = smoothstep(metrics.get("lengthKm"), 250.0, 800.0)
    short = 1.0 - length
    terrain = smoothstep(metrics.get("terrainFraction"), 0.25, 0.70)
    sinuosity = smoothstep(metrics.get("sinuosity"), 1.4, 2.6)
    wind = max(smoothstep(metrics.get("windShiftMs"), 1.5, 7.0),
               smoothstep(metrics.get("windShiftAngleDeg"), 8.0, 42.0))
    convergence = max(smoothstep(metrics.get("convergenceMs"), 0.05, 2.0),
                      smoothstep(metrics.get("convergenceFraction"), 0.42, 0.78))
    vorticity = smoothstep(metrics.get("vorticity1e5"), -0.5, 4.0)
    frontogenesis = smoothstep(metrics.get("frontogenesis"), -0.6, 2.8)
    ascent = _neutral(metrics.get("omega700PaS"), -0.03, -0.20, 0.5)
    if np.isfinite(_finite(metrics.get("omega700PaS"), np.nan)):
        ascent = smoothstep(-_finite(metrics.get("omega700PaS")), -0.03, 0.20)
    lower = _neutral(metrics.get("lowerLevelSupport"), 0.35, 0.80, 0.5)
    lower_contrast = _neutral(metrics.get("deltaThetaW925"), 0.7, 3.2, 0.5)
    vertical = 0.58 * lower + 0.42 * lower_contrast
    trough = _neutral(metrics.get("pressureTroughHpa"), -0.15, 1.8, 0.45)

    dynamic = 0.48 * wind + 0.30 * convergence + 0.12 * vorticity + 0.10 * frontogenesis
    # All three contrasts must be present -- theta_w, dry temperature and
    # density -- but the AND is soft.  A hard ``min`` lets the single weakest
    # ingredient annihilate the score, and the weakest is almost always the
    # density contrast: over one ICON-2I run a 900-km boundary with a clean
    # 2 K theta_w contrast and a clean dry contrast scored 0.14 because
    # delta theta_v was 0.5 K, and the same boundary flipped between
    # "synoptic front" and "mesoscale boundary" from one hour to the next.
    # The geometric mean still collapses to zero when an ingredient is truly
    # absent -- which is what makes a front a front -- but it degrades
    # smoothly when one is merely modest, which is the common case in a
    # summer air mass.
    thermal_core = float(np.cbrt(max(theta_w, 0.0) * max(dry, 0.0)
                                 * max(density, 0.0)))

    # --- temporal evolution: a front persists and moves coherently, a
    # mesoscale/outflow boundary is transient and erratic (documento sez. 12).
    persistence = _neutral(metrics.get("lifetimeH"), 3.0, 15.0, 0.5)
    coverage_t = _neutral(metrics.get("temporalCoverage"), 0.5, 0.95, 0.5)
    mad = _finite(metrics.get("motionMadKmh"), np.nan)
    motion_coherence = (
        1.0 - smoothstep(mad, 6.0, 26.0) if np.isfinite(mad) else 0.5
    )
    temporal = 0.45 * persistence + 0.30 * coverage_t + 0.25 * motion_coherence
    transient = 1.0 - persistence

    # --- hypothesis supports --------------------------------------------
    synoptic_front = (
        0.28 * thermal_core + 0.12 * dry_gradient + 0.18 * synoptic
        + 0.09 * length + 0.11 * vertical + 0.08 * dynamic + 0.04 * trough
        + 0.10 * temporal
    ) * (1.0 - 0.5 * terrain)

    # Sharp theta-w/theta-e but no dry or density separation and shallow.
    moisture = (
        max(theta_e, theta_w)
        * (1.0 - dry) * (1.0 - density) * (1.0 - dry_gradient)
    )
    moisture = 0.65 * moisture + 0.20 * (1.0 - vertical) + 0.15 * (1.0 - synoptic)

    # Short, locally convergent, shallow, transient, no synoptic corridor.
    mesoscale = (0.30 * short + 0.22 * convergence + 0.20 * (1.0 - synoptic)
                 + 0.16 * (1.0 - vertical) + 0.12 * transient) * max(convergence, wind)

    # Short, curved/closed, convective ascent, transient, no corridor.
    outflow = (0.24 * short + 0.22 * sinuosity + 0.18 * convergence
               + 0.16 * ascent + 0.10 * (1.0 - synoptic)
               + 0.10 * transient) * max(convergence, ascent)

    orographic = (0.6 * terrain + 0.4 * (1.0 - synoptic)) * terrain

    noise = (1.0 - max(theta_w, dry, synoptic, dynamic)) * (0.6 + 0.4 * transient)

    supports = {
        "synoptic-front": synoptic_front,
        "moisture-boundary": moisture,
        "mesoscale-boundary": mesoscale,
        "outflow-boundary": outflow,
        "orographic-boundary": orographic,
        "noise": noise,
    }
    ranked = sorted(supports.items(), key=lambda kv: kv[1], reverse=True)
    best, top = ranked[0]
    # A located candidate that already passed TFL/TFP/ABZ is a *plausible*
    # front. Like a forecaster, we keep the synoptic-front reading unless a
    # competing explanation CLEARLY wins - a narrowly higher "noise" score
    # must not veto a genuine but modest boundary. This margin is what keeps
    # real fronts from being dropped while still catching clear drylines,
    # orographic and mesoscale look-alikes.
    sf = supports["synoptic-front"]
    # The air-mass thermal contrast (theta_w AND dry T AND density together)
    # is what makes a front a front. When it is essentially absent the
    # candidate is not a synoptic front even if structure/vertical scores are
    # high; otherwise a located, plausible candidate keeps the front reading
    # unless a rival CLEARLY dominates.
    # The margin is graded, not switched.  It used to vanish the instant
    # ``thermal_core`` fell below a hard 0.20, so at that value a difference
    # of a thousandth between two hypotheses decided the verdict -- and a
    # boundary sitting on the step was read as a front at 03 UTC, as a
    # mesoscale line at 04 UTC and as a front again at 05 UTC.  Nothing in
    # the atmosphere had changed.  Now the protection fades as the air-mass
    # contrast fades: no contrast, no protection (a dryline still loses);
    # solid contrast, full protection.
    thermal_weight = smoothstep(thermal_core, 0.05, 0.35)
    required_margin = 0.12 * thermal_weight
    # A front reading is shielded from being overturned only when it is both
    # well supported AND backed by a real air-mass contrast.  Structure and
    # persistence alone must never keep the label: that is how a dryline or
    # a lee convergence line gets promoted to a front.
    protected = sf >= 0.55 and thermal_weight >= 0.35
    if best != "synoptic-front" and top >= sf + required_margin and not protected:
        verdict = best
    else:
        verdict = "synoptic-front"
    alt_name = next(k for k, _ in ranked if k != verdict)
    runner = alt_name
    margin = float(supports[verdict] - supports[alt_name])

    # Why the winner beat the field (human-readable discriminators).
    reasons: list[str] = []
    if verdict == "synoptic-front":
        if thermal_core >= 0.5:
            reasons.append("contrasto simultaneo di θw, temperatura secca e densità")
        if synoptic >= 0.5:
            reasons.append("struttura coerente alla scala sinottica")
        if vertical >= 0.5:
            reasons.append("contrasto che si estende in verticale (925 hPa)")
        if dynamic >= 0.5:
            reasons.append("risposta dinamica del vento coerente")
        if persistence >= 0.55 and motion_coherence >= 0.5:
            reasons.append("persistente e in moto coerente nel tempo")
        elif persistence >= 0.55:
            reasons.append("persistente nel tempo")
        reasons.append(
            f"spiega il quadro meglio del caso «{_HYPOTHESIS_LABEL[runner]}»"
        )
    elif verdict == "moisture-boundary":
        reasons.append("forte gradiente di umidità ma contrasto termico/densità debole")
    elif verdict == "mesoscale-boundary":
        reasons.append("corta, convergente e senza corridoio sinottico")
    elif verdict == "outflow-boundary":
        reasons.append("corta, curva e con ascesa convettiva locale")
    elif verdict == "orographic-boundary":
        reasons.append("segnale confinato sui rilievi, senza struttura sinottica")
    else:
        reasons.append("prove fisiche complessivamente deboli")

    return {
        "verdict": verdict,
        "margin": round(margin, 3),
        "supports": {k: round(float(v), 3) for k, v in supports.items()},
        "ranking": [k for k, _ in ranked],
        "reasons": reasons,
    }


def candidate_hypothesis(metrics: dict) -> tuple[str, list[str]]:
    """Backward-compatible view of the reasoning engine: (verdict, reasons)."""
    report = differential_diagnosis(metrics)
    return report["verdict"], report["reasons"]


_DIAGNOSIS_LABEL = {
    "synoptic-front": "fronte sinottico",
    "moisture-boundary": "confine di umidità / dryline",
    "orographic-boundary": "segnale orografico",
    "mesoscale-boundary": "confine mesoscalare (brezza/outflow)",
    "ridge-boundary": "gradiente debole sotto promontorio barico",
}


def frontal_explanation(
    diagnostics: dict,
    classification: dict | None = None,
    diagnosis: str = "synoptic-front",
    lifetime_h: float | None = None,
) -> list[str]:
    """Plain-language reasons a front was accepted (the science, exposed).

    Reads only diagnostics already computed for the track, so it invents no
    new physics: it verbalises the evidence that the numeric scores encode.
    Ordered strongest-first, thermodynamics before dynamics before context.
    """
    reasons: list[str] = []
    classification = classification or {}

    delta_tw = _finite(diagnostics.get("deltaThetaW"))
    abz = _finite(diagnostics.get("medianAbzGradient"))
    delta_t = _finite(diagnostics.get("deltaTemperature"))
    delta_tv = _finite(diagnostics.get("deltaThetaV"))
    if np.isfinite(delta_tw) and delta_tw >= 1.2:
        reasons.append(
            f"masse d'aria termicamente distinte (Δθw {delta_tw:.1f} K)"
        )
    if np.isfinite(abz) and abz >= 1.0:
        reasons.append(f"zona baroclina adiacente marcata ({abz:.1f} K/100 km)")
    if np.isfinite(delta_t) and np.isfinite(delta_tv) and delta_t >= 0.6 and delta_tv >= 0.25:
        reasons.append("contrasto reale di temperatura secca e densità")

    wind_angle = _finite(diagnostics.get("windShiftAngleDeg"))
    convergence = _finite(diagnostics.get("convergenceMs"))
    vorticity = _finite(diagnostics.get("vorticity1e5"))
    frontogenesis = _finite(diagnostics.get("frontogenesis"))
    if np.isfinite(wind_angle) and wind_angle >= 10.0:
        reasons.append(f"rotazione del vento attraverso la linea ({wind_angle:.0f}°)")
    if np.isfinite(convergence) and convergence >= 0.12:
        reasons.append("convergenza del vento verso il fronte")
    if np.isfinite(vorticity) and vorticity >= 0.8:
        reasons.append("striscia di vorticità ciclonica")
    if np.isfinite(frontogenesis) and frontogenesis >= 0.25:
        reasons.append("frontogenesi attiva (gradiente in intensificazione)")

    trough = _finite(diagnostics.get("pressureTroughHpa"))
    if np.isfinite(trough) and trough >= 0.2:
        reasons.append(f"linea in saccatura barica ({trough:.1f} hPa)")

    lower = _finite(diagnostics.get("lowerLevelSupport"))
    lower_contrast = _finite(diagnostics.get("deltaThetaW925"))
    if np.isfinite(lower) and lower >= 0.35 and np.isfinite(lower_contrast) and lower_contrast >= 0.7:
        reasons.append("contrasto coerente anche a 925 hPa (struttura profonda)")

    omega = _finite(diagnostics.get("omega700PaS"))
    if np.isfinite(omega) and omega <= -0.05:
        reasons.append("moto verticale ascendente a 700 hPa")

    if lifetime_h is not None and np.isfinite(_finite(lifetime_h)) and lifetime_h >= 6:
        reasons.append(f"persistente e coerente per {int(lifetime_h)} ore")

    votes = classification.get("motionVotes", {})
    agree = [k for k, v in votes.items() if v == classification.get("frontType")]
    if classification.get("frontType") in ("cold", "warm") and len(agree) >= 2:
        reasons.append("moto geometrico e vento concordi sul tipo")

    if diagnosis != "synoptic-front":
        reasons.append(
            f"attenzione: interpretazione più probabile = "
            f"{_DIAGNOSIS_LABEL.get(diagnosis, diagnosis)}"
        )
    if not reasons:
        reasons.append("supporto fisico minimo ma coerente")
    return reasons


def candidate_gate_report(metrics: dict, evidence: dict | None = None) -> dict:
    """Apply a small set of non-compensating physical gates.

    Thermodynamic air-mass separation and a synoptic-scale corridor are the
    hard requirements.  Wind, pressure and 925-hPa coherence remain explicit
    diagnostics: they strengthen confidence and the frontal type, but do not
    make a physically coherent boundary vanish for one forecast hour.
    """
    evidence = evidence or candidate_evidence(metrics)
    reasons: list[str] = []

    delta_tw = _finite(metrics.get("deltaThetaW"), -999.0)
    delta_t = _finite(metrics.get("deltaTemperature"), -999.0)
    delta_tv = _finite(metrics.get("deltaThetaV"), -999.0)
    dry_gradient = _finite(metrics.get("dryThermalGradient"), -999.0)
    alignment = _finite(metrics.get("thermalAlignment"), -999.0)
    thermal_fraction = _finite(metrics.get("thermalContrastFraction"), 0.0)
    alignment_fraction = _finite(metrics.get("thermalAlignmentFraction"), 0.0)
    locator = _finite(metrics.get("locatorConfidence"), 0.0)
    synoptic = _finite(metrics.get("synopticSupport"), 0.0)
    terrain = _finite(metrics.get("terrainFraction"), 1.0)

    wind_shift = _finite(metrics.get("windShiftMs"), -999.0)
    wind_angle = _finite(metrics.get("windShiftAngleDeg"), -999.0)
    convergence = _finite(metrics.get("convergenceMs"), -999.0)
    convergence_fraction = _finite(metrics.get("convergenceFraction"), 0.0)
    vorticity = _finite(metrics.get("vorticity1e5"), -999.0)
    frontogenesis = _finite(metrics.get("frontogenesis"), -999.0)

    pressure_trough = _finite(metrics.get("pressureTroughHpa"))
    pressure_fraction = _finite(metrics.get("pressureTroughFraction"))
    lower_valid = _finite(metrics.get("lowerValidFraction"), 0.0)
    lower_support = _finite(metrics.get("lowerLevelSupport"))
    lower_contrast = _finite(metrics.get("deltaThetaW925"))
    cross_distance_raw = _finite(metrics.get("crossDistanceThermalSupport"))
    cross_distance = cross_distance_raw if np.isfinite(cross_distance_raw) else 0.0
    diagnosis, diagnosis_reasons = candidate_hypothesis(metrics)

    thermal_core = (
        delta_tw >= 1.20
        and delta_t >= 0.60
        and delta_tv >= 0.25
        and dry_gradient >= 0.80
        and alignment >= 0.05
        and thermal_fraction >= 0.48
        and alignment_fraction >= 0.48
        and (cross_distance >= 0.34 or not np.isfinite(cross_distance_raw))
    )
    if not thermal_core:
        reasons.append("thermal-core")

    structure_core = (
        locator >= 0.15 and synoptic >= 0.58 and terrain <= 0.65
    )
    if not structure_core:
        reasons.append("synoptic-structure")

    wind_boundary = wind_shift >= 1.8 or wind_angle >= 10.0
    if not wind_boundary:
        reasons.append("wind-boundary")

    dynamic_core = (
        convergence >= 0.12
        or convergence_fraction >= 0.52
        or vorticity >= 0.80
        or frontogenesis >= 0.25
    )
    if not dynamic_core:
        reasons.append("dynamic-core")

    # Strong cross-frontal divergence contradicts a material boundary unless
    # a clearly cyclonic/deforming flow independently supports it.
    divergent_contradiction = (
        convergence < -0.75
        and convergence_fraction < 0.32
        and vorticity < 1.8
        and frontogenesis < 1.0
    )
    if divergent_contradiction:
        reasons.append("divergent-wind")

    pressure_contradiction = (
        np.isfinite(pressure_trough) and pressure_trough < -0.25
    ) or (
        np.isfinite(pressure_fraction) and pressure_fraction < 0.28
        and pressure_trough < 0.05
    )
    if pressure_contradiction:
        reasons.append("pressure-ridge-warning")

    # Where 925 hPa is actually above ground over most of the line, the same
    # air-mass boundary must also be visible in the lower troposphere.
    lower_contradiction = (
        lower_valid >= 0.60
        and (
            not np.isfinite(lower_support)
            or not np.isfinite(lower_contrast)
            or lower_support < 0.25
            or lower_contrast < 0.50
        )
    )
    if lower_contradiction:
        reasons.append("lower-level-incoherence-warning")

    if diagnosis != "synoptic-front":
        reasons.append(diagnosis)

    continuation_pass = bool(
        thermal_core
        and structure_core
        and not divergent_contradiction
        and diagnosis == "synoptic-front"
    )

    strong_thermal = (
        delta_tw >= 1.55
        and delta_t >= 0.80
        and delta_tv >= 0.35
        and dry_gradient >= 0.95
        and alignment >= 0.12
        and thermal_fraction >= 0.56
        and alignment_fraction >= 0.54
        and cross_distance >= 0.50
    )
    strong_dynamic = wind_boundary and dynamic_core and (
        convergence >= -0.20 or convergence_fraction >= 0.44
    )
    pressure_support = (
        not np.isfinite(pressure_trough)
        or pressure_trough >= 0.0
        or (
            pressure_trough >= -0.10
            and np.isfinite(pressure_fraction)
            and pressure_fraction >= 0.48
        )
    )
    strong_pass = bool(
        continuation_pass
        and strong_thermal
        and pressure_support
        and not lower_contradiction
        and evidence["candidateEvidence"] >= (0.43 if strong_dynamic else 0.53)
    )
    return {
        "strongPass": strong_pass,
        "continuationPass": continuation_pass,
        "gateStatus": "strong" if strong_pass else (
            "continuation" if continuation_pass else "rejected"
        ),
        "rejectionReasons": reasons,
        "diagnosis": diagnosis,
        "diagnosisReasons": diagnosis_reasons,
        "windBoundary": wind_boundary,
        "dynamicSupport": dynamic_core,
        "pressureRidge": pressure_contradiction,
        "lowerLevelIncoherent": lower_contradiction,
    }


def candidate_is_plausible(metrics: dict, evidence: dict | None = None) -> bool:
    """Backward-compatible boolean view of the non-compensating gates."""
    return candidate_gate_report(metrics, evidence)["continuationPass"]
