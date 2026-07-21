"""Thermodynamic fields for objective frontal analysis (v10 core).

Self-contained, fully vectorised NumPy.  All internal computation uses
explicit SI-consistent units:

    pressure    Pa
    temperature K
    specific humidity q   kg/kg
    mixing ratio          kg/kg (dimensionless)
    vapour pressure       Pa

The primary field of the v10 detector is the **wet-bulb potential
temperature** theta_w, computed with the Davies-Jones (2008) rational
approximation from theta_e.  theta (dry potential temperature) and theta_e
are provided as auxiliary fields: theta checks real (dry) baroclinicity,
theta_e describes the moist contrast.

References
----------
Bolton (1980), MWR 108, 1046-1053 - theta_e and saturation vapour pressure.
Davies-Jones (2008), MWR 136, 2764-2785 - theta_w from theta_e.
The Davies-Jones polynomial coefficients match those in MetPy's
``wet_bulb_potential_temperature`` (validated to <1e-3 K in tests).
"""

from __future__ import annotations

import numpy as np


# --- physical constants (Bolton-consistent) --------------------------------
KAPPA = 0.2854            # Rd/cp for the theta_dl exponent (Bolton eq. 38)
EPSILON = 0.62197         # Rd/Rv
P0_PA = 100_000.0         # reference pressure for potential temperature
CELSIUS0 = 273.15

# Davies-Jones (2008) theta_w(theta_e) rational approximation.
_A = (7.101574, -20.68208, 16.11182, 2.574631, -5.205688)
_B = (1.0, -3.552497, 3.781782, -0.6899655, -0.5929340)
_DJ_FLOOR_K = 173.15      # below this theta_e, theta_w = theta_e

# Nominal validity domain (meteorological range in which the chain and the
# Davies-Jones fit are trustworthy).  Outside it results are flagged.
DOMAIN = {
    "pressure_pa": (40_000.0, 105_000.0),
    "temperature_k": (200.0, 330.0),
    "specific_humidity": (0.0, 0.05),
}


def saturation_vapour_pressure_pa(temperature_k: np.ndarray) -> np.ndarray:
    """Bolton (1980) eq. 10 saturation vapour pressure, in Pa.

    e_s = 611.2 Pa * exp(17.67 * Tc / (Tc + 243.5)),  Tc in degrees Celsius.
    """
    tc = np.asarray(temperature_k, dtype=float) - CELSIUS0
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def dewpoint_k_from_vapour_pressure(vapour_pressure_pa: np.ndarray) -> np.ndarray:
    """Invert Bolton's saturation vapour pressure for the dew point (K)."""
    e_hpa = np.asarray(vapour_pressure_pa, dtype=float) / 100.0
    logarithm = np.log(np.maximum(e_hpa, 1.0e-12) / 6.112)
    dewpoint_c = 243.5 * logarithm / (17.67 - logarithm)
    return dewpoint_c + CELSIUS0


def potential_temperature_k(pressure_pa: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    """Dry potential temperature theta = T (P0/p)^kappa."""
    return np.asarray(temperature_k, dtype=float) * (
        P0_PA / np.asarray(pressure_pa, dtype=float)
    ) ** KAPPA


def equivalent_potential_temperature_k(
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    specific_humidity: np.ndarray,
) -> np.ndarray:
    """Bolton (1980) equivalent potential temperature (K) from p, T, q.

    Mirrors the Bolton theta_e formula used by MetPy, but with a
    self-consistent Bolton saturation vapour pressure (rather than a mixed
    modern svp), as required for a clean Bolton chain feeding Davies-Jones.
    """
    p = np.asarray(pressure_pa, dtype=float)
    t = np.asarray(temperature_k, dtype=float)
    q = np.clip(np.asarray(specific_humidity, dtype=float), 0.0, None)

    mixing_ratio = q / np.maximum(1.0 - q, 1.0e-9)
    vapour_pressure = p * mixing_ratio / (EPSILON + mixing_ratio)
    dewpoint = dewpoint_k_from_vapour_pressure(vapour_pressure)
    # The dew point cannot exceed the temperature (supersaturation guard).
    dewpoint = np.minimum(dewpoint, t)

    # Bolton (1980) eq. 15 LCL temperature, then eq. 24 theta_e.
    t_lcl = 56.0 + 1.0 / (
        1.0 / (dewpoint - 56.0) + np.log(t / dewpoint) / 800.0
    )
    theta_dl = potential_temperature_k(p - vapour_pressure, t) * (
        t / t_lcl
    ) ** (0.28 * mixing_ratio)
    return theta_dl * np.exp(
        mixing_ratio * (1.0 + 0.448 * mixing_ratio) * (3036.0 / t_lcl - 1.78)
    )


def wet_bulb_potential_temperature_k(theta_e_k: np.ndarray) -> np.ndarray:
    """Davies-Jones (2008) theta_w from theta_e (both K), vectorised."""
    theta_e = np.asarray(theta_e_k, dtype=float)
    x = theta_e / CELSIUS0
    numerator = _A[0] + x * (_A[1] + x * (_A[2] + x * (_A[3] + x * _A[4])))
    denominator = _B[0] + x * (_B[1] + x * (_B[2] + x * (_B[3] + x * _B[4])))
    with np.errstate(over="ignore", invalid="ignore"):
        theta_w = theta_e - np.exp(numerator / denominator)
    return np.where(theta_e <= _DJ_FLOOR_K, theta_e, theta_w)


def _theta_e_of_saturated_parcel(theta_w_k: np.ndarray) -> np.ndarray:
    """theta_e of a saturated parcel sitting at 1000 hPa with T = theta_w.

    Used only by the optional Newton refinement: at P0 a saturated parcel of
    temperature theta_w has, by definition, the theta_e whose theta_w we seek.
    """
    t = np.asarray(theta_w_k, dtype=float)
    e_s = saturation_vapour_pressure_pa(t)
    mixing_ratio = EPSILON * e_s / np.maximum(P0_PA - e_s, 1.0e-6)
    t_lcl = t  # saturated: LCL is the parcel itself
    theta_dl = potential_temperature_k(P0_PA - e_s, t) * (t / t_lcl) ** (
        0.28 * mixing_ratio
    )
    return theta_dl * np.exp(
        mixing_ratio * (1.0 + 0.448 * mixing_ratio) * (3036.0 / t_lcl - 1.78)
    )


def _newton_refine_theta_w(theta_w_k: np.ndarray, theta_e_k: np.ndarray) -> np.ndarray:
    """One Newton step so that theta_e(theta_w) matches the input theta_e.

    Davies-Jones' single iteration brings the residual well below 0.002 K.
    Not used in the ordinary workflow (the direct polynomial is enough);
    available as a validation/fallback refinement, e.g. near the domain edge.
    """
    theta_w = np.asarray(theta_w_k, dtype=float)
    delta = 0.01
    f0 = _theta_e_of_saturated_parcel(theta_w) - theta_e_k
    f1 = _theta_e_of_saturated_parcel(theta_w + delta) - theta_e_k
    slope = (f1 - f0) / delta
    with np.errstate(invalid="ignore", divide="ignore"):
        step = np.where(np.abs(slope) > 1.0e-6, f0 / slope, 0.0)
    return theta_w - step


def thermodynamic_fields(
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    specific_humidity: np.ndarray,
    method: str = "davies_jones",
) -> dict:
    """Compute the three thermodynamic fields plus an out-of-domain mask.

    Parameters
    ----------
    method : "davies_jones" (default, direct polynomial - the operational
        path) or "davies_jones_newton" (one Newton refinement on top; for
        validation or controlled fallback, not for the whole ICON grid).

    Returns a dict with ``theta``, ``theta_w``, ``theta_e`` (all K) and
    ``out_of_domain`` (bool array).  Non-finite inputs and unphysical
    thermodynamic states are set to NaN, never silently clipped.
    """
    if method not in ("davies_jones", "davies_jones_newton"):
        raise ValueError(f"metodo sconosciuto: {method!r}")

    p = np.asarray(pressure_pa, dtype=float)
    t = np.asarray(temperature_k, dtype=float)
    q = np.asarray(specific_humidity, dtype=float)

    finite = np.isfinite(p) & np.isfinite(t) & np.isfinite(q)
    physical = finite & (p > 0.0) & (t > 150.0) & (q >= 0.0)

    p_lo, p_hi = DOMAIN["pressure_pa"]
    t_lo, t_hi = DOMAIN["temperature_k"]
    q_lo, q_hi = DOMAIN["specific_humidity"]
    out_of_domain = ~(
        (p >= p_lo) & (p <= p_hi)
        & (t >= t_lo) & (t <= t_hi)
        & (q >= q_lo) & (q <= q_hi)
    )

    safe_p = np.where(physical, p, np.nan)
    safe_t = np.where(physical, t, np.nan)
    safe_q = np.where(physical, q, np.nan)

    theta = potential_temperature_k(safe_p, safe_t)
    theta_e = equivalent_potential_temperature_k(safe_p, safe_t, safe_q)
    theta_w = wet_bulb_potential_temperature_k(theta_e)
    if method == "davies_jones_newton":
        refined = _newton_refine_theta_w(theta_w, theta_e)
        theta_w = np.where(np.isfinite(refined), refined, theta_w)

    invalid = ~physical
    theta = np.where(invalid, np.nan, theta)
    theta_e = np.where(invalid, np.nan, theta_e)
    theta_w = np.where(invalid, np.nan, theta_w)

    return {
        "theta": theta,
        "theta_w": theta_w,
        "theta_e": theta_e,
        "out_of_domain": out_of_domain | invalid,
    }
