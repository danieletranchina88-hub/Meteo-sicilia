"""Numerical validation of thermodynamics.py against MetPy 1.7.1.

Not part of the production workflow (MetPy is a validation-only
dependency, intentionally NOT in requirements.txt).  Run manually:

    pip install metpy
    python scripts/tests/test_thermodynamics.py

Grid: p 700-1000 hPa, T 235-315 K, RH 5-100%.  Two separate comparisons:
  A) Davies-Jones theta_w(theta_e) fed the SAME theta_e as MetPy -> isolates
     the polynomial (must match to ~machine precision).
  B) full chain theta_w(p,T,q) -> includes the upstream theta_e difference
     (self-consistent Bolton here vs MetPy's Ambaum svp).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import thermodynamics as td

from metpy.calc import (
    dewpoint_from_relative_humidity,
    equivalent_potential_temperature,
    specific_humidity_from_dewpoint,
    wet_bulb_potential_temperature,
)
from metpy.units import units


def main() -> int:
    pressure = np.arange(700.0, 1000.1, 25.0)
    temperature = np.arange(235.0, 315.1, 5.0)
    humidity = np.array([5, 10, 20, 30, 50, 70, 90, 100], dtype=float) / 100.0
    grid_p, grid_t, grid_rh = np.meshgrid(
        pressure, temperature, humidity, indexing="ij"
    )

    p_hpa = grid_p.ravel() * units.hPa
    t_k = grid_t.ravel() * units.kelvin
    rh = grid_rh.ravel() * units.dimensionless

    dew = dewpoint_from_relative_humidity(t_k, rh)
    q = specific_humidity_from_dewpoint(p_hpa, dew).m_as("dimensionless")

    theta_e_ref = equivalent_potential_temperature(p_hpa, t_k, dew).m_as("kelvin")
    theta_w_ref = wet_bulb_potential_temperature(p_hpa, t_k, dew).m_as("kelvin")
    valid = np.isfinite(theta_e_ref) & np.isfinite(theta_w_ref) & (theta_e_ref > 173.15)

    # A) polynomial isolated
    theta_w_from_ref = td.wet_bulb_potential_temperature_k(theta_e_ref)
    err_poly = np.abs(theta_w_from_ref - theta_w_ref)[valid]

    # B) full chain
    p_pa = grid_p.ravel() * 100.0
    theta_e_mine = td.equivalent_potential_temperature_k(p_pa, grid_t.ravel(), q)
    theta_w_mine = td.wet_bulb_potential_temperature_k(theta_e_mine)
    err_theta_e = np.abs(theta_e_mine - theta_e_ref)[valid]
    err_chain = np.abs(theta_w_mine - theta_w_ref)[valid]

    # C) Newton refinement residual
    refined = td.thermodynamic_fields(
        p_pa, grid_t.ravel(), q, method="davies_jones_newton"
    )["theta_w"]
    recon = td._theta_e_of_saturated_parcel(refined)
    residual = np.abs(recon - theta_e_mine)[valid & np.isfinite(refined)]

    # D) monotonicity / continuity
    sweep = np.linspace(200.0, 360.0, 400)
    theta_w_sweep = td.wet_bulb_potential_temperature_k(sweep)
    monotone = bool(np.all(np.diff(theta_w_sweep) > 0.0))

    print(f"A) Davies-Jones polinomio: max {err_poly.max() * 1000:.4f} mK")
    print(f"B) catena completa theta_w: max {err_chain.max():.4f} K "
          f"(theta_e upstream max {err_theta_e.max():.4f} K)")
    print(f"C) residuo Newton: max {residual.max() * 1000:.3f} mK")
    print(f"D) theta_w monotona in theta_e: {monotone}")

    ok = (
        err_poly.max() < 1.0e-3
        and err_chain.max() <= 0.05
        and residual.max() < 2.0e-3
        and monotone
    )
    print("ESITO:", "SUPERATO" if ok else "DA RIVEDERE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
