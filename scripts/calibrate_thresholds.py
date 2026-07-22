"""Climatological calibration of the OFA detection thresholds (documento sez. 17).

The per-map adaptive quantiles inside ``front_locator`` are stable only within
one analysis time: a single quiet map can push the gradient threshold too low
and a single stormy map too high. This script builds a **climatology** of the
two decision fields the locator uses — the adjacent-baroclinic-zone gradient
``|grad theta_w|`` (K/100 km) and the thermal front parameter TFP (K/km^2,
negative on the warm edge) — over many ICON-2I runs, at the *same* two
smoothing scales the detector uses, split by month and latitude band. The
resulting quantiles give a stable base threshold per season/latitude; the
existing per-map tightening still applies on top.

It copies the *method*, not foreign absolute values: thresholds are derived
from this domain's own ICON-2I fields.

Run:  python calibrate_thresholds.py <t850,q850>... --out climatology_thresholds.json
The pairs are auto-discovered from the standard scratch layout when omitted.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
import front_locator as fl
import thermodynamics as thermo

BOUNDS = (3.0, 22.0, 33.7, 48.9)
FACTOR = 4
ANALYSIS_PRESSURE_PA = 85000.0
# Same smoothing scales as front_analysis_v12 / detect_fronts_two_scale.
SCALES = {
    "synoptic": {"sigma_km": 100.0, "deriv_km": max(15.0, 100.0 * 0.3)},
    "refined": {"sigma_km": 45.0, "deriv_km": 15.0},
}
LAT_BANDS = {"south": (33.7, 40.0), "central": (40.0, 44.0), "north": (44.0, 48.9)}
LEADS = [0, 6, 12, 18, 24, 36, 48, 60, 72]


def _read_level(path: str, step_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    var = next(iter(ds.data_vars))
    data = ds[var]
    if "step" in data.dims:
        data = data.isel(step=min(step_index, data.sizes["step"] - 1))
    data = data.squeeze(drop=True).sortby("latitude").sortby("longitude")
    lon_min, lon_max, lat_min, lat_max = BOUNDS
    data = data.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    data = data.transpose("latitude", "longitude")
    values = np.asarray(data.values, dtype=float)[::FACTOR, ::FACTOR]
    lats = np.asarray(data.latitude.values, dtype=float)[::FACTOR]
    lons = np.asarray(data.longitude.values, dtype=float)[::FACTOR]
    ds.close()
    return values, lons, lats


def _decision_fields(theta_w, lons, lats, sigma_km, deriv_km):
    metrics = fl.grid_metrics(lons, lats)
    field = fl.smooth_km(theta_w, sigma_km, metrics)
    grad_e, grad_n = fl.gradient(field, metrics)
    grad_mag = fl.smooth_km(np.hypot(grad_e, grad_n), deriv_km, metrics)
    gm_e, gm_n = fl.gradient(grad_mag, metrics)
    safe = np.maximum(grad_mag, 1.0e-9)
    tfp = (gm_e * grad_e + gm_n * grad_n) / safe
    local_grid_km = np.sqrt(metrics["dx_km_col"] * metrics["dy_km"])
    abz = (grad_mag + (local_grid_km / np.sqrt(2.0)) * np.hypot(gm_e, gm_n)) * 100.0
    return tfp, abz


def calibrate(pairs: list[tuple[str, str, str]]) -> dict:
    # month -> scale -> band -> {"tfp": [...], "abz": [...]}
    acc: dict = {}
    for t_path, q_path, tag in pairs:
        month = tag[4:6] if len(tag) >= 6 else "all"
        for lead in LEADS:
            try:
                temp, lons, lats = _read_level(t_path, lead)
                hum, _, _ = _read_level(q_path, lead)
            except Exception as error:
                print(f"   {tag}+{lead}h saltato: {error}", flush=True)
                continue
            pressure = np.full_like(temp, ANALYSIS_PRESSURE_PA)
            fields = thermo.thermodynamic_fields(pressure, temp, hum, method="davies_jones")
            theta_w = np.where(fields["out_of_domain"], np.nan, fields["theta_w"])
            lat_grid = np.broadcast_to(lats[:, None], theta_w.shape)
            for scale, cfg in SCALES.items():
                tfp, abz = _decision_fields(theta_w, lons, lats, cfg["sigma_km"], cfg["deriv_km"])
                finite = np.isfinite(tfp) & np.isfinite(abz)
                for band, (lo, hi) in {**LAT_BANDS, "all": (BOUNDS[2], BOUNDS[3])}.items():
                    mask = finite & (lat_grid >= lo) & (lat_grid < hi)
                    if np.count_nonzero(mask) < 50:
                        continue
                    node = acc.setdefault(month, {}).setdefault(scale, {}).setdefault(
                        band, {"tfp": [], "abz": []}
                    )
                    # subsample to keep memory bounded but representative
                    node["tfp"].append(np.random.choice(tfp[mask], size=min(4000, mask.sum()), replace=False))
                    node["abz"].append(np.random.choice(abz[mask], size=min(4000, mask.sum()), replace=False))
        print(f"   elaborato run {tag}", flush=True)

    out: dict = {"_meta": {
        "source": "ICON-2I",
        "runs": [tag for _, _, tag in pairs],
        "method": "domain quantiles of |grad theta_w| and TFP at the detector scales",
        "note": "tfp in K/km^2 (negative warm edge); abz in K/100km",
    }}
    for month, scales in acc.items():
        for scale, bands in scales.items():
            for band, arrays in bands.items():
                tfp = np.concatenate(arrays["tfp"])
                abz = np.concatenate(arrays["abz"])
                out.setdefault(month, {}).setdefault(scale, {})[band] = {
                    "n": int(tfp.size),
                    "tfp_weak": round(float(np.quantile(tfp, 0.25)), 8),
                    "tfp_full": round(float(np.quantile(tfp, 0.05)), 8),
                    "abz_weak": round(float(np.quantile(abz, 0.50)), 4),
                    "abz_full": round(float(np.quantile(abz, 0.85)), 4),
                }
    return out


def discover_pairs() -> list[tuple[str, str, str]]:
    scratch = os.environ.get("CLIM_SCRATCH", "")
    pairs = []
    # explicit clim/ downloads: t850_<tag>.grib / q850_<tag>.grib
    for t_path in sorted(glob.glob(os.path.join(scratch, "clim", "t850_*.grib"))):
        tag = re.search(r"t850_(\d+)\.grib", t_path).group(1)
        q_path = t_path.replace("t850_", "q850_")
        if os.path.exists(q_path):
            pairs.append((t_path, q_path, tag))
    # standard per-run dirs with t850.grib/q850.grib
    for d, tag in (("gribs", "2026072000"), ("gribs12", "2026072012"), ("g22", "2026072200")):
        t_path = os.path.join(scratch, d, "t850.grib")
        q_path = os.path.join(scratch, d, "q850.grib")
        if os.path.exists(t_path) and os.path.exists(q_path):
            pairs.append((t_path, q_path, tag))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "climatology_thresholds.json"))
    args = ap.parse_args()
    np.random.seed(0)
    pairs = discover_pairs()
    if not pairs:
        raise SystemExit("nessuna coppia t850/q850 trovata (imposta CLIM_SCRATCH)")
    print(f"Calibro su {len(pairs)} run: {[p[2] for p in pairs]}", flush=True)
    climatology = calibrate(pairs)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(climatology, handle, indent=1, ensure_ascii=False)
    print(f"Scritto {args.out}", flush=True)
    # brief human summary
    for month, scales in climatology.items():
        if month == "_meta":
            continue
        for scale, bands in scales.items():
            a = bands.get("all", {})
            print(f"  mese {month} {scale:8s} all: abz {a.get('abz_weak')}->{a.get('abz_full')} "
                  f"tfp {a.get('tfp_weak')}->{a.get('tfp_full')} (n={a.get('n')})")


if __name__ == "__main__":
    main()
