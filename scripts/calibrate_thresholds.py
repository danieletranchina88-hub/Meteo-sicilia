"""Estimate candidate monthly thresholds for the front detector.

This script samples the two decision fields used by ``front_locator`` from
many independent ICON-2I runs.  The resulting quantiles are *candidate*
distributional thresholds: they are not promoted to operational use until an
independent labelled benchmark has shown an improvement over the frozen
defaults.

Examples:

    python calibrate_thresholds.py --scratch /data/icon-history
    python calibrate_thresholds.py T.grib,Q.grib,2026072300 --out candidates.json

Each explicit positional input is ``T850_PATH,Q850_PATH,RUN_TAG``.  ``RUN_TAG``
must start with YYYYMMDDHH so that months and independent days can be counted.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
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
SCALES = {
    "synoptic": {"sigma_km": 100.0, "deriv_km": 30.0},
    "refined": {"sigma_km": 45.0, "deriv_km": 15.0},
}
LAT_BANDS = {
    "south": (33.7, 40.0),
    "central": (40.0, 44.0),
    "north": (44.0, 48.9),
}
LEADS = [0, 6, 12, 18, 24, 36, 48, 60, 72]
MIN_OPERATIONAL_RUNS = 30
MIN_OPERATIONAL_DAYS = 15
MAX_STEP_MISMATCH_HOURS = 1.0
RUN_TAG_RE = re.compile(r"(?<!\d)(\d{10})(?!\d)")


def _coordinate_hours(values: np.ndarray) -> np.ndarray:
    """Convert a GRIB step coordinate to forecast hours."""
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.timedelta64):
        return values / np.timedelta64(1, "h")
    if values.dtype == object:
        converted = []
        for value in values:
            if hasattr(value, "total_seconds"):
                converted.append(value.total_seconds() / 3600.0)
            else:
                converted.append(float(value))
        return np.asarray(converted, dtype=float)
    return values.astype(float)


def _read_level(path: str, lead_hour: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the requested lead by coordinate value, never by positional index."""
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        var = next(iter(ds.data_vars))
        data = ds[var]
        if "step" in data.dims:
            step_hours = _coordinate_hours(data["step"].values)
            step_index = int(np.argmin(np.abs(step_hours - lead_hour)))
            mismatch = abs(float(step_hours[step_index]) - lead_hour)
            if mismatch > MAX_STEP_MISMATCH_HOURS:
                raise ValueError(
                    f"lead {lead_hour}h assente; il più vicino è "
                    f"{float(step_hours[step_index]):g}h"
                )
            data = data.isel(step=step_index)
        elif lead_hour != 0:
            raise ValueError("file senza coordinata step: utilizzabile solo a +0h")

        data = data.squeeze(drop=True).sortby("latitude").sortby("longitude")
        lon_min, lon_max, lat_min, lat_max = BOUNDS
        data = data.sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max),
        ).transpose("latitude", "longitude")
        values = np.asarray(data.values, dtype=float)[::FACTOR, ::FACTOR]
        lats = np.asarray(data.latitude.values, dtype=float)[::FACTOR]
        lons = np.asarray(data.longitude.values, dtype=float)[::FACTOR]
        return values, lons, lats
    finally:
        ds.close()


def _decision_fields(theta_w, lons, lats, sigma_km, deriv_km):
    metrics = fl.grid_metrics(lons, lats)
    field = fl.smooth_km(theta_w, sigma_km, metrics)
    grad_e, grad_n = fl.gradient(field, metrics)
    grad_mag = fl.smooth_km(np.hypot(grad_e, grad_n), deriv_km, metrics)
    gm_e, gm_n = fl.gradient(grad_mag, metrics)
    safe = np.maximum(grad_mag, 1.0e-9)
    tfp = (gm_e * grad_e + gm_n * grad_n) / safe
    local_grid_km = np.sqrt(metrics["dx_km_col"] * metrics["dy_km"])
    abz = (
        grad_mag
        + (local_grid_km / np.sqrt(2.0)) * np.hypot(gm_e, gm_n)
    ) * 100.0
    return tfp, abz


def _sample_pair(
    tfp_values: np.ndarray,
    abz_values: np.ndarray,
    size: int,
    seed_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Take a deterministic joint sample, reproducible across executions."""
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    sample_size = min(size, tfp_values.size)
    indexes = rng.choice(tfp_values.size, size=sample_size, replace=False)
    return tfp_values[indexes], abz_values[indexes]


def calibrate(pairs: list[tuple[str, str, str]]) -> dict:
    # month -> scale -> band -> {"tfp": [...], "abz": [...]}
    acc: dict = {}
    accepted_runs: list[str] = []
    for t_path, q_path, tag in pairs:
        if not RUN_TAG_RE.search(tag):
            print(f"   run {tag!r} saltato: tag privo di YYYYMMDDHH", flush=True)
            continue
        run_tag = RUN_TAG_RE.search(tag).group(1)
        month = run_tag[4:6]
        accepted_any = False
        for lead in LEADS:
            try:
                temp, lons, lats = _read_level(t_path, lead)
                hum, q_lons, q_lats = _read_level(q_path, lead)
            except Exception as error:
                print(f"   {run_tag}+{lead}h saltato: {error}", flush=True)
                continue
            if temp.shape != hum.shape or not (
                np.array_equal(lons, q_lons) and np.array_equal(lats, q_lats)
            ):
                print(f"   {run_tag}+{lead}h saltato: griglie T/Q diverse", flush=True)
                continue

            pressure = np.full_like(temp, ANALYSIS_PRESSURE_PA)
            fields = thermo.thermodynamic_fields(
                pressure, temp, hum, method="davies_jones"
            )
            theta_w = np.where(fields["out_of_domain"], np.nan, fields["theta_w"])
            lat_grid = np.broadcast_to(lats[:, None], theta_w.shape)
            for scale, cfg in SCALES.items():
                tfp, abz = _decision_fields(
                    theta_w, lons, lats, cfg["sigma_km"], cfg["deriv_km"]
                )
                finite = np.isfinite(tfp) & np.isfinite(abz)
                bands = {**LAT_BANDS, "all": (BOUNDS[2], BOUNDS[3])}
                for band, (lo, hi) in bands.items():
                    mask = finite & (lat_grid >= lo) & (lat_grid < hi)
                    if np.count_nonzero(mask) < 50:
                        continue
                    sampled_tfp, sampled_abz = _sample_pair(
                        tfp[mask],
                        abz[mask],
                        4000,
                        f"{run_tag}:{lead}:{scale}:{band}",
                    )
                    node = (
                        acc.setdefault(month, {})
                        .setdefault(scale, {})
                        .setdefault(band, {"tfp": [], "abz": []})
                    )
                    node["tfp"].append(sampled_tfp)
                    node["abz"].append(sampled_abz)
                    accepted_any = True
        if accepted_any:
            accepted_runs.append(run_tag)
        print(f"   elaborato run {run_tag}", flush=True)

    unique_runs = sorted(set(accepted_runs))
    month_coverage = {}
    for month in sorted(acc):
        month_runs = [tag for tag in unique_runs if tag[4:6] == month]
        month_days = sorted({tag[:8] for tag in month_runs})
        month_coverage[month] = {
            "independentRuns": len(month_runs),
            "distinctDays": len(month_days),
            "minimumRuns": MIN_OPERATIONAL_RUNS,
            "minimumDays": MIN_OPERATIONAL_DAYS,
            "sampleSizeEligible": (
                len(month_runs) >= MIN_OPERATIONAL_RUNS
                and len(month_days) >= MIN_OPERATIONAL_DAYS
            ),
        }

    out: dict = {
        "_meta": {
            "schemaVersion": 2,
            "source": "ICON-2I",
            "runs": unique_runs,
            "monthCoverage": month_coverage,
            "method": (
                "domain quantiles of |grad theta_w| and TFP at the detector scales"
            ),
            "note": "tfp in K/km^2 (negative warm edge); abz in K/100km",
            "operationalValidated": False,
            "validationNote": (
                "Candidate quantiles only. Set operationalValidated=true manually "
                "only after a frozen independent labelled benchmark improves."
            ),
        }
    }
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


def parse_pair_specs(specs: list[str]) -> list[tuple[str, str, str]]:
    pairs = []
    for spec in specs:
        parts = [part.strip() for part in spec.split(",", 2)]
        if len(parts) != 3 or not all(parts):
            raise SystemExit(
                f"coppia non valida {spec!r}; usa T850_PATH,Q850_PATH,YYYYMMDDHH"
            )
        t_path, q_path, tag = parts
        if not os.path.isfile(t_path) or not os.path.isfile(q_path):
            raise SystemExit(f"file non trovato nella coppia {spec!r}")
        pairs.append((t_path, q_path, tag))
    return pairs


def discover_pairs(scratch: str) -> list[tuple[str, str, str]]:
    """Recursively discover t850/q850 pairs without hard-coded run folders."""
    if not scratch:
        return []
    pairs = []
    seen = set()
    pattern = os.path.join(os.path.abspath(scratch), "**", "t850*.grib")
    for t_path in sorted(glob.glob(pattern, recursive=True)):
        basename = os.path.basename(t_path)
        q_name = re.sub(r"^t850", "q850", basename, count=1)
        q_path = os.path.join(os.path.dirname(t_path), q_name)
        if not os.path.isfile(q_path):
            continue
        match = RUN_TAG_RE.search(t_path)
        if match is None:
            print(f"   coppia ignorata, run tag assente: {t_path}", flush=True)
            continue
        key = (os.path.realpath(t_path), os.path.realpath(q_path), match.group(1))
        if key not in seen:
            seen.add(key)
            pairs.append((t_path, q_path, match.group(1)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stima soglie candidate; non le valida automaticamente."
    )
    parser.add_argument(
        "pairs",
        nargs="*",
        metavar="T850,Q850,RUN",
        help="coppia esplicita; RUN deve iniziare con YYYYMMDDHH",
    )
    parser.add_argument(
        "--scratch",
        default=os.environ.get("CLIM_SCRATCH", ""),
        help="radice ricorsiva dell'archivio, oppure imposta CLIM_SCRATCH",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "climatology_thresholds.json"),
    )
    args = parser.parse_args()

    pairs = parse_pair_specs(args.pairs) if args.pairs else discover_pairs(args.scratch)
    if not pairs:
        raise SystemExit(
            "nessuna coppia t850/q850 trovata; passa T,Q,RUN o --scratch"
        )
    print(f"Stimo su {len(pairs)} run: {[p[2] for p in pairs]}", flush=True)
    climatology = calibrate(pairs)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(climatology, handle, indent=1, ensure_ascii=False)
    print(f"Scritto {args.out}", flush=True)
    for month, scales in climatology.items():
        if month == "_meta":
            continue
        for scale, bands in scales.items():
            whole_domain = bands.get("all", {})
            print(
                f"  mese {month} {scale:8s} all: "
                f"abz {whole_domain.get('abz_weak')}->{whole_domain.get('abz_full')} "
                f"tfp {whole_domain.get('tfp_weak')}->{whole_domain.get('tfp_full')} "
                f"(n={whole_domain.get('n')})"
            )


if __name__ == "__main__":
    main()
