#!/usr/bin/env python3
"""Create the leakage-safe DWD/ERA5 front training table.

The public DWD polyline archive is authoritative manual analysis. ERA5 is used
only as the historical predictor source; the operational model is then applied
to equivalent, unit-checked ICON-2I fields.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
from pathlib import Path
import sys

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meteo_analysis.ml.dwd_labels import iter_archive
from meteo_analysis.ml.era5 import EarthmoverERA5
from meteo_analysis.ml.features import (
    ERA5_TRANSFER_FEATURE_COLUMNS,
    compute_feature_frame,
    theta_gradient_50,
)
from meteo_analysis.ml.labels import grid_labels

DWD_URL = (
    "https://zenodo.org/api/records/5785817/files/"
    "DWDFrontsNA.tar.gz/content"
)
DWD_MD5 = "e9a9c26a5d5d10b6f83d7d5726115a50"


def download_archive(destination):
    destination = Path(destination)
    if destination.exists():
        digest = hashlib.md5(destination.read_bytes()).hexdigest()
        if digest == DWD_MD5:
            return destination
        raise ValueError("archivio DWD esistente con checksum errato")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(DWD_URL, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with partial.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
    if hashlib.md5(partial.read_bytes()).hexdigest() != DWD_MD5:
        partial.unlink(missing_ok=True)
        raise ValueError("checksum MD5 dell'archivio DWD non valido")
    partial.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="training_data/DWDFrontsNA.tar.gz")
    parser.add_argument("--output", default="training_data/front_features.parquet")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2019-12-31")
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Usa una analisi ogni N (1 = tutte le 00Z).",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    archive = download_archive(args.archive)
    analyses = sorted(
        iter_archive(archive, start=args.start, end=args.end, hours=(0,)),
        key=lambda item: item[0],
    )[::max(1, args.stride)]
    if args.limit:
        analyses = analyses[:args.limit]
    if not analyses:
        raise SystemExit("nessuna analisi DWD selezionata")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    era5 = EarthmoverERA5()
    try:
        first_write = True
        writer = None
        import pyarrow as pa
        import pyarrow.parquet as pq

        for index, (valid_time, fronts) in enumerate(analyses, 1):
            fields = era5.fields(valid_time)
            previous = era5.single_field(
                "msl", valid_time - timedelta(hours=3)
            )
            history = [
                theta_gradient_50(
                    era5.pressure_field(
                        "t", 850, valid_time - timedelta(hours=hours)
                    ),
                    era5.target_latitudes,
                    era5.target_longitudes,
                )
                for hours in (2, 1)
            ]
            features = compute_feature_frame(
                fields, era5.target_latitudes, era5.target_longitudes,
                valid_time=valid_time,
                previous_pmsl_3h=previous,
                gradient_history=history,
            )
            labels = grid_labels(fronts, valid_time=valid_time)
            frame = features[
                ["time", *ERA5_TRANSFER_FEATURE_COLUMNS]
            ].copy()
            frame["y"] = labels["y"].to_numpy()
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if first_write:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
                first_write = False
            writer.write_table(table)
            positives = int(frame.y.sum())
            print(
                f"[{index}/{len(analyses)}] {valid_time:%Y-%m-%d %HZ}: "
                f"{len(fronts)} linee, {positives} celle positive",
                flush=True,
            )
        if writer:
            writer.close()
    finally:
        era5.close()


if __name__ == "__main__":
    main()
