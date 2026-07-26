"""Reader for S. Niebler's DWD manual-front polyline archive."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import tarfile

import numpy as np

TYPE_MAP = {
    "cold": "cold",
    "warm": "warm",
    "occ": "occluded",
    "occluded": "occluded",
    "stnry": "stationary",
    "stationary": "stationary",
}
DATE_PATTERN = re.compile(r"(19|20)\d{6}(?:[_-]?([0-2]\d))?")
PAIR_PATTERN = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def timestamp_from_name(name: str):
    match = DATE_PATTERN.search(Path(name).stem)
    if not match:
        return None
    token = match.group(0).replace("_", "").replace("-", "")
    if len(token) == 8:
        token += "00"
    try:
        return datetime.strptime(token[:10], "%Y%m%d%H").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def parse_dwd_text(text: str):
    result = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        kind_token = line.split(None, 1)[0].lower().rstrip(":")
        kind = TYPE_MAP.get(kind_token)
        if kind is None:
            continue
        pairs = [(float(a), float(b)) for a, b in PAIR_PATTERN.findall(line)]
        # Native format is [lat, lon]; convert 0..360 longitudes to -180..180.
        coordinates = [
            [((lon + 180.0) % 360.0) - 180.0, lat] for lat, lon in pairs
            if -90 <= lat <= 90
        ]
        if len(coordinates) >= 2:
            result.append({"tipo": kind, "coordinates": coordinates})
    return result


def _line_length_km(coordinates):
    values = np.asarray(coordinates, float)
    if len(values) < 2:
        return 0.0
    mean_lat = np.deg2rad((values[1:, 1] + values[:-1, 1]) / 2.0)
    dx = np.diff(values[:, 0]) * 111.32 * np.cos(mean_lat)
    dy = np.diff(values[:, 1]) * 110.57
    return float(np.sum(np.hypot(dx, dy)))


def _intersects_bounds(coordinates, bounds, halo):
    west, east, south, north = bounds
    values = np.asarray(coordinates, float)
    return bool(
        np.any(
            (values[:, 0] >= west - halo)
            & (values[:, 0] <= east + halo)
            & (values[:, 1] >= south - halo)
            & (values[:, 1] <= north + halo)
        )
    )


def iter_archive(
    archive,
    *,
    bounds=(3.0, 22.0, 33.7, 48.9),
    halo_deg=0.75,
    minimum_length_km=80.0,
    start=None,
    end=None,
    hours=(0,),
):
    """Yield ``(timestamp, fronts)`` analyses selected without extracting."""
    start = None if start is None else np.datetime64(start)
    end = None if end is None else np.datetime64(end)
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            valid_time = timestamp_from_name(member.name)
            if valid_time is None or valid_time.hour not in set(hours):
                continue
            value = np.datetime64(valid_time.replace(tzinfo=None))
            if start is not None and value < start:
                continue
            if end is not None and value > end:
                continue
            source = bundle.extractfile(member)
            if source is None:
                continue
            fronts = []
            for front in parse_dwd_text(
                source.read().decode("utf-8", errors="replace")
            ):
                if (
                    _line_length_km(front["coordinates"]) >= minimum_length_km
                    and _intersects_bounds(front["coordinates"], bounds, halo_deg)
                ):
                    fronts.append(front)
            # Empty analyses are valid negative examples.
            yield valid_time, fronts
