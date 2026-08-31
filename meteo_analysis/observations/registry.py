"""Canonical station registry: merge, identify and deduplicate stations
coming from independent providers.

The registry never destroys a station because it looks similar to another:
it only ever *annotates* a confidence score for a possible correspondence
(requirement 8).  Two stations are only marked as duplicates of one another
above a high-confidence threshold, and the lower-priority one is flagged
``duplicateOfStationId`` while remaining fully present in the registry and
in ``observations`` (nothing is dropped).
"""

from __future__ import annotations

import math
from typing import Any

from meteo_analysis.observations.model import NETWORK_TYPE_INSTITUTIONAL

# A station from an institutional/aviation network is preferred as the
# "primary" record when two networks report the same physical station.
_SOURCE_PRIORITY = {"italiameteo": 0, "metar": 1, "meteonetwork": 2}

DUPLICATE_CONFIDENCE_THRESHOLD = 0.85


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return radius * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _name_similarity(a: str, b: str) -> float:
    """Cheap, dependency-free token-overlap similarity in [0, 1]."""

    tokens_a = {token for token in a.lower().replace("-", " ").split() if len(token) > 2}
    tokens_b = {token for token in b.lower().replace("-", " ").split() if len(token) > 2}
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    return overlap / max(len(tokens_a), len(tokens_b))


def match_confidence(station_a: dict[str, Any], station_b: dict[str, Any]) -> float:
    """Confidence in [0, 1] that two canonical stations are the same site."""

    if station_a.get("icaoId") and station_a.get("icaoId") == station_b.get("icaoId"):
        return 1.0
    if station_a.get("wmoId") and station_a.get("wmoId") == station_b.get("wmoId"):
        return 1.0
    distance_km = _haversine_km(
        station_a["lat"], station_a["lon"], station_b["lat"], station_b["lon"]
    )
    if distance_km > 2.0:
        return 0.0
    elevation_a = station_a.get("elevationM")
    elevation_b = station_b.get("elevationM")
    elevation_penalty = 0.0
    if elevation_a is not None and elevation_b is not None:
        elevation_penalty = min(abs(elevation_a - elevation_b) / 200.0, 0.4)
    distance_score = max(0.0, 1.0 - distance_km / 2.0)
    name_score = _name_similarity(station_a.get("name") or "", station_b.get("name") or "")
    confidence = 0.6 * distance_score + 0.25 * name_score - elevation_penalty
    if distance_km <= 0.05 and elevation_penalty <= 0.05:
        # Two stations essentially co-located (<50 m) and at the same
        # elevation are almost certainly the same physical site, even if
        # naming conventions differ between providers.
        confidence = max(confidence, 0.9)
    return max(0.0, min(1.0, confidence))


def build_registry(stations: list[dict[str, Any]]) -> dict[str, Any]:
    """Assign internal ids, detect duplicate candidates, return a registry.

    ``stations`` is the concatenation of every provider's canonical station
    list.  The result preserves every station (see module docstring) and
    adds registry-level metadata: ``internalStationId``,
    ``duplicateCandidates`` and ``duplicateOfStationId``.
    """

    registry: list[dict[str, Any]] = []
    for station in stations:
        record = dict(station)
        record["internalStationId"] = f"{record['source']}:{record['sourceStationId']}"
        record["duplicateCandidates"] = []
        record["duplicateOfStationId"] = None
        registry.append(record)

    # O(n^2) pairwise comparison is acceptable at today's station counts
    # (low thousands); a spatial index (geohash/grid bucket) should replace
    # this once volumes grow enough to matter.
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, record in enumerate(registry):
        key = (round(record["lat"] * 20), round(record["lon"] * 20))  # ~5 km cells
        buckets.setdefault(key, []).append(index)

    def neighbours(index: int) -> list[int]:
        record = registry[index]
        cell = (round(record["lat"] * 20), round(record["lon"] * 20))
        candidates: list[int] = []
        for dlat in (-1, 0, 1):
            for dlon in (-1, 0, 1):
                candidates.extend(buckets.get((cell[0] + dlat, cell[1] + dlon), []))
        return candidates

    for index, record in enumerate(registry):
        for other_index in neighbours(index):
            if other_index <= index:
                continue
            other = registry[other_index]
            if other["source"] == record["source"]:
                continue
            confidence = match_confidence(record, other)
            if confidence <= 0.0:
                continue
            record["duplicateCandidates"].append(
                {"internalStationId": other["internalStationId"], "confidence": round(confidence, 3)}
            )
            other["duplicateCandidates"].append(
                {"internalStationId": record["internalStationId"], "confidence": round(confidence, 3)}
            )
            if confidence >= DUPLICATE_CONFIDENCE_THRESHOLD:
                primary, secondary = sorted(
                    (record, other),
                    key=lambda item: _SOURCE_PRIORITY.get(item["source"], 9),
                )
                secondary["duplicateOfStationId"] = primary["internalStationId"]

    return {
        "stations": registry,
        "totalBeforeDeduplication": len(registry),
        "duplicatesFlagged": sum(
            1 for record in registry if record["duplicateOfStationId"]
        ),
    }
