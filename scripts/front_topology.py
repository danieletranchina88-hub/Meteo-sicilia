"""Topological consistency for published synoptic-front objects (v19).

Detection and tracking decide whether a physical front exists.  This module
operates later, on the exact geometry sent to the map, because ridge snapping
and shared-trunk removal can change a line's centroid and retained extent.
Its job is deliberately narrow:

* keep one exclusive type on every frontal arc;
* prevent a trimmed branch from jumping to the opposite end of a shared
  boundary between adjacent hours;
* remove interpolated geometries whose endpoints are not the same physical
  object and split the published identity at any remaining discontinuity;
* expose quantitative post-publication motion checks.

It never creates a front and never overrides the thermodynamic gates.
"""

from __future__ import annotations

import numpy as np

import front_tracking as ftk


MOVING_TYPES = {"cold", "warm"}
FRONT_TYPES = MOVING_TYPES | {"stationary", "occluded"}


def _merge_segments(segments: list[dict]) -> list[dict]:
    """Merge adjacent equal labels and force an exact [0, 1] tiling."""
    ordered = sorted(segments, key=lambda item: float(item.get("start", 0.0)))
    merged: list[dict] = []
    for raw in ordered:
        try:
            start = float(raw.get("start", 0.0))
            end = float(raw.get("end", 1.0))
        except (TypeError, ValueError):
            continue
        label = raw.get("type")
        if label not in FRONT_TYPES or not np.isfinite(start + end) or end <= start:
            continue
        item = dict(raw)
        item["start"] = float(np.clip(start, 0.0, 1.0))
        item["end"] = float(np.clip(end, 0.0, 1.0))
        if item["end"] <= item["start"]:
            continue
        if merged and merged[-1]["type"] == label:
            merged[-1]["end"] = item["end"]
            merged[-1]["certainty"] = round(min(
                float(merged[-1].get("certainty", 0.0)),
                float(item.get("certainty", 0.0)),
            ), 2)
        else:
            merged.append(item)
    if not merged:
        return []
    merged[0]["start"] = 0.0
    for previous, following in zip(merged[:-1], merged[1:]):
        boundary = 0.5 * (float(previous["end"]) + float(following["start"]))
        previous["end"] = boundary
        following["start"] = boundary
    merged[-1]["end"] = 1.0
    for item in merged:
        item["start"] = round(float(item["start"]), 3)
        item["end"] = round(float(item["end"]), 3)
    return merged


def cohere_feature_type(properties: dict) -> dict:
    """Make ``frontType`` and ``segmentTypes`` mutually consistent.

    A single tracked arc may contain its moving type plus stationary pieces,
    but never simultaneous cold and warm pieces.  An occluded arc is wholly
    occluded.  When trimming leaves only a stationary piece, the feature's
    main label is changed too, so metadata and rendered symbols agree.
    """
    segments = properties.get("segmentTypes")
    anchor = properties.get("frontType")
    original_anchor = anchor
    if anchor == "occluded":
        properties["segmentTypes"] = [{
            "start": 0.0, "end": 1.0, "type": "occluded", "certainty": 1.0,
        }]
        return properties
    if not isinstance(segments, list) or not segments:
        return properties

    cleaned = _merge_segments([dict(item) for item in segments])
    if not cleaned:
        properties.pop("segmentTypes", None)
        return properties

    if anchor == "stationary":
        for item in cleaned:
            item["type"] = "stationary"
    elif anchor in MOVING_TYPES:
        opposite = "warm" if anchor == "cold" else "cold"
        for item in cleaned:
            if item["type"] == opposite:
                item["type"] = "stationary"
    else:
        # Legacy/uncertain metadata: retain the moving type with the greatest
        # arc-length support; the competing sign is demoted to stationary.
        coverage = {
            moving: sum(
                float(item["end"]) - float(item["start"])
                for item in cleaned if item["type"] == moving
            )
            for moving in MOVING_TYPES
        }
        chosen = max(coverage, key=coverage.get)
        if coverage[chosen] > 0.0:
            for item in cleaned:
                if item["type"] in MOVING_TYPES and item["type"] != chosen:
                    item["type"] = "stationary"
            anchor = chosen

    cleaned = _merge_segments(cleaned)
    moving = {item["type"] for item in cleaned if item["type"] in MOVING_TYPES}
    if len(moving) == 1:
        properties["frontType"] = next(iter(moving))
    elif not moving and any(item["type"] == "stationary" for item in cleaned):
        properties["frontType"] = "stationary"
    properties["segmentTypes"] = cleaned
    if properties.get("frontType") != original_anchor:
        properties["typeAdjustedAfterTopology"] = True
        certainties = [
            float(item.get("certainty", 0.0)) for item in cleaned
            if isinstance(item.get("certainty"), (int, float))
        ]
        if certainties:
            certainty = round(float(np.median(certainties)), 2)
            properties["typeConfidence"] = certainty
            if properties.get("classificationConfidence") is not None:
                properties["classificationConfidence"] = certainty
    return properties


def _centroid_distance_km(first: np.ndarray, second: np.ndarray) -> float:
    mean_lat = ftk.math_mean_lat(first, second)
    a = ftk._project_km(np.asarray(first, dtype=float), mean_lat).mean(axis=0)
    b = ftk._project_km(np.asarray(second, dtype=float), mean_lat).mean(axis=0)
    return float(np.hypot(*(b - a)))


def published_transition_is_coherent(
    first: np.ndarray,
    second: np.ndarray,
    gap_hours: int,
    *,
    max_centroid_speed_kmh: float = 110.0,
) -> bool:
    """Whether two final map geometries can be the same visible branch."""
    gap = max(int(gap_hours), 1)
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    overlap = ftk._shorter_line_overlap_fraction(
        first, second, 55.0 + 20.0 * max(gap - 1, 0)
    )
    if ftk._orientation_diff(first, second) > 80.0 and overlap < 0.60:
        return False
    speed = _centroid_distance_km(first, second) / gap
    return speed <= max_centroid_speed_kmh or overlap >= 0.45


def _retention_score(coordinates: np.ndarray, properties: dict) -> float:
    """Preference among competing deconflicted branches of one track."""
    original = max(float(properties.get("originalExtentKm") or 0.0), 1.0)
    retained = float(
        properties.get("publishedBranchKm")
        or ftk._length_km(np.asarray(coordinates, dtype=float))
    )
    ratio = min(retained / original, 1.0)
    quality = float(properties.get("qualityScore") or 0.0)
    direct = 0.0 if properties.get("interpolated") else 0.08
    return 0.55 * ratio + 0.37 * quality + direct


def stabilize_deconflicted_branches(
    by_hour: dict[int, list[tuple[np.ndarray, dict]]],
    *,
    max_centroid_speed_kmh: float = 110.0,
) -> tuple[int, int]:
    """Remove only trimmed fragments that create a visible branch teleport.

    Shared-trunk ownership is decided independently for each hour.  Without
    this pass, the longest surviving fragment of the weaker front can be its
    western branch at hour *t* and its eastern branch at *t+1*, producing a
    several-hundred-kilometre jump even though the underlying tracked front
    was continuous.  Conflicts are resolved by dropping the less complete
    deconflicted fragment; untouched physical detections are never deleted.

    Returns ``(suppressed_fragments, unresolved_transitions)``.
    """
    records_by_track: dict[int, list[tuple[int, int, np.ndarray, dict]]] = {}
    for hour, entries in by_hour.items():
        for index, (coordinates, properties) in enumerate(entries):
            track_id = properties.get("trackId")
            if isinstance(track_id, (int, np.integer)):
                records_by_track.setdefault(int(track_id), []).append(
                    (int(hour), index, np.asarray(coordinates, dtype=float), properties)
                )

    removed: set[tuple[int, int]] = set()
    unresolved = 0
    for records in records_by_track.values():
        records.sort(key=lambda item: item[0])
        while True:
            active = [record for record in records if (record[0], record[1]) not in removed]
            conflict = None
            for first, second in zip(active[:-1], active[1:]):
                if published_transition_is_coherent(
                    first[2], second[2], second[0] - first[0],
                    max_centroid_speed_kmh=max_centroid_speed_kmh,
                ):
                    continue
                conflict = (first, second)
                break
            if conflict is None:
                break
            first, second = conflict
            droppable = [
                record for record in (first, second)
                if bool(record[3].get("topologyDeconflicted"))
            ]
            if not droppable:
                unresolved += 1
                # There is no safe post-processing action: both geometries
                # are direct detections.  Stop this track instead of looping.
                break
            victim = min(
                droppable,
                key=lambda record: _retention_score(record[2], record[3]),
            )
            removed.add((victim[0], victim[1]))

    for hour, entries in list(by_hour.items()):
        by_hour[hour] = [
            entry for index, entry in enumerate(entries)
            if (int(hour), index) not in removed
        ]
    return len(removed), unresolved


def repair_published_identities(
    by_hour: dict[int, list[tuple[np.ndarray, dict]]],
    *,
    max_centroid_speed_kmh: float = 110.0,
) -> tuple[int, int]:
    """Remove invalid interpolation and split identities at discontinuities.

    A tracker may occasionally associate two independently valid fronts that
    occupy distant parts of the domain. Drawing an interpolated line between
    those detections invents a boundary that the model never resolved;
    keeping the same public ``trackId`` then claims an impossible motion. The
    conservative response is to keep both direct detections, remove the
    synthetic bridge, and begin a new public identity.

    ``trackingSourceId`` preserves the internal association for audit. The
    first feature of a split identity has no published motion, because motion
    cannot be estimated across a declared discontinuity.

    Returns ``(removed_interpolations, identity_splits)``.
    """
    records_by_track: dict[int, list[tuple[int, int, np.ndarray, dict]]] = {}
    maximum_track_id = 0
    for hour, entries in by_hour.items():
        for index, (coordinates, properties) in enumerate(entries):
            track_id = properties.get("trackId")
            if not isinstance(track_id, (int, np.integer)):
                continue
            source_id = int(track_id)
            maximum_track_id = max(maximum_track_id, source_id)
            records_by_track.setdefault(source_id, []).append(
                (int(hour), index, np.asarray(coordinates, dtype=float), properties)
            )

    removed: set[tuple[int, int]] = set()
    for records in records_by_track.values():
        records.sort(key=lambda item: item[0])
        for position, record in enumerate(records):
            if not bool(record[3].get("interpolated")):
                continue
            neighbours = []
            if position > 0:
                neighbours.append(records[position - 1])
            if position + 1 < len(records):
                neighbours.append(records[position + 1])
            if any(
                not published_transition_is_coherent(
                    neighbour[2], record[2],
                    abs(record[0] - neighbour[0]),
                    max_centroid_speed_kmh=max_centroid_speed_kmh,
                )
                for neighbour in neighbours
            ):
                removed.add((record[0], record[1]))

    for hour, entries in list(by_hour.items()):
        by_hour[hour] = [
            entry for index, entry in enumerate(entries)
            if (int(hour), index) not in removed
        ]

    survivors_by_track: dict[int, list[tuple[int, np.ndarray, dict]]] = {}
    for hour, entries in by_hour.items():
        for coordinates, properties in entries:
            track_id = properties.get("trackId")
            if isinstance(track_id, (int, np.integer)):
                survivors_by_track.setdefault(int(track_id), []).append(
                    (int(hour), np.asarray(coordinates, dtype=float), properties)
                )

    next_track_id = maximum_track_id + 1
    split_count = 0
    for source_id, records in survivors_by_track.items():
        records.sort(key=lambda item: item[0])
        published_id = source_id
        previous = None
        for record in records:
            if previous is not None and not published_transition_is_coherent(
                previous[1], record[1], record[0] - previous[0],
                max_centroid_speed_kmh=max_centroid_speed_kmh,
            ):
                published_id = next_track_id
                next_track_id += 1
                split_count += 1
                properties = record[2]
                properties["publishedIdentityStart"] = True
                properties["temporalIdentitySplit"] = True
                properties["motionKmh"] = None
                properties["geoMotionKmh"] = None
                properties["motionBearingDeg"] = None
            properties = record[2]
            if published_id != source_id:
                properties["trackingSourceId"] = source_id
                properties["trackId"] = published_id
            previous = record

    return len(removed), split_count


def published_motion_statistics(
    by_hour: dict[int, list[tuple[np.ndarray, dict]]]
) -> dict:
    """Post-processing QC on the exact polylines that the viewer will draw."""
    records_by_track: dict[int, list[tuple[int, np.ndarray]]] = {}
    for hour, entries in by_hour.items():
        for coordinates, properties in entries:
            track_id = properties.get("trackId")
            if isinstance(track_id, (int, np.integer)):
                records_by_track.setdefault(int(track_id), []).append(
                    (int(hour), np.asarray(coordinates, dtype=float))
                )
    speeds = []
    implausible = 0
    for records in records_by_track.values():
        records.sort(key=lambda item: item[0])
        for first, second in zip(records[:-1], records[1:]):
            gap = max(second[0] - first[0], 1)
            speed = _centroid_distance_km(first[1], second[1]) / gap
            speeds.append(speed)
            if not published_transition_is_coherent(first[1], second[1], gap):
                implausible += 1
    return {
        "transitions": len(speeds),
        "maximumCentroidMotionKmh": round(max(speeds), 1) if speeds else None,
        "p95CentroidMotionKmh": (
            round(float(np.percentile(speeds, 95)), 1) if speeds else None
        ),
        "implausibleTransitions": implausible,
    }
