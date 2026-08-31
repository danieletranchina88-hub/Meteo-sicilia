"""Quality control applied to every observation before it reaches the
verification archive or the AI observation layer.

Only two of the four QC families from the specification are implemented as
pure functions here without any external state: physical bounds and a
best-effort temporal consistency check driven by an optional short rolling
history.  Spatial buddy-check and full cross-provider correlation need a
larger observation window than a single collection run provides and are left
as a documented next step (see ``docs/rete_stazioni.md``); the registry
already carries everything (coordinates, elevation, duplicate candidates)
that a future buddy-check pass would need.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from meteo_analysis.observations.model import CANONICAL_VARIABLES, provider_defaults

FLAG_OK = "ok"
FLAG_OUT_OF_BOUNDS = "out_of_bounds"
FLAG_STUCK = "stuck_sensor"
FLAG_FUTURE_TIMESTAMP = "future_timestamp"
FLAG_TOO_OLD = "too_old"
FLAG_UNKNOWN_VARIABLE = "unknown_variable"

# A sensor reporting the *exact* same value for this many consecutive
# observations is flagged as possibly stuck.  Genuine calm/dry spells can
# legitimately repeat a value a few times, so the bar is deliberately high.
STUCK_SENSOR_MIN_REPEATS = 6

# An observation timestamped further in the past than this is considered
# unusable for real-time verification, independent of provider latency.
MAX_OBSERVATION_AGE_HOURS = 24
MAX_FUTURE_SKEW_MINUTES = 10


def physical_bounds_flag(variable: str, value: float) -> str:
    bounds = CANONICAL_VARIABLES.get(variable)
    if bounds is None:
        return FLAG_UNKNOWN_VARIABLE
    if value < bounds["min"] or value > bounds["max"]:
        return FLAG_OUT_OF_BOUNDS
    return FLAG_OK


def temporal_consistency_flags(
    variable: str,
    value: float,
    observed_at_epoch: float | None,
    *,
    recent_values: list[float] | None = None,
    now: datetime | None = None,
) -> list[str]:
    flags: list[str] = []
    now = now or datetime.now(timezone.utc)
    if observed_at_epoch is not None:
        observed = datetime.fromtimestamp(float(observed_at_epoch), tz=timezone.utc)
        skew_minutes = (observed - now).total_seconds() / 60.0
        if skew_minutes > MAX_FUTURE_SKEW_MINUTES:
            flags.append(FLAG_FUTURE_TIMESTAMP)
        elif (now - observed).total_seconds() > MAX_OBSERVATION_AGE_HOURS * 3600:
            flags.append(FLAG_TOO_OLD)
    if recent_values and len(recent_values) >= STUCK_SENSOR_MIN_REPEATS:
        window = recent_values[-STUCK_SENSOR_MIN_REPEATS:]
        if all(abs(previous - value) < 1e-9 for previous in window):
            flags.append(FLAG_STUCK)
    return flags


def evaluate_observation(
    variable: str,
    value: float,
    *,
    source: str,
    observed_at_epoch: float | None = None,
    recent_values: list[float] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return QC flags and a ``quality_score`` in ``[0, 1]`` for one value."""

    flags = [physical_bounds_flag(variable, value)]
    flags = [flag for flag in flags if flag != FLAG_OK]
    flags.extend(
        temporal_consistency_flags(
            variable, value, observed_at_epoch,
            recent_values=recent_values, now=now,
        )
    )

    base_weight = provider_defaults(source)["baseQualityWeight"]
    score = base_weight
    if FLAG_OUT_OF_BOUNDS in flags:
        score = 0.0
    else:
        if FLAG_STUCK in flags:
            score *= 0.5
        if FLAG_TOO_OLD in flags:
            score *= 0.3
        if FLAG_FUTURE_TIMESTAMP in flags:
            score = 0.0
    return {
        "qualityFlags": flags or [FLAG_OK],
        "qualityScore": round(score, 3),
    }


def apply_quality_control(
    stations: list[dict[str, Any]],
    *,
    history: dict[str, dict[str, list[float]]] | None = None,
    now: datetime | None = None,
) -> None:
    """Mutate ``stations`` in place, adding QC results to each observation."""

    history = history or {}
    for station in stations:
        station_history = history.get(station.get("internalStationId", ""), {})
        for variable, measurement in list(station.get("observations", {}).items()):
            if not isinstance(measurement, dict) or measurement.get("value") is None:
                continue
            result = evaluate_observation(
                variable,
                float(measurement["value"]),
                source=station["source"],
                observed_at_epoch=measurement.get("observedAt"),
                recent_values=station_history.get(variable),
                now=now,
            )
            measurement.update(result)
