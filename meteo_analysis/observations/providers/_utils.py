"""Small, dependency-free helpers shared by the REST-based providers.

Extracted once both ItaliaMeteo and MeteoNetwork adapters needed the exact
same defensive field-extraction logic, so a future fix only needs to happen
in one place.
"""

from __future__ import annotations

from typing import Any


def first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def as_float(mapping: dict[str, Any], *keys: str) -> float | None:
    value = first(mapping, *keys)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # reject NaN


def as_list(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "items", "results", "stations", "observations"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []
