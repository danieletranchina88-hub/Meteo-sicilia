"""Multi-provider surface observation network.

This package extends the original single-source (METAR/AWC) verification
feed into a pluggable ``ObservationProvider`` architecture able to merge
several Italian observation networks (METAR, Agenzia ItaliaMeteo/MeteoHub,
MeteoNetwork, regional services, ...) into one canonical, deduplicated
station registry with quality control and health classification.

Every provider is independent: a failure in one network must never take
down the others, and must never be silently rendered as "no data" on the
public map (see :mod:`meteo_analysis.observations.pipeline`).
"""

from __future__ import annotations

SCHEMA_VERSION = 1

__all__ = ["SCHEMA_VERSION"]
