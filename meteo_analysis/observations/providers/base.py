"""Base contract every observation provider must satisfy.

Adding or removing a network (requirement: "in modo che sia semplice
aggiungere o rimuovere fonti") should only require writing a new subclass of
:class:`ObservationProvider` and registering it in
:mod:`meteo_analysis.observations.pipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ProviderResult:
    """Uniform result returned by every provider's ``fetch()``.

    ``stations`` uses the canonical schema from
    :mod:`meteo_analysis.observations.model`.  ``ok`` is False when the
    provider could not be reached/parsed; ``error`` then carries a short,
    human-readable reason.  A failed provider must never raise past this
    boundary: the pipeline keeps working with the remaining sources.
    """

    source: str
    ok: bool
    stations: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    fetched_at: str | None = None
    configured: bool = True

    def __post_init__(self) -> None:
        if self.fetched_at is None:
            self.fetched_at = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )


class ObservationProvider:
    """Common interface for a surface-observation source."""

    #: Canonical identifier, e.g. ``"metar"``, ``"italiameteo"``.
    source: str = "unknown"

    def is_configured(self) -> bool:
        """Return whether credentials/environment allow calling this source.

        Providers that require registration (e.g. MeteoHub) must return
        ``False`` rather than raising, so the pipeline can report a clear
        "not configured" status instead of a spurious failure.
        """

        return True

    def fetch(self, *, session=None, timeout: tuple[int, int] = (15, 45)) -> ProviderResult:
        raise NotImplementedError

    def safe_fetch(self, *, session=None, timeout: tuple[int, int] = (15, 45)) -> ProviderResult:
        """Call :meth:`fetch` and never let an exception cross this boundary."""

        if not self.is_configured():
            return ProviderResult(
                source=self.source,
                ok=False,
                configured=False,
                error="provider non configurato (variabili d'ambiente mancanti)",
            )
        try:
            return self.fetch(session=session, timeout=timeout)
        except Exception as error:  # noqa: BLE001 - provider isolation is intentional
            return ProviderResult(source=self.source, ok=False, error=str(error))
