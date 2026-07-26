"""Deterministic Italian natural-language weather bulletins.

The templates only describe signals that are present in the model fields.
They never infer a warm/cold front from the national mean temperature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

NLG_METHOD = "icon2i-conditional-nlg-v2"

FRONT_NAMES = {
    "cold": "freddo",
    "warm": "caldo",
    "stationary": "stazionario",
    "occluded": "occluso",
}


def _number(value, digits=1):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return round(result, digits) if np.isfinite(result) else None


def _field_stats(values, mask=None) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    valid = np.isfinite(array)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    sample = array[valid]
    if not sample.size:
        return {
            "minimum": None,
            "p10": None,
            "median": None,
            "mean": None,
            "p90": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "minimum": _number(np.min(sample)),
        "p10": _number(np.percentile(sample, 10)),
        "median": _number(np.median(sample)),
        "mean": _number(np.mean(sample)),
        "p90": _number(np.percentile(sample, 90)),
        "p95": _number(np.percentile(sample, 95)),
        "maximum": _number(np.max(sample)),
    }


def _coverage(values, threshold, mask=None) -> float | None:
    array = np.asarray(values, dtype=float)
    valid = np.isfinite(array)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not valid.any():
        return None
    return _number(np.mean(array[valid] >= threshold) * 100.0, 2)


def _front_types(fronts: dict | None) -> tuple[str, ...]:
    if not isinstance(fronts, dict):
        return ()
    detected = []
    for feature in fronts.get("features", []):
        value = (feature.get("properties") or {}).get("frontType")
        if value in FRONT_NAMES and value not in detected:
            detected.append(value)
    return tuple(detected)


@dataclass(frozen=True)
class BulletinInputs:
    area: str = "Italia"
    valid_time: str | None = None
    front_types: tuple[str, ...] = ()
    front_count: int = 0
    convection: dict[str, Any] = field(default_factory=dict)
    temperature: dict[str, Any] = field(default_factory=dict)
    precipitation: dict[str, Any] = field(default_factory=dict)
    cloud: dict[str, Any] = field(default_factory=dict)
    pressure: dict[str, Any] = field(default_factory=dict)
    wind: dict[str, Any] = field(default_factory=dict)
    hail: dict[str, Any] = field(default_factory=dict)
    temperature_trend_c: float | None = None
    pressure_trend_hpa: float | None = None
    trend_hours: int | None = None
    confidence: str = "media"
    unavailable: tuple[str, ...] = ()


def build_bulletin_inputs(
    *,
    valid_time: str | None,
    fronts: dict | None,
    convection_probability,
    temperature,
    precipitation,
    cloud,
    pressure,
    u_wind,
    v_wind,
    hail_threat=None,
    mask=None,
    previous_temperature_mean: float | None = None,
    previous_pressure_mean: float | None = None,
    trend_hours: int | None = None,
    area: str = "Italia",
) -> BulletinInputs:
    """Reduce gridded fields to robust, auditable bulletin evidence."""
    temperature_stats = _field_stats(temperature, mask)
    precipitation_stats = _field_stats(precipitation, mask)
    cloud_stats = _field_stats(cloud, mask)
    pressure_stats = _field_stats(pressure, mask)
    speed_kmh = np.hypot(np.asarray(u_wind, dtype=float), np.asarray(v_wind, dtype=float)) * 3.6
    wind_stats = _field_stats(speed_kmh, mask)
    convection_stats = _field_stats(convection_probability, mask)
    convection_stats.update(
        {
            "status": (
                "available"
                if convection_stats["maximum"] is not None
                else "unavailable"
            ),
            "areaAbove40Pct": _coverage(convection_probability, 40.0, mask),
            "areaAbove70Pct": _coverage(convection_probability, 70.0, mask),
        }
    )
    precipitation_stats["areaAbove01Pct"] = _coverage(precipitation, 0.1, mask)
    precipitation_stats["areaAbove10Pct"] = _coverage(precipitation, 1.0, mask)

    if hail_threat is None:
        hail_stats = {"status": "unavailable", "areaHighPct": None}
    else:
        stats = _field_stats(hail_threat, mask)
        hail_stats = {
            **stats,
            "status": "available" if stats["maximum"] is not None else "unavailable",
            "areaHighPct": _coverage(hail_threat, 2.0, mask),
        }

    current_temp_mean = temperature_stats["mean"]
    current_pressure_mean = pressure_stats["mean"]
    temp_trend = (
        current_temp_mean - previous_temperature_mean
        if current_temp_mean is not None and previous_temperature_mean is not None
        else None
    )
    pressure_trend = (
        current_pressure_mean - previous_pressure_mean
        if current_pressure_mean is not None and previous_pressure_mean is not None
        else None
    )

    unavailable = []
    if convection_stats["status"] != "available":
        unavailable.append("probabilità temporali")
    for name, stats in (
        ("temperatura", temperature_stats),
        ("precipitazioni", precipitation_stats),
        ("pressione", pressure_stats),
        ("vento", wind_stats),
    ):
        if stats["maximum"] is None:
            unavailable.append(name)

    return BulletinInputs(
        area=area,
        valid_time=valid_time,
        front_types=_front_types(fronts),
        front_count=len((fronts or {}).get("features", [])),
        convection=convection_stats,
        temperature=temperature_stats,
        precipitation=precipitation_stats,
        cloud=cloud_stats,
        pressure=pressure_stats,
        wind=wind_stats,
        hail=hail_stats,
        temperature_trend_c=_number(temp_trend),
        pressure_trend_hpa=_number(pressure_trend),
        trend_hours=trend_hours,
        confidence="alta" if not unavailable else "media",
        unavailable=tuple(unavailable),
    )


def _front_sentence(inputs: BulletinInputs) -> str | None:
    if not inputs.front_types:
        return None
    names = [FRONT_NAMES[value] for value in inputs.front_types]
    if len(names) == 1:
        subject = f"un fronte {names[0]}"
    else:
        subject = "più strutture frontali (" + ", ".join(names) + ")"
    return (
        f"L'analisi oggettiva individua {subject} su {inputs.area}. "
        "La posizione resta una stima del modello e può differire localmente."
    )


def _convection_sentence(inputs: BulletinInputs) -> str:
    convection = inputs.convection
    if convection.get("status") != "available":
        return (
            "La probabilità di temporali non è disponibile per questa scadenza: "
            "il sistema non sostituisce i campi mancanti con valori simulati."
        )
    p95 = convection.get("p95") or 0.0
    maximum = convection.get("maximum") or 0.0
    area70 = convection.get("areaAbove70Pct") or 0.0
    area40 = convection.get("areaAbove40Pct") or 0.0
    if p95 >= 70.0 or area70 >= 1.0:
        return (
            "Innesco temporalesco probabile nelle aree più favorevoli, "
            f"con picchi del {maximum:.0f}% e segnali alti sul {area70:.1f}% "
            "dell'area analizzata."
        )
    if p95 >= 40.0 or area40 >= 2.0:
        return (
            "Possibili temporali locali: "
            f"i valori raggiungono il {maximum:.0f}%, ma il segnale medio resta "
            "confinato e non indica fenomeni diffusi."
        )
    return (
        "Segnale temporalesco generalmente basso; eventuali rovesci isolati "
        "restano possibili dove convergenze locali non sono risolte dal modello."
    )


def _precipitation_sentence(inputs: BulletinInputs) -> str | None:
    rain = inputs.precipitation
    coverage = rain.get("areaAbove01Pct")
    strong_coverage = rain.get("areaAbove10Pct")
    if coverage is None:
        return None
    if coverage >= 25.0:
        return (
            "Precipitazioni abbastanza estese nell'ora considerata"
            + (
                f", più organizzate sul {strong_coverage:.1f}% dell'area."
                if strong_coverage
                else "."
            )
        )
    if coverage >= 2.0:
        return (
            "Precipitazioni sparse o locali, senza un segnale uniforme sul "
            "territorio."
        )
    return "Precipitazioni assenti o poco significative sulla maggior parte dell'area."


def _temperature_sentence(inputs: BulletinInputs) -> str | None:
    temperature = inputs.temperature
    low = temperature.get("p10")
    high = temperature.get("p90")
    if low is None or high is None:
        return None
    sentence = f"Temperature prevalentemente comprese tra {low:.0f} e {high:.0f} °C."
    trend = inputs.temperature_trend_c
    hours = inputs.trend_hours
    if trend is not None and hours and abs(trend) >= 1.0:
        direction = "aumento" if trend > 0 else "calo"
        sentence += (
            f" Tendenza media in {direction} di circa {abs(trend):.1f} °C "
            f"rispetto a {hours} ore prima."
        )
    return sentence


def _wind_pressure_sentence(inputs: BulletinInputs) -> str | None:
    pieces = []
    wind_p95 = inputs.wind.get("p95")
    if wind_p95 is not None:
        if wind_p95 >= 70.0:
            pieces.append(f"venti forti, localmente oltre {wind_p95:.0f} km/h")
        elif wind_p95 >= 40.0:
            pieces.append(f"venti moderati o tesi, fino a circa {wind_p95:.0f} km/h")
        else:
            pieces.append("venti in prevalenza deboli o moderati")
    trend = inputs.pressure_trend_hpa
    hours = inputs.trend_hours
    if trend is not None and hours and abs(trend) >= 1.0:
        direction = "aumento" if trend > 0 else "diminuzione"
        pieces.append(
            f"pressione media in {direction} di {abs(trend):.1f} hPa in {hours} ore"
        )
    return "; ".join(pieces).capitalize() + "." if pieces else None


def _hail_sentence(inputs: BulletinInputs) -> str | None:
    if inputs.hail.get("status") != "available":
        return None
    high_area = inputs.hail.get("areaHighPct") or 0.0
    if high_area >= 0.5:
        return (
            "Nelle celle temporalesche più intense è presente anche un segnale "
            "favorevole alla grandine; non equivale a una previsione puntuale."
        )
    return None


def generate_bulletin_details(inputs: BulletinInputs) -> dict:
    paragraphs = [
        sentence
        for sentence in (
            _front_sentence(inputs),
            _convection_sentence(inputs),
            _precipitation_sentence(inputs),
            _temperature_sentence(inputs),
            _wind_pressure_sentence(inputs),
            _hail_sentence(inputs),
        )
        if sentence
    ]
    headline = paragraphs[0] if paragraphs else "Dati insufficienti per il bollettino."
    text = " ".join(paragraphs)
    return {
        "schemaVersion": 2,
        "method": NLG_METHOD,
        "title": f"Bollettino automatico · {inputs.area}",
        "validTime": inputs.valid_time,
        "headline": headline,
        "paragraphs": paragraphs,
        "text": text,
        "confidence": inputs.confidence,
        "unavailableInputs": list(inputs.unavailable),
        "disclaimer": (
            "Sintesi automatica del modello ICON-2I: non è un'allerta né un "
            "bollettino ufficiale."
        ),
    }


def generate_bulletin(
    inputs: BulletinInputs | None = None,
    front_type: str | None = None,
    prob_thunderstorm: str | None = None,
    hail_threat: str | None = None,
    t_trend: str | None = None,
) -> str:
    """Return bulletin text.

    The legacy scalar arguments are retained only for API compatibility.  New
    code should pass :class:`BulletinInputs`, which is evidence-driven.
    """
    if inputs is not None:
        return generate_bulletin_details(inputs)["text"]

    # Conservative legacy fallback: do not claim a front unless explicitly
    # supplied by an upstream detector.
    parts = []
    if front_type in {"freddo", "caldo", "stazionario", "occluso"}:
        parts.append(f"L'analisi indica un fronte {front_type} sull'area.")
    if prob_thunderstorm == "alta":
        parts.append("La probabilità di temporali è elevata nelle zone più favorevoli.")
    elif prob_thunderstorm == "media":
        parts.append("Sono possibili temporali locali.")
    else:
        parts.append("Il segnale temporalesco è basso o non disponibile.")
    if hail_threat == "alto":
        parts.append("È presente un segnale favorevole alla grandine.")
    if t_trend in {"calo", "aumento"}:
        parts.append(f"Temperature mediamente in {t_trend}.")
    return " ".join(parts)
