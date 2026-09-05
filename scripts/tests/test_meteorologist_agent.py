#!/usr/bin/env python3
"""Regression tests for the evidence-grounded dual-LLM supervisor."""

from __future__ import annotations

import copy
import gzip
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from urllib.error import HTTPError


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from meteo_analysis.agents.meteorologist import (  # noqa: E402
    AgentError,
    SECTION_IDS,
    SECTION_TITLES,
    _post_json,
    build_evidence_packet,
    generate_verified_bulletin,
    validate_primary_analysis,
)
from scripts.generate_ai_bulletin import run as run_agent_script  # noqa: E402


def deterministic_bulletin():
    analyses = []
    for hour in range(73):
        analyses.append({
            "leadHours": hour,
            "validTime": f"2026-09-{1 + hour // 24:02d}T{hour % 24:02d}:00:00Z",
            "signalConfidence": "scenario plausibile",
            "operationalSummary": [
                f"Alla scadenza +{hour} h il minimo MSLP e il flusso in quota sono valutati congiuntamente."
            ],
            "sections": [
                {
                    "id": "synoptic", "title": "Analisi sinottica",
                    "paragraphs": [f"A +{hour} h la pressione minima è 1004 hPa con gradiente barico organizzato."],
                },
                {
                    "id": "upper", "title": "Dinamica in quota",
                    "paragraphs": [f"A +{hour} h PVA e omega 500 hPa indicano un forcing concorde."],
                },
                {
                    "id": "low", "title": "Bassa troposfera",
                    "paragraphs": [f"A +{hour} h theta-e 850 hPa e convergenza sostengono la struttura."],
                },
                {
                    "id": "convection", "title": "Stabilità e convezione",
                    "paragraphs": [f"A +{hour} h instabilità, umidità e innesco risultano valutati insieme."],
                },
                {
                    "id": "precipitation", "title": "Precipitazioni",
                    "paragraphs": [f"A +{hour} h precipitazione e persistenza sono trattate separatamente."],
                },
                {
                    "id": "uncertainty", "title": "Incertezze",
                    "paragraphs": [f"A +{hour} h resta il limite del singolo run deterministico."],
                },
            ],
            "why": [],
            "unavailableInputs": ["DCAPE", "SRH 0–3 km"],
        })
    return {
        "schemaVersion": 1,
        "method": "icon2i-multifield-synoptic-engine-v1",
        "model": "ICON-2I",
        "runTime": "2026-09-01T00:00:00Z",
        "area": "Italia e dominio ICON-2I",
        "spatialResolutionKm": 2.2,
        "temporalResolutionHours": 1,
        "forecastHorizonHours": 72,
        "semantics": {
            "source": "single-deterministic-nwp-run",
            "stormScore": "diagnostic-not-calibrated-probability",
        },
        "analyses": analyses,
    }


def primary_for(packet):
    evidence = packet["evidence"]

    def refs_for(period=None):
        candidates = evidence
        if period is not None:
            candidates = [
                item for item in evidence
                if period["fromHour"] <= item["leadHours"] <= period["toHour"]
            ]
        first = next(item for item in candidates if item["section"] == "synoptic")
        second = next(item for item in candidates if item["section"] == "upper")
        return [first["id"], second["id"]]

    counter = 0

    def claim(period=None):
        nonlocal counter
        counter += 1
        return {
            "id": f"C{counter:03d}",
            "text": "Il quadro modellistico è descritto da più famiglie diagnostiche concordi.",
            "confidence": "scenario plausibile",
            "evidenceIds": refs_for(period),
        }

    return {
        "language": "it",
        "overview": {
            "headline": "Evoluzione sinottica del run",
            "claims": [claim(), claim(), claim()],
        },
        "periods": [
            {
                **period,
                "headline": "Quadro meteorologico del periodo",
                "sections": [
                    {
                        "id": section_id,
                        "title": SECTION_TITLES[section_id],
                        "claims": [claim(period)],
                    }
                    for section_id in SECTION_IDS
                ],
            }
            for period in packet["periods"]
        ],
    }


def fake_post_factory(packet, primary, *, approve=True):
    calls = []

    def fake_post(url, *, headers, payload, attempts=3):
        calls.append((url, headers, payload))
        if "googleapis.com" in url:
            assert headers["x-goog-api-key"] == "gemini-secret"
            request_packet = json.loads(payload["contents"][0]["parts"][0]["text"])
            if request_packet["scope"] == "overview":
                generated = {
                    "language": primary["language"],
                    "overview": primary["overview"],
                }
            else:
                period_ids = {
                    item["periodId"] for item in request_packet["requestedPeriods"]
                }
                generated = {
                    "language": primary["language"],
                    "periods": [
                        item for item in primary["periods"]
                        if item["periodId"] in period_ids
                    ],
                }
            return {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {"parts": [
                        {"thought": True, "text": "ragionamento non pubblicabile"},
                        {"text": json.dumps(generated)},
                    ]},
                }],
                "usageMetadata": {"promptTokenCount": 1200, "candidatesTokenCount": 500},
            }
        review_input = json.loads(payload["messages"][1]["content"])
        claim_ids = [item["id"] for item in review_input["claims"]]
        return {
            "choices": [{"message": {"content": json.dumps({
                "approved": approve,
                "rejectedClaimIds": [] if approve else [claim_ids[0]],
                "downgradeClaimIds": (
                    [claim_ids[-1]]
                    if approve and review_input["scope"] == "overview" else []
                ),
                "issues": [],
            })}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 80},
        }

    return fake_post, calls


def test_packet_and_verified_product():
    source = deterministic_bulletin()
    packet = build_evidence_packet(source)
    assert packet["periods"][-1] == {
        "periodId": "P061-072", "fromHour": 61, "toHour": 72,
    }
    assert {item["leadHours"] for item in packet["evidence"]} == set(range(0, 73, 6))
    primary = primary_for(packet)
    validation = validate_primary_analysis(primary, packet)
    assert validation["claimCount"] == 3 + 6 * len(packet["periods"])
    generated_primary = copy.deepcopy(primary)
    generated_primary["overview"]["headline"] = "ICON-2I: evoluzione nelle prossime 72 ore"
    generated_primary["periods"][0]["headline"] = "Scenario tra 0 e 12 ore"
    fake_post, calls = fake_post_factory(packet, generated_primary)
    product = generate_verified_bulletin(
        source,
        gemini_api_key="gemini-secret",
        groq_api_key="groq-secret",
        post_json=fake_post,
    )
    assert product["status"] == "validated"
    assert product["source"]["fieldsModifiedByLlm"] is False
    assert product["providers"]["reviewer"]["downgradedClaimCount"] == 1
    assert product["overview"]["headline"] == "ICON: evoluzione nelle prossime ore"
    assert product["periods"][0]["headline"] == "Scenario nel periodo"
    gemini_calls = [item for item in calls if "googleapis.com" in item[0]]
    groq_calls = [item for item in calls if "groq.com" in item[0]]
    assert len(gemini_calls) == 1 + (len(packet["periods"]) + 1) // 2
    assert len(groq_calls) == 1 + len(packet["periods"])
    assert product["providers"]["primary"]["requestCount"] == len(gemini_calls)
    encoded = json.dumps(product, ensure_ascii=False, allow_nan=False)
    assert "gemini-secret" not in encoded and "groq-secret" not in encoded


def test_hallucinated_reference_and_number_are_rejected():
    packet = build_evidence_packet(deterministic_bulletin())
    primary = primary_for(packet)
    broken = copy.deepcopy(primary)
    broken["overview"]["claims"][0]["evidenceIds"][0] = "INESISTENTE"
    try:
        validate_primary_analysis(broken, packet)
    except AgentError as error:
        assert "inesistente" in str(error)
    else:
        raise AssertionError("riferimento inventato accettato")

    broken = copy.deepcopy(primary)
    broken["overview"]["claims"][0]["text"] = (
        "Il quadro indica esattamente 9876 hPa, valore non presente nelle prove."
    )
    try:
        validate_primary_analysis(broken, packet)
    except AgentError as error:
        assert "numeri non documentati" in str(error)
    else:
        raise AssertionError("numero inventato accettato")


def test_reviewer_rejection_fails_closed():
    source = deterministic_bulletin()
    packet = build_evidence_packet(source)
    primary = primary_for(packet)
    fake_post, _ = fake_post_factory(packet, primary, approve=False)
    try:
        generate_verified_bulletin(
            source,
            gemini_api_key="gemini-secret",
            groq_api_key="groq-secret",
            post_json=fake_post,
        )
    except AgentError as error:
        assert "non approvata" in str(error)
    else:
        raise AssertionError("prodotto respinto pubblicato")


def test_provider_http_failure_is_identifiable_without_error_body_leak():
    url = "https://generativelanguage.googleapis.com/v1beta/models/test:generateContent"
    response = BytesIO(json.dumps({
        "error": {
            "code": 429,
            "message": "quota detail that must not be copied",
            "status": "RESOURCE_EXHAUSTED",
        }
    }).encode("utf-8"))
    error = HTTPError(
        url,
        429,
        "quota",
        {"retry-after": "not-a-number"},
        response,
    )
    with patch("meteo_analysis.agents.meteorologist.urlopen", side_effect=error):
        try:
            _post_json(url, headers={}, payload={}, attempts=1)
        except AgentError as caught:
            message = str(caught)
            assert "Gemini API" in message
            assert "HTTP 429" in message
            assert "RESOURCE_EXHAUSTED" in message
            assert "quota detail" not in message
        else:
            raise AssertionError("errore HTTP del provider non propagato")


def test_missing_secrets_preserve_deterministic_fallback():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        with gzip.open(directory / "expert_bulletin.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(deterministic_bulletin(), handle)
        (directory / "catalog.json").write_text(
            json.dumps([{"hour": hour, "leadHours": hour} for hour in range(73)]),
            encoding="utf-8",
        )
        (directory / "archive_manifest.json").write_text(json.dumps({
            "model": {"runTime": "2026-09-01T00:00:00Z"},
            "algorithms": {"expertBulletin": "icon2i-multifield-synoptic-engine-v1"},
            "sourceAssets": [],
            "domain": {"south": 33.7, "north": 48.9, "west": 3.0, "east": 22.0},
            "objectStorage": {"schemaVersion": 1, "mode": "off"},
        }), encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            status = run_agent_script(directory)
        assert status["status"] == "fallback"
        assert (directory / "expert_bulletin.json.gz").exists()
        assert not (directory / "ai_expert_bulletin.json.gz").exists()
        manifest = json.loads(
            (directory / "archive_manifest.json").read_text(encoding="utf-8")
        )
        paths = {item["path"] for item in manifest["publishedAssets"]}
        assert "ai_agent_status.json" in paths
        assert manifest["algorithms"]["aiMeteorologist"] is None


if __name__ == "__main__":
    test_packet_and_verified_product()
    test_hallucinated_reference_and_number_are_rejected()
    test_reviewer_rejection_fails_closed()
    test_provider_http_failure_is_identifiable_without_error_body_leak()
    test_missing_secrets_preserve_deterministic_fallback()
    print("Meteorological agent tests passed")
