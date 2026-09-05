"""Evidence-grounded meteorological LLM supervisor.

The language models never receive raw GRIB arrays and never alter forecast
fields.  They synthesize a compact evidence catalogue emitted by the audited
ICON-2I synoptic engine.  Every published claim must cite at least two items
from different diagnostic families, pass deterministic schema and numerical
grounding checks, and be approved by an independent reviewer.

Failure is deliberately non-fatal for the weather pipeline: callers keep the
deterministic expert bulletin as the authoritative fallback.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

AGENT_METHOD = "icon2i-evidence-grounded-dual-llm-v2"
PACKET_METHOD = "icon2i-synoptic-evidence-packet-v1"
GEMINI_MODEL = "gemini-3.5-flash"
GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

CONFIDENCE_LEVELS = (
    "segnale molto robusto",
    "segnale abbastanza robusto",
    "scenario plausibile",
    "scenario condizionale",
    "segnale debole",
    "dato insufficiente",
)
SECTION_TITLES = {
    "synoptic": "Analisi sinottica",
    "upper": "Dinamica in quota",
    "low": "Bassa troposfera",
    "convection": "Stabilità e convezione",
    "precipitation": "Precipitazioni",
    "uncertainty": "Incertezze",
}
SECTION_IDS = tuple(SECTION_TITLES)
INDEPENDENT_EVIDENCE_FAMILIES = frozenset({
    "synoptic", "upper", "low", "convection", "precipitation",
    "boundary", "evolution", "uncertainty",
})
CONFIDENCE_DOWNGRADE = {
    "segnale molto robusto": "segnale abbastanza robusto",
    "segnale abbastanza robusto": "scenario plausibile",
    "scenario plausibile": "scenario condizionale",
    "scenario condizionale": "segnale debole",
    "segnale debole": "dato insufficiente",
    "dato insufficiente": "dato insufficiente",
}
NUMBER_RE = re.compile(r"(?<![A-Za-zÀ-ÿ])[-+]?\d+(?:[.,]\d+)?")


class AgentError(RuntimeError):
    """Raised when an LLM product cannot be published safely."""


def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


CLAIM_SCHEMA = _object_schema(
    {
        "id": {"type": "string"},
        "text": {"type": "string"},
        "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
        "evidenceIds": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 5,
        },
    },
    ["id", "text", "confidence", "evidenceIds"],
)

SECTION_SCHEMA = _object_schema(
    {
        "id": {"type": "string", "enum": list(SECTION_IDS)},
        "title": {"type": "string"},
        "claims": {
            "type": "array",
            "items": CLAIM_SCHEMA,
            "minItems": 1,
            "maxItems": 1,
        },
    },
    ["id", "title", "claims"],
)

PERIOD_SCHEMA = _object_schema(
    {
        "periodId": {"type": "string"},
        "fromHour": {"type": "integer", "minimum": 0, "maximum": 240},
        "toHour": {"type": "integer", "minimum": 0, "maximum": 240},
        "headline": {"type": "string"},
        "sections": {
            "type": "array",
            "items": SECTION_SCHEMA,
            "minItems": len(SECTION_IDS),
            "maxItems": len(SECTION_IDS),
        },
    },
    ["periodId", "fromHour", "toHour", "headline", "sections"],
)

PRIMARY_SCHEMA = _object_schema(
    {
        "language": {"type": "string", "enum": ["it"]},
        "overview": _object_schema(
            {
                "headline": {"type": "string"},
                "claims": {
                    "type": "array",
                    "items": CLAIM_SCHEMA,
                    "minItems": 3,
                    "maxItems": 4,
                },
            },
            ["headline", "claims"],
        ),
        "periods": {
            "type": "array",
            "items": PERIOD_SCHEMA,
            "minItems": 1,
            "maxItems": 12,
        },
    },
    ["language", "overview", "periods"],
)

REVIEW_SCHEMA = _object_schema(
    {
        "approved": {"type": "boolean"},
        "rejectedClaimIds": {
            "type": "array", "items": {"type": "string"}, "maxItems": 12,
        },
        "downgradeClaimIds": {
            "type": "array", "items": {"type": "string"}, "maxItems": 20,
        },
        "issues": {
            "type": "array", "items": {"type": "string"}, "maxItems": 8,
        },
    },
    ["approved", "rejectedClaimIds", "downgradeClaimIds", "issues"],
)

PRIMARY_SYSTEM_PROMPT = """Sei un meteorologo sinottico e mesoscalare esperto.
Devi sintetizzare esclusivamente il catalogo di prove deterministiche ICON-2I
fornito dall'utente. Non usare il web, climatologia, memoria esterna o valori
non presenti. Non chiamare osservato un campo prognostico. Non trasformare
score diagnostici in probabilità calibrate o allerte.

Regole inderogabili:
1. Ogni claim cita da 2 a 5 evidenceIds esistenti, provenienti da almeno due
   famiglie diagnostiche differenti. Nessun fenomeno deriva da un solo indice.
2. Copia qualsiasi numero esattamente dalle prove citate; se non serve, usa
   linguaggio qualitativo. Non inventare soglie, località o orari.
3. Riproduci esattamente periodId, fromHour e toHour richiesti e crea una sola
   sezione per ciascuno dei sei id indicati.
4. Se le prove sono contraddittorie o mancanti, scrivilo e abbassa la
   confidence. 'Segnale molto robusto' è ammesso soltanto con più livelli,
   evoluzione temporale coerente e nessuna contraddizione rilevante.
5. CAPE, STP, LPI, UH, pressione o un singolo gradiente non costituiscono mai
   da soli una previsione. Distingui ingrediente, segnale favorevole ed
   evidenza dinamica robusta.
6. I titoli devono essere sobri, tecnici, senza numeri e senza sensazionalismo.
7. Rispondi soltanto con il JSON conforme allo schema richiesto, in italiano.
"""

REVIEW_SYSTEM_PROMPT = """Sei il revisore indipendente di un bollettino NWP.
Controlla esclusivamente claim e prove forniti. Respingi un claim se introduce
dati, fenomeni, causalità o certezza non sostenuti dalle evidenceIds. Richiedi
un downgrade se la conclusione è plausibile ma la confidence è eccessiva.
Non riscrivere il bollettino e non aggiungere conoscenza esterna. Approva solo
se nessun claim deve essere respinto. Restituisci esclusivamente JSON conforme
allo schema."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean_text(value: Any, limit: int = 700) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit].rstrip()


def _periods(maximum_hour: int) -> list[dict[str, int | str]]:
    periods = []
    start = 0
    while start <= maximum_hour:
        end = min(maximum_hour, 12 if start == 0 else start + 11)
        periods.append({
            "periodId": f"P{start:03d}-{end:03d}",
            "fromHour": start,
            "toHour": end,
        })
        start = end + 1
    return periods


def _selected_analyses(analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hour = {
        int(item["leadHours"]): item
        for item in analyses
        if isinstance(item, dict) and "leadHours" in item
    }
    if not by_hour:
        raise AgentError("bollettino deterministico privo di analisi")
    maximum = max(by_hour)
    targets = set(range(0, maximum + 1, 6)) | {maximum}
    return [by_hour[hour] for hour in sorted(targets) if hour in by_hour]


def _append_evidence(
    target: list[dict[str, Any]],
    *,
    hour: int,
    section: str,
    index: int,
    text: Any,
) -> None:
    cleaned = _clean_text(text)
    if not cleaned:
        return
    target.append({
        "id": f"H{hour:03d}:{section.upper()}:{index:02d}",
        "leadHours": hour,
        "section": section,
        "text": cleaned,
    })


def build_evidence_packet(bulletin: dict[str, Any]) -> dict[str, Any]:
    """Reduce the deterministic run bulletin to a compact auditable packet."""
    analyses = _selected_analyses(list(bulletin.get("analyses") or []))
    maximum_hour = max(int(item["leadHours"]) for item in analyses)
    evidence: list[dict[str, Any]] = []
    for analysis in analyses:
        hour = int(analysis["leadHours"])
        counters: dict[str, int] = {}

        def add(section: str, text: Any) -> None:
            index = counters.get(section, 0)
            _append_evidence(
                evidence, hour=hour, section=section, index=index, text=text
            )
            counters[section] = index + 1

        for text in analysis.get("operationalSummary") or []:
            add("operational", text)
        for section in analysis.get("sections") or []:
            section_id = str(section.get("id") or "other")
            for text in (section.get("paragraphs") or [])[:3]:
                add(section_id, text)
            for name, value in sorted((section.get("components") or {}).items()):
                add(section_id, f"{name}: {value}")
            for item in (section.get("items") or [])[:3]:
                add(
                    section_id,
                    f"{item.get('phenomenon', 'Fenomeno')} su "
                    f"{item.get('area', 'area non specificata')}; periodo "
                    f"{item.get('period', 'non specificato')}; confidence "
                    f"{item.get('confidence', 'non specificata')}; "
                    f"motivazione: {item.get('reason', 'non disponibile')}",
                )
        for chain in analysis.get("why") or []:
            for text in chain.get("evidence") or []:
                add("causal", f"{chain.get('phenomenon', 'Processo')}: {text}")
            for text in chain.get("limitingFactors") or []:
                add("uncertainty", f"Limite per {chain.get('phenomenon', 'processo')}: {text}")
            add("causal", chain.get("conclusion"))
        unavailable = analysis.get("unavailableInputs") or []
        if unavailable:
            add("uncertainty", "Diagnostiche non disponibili: " + "; ".join(unavailable))
        add(
            "uncertainty",
            f"Confidence interna deterministica per +{hour} h: "
            f"{analysis.get('signalConfidence', 'dato insufficiente')}",
        )

    if not evidence:
        raise AgentError("catalogo delle prove vuoto")
    return {
        "schemaVersion": 1,
        "method": PACKET_METHOD,
        "model": bulletin.get("model", "ICON-2I"),
        "runTime": bulletin.get("runTime"),
        "area": bulletin.get("area"),
        "spatialResolutionKm": bulletin.get("spatialResolutionKm"),
        "temporalResolutionHours": bulletin.get("temporalResolutionHours"),
        "forecastHorizonHours": bulletin.get("forecastHorizonHours", maximum_hour),
        "sourceMethod": bulletin.get("method"),
        "sourceSemantics": bulletin.get("semantics") or {},
        "requiredSections": [
            {"id": section_id, "title": SECTION_TITLES[section_id]}
            for section_id in SECTION_IDS
        ],
        "periods": _periods(maximum_hour),
        "evidence": evidence,
    }


def _normal_number(token: str) -> str:
    token = token.replace(",", ".")
    try:
        value = float(token)
    except ValueError:
        return token
    return f"{value:.8g}"


def _numbers(text: str) -> set[str]:
    return {_normal_number(item) for item in NUMBER_RE.findall(text)}


def _require_exact_keys(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise AgentError(f"struttura non ammessa in {label}")


def _validate_headline(value: Any, label: str) -> None:
    text = _clean_text(value, 161)
    if not text or len(text) > 160 or text != value:
        raise AgentError(f"titolo {label} assente, eccessivo o non normalizzato")
    if _numbers(text):
        raise AgentError(f"titolo {label} con numeri non verificabili")


def _iter_claims(primary: dict[str, Any]):
    for claim in (primary.get("overview") or {}).get("claims") or []:
        yield claim, None
    for period in primary.get("periods") or []:
        for section in period.get("sections") or []:
            for claim in section.get("claims") or []:
                yield claim, period


def validate_primary_analysis(
    primary: dict[str, Any], packet: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed on structure, provenance, or invented numerical values."""
    if not isinstance(primary, dict) or primary.get("language") != "it":
        raise AgentError("output primario non italiano o non strutturato")
    _require_exact_keys(primary, {"language", "overview", "periods"}, "radice IA")
    evidence = {
        item["id"]: item for item in packet.get("evidence") or []
        if isinstance(item, dict) and item.get("id")
    }
    expected_periods = packet.get("periods") or []
    actual_periods = primary.get("periods") or []
    if len(actual_periods) != len(expected_periods):
        raise AgentError("numero di periodi IA non coerente con il run")
    overview = primary.get("overview") or {}
    _require_exact_keys(overview, {"headline", "claims"}, "sintesi generale")
    if not 3 <= len(overview.get("claims") or []) <= 4:
        raise AgentError("sintesi generale priva del numero richiesto di claim")
    _validate_headline(overview.get("headline"), "generale")
    for expected, actual in zip(expected_periods, actual_periods):
        _require_exact_keys(
            actual,
            {"periodId", "fromHour", "toHour", "headline", "sections"},
            "periodo IA",
        )
        for key in ("periodId", "fromHour", "toHour"):
            if actual.get(key) != expected.get(key):
                raise AgentError(f"periodo IA alterato: {key}")
        sections = actual.get("sections") or []
        section_ids = [item.get("id") for item in sections]
        if len(sections) != len(SECTION_IDS) or set(section_ids) != set(SECTION_IDS):
            raise AgentError("sezioni IA incomplete o duplicate")
        for section in sections:
            _require_exact_keys(section, {"id", "title", "claims"}, "sezione IA")
            if section.get("title") != SECTION_TITLES.get(section.get("id")):
                raise AgentError("titolo di sezione IA non canonico")
            if len(section.get("claims") or []) != 1:
                raise AgentError("ogni sezione IA deve contenere un solo claim")
        _validate_headline(actual.get("headline"), "di periodo")

    seen_claims: set[str] = set()
    referenced: set[str] = set()
    numeric_claims = 0
    for claim, period in _iter_claims(primary):
        if not isinstance(claim, dict):
            raise AgentError("claim IA non strutturato")
        _require_exact_keys(
            claim, {"id", "text", "confidence", "evidenceIds"}, "claim IA"
        )
        claim_id = str(claim.get("id") or "")
        if not claim_id or claim_id in seen_claims:
            raise AgentError("identificatore claim IA assente o duplicato")
        seen_claims.add(claim_id)
        text = _clean_text(claim.get("text"), 421)
        if not text or len(text) > 420 or text != claim.get("text"):
            raise AgentError(f"testo claim {claim_id} assente, eccessivo o non normalizzato")
        if claim.get("confidence") not in CONFIDENCE_LEVELS:
            raise AgentError(f"confidence non valida nel claim {claim_id}")
        refs = list(dict.fromkeys(claim.get("evidenceIds") or []))
        if not 2 <= len(refs) <= 5 or len(refs) != len(claim.get("evidenceIds") or []):
            raise AgentError(f"prove insufficienti o duplicate nel claim {claim_id}")
        if any(ref not in evidence for ref in refs):
            raise AgentError(f"prova inesistente nel claim {claim_id}")
        categories = {
            str(evidence[ref].get("section"))
            for ref in refs
            if str(evidence[ref].get("section")) in INDEPENDENT_EVIDENCE_FAMILIES
        }
        if len(categories) < 2:
            raise AgentError(f"claim {claim_id} basato su una sola famiglia diagnostica")
        if period is not None:
            lower = int(period["fromHour"])
            upper = int(period["toHour"])
            if any(
                not lower <= int(evidence[ref].get("leadHours", -1)) <= upper
                for ref in refs
            ):
                raise AgentError(f"claim {claim_id} usa prove fuori dal proprio periodo")
        claim_numbers = _numbers(text)
        if claim_numbers:
            numeric_claims += 1
            allowed = set()
            for ref in refs:
                allowed |= _numbers(str(evidence[ref].get("text") or ""))
            if not claim_numbers <= allowed:
                missing = ", ".join(sorted(claim_numbers - allowed))
                raise AgentError(f"numeri non documentati nel claim {claim_id}: {missing}")
        referenced.update(refs)
    return {
        "claimCount": len(seen_claims),
        "referencedEvidenceCount": len(referenced),
        "numericClaimsChecked": numeric_claims,
        "schemaValid": True,
        "evidenceGrounded": True,
    }


def _post_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    attempts: int = 3,
) -> dict[str, Any]:
    provider = "Gemini" if "googleapis.com" in url else "Groq"
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urlopen(request, timeout=150) as response:
                body = json.loads(response.read().decode("utf-8"))
            if not isinstance(body, dict):
                raise AgentError("risposta API non strutturata")
            return body
        except HTTPError as error:
            detail = ""
            try:
                error_body = json.loads(error.read().decode("utf-8", errors="replace"))
                api_error = error_body.get("error") if isinstance(error_body, dict) else None
                if isinstance(api_error, dict):
                    identifiers = [
                        api_error.get("status"),
                        api_error.get("type"),
                        api_error.get("code"),
                    ]
                    safe_identifiers = []
                    for identifier in identifiers:
                        normalized = re.sub(r"[^0-9A-Za-z_.-]", "", str(identifier or ""))
                        if normalized and normalized not in safe_identifiers:
                            safe_identifiers.append(normalized[:80])
                    if safe_identifiers:
                        detail = " (" + ", ".join(safe_identifiers) + ")"
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
            last_error = AgentError(f"HTTP {error.code}{detail}")
            if error.code == 429 or error.code >= 500:
                retry_after = error.headers.get("retry-after")
                try:
                    retry_seconds = float(retry_after) if retry_after else float(2 ** (attempt + 1))
                except (TypeError, ValueError):
                    retry_seconds = float(2 ** (attempt + 1))
                wait = min(
                    30.0,
                    max(0.0, retry_seconds),
                )
                if attempt + 1 < attempts:
                    time.sleep(wait)
                    continue
            break
        except (URLError, TimeoutError, ValueError, AgentError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(8, 2 ** attempt))
    if last_error is None:
        raise AgentError(f"{provider} API fallita senza risposta")
    if isinstance(last_error, AgentError):
        raise AgentError(f"{provider} API fallita: {last_error}")
    raise AgentError(f"{provider} API fallita: {type(last_error).__name__}")


def _gemini_analysis(
    packet: dict[str, Any],
    api_key: str,
    post_json: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = post_json(
        GEMINI_ENDPOINT,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        payload={
            "systemInstruction": {"parts": [{"text": PRIMARY_SYSTEM_PROMPT}]},
            "contents": [{
                "role": "user",
                "parts": [{"text": json.dumps(packet, ensure_ascii=False, separators=(",", ":"))}],
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": PRIMARY_SCHEMA,
                "temperature": 0.1,
                "maxOutputTokens": 8192,
                "thinkingConfig": {"thinkingLevel": "HIGH"},
            },
        },
    )
    try:
        candidate = response["candidates"][0]
        parts = candidate["content"]["parts"]
        text = "".join(str(part.get("text") or "") for part in parts)
        result = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise AgentError("Gemini non ha restituito JSON utilizzabile") from error
    usage = response.get("usageMetadata") or {}
    return result, {
        "provider": "google-gemini-api",
        "model": GEMINI_MODEL,
        "promptTokens": usage.get("promptTokenCount"),
        "outputTokens": usage.get("candidatesTokenCount"),
    }


def _review_payload(
    primary: dict[str, Any],
    packet: dict[str, Any],
    claim_ids: set[str],
    scope: str,
) -> dict[str, Any]:
    evidence_by_id = {item["id"]: item for item in packet["evidence"]}
    claims = []
    referenced: set[str] = set()
    for claim, period in _iter_claims(primary):
        if claim["id"] not in claim_ids:
            continue
        refs = list(claim["evidenceIds"])
        referenced.update(refs)
        claims.append({
            "id": claim["id"],
            "text": claim["text"],
            "confidence": claim["confidence"],
            "periodId": period.get("periodId") if period else "overview",
            "evidenceIds": refs,
        })
    compact = {
        "scope": scope,
        "model": packet["model"],
        "runTime": packet["runTime"],
        "semantics": packet.get("sourceSemantics") or {},
        "claims": claims,
        "evidence": [
            {
                "id": evidence_by_id[item]["id"],
                "leadHours": evidence_by_id[item]["leadHours"],
                "section": evidence_by_id[item]["section"],
                "text": _clean_text(evidence_by_id[item]["text"], 360),
            }
            for item in sorted(referenced)
        ],
    }
    encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > 18_000:
        raise AgentError("pacchetto del revisore oltre il limite gratuito sicuro")
    return compact


def _groq_review(
    primary: dict[str, Any],
    packet: dict[str, Any],
    api_key: str,
    post_json: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scopes: list[tuple[str, set[str]]] = [(
        "overview",
        {item["id"] for item in primary["overview"]["claims"]},
    )]
    for period in primary["periods"]:
        scopes.append((
            period["periodId"],
            {
                claim["id"]
                for section in period["sections"]
                for claim in section["claims"]
            },
        ))
    aggregate = {
        "approved": True,
        "rejectedClaimIds": [],
        "downgradeClaimIds": [],
        "issues": [],
    }
    prompt_tokens = 0
    output_tokens = 0
    for scope, claim_ids in scopes:
        review_input = _review_payload(primary, packet, claim_ids, scope)
        response = post_json(
            GROQ_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            payload={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(review_input, ensure_ascii=False, separators=(",", ":")),
                    },
                ],
                "reasoning_effort": "medium",
                "reasoning_format": "hidden",
                "temperature": 0,
                "max_completion_tokens": 1536,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "meteorological_claim_review",
                        "strict": True,
                        "schema": REVIEW_SCHEMA,
                    },
                },
            },
        )
        try:
            text = response["choices"][0]["message"]["content"]
            result = json.loads(text)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise AgentError("Groq non ha restituito una revisione JSON utilizzabile") from error
        returned_ids = set(result.get("rejectedClaimIds") or []) | set(
            result.get("downgradeClaimIds") or []
        )
        if returned_ids - claim_ids:
            raise AgentError(f"revisione {scope} riferita a claim fuori ambito")
        aggregate["approved"] = bool(aggregate["approved"] and result.get("approved"))
        aggregate["rejectedClaimIds"].extend(result.get("rejectedClaimIds") or [])
        aggregate["downgradeClaimIds"].extend(result.get("downgradeClaimIds") or [])
        aggregate["issues"].extend(
            f"{scope}: {_clean_text(item, 200)}" for item in result.get("issues") or []
        )
        usage = response.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        output_tokens += int(usage.get("completion_tokens") or 0)
    aggregate["rejectedClaimIds"] = sorted(set(aggregate["rejectedClaimIds"]))
    aggregate["downgradeClaimIds"] = sorted(set(aggregate["downgradeClaimIds"]))
    aggregate["issues"] = aggregate["issues"][:8]
    return aggregate, {
        "provider": "groq-cloud",
        "model": GROQ_MODEL,
        "requestCount": len(scopes),
        "promptTokens": prompt_tokens,
        "outputTokens": output_tokens,
    }


def _apply_review(primary: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        review,
        {"approved", "rejectedClaimIds", "downgradeClaimIds", "issues"},
        "revisione",
    )
    if not isinstance(review["approved"], bool):
        raise AgentError("esito del revisore non booleano")
    if any(not isinstance(review[name], list) for name in (
        "rejectedClaimIds", "downgradeClaimIds", "issues"
    )):
        raise AgentError("liste del revisore non valide")
    all_ids = {claim["id"] for claim, _ in _iter_claims(primary)}
    rejected = set(review.get("rejectedClaimIds") or [])
    downgraded = set(review.get("downgradeClaimIds") or [])
    unknown = (rejected | downgraded) - all_ids
    if unknown:
        raise AgentError("il revisore ha indicato claim inesistenti")
    if rejected or not review.get("approved"):
        raise AgentError("revisione meteorologica indipendente non approvata")
    for claim, _ in _iter_claims(primary):
        if claim["id"] in downgraded:
            claim["confidence"] = CONFIDENCE_DOWNGRADE[claim["confidence"]]
    return primary


def generate_verified_bulletin(
    deterministic_bulletin: dict[str, Any],
    *,
    gemini_api_key: str,
    groq_api_key: str,
    post_json: Callable[..., dict[str, Any]] = _post_json,
) -> dict[str, Any]:
    """Generate, independently review, and package an AI run synopsis."""
    if not gemini_api_key or not groq_api_key:
        raise AgentError("chiavi API non disponibili")
    packet = build_evidence_packet(deterministic_bulletin)
    primary, primary_usage = _gemini_analysis(packet, gemini_api_key, post_json)
    validation = validate_primary_analysis(primary, packet)
    review, reviewer_usage = _groq_review(
        primary, packet, groq_api_key, post_json
    )
    primary = _apply_review(primary, review)
    validation = validate_primary_analysis(primary, packet)
    referenced = {
        ref
        for claim, _ in _iter_claims(primary)
        for ref in claim["evidenceIds"]
    }
    evidence_catalogue = [
        item for item in packet["evidence"] if item["id"] in referenced
    ]
    return {
        "schemaVersion": 1,
        "method": AGENT_METHOD,
        "status": "validated",
        "generatedAt": _utc_now(),
        "model": packet["model"],
        "runTime": packet["runTime"],
        "area": packet["area"],
        "forecastHorizonHours": packet["forecastHorizonHours"],
        "source": {
            "method": packet["sourceMethod"],
            "nature": "single-deterministic-nwp-run",
            "fieldsModifiedByLlm": False,
        },
        "providers": {
            "primary": primary_usage,
            "reviewer": {
                **reviewer_usage,
                "approved": True,
                "downgradedClaimCount": len(review.get("downgradeClaimIds") or []),
                "issues": [_clean_text(item, 240) for item in review.get("issues") or []],
            },
        },
        "semantics": {
            "claimsRequireEvidence": True,
            "minimumIndependentEvidenceFamilies": 2,
            "numbersMustExistInCitedEvidence": True,
            "confidenceIsQualitativeInternalEvidence": True,
            "notAnOfficialWarning": True,
        },
        "overview": primary["overview"],
        "periods": primary["periods"],
        "evidenceCatalog": evidence_catalogue,
        "validation": validation,
        "disclaimer": (
            "Sintesi IA del singolo run deterministico ICON-2I, vincolata alle "
            "prove del motore fisico e approvata da un secondo modello. Non è "
            "un’allerta, non corregge i campi numerici e non misura un ensemble."
        ),
    }
