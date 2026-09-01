"""Reproducible ingestion of the official legacy ICON-2I archives.

MeteoHub historical extracts are asynchronous: a request is submitted, its
status is polled and the resulting file is downloaded later.  This module
keeps that state explicit and resumable.  Credentials never enter the state
manifest; downloaded bytes are identified by size and SHA-256 before they can
be retained by the immutable object-store layer.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.parse import quote

from .archive import sha256_file


SCHEMA_VERSION = 1
DEFAULT_BASE_URL = "https://meteohub.agenziaitaliameteo.it"
SUCCESS_STATES = {"SUCCESS", "SUCCEEDED", "COMPLETED", "COMPLETE", "DONE"}
FAILURE_STATES = {"FAILURE", "FAILED", "ERROR", "CANCELLED", "CANCELED"}
RETENTION_ARCHIVE = "archive"
RETENTION_SOURCE_REFERENCE = "source-reference"
RETENTION_MODES = {RETENTION_ARCHIVE, RETENTION_SOURCE_REFERENCE}


class HistoricalArchiveError(RuntimeError):
    """Raised when a historical request cannot be represented safely."""


@dataclass(frozen=True)
class HistoricalDataset:
    name: str
    display_name: str
    first_day: str
    last_day: str
    run_utc: int
    horizontal_resolution_km: float
    domain: str
    vertical_content: str
    kind: str
    license: str = "CC BY 4.0"
    provider: str = "Agenzia ItaliaMeteo / Arpae Emilia-Romagna / CINECA"
    operational: bool = False
    configuration_era: str = "legacy-before-2026-06-17-soil-analysis-change"

    @property
    def start(self) -> date:
        return date.fromisoformat(self.first_day)

    @property
    def end(self) -> date:
        return date.fromisoformat(self.last_day)


HISTORICAL_DATASETS: dict[str, HistoricalDataset] = {
    "ICON_2I_ita2km": HistoricalDataset(
        name="ICON_2I_ita2km",
        display_name="ICON-2I_ita2km [NOT OPERATIONAL]",
        first_day="2024-09-29",
        last_day="2025-05-26",
        run_utc=0,
        horizontal_resolution_km=2.2,
        domain="Italy 47N-35N, 6E-18E",
        vertical_content="surface and model levels",
        kind="deterministic-forecast",
    ),
    "ICON_2I_all2km": HistoricalDataset(
        name="ICON_2I_all2km",
        display_name="ICON-2I_all2km [NOT OPERATIONAL]",
        first_day="2024-09-29",
        last_day="2025-05-26",
        run_utc=0,
        horizontal_resolution_km=2.2,
        domain="full 49N-33N, 3E-22E",
        vertical_content="surface and pressure levels",
        kind="deterministic-forecast",
    ),
    "ICON_2I_ASSIM_ita2km": HistoricalDataset(
        name="ICON_2I_ASSIM_ita2km",
        display_name="ICON-2I_ASSIM_ita2km [NOT OPERATIONAL]",
        first_day="2025-02-04",
        last_day="2025-05-26",
        run_utc=0,
        horizontal_resolution_km=2.2,
        domain="Italy 47N-35N, 6E-18E",
        vertical_content="surface and model levels",
        kind="assimilation-background-not-observational-truth",
    ),
    "ICON_2I_ASSIM_all2km": HistoricalDataset(
        name="ICON_2I_ASSIM_all2km",
        display_name="ICON-2I_ASSIM_all2km [NOT OPERATIONAL]",
        first_day="2025-02-04",
        last_day="2025-05-26",
        run_utc=0,
        horizontal_resolution_km=2.2,
        domain="full 49N-33N, 3E-22E",
        vertical_content="surface and pressure levels",
        kind="assimilation-background-not-observational-truth",
    ),
    "ICON_2I_FCENS": HistoricalDataset(
        name="ICON_2I_FCENS",
        display_name="ICON-2I_FCENS [NOT OPERATIONAL]",
        first_day="2024-06-18",
        last_day="2025-05-26",
        run_utc=0,
        horizontal_resolution_km=2.2,
        domain="full 49N-33N, 3E-22E",
        vertical_content="surface and pressure levels",
        kind="ensemble-forecast",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _days(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _dataset(name: str) -> HistoricalDataset:
    try:
        return HISTORICAL_DATASETS[name]
    except KeyError as error:
        choices = ", ".join(sorted(HISTORICAL_DATASETS))
        raise HistoricalArchiveError(
            f"dataset storico sconosciuto {name!r}; valori ammessi: {choices}"
        ) from error


def load_request_template(path: str | Path | None = None, *, environ=None):
    """Load the exact JSON copied from MeteoHub's request interface.

    The environment form is convenient for a GitHub encrypted secret and is
    deliberately never written back to the state manifest verbatim before
    scheduling fields and date-specific fields have been normalised.
    """

    environment = os.environ if environ is None else environ
    raw = None
    if path:
        raw = Path(path).read_text(encoding="utf-8")
    elif environment.get("METEOHUB_REQUEST_TEMPLATE_JSON"):
        raw = environment["METEOHUB_REQUEST_TEMPLATE_JSON"]
    if not raw:
        return None
    try:
        template = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HistoricalArchiveError("template MeteoHub non e JSON valido") from error
    if not isinstance(template, (dict, list)):
        raise HistoricalArchiveError(
            "il template MeteoHub deve essere un oggetto o una lista di oggetti JSON"
        )
    templates = template if isinstance(template, list) else [template]
    if not templates or not all(isinstance(item, dict) for item in templates):
        raise HistoricalArchiveError("ogni template MeteoHub deve essere un oggetto JSON")
    for item in templates:
        _reject_secret_fields(item)
    return template


def _reject_secret_fields(payload: Any, *, path: str = "template") -> None:
    secret_fragments = ("password", "passwd", "secret", "token", "authorization")
    if isinstance(payload, dict):
        for key, value in payload.items():
            label = str(key).lower()
            if any(fragment in label for fragment in secret_fragments):
                raise HistoricalArchiveError(
                    f"campo sensibile non ammesso nel template: {path}.{key}"
                )
            _reject_secret_fields(value, path=f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _reject_secret_fields(value, path=f"{path}[{index}]")


def _safe_group_name(value: Any, index: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "")).strip("-")
    return (clean or f"group-{index:02d}")[:48]


def _template_groups(template: Any) -> list[tuple[str, dict[str, Any] | None]]:
    templates = template if isinstance(template, list) else [template]
    groups = []
    used = set()
    for index, item in enumerate(templates, start=1):
        if item is not None and not isinstance(item, dict):
            raise HistoricalArchiveError("template MeteoHub non valido")
        if item is not None:
            _reject_secret_fields(item)
        base = _safe_group_name((item or {}).get("request_name"), index)
        group = base
        suffix = 2
        while group in used:
            group = f"{base[:42]}-{suffix}"
            suffix += 1
        used.add(group)
        groups.append((group, item))
    return groups


def _normalise_template(template: dict[str, Any] | None) -> dict[str, Any]:
    payload = deepcopy(template or {})
    for unsafe in (
        "request_name", "reftime", "dataset_names", "on-data-ready",
        "crontab-settings", "schedule", "scheduled",
    ):
        payload.pop(unsafe, None)
    return payload


def _has_effective_filters(template: dict[str, Any] | None) -> bool:
    filters = _normalise_template(template).get("filters")
    if not isinstance(filters, dict):
        return False
    for values in filters.values():
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict) and value:
                return True
            if not isinstance(value, dict) and bool(value):
                return True
    return False


def build_request_payload(
    specification: HistoricalDataset,
    day: date,
    *,
    template: dict[str, Any] | None = None,
    group: str | None = None,
) -> dict[str, Any]:
    if not specification.start <= day <= specification.end:
        raise HistoricalArchiveError(
            f"{day.isoformat()} fuori dall'archivio {specification.name} "
            f"({specification.first_day}..{specification.last_day})"
        )
    payload = _normalise_template(template)
    reference = f"{day.isoformat()}T{specification.run_utc:02d}:00:00.000Z"
    request_name = f"icon2i-history-{day.strftime('%Y%m%d')}"
    if group:
        request_name += f"-{_safe_group_name(group, 1)}"
    payload.update({
        "request_name": request_name,
        "reftime": {"from": reference, "to": reference},
        "dataset_names": [specification.name],
    })
    return payload


def build_plan(
    dataset_name: str,
    start: str | date,
    end: str | date,
    *,
    template: dict[str, Any] | list[dict[str, Any]] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    specification = _dataset(dataset_name)
    first = date.fromisoformat(start) if isinstance(start, str) else start
    last = date.fromisoformat(end) if isinstance(end, str) else end
    if first > last:
        raise HistoricalArchiveError("la data iniziale e successiva alla data finale")
    if first < specification.start or last > specification.end:
        raise HistoricalArchiveError(
            f"intervallo fuori copertura: {specification.first_day}.."
            f"{specification.last_day}"
        )
    groups = _template_groups(template)
    normalised_templates = [_normalise_template(item) for _, item in groups]
    entries = []
    for day in _days(first, last):
        for group, group_template in groups:
            payload = build_request_payload(
                specification, day, template=group_template, group=group
            )
            entries.append({
                "day": day.isoformat(),
                "group": group,
                "runTime": payload["reftime"]["from"].replace(".000Z", "Z"),
                "requestKey": canonical_sha256(payload),
                "status": "PLANNED",
                "payload": payload,
            })
    timestamp = created_at or utc_now()
    dataset_payload = asdict(specification)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "createdAt": timestamp,
        "updatedAt": timestamp,
        "provider": {
            "name": "Agenzia ItaliaMeteo MeteoHub",
            "apiBase": DEFAULT_BASE_URL,
            "licenseCatalog": (
                "https://meteohub.agenziaitaliameteo.it/app/license"
            ),
        },
        "model": {
            "name": "ICON-2I",
            "currentConfigurationBoundary": "2026-06-17T00:00:00Z",
            "historicalArchiveComparableToCurrentWithoutVersionFeature": False,
        },
        "dataset": dataset_payload,
        "selection": {
            "start": first.isoformat(),
            "end": last.isoformat(),
            "days": (last - first).days + 1,
            "requestGroups": [group for group, _ in groups],
            "requests": len(entries),
            "filtered": all(
                _has_effective_filters(item) for _, item in groups
            ),
            "requestTemplateSha256": canonical_sha256(normalised_templates),
        },
        "requests": entries,
    }


def write_state(path: str | Path, state: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state["updatedAt"] = utc_now()
    partial = target.with_suffix(target.suffix + ".part")
    try:
        partial.write_text(
            json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        os.replace(partial, target)
    finally:
        if partial.exists():
            partial.unlink()


def read_state(path: str | Path) -> dict[str, Any]:
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    if state.get("schemaVersion") != SCHEMA_VERSION:
        raise HistoricalArchiveError("versione manifest storico non supportata")
    if not isinstance(state.get("requests"), list):
        raise HistoricalArchiveError("manifest storico privo delle richieste")
    return state


def _find_string(payload: Any, names: Iterable[str]) -> str | None:
    ordered_names = tuple(str(name).lower() for name in names)
    if isinstance(payload, dict):
        for wanted in ordered_names:
            for key, value in payload.items():
                if str(key).lower() == wanted and isinstance(value, (str, int)):
                    return str(value)
        for value in payload.values():
            found = _find_string(value, ordered_names)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_string(value, ordered_names)
            if found:
                return found
    return None


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("requests", "items", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = _records(value)
                if nested:
                    return nested
        if any(key in payload for key in ("request_id", "requestId", "id")):
            return [payload]
    return []


def _safe_filename(value: str) -> str:
    base = Path(str(value)).name
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip(".-")
    if not clean:
        raise HistoricalArchiveError("nome output MeteoHub non valido")
    return clean[:220]


def identify_payload(path: str | Path) -> str:
    with open(path, "rb") as handle:
        head = handle.read(512)
    if head.startswith(b"GRIB"):
        return "grib"
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if head.startswith(b"\x1f\x8b"):
        return "gzip"
    if len(head) >= 262 and head[257:262] == b"ustar":
        return "tar"
    return "binary"


class MeteoHubClient:
    """Small authenticated client around the documented MeteoHub API."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        token: str | None = None,
        username: str | None = None,
        password: str | None = None,
        session=None,
        timeout: tuple[int, int] = (20, 180),
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.username = username
        self.password = password
        if session is None:
            try:
                import requests
            except ImportError as error:  # pragma: no cover - workflows install it
                raise HistoricalArchiveError(
                    "requests non installato: impossibile contattare MeteoHub"
                ) from error
            session = requests.Session()
        self.session = session
        self.timeout = timeout

    @classmethod
    def from_environment(cls, *, environ=None, session=None):
        environment = os.environ if environ is None else environ
        return cls(
            base_url=(environment.get("METEOHUB_BASE_URL") or DEFAULT_BASE_URL),
            token=(environment.get("METEOHUB_TOKEN") or "").strip() or None,
            username=(environment.get("METEOHUB_USERNAME") or "").strip() or None,
            password=environment.get("METEOHUB_PASSWORD") or None,
            session=session,
        )

    def authenticate(self) -> str:
        if self.token:
            return self.token
        if not self.username or not self.password:
            raise HistoricalArchiveError(
                "servono METEOHUB_TOKEN oppure METEOHUB_USERNAME e METEOHUB_PASSWORD"
            )
        response = self.session.post(
            f"{self.base_url}/auth/login",
            json={"username": self.username, "password": self.password},
            timeout=self.timeout,
        )
        response.raise_for_status()
        token = _find_string(
            response.json(), ("access_token", "accesstoken", "token", "jwt")
        )
        if not token:
            raise HistoricalArchiveError("login MeteoHub riuscito ma token assente")
        self.token = token
        return token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.authenticate()}",
            "Accept-Encoding": "identity",
        }

    def dataset(self, dataset_id: str) -> dict[str, Any]:
        """Resolve an exact public catalogue ID before submitting extracts."""

        response = self.session.get(
            f"{self.base_url}/api/datasets",
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        records = payload if isinstance(payload, list) else _records(payload)
        for record in records:
            if str(record.get("id") or "") == dataset_id:
                if str(record.get("category") or "").upper() != "FOR":
                    raise HistoricalArchiveError(
                        f"{dataset_id} non e catalogato come previsione MeteoHub"
                    )
                return record
        raise HistoricalArchiveError(
            f"dataset ID {dataset_id!r} assente dal catalogo MeteoHub corrente"
        )

    def submit(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        response = self.session.post(
            f"{self.base_url}/api/data",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        request_id = _find_string(
            body, ("request_id", "requestid", "task_id", "taskid", "id")
        )
        if not request_id:
            raise HistoricalArchiveError("MeteoHub non ha restituito request_id")
        return request_id, body

    def list_requests(self) -> list[dict[str, Any]]:
        response = self.session.get(
            f"{self.base_url}/api/requests",
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return _records(response.json())

    def download_url(self, filename: str) -> str:
        return f"{self.base_url}/api/data/{quote(filename, safe='')}"

    def download(self, filename: str, destination: str | Path) -> dict[str, Any]:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        url = self.download_url(filename)
        try:
            with self.session.get(
                url, headers=self._headers(), timeout=self.timeout, stream=True
            ) as response:
                response.raise_for_status()
                expected = response.headers.get("Content-Length")
                with open(partial, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                actual_size = int(partial.stat().st_size)
                if expected is not None and actual_size != int(expected):
                    raise HistoricalArchiveError(
                        f"download incompleto: {actual_size} byte, attesi {expected}"
                    )
                if actual_size <= 0:
                    raise HistoricalArchiveError("MeteoHub ha restituito un file vuoto")
                os.replace(partial, target)
        finally:
            if partial.exists():
                partial.unlink()
        return {
            "filename": filename,
            "sourceUrl": url,
            "sizeBytes": int(target.stat().st_size),
            "sha256": sha256_file(target),
            "container": identify_payload(target),
        }


def _record_id(record: dict[str, Any]) -> str | None:
    return _find_string(
        record, ("request_id", "requestid", "task_id", "taskid", "id")
    )


def _record_status(record: dict[str, Any]) -> str:
    return (
        _find_string(record, ("status", "request_status", "requeststatus", "state"))
        or "UNKNOWN"
    ).upper()


def _record_output(record: dict[str, Any]) -> str | None:
    return _find_string(
        record,
        ("fileoutput", "file_output", "output_file", "outputfile", "filename"),
    )


def submit_planned(
    state: dict[str, Any],
    client: MeteoHubClient,
    *,
    limit: int = 5,
    allow_unfiltered: bool = False,
) -> int:
    if limit < 1 or limit > 10:
        raise HistoricalArchiveError("limit deve essere compreso tra 1 e 10")
    if not state.get("selection", {}).get("filtered") and not allow_unfiltered:
        raise HistoricalArchiveError(
            "estrazione non filtrata bloccata: l'output MeteoHub puo superare 1 GB; "
            "usa il JSON di una richiesta filtrata oppure --allow-unfiltered"
        )
    planned_entries = [
        entry for entry in state["requests"] if entry.get("status") == "PLANNED"
    ][:limit]
    if not planned_entries:
        return 0
    dataset_id = str(state.get("dataset", {}).get("name") or "")
    catalogue_record = client.dataset(dataset_id)
    state["dataset"]["catalogVerifiedAt"] = utc_now()
    state["dataset"]["catalogDisplayName"] = catalogue_record.get("name")
    state["dataset"]["catalogIsPublic"] = bool(catalogue_record.get("is_public"))
    submitted = 0
    for entry in planned_entries:
        request_id, _ = client.submit(entry["payload"])
        entry.update({
            "requestId": request_id,
            "status": "SUBMITTED",
            "submittedAt": utc_now(),
        })
        submitted += 1
    return submitted


def sync_submitted(
    state: dict[str, Any],
    client: MeteoHubClient,
    archive_factory,
    *,
    download_dir: str | Path,
    keep_local: bool = False,
    retention_mode: str = RETENTION_ARCHIVE,
) -> dict[str, int]:
    if retention_mode not in RETENTION_MODES:
        raise HistoricalArchiveError(
            "retention_mode deve essere 'archive' o 'source-reference'"
        )
    if retention_mode == RETENTION_SOURCE_REFERENCE and keep_local:
        raise HistoricalArchiveError(
            "source-reference elimina sempre l'estratto; --keep-local non e ammesso"
        )
    remote = {
        request_id: record
        for record in client.list_requests()
        if (request_id := _record_id(record))
    }
    counts = {
        "completed": 0,
        "sourceVerified": 0,
        "failed": 0,
        "pending": 0,
        "missing": 0,
    }
    destination = Path(download_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for entry in state["requests"]:
        if entry.get("status") not in {"SUBMITTED", "PROCESSING", "UNKNOWN"}:
            continue
        request_id = str(entry.get("requestId") or "")
        record = remote.get(request_id)
        if not record:
            entry["lastCheckedAt"] = utc_now()
            counts["missing"] += 1
            continue
        status = _record_status(record)
        entry["providerStatus"] = status
        entry["lastCheckedAt"] = utc_now()
        if status in FAILURE_STATES:
            entry["status"] = "FAILED"
            counts["failed"] += 1
            continue
        if status not in SUCCESS_STATES:
            entry["status"] = "PROCESSING"
            counts["pending"] += 1
            continue
        filename = _record_output(record)
        if not filename:
            entry["status"] = "PROCESSING"
            counts["pending"] += 1
            continue
        archive = archive_factory(entry["runTime"])
        if (
            retention_mode == RETENTION_ARCHIVE
            and not archive.enabled
            and not keep_local
        ):
            raise HistoricalArchiveError(
                "storage raw non attivo: download storico bloccato prima del trasferimento"
            )
        local_name = (
            f"{entry['day']}-{_safe_group_name(entry.get('group'), 1)}-"
            f"{_safe_filename(filename)}"
        )
        local_path = destination / local_name
        download = client.download(filename, local_path)
        archive_object = None
        if retention_mode == RETENTION_ARCHIVE:
            archive_object = archive.retain_source(
                path=local_path,
                name=filename,
                role=f"historical-{state['dataset']['name']}",
                source_url=download["sourceUrl"],
                expected_sha256=download["sha256"],
            )
        download["archiveObject"] = archive_object
        download["retainedInArchive"] = archive_object is not None
        download["localPathRetained"] = bool(
            keep_local and retention_mode == RETENTION_ARCHIVE
        )
        download["retentionMode"] = retention_mode
        download["sourceReference"] = {
            "provider": state["provider"]["name"],
            "datasetId": state["dataset"]["name"],
            "requestId": request_id,
            "requestKey": entry["requestKey"],
            "regenerationPayloadSha256": canonical_sha256(entry["payload"]),
        }
        source_only = retention_mode == RETENTION_SOURCE_REFERENCE
        entry.update({
            "status": "SOURCE_VERIFIED" if source_only else "COMPLETED",
            "completedAt": None if source_only else utc_now(),
            "sourceVerifiedAt": utc_now() if source_only else None,
            "fileOutput": filename,
            "download": download,
            "rawRetained": not source_only and bool(
                archive_object is not None or keep_local
            ),
            "derivedProductCreated": False,
            "trainingReady": False,
        })
        if source_only or not keep_local:
            local_path.unlink()
        if source_only:
            counts["sourceVerified"] += 1
        else:
            counts["completed"] += 1
    return counts
