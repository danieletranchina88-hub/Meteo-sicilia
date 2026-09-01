#!/usr/bin/env python3
"""Contract tests for the resumable MeteoHub historical importer."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from meteo_analysis.verification.historical import (
    HistoricalArchiveError,
    MeteoHubClient,
    build_plan,
    build_request_payload,
    read_state,
    submit_planned,
    sync_submitted,
    write_state,
    HISTORICAL_DATASETS,
)


FILTER_TEMPLATE = {
    "request_name": "copied-from-ui",
    "reftime": {"from": "2000-01-01T00:00:00Z", "to": "2000-01-01T00:00:00Z"},
    "dataset_names": ["wrong-dataset"],
    "filters": {"quantity": [{"name": "temperature"}]},
    "on-data-ready": True,
    "crontab-settings": {"hour": 1},
}


class FakeResponse:
    def __init__(self, payload=None, *, content=b"", headers=None):
        self.payload = payload
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, chunk_size):
        midpoint = max(1, len(self.content) // 2)
        yield self.content[:midpoint]
        yield self.content[midpoint:]


class FakeSession:
    def __init__(self):
        self.submissions = 0

    def post(self, url, **kwargs):
        if url.endswith("/auth/login"):
            assert kwargs["json"]["password"] == "secret"
            return FakeResponse({"access_token": "test-token"})
        assert url.endswith("/api/data")
        assert kwargs["headers"]["Authorization"] == "Bearer test-token"
        self.submissions += 1
        return FakeResponse({"request_id": f"request-{self.submissions}"})

    def get(self, url, **kwargs):
        if url.endswith("/api/requests"):
            return FakeResponse({"requests": [{
                "request_id": "request-1",
                "status": "SUCCESS",
                "fileoutput": "extract.grib",
            }]})
        content = b"GRIB" + b"historical-test-payload"
        return FakeResponse(
            content=content, headers={"Content-Length": str(len(content))}
        )


class MemoryArchive:
    enabled = True

    def __init__(self, run_time):
        self.run_time = run_time

    def retain_source(self, **kwargs):
        assert kwargs["expected_sha256"]
        return {
            "key": f"history/{self.run_time}/{Path(kwargs['path']).name}",
            "sha256": kwargs["expected_sha256"],
        }


def test_catalog_and_date_boundaries():
    specification = HISTORICAL_DATASETS["ICON-2I_ita2km"]
    assert specification.first_day == "2024-09-29"
    assert specification.last_day == "2025-05-26"
    assert specification.vertical_content == "surface and model levels"
    try:
        build_plan("ICON-2I_ita2km", "2024-09-28", "2024-09-29")
    except HistoricalArchiveError:
        pass
    else:
        raise AssertionError("intervallo esterno accettato")


def test_plan_is_deterministic_and_removes_scheduling():
    fixed = "2026-09-01T08:00:00Z"
    first = build_plan(
        "ICON-2I_ita2km", "2024-09-29", "2024-09-30",
        template=FILTER_TEMPLATE, created_at=fixed,
    )
    second = build_plan(
        "ICON-2I_ita2km", "2024-09-29", "2024-09-30",
        template=FILTER_TEMPLATE, created_at=fixed,
    )
    assert first == second
    assert first["selection"]["filtered"] is True
    assert first["selection"]["days"] == 2
    assert first["selection"]["requests"] == 2
    payload = first["requests"][0]["payload"]
    assert payload["dataset_names"] == ["ICON-2I_ita2km"]
    assert payload["reftime"]["from"] == "2024-09-29T00:00:00.000Z"
    assert "on-data-ready" not in payload
    assert "crontab-settings" not in payload
    assert first["model"]["historicalArchiveComparableToCurrentWithoutVersionFeature"] is False

    sharded = build_plan(
        "ICON-2I_ita2km", "2024-09-29", "2024-09-30",
        template=[
            FILTER_TEMPLATE,
            {**FILTER_TEMPLATE, "request_name": "upper-air"},
        ],
        created_at=fixed,
    )
    assert sharded["selection"]["days"] == 2
    assert sharded["selection"]["requests"] == 4
    assert len(set(item["requestKey"] for item in sharded["requests"])) == 4


def test_unfiltered_submission_fails_closed():
    state = build_plan("ICON-2I_ita2km", "2024-09-29", "2024-09-29")
    client = MeteoHubClient(token="token", session=FakeSession())
    try:
        submit_planned(state, client)
    except HistoricalArchiveError:
        pass
    else:
        raise AssertionError("richiesta non filtrata inviata senza opt-in")

    try:
        build_plan(
            "ICON-2I_ita2km", "2024-09-29", "2024-09-29",
            template={
                "filters": {"quantity": [{"name": "temperature"}]},
                "authorizationToken": "must-never-enter-the-manifest",
            },
        )
    except HistoricalArchiveError:
        pass
    else:
        raise AssertionError("campo sensibile accettato nel manifest")


def test_submit_download_archive_and_resume_state():
    state = build_plan(
        "ICON-2I_ita2km", "2024-09-29", "2024-09-29",
        template=FILTER_TEMPLATE,
    )
    session = FakeSession()
    client = MeteoHubClient(
        username="user@example.test", password="secret", session=session
    )
    assert submit_planned(state, client, limit=1) == 1
    assert submit_planned(state, client, limit=1) == 0
    assert state["requests"][0]["requestId"] == "request-1"
    with tempfile.TemporaryDirectory() as temporary:
        counts = sync_submitted(
            state, client, lambda run: MemoryArchive(run),
            download_dir=temporary,
        )
        assert counts["completed"] == 1
        entry = state["requests"][0]
        assert entry["status"] == "COMPLETED"
        assert entry["download"]["container"] == "grib"
        assert entry["download"]["retainedInArchive"] is True
        assert not list(Path(temporary).iterdir())
        state_path = Path(temporary) / "state.json"
        write_state(state_path, state)
        restored = read_state(state_path)
        assert restored["requests"][0]["download"]["sha256"]
        json.dumps(restored, allow_nan=False)


def test_payload_date_guard():
    try:
        build_request_payload(
            HISTORICAL_DATASETS["ICON-2I_ita2km"],
            datetime(2026, 1, 1, tzinfo=timezone.utc).date(),
        )
    except HistoricalArchiveError:
        pass
    else:
        raise AssertionError("giorno inesistente nell'archivio accettato")


if __name__ == "__main__":
    test_catalog_and_date_boundaries()
    test_plan_is_deterministic_and_removes_scheduling()
    test_unfiltered_submission_fails_closed()
    test_submit_download_archive_and_resume_state()
    test_payload_date_guard()
    print("Historical ICON-2I archive tests passed")
