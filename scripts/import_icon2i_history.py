#!/usr/bin/env python3
"""Plan, submit and retain official historical ICON-2I extracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meteo_analysis.verification.historical import (
    HISTORICAL_DATASETS,
    HistoricalArchiveError,
    MeteoHubClient,
    RETENTION_ARCHIVE,
    RETENTION_MODES,
    RETENTION_SOURCE_REFERENCE,
    build_plan,
    load_request_template,
    read_state,
    submit_planned,
    sync_submitted,
    write_state,
)
from meteo_analysis.verification.object_storage import RawGribArchive
from meteo_analysis.verification.object_storage import (
    ObjectStoreConfigurationError,
    ObjectStoreSettings,
)


def _archive_factory(run_time: str):
    return RawGribArchive.from_environment(run_time=run_time)


def _summary(state):
    counts = {}
    for entry in state["requests"]:
        status = entry.get("status", "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _create_plan(arguments, template):
    state = build_plan(
        arguments.dataset,
        arguments.start,
        arguments.end,
        template=template,
    )
    write_state(arguments.state, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Importazione riprendibile degli archivi ufficiali ICON-2I MeteoHub"
        )
    )
    parser.add_argument(
        "action", choices=("catalog", "plan", "submit", "sync", "ingest")
    )
    parser.add_argument(
        "--dataset", choices=sorted(HISTORICAL_DATASETS),
        default="ICON_2I_ita2km",
    )
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--state", default="historical_import/state.json")
    parser.add_argument("--request-template")
    parser.add_argument("--max-requests", type=int, default=5)
    parser.add_argument("--download-dir", default="historical_import/downloads")
    parser.add_argument("--keep-local", action="store_true")
    parser.add_argument(
        "--retention-mode",
        choices=sorted(RETENTION_MODES),
        default=RETENTION_ARCHIVE,
        help=(
            "archive conserva i byte in S3/locale; source-reference verifica "
            "un estratto e lo elimina, senza dichiararlo archiviato"
        ),
    )
    parser.add_argument("--allow-unfiltered", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=3000)
    arguments = parser.parse_args()

    if arguments.action == "catalog":
        print(json.dumps(
            {name: specification.__dict__ for name, specification in HISTORICAL_DATASETS.items()},
            indent=2,
            sort_keys=True,
        ))
        return

    template = load_request_template(arguments.request_template)
    state_path = Path(arguments.state)
    if arguments.action == "plan":
        if not arguments.start or not arguments.end:
            parser.error("plan richiede --start e --end")
        state = _create_plan(arguments, template)
        print(json.dumps(_summary(state), sort_keys=True))
        return

    if state_path.exists():
        state = read_state(state_path)
    else:
        if arguments.action != "ingest" or not arguments.start or not arguments.end:
            parser.error("manifest assente: usa plan o fornisci date a ingest")
        state = _create_plan(arguments, template)

    if (
        arguments.action in {"submit", "ingest"}
        and not state.get("selection", {}).get("filtered")
        and not arguments.allow_unfiltered
    ):
        raise HistoricalArchiveError(
            "estrazione non filtrata bloccata: configura il template MeteoHub"
        )
    if (
        arguments.action == "ingest"
        and arguments.retention_mode == RETENTION_SOURCE_REFERENCE
        and arguments.keep_local
    ):
        parser.error("--keep-local non e compatibile con source-reference")
    if (
        arguments.action == "ingest"
        and arguments.retention_mode == RETENTION_ARCHIVE
        and not arguments.keep_local
    ):
        try:
            storage = ObjectStoreSettings.from_environment()
        except ObjectStoreConfigurationError as error:
            raise HistoricalArchiveError(str(error)) from error
        if not storage.enabled:
            raise HistoricalArchiveError(
                "ingest richiede lo storage raw oppure l'opzione --keep-local"
            )

    client = MeteoHubClient.from_environment()
    if arguments.action in {"submit", "ingest"}:
        submitted = submit_planned(
            state,
            client,
            limit=arguments.max_requests,
            allow_unfiltered=arguments.allow_unfiltered,
        )
        write_state(state_path, state)
        print(f"Richieste inviate: {submitted}", flush=True)
        if arguments.action == "submit":
            return

    if arguments.action == "sync":
        counts = sync_submitted(
            state, client, _archive_factory,
            download_dir=arguments.download_dir,
            keep_local=arguments.keep_local,
            retention_mode=arguments.retention_mode,
        )
        write_state(state_path, state)
        print(json.dumps(counts, sort_keys=True))
        return

    deadline = time.monotonic() + arguments.timeout_seconds
    while True:
        counts = sync_submitted(
            state, client, _archive_factory,
            download_dir=arguments.download_dir,
            keep_local=arguments.keep_local,
            retention_mode=arguments.retention_mode,
        )
        write_state(state_path, state)
        print(json.dumps(counts, sort_keys=True), flush=True)
        active = sum(
            entry.get("status") in {"SUBMITTED", "PROCESSING", "UNKNOWN"}
            for entry in state["requests"]
        )
        if not active:
            break
        if time.monotonic() >= deadline:
            raise HistoricalArchiveError(
                "tempo di attesa esaurito; il manifest e valido e puo essere ripreso con sync"
            )
        time.sleep(max(5, min(arguments.poll_seconds, 60)))


if __name__ == "__main__":
    try:
        main()
    except HistoricalArchiveError as error:
        raise SystemExit(f"Errore archivio storico: {error}") from error
