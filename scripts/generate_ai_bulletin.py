#!/usr/bin/env python3
"""Generate the optional, evidence-grounded AI synopsis for one ICON-2I run."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meteo_analysis.agents.meteorologist import (  # noqa: E402
    AGENT_METHOD,
    GEMINI_MODEL,
    GROQ_MODEL,
    generate_verified_bulletin,
)
from meteo_analysis.verification.archive import (  # noqa: E402
    build_run_manifest,
    write_run_manifest,
)


def _read_gzip_json(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON non strutturato: {path.name}")
    return payload


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        partial.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _write_gzip_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        with gzip.open(partial, "wt", encoding="utf-8", compresslevel=9) as handle:
            json.dump(
                payload, handle, ensure_ascii=False,
                separators=(",", ":"), allow_nan=False,
            )
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _safe_error(error: Exception) -> str:
    message = " ".join(str(error).split())
    message = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[secret]", message)
    message = re.sub(r"gsk_[0-9A-Za-z_-]{16,}", "[secret]", message)
    return f"{type(error).__name__}: {message[:240]}"


def _refresh_manifest(data_dir: Path, status: dict) -> None:
    path = data_dir / "archive_manifest.json"
    if not path.exists():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    catalog = json.loads((data_dir / "catalog.json").read_text(encoding="utf-8"))
    algorithms = dict(previous.get("algorithms") or {})
    algorithms["aiMeteorologist"] = (
        AGENT_METHOD if status.get("status") == "validated" else None
    )
    refreshed = build_run_manifest(
        data_dir,
        run_time=previous["model"]["runTime"],
        catalog=catalog,
        source_assets=previous.get("sourceAssets") or [],
        algorithms=algorithms,
        domain=previous.get("domain") or {},
        object_storage=previous.get("objectStorage") or None,
    )
    write_run_manifest(path, refreshed)


def run(data_dir: Path) -> dict:
    deterministic_path = data_dir / "expert_bulletin.json.gz"
    output_path = data_dir / "ai_expert_bulletin.json.gz"
    status_path = data_dir / "ai_agent_status.json"
    status = {
        "schemaVersion": 1,
        "method": AGENT_METHOD,
        "status": "fallback",
        "primaryModel": GEMINI_MODEL,
        "reviewerModel": GROQ_MODEL,
        "reason": None,
    }
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not gemini_key or not groq_key:
        missing = []
        if not gemini_key:
            missing.append("GEMINI_API_KEY")
        if not groq_key:
            missing.append("GROQ_API_KEY")
        status["reason"] = "segreti non disponibili: " + ", ".join(missing)
        if output_path.exists():
            output_path.unlink()
        _write_json_atomic(status_path, status)
        _refresh_manifest(data_dir, status)
        return status

    try:
        deterministic = _read_gzip_json(deterministic_path)
        product = generate_verified_bulletin(
            deterministic,
            gemini_api_key=gemini_key,
            groq_api_key=groq_key,
        )
        _write_gzip_json_atomic(output_path, product)
        status.update({
            "status": "validated",
            "runTime": product["runTime"],
            "generatedAt": product["generatedAt"],
            "reason": "analisi primaria, revisione indipendente e controlli deterministici superati",
            "claimCount": product["validation"]["claimCount"],
        })
    except Exception as error:  # fallback is an explicit operational feature
        if output_path.exists():
            output_path.unlink()
        status["reason"] = _safe_error(error)
    _write_json_atomic(status_path, status)
    _refresh_manifest(data_dir, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_weather")
    arguments = parser.parse_args()
    status = run(Path(arguments.data_dir))
    print(json.dumps(status, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
