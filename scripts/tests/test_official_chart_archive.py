"""Offline tests for official_chart_archive.py."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import official_chart_archive as archive


HTML = b"""<html><body>
<a href="Z__C_EDZW_x_tka01,ana_bwkman_dwdna_O_000000_000000_202607250000_WV12.png">colour</a>
<a href="Z__C_EDZW_x_tka01,ana_bwkman_dwdna_O_000000_000000_202607250000_WV12SW.png">bw</a>
<a href="Z__C_EDZW_x_tka01,ana_bwkman_dwdc_O_000000_000000_202607250000_WV12.png">other</a>
</body></html>"""
PNG = b"\x89PNG\r\n\x1a\n" + b"test-payload"
ok = True
MET_HTML = b'<a href="https://data.consumer-digital.api.metoffice.gov.uk/v1/surface-pressure/colour/2026-07-25T0000/FSXX00T_00.gif">analysis</a>'



original_fetch = archive._fetch
try:
    archive._fetch = lambda url, timeout=45.0: HTML
    records = archive.discover(
        archive.DWD_CURRENT_INDEX, archive.DWD_ANALYSIS_RE
    )
finally:
    archive._fetch = original_fetch

print(f"A) discovery: {len(records)} carta colore North Atlantic")
if len(records) != 1:
    print("  FAIL: duplicato BW o prodotto geografico errato non escluso")
    ok = False
elif records[0]["validTime"] != "2026-07-25T00:00:00+00:00":
    print("  FAIL: ora valida DWD interpretata male")
    ok = False

try:
    archive._fetch = lambda url, timeout=45.0: MET_HTML
    met_records = archive.discover(
        archive.METOFFICE_CURRENT_PAGE, archive.METOFFICE_ANALYSIS_RE
    )
finally:
    archive._fetch = original_fetch
print(f"A2) discovery Met Office: {len(met_records)} analisi")
if (len(met_records) != 1
        or met_records[0]["validTime"] != "2026-07-25T00:00:00+00:00"):
    print("  FAIL: analisi ASXX Met Office o ora valida non riconosciuta")
    ok = False

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    args = SimpleNamespace(
        output=str(root / "charts"),
        catalog=str(root / "catalog.json"),
        index_url=archive.DWD_CURRENT_INDEX,
        since=None,
        limit=None,
        dry_run=False,
        pattern=archive.DWD_ANALYSIS_RE,
        provider="DWD",
        product="analysed-surface-front-chart-north-atlantic-europe",
        terms=archive.DWD_TERMS,
        magics=(b"\x89PNG\r\n\x1a\n",),
    )

    def fake_fetch(url, timeout=45.0):
        return HTML if url.endswith("/analysis/") else PNG

    try:
        archive._fetch = fake_fetch
        first = archive.collect_current(args)
        second = archive.collect_current(args)
    finally:
        archive._fetch = original_fetch

    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    entries = catalog.get("charts", [])
    print(f"B) catalogo immutabile/deduplicato: {len(entries)} record")
    if first != 0 or second != 0 or len(entries) != 1:
        print("  FAIL: raccolta ripetuta non idempotente")
        ok = False
    elif entries[0].get("sha256") != archive._sha256(PNG):
        print("  FAIL: impronta SHA-256 errata")
        ok = False
    else:
        local = root / entries[0]["localPath"]
        if not local.exists() or local.read_bytes() != PNG:
            print("  FAIL: file catalogato non verificabile")
            ok = False

print("ESITO:", "SUPERATO" if ok else "DA RIVEDERE")
raise SystemExit(0 if ok else 1)
