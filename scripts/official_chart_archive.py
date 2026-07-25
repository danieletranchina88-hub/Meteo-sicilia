"""Acquire immutable official synoptic-chart references for front validation.

The charts are evidence sources, not automatically trustworthy vector labels.
They must be digitised without viewing this algorithm's prediction and retain
provider, valid time, URL and SHA-256 provenance in the benchmark manifest.

Only Python's standard library is used so the collector can run in CI.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen


DWD_CURRENT_INDEX = "https://opendata.dwd.de/weather/charts/analysis/"
DWD_EMB_INDEX = "https://download.dwd.de/pub/EMB/"
DWD_TERMS = "https://www.dwd.de/EN/service/copyright/copyright_node.html"
USER_AGENT = "Meteo-sicilia-front-validation/1.0"
METOFFICE_CURRENT_PAGE = "https://weather.metoffice.gov.uk/maps-and-charts/surface-pressure"
METOFFICE_TERMS = "https://www.metoffice.gov.uk/policies/legal"

# North Atlantic/Europe, analysed surface chart, colour version.  The SW
# suffix is the black-and-white duplicate and is intentionally excluded.
DWD_ANALYSIS_RE = re.compile(
    r"ana_bwkman_dwdna_O_000000_000000_(?P<valid>\d{12})_WV12\.png$",
    re.IGNORECASE,
)
DWD_EMB_RE = re.compile(r"EMB_[A-Za-z]+_\d{4}\.zip$", re.IGNORECASE)

METOFFICE_ANALYSIS_RE = re.compile(
    r"surface-pressure/colour/(?P<valid_iso>\d{4}-\d{2}-\d{2}T\d{4})/FSXX00T_00\.gif$",
    re.IGNORECASE,
)

class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs) -> None:
        values = dict(attrs)
        for attribute in ("href", "src"):
            url = values.get(attribute)
            if url and url not in self.hrefs:
                self.hrefs.append(url)


def _safe_url(base: str, href: str) -> str:
    """Join an Apache-index href while preserving commas in DWD filenames."""
    joined = urljoin(base, href)
    parts = urlsplit(joined)
    path = quote(parts.path, safe="/%:,=+@")
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def _fetch(url: str, timeout: float = 45.0) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def discover(index_url: str, pattern: re.Pattern[str]) -> list[dict]:
    parser = _Links()
    parser.feed(_fetch(index_url).decode("utf-8", errors="replace"))
    records = []
    for href in parser.hrefs:
        filename = Path(urlsplit(href).path).name
        match = pattern.search(href)
        if not match:
            continue
        record = {"filename": filename, "url": _safe_url(index_url, href)}
        if "valid" in pattern.groupindex:
            valid = datetime.strptime(match.group("valid"), "%Y%m%d%H%M")
        elif "valid_iso" in pattern.groupindex:
            valid = datetime.strptime(match.group("valid_iso"), "%Y-%m-%dT%H%M")
        else:
            valid = None
        if valid is not None:
            record["validTime"] = valid.replace(tzinfo=timezone.utc).isoformat()
        records.append(record)
    return sorted(records, key=lambda item: item.get("validTime", item["filename"]))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_catalog(
    path: Path, provider: str, product: str, terms: str
) -> dict:
    if not path.exists():
        return {
            "schemaVersion": 1,
            "provider": provider,
            "product": product,
            "terms": terms,
            "charts": [],
        }
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_catalog(path: Path, catalog: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(catalog, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def collect_current(args) -> int:
    output = Path(args.output).resolve()
    catalog_path = Path(args.catalog).resolve()
    records = discover(args.index_url, args.pattern)
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        records = [r for r in records if datetime.fromisoformat(r["validTime"]) >= since]
    if args.limit is not None:
        records = records[-max(0, args.limit):]

    catalog = _load_catalog(
        catalog_path, args.provider, args.product, args.terms
    )
    known = {item["url"]: item for item in catalog.get("charts", [])}
    downloaded = 0
    for record in records:
        if record["url"] in known:
            continue
        destination = output / record["validTime"][:4] / record["validTime"][5:7]
        destination = destination / record["filename"]
        if args.dry_run:
            print(record["url"])
            continue
        payload = _fetch(record["url"])
        if not any(payload.startswith(magic) for magic in args.magics):
            raise RuntimeError(f"unexpected chart payload: {record['url']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(destination)
        item = {
            **record,
            "provider": args.provider,
            "productKind": "analysis",
            "localPath": destination.relative_to(catalog_path.parent).as_posix(),
            "sha256": _sha256(payload),
            "bytes": len(payload),
            "accessedAt": datetime.now(timezone.utc).isoformat(),
        }
        known[item["url"]] = item
        downloaded += 1

    if not args.dry_run:
        catalog["updatedAt"] = datetime.now(timezone.utc).isoformat()
        catalog["charts"] = sorted(
            known.values(), key=lambda item: (item.get("validTime", ""), item["url"])
        )
        _write_catalog(catalog_path, catalog)
    print(f"discovered={len(records)} downloaded={downloaded} catalog={catalog_path}")
    return 0


def list_emb(args) -> int:
    records = discover(args.index_url, DWD_EMB_RE)
    document = {
        "provider": "DWD",
        "product": "European Meteorological Bulletin monthly archives",
        "terms": DWD_TERMS,
        "archives": records,
    }
    print(json.dumps(document, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    current = commands.add_parser("collect-current", help="archive current DWD analyses")
    current.add_argument("--index-url", default=DWD_CURRENT_INDEX)
    current.add_argument("--output", default="benchmarks/fronts/source_charts/dwd")
    current.add_argument(
        "--catalog", default="benchmarks/fronts/source_charts/dwd/catalog.json"
    )
    current.add_argument("--since", help="earliest UTC date, YYYY-MM-DD")
    current.add_argument("--limit", type=int, help="keep only the latest N discoveries")
    current.add_argument("--dry-run", action="store_true")
    current.set_defaults(
        handler=collect_current,
        pattern=DWD_ANALYSIS_RE,
        provider="DWD",
        product="analysed-surface-front-chart-north-atlantic-europe",
        terms=DWD_TERMS,
        magics=(b"\x89PNG\r\n\x1a\n",),
    )

    emb = commands.add_parser("list-emb", help="list official six-month EMB ZIP files")
    metoffice = commands.add_parser(
        "collect-metoffice", help="archive current Met Office analysis"
    )
    metoffice.add_argument("--index-url", default=METOFFICE_CURRENT_PAGE)
    metoffice.add_argument(
        "--output", default="benchmarks/fronts/source_charts/metoffice"
    )
    metoffice.add_argument(
        "--catalog", default="benchmarks/fronts/source_charts/metoffice/catalog.json"
    )
    metoffice.add_argument("--since", help="earliest UTC date, YYYY-MM-DD")
    metoffice.add_argument("--limit", type=int, help="keep only the latest N discoveries")
    metoffice.add_argument("--dry-run", action="store_true")
    metoffice.set_defaults(
        handler=collect_current,
        pattern=METOFFICE_ANALYSIS_RE,
        provider="Met Office",
        product="ASXX-surface-pressure-analysis",
        terms=METOFFICE_TERMS,
        magics=(b"GIF87a", b"GIF89a"),
    )

    emb.add_argument("--index-url", default=DWD_EMB_INDEX)
    emb.set_defaults(handler=list_emb)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except Exception as exc:
        print(f"official chart archive failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
