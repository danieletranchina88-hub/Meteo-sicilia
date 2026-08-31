#!/usr/bin/env python3
"""Scientific regression tests for the forecast-observation foundation."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from meteo_analysis.verification.archive import (  # noqa: E402
    build_run_manifest,
    source_asset_record,
)
from meteo_analysis.verification.metrics import verify_station_forecasts  # noqa: E402
from meteo_analysis.verification.object_storage import (  # noqa: E402
    ObjectStoreConfigurationError,
    RawGribArchive,
)
from meteo_analysis.verification.observations import (  # noqa: E402
    fetch_italy_metar_observations,
    normalize_italian_station_network,
    normalize_metar_reports,
)
from meteo_analysis.verification.stations import (  # noqa: E402
    StationForecastArchive,
    bilinear_sample,
)
from scripts.archive_run import build_bundle  # noqa: E402


def test_observation_semantics():
    reports = [{
        "icaoId": "LICJ",
        "name": "Palermo",
        "lat": 38.18,
        "lon": 13.10,
        "elev": 21,
        "obsTime": "2026-08-30T00:20:00Z",
        "temp": 24,
        "dewp": 18,
        "wspd": 10,
        "wgst": 17,
        "wdir": 270,
        "slp": 1011.7,
        "altim": 29.91,
        "rawOb": "LICJ 300020Z 27010G17KT ...",
    }]
    payload = normalize_metar_reports(
        reports,
        domain=(33.7, 3.0, 48.9, 22.0),
        captured_at=datetime(2026, 8, 30, 0, 25, tzinfo=timezone.utc),
    )
    station = payload["stations"][0]
    assert station["obsTime"] == 1788049200
    assert station["pressHpa"] == 1011.7
    assert station["seaLevelPressureHpa"] == 1011.7
    assert 1012.0 < station["altimeterHpa"] < 1013.5
    assert station["windGustKmh"] == 31.48

    no_slp = normalize_metar_reports(
        [{**reports[0], "slp": None}], domain=(33.7, 3.0, 48.9, 22.0)
    )["stations"][0]
    assert no_slp["pressHpa"] is None
    assert no_slp["altimeterHpa"] is not None


def test_italian_network_and_live_subset_are_distinct():
    catalogue = normalize_italian_station_network([
        {
            "icaoId": "LICJ", "site": "Palermo Arpt", "lat": 38.176,
            "lon": 13.091, "elev": 20, "country": "IT",
            "siteType": ["METAR", "TAF"],
        },
        {
            "icaoId": "LIVP", "site": "Monte Paganella", "lat": 46.143,
            "lon": 11.038, "elev": 2125, "country": "IT",
            "siteType": ["METAR"],
        },
        {
            "icaoId": "LFMN", "site": "Nice", "lat": 43.658,
            "lon": 7.216, "elev": 4, "country": "FR",
            "siteType": ["METAR"],
        },
        {
            "icaoId": "LIEE", "site": "Cagliari", "lat": 39.243,
            "lon": 9.06, "elev": 1, "country": "IT",
            "siteType": ["TAF"],
        },
    ])
    assert catalogue["count"] == 2
    assert [item["id"] for item in catalogue["stations"]] == ["LICJ", "LIVP"]

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Session:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            if "stationinfo" in url:
                return Response([
                    {
                        "icaoId": "LICJ", "site": "Palermo Arpt",
                        "lat": 38.176, "lon": 13.091, "elev": 20,
                        "country": "IT", "siteType": ["METAR", "TAF"],
                    },
                    {
                        "icaoId": "LIVP", "site": "Monte Paganella",
                        "lat": 46.143, "lon": 11.038, "elev": 2125,
                        "country": "IT", "siteType": ["METAR"],
                    },
                ])
            return Response([{
                "icaoId": "LICJ", "obsTime": "2026-08-30T00:20:00Z",
                "temp": 24, "dewp": 18, "wspd": 10, "wdir": 270,
                "altim": 1013, "lat": 38.176, "lon": 13.091,
            }])

    session = Session()
    payload = fetch_italy_metar_observations(session=session)
    assert payload["stationNetwork"]["count"] == 2
    assert payload["coverage"]["registeredMetarStations"] == 2
    assert payload["coverage"]["reportingMetarStations"] == 1
    assert payload["stations"][0]["name"] == "Palermo Arpt"
    assert len(session.calls) == 2
    assert session.calls[1][1]["params"]["bbox"] == "35.0,6.0,48.0,19.0"


def test_native_bilinear_sampling_and_archive():
    latitudes = np.asarray([42.0, 41.0, 40.0])
    longitudes = np.asarray([10.0, 11.0, 12.0])
    lon2d, lat2d = np.meshgrid(longitudes, latitudes)
    field = 2.0 * lat2d + 3.0 * lon2d
    sampled = bilinear_sample(
        field, latitudes, longitudes, [(40.5, 10.25), (60.0, 10.0)]
    )
    assert abs(sampled[0] - (2.0 * 40.5 + 3.0 * 10.25)) < 1.0e-10
    assert np.isnan(sampled[1])

    archive = StationForecastArchive(
        latitudes,
        longitudes,
        [{"id": "LICJ", "name": "Palermo", "lat": 40.5, "lon": 10.25}],
        run_time="2026-08-30T00:00:00Z",
    )
    archive.add(0, "2026-08-30T00:00:00Z", {"temperature2m": field})
    archive.add(1, "2026-08-30T01:00:00Z", {"temperature2m": field + 1})
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "forecast_samples.json.gz"
        payload = archive.write(path)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            reloaded = json.load(handle)
        assert payload == reloaded
        values = payload["fields"]["temperature2m"]["valuesByStation"][0]
        assert values == [111.75, 112.75]
        json.dumps(payload, allow_nan=False)


def test_time_matching_metrics_and_pressure_exclusion():
    forecast = {
        "model": "ICON-2I",
        "runTime": "2026-08-30T00:00:00Z",
        "stations": [{"id": "LICJ"}],
        "times": [
            {"leadHours": 0, "validTime": "2026-08-30T00:00:00Z"},
            {"leadHours": 1, "validTime": "2026-08-30T01:00:00Z"},
        ],
        "fields": {
            "temperature2m": {"valuesByStation": [[22.0, 24.0]]},
            "relativeHumidity2m": {"valuesByStation": [[70.0, 65.0]]},
            "pressureMsl": {"valuesByStation": [[1012.0, 1014.0]]},
            "windU10": {"valuesByStation": [[5.0, 6.0]]},
            "windV10": {"valuesByStation": [[0.0, 0.0]]},
        },
    }
    observations = [{
        "stations": [
            {
                "id": "LICJ",
                "obsTime": "2026-08-30T00:10:00Z",
                "tempC": 20.0,
                "dewpC": 15.0,
                "wspdKmh": 18.0,
                "wdir": 270.0,
                "seaLevelPressureHpa": 1010.0,
                "altimeterHpa": 1015.0,
            },
            {
                "id": "LICJ",
                "obsTime": "2026-08-30T01:10:00Z",
                "tempC": 23.0,
                "dewpC": 16.0,
                "wspdKmh": 18.0,
                "wdir": 270.0,
                "seaLevelPressureHpa": None,
                "altimeterHpa": 1014.0,
            },
        ]
    }]
    result = verify_station_forecasts(forecast, observations)
    assert result["matchedForecastStationTimes"] == 2
    assert result["metrics"]["temperature2m"] == {
        "count": 2, "bias": 1.5, "mae": 1.5, "rmse": 1.5811
    }
    # The second altimeter value must never masquerade as observed MSLP.
    assert result["metrics"]["pressureMsl"]["count"] == 1
    assert result["metrics"]["pressureMsl"]["bias"] == 2.0
    assert result["metrics"]["windU10"]["mae"] == 0.5
    json.dumps(result, allow_nan=False)


def test_checksum_manifest_is_truthful():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "catalog.json").write_text("[]", encoding="utf-8")
        (directory / "step_0.json.gz").write_bytes(b"forecast")
        source = directory / "source.grib"
        source.write_bytes(b"GRIB-data-7777")
        record = source_asset_record(
            name="surface",
            url="https://example.invalid/source.grib",
            path=source,
            role="surface-run",
            required=True,
            retained=False,
        )
        manifest = build_run_manifest(
            directory,
            run_time="2026-08-30T00:00:00Z",
            catalog=[{
                "leadHours": 0,
                "validTime": "2026-08-30T00:00:00Z",
            }],
            source_assets=[record],
            created_at=datetime(2026, 8, 30, 3, tzinfo=timezone.utc),
        )
        assert manifest["completeness"]["complete0To72Hourly"] is False
        assert manifest["archiveCapability"]["fullFieldHistoricalTrainingReady"] is False
        assert record["sha256"]
        assert all(item["sha256"] for item in manifest["publishedAssets"])
        json.dumps(manifest, allow_nan=False)
        (directory / "archive_manifest.json").write_text(
            json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
        )
        bundle_root = directory / "bundle-output"
        bundle_path = build_bundle(directory, bundle_root)
        assert (bundle_path / "archive_manifest.json").exists()
        bundle = json.loads(
            (bundle_path / "bundle.json").read_text(encoding="utf-8")
        )
        assert bundle["mode"] == "verification"
        assert bundle["rawGribIncluded"] is False


def test_immutable_s3_raw_archive_contract():
    class NotFound(Exception):
        response = {"Error": {"Code": "404"}}

    class MemoryS3:
        def __init__(self):
            self.objects = {}
            self.uploads = 0

        def head_object(self, *, Bucket, Key):
            if (Bucket, Key) not in self.objects:
                raise NotFound()
            return self.objects[(Bucket, Key)]

        def upload_file(self, filename, bucket, key, ExtraArgs):
            self.uploads += 1
            self.objects[(bucket, key)] = {
                "ContentLength": Path(filename).stat().st_size,
                "Metadata": {
                    str(name).lower(): str(value)
                    for name, value in ExtraArgs["Metadata"].items()
                },
                "ETag": "memory-etag",
                "VersionId": "1",
            }

    environment = {
        "ICON_ARCHIVE_MODE": "raw",
        "ICON_ARCHIVE_S3_BUCKET": "private-icon-history",
        "ICON_ARCHIVE_S3_ENDPOINT": "https://example.invalid",
        "ICON_ARCHIVE_S3_ACCESS_KEY_ID": "test-key",
        "ICON_ARCHIVE_S3_SECRET_ACCESS_KEY": "test-secret",
        "ICON_ARCHIVE_S3_PREFIX": "meteo/icon2i",
    }
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "surface.grib"
        source.write_bytes(b"GRIB-data-7777")
        client = MemoryS3()
        archive = RawGribArchive.from_environment(
            run_time="2026-08-30T00:00:00Z", environ=environment, client=client
        )
        record = archive.retain_source(
            path=source, name="surface", role="surface-run",
            source_url="https://example.invalid/surface.grib",
        )
        assert record["key"].startswith("meteo/icon2i/20260830T000000Z/raw/surface-run/")
        assert record["sha256"]
        assert client.uploads == 1
        assert archive.retain_source(
            path=source, name="surface", role="surface-run",
            source_url="https://example.invalid/surface.grib",
        )["key"] == record["key"]
        assert client.uploads == 1
        assert archive.public_status()["kind"] == "private-s3-compatible-raw-grib"

    try:
        RawGribArchive.from_environment(
            run_time="2026-08-30T00:00:00Z",
            environ={"ICON_ARCHIVE_MODE": "raw"},
        )
    except ObjectStoreConfigurationError:
        pass
    else:
        raise AssertionError("configurazione raw incompleta accettata")


if __name__ == "__main__":
    test_observation_semantics()
    test_italian_network_and_live_subset_are_distinct()
    test_native_bilinear_sampling_and_archive()
    test_time_matching_metrics_and_pressure_exclusion()
    test_checksum_manifest_is_truthful()
    test_immutable_s3_raw_archive_contract()
    print("Forecast-observation verification tests passed")
