"""Prove dell'ambiente ICON-2I per le nubi 3D (solo numpy)."""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from meteo_analysis.clouds.environment import (  # noqa: E402
    CloudEnvironmentWriter,
    EnvironmentUnavailable,
    Tile,
    lapse_rate,
    lcl_height_asl_m,
    merge_previous,
    temporal_blend,
)

RUN = datetime(2026, 9, 24, 0, tzinfo=timezone.utc)


def _writer(run=RUN, hours=(0, 1, 2), cape=1000.0):
    lat = np.linspace(48.9, 33.7, 60)  # decrescente, come in process_data
    lon = np.linspace(3.0, 22.0, 80)
    shape = (lat.size, lon.size)
    writer = CloudEnvironmentWriter(run, lat, lon, target_points=1000)
    hsurf = np.zeros(shape)
    hsurf[:10, :] = 2000.0  # le Alpi, a nord
    for h in hours:
        writer.add(h, np.full(shape, 20.0), np.full(shape, 12.0 + 273.15),
                   np.full(shape, cape * h), t500_k=np.full(shape, 20.0 + 273.15 - 33.0),
                   hsurf_m=hsurf)
    return writer


def test_lcl_is_lawrence_above_orography():
    base = lcl_height_asl_m(np.array([25.0]), np.array([15.0]), np.array([400.0]))
    assert np.isclose(base[0], 1650.0)


def test_lapse_rate_is_bounded_and_falls_back_to_standard():
    assert np.isclose(lapse_rate(np.array([288.0]), None, None)[0], 6.5)
    assert np.isclose(lapse_rate(np.array([288.0]), np.array([288.0 - 33.0]), np.array([0.0]))[0],
                      33.0 / 5.574)
    assert lapse_rate(np.array([288.0]), np.array([300.0]), np.array([0.0]))[0] == 4.0


def test_tile_roundtrip_keeps_orientation_and_values():
    writer = _writer()
    tile = writer.tiles[2]
    back = Tile.from_bytes(tile.to_bytes(), tile.valid)
    assert back.latitudes[0] < back.latitudes[-1], "riga 0 = sud"
    assert np.allclose(back.fields["cape"], 2000.0)
    assert np.allclose(back.fields["t2m"], 293.15, atol=0.01)
    # Le Alpi sono a nord: ultima riga, non la prima.
    assert back.fields["hsurf"][-1].mean() > 1500 and back.fields["hsurf"][0].mean() < 1
    assert np.allclose(back.fields["lcl"][0], 1000.0, atol=1)
    assert len(tile.to_bytes()) < 60_000


def test_writer_refuses_incomplete_and_out_of_range_hours():
    writer = _writer(hours=())
    shape = (60, 80)
    assert not writer.add(40, np.zeros(shape), np.zeros(shape), np.zeros(shape))
    assert not writer.add(3, np.zeros(shape), None, np.zeros(shape))


def test_blend_interpolates_extrapolates_and_refuses():
    hours = [RUN + timedelta(hours=h) for h in (0, 1, 2)]
    b = temporal_blend(hours, RUN + timedelta(minutes=80))
    assert b.mode == "interpolated" and abs(b.weight_after - 1 / 3) < 1e-9
    b = temporal_blend(hours, RUN - timedelta(hours=2))
    assert b.mode == "nearest" and b.before == "2026-09-24T00:00:00Z"
    try:
        temporal_blend(hours, RUN - timedelta(hours=4))
    except EnvironmentUnavailable:
        pass
    else:
        raise AssertionError("4 ore fuori dall'ambiente vanno rifiutate")
    # Un buco di sei ore (run saltato) non si interpola attraverso.
    gap = [RUN, RUN + timedelta(hours=6)]
    assert temporal_blend(gap, RUN + timedelta(hours=1)).mode == "nearest"


def test_merge_keeps_past_hours_of_previous_runs_and_prefers_new():
    with tempfile.TemporaryDirectory() as root:
        old_dir, new_dir = os.path.join(root, "old"), os.path.join(root, "new")
        _writer(RUN, hours=range(0, 16), cape=10.0).write(old_dir)
        _writer(RUN + timedelta(hours=12), hours=range(0, 4), cape=500.0).write(new_dir)
        index = merge_previous(new_dir, old_dir)
        valid = [e["valid"] for e in index["hours"]]
        assert valid == sorted(valid)
        assert valid[0] == "2026-09-24T00:00:00Z" and valid[-1] == "2026-09-24T15:00:00Z"
        # Le 12-15 sono del run nuovo: vince il run nuovo.
        twelve = [e for e in index["hours"] if e["valid"] == "2026-09-24T12:00:00Z"][0]
        assert twelve["run"] == "2026-09-24T12:00:00Z"
        with open(os.path.join(new_dir, twelve["file"]), "rb") as handle:
            assert np.allclose(Tile.from_bytes(handle.read(), twelve["valid"]).fields["cape"], 0.0)
        with open(os.path.join(new_dir, "index.json"), encoding="utf-8") as handle:
            assert json.load(handle) == index


def test_merge_forgets_hours_older_than_the_window():
    with tempfile.TemporaryDirectory() as root:
        old_dir, new_dir = os.path.join(root, "old"), os.path.join(root, "new")
        _writer(RUN, hours=(0, 1)).write(old_dir)
        _writer(RUN + timedelta(hours=72), hours=(0,)).write(new_dir)
        index = merge_previous(new_dir, old_dir)
        assert [e["valid"] for e in index["hours"]] == ["2026-09-27T00:00:00Z"]


def test_browser_fixture_matches_the_writer():
    """scripts/tests/fixtures_cloud_env.bin.gz e' la piastrella che legge
    test_nubi_icon.js: se il writer cambia formato, va rigenerata."""
    lat = np.linspace(48.9, 33.7, 24)
    lon = np.linspace(3, 22, 30)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    writer = CloudEnvironmentWriter(RUN, lat, lon, target_points=10_000)
    writer.add(12, 20.0 + 0 * la, 12.0 + 0 * la, 3200.0 * (lo > 12),
               t500_k=20 + 273.15 - 33 + 0 * la, hsurf_m=np.where(la > 46, 1500.0, 0.0))
    fixture = os.path.join(os.path.dirname(__file__), "fixtures_cloud_env.bin.gz")
    with open(fixture, "rb") as handle:
        assert handle.read() == writer.tiles[12].to_bytes(), "fixture del browser da rigenerare"


if __name__ == "__main__":
    for name, function in sorted(list(globals().items())):
        if name.startswith("test_") and callable(function):
            function()
    print("Cloud environment tests passed")
