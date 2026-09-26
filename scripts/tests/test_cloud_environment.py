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


def test_inference_fields_are_optional_and_derived():
    lat = np.linspace(48.9, 33.7, 60)
    lon = np.linspace(3.0, 22.0, 80)
    shape = (lat.size, lon.size)
    writer = CloudEnvironmentWriter(RUN, lat, lon, target_points=1000)
    extras = {
        "u250": np.full(shape, 30.0), "v250": np.full(shape, -10.0),
        "rh850": np.full(shape, 95.0), "clcl": np.full(shape, 80.0),
        "t700": np.full(shape, 273.15), "q700": np.full(shape, 0.0035),
        "rain_con": np.zeros(shape), "cin": np.full(shape, -40.0),
        "rh500": None,
    }
    assert writer.add(1, np.full(shape, 20.0), np.full(shape, 12.0), np.zeros(shape),
                      extras=extras)
    back = Tile.from_bytes(writer.tiles[1].to_bytes(), writer.tiles[1].valid)
    assert np.allclose(back.fields["u250"], 30.0) and np.allclose(back.fields["v250"], -10.0)
    assert np.allclose(back.fields["cin"], 40.0), "CIN in modulo"
    # QV 3,5 g/kg a 700 hPa e 0 C: circa 60% di umidita' relativa.
    assert 55 < float(np.nanmean(back.fields["rh700"])) < 65
    assert "rh500" not in back.fields and "clch" not in back.fields
    # Senza campi facoltativi la piastrella resta quella di prima.
    assert set(_writer().tiles[1].fields) == {"lcl", "t2m", "lapse", "cape", "hsurf"}


def test_vertical_profile_depth_stability_and_moisture():
    """Il profilo verticale dai livelli ICON-2I: una colonna instabile ha una
    nube profonda fino al livello di equilibrio, una colonna stabile e secca
    una nube bassa, uno strato umido stabile lo spessore dello strato."""
    lat = np.linspace(40.0, 38.0, 3)
    lon = np.linspace(13.0, 16.0, 3)
    shape = (lat.size, lon.size)
    ones = np.ones(shape)

    def tile(t2m, td, t700, t500, t250, rh, q700=0.001):
        writer = CloudEnvironmentWriter(RUN, lat, lon, target_points=100)
        assert writer.add(3, t2m * ones, td * ones, np.zeros(shape), t500_k=t500 * ones,
                          hsurf_m=np.zeros(shape),
                          extras={"t700": t700 * ones, "t250": t250 * ones,
                                  "q700": q700 * ones,
                                  "rh850": rh * ones, "rh500": rh * ones})
        return Tile.from_bytes(writer.tiles[3].to_bytes(), writer.tiles[3].valid).fields

    # Estate, suolo 30/22 C, 700 hPa +4 C, 500 hPa -16 C, 250 hPa -48 C:
    # CAPE abbondante, la particella galleggia oltre i 250 hPa.
    instabile = tile(30.0, 22.0, 277.15, 257.15, 225.15, 60.0)
    # Stessa superficie, ma aria calda in quota (inversione di subsidenza).
    stabile = tile(30.0, 22.0, 288.15, 268.15, 235.15, 40.0)
    # Autunno: suolo 14/13 C quasi saturo, stabile ma umido fino a 500 hPa.
    strato = tile(14.0, 13.0, 275.15, 257.15, 222.15, 92.0, q700=0.0058)
    d_i, d_s, d_u = (float(np.nanmean(f["depth"])) / 1000 for f in (instabile, stabile, strato))
    assert d_i > 8.0, f"torre convettiva troppo bassa: {d_i:.1f} km"
    assert d_s < 2.0, f"colonna stabile e secca con nube alta: {d_s:.1f} km"
    assert 3.5 < d_u < 6.0, f"strato umido fino a 500 hPa: {d_u:.1f} km"
    # 700-500 hPa: (4 + 16) K su 2,56 km = 7,8 K/km.
    assert abs(float(np.nanmean(instabile["stab"])) - 20.0 / 2.562) < 0.05
    assert float(np.nanmean(strato["rhmid"])) > 85.0, "umidita' media dello strato"
    assert float(np.nanmean(stabile["rhmid"])) < 45.0


def test_icon_eu_levels_are_interpolated_and_written():
    """La copertura per livello di ICON-EU (DWD) passa dalla sua griglia a
    quella di ICON-2I e finisce nella piastrella come c<livello> (%)."""
    from datetime import timedelta
    from meteo_analysis.clouds.icon_eu import IconEuCloudProfile, clc_url, field_name

    assert field_name(850) == "c850"
    url = clc_url(datetime(2026, 9, 26, 12, tzinfo=timezone.utc), 7, 850)
    assert url.endswith("/12/clc/icon-eu_europe_regular-lat-lon_pressure-level_2026092612_007_850_CLC.grib2.bz2")
    profile = IconEuCloudProfile((36.0, 40.0), (12.0, 16.0))
    profile.latitudes = np.arange(35.8, 40.3, 0.0625)
    profile.longitudes = np.arange(11.8, 16.3, 0.0625)
    la, lo = np.meshgrid(profile.latitudes, profile.longitudes, indexing="ij")
    valid = RUN + timedelta(hours=2)
    # 850 hPa: coltre piena a ovest di 14 E; 500 hPa: sereno; un buco mancante.
    c850 = np.where(lo < 14.0, 100, 0).astype(np.uint8)
    c500 = np.zeros_like(c850)
    c500[0, 0] = 255
    profile.data[valid] = {850: c850, 500: c500}
    lat = np.linspace(39.5, 36.5, 12)
    lon = np.linspace(12.5, 15.5, 16)
    levels = profile.levels_at(valid, lat, lon)
    assert set(levels) == {850, 500}
    assert np.allclose(levels[850][:, lon < 13.8], 100) and np.allclose(levels[850][:, lon > 14.2], 0)
    assert np.nanmax(levels[500]) == 0
    assert profile.levels_at(RUN, lat, lon) == {}
    # La serie a parte per il browser: cloud_eu/, stesso formato NUBA e indice.
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as cartella:
        profile.run = RUN - timedelta(hours=3)
        index = profile.write(cartella, RUN)
        assert index["method"] == "icon-eu-cloud-profile-v1"
        assert index["hours"][0]["lead"] == 2 and index["hours"][0]["run"].startswith("2026")
        with open(os.path.join(cartella, "index.json"), encoding="utf-8") as handle:
            assert json.load(handle)["hours"][0]["file"] == index["hours"][0]["file"]
        with open(os.path.join(cartella, index["hours"][0]["file"]), "rb") as handle:
            back = Tile.from_bytes(handle.read(), valid).fields
    assert set(back) == {"c850", "c500"}
    assert float(np.nanmax(back["c850"])) == 100.0 and float(np.nanmin(back["c850"])) == 0.0
    assert np.isnan(back["c500"][0, 0]), "il valore mancante resta mancante"


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
