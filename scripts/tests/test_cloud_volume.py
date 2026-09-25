"""Prove della fusione EUMETSAT + ICON-2I per le nubi in volume.

Solo numpy (e Pillow se c'e'): girano anche nel workflow di cottura, che non
installa eccodes.
"""

import os
import sys
import zlib
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from meteo_analysis.clouds.environment import (  # noqa: E402
    CloudEnvironment,
    CloudEnvironmentWriter,
    EnvironmentUnavailable,
    lcl_height_asl_m,
    resample_rectilinear,
    temporal_blend,
)
from meteo_analysis.clouds.volume_texture import (  # noqa: E402
    MIN_THICKNESS_KM,
    SCALE_KM,
    MercatorGrid,
    SatelliteFrame,
    brightness_temperature_from_grey,
    cloud_top_height_km,
    decode_rgba,
    extend_outside,
    fuse,
    png_bytes,
)

RUN = datetime(2026, 9, 24, 0, tzinfo=timezone.utc)


def test_temporal_blend_interpolates_between_bracketing_hours():
    blend = temporal_blend(RUN, RUN + timedelta(hours=12, minutes=40), [11, 12, 13, 14])
    assert blend.mode == "interpolated"
    assert (blend.hour_before, blend.hour_after) == (12, 13)
    assert abs(blend.weight_after - 40 / 60) < 1e-9


def test_temporal_blend_uses_nearest_just_outside_and_refuses_far():
    blend = temporal_blend(RUN, RUN - timedelta(hours=1), [0, 1, 2])
    assert blend.mode == "nearest" and blend.hour_before == 0
    assert abs(blend.distance_hours - 1.0) < 1e-9
    try:
        temporal_blend(RUN, RUN + timedelta(hours=10), [0, 1, 2])
    except EnvironmentUnavailable:
        return
    raise AssertionError("un ambiente a 8 ore dall'ultima scadenza va rifiutato")


def test_lcl_is_lawrence_above_orography_and_accepts_kelvin_dewpoint():
    base = lcl_height_asl_m(np.array([25.0]), np.array([15.0 + 273.15]), np.array([400.0]))
    assert np.isclose(base[0], 400.0 + 1250.0)


def _environment():
    lat = np.linspace(48.9, 33.7, 40)  # decrescente, come in process_data
    lon = np.linspace(3.0, 22.0, 50)
    writer = CloudEnvironmentWriter(RUN, lat, lon, target_points=500)
    shape = (lat.size, lon.size)
    for hour, cape in ((12, 0.0), (13, 3200.0)):
        writer.add(
            hour,
            np.full(shape, 25.0),
            np.full(shape, 17.0),
            np.full(shape, cape),
            t500_k=np.full(shape, 25.0 + 273.15 - 36.0),
            hsurf_m=np.zeros(shape),
        )
    return CloudEnvironment.from_bytes(writer.to_bytes())


def test_environment_roundtrip_and_time_interpolation():
    env = _environment()
    assert env.hours == [12, 13]
    assert env.latitudes[0] < env.latitudes[-1], "latitudini riportate crescenti"
    fields, blend = env.at(RUN + timedelta(hours=12, minutes=30))
    assert blend.mode == "interpolated"
    assert np.allclose(fields["cape"], 1600.0, atol=1.0)
    assert np.allclose(fields["lcl_asl_m"], 1000.0, atol=1.0)
    assert np.allclose(fields["lapse_k_km"], 36.0 / 5.574, atol=0.01)


def test_resample_is_nan_outside_model_domain():
    src = np.arange(12.0).reshape(3, 4)
    out = resample_rectilinear(src, [0, 1, 2], [0, 1, 2, 3], np.array([0.5, 5.0]),
                               np.array([1.5, -1.0]))
    assert np.isclose(out[0, 0], 0.5 * (5.0 + 6.0) / 2 + 0.5 * (1.0 + 2.0) / 2)
    assert np.isnan(out[1, 0]) and np.isnan(out[0, 1])


def test_ir_grey_scale_follows_official_legend():
    bt = brightness_temperature_from_grey(np.array([254.0, 1.0]))
    assert np.allclose(bt - 273.15, [-73.0, 30.0], atol=0.2)
    assert np.isnan(brightness_temperature_from_grey(np.array([128.0]), np.array([0]))[0])


def test_cloud_top_climbs_the_model_lapse_rate_and_overshoots():
    top = cloud_top_height_km(np.array([288.0 - 6.5 * 5.0]), 288.0, 6.5, 0.0)
    assert np.isclose(top[0], 5.0)
    t_tropo = 288.0 - 6.5 * 12.0
    over = cloud_top_height_km(np.array([t_tropo - 7.0]), 288.0, 6.5, 0.0)
    assert np.isclose(over[0], 13.0), "7 K sotto la tropopausa = 1 km di sfondamento"


def _frame(grid, when):
    h, w = grid.height, grid.width
    bt = np.full((h, w), 297.0)  # sereno: temperatura del suolo
    mask = np.zeros((h, w))
    vis = np.full((h, w), 0.08)
    # Colonna spessa e fredda a sinistra, velo alto e sottile a destra.
    bt[:, : w // 3] = 215.0
    mask[:, : w // 3] = 1.0
    vis[:, : w // 3] = 0.8
    bt[:, 2 * w // 3:] = 235.0
    mask[:, 2 * w // 3:] = 1.0
    vis[:, 2 * w // 3:] = 0.14
    return SatelliteFrame(when, bt, vis, mask, {"ir": {"layer": "test"}})


def _fused(minutes):
    env = _environment()
    grid = MercatorGrid(10.0, 36.0, 16.0, 40.0, 90)
    when = RUN + timedelta(hours=12, minutes=minutes)
    fields, blend = env.at(when)
    fields["latitudes"] = env.latitudes
    fields["longitudes"] = env.longitudes
    return grid, fuse(grid, _frame(grid, when), fields, blend, env.run_time)


def test_satellite_mask_is_exact():
    grid, texture = _fused(59)
    w = grid.width
    clear = texture.rgba[:, w // 3 + 3: 2 * w // 3 - 3]
    assert (clear[..., 1] == 0).all(), "dove il satellite vede sereno G deve essere zero"
    assert (texture.rgba[:, : w // 3 - 2, 1] > 0).all()


def test_channels_follow_the_physics():
    grid, texture = _fused(59)
    w = grid.width
    thick = decode_rgba(texture.rgba[:, 2: w // 3 - 2])
    veil = decode_rgba(texture.rgba[:, 2 * w // 3 + 2: -2])
    assert thick["top_km"].mean() > veil["top_km"].mean() > 5.0
    assert thick["density"].mean() > veil["density"].mean()
    # CAPE alta: la colonna spessa scende fino all'LCL, il velo no.
    assert np.allclose(thick["base_km"], 1.0, atol=0.15)
    assert veil["base_km"].mean() > 3.0
    assert (texture.base_km[texture.density > 0] <= texture.top_km[texture.density > 0]
            - MIN_THICKNESS_KM + 1e-9).all()
    assert thick["convective"].mean() > 0.5


def test_convective_channel_grows_with_cape_between_hours():
    _, early = _fused(0)
    _, late = _fused(59)
    assert early.convective.max() < 0.05
    assert late.convective.max() > early.convective.max() + 0.5


def test_extend_outside_keeps_inside_and_fills_neighbours():
    values = np.zeros((5, 9))
    inside = np.zeros((5, 9), dtype=bool)
    values[:, :3] = 8.0
    inside[:, :3] = True
    out = extend_outside(values, inside, radii=(2,))
    assert (out[:, :3] == 8.0).all()
    assert np.allclose(out[:, 3:5], 8.0)
    assert (out[:, 6:] == 0.0).all()


def test_png_is_valid_rgba_and_lossless():
    rgba = np.random.default_rng(3).integers(0, 256, (7, 5, 4), dtype=np.uint8)
    rgba[..., 3] = 0  # l'alfa e' un dato: zero non deve cancellare RGB
    data = png_bytes(rgba)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    try:
        from PIL import Image
        import io

        back = np.asarray(Image.open(io.BytesIO(data)))
        assert back.shape == rgba.shape and (back == rgba).all()
    except ImportError:
        assert zlib.decompress(data[data.index(b"IDAT") + 4:-16])


def test_scale_constant_matches_client():
    client = open(os.path.join(os.path.dirname(__file__), "..", "..", "nubi_fusione.js"),
                  encoding="utf-8").read()
    assert "meta.scaleKm || 16" in client and SCALE_KM == 16.0


if __name__ == "__main__":
    for name, function in sorted(list(globals().items())):
        if name.startswith("test_") and callable(function):
            function()
    print("Cloud volume fusion tests passed")
