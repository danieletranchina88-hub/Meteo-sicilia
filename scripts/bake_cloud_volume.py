#!/usr/bin/env python3
"""Cuoce la texture RGBA del volume delle nubi: EUMETSAT + ICON-2I.

Prende l'ultimo fotogramma MTG disponibile su EUMETView (IR 10,5 um e VIS
0,6 um) con la maschera nubi MSG dello stesso istante, porta l'ambiente
ICON-2I del run piu' recente all'istante esatto del satellite e scrive

    <output-dir>/volume.png   RGBA8: cima, densita', base, convezione
    <output-dir>/volume.json  metadati: dominio, istanti, pesi temporali

Uscita 0 se la texture e' stata scritta, 2 se non c'era nulla da fare
(satellite o ambiente indisponibili): il workflow non deve fallire per
questo, deve lasciare in linea la texture precedente.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meteo_analysis.clouds.environment import (  # noqa: E402
    CloudEnvironment,
    EnvironmentUnavailable,
)
from meteo_analysis.clouds.volume_texture import (  # noqa: E402
    DAY_COS_ZENITH,
    MercatorGrid,
    SatelliteFrame,
    brightness_temperature_from_grey,
    fuse,
    solar_cos_zenith,
    write_texture,
)

WMS = "https://view.eumetsat.int/geoserver/wms"
# Gli stessi prodotti e le stesse latenze misurate che usa la pagina.
PRODUCTS = {
    "ir": {"layer": "mtg_fd:ir105_hrfi", "slot_min": 10, "latency_min": 22},
    "vis": {"layer": "mtg_fd:vis06_hrfi", "slot_min": 10, "latency_min": 22},
    "clm": {"layer": "msg_fes:clm", "slot_min": 15, "latency_min": 20},
}
USER_AGENT = "meteo-sicilia-cloud-volume/1 (+https://github.com/danieletranchina88-hub/meteo)"


def slot(when: datetime, minutes: int) -> datetime:
    epoch = int(when.timestamp()) // (minutes * 60) * (minutes * 60)
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def iso_ms(when: datetime) -> str:
    return when.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def fetch_rgba(session, layer: str, grid: MercatorGrid, when: datetime):
    """Un GetMap PNG come array (H, W, 4); None se lo slot non c'e'."""
    from PIL import Image

    bbox = ",".join(f"{v:.1f}" for v in grid.bbox_3857())
    params = {
        "service": "WMS", "version": "1.1.1", "request": "GetMap", "styles": "",
        "layers": layer, "srs": "EPSG:3857", "format": "image/png", "transparent": "true",
        "bbox": bbox, "width": grid.width, "height": grid.height, "time": iso_ms(when),
    }
    for attempt in range(3):
        try:
            response = session.get(WMS, params=params, timeout=90)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if response.status_code == 429:
            time.sleep(2 * (attempt + 1))
            continue
        # Uno slot non ancora pubblicato torna come ServiceException XML.
        if response.status_code != 200 or not response.headers.get(
                "content-type", "").startswith("image/png"):
            return None
        image = Image.open(io.BytesIO(response.content)).convert("RGBA")
        arr = np.asarray(image)
        if arr.shape[:2] != (grid.height, grid.width):
            return None
        return arr
    return None


def fetch_frame(grid: MercatorGrid, requested: datetime | None, back_slots: int) -> SatelliteFrame | None:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    ir = PRODUCTS["ir"]
    now = datetime.now(timezone.utc)
    start = slot(requested, ir["slot_min"]) if requested else slot(
        now - timedelta(minutes=ir["latency_min"]), ir["slot_min"])
    for back in range(back_slots + 1):
        when = start - timedelta(minutes=ir["slot_min"] * back)
        ir_rgba = fetch_rgba(session, ir["layer"], grid, when)
        if ir_rgba is None or not (ir_rgba[..., 3] > 127).any():
            continue
        bt = brightness_temperature_from_grey(ir_rgba[..., 0], ir_rgba[..., 3])
        sources = {"ir": {"layer": ir["layer"], "time": iso_ms(when)}}

        # La maschera MSG dello slot che contiene quello MTG: MSG passa
        # sull'Europa a fine scansione, pochi minuti dopo l'orario nominale.
        clm_time = slot(when, PRODUCTS["clm"]["slot_min"])
        mask = None
        for clm_when in (clm_time, clm_time - timedelta(minutes=15)):
            clm = fetch_rgba(session, PRODUCTS["clm"]["layer"], grid, clm_when)
            if clm is not None and (clm[..., 3] > 127).any():
                # Bianco = nube, verde/blu = sereno: il rosso e' la frazione.
                mask = np.where(clm[..., 3] > 127, clm[..., 0] / 255.0, 0.0)
                sources["clm"] = {"layer": PRODUCTS["clm"]["layer"], "time": iso_ms(clm_when)}
                break

        vis = None
        mu = solar_cos_zenith(when, grid.latitudes, grid.longitudes)
        if float(mu.max()) > DAY_COS_ZENITH[0]:
            vis_rgba = fetch_rgba(session, PRODUCTS["vis"]["layer"], grid, when)
            if vis_rgba is not None:
                vis = np.where(vis_rgba[..., 3] > 127, vis_rgba[..., 0] / 255.0, np.nan)
                sources["vis"] = {"layer": PRODUCTS["vis"]["layer"], "time": iso_ms(when)}
        return SatelliteFrame(when, bt, vis, mask, sources)
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--environment", required=True,
                        help="cloud_environment.npz scritto da process_data.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--time", help="istante satellite ISO (default: ultimo disponibile)")
    parser.add_argument("--back-slots", type=int, default=6)
    args = parser.parse_args(argv)

    if not os.path.exists(args.environment):
        print(f"Ambiente ICON-2I assente ({args.environment}): nessuna texture.")
        return 2
    environment = CloudEnvironment.load(args.environment)
    grid = MercatorGrid.for_domain(args.width)
    requested = None
    if args.time:
        requested = datetime.fromisoformat(args.time.replace("Z", "+00:00"))

    frame = fetch_frame(grid, requested, args.back_slots)
    if frame is None:
        print("Nessun fotogramma IR EUMETSAT disponibile: nessuna texture.")
        return 2
    try:
        fields, blend = environment.at(frame.time)
    except EnvironmentUnavailable as error:
        print(f"Ambiente ICON-2I troppo lontano dal satellite: {error}")
        return 2
    fields["latitudes"] = environment.latitudes
    fields["longitudes"] = environment.longitudes

    texture = fuse(grid, frame, fields, blend, environment.run_time)
    os.makedirs(args.output_dir, exist_ok=True)
    write_texture(texture,
                  os.path.join(args.output_dir, "volume.png"),
                  os.path.join(args.output_dir, "volume.json"))
    stats = texture.metadata["stats"]
    print(
        f"Volume nubi {texture.metadata['satelliteTime']} · run {texture.metadata['model']['runTime']} "
        f"({blend.mode}, +{blend.lead_hours:.2f} h) · nubi {stats['cloudyPct']}% · "
        f"cima max {stats['topMaxKm']} km · maschera {texture.metadata['mask']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
