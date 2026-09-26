"""La copertura nuvolosa per livelli di ICON-EU (DWD, dati aperti).

ICON-2I su MeteoHub pubblica sei livelli isobarici e solo la copertura per
piani (bassa, media, alta).  ICON-EU, il modello da cui ICON-2I prende i bordi
nella catena europea del DWD, pubblica invece la frazione di nube CLC su
venti livelli isobarici (1000-50 hPa) a 0,0625 gradi (circa 6,5 km), senza
registrazione, su https://opendata.dwd.de/weather/nwp/icon-eu/grib/.

E' la struttura verticale che il satellite non vede: quanti strati ci sono,
a che quota, con che copertura -- anche sotto una coltre alta.  Qui si
scaricano i livelli fino a 200 hPa per le ore del run ICON-2I, si ritagliano
sul dominio del volume e si interpolano sulla griglia ICON-2I.

Tutto facoltativo: un file che manca lascia il livello assente, un run non
ancora pubblicato fa ripiegare sul run precedente (il DWD gira ogni 3 ore).
"""

from __future__ import annotations

import bz2
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import requests

BASE_URL = "https://opendata.dwd.de/weather/nwp/icon-eu/grib"
# Dal suolo alla tropopausa: sotto 200 hPa (11,8 km) le nubi del nostro volume.
LEVELS = (1000, 950, 925, 900, 875, 850, 825, 800, 775, 700, 600, 500, 400, 300, 250, 200)
# Quote dell'atmosfera standard ICAO dei livelli, km: le usa anche il browser.
LEVEL_HEIGHT_KM = {
    1000: 0.111, 950: 0.540, 925: 0.762, 900: 0.988, 875: 1.220, 850: 1.457,
    825: 1.700, 800: 1.949, 775: 2.204, 700: 3.012, 600: 4.206, 500: 5.574,
    400: 7.185, 300: 9.164, 250: 10.363, 200: 11.784,
}
# Il DWD pubblica un run ogni 3 ore; si risale al massimo di 12.
RUN_STEP_HOURS = 3
MAX_RUN_LOOKBACK_HOURS = 12
MAX_STEP_HOURS = 120
WORKERS = 8
USER_AGENT = "MeteoHub-Mobile-Synoptic/1.0 (nubi 3D)"


def field_name(level: int) -> str:
    """Nome del campo nella piastrella ambiente (8 caratteri al massimo)."""
    return f"c{int(level)}"


def clc_url(run: datetime, step: int, level: int) -> str:
    return (f"{BASE_URL}/{run:%H}/clc/icon-eu_europe_regular-lat-lon_pressure-level_"
            f"{run:%Y%m%d%H}_{int(step):03d}_{int(level)}_CLC.grib2.bz2")


def decode_regular_grib(data: bytes):
    """Valori, latitudini e longitudini (crescenti) di un GRIB2 regular_ll."""
    import eccodes

    gid = eccodes.codes_new_from_message(data)
    try:
        ni = eccodes.codes_get(gid, "Ni")
        nj = eccodes.codes_get(gid, "Nj")
        lat0 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
        lat1 = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
        lon0 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
        lon1 = eccodes.codes_get(gid, "longitudeOfLastGridPointInDegrees")
        values = eccodes.codes_get_values(gid).reshape(nj, ni).astype(np.float32)
        missing = eccodes.codes_get(gid, "missingValue")
    finally:
        eccodes.codes_release(gid)
    values[values == missing] = np.nan
    if lon0 > 180:
        lon0 -= 360
    if lon1 > 180:
        lon1 -= 360
    lats = np.linspace(lat0, lat1, nj)
    lons = np.linspace(lon0, lon1, ni)
    if lats[0] > lats[-1]:
        lats, values = lats[::-1], values[::-1, :]
    return values, lats, lons


class IconEuCloudProfile:
    """CLC per livello e per ora di validita', ritagliata sul dominio."""

    def __init__(self, lat_bounds, lon_bounds, margin_deg: float = 0.2) -> None:
        self.south, self.north = min(lat_bounds) - margin_deg, max(lat_bounds) + margin_deg
        self.west, self.east = min(lon_bounds) - margin_deg, max(lon_bounds) + margin_deg
        self.latitudes = None
        self.longitudes = None
        # valid time -> livello -> copertura % (uint8, 255 = mancante)
        self.data: dict[datetime, dict[int, np.ndarray]] = {}
        self.run: datetime | None = None
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    # --- scaricamento --------------------------------------------------------
    def _exists(self, url: str) -> bool:
        try:
            r = self.session.head(url, timeout=(10, 30), allow_redirects=True)
            return r.status_code == 200
        except Exception:
            return False

    def choose_run(self, target_run: datetime, first_lead: int = 0):
        """Il run ICON-EU piu' recente, non successivo a quello ICON-2I, che
        ha gia' pubblicato l'ora di validita' iniziale."""
        base = target_run.replace(minute=0, second=0, microsecond=0)
        base -= timedelta(hours=base.hour % RUN_STEP_HOURS)
        for back in range(0, MAX_RUN_LOOKBACK_HOURS + 1, RUN_STEP_HOURS):
            run = base - timedelta(hours=back)
            step = int((target_run - run).total_seconds() // 3600) + int(first_lead)
            if 0 <= step <= MAX_STEP_HOURS and self._exists(clc_url(run, step, LEVELS[-1])):
                return run
        return None

    def _fetch(self, run: datetime, step: int, level: int):
        url = clc_url(run, step, level)
        for _ in range(3):
            try:
                r = self.session.get(url, timeout=(15, 120))
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return decode_regular_grib(bz2.decompress(r.content))
            except Exception:
                continue
        return None

    def download(self, target_run: datetime, leads, workers: int = WORKERS) -> int:
        """Scarica le ore ``target_run + lead``; restituisce i campi letti."""
        leads = sorted({int(v) for v in leads})
        if not leads:
            return 0
        run = self.choose_run(target_run, leads[0])
        if run is None:
            return 0
        self.run = run
        offset = int((target_run - run).total_seconds() // 3600)
        jobs = [(lead, level) for lead in leads for level in LEVELS
                if 0 <= lead + offset <= MAX_STEP_HOURS]

        def one(job):
            lead, level = job
            return job, self._fetch(run, lead + offset, level)

        count = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for (lead, level), result in pool.map(one, jobs):
                if result is None:
                    continue
                values, lats, lons = result
                jlat = (lats >= self.south) & (lats <= self.north)
                jlon = (lons >= self.west) & (lons <= self.east)
                if not jlat.any() or not jlon.any():
                    continue
                if self.latitudes is None:
                    self.latitudes, self.longitudes = lats[jlat], lons[jlon]
                crop = values[np.ix_(jlat, jlon)]
                if crop.shape != (self.latitudes.size, self.longitudes.size):
                    continue
                packed = np.where(np.isfinite(crop), np.clip(np.round(crop), 0, 100), 255).astype(np.uint8)
                valid = target_run + timedelta(hours=lead)
                self.data.setdefault(valid, {})[level] = packed
                count += 1
        return count

    # --- lettura ---------------------------------------------------------------
    def levels_at(self, valid: datetime, lat, lon) -> dict:
        """Copertura % per livello sulla griglia (lat, lon) data, o {}."""
        found = self.data.get(valid)
        if not found or self.latitudes is None:
            return {}
        from scipy.interpolate import RegularGridInterpolator

        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        la, lo = np.meshgrid(lat, lon, indexing="ij")
        points = np.stack([np.clip(la, self.latitudes[0], self.latitudes[-1]),
                           np.clip(lo, self.longitudes[0], self.longitudes[-1])], axis=-1)
        out = {}
        for level, packed in found.items():
            values = packed.astype(np.float32)
            values[packed == 255] = np.nan
            interp = RegularGridInterpolator((self.latitudes, self.longitudes), values,
                                             bounds_error=False, fill_value=np.nan)
            out[level] = interp(points)
        return out
