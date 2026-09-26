"""L'ambiente termodinamico ICON-2I che accompagna le nubi 3D.

Il satellite osserva le nubi ogni dieci minuti; ICON-2I gira due volte al
giorno e pubblica una scadenza ogni ora.  I due orologi non coincidono mai.
Qui si conserva, durante la pipeline del modello, l'unica parte del run che
serve al volume -- base delle nubi, temperatura al suolo, gradiente verticale,
CAPE e orografia -- su una griglia ridotta, una piastrella per ora di
validita'.  Il browser interpola fra le due ore che racchiudono il fotogramma
satellitare che si sta guardando, anche indietro nella timeline.

Il modello non decide MAI dove c'e' una nube: e' un modificatore ambientale.
Presenza, sagoma e cima restano del satellite.

Formato di una piastrella (gzip, little-endian)::

    "NUBA"  u8 versione  u8 campi  u16 nx  u16 ny
    f32 lat_sud  f32 lat_nord  f32 lon_ovest  f32 lon_est   (centri, crescenti)
    per campo: 8 byte nome ASCII, f32 scala, f32 offset
    per campo: ny*nx int16, riga 0 = sud; -32768 = mancante

``index.json`` elenca le ore disponibili con il run che le ha prodotte.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import struct
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

METHOD = "icon2i-cloud-environment-v1"
MAGIC = b"NUBA"
VERSION = 1

# Quota standard della superficie isobarica di 500 hPa (atmosfera ICAO).  Il
# gradiente T2m-T500 si divide per lo spessore fra il suolo e questa quota.
Z500_STANDARD_KM = 5.574
# Gradiente dell'atmosfera standard, usato solo dove T500 manca.
STANDARD_LAPSE_K_PER_KM = 6.5
# Oltre questi limiti il gradiente T2m-T500 non descrive la troposfera libera
# ma un'inversione al suolo (notte, nebbia) o una superadiabatica di pochi
# metri: si limita all'intervallo fisico della colonna.
LAPSE_MIN_K_PER_KM = 4.0
LAPSE_MAX_K_PER_KM = 9.8
# Il run piu' recente esce con 2-4 ore di ritardo e viene sostituito dopo
# dodici: 36 scadenze coprono il satellite con margine anche se un run salta.
MAX_LEAD_HOURS = 36
# Le ore passate dei run precedenti restano per la timeline satellitare.
KEEP_PAST_HOURS = 48
# Il volume vuole un ambiente liscio, non il dettaglio a 2,2 km: circa
# quarantamila punti (9-10 km sull'Italia) e una piastrella da ~150 kB.
TARGET_POINTS = 40_000
# Oltre questa distanza dall'ora disponibile piu' vicina l'ambiente non
# descrive piu' l'istante del satellite e non si usa.
MAX_EXTRAPOLATION_HOURS = 3.0

_NODATA = -32768
# valore = codice * scala + offset
FIELDS = (
    ("lcl", 1.0, 0.0),        # base delle nubi (LCL) sul livello del mare, m
    ("t2m", 0.01, 273.15),    # temperatura a 2 m, K
    ("lapse", 0.001, 0.0),    # gradiente medio suolo-500 hPa, K/km
    ("cape", 0.5, 0.0),       # CAPE (massimo di blocco), J/kg
    ("hsurf", 1.0, 0.0),      # orografia, m
    # --- per il motore d'inferenza delle nubi (tutti facoltativi) ---
    ("cin", 0.5, 0.0),        # |CIN| ML, J/kg
    ("hzero", 1.0, 0.0),      # quota dello zero termico, m
    ("u250", 0.01, 0.0),      # vento a 250 hPa (incudini, cirri), m/s
    ("v250", 0.01, 0.0),
    ("u500", 0.01, 0.0),      # vento a 500 hPa (moto dei sistemi), m/s
    ("v500", 0.01, 0.0),
    ("shu", 0.01, 0.0),       # shear 0-6 km, m/s (inclinazione delle torri)
    ("shv", 0.01, 0.0),
    ("rh850", 0.01, 0.0),     # umidita' relativa, %
    ("rh700", 0.01, 0.0),     # (da T e QV a 700 hPa)
    ("rh500", 0.01, 0.0),
    ("clcl", 0.01, 0.0),      # copertura del modello per piani, %
    ("clcm", 0.01, 0.0),
    ("clch", 0.01, 0.0),
    ("rcon", 0.001, 0.0),     # pioggia convettiva, mm/h (massimo di blocco)
    ("rgsp", 0.001, 0.0),     # pioggia di scala, mm/h
)
# Come si riduce ogni campo facoltativo sulla griglia larga: il massimo per
# cio' che e' piccolo e intenso (una cella convettiva), la media per il resto.
_EXTRA_BLOCK = {"rcon": "max", "cin": "mean"}


def relative_humidity(q_kg_kg, t_k, p_hpa: float) -> np.ndarray:
    """Umidita' relativa (%) da umidita' specifica e temperatura (Bolton)."""
    q = np.asarray(q_kg_kg, dtype=np.float64)
    # Alcuni GRIB pubblicano QV in g/kg: nessuna umidita' specifica reale
    # supera 0,05 kg/kg.
    if np.nanmax(q) > 0.1:
        q = q / 1000.0
    tc = np.asarray(t_k, dtype=np.float64) - 273.15
    e = q * p_hpa / (0.622 + 0.378 * q)
    es = 6.112 * np.exp(17.67 * tc / (tc + 243.5))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(100.0 * e / es, 0.0, 110.0)


class EnvironmentUnavailable(RuntimeError):
    """Nessuna ora ICON-2I abbastanza vicina all'istante del satellite."""


def parse_time(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tile_name(valid: datetime) -> str:
    return valid.astimezone(timezone.utc).strftime("%Y%m%d%H") + ".bin.gz"


def coarsen_factor(ny: int, nx: int, target_points: int = TARGET_POINTS) -> int:
    return max(1, int(math.ceil(math.sqrt(ny * nx / float(target_points)))))


def _block(values, factor: int, how: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if factor <= 1:
        return arr.copy()
    ny, nx = arr.shape
    padded = np.pad(arr, ((0, (-ny) % factor), (0, (-nx) % factor)), constant_values=np.nan)
    blocks = padded.reshape(padded.shape[0] // factor, factor, padded.shape[1] // factor, factor)
    with warnings.catch_warnings():
        # Un blocco tutto NaN (fuori dominio) resta NaN senza avvisi.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if how == "max":
            return np.nanmax(blocks, axis=(1, 3))
        return np.nanmean(blocks, axis=(1, 3))


def _block_axis(values, factor: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if factor <= 1:
        return arr.copy()
    pad = (-arr.size) % factor
    if pad:
        step = arr[-1] - arr[-2]
        arr = np.concatenate([arr, arr[-1] + step * np.arange(1, pad + 1)])
    return arr.reshape(-1, factor).mean(axis=1)


def lapse_rate(t2m_k, t500_k, hsurf_m) -> np.ndarray:
    """Gradiente verticale medio suolo-500 hPa in K/km, limitato al fisico."""
    t2m = np.asarray(t2m_k, dtype=np.float64)
    if t500_k is None:
        return np.full(t2m.shape, STANDARD_LAPSE_K_PER_KM)
    zs = np.zeros_like(t2m) if hsurf_m is None else np.asarray(hsurf_m, float) / 1000.0
    depth = np.maximum(Z500_STANDARD_KM - np.nan_to_num(zs), 1.0)
    with np.errstate(invalid="ignore"):
        gamma = (t2m - np.asarray(t500_k, dtype=np.float64)) / depth
    gamma = np.where(np.isfinite(gamma), gamma, STANDARD_LAPSE_K_PER_KM)
    return np.clip(gamma, LAPSE_MIN_K_PER_KM, LAPSE_MAX_K_PER_KM)


def lcl_height_asl_m(t2m_c, td2m, hsurf_m) -> np.ndarray:
    """Base delle nubi: LCL di Lawrence (125 m per grado di scarto) + suolo."""
    t = np.asarray(t2m_c, dtype=np.float64)
    td = np.asarray(td2m, dtype=np.float64)
    # ICON pubblica il punto di rugiada in kelvin: si riconosce dall'ordine
    # di grandezza, come fa gia' la pipeline dei temporali.
    td = np.where(np.isfinite(td) & (td > 150.0), td - 273.15, td)
    spread = np.maximum(t - np.minimum(td, t), 0.0)
    ground = 0.0 if hsurf_m is None else np.nan_to_num(np.asarray(hsurf_m, float))
    return ground + 125.0 * spread


@dataclass
class Tile:
    valid: datetime
    latitudes: np.ndarray
    longitudes: np.ndarray
    fields: dict

    def to_bytes(self) -> bytes:
        ny, nx = self.fields["lcl"].shape
        body = b""
        present = [spec for spec in FIELDS if spec[0] in self.fields]
        head = MAGIC + struct.pack("<BBHH", VERSION, len(present), nx, ny)
        head += struct.pack("<ffff", float(self.latitudes[0]), float(self.latitudes[-1]),
                            float(self.longitudes[0]), float(self.longitudes[-1]))
        for name, scale, offset in present:
            head += name.encode("ascii").ljust(8, b"\0") + struct.pack("<ff", scale, offset)
            values = np.asarray(self.fields[name], dtype=np.float64)
            codes = np.full(values.shape, _NODATA, dtype="<i2")
            ok = np.isfinite(values)
            codes[ok] = np.clip(np.round((values[ok] - offset) / scale), -32000, 32000)
            body += codes.tobytes()
        return gzip.compress(head + body, compresslevel=9, mtime=0)

    @classmethod
    def from_bytes(cls, data: bytes, valid) -> "Tile":
        raw = gzip.decompress(data)
        if raw[:4] != MAGIC:
            raise ValueError("piastrella ambiente non riconosciuta")
        version, count, nx, ny = struct.unpack_from("<BBHH", raw, 4)
        if version != VERSION:
            raise ValueError(f"versione piastrella {version}")
        s, n, w, e = struct.unpack_from("<ffff", raw, 10)
        pos = 26
        specs = []
        for _ in range(count):
            name = raw[pos:pos + 8].rstrip(b"\0").decode("ascii")
            scale, offset = struct.unpack_from("<ff", raw, pos + 8)
            specs.append((name, scale, offset))
            pos += 16
        fields = {}
        for name, scale, offset in specs:
            codes = np.frombuffer(raw, dtype="<i2", count=nx * ny, offset=pos).reshape(ny, nx)
            values = codes.astype(np.float64) * scale + offset
            values[codes == _NODATA] = np.nan
            fields[name] = values
            pos += 2 * nx * ny
        return cls(parse_time(valid), np.linspace(s, n, ny), np.linspace(w, e, nx), fields)


class CloudEnvironmentWriter:
    """Raccoglie, scadenza per scadenza, l'ambiente del volume durante il run."""

    def __init__(self, run_time, latitudes, longitudes,
                 max_lead_hours: int = MAX_LEAD_HOURS,
                 target_points: int = TARGET_POINTS) -> None:
        lat = np.asarray(latitudes, dtype=np.float64)
        lon = np.asarray(longitudes, dtype=np.float64)
        if lat.ndim != 1 or lon.ndim != 1 or lat.size < 2 or lon.size < 2:
            raise ValueError("griglia lat/lon non regolare")
        self.run_time = parse_time(run_time)
        self.max_lead_hours = int(max_lead_hours)
        self._flip_lat = lat[0] > lat[-1]
        self._flip_lon = lon[0] > lon[-1]
        lat = lat[::-1] if self._flip_lat else lat
        lon = lon[::-1] if self._flip_lon else lon
        self.factor = coarsen_factor(lat.size, lon.size, target_points)
        self.latitudes = _block_axis(lat, self.factor)
        self.longitudes = _block_axis(lon, self.factor)
        self.tiles: dict[int, Tile] = {}

    def _orient(self, values):
        if values is None:
            return None
        arr = np.asarray(values, dtype=np.float64)
        if self._flip_lat:
            arr = arr[::-1, :]
        if self._flip_lon:
            arr = arr[:, ::-1]
        return arr

    @property
    def hours(self) -> list[int]:
        return sorted(self.tiles)

    def add(self, lead_hours: int, t2m_c, td2m, cape, t500_k=None, hsurf_m=None,
            extras=None) -> bool:
        """Aggiunge una scadenza; False se fuori orizzonte o incompleta."""
        lead_hours = int(lead_hours)
        if lead_hours < 0 or lead_hours > self.max_lead_hours or lead_hours in self.tiles:
            return False
        if t2m_c is None or td2m is None or cape is None:
            return False
        t2m_c, td2m, cape = self._orient(t2m_c), self._orient(td2m), self._orient(cape)
        hsurf, t500 = self._orient(hsurf_m), self._orient(t500_k)
        t2m_k = t2m_c + 273.15
        f = self.factor
        fields = {
            "lcl": _block(lcl_height_asl_m(t2m_c, td2m, hsurf), f, "mean"),
            "t2m": _block(t2m_k, f, "mean"),
            "lapse": _block(lapse_rate(t2m_k, t500, hsurf), f, "mean"),
            # Il massimo e non la media: una cella convettiva larga dieci
            # chilometri non deve sparire diluita nel blocco.
            "cape": _block(np.maximum(np.nan_to_num(cape, nan=0.0), 0.0), f, "max"),
            "hsurf": (_block(hsurf, f, "mean") if hsurf is not None
                      else np.zeros((self.latitudes.size, self.longitudes.size))),
        }
        extras = dict(extras or {})
        if extras.get("q700") is not None and extras.get("t700") is not None:
            extras["rh700"] = relative_humidity(extras["q700"], extras["t700"], 700.0)
        if extras.get("shear_u") is not None:
            extras["shu"], extras["shv"] = extras.get("shear_u"), extras.get("shear_v")
        if extras.get("rain_con") is not None:
            extras["rcon"] = extras["rain_con"]
        if extras.get("rain_gsp") is not None:
            extras["rgsp"] = extras["rain_gsp"]
        if extras.get("cin") is not None:
            extras["cin"] = np.abs(np.asarray(extras["cin"], dtype=np.float64))
        known = {spec[0] for spec in FIELDS}
        for name, values in extras.items():
            if name in known and name not in fields and values is not None:
                arr = self._orient(values)
                if arr is None or arr.shape != t2m_k.shape:
                    continue
                fields[name] = _block(arr, f, _EXTRA_BLOCK.get(name, "mean"))
        valid = self.run_time + timedelta(hours=lead_hours)
        self.tiles[lead_hours] = Tile(valid, self.latitudes, self.longitudes, fields)
        return True

    def write(self, directory) -> dict:
        """Scrive piastrelle e indice; restituisce l'indice."""
        if not self.tiles:
            raise ValueError("nessuna scadenza raccolta")
        os.makedirs(directory, exist_ok=True)
        entries = []
        for lead in self.hours:
            tile = self.tiles[lead]
            name = tile_name(tile.valid)
            _write_atomic(os.path.join(directory, name), tile.to_bytes())
            entries.append({"valid": iso(tile.valid), "run": iso(self.run_time),
                            "lead": lead, "file": name})
        index = {"method": METHOD, "latestRun": iso(self.run_time), "hours": entries}
        _write_atomic(os.path.join(directory, "index.json"),
                      json.dumps(index, separators=(",", ":")).encode("utf-8"))
        return index


def _write_atomic(path, data: bytes) -> None:
    partial = f"{path}.part"
    with open(partial, "wb") as handle:
        handle.write(data)
    os.replace(partial, path)


def merge_previous(new_dir, old_dir, keep_past_hours: int = KEEP_PAST_HOURS) -> dict:
    """Porta nel nuovo ambiente le ore passate che il nuovo run non copre.

    La pubblicazione su gh-pages e' un commit orfano: senza questo passo,
    appena esce il run delle 12 le ore 06-11 della timeline satellitare
    perderebbero l'ambiente.  Le ore coperte dal nuovo run vincono sempre;
    si conservano solo quelle entro ``keep_past_hours`` dall'inizio del
    nuovo run.
    """
    with open(os.path.join(new_dir, "index.json"), encoding="utf-8") as handle:
        index = json.load(handle)
    old_index_path = os.path.join(old_dir, "index.json")
    if not os.path.exists(old_index_path):
        return index
    with open(old_index_path, encoding="utf-8") as handle:
        old = json.load(handle)
    have = {entry["valid"] for entry in index["hours"]}
    oldest = parse_time(index["latestRun"]) - timedelta(hours=keep_past_hours)
    for entry in old.get("hours", []):
        if entry["valid"] in have or parse_time(entry["valid"]) < oldest:
            continue
        source = os.path.join(old_dir, entry["file"])
        if not os.path.exists(source):
            continue
        with open(source, "rb") as handle:
            data = handle.read()
        try:
            Tile.from_bytes(data, entry["valid"])
        except Exception:
            continue
        _write_atomic(os.path.join(new_dir, entry["file"]), data)
        index["hours"].append(entry)
        have.add(entry["valid"])
    index["hours"].sort(key=lambda e: e["valid"])
    _write_atomic(os.path.join(new_dir, "index.json"),
                  json.dumps(index, separators=(",", ":")).encode("utf-8"))
    return index


@dataclass
class TemporalBlend:
    """Come l'ambiente e' stato portato all'istante del satellite."""

    before: str
    after: str
    weight_after: float
    distance_hours: float
    mode: str  # "interpolated" | "exact" | "nearest"


def temporal_blend(valid_times, when,
                   max_extrapolation_hours: float = MAX_EXTRAPOLATION_HOURS) -> TemporalBlend:
    """Le due ore che racchiudono ``when`` e il peso lineare della seconda.

    E' la stessa regola che applica il browser (``ambienteAllIstante``):
    dentro l'intervallo si interpola fra ore adiacenti; appena fuori si usa
    l'ora piu' vicina fino a ``max_extrapolation_hours``; oltre si rifiuta.
    Un buco nell'elenco (un run saltato) conta come un fuori intervallo.
    """
    times = sorted({parse_time(v) for v in valid_times})
    if not times:
        raise EnvironmentUnavailable("nessuna ora ICON-2I")
    t = parse_time(when)
    before = max((v for v in times if v <= t), default=None)
    after = min((v for v in times if v >= t), default=None)
    if before is not None and after is not None:
        span = (after - before).total_seconds() / 3600.0
        if span == 0:
            return TemporalBlend(iso(before), iso(after), 0.0, 0.0, "exact")
        if span <= 3.0:
            weight = (t - before).total_seconds() / 3600.0 / span
            distance = min(weight, 1 - weight) * span
            return TemporalBlend(iso(before), iso(after), weight, distance, "interpolated")
    candidates = [v for v in (before, after) if v is not None]
    nearest = min(candidates, key=lambda v: abs((v - t).total_seconds()))
    distance = abs((nearest - t).total_seconds()) / 3600.0
    if distance > max_extrapolation_hours:
        raise EnvironmentUnavailable(f"ora ICON-2I piu' vicina a {distance:.1f} h")
    return TemporalBlend(iso(nearest), iso(nearest), 0.0, distance, "nearest")
