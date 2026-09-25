"""L'ambiente termodinamico ICON-2I che accompagna il volume delle nubi.

Il satellite osserva una nube ogni dieci minuti; ICON-2I gira due volte al
giorno e pubblica una scadenza ogni ora.  I due orologi non coincidono mai, e
il volume non puo' aspettare il run successivo: qui si conserva, durante la
pipeline del modello, l'unica parte del run che serve al volume -- base delle
nubi, temperatura al suolo, gradiente verticale e CAPE -- su una griglia
ridotta, per poterla poi interpolare all'istante esatto del fotogramma
satellitare.

Il modello non decide MAI dove c'e' una nube: e' un modificatore ambientale.
La presenza resta del satellite (vedi ``volume_texture``).
"""

from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

# Quota standard della superficie isobarica di 500 hPa (atmosfera ICAO).  Il
# gradiente T2m-T500 si divide per lo spessore fra il suolo e questa quota: su
# una colonna reale 500 hPa oscilla di qualche centinaio di metri, che sul
# gradiente vale pochi decimi di grado per chilometro.
Z500_STANDARD_KM = 5.574
# Gradiente dell'atmosfera standard, usato solo dove T500 manca.
STANDARD_LAPSE_K_PER_KM = 6.5
# Oltre questi limiti il gradiente T2m-T500 non descrive la troposfera libera
# ma un'inversione al suolo (notte, nebbia) o una superadiabatica di pochi
# metri: si limita all'intervallo fisico della colonna.
LAPSE_MIN_K_PER_KM = 4.0
LAPSE_MAX_K_PER_KM = 9.8
# Il run piu' recente viene pubblicato con 2-4 ore di ritardo e sostituito
# dopo dodici: 36 scadenze coprono il satellite con ampio margine anche
# quando un run salta.
MAX_LEAD_HOURS = 36
# Il volume vuole un ambiente liscio, non il dettaglio a 2,2 km: circa
# sessantamila punti (8 km sull'Italia) bastano e tengono il file piccolo.
TARGET_POINTS = 60_000
# Oltre questa distanza dall'ultima scadenza disponibile l'ambiente non
# descrive piu' l'istante del satellite e non si usa.
MAX_EXTRAPOLATION_HOURS = 3.0

# Quantizzazione int16: valore = codice * scala + offset.
_NODATA = -32768
_QUANT = {
    "lcl_asl_m": (1.0, 0.0),
    "t2m_k": (0.01, 273.15),
    "lapse_k_km": (0.001, 0.0),
    "cape": (0.5, 0.0),
    "hsurf_m": (1.0, 0.0),
}
TIME_VARYING = ("lcl_asl_m", "t2m_k", "lapse_k_km", "cape")


class EnvironmentUnavailable(RuntimeError):
    """Nessuna scadenza ICON-2I abbastanza vicina all'istante del satellite."""


def _parse_time(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _quantize(name: str, values: np.ndarray) -> np.ndarray:
    scale, offset = _QUANT[name]
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, _NODATA, dtype=np.int16)
    finite = np.isfinite(arr)
    out[finite] = np.clip(np.round((arr[finite] - offset) / scale), -32000, 32000)
    return out


def _dequantize(name: str, codes: np.ndarray) -> np.ndarray:
    scale, offset = _QUANT[name]
    arr = codes.astype(np.float64) * scale + offset
    arr[codes == _NODATA] = np.nan
    return arr


def coarsen_factor(ny: int, nx: int, target_points: int = TARGET_POINTS) -> int:
    return max(1, int(math.ceil(math.sqrt(ny * nx / float(target_points)))))


def _block(values, factor: int, how: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if factor <= 1:
        return arr.copy()
    ny, nx = arr.shape
    py, px = (-ny) % factor, (-nx) % factor
    padded = np.pad(arr, ((0, py), (0, px)), constant_values=np.nan)
    blocks = padded.reshape(
        padded.shape[0] // factor, factor, padded.shape[1] // factor, factor
    )
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
        step = arr[-1] - arr[-2] if arr.size > 1 else 0.0
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


def lcl_height_asl_m(t2m_c, td2m_c, hsurf_m) -> np.ndarray:
    """Base delle nubi: LCL di Lawrence (125 m per grado di scarto) + suolo."""
    t = np.asarray(t2m_c, dtype=np.float64)
    td = np.asarray(td2m_c, dtype=np.float64)
    # ICON pubblica il punto di rugiada in kelvin: si riconosce dall'ordine
    # di grandezza, come fa gia' la pipeline dei temporali.
    td = np.where(np.isfinite(td) & (td > 150.0), td - 273.15, td)
    spread = np.maximum(t - np.minimum(td, t), 0.0)
    ground = 0.0 if hsurf_m is None else np.nan_to_num(np.asarray(hsurf_m, float))
    return ground + 125.0 * spread


class CloudEnvironmentWriter:
    """Raccoglie, scadenza per scadenza, l'ambiente del volume durante il run."""

    def __init__(self, run_time, latitudes, longitudes,
                 max_lead_hours: int = MAX_LEAD_HOURS,
                 target_points: int = TARGET_POINTS) -> None:
        lat = np.asarray(latitudes, dtype=np.float64)
        lon = np.asarray(longitudes, dtype=np.float64)
        if lat.ndim != 1 or lon.ndim != 1 or lat.size < 2 or lon.size < 2:
            raise ValueError("griglia lat/lon non regolare")
        self.run_time = _parse_time(run_time)
        self.max_lead_hours = int(max_lead_hours)
        self._flip_lat = lat[0] > lat[-1]
        self._flip_lon = lon[0] > lon[-1]
        lat = lat[::-1] if self._flip_lat else lat
        lon = lon[::-1] if self._flip_lon else lon
        self.factor = coarsen_factor(lat.size, lon.size, target_points)
        self.latitudes = _block_axis(lat, self.factor)
        self.longitudes = _block_axis(lon, self.factor)
        self.hours: list[int] = []
        self.fields: dict[str, list[np.ndarray]] = {name: [] for name in TIME_VARYING}
        self.hsurf: np.ndarray | None = None

    def _orient(self, values):
        arr = np.asarray(values, dtype=np.float64)
        if self._flip_lat:
            arr = arr[::-1, :]
        if self._flip_lon:
            arr = arr[:, ::-1]
        return arr

    def add(self, lead_hours: int, t2m_c, td2m_c, cape, t500_k=None, hsurf_m=None) -> bool:
        """Aggiunge una scadenza; False se fuori orizzonte o incompleta."""
        if lead_hours > self.max_lead_hours or lead_hours in self.hours:
            return False
        if t2m_c is None or td2m_c is None or cape is None:
            return False
        t2m_c = self._orient(t2m_c)
        td2m_c = self._orient(td2m_c)
        cape = self._orient(cape)
        hsurf = self._orient(hsurf_m) if hsurf_m is not None else None
        t500 = self._orient(t500_k) if t500_k is not None else None
        t2m_k = t2m_c + 273.15
        derived = {
            "lcl_asl_m": _block(lcl_height_asl_m(t2m_c, td2m_c, hsurf), self.factor, "mean"),
            "t2m_k": _block(t2m_k, self.factor, "mean"),
            "lapse_k_km": _block(lapse_rate(t2m_k, t500, hsurf), self.factor, "mean"),
            # Il massimo e non la media: una cella convettiva larga dieci
            # chilometri non deve sparire diluita nel blocco.
            "cape": _block(np.maximum(np.nan_to_num(cape, nan=0.0), 0.0), self.factor, "max"),
        }
        for name, values in derived.items():
            self.fields[name].append(values)
        if self.hsurf is None and hsurf is not None:
            self.hsurf = _block(hsurf, self.factor, "mean")
        self.hours.append(int(lead_hours))
        return True

    def to_bytes(self) -> bytes:
        if not self.hours:
            raise ValueError("nessuna scadenza raccolta")
        order = np.argsort(self.hours)
        payload = {
            "run_time": np.array(_iso(self.run_time)),
            "hours": np.asarray(self.hours, dtype=np.int16)[order],
            "latitudes": self.latitudes.astype(np.float32),
            "longitudes": self.longitudes.astype(np.float32),
        }
        for name in TIME_VARYING:
            stack = np.stack(self.fields[name])[order]
            payload[name] = _quantize(name, stack)
        hsurf = self.hsurf if self.hsurf is not None else np.zeros(
            (self.latitudes.size, self.longitudes.size))
        payload["hsurf_m"] = _quantize("hsurf_m", hsurf)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **payload)
        return buffer.getvalue()

    def save(self, path) -> None:
        import os

        data = self.to_bytes()
        partial = f"{path}.part"
        with open(partial, "wb") as handle:
            handle.write(data)
        os.replace(partial, path)


@dataclass
class TemporalBlend:
    """Come l'ambiente e' stato portato all'istante del satellite."""

    hour_before: int
    hour_after: int
    weight_after: float
    lead_hours: float
    distance_hours: float
    mode: str  # "interpolated" | "exact" | "nearest"

    def as_dict(self) -> dict:
        return {
            "hourBefore": self.hour_before,
            "hourAfter": self.hour_after,
            "weightAfter": round(self.weight_after, 4),
            "leadHours": round(self.lead_hours, 3),
            "distanceHours": round(self.distance_hours, 3),
            "mode": self.mode,
        }


def temporal_blend(run_time, valid_time, hours,
                   max_extrapolation_hours: float = MAX_EXTRAPOLATION_HOURS) -> TemporalBlend:
    """Scadenze che racchiudono l'istante del satellite e peso lineare.

    Dentro l'intervallo delle scadenze disponibili si interpola fra le due
    ore adiacenti; appena fuori (run in ritardo, satellite piu' vecchio del
    run) si usa la scadenza piu' vicina, fino a ``max_extrapolation_hours``.
    Oltre si rifiuta: un ambiente di ieri non modula le nubi di oggi.
    """
    available = sorted({int(h) for h in hours})
    if not available:
        raise EnvironmentUnavailable("nessuna scadenza ICON-2I")
    lead = (_parse_time(valid_time) - _parse_time(run_time)).total_seconds() / 3600.0
    if lead <= available[0] or lead >= available[-1]:
        nearest = available[0] if lead <= available[0] else available[-1]
        distance = abs(lead - nearest)
        if distance > max_extrapolation_hours:
            raise EnvironmentUnavailable(
                f"satellite a {lead:+.1f} h dal run, scadenze {available[0]}-{available[-1]} h"
            )
        mode = "exact" if distance < 1e-6 else "nearest"
        return TemporalBlend(nearest, nearest, 0.0, lead, distance, mode)
    before = max(h for h in available if h <= lead)
    after = min(h for h in available if h >= lead)
    if before == after:
        return TemporalBlend(before, after, 0.0, lead, 0.0, "exact")
    weight = (lead - before) / float(after - before)
    distance = min(lead - before, after - lead)
    return TemporalBlend(before, after, weight, lead, distance, "interpolated")


class CloudEnvironment:
    """Ambiente di un run, interpolabile nel tempo e nello spazio."""

    def __init__(self, run_time, hours, latitudes, longitudes, fields, hsurf_m) -> None:
        self.run_time = _parse_time(run_time)
        self.hours = [int(h) for h in hours]
        self.latitudes = np.asarray(latitudes, dtype=np.float64)
        self.longitudes = np.asarray(longitudes, dtype=np.float64)
        self.fields = {name: np.asarray(v, dtype=np.float64) for name, v in fields.items()}
        self.hsurf_m = np.asarray(hsurf_m, dtype=np.float64)

    @classmethod
    def from_bytes(cls, data: bytes) -> "CloudEnvironment":
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            fields = {name: _dequantize(name, archive[name]) for name in TIME_VARYING}
            return cls(
                str(archive["run_time"]),
                archive["hours"].tolist(),
                archive["latitudes"],
                archive["longitudes"],
                fields,
                _dequantize("hsurf_m", archive["hsurf_m"]),
            )

    @classmethod
    def load(cls, path) -> "CloudEnvironment":
        with open(path, "rb") as handle:
            return cls.from_bytes(handle.read())

    def valid_time(self, hour: int) -> datetime:
        return self.run_time + timedelta(hours=int(hour))

    def at(self, valid_time) -> tuple[dict[str, np.ndarray], TemporalBlend]:
        """I campi all'istante ``valid_time``, sulla griglia del run."""
        blend = temporal_blend(self.run_time, valid_time, self.hours)
        i0 = self.hours.index(blend.hour_before)
        i1 = self.hours.index(blend.hour_after)
        w = blend.weight_after
        out = {}
        for name, stack in self.fields.items():
            a, b = stack[i0], stack[i1]
            # Se una delle due scadenze manca in un punto, vale l'altra.
            mixed = a * (1.0 - w) + b * w
            out[name] = np.where(np.isfinite(mixed), mixed, np.where(np.isfinite(a), a, b))
        out["hsurf_m"] = self.hsurf_m
        return out, blend


def resample_rectilinear(values, src_lat, src_lon, dst_lat, dst_lon) -> np.ndarray:
    """Bilineare da una griglia lat/lon crescente a righe/colonne arbitrarie.

    ``dst_lat`` indicizza le righe e ``dst_lon`` le colonne del risultato:
    la griglia Mercatore del volume e' separabile in latitudine e longitudine.
    Fuori dal dominio del modello il valore e' NaN, non il bordo ripetuto.
    """
    src = np.asarray(values, dtype=np.float64)
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = np.asarray(src_lon, dtype=np.float64)
    dst_lat = np.asarray(dst_lat, dtype=np.float64)
    dst_lon = np.asarray(dst_lon, dtype=np.float64)

    def indices(src_axis, dst_axis):
        f = np.interp(dst_axis, src_axis, np.arange(src_axis.size, dtype=np.float64))
        inside = (dst_axis >= src_axis[0]) & (dst_axis <= src_axis[-1])
        i0 = np.clip(np.floor(f).astype(int), 0, src_axis.size - 2)
        return i0, f - i0, inside

    r0, tr, rin = indices(src_lat, dst_lat)
    c0, tc, cin = indices(src_lon, dst_lon)
    tr = tr[:, None]
    tc = tc[None, :]
    a = src[r0][:, c0]
    b = src[r0][:, c0 + 1]
    c = src[r0 + 1][:, c0]
    d = src[r0 + 1][:, c0 + 1]
    out = (a * (1 - tc) + b * tc) * (1 - tr) + (c * (1 - tc) + d * tc) * tr
    # Un vicino NaN (mare senza CAPE, bordo) non deve annullare il punto:
    # si ripiega sul vicino piu' prossimo valido.
    if not np.all(np.isfinite(out)):
        nearest = src[np.clip(np.round(r0 + tr[:, 0]).astype(int), 0, src_lat.size - 1)][
            :, np.clip(np.round(c0 + tc[0]).astype(int), 0, src_lon.size - 1)]
        out = np.where(np.isfinite(out), out, nearest)
    out[~rin, :] = np.nan
    out[:, ~cin] = np.nan
    return out
