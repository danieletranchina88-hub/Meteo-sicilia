"""Fusione EUMETSAT + ICON-2I in un'unica texture RGBA per il ray marcher.

Il fotogramma satellitare e' la MASCHERA SPAZIALE ESATTA: dove il satellite
non vede nube, la texture e' zero in tutti e quattro i canali e il volume non
esiste, qualunque cosa dica il modello.  ICON-2I, portato all'istante del
satellite (``environment.temporal_blend``), e' solo il MODIFICATORE
TERMODINAMICO: dice a che quota sta la base, quanto e' fredda l'aria a una
certa quota e quanto e' violenta la convezione possibile.

I quattro canali, a otto bit, normalizzati su ``SCALE_KM``:

    R  cima       temperatura di brillanza IR 10,5 um invertita sul profilo
                  termico ICON-2I (T2m, gradiente T2m-T500, tropopausa)
    G  densita'   albedo VIS 0,6 um oltre il fondo di cielo sereno locale,
                  corretto per l'altezza del sole; di notte dal contrasto IR
    B  base       LCL ICON-2I (formula di Lawrence) sopra l'orografia,
                  limitata in basso dallo spessore plausibile del genere
    A  convezione corrente ascensionale potenziale 0,45*sqrt(2*CAPE),
                  allargata sul vicinato per assorbire lo sfasamento fra
                  cella osservata e cella simulata
"""

from __future__ import annotations

import json
import math
import struct
import warnings
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from meteo_analysis.clouds.environment import resample_rectilinear

METHOD = "eumetsat-mask-icon2i-environment-v1"
SCALE_KM = 16.0
RADIUS_EARTH_M = 6378137.0

# Dominio ICON-2I: lo stesso di process_data.py.
DOMAIN = {"west": 3.0, "south": 33.7, "east": 22.0, "north": 48.9}
DEFAULT_WIDTH = 1024

# --- IR 10,5 um -------------------------------------------------------------
# EUMETView distribuisce FCI IR 10,5 um come conteggi a otto bit.  La scala
# fra conteggio e temperatura di brillanza non e' nel PNG: e' quella della
# legenda ufficiale dello stile "mtg_fd_ir105_hrfi_style_02" (GetLegendGraphic),
# lineare in temperatura con tre ancore -- conteggio 1 a -73 C, il passaggio
# fra colori e grigi (conteggio 111) a -32 C, conteggio 255 a +30 C.  Lo stile
# in grigio consegnato dal WMS ("grayscale") mappa il conteggio 1 su #fefefe e
# il 255 su #010101: il grigio e' il conteggio rovesciato.
IR_COUNT_ANCHORS = ((1.0, -73.0), (111.0, -32.0), (255.0, 30.0))
# La tropopausa alle nostre latitudini: sopra, la temperatura smette di
# scendere, e una cima piu' fredda della tropopausa e' una torre che sfonda
# (overshooting top), non una nube a venti chilometri.
TROPOPAUSE_KM = 12.0
# Raffreddamento di una torre oltre la tropopausa: circa 7 K per chilometro
# di sfondamento (Griffin et al., 2016, "Characteristics of overshooting
# top and above-anvil cirrus plumes").
OVERSHOOT_K_PER_KM = 7.0

# --- VIS 0,6 um -------------------------------------------------------------
# Riflettanza come frazione del grigio (dichiarato: EUMETView non pubblica la
# curva del VIS, che e' distribuito linearizzato).  Il sole basso allunga il
# cammino: l'albedo si ricava dividendo per cos(zenit), mai sotto 0,15.
VIS_MIN_COS_ZENITH = 0.15
# Albedo di una nube otticamente spessa (tau > 30, Liou 2002): un cumulonembo
# o un nembostrato superano 0,8.
THICK_CLOUD_ALBEDO = 0.82
# Sotto questo coseno dello zenit il VIS non si usa piu': alba e tramonto
# passano con continuita' alla densita' dal solo infrarosso.
DAY_COS_ZENITH = (0.05, 0.25)
# Il fondo di cielo sereno, stimato a blocchi sui pixel che la maschera nubi
# dice sereni: la sabbia del Sahara e il mare non hanno la stessa albedo.
BACKGROUND_BLOCK_PX = 48

# Contrasto termico alla superficie per la densita' notturna: 35 K di nube
# piu' fredda del suolo sono una nube otticamente spessa.
IR_THICK_DEPRESSION_K = 35.0
# Senza maschera CLM: la nube comincia 4 K sotto la temperatura al suolo ed e'
# certa da 12 K in giu'.
IR_MASK_DEPRESSION_K = (4.0, 12.0)

# --- geometria della colonna ---------------------------------------------
MIN_THICKNESS_KM = 0.25
# Lo spessore massimo cresce con la densita' (un nembostrato e' spesso, un
# cirro e' una lama) e con la convezione (un cumulonembo va dall'LCL alla
# tropopausa).  Dichiarato, non misurato: il satellite non vede la base.
THICKNESS_BASE_KM = 0.8
THICKNESS_PER_DENSITY_KM = 4.0
THICKNESS_PER_CONVECTION_KM = 11.0

# --- convezione ---------------------------------------------------------
# Densita' a partire dalla quale una nube puo' essere la torre di un
# cumulonembo: sotto, il CAPE non ne allunga la colonna fino all'LCL.
DEEP_DENSITY = (0.45, 0.85)
PARCEL_EFFICIENCY = 0.45  # come meteo_analysis.hazards.storms
# Una corrente di 40 m/s e' il massimo delle supercelle mediterranee:
# normalizza il canale A a 1.
UPDRAFT_REFERENCE_MS = 40.0
# Vicinato su cui si prende il massimo del CAPE: 25 km, la distanza tipica
# fra la cella osservata e la stessa cella nel modello a 2,2 km.
CAPE_NEIGHBOURHOOD_KM = 25.0


@dataclass
class MercatorGrid:
    """Griglia regolare in Web Mercator: la texture si stende senza distorsione."""

    west: float
    south: float
    east: float
    north: float
    width: int
    height: int = 0
    longitudes: np.ndarray = field(init=False, repr=False)
    latitudes: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        x0, x1 = merc_x(self.west), merc_x(self.east)
        y0, y1 = merc_y(self.south), merc_y(self.north)
        if not self.height:
            self.height = max(16, int(round(self.width * (y1 - y0) / (x1 - x0))))
        xs = x0 + (np.arange(self.width) + 0.5) * (x1 - x0) / self.width
        # Riga 0 = nord, come le immagini WMS e la texture WebGL.
        ys = y1 - (np.arange(self.height) + 0.5) * (y1 - y0) / self.height
        self.longitudes = np.degrees(xs / RADIUS_EARTH_M)
        self.latitudes = np.degrees(2.0 * np.arctan(np.exp(ys / RADIUS_EARTH_M)) - math.pi / 2)

    @classmethod
    def for_domain(cls, width: int = DEFAULT_WIDTH, domain: dict | None = None) -> "MercatorGrid":
        d = domain or DOMAIN
        return cls(d["west"], d["south"], d["east"], d["north"], width)

    def bbox_3857(self) -> tuple[float, float, float, float]:
        return (merc_x(self.west), merc_y(self.south), merc_x(self.east), merc_y(self.north))

    def km_per_pixel(self) -> float:
        lat = math.radians((self.south + self.north) / 2)
        return (self.east - self.west) * 111.32 * math.cos(lat) / self.width


def merc_x(lon: float) -> float:
    return RADIUS_EARTH_M * math.radians(lon)


def merc_y(lat: float) -> float:
    lat = max(-85.0511, min(85.0511, lat))
    return RADIUS_EARTH_M * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def smoothstep(edge0, edge1, x) -> np.ndarray:
    t = np.clip((np.asarray(x, dtype=np.float64) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --- satellite ------------------------------------------------------------

def brightness_temperature_from_counts(counts) -> np.ndarray:
    """Conteggi EUMETView IR 10,5 um -> temperatura di brillanza in kelvin."""
    c = np.asarray(counts, dtype=np.float64)
    xs = [a[0] for a in IR_COUNT_ANCHORS]
    ys = [a[1] for a in IR_COUNT_ANCHORS]
    bt = np.interp(c, xs, ys) + 273.15
    return np.where(np.isfinite(c) & (c >= 1.0), bt, np.nan)


def brightness_temperature_from_grey(grey, alpha=None) -> np.ndarray:
    """Grigio del WMS IR (stile grayscale, freddo = chiaro) -> kelvin."""
    g = np.asarray(grey, dtype=np.float64)
    counts = 1.0 + (254.0 - g) * 254.0 / 253.0
    bt = brightness_temperature_from_counts(np.clip(counts, 1.0, 255.0))
    if alpha is not None:
        bt = np.where(np.asarray(alpha) > 127, bt, np.nan)
    return bt


def solar_cos_zenith(when: datetime, latitudes, longitudes) -> np.ndarray:
    """cos(zenit solare) su una griglia (righe=lat, colonne=lon); NOAA/Meeus."""
    when = when.astimezone(timezone.utc)
    day = when.timetuple().tm_yday
    hours = when.hour + when.minute / 60.0 + when.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (day - 1 + (hours - 12) / 24.0)
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    lat = np.radians(np.asarray(latitudes, dtype=np.float64))[:, None]
    lon = np.asarray(longitudes, dtype=np.float64)[None, :]
    true_solar_min = hours * 60.0 + eqtime + 4.0 * lon
    hour_angle = np.radians(true_solar_min / 4.0 - 180.0)
    return np.sin(lat) * math.sin(decl) + np.cos(lat) * math.cos(decl) * np.cos(hour_angle)


def _box_mean(values, radius: int) -> np.ndarray:
    """Media mobile separabile con somme cumulative, bordi replicati."""
    if radius < 1:
        return np.asarray(values, dtype=np.float64).copy()
    arr = np.asarray(values, dtype=np.float64)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius + 1, radius)
        p = np.pad(arr, pad, mode="edge")
        c = np.cumsum(p, axis=axis)
        n = arr.shape[axis]
        hi = np.take(c, np.arange(2 * radius + 1, 2 * radius + 1 + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        arr = (hi - lo) / (2 * radius + 1)
    return arr


def _max_filter(values, radius: int) -> np.ndarray:
    """Massimo su una finestra quadrata, separabile (dilatazione)."""
    arr = np.asarray(values, dtype=np.float64)
    if radius < 1:
        return arr.copy()
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        p = np.pad(arr, pad, mode="edge")
        n = arr.shape[axis]
        out = np.take(p, np.arange(0, n), axis=axis)
        for k in range(1, 2 * radius + 1):
            out = np.maximum(out, np.take(p, np.arange(k, k + n), axis=axis))
        arr = out
    return arr


def clear_sky_background(values, clear, block: int = BACKGROUND_BLOCK_PX,
                         percentile: float = 50.0) -> np.ndarray:
    """Valore tipico del cielo sereno, per blocchi, esteso dove e' coperto."""
    arr = np.asarray(values, dtype=np.float64)
    ok = np.asarray(clear, dtype=bool) & np.isfinite(arr)
    h, w = arr.shape
    ny, nx = max(1, -(-h // block)), max(1, -(-w // block))
    global_value = float(np.percentile(arr[ok], percentile)) if ok.any() else 0.0
    grid = np.full((ny, nx), np.nan)
    for by in range(ny):
        for bx in range(nx):
            sl = (slice(by * block, (by + 1) * block), slice(bx * block, (bx + 1) * block))
            sample = arr[sl][ok[sl]]
            if sample.size >= block:
                grid[by, bx] = np.percentile(sample, percentile)
    # I blocchi interamente coperti prendono il fondo dei vicini sereni.
    for _ in range(max(ny, nx)):
        missing = ~np.isfinite(grid)
        if not missing.any():
            break
        padded = np.pad(grid, 1, constant_values=np.nan)
        stack = np.stack([padded[dy:dy + ny, dx:dx + nx]
                          for dy in range(3) for dx in range(3)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            neighbours = np.nanmean(stack, axis=0)
        grid = np.where(missing & np.isfinite(neighbours), neighbours, grid)
    grid = np.where(np.isfinite(grid), grid, global_value)
    rows = np.minimum(np.arange(h) // block, ny - 1)
    cols = np.minimum(np.arange(w) // block, nx - 1)
    return _box_mean(grid[rows][:, cols], block // 2)


# --- i quattro canali -----------------------------------------------------

def cloud_top_height_km(bt_k, t_surface_k, lapse_k_per_km, terrain_km) -> np.ndarray:
    """Canale R: la quota a cui l'ambiente ICON-2I e' freddo quanto la cima.

    Sotto la tropopausa si risale il gradiente del modello; oltre, la
    temperatura non scende piu' e il di piu' di freddo e' sfondamento.  Una
    nube semitrasparente (cirro sottile) appare piu' calda della sua cima:
    la quota e' allora sottostimata, limite noto di ogni metodo IR a canale
    singolo.
    """
    bt = np.asarray(bt_k, dtype=np.float64)
    ts = np.asarray(t_surface_k, dtype=np.float64)
    gamma = np.maximum(np.asarray(lapse_k_per_km, dtype=np.float64), 1.0)
    zs = np.nan_to_num(np.asarray(terrain_km, dtype=np.float64))
    t_tropo = ts - gamma * np.maximum(TROPOPAUSE_KM - zs, 0.0)
    below = zs + (ts - bt) / gamma
    above = TROPOPAUSE_KM + (t_tropo - bt) / OVERSHOOT_K_PER_KM
    top = np.where(bt >= t_tropo, below, above)
    return np.clip(top, zs, SCALE_KM)


def optical_density(vis_reflectance, cos_zenith, clear, ir_depression_k) -> np.ndarray:
    """Canale G: densita' iniziale delle particelle, 0-1.

    Di giorno e' l'albedo VIS eccedente il fondo sereno locale, rapportata
    all'albedo di una nube spessa: e' in pratica lo spessore ottico
    compresso in [0,1].  Di notte, e dove il VIS manca, si ricava dal
    contrasto termico con la superficie.  Il passaggio e' continuo.
    """
    depression = np.nan_to_num(np.asarray(ir_depression_k, dtype=np.float64))
    ir_density = np.clip(depression / IR_THICK_DEPRESSION_K, 0.0, 1.0) ** 0.8
    if vis_reflectance is None:
        return ir_density
    mu = np.asarray(cos_zenith, dtype=np.float64)
    vis = np.asarray(vis_reflectance, dtype=np.float64)
    albedo = vis / np.maximum(mu, VIS_MIN_COS_ZENITH)
    background = clear_sky_background(albedo, clear)
    span = np.maximum(THICK_CLOUD_ALBEDO - background, 0.1)
    vis_density = np.clip((albedo - background) / span, 0.0, 1.0) ** 0.75
    day = smoothstep(DAY_COS_ZENITH[0], DAY_COS_ZENITH[1], mu)
    day = np.where(np.isfinite(vis), day, 0.0)
    vis_density = np.nan_to_num(vis_density)
    return day * vis_density + (1.0 - day) * ir_density


def convective_modifier(cape_j_kg, radius_px: int = 0) -> np.ndarray:
    """Canale A: corrente ascensionale potenziale normalizzata, 0-1."""
    cape = np.maximum(np.nan_to_num(np.asarray(cape_j_kg, dtype=np.float64)), 0.0)
    cape = _max_filter(cape, radius_px)
    updraft = PARCEL_EFFICIENCY * np.sqrt(2.0 * cape)
    return np.clip(updraft / UPDRAFT_REFERENCE_MS, 0.0, 1.0)


def cloud_base_km(lcl_asl_km, top_km, density, convective, terrain_km) -> np.ndarray:
    """Canale B: base della nube.

    La base e' l'LCL del modello, cioe' la quota a cui l'aria del suolo
    condensa.  Vale per le nubi che nascono dal suolo -- cumuli, cumulonembi,
    strati.  Un cirro a nove chilometri non scende fino all'LCL: lo spessore
    e' limitato da una regola dichiarata che cresce con densita' e
    convezione, e la base non sale mai oltre ``top - MIN_THICKNESS_KM``.
    """
    top = np.asarray(top_km, dtype=np.float64)
    zs = np.nan_to_num(np.asarray(terrain_km, dtype=np.float64))
    lcl = np.asarray(lcl_asl_km, dtype=np.float64)
    lcl = np.where(np.isfinite(lcl), lcl, zs + 1.0)
    thickness = (THICKNESS_BASE_KM + THICKNESS_PER_DENSITY_KM * np.asarray(density)
                 + THICKNESS_PER_CONVECTION_KM * np.asarray(convective))
    base = np.maximum(np.maximum(lcl, zs), top - thickness)
    return np.clip(np.minimum(base, top - MIN_THICKNESS_KM), 0.0, SCALE_KM)


@dataclass
class SatelliteFrame:
    """Un fotogramma EUMETSAT gia' riportato sulla griglia del volume."""

    time: datetime
    bt_k: np.ndarray                 # IR 10,5 um, kelvin (NaN = nessun dato)
    vis_reflectance: np.ndarray | None = None  # VIS 0,6 um, 0-1
    cloud_mask: np.ndarray | None = None       # CLM, frazione di nube 0-1
    sources: dict = field(default_factory=dict)


@dataclass
class VolumeTexture:
    rgba: np.ndarray
    metadata: dict
    top_km: np.ndarray
    base_km: np.ndarray
    density: np.ndarray
    convective: np.ndarray


def fuse(grid: MercatorGrid, frame: SatelliteFrame, environment: dict, blend=None,
         run_time=None) -> VolumeTexture:
    """Il cuore della pipeline: satellite come maschera, modello come ambiente.

    ``environment`` sono i campi di ``CloudEnvironment.at`` con le coordinate
    ``latitudes``/``longitudes`` della griglia del run.
    """
    shape = (grid.height, grid.width)
    if frame.bt_k.shape != shape:
        raise ValueError(f"IR {frame.bt_k.shape}, attesa {shape}")

    def on_grid(name, fallback):
        values = environment.get(name)
        if values is None:
            return np.full(shape, fallback)
        out = resample_rectilinear(values, environment["latitudes"], environment["longitudes"],
                                   grid.latitudes, grid.longitudes)
        return np.where(np.isfinite(out), out, fallback)

    terrain_km = on_grid("hsurf_m", 0.0) / 1000.0
    t_surface = on_grid("t2m_k", 288.15)
    lapse = on_grid("lapse_k_km", 6.5)
    lcl_km = on_grid("lcl_asl_m", np.nan) / 1000.0
    cape = on_grid("cape", 0.0)
    # Fuori dal dominio del modello non c'e' ambiente: il volume non si fa.
    inside_model = np.isfinite(on_grid("t2m_k", np.nan))

    bt = np.asarray(frame.bt_k, dtype=np.float64)
    depression = t_surface - bt
    mu = solar_cos_zenith(frame.time, grid.latitudes, grid.longitudes)

    # LA MASCHERA E' DEL SATELLITE.  La CLM (classificazione multicanale
    # EUMETSAT) quando c'e'; altrimenti il contrasto IR con la superficie.
    if frame.cloud_mask is not None:
        # La CLM ha pixel di 3 km consegnati a gradini dal WMS: un pixel di
        # raccordo toglie la scalinatura senza spostare il bordo.
        mask = _box_mean(np.clip(np.nan_to_num(frame.cloud_mask), 0.0, 1.0), 1)
        mask_source = "clm"
    else:
        mask = smoothstep(IR_MASK_DEPRESSION_K[0], IR_MASK_DEPRESSION_K[1], depression)
        mask_source = "ir-contrast"
    mask = np.where(np.isfinite(bt) & inside_model, mask, 0.0)
    clear = mask < 0.05

    top = cloud_top_height_km(bt, t_surface, lapse, terrain_km)
    density = optical_density(frame.vis_reflectance, mu, clear, depression)
    radius_px = max(0, int(round(CAPE_NEIGHBOURHOOD_KM / grid.km_per_pixel())))
    convective = convective_modifier(cape, radius_px)
    # Il CAPE dice quanto POTREBBE salire una corrente; la colonna scende fino
    # all'LCL solo se il satellite vede davvero una nube spessa. Un cirro
    # sottile sopra aria instabile resta un cirro.
    deep = convective * smoothstep(DEEP_DENSITY[0], DEEP_DENSITY[1], density)
    base = cloud_base_km(lcl_km, top, density, deep, terrain_km)
    # Una nube bassa e calda (nebbia, strato) ha la cima IR quasi al suolo:
    # la colonna non puo' essere piu' sottile dello spessore minimo.
    top = np.maximum(top, base + MIN_THICKNESS_KM)

    # Anche una nube che il VIS vede appena resta una nube: la densita' non
    # scende sotto un minimo dove la maschera e' piena, altrimenti la CLM
    # direbbe "nube" e il volume "vuoto".
    density = np.maximum(density, 0.12) * mask
    cloudy = density > 1.0 / 255.0

    # G e' la maschera. R, B e A fuori dalla nube non sono zero ma le quote
    # delle nubi vicine: il ray marcher legge le quote a scala piu' grossa
    # della sagoma, e degli zeri trascinerebbero a terra cime e basi di bordo.
    top_c = extend_outside(top, cloudy)
    base_c = extend_outside(base, cloudy)
    conv_c = extend_outside(convective, cloudy)
    rgba = encode_rgba(top_c, density, base_c, conv_c)

    metadata = {
        "method": METHOD,
        "satelliteTime": frame.time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bounds": {"west": grid.west, "south": grid.south, "east": grid.east, "north": grid.north},
        "projection": "EPSG:3857",
        "width": grid.width,
        "height": grid.height,
        "scaleKm": SCALE_KM,
        "channels": {
            "r": "cloud top height (IR 10.5 um brightness temperature on the ICON-2I thermal profile), km / scaleKm",
            "g": "initial particle density (VIS 0.6 um albedo above clear sky; IR contrast at night), 0-1",
            "b": "cloud base height (ICON-2I LCL above orography, capped by genus thickness), km / scaleKm",
            "a": "convective modifier (ICON-2I potential updraft 0.45*sqrt(2*CAPE) / 40 m/s), 0-1",
        },
        "mask": mask_source,
        "sources": frame.sources,
        "model": {
            "name": "ICON-2I",
            "runTime": (run_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                        if isinstance(run_time, datetime) else run_time),
            "blend": blend.as_dict() if blend is not None else None,
        },
        "stats": {
            "cloudyPct": round(100.0 * float(cloudy.mean()), 2),
            "topMaxKm": round(float(top[cloudy].max()), 2) if cloudy.any() else 0.0,
            "convectiveMax": round(float(convective[cloudy].max()), 3) if cloudy.any() else 0.0,
            "dayFraction": round(float((mu > DAY_COS_ZENITH[1]).mean()), 3),
        },
    }
    return VolumeTexture(rgba, metadata, top_c, base_c, density, conv_c)


def extend_outside(values, inside, radii=(2, 6, 18, 54)) -> np.ndarray:
    """Dentro ``inside`` i valori veri; fuori la media dei valori vicini.

    Convoluzione normalizzata a raggi crescenti: ogni pixel sereno prende la
    media delle colonne nuvolose entro il raggio piu' piccolo che ne contiene
    qualcuna. Lontano da ogni nube resta zero.
    """
    vals = np.where(inside, np.nan_to_num(np.asarray(values, dtype=np.float64)), 0.0)
    weight = np.asarray(inside, dtype=np.float64)
    out = vals.copy()
    filled = np.asarray(inside, dtype=bool).copy()
    for radius in radii:
        w = _box_mean(weight, radius)
        v = _box_mean(vals, radius)
        take = ~filled & (w > 1e-6)
        out[take] = v[take] / w[take]
        filled |= take
    return out


def encode_rgba(top_km, density, base_km, convective) -> np.ndarray:
    """Quattro campi fisici -> RGBA8 (H, W, 4)."""
    def byte(values):
        return np.round(np.clip(np.nan_to_num(values), 0.0, 1.0) * 255.0).astype(np.uint8)

    return np.stack([
        byte(np.asarray(top_km) / SCALE_KM),
        byte(density),
        byte(np.asarray(base_km) / SCALE_KM),
        byte(convective),
    ], axis=-1)


def decode_rgba(rgba) -> dict[str, np.ndarray]:
    """L'inverso di ``encode_rgba`` (per le prove e per chi legge il file)."""
    arr = np.asarray(rgba, dtype=np.float64) / 255.0
    return {
        "top_km": arr[..., 0] * SCALE_KM,
        "density": arr[..., 1],
        "base_km": arr[..., 2] * SCALE_KM,
        "convective": arr[..., 3],
    }


def png_bytes(rgba) -> bytes:
    """PNG RGBA8 senza dipendenze: filtro 'Up' per riga, zlib livello 9.

    Il filtro Up rende quasi nulle le righe in un campo liscio come questo e
    il file resta piccolo senza bisogno di Pillow.
    """
    arr = np.ascontiguousarray(np.asarray(rgba, dtype=np.uint8))
    h, w, c = arr.shape
    if c != 4:
        raise ValueError("servono quattro canali")
    rows = arr.reshape(h, w * 4).astype(np.int16)
    up = np.empty_like(rows)
    up[0] = rows[0]
    up[1:] = rows[1:] - rows[:-1]
    filtered = np.empty((h, w * 4 + 1), dtype=np.uint8)
    filtered[:, 0] = 2
    filtered[0, 0] = 0
    filtered[:, 1:] = (up & 0xFF).astype(np.uint8)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(filtered.tobytes(), 9)) + chunk(b"IEND", b""))


def write_texture(texture: VolumeTexture, png_path, json_path) -> None:
    """Scrive PNG e metadati in modo atomico: prima il PNG, poi il JSON.

    Il client legge il JSON e da li' il PNG con il suo stesso marcatore di
    tempo: un lettore non vede mai metadati nuovi con un'immagine vecchia.
    """
    import os

    stamp = texture.metadata["satelliteTime"].replace(":", "").replace("-", "")
    texture.metadata["image"] = os.path.basename(str(png_path)) + "?t=" + stamp
    for path, data in ((png_path, png_bytes(texture.rgba)),
                       (json_path, json.dumps(texture.metadata, ensure_ascii=False,
                                              allow_nan=False, indent=1).encode("utf-8"))):
        partial = f"{path}.part"
        with open(partial, "wb") as handle:
            handle.write(data)
        os.replace(partial, path)
