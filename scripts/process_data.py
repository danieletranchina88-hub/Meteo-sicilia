import math
import requests
import xarray as xr
import numpy as np
import json
import gzip
import os
import sys
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from front_analysis_v12 import FRONT_METHOD as ICON_FRONT_METHOD
from front_analysis_v12 import FrontalAnalysisV12

# Add meteo_analysis imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from meteo_analysis.core.icon_fields import IconRunFields
from meteo_analysis.hazards.storms import (
    bowen_ratio,
    coarsen,
    sea_breeze_lift,
    trigger_index,
    upslope_flow,
    bulk_shear,
    downburst_potential,
    hail_potential,
    intelligent_storm_probability,
    lifting_condensation_level,
    potential_updraft,
    storm_mode,
    strongest_updraft,
    summarize_storms,
    updraft_from_omega,
)
from meteo_analysis.hazards.convection import (
    front_distance_km,
    horizontal_convergence,
    normalize_cin,
    relative_humidity_from_specific_humidity,
    summarize_convection,
)
from meteo_analysis.hazards.visibility import (
    calculate_fog_probability,
    classify_fog_type,
    estimate_visibility,
)
from meteo_analysis.hazards.winter import detect_freezing_rain
from meteo_analysis.orography.foehn import (
    alpine_domain_mask,
    cross_alpine_pressure_difference,
    detect_foehn,
)
from meteo_analysis.products.nlg import (
    NLG_METHOD,
    build_bulletin_inputs,
    generate_bulletin_details,
)
from meteo_analysis.products.meteograms import MeteogramArchive
from meteo_analysis.ml.icon2i import Icon2IStore
from meteo_analysis.ml.model import FrontModel
from ml_fronts import predict_store as predict_ml_fronts

# --- CONFIGURAZIONE ---
DATASET_ID = "ICON_2I_SURFACE_PRESSURE_LEVELS"
API_LIST_URL = f"https://meteohub.agenziaitaliameteo.it/api/datasets/{DATASET_ID}/opendata"
API_DOWNLOAD_URL = "https://meteohub.agenziaitaliameteo.it/api/opendata"

FINAL_DIR = "data_weather"
TEMP_DIR = "temp_processing"
TEMP_FILE = "temp.grib2"
FRONT_TEMP_DIR = "temp_front_processing"
HAZARD_TEMP_DIR = "temp_hazard_processing"
# The API dataset identifier uses an underscore, while the public NWP
# directory really uses a hyphen.  Keep the two identifiers separate.
NWP_DIRECTORY_ID = "ICON-2I_SURFACE_PRESSURE_LEVELS"
NWP_DIRECT_BASE = "https://meteohub.agenziaitaliameteo.it/nwp"

# ============================================================
# DOMINIO MASSIMO PUBBLICATO DA ICON-2I / METEOHUB.
# Verificato direttamente sui GRIB: 761 x 761 punti.
# ============================================================
LAT_MIN, LAT_MAX = 33.7, 48.9
LON_MIN, LON_MAX = 3.0, 22.0

# ============================================================
# CONTROLLO DIMENSIONE GRIGLIA
# FIX 1: Portato a 600_000 per mantenere la risoluzione nativa a 2.2km
# ============================================================
MAX_PIXELS = 600_000
FEELS_LIKE_METHOD = "heat-index-wind-chill-v1"
CONVECTION_METHOD = "icon2i-physical-evidence-fusion-v2"
MAX_CONVECTIVE_HIGH_AREA_PCT = 15.0
MAX_FIXED_80_AREA_PCT = 5.0


def download_grib_file(url, destination):
    """Download a large MeteoHub GRIB with validation and resumable retries."""
    partial = destination + ".part"
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "MeteoHub-Mobile-Synoptic/1.0",
    }
    expected_size = 0
    try:
        response = requests.head(
            url,
            headers=headers,
            timeout=(20, 60),
            allow_redirects=True,
        )
        response.raise_for_status()
        expected_size = int(response.headers.get("Content-Length") or 0)
    except Exception:
        pass

    for attempt in range(1, 5):
        try:
            current_size = os.path.getsize(partial) if os.path.exists(partial) else 0
            request_headers = dict(headers)
            if current_size:
                request_headers["Range"] = f"bytes={current_size}-"

            with requests.get(
                url,
                headers=request_headers,
                stream=True,
                timeout=(25, 300),
            ) as response:
                response.raise_for_status()
                append = current_size > 0 and response.status_code == 206
                if not append:
                    current_size = 0
                response_size = int(response.headers.get("Content-Length") or 0)
                if expected_size <= 0:
                    if response.status_code == 206:
                        content_range = response.headers.get("Content-Range", "")
                        if "/" in content_range:
                            expected_size = int(content_range.rsplit("/", 1)[-1])
                    elif response_size:
                        expected_size = response_size

                with open(partial, "ab" if append else "wb") as output:
                    for chunk in response.iter_content(chunk_size=2 * 1024 * 1024):
                        if chunk:
                            output.write(chunk)

            downloaded_size = os.path.getsize(partial)
            if expected_size and downloaded_size != expected_size:
                if downloaded_size > expected_size:
                    os.remove(partial)
                raise IOError(
                    f"download incompleto: {downloaded_size}/{expected_size} byte"
                )
            if downloaded_size < 1_000_000:
                raise IOError(f"risposta troppo piccola: {downloaded_size} byte")
            with open(partial, "rb") as check:
                if check.read(4) != b"GRIB":
                    raise IOError("firma GRIB iniziale non valida")
                check.seek(-4, os.SEEK_END)
                if check.read(4) != b"7777":
                    raise IOError("file GRIB troncato")
            os.replace(partial, destination)
            return destination
        except Exception as error:
            print(f"   retry {attempt}/4: {error}", flush=True)
            if attempt == 4:
                if os.path.exists(partial):
                    os.remove(partial)
                raise
            if os.path.exists(partial) and os.path.getsize(partial) < 1_000_000:
                os.remove(partial)
            time.sleep(attempt * 2)


def write_json_atomic(path, payload):
    """Write a complete strict JSON document or leave no target at all."""
    partial = path + ".part"
    try:
        with open(partial, "w", encoding="utf-8") as output:
            json.dump(payload, output, separators=(",", ":"), allow_nan=False)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def write_json_gzip_atomic(path, payload, compresslevel=6):
    """Write deterministic compressed JSON, atomically.

    Full-domain ICON-2I fields are large.  Manual gzip decompression in the
    browser keeps the gh-pages snapshot and mobile transfer size manageable
    without reducing the 761 x 761 source domain.
    """
    partial = path + ".part"
    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    try:
        with open(partial, "wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                compresslevel=int(compresslevel),
                mtime=0,
            ) as output:
                output.write(encoded)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def write_observations(output_dir):
    """Fetch Italian-domain METAR observations and publish them for the
    browser-side observed analysis (Cressman fusion). Server-side fetch avoids
    the browser CORS block on aviationweather.gov. Non-fatal: on any error no
    file is written and the fusion layer simply reports observations absent.
    """
    def _num(mapping, key):
        try:
            value = float(mapping.get(key))
        except (TypeError, ValueError):
            return None
        return value if value == value else None  # drop NaN

    try:
        url = (
            "https://aviationweather.gov/api/data/metar?format=json"
            f"&bbox={LAT_MIN},{LON_MIN},{LAT_MAX},{LON_MAX}"
        )
        response = requests.get(
            url, timeout=(15, 45),
            headers={"User-Agent": "Meteo-Sicilia/1.0"},
        )
        response.raise_for_status()
        raw = response.json()
        stations = []
        latest = 0
        for report in raw if isinstance(raw, list) else []:
            latitude = _num(report, "lat")
            longitude = _num(report, "lon")
            if latitude is None or longitude is None:
                continue
            if not (LAT_MIN <= latitude <= LAT_MAX and LON_MIN <= longitude <= LON_MAX):
                continue
            wind_kt = _num(report, "wspd")
            pressure = _num(report, "altim")
            obs_time = report.get("obsTime")
            if isinstance(obs_time, (int, float)):
                latest = max(latest, int(obs_time))
            stations.append({
                "id": report.get("icaoId") or "",
                "name": report.get("name") or "",
                "lat": round(latitude, 4),
                "lon": round(longitude, 4),
                "tempC": _num(report, "temp"),
                "dewpC": _num(report, "dewp"),
                "wspdKmh": round(wind_kt * 1.852, 1) if wind_kt is not None else None,
                "wdir": _num(report, "wdir"),
                "pressHpa": (
                    pressure if pressure is not None and 850 < pressure < 1080 else None
                ),
            })
        payload = {
            "source": "NOAA aviationweather METAR",
            "obsTime": latest,
            "count": len(stations),
            "stations": stations,
        }
        write_json_atomic(os.path.join(output_dir, "observations.json"), payload)
        print(f"   Osservazioni METAR: {len(stations)} stazioni.", flush=True)
    except Exception as error:
        print(f"   Osservazioni METAR non disponibili: {error}", flush=True)


def prepare_icon_front_analyzer(run_dt):
    """Build the ICON-2I-only, hourly, multilayer frontal analysis.

    T/QV/U/V at 850 hPa are mandatory because they define the objective
    frontal geometry. T/QV and U/V at 925 hPa test whether both the air-mass
    boundary and the cross-front flow survive closer to the surface. T/QV/U/V
    at 700 hPa measure vertical coherence and tilt; omega is a secondary
    ascent diagnostic. Consecutive 10 m winds supply a temporal WND check.
    """
    run_tag = run_dt.strftime("%Y%m%d%H")
    common = f"ICON_2I_SURFACE_PRESSURE_LEVELS_{run_tag}"
    run_base = f"{NWP_DIRECT_BASE}/{NWP_DIRECTORY_ID}/{run_tag}"
    pressure_file_850 = f"{common}_isobaricInhPa-850.grib"
    pressure_file_925 = f"{common}_isobaricInhPa-925.grib"
    pressure_file_700 = f"{common}_isobaricInhPa-700.grib"
    pressure_file_500 = f"{common}_isobaricInhPa-500.grib"
    height_file_10 = f"{common}_heightAboveGround-10.grib"
    surface_file = f"{common}_surface-0.grib"
    mean_sea_file = f"{common}_meanSea-0.grib"
    requests_to_make = {
        "temperature": (f"{run_base}/T/{pressure_file_850}", "t850.grib"),
        "humidity": (f"{run_base}/QV/{pressure_file_850}", "q850.grib"),
        "u_wind": (f"{run_base}/U/{pressure_file_850}", "u850.grib"),
        "v_wind": (f"{run_base}/V/{pressure_file_850}", "v850.grib"),
        "temperature_925": (f"{run_base}/T/{pressure_file_925}", "t925.grib"),
        "humidity_925": (f"{run_base}/QV/{pressure_file_925}", "q925.grib"),
        "u_wind_925": (f"{run_base}/U/{pressure_file_925}", "u925.grib"),
        "v_wind_925": (f"{run_base}/V/{pressure_file_925}", "v925.grib"),
        "omega_700": (f"{run_base}/OMEGA/{pressure_file_700}", "omega700.grib"),
        "temperature_700": (f"{run_base}/T/{pressure_file_700}", "t700_ml.grib"),
        "humidity_700": (f"{run_base}/QV/{pressure_file_700}", "q700_ml.grib"),
        "u_wind_700": (f"{run_base}/U/{pressure_file_700}", "u700.grib"),
        "v_wind_700": (f"{run_base}/V/{pressure_file_700}", "v700.grib"),
        "u_wind_500": (f"{run_base}/U/{pressure_file_500}", "u500_ml.grib"),
        "v_wind_500": (f"{run_base}/V/{pressure_file_500}", "v500_ml.grib"),
        "geopotential_500": (f"{run_base}/FI/{pressure_file_500}", "fi500_ml.grib"),
        "u_wind_10": (f"{run_base}/U_10M/{height_file_10}", "u10_ml.grib"),
        "v_wind_10": (f"{run_base}/V_10M/{height_file_10}", "v10_ml.grib"),
        "orography": (f"{run_base}/HSURF/{surface_file}", "hsurf.grib"),
        "pressure": (f"{run_base}/PMSL/{mean_sea_file}", "pmsl.grib"),
    }
    # Questi campi aggiungono prove indipendenti, ma non definiscono la linea.
    # Sono opzionali per non inventare dati e non bloccare l'analisi primaria.
    optional_fields = {
        "pressure", "temperature_925", "humidity_925",
        "u_wind_925", "v_wind_925", "omega_700",
        "temperature_700", "humidity_700", "u_wind_700", "v_wind_700",
        "u_wind_500", "v_wind_500",
        "geopotential_500", "u_wind_10", "v_wind_10",
    }

    if os.path.exists(FRONT_TEMP_DIR):
        shutil.rmtree(FRONT_TEMP_DIR)
    os.makedirs(FRONT_TEMP_DIR)
    paths = {}
    print(
        "2a. Scarico ICON-2I 850 hPa e diagnostica opzionale 925/700 hPa…",
        flush=True,
    )

    try:
        # Due connessioni riducono i tempi senza sovraccaricare MeteoHub.
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            for name, (url, filename) in requests_to_make.items():
                destination = os.path.join(FRONT_TEMP_DIR, filename)
                future = executor.submit(download_grib_file, url, destination)
                futures[future] = (name, destination)
            for future in as_completed(futures):
                name, destination = futures[future]
                try:
                    future.result()
                except Exception as error:
                    if name in optional_fields:
                        print(f"   {name} non disponibile: {error}", flush=True)
                        continue
                    raise
                paths[name] = destination
                print(
                    f"   {name}: {os.path.getsize(destination) / 1048576:.1f} MB",
                    flush=True,
                )

        ml_guidance = None
        model_path = os.path.join("models", "front_model.json.gz.b64")
        ml_path_map = {
            "t850": paths.get("temperature"),
            "q850": paths.get("humidity"),
            "u850": paths.get("u_wind"),
            "v850": paths.get("v_wind"),
            "t700": paths.get("temperature_700"),
            "q700": paths.get("humidity_700"),
            "u500": paths.get("u_wind_500"),
            "v500": paths.get("v_wind_500"),
            "fi500": paths.get("geopotential_500"),
            "u10": paths.get("u_wind_10"),
            "v10": paths.get("v_wind_10"),
            "pmsl": paths.get("pressure"),
        }
        if os.path.exists(model_path) and all(ml_path_map.values()):
            ml_store = None
            try:
                ml_store = Icon2IStore(ml_path_map, run_tag)
                ml_model = FrontModel.load(model_path)
                common_hours = set.intersection(*[
                    ml_store.available_hours(name) for name in ml_path_map
                ])
                ml_guidance = predict_ml_fronts(
                    ml_store,
                    ml_model,
                    hours=range(min(common_hours), max(common_hours) + 1, 3),
                    output_dir=os.path.join(TEMP_DIR, "fronts_ml"),
                )
                print(
                    "   Fusione ML pronta: griglia 0.20°, passo 3 h "
                    "interpolato solo come conferma.",
                    flush=True,
                )
            except Exception as error:
                print(f"   ML frontale disabilitato: {error}", flush=True)
                ml_guidance = None
            finally:
                if ml_store is not None:
                    ml_store.close()

        analyzer = FrontalAnalysisV12(
            paths["temperature"],
            paths["humidity"],
            paths["u_wind"],
            paths["v_wind"],
            paths["orography"],
            pressure_path=paths.get("pressure"),
            lower_temperature_path=paths.get("temperature_925"),
            lower_humidity_path=paths.get("humidity_925"),
            lower_u_wind_path=paths.get("u_wind_925"),
            lower_v_wind_path=paths.get("v_wind_925"),
            upper_temperature_path=paths.get("temperature_700"),
            upper_humidity_path=paths.get("humidity_700"),
            upper_u_wind_path=paths.get("u_wind_700"),
            upper_v_wind_path=paths.get("v_wind_700"),
            surface_u_wind_path=paths.get("u_wind_10"),
            surface_v_wind_path=paths.get("v_wind_10"),
            omega_700_path=paths.get("omega_700"),
            downsample=4,
            bounds=(3.0, 22.0, 33.7, 48.9),
            method=ICON_FRONT_METHOD,
            source="ICON-2I",
            tendency_window_hours=3,
            ml_guidance=ml_guidance,
        )
        if len(analyzer.available_hours) < 70:
            analyzer.close()
            raise ValueError(
                f"solo {len(analyzer.available_hours)} scadenze ICON-2I disponibili"
            )
        # Il primo accesso esegue e valida l'intera sequenza oraria. Meglio
        # interrompere il deploy che pubblicare silenziosamente un layer
        # frontale tecnicamente vuoto o parziale.
        analyzer.analyze(analyzer.available_hours[0])
        summary = analyzer.analysis_summary or {}
        published = int(summary.get("publishedTracks", 0))
        print(
            "   Analisi ICON-2I pronta: "
            f"{len(analyzer.available_hours)} ore, "
            + (
                f"{published} tracce pubblicabili."
                if published else
                "nessun fronte sinottico robusto: pubblico un'analisi vuota valida."
            ),
            flush=True,
        )
        return analyzer
    except Exception as error:
        print(f"   Analisi frontale ICON-2I non disponibile: {error}", flush=True)
        return None


STORM_METHOD = "icon2i-physical-evidence-fusion-v2"
# L'evento e' "temporale entro 10 km"; la lisciatura a 25 km esprime il fatto
# che il modello sa che il temporale ci sara' ma non su quale paese.
STORM_EVENT_RADIUS_KM = 10.0
STORM_SMOOTHING_RADIUS_KM = 25.0
STORM_LPI_THRESHOLD = 1.0
# La griglia pubblicata e' dimezzata rispetto a quella del modello: a piena
# risoluzione la sezione pesava 3,6 MB per scadenza, cioe' 260 MB per corsa.
STORM_COARSEN = 2


def build_storm_payload(
    fields,
    hour,
    latitudes,
    longitudes,
    cape_ml,
    deep_layer_shear,
    omega_700,
    temperature_2m_c,
    u_wind_10m=None,
    v_wind_10m=None,
    convergence_10m=None,
    cin_ml=None,
    surface_rh=None,
    mid_level_rh=None,
    nearest_front_km=None,
    include_native=False,
):
    """Campi della sezione temporali per una scadenza, o None se mancano.

    Vive in un file separato da quello dello step: sono una dozzina di
    griglie e chi non apre la sezione temporali non deve scaricarle.  E' lo
    stesso schema gia' usato per i campi in quota.
    """
    if fields is None:
        return None
    lpi = fields.field("lpi", hour, latitudes, longitudes)
    if lpi is None:
        return None

    latitude = np.asarray(latitudes, dtype=float)
    longitude = np.asarray(longitudes, dtype=float)
    if latitude.size < 2 or longitude.size < 2:
        return None
    # Passo della griglia in chilometri: serve al vicinato, che ragiona in
    # distanze vere e non in celle.
    mean_latitude = float(np.nanmean(latitude))
    cell_y_km = abs(float(latitude[1] - latitude[0])) * 111.32
    cell_x_km = (
        abs(float(longitude[1] - longitude[0]))
        * 111.32
        * math.cos(math.radians(mean_latitude))
    )
    cell_km = float(np.mean([cell_x_km, cell_y_km]))
    if not np.isfinite(cell_km) or cell_km <= 0:
        return None

    previous_lpi = (
        fields.field("lpi", hour - 1, latitudes, longitudes)
        if hour > 0 else None
    )
    next_lpi = fields.field("lpi", hour + 1, latitudes, longitudes)

    # Correnti ascensionali risolte: il nucleo di una cella non sta sempre
    # allo stesso livello, quindi si prende il massimo sulla colonna.
    omega_levels = {}
    if omega_700 is not None:
        omega_levels[700.0] = omega_700
    for level, name in ((500.0, "omega500"), (850.0, "omega850")):
        value = fields.field(name, hour, latitudes, longitudes)
        if value is not None:
            omega_levels[level] = value
    updraft = strongest_updraft(omega_levels) if omega_levels else None

    # Salita sinottica: lo stesso omega letto su un'altra scala. E' la
    # forzante larga che consuma il coperchio su intere regioni, tre ordini di
    # grandezza sotto il nucleo convettivo, e va mostrata separata.
    ascent = None
    if 700.0 in omega_levels:
        ascent = updraft_from_omega(omega_levels[700.0], 700.0) * 100.0  # cm/s

    dewpoint = fields.field("td_2m", hour, latitudes, longitudes)
    cloud_base = None
    if dewpoint is not None and temperature_2m_c is not None:
        dewpoint_c = np.where(
            np.isfinite(dewpoint) & (dewpoint > 150.0),
            dewpoint - 273.15,
            dewpoint,
        )
        cloud_base = lifting_condensation_level(temperature_2m_c, dewpoint_c)

    helicity = fields.field("uh_max", hour, latitudes, longitudes)
    cape_mu = fields.field("cape_con", hour, latitudes, longitudes)
    graupel = fields.field("graupel", hour, latitudes, longitudes)
    freezing_level = fields.field("hzerocl", hour, latitudes, longitudes)
    gust = fields.field("vmax_10m", hour, latitudes, longitudes)

    hail = None
    if graupel is not None and cape_ml is not None:
        hail = hail_potential(
            graupel,
            freezing_level if freezing_level is not None else np.nan,
            cape_ml,
            deep_layer_shear if deep_layer_shear is not None else np.nan,
        ) * 100.0

    downburst = None
    if cape_ml is not None and (cloud_base is not None or gust is not None):
        downburst = downburst_potential(
            cloud_base if cloud_base is not None else np.nan,
            gust if gust is not None else np.nan,
            cape_ml,
        ) * 100.0

    mode = None
    if cape_ml is not None and deep_layer_shear is not None:
        mode = storm_mode(cape_ml, deep_layer_shear)

    # --- Innesco: cosa puo' rompere il coperchio in questo punto ---
    # Il rapporto di Bowen non e' un innesco, e' il carattere della giornata:
    # dice se l'energia del Sole finisce a scaldare l'aria o a evaporare
    # acqua, e quindi se avremo celle isolate e violente o temporali diffusi.
    sensible = fields.field("ashfl", hour, latitudes, longitudes)
    latent = fields.field("alhfl", hour, latitudes, longitudes)
    bowen = (
        bowen_ratio(sensible, latent)
        if sensible is not None and latent is not None else None
    )

    orography = fields.field("hsurf", hour, latitudes, longitudes)
    land_fraction = fields.field("fr_land", hour, latitudes, longitudes)
    upslope = None
    if (
        orography is not None
        and u_wind_10m is not None
        and v_wind_10m is not None
    ):
        try:
            upslope = upslope_flow(
                u_wind_10m, v_wind_10m, orography, latitudes, longitudes
            )
        except ValueError:
            upslope = None

    breeze = None
    if (
        land_fraction is not None
        and convergence_10m is not None
        and u_wind_10m is not None
        and v_wind_10m is not None
    ):
        try:
            breeze = sea_breeze_lift(
                convergence_10m,
                u_wind_10m,
                v_wind_10m,
                land_fraction,
                latitudes,
                longitudes,
            )
        except ValueError:
            breeze = None

    trigger = None
    if upslope is not None or breeze is not None or convergence_10m is not None:
        trigger = trigger_index(
            upslope_ms=upslope,
            sea_breeze=breeze,
            convergence=convergence_10m,
        ) * 100.0

    # --- Ipotesi temporale: fusione delle prove e delle contraddizioni ---
    # L'indice trigger e' pubblicato in percentuale, mentre la fusione lavora
    # fra zero e uno. Ogni componente resta disponibile separatamente: la
    # probabilita' finale non e' una scatola nera.
    storm_evidence = intelligent_storm_probability(
        lpi,
        cape_ml,
        cin_ml,
        trigger / 100.0 if trigger is not None else None,
        cell_km,
        updraft_ms=updraft,
        cape_mu=cape_mu,
        surface_rh=surface_rh,
        mid_level_rh=mid_level_rh,
        cloud_base_m=cloud_base,
        omega_700=omega_700,
        front_distance_km=nearest_front_km,
        shear=deep_layer_shear,
        helicity=helicity,
        previous_lightning_potential=previous_lpi,
        next_lightning_potential=next_lpi,
        lpi_threshold=STORM_LPI_THRESHOLD,
        event_radius_km=STORM_EVENT_RADIUS_KM,
        smoothing_radius_km=STORM_SMOOTHING_RADIUS_KM,
    )
    probability = storm_evidence["probability"] * 100.0

    # Il riepilogo va calcolato a piena risoluzione, prima di ridurre la
    # griglia: e' quello che finisce nel controllo qualita', e un massimo
    # mediato non sarebbe piu' un massimo.
    summary = summarize_storms(
        probability,
        updraft,
        storm_evidence["confidence"] * 100.0,
        storm_evidence["contradiction"] * 100.0,
    )

    # Griglia dimezzata per il peso del file. Il modo di aggregare cambia da
    # campo a campo: media dove il campo e' gia' liscio, massimo dove il
    # segnale sta in pochi punti appuntiti, valore piu' vicino per i codici.
    def reduce_field(values, how, decimals):
        if values is None:
            return None
        return clean_for_json(coarsen(values, STORM_COARSEN, how), decimals)

    payload = {
        "method": STORM_METHOD,
        "eventRadiusKm": STORM_EVENT_RADIUS_KM,
        "smoothingRadiusKm": STORM_SMOOTHING_RADIUS_KM,
        "lpiThreshold": STORM_LPI_THRESHOLD,
        "cellKm": round(cell_km * STORM_COARSEN, 3),
        "probability": reduce_field(probability, "mean", 0),
        "directEvidence": reduce_field(
            storm_evidence["direct"] * 100.0, "mean", 0
        ),
        "environmentSupport": reduce_field(
            storm_evidence["environment"] * 100.0, "mean", 0
        ),
        "instabilitySupport": reduce_field(
            storm_evidence["instability"] * 100.0, "mean", 0
        ),
        "moistureSupport": reduce_field(
            storm_evidence["moisture"] * 100.0, "mean", 0
        ),
        "liftSupport": reduce_field(
            storm_evidence["lift"] * 100.0, "mean", 0
        ),
        "temporalSupport": reduce_field(
            storm_evidence["temporal"] * 100.0, "mean", 0
        ),
        "organisationSupport": reduce_field(
            storm_evidence["organisation"] * 100.0, "mean", 0
        ),
        "contradiction": reduce_field(
            storm_evidence["contradiction"] * 100.0, "mean", 0
        ),
        "confidence": reduce_field(
            storm_evidence["confidence"] * 100.0, "mean", 0
        ),
        "lpi": reduce_field(lpi, "max", 1),
        "updraft": reduce_field(updraft, "max", 1),
        "ascent": reduce_field(ascent, "mean", 1),
        "potentialUpdraft": reduce_field(
            potential_updraft(cape_ml) if cape_ml is not None else None,
            "mean",
            0,
        ),
        "cape": reduce_field(cape_ml, "mean", 0),
        "capeMu": reduce_field(cape_mu, "mean", 0),
        "shear": reduce_field(deep_layer_shear, "mean", 1),
        "helicity": reduce_field(helicity, "max", 0),
        "cloudBase": reduce_field(cloud_base, "mean", 0),
        "freezingLevel": reduce_field(freezing_level, "mean", 0),
        "hail": reduce_field(hail, "max", 0),
        "downburst": reduce_field(downburst, "max", 0),
        "mode": reduce_field(mode, "nearest", 0),
        "trigger": reduce_field(trigger, "max", 0),
        "bowen": reduce_field(bowen, "mean", 2),
        "upslope": reduce_field(upslope, "max", 2),
        "seaBreeze": (
            reduce_field(breeze * 1.0e5, "max", 1) if breeze is not None else None
        ),
        "summary": summary,
    }
    if include_native:
        # Questi campi restano in memoria soltanto durante la pipeline. Servono
        # a bollettino e meteogrammi sulla griglia nativa e vengono rimossi
        # prima di serializzare il file pubblico dei temporali.
        payload["_native"] = {
            "probability": probability,
            "confidence": storm_evidence["confidence"] * 100.0,
            "contradiction": storm_evidence["contradiction"] * 100.0,
            "direct": storm_evidence["direct"] * 100.0,
            "environment": storm_evidence["environment"] * 100.0,
        }
    return payload


def prepare_icon_hazard_fields(run_dt):
    """Download real convective and 700-hPa hazard fields for the ICON run.

    These files replace the former temperature-derived CAPE and constant
    convergence placeholders.  Failure is non-fatal: the hazard layer is then
    explicitly unavailable instead of being filled with synthetic values.
    """
    run_tag = run_dt.strftime("%Y%m%d%H")
    common = f"ICON_2I_SURFACE_PRESSURE_LEVELS_{run_tag}"
    run_base = f"{NWP_DIRECT_BASE}/{NWP_DIRECTORY_ID}/{run_tag}"
    pressure_file_700 = f"{common}_isobaricInhPa-700.grib"
    pressure_file_500 = f"{common}_isobaricInhPa-500.grib"
    surface_file = f"{common}_surface-0.grib"
    height_file_10 = f"{common}_heightAboveGround-10.grib"
    shear_layer_file = f"{common}_heightAboveGroundLayer-6000.grib"
    requests_to_make = {
        "cape_ml": (
            f"{run_base}/CAPE_ML/{common}_atmML-0.grib",
            "cape_ml.grib",
        ),
        "cin_ml": (
            f"{run_base}/CIN_ML/{common}_atmML-0.grib",
            "cin_ml.grib",
        ),
        "t700": (
            f"{run_base}/T/{pressure_file_700}",
            "t700.grib",
        ),
        "u700": (
            f"{run_base}/U/{pressure_file_700}",
            "u700.grib",
        ),
        "v700": (
            f"{run_base}/V/{pressure_file_700}",
            "v700.grib",
        ),
        # Mid-tropospheric moisture (700 hPa) and deep-layer shear (500 hPa
        # wind relative to the 10 m wind) for the ingredients-based convective
        # probability; 500-hPa geopotential (FI) for the isohypse overlay.
        # All optional: their absence only removes those ingredients/layer, it
        # never disables the convective layer.
        "q700": (
            f"{run_base}/QV/{pressure_file_700}",
            "q700.grib",
        ),
        "u500": (
            f"{run_base}/U/{pressure_file_500}",
            "u500.grib",
        ),
        "v500": (
            f"{run_base}/V/{pressure_file_500}",
            "v500.grib",
        ),
        "fi500": (
            f"{run_base}/FI/{pressure_file_500}",
            "fi500.grib",
        ),
        # --- SEZIONE TEMPORALI ---
        # ICON-2I gira a 2,2 km e risolve esplicitamente le celle convettive:
        # questi campi sono la sua diagnostica diretta, non una ricostruzione.
        # Tutti opzionali, come i precedenti: se MeteoHub non li espone la
        # sezione temporali resta assente e il resto del sito non se ne accorge.
        #
        # LPI, indice di potenziale fulminazione, calcolato dalla microfisica
        # (graupel, acqua sopraffusa, ghiaccio) dentro le correnti ascendenti.
        "lpi": (
            f"{run_base}/LPI/{surface_file}",
            "lpi.grib",
        ),
        # Elicita' dell'updraft: updraft che ruota, cioe' mesociclone.
        "uh_max": (
            f"{run_base}/UH_MAX/{common}_heightAboveSeaLayer-2000.grib",
            "uh_max.grib",
        ),
        # Shear di massa 0-6 km calcolato dal modello sui livelli veri: molto
        # meglio della differenza fra vento a 500 hPa e vento a 10 m, che
        # approssima lo stesso strato ignorando tutto quello che c'e' in mezzo.
        "wshear_u": (
            f"{run_base}/WSHEAR_U/{shear_layer_file}",
            "wshear_u.grib",
        ),
        "wshear_v": (
            f"{run_base}/WSHEAR_V/{shear_layer_file}",
            "wshear_v.grib",
        ),
        # CAPE della particella piu' instabile: confrontata con quella dello
        # strato mescolato dice se l'instabilita' e' al suolo o sollevata.
        "cape_con": (
            f"{run_base}/CAPE_CON/{surface_file}",
            "cape_con.grib",
        ),
        # Punto di rugiada a 2 m: con la temperatura da la base delle nubi.
        "td_2m": (
            f"{run_base}/TD_2M/{common}_heightAboveGround-2.grib",
            "td_2m.grib",
        ),
        # Zero termico: quanto fonde un chicco di grandine scendendo.
        "hzerocl": (
            f"{run_base}/HZEROCL/{common}_isothermZero-0.grib",
            "hzerocl.grib",
        ),
        # Graupel accumulato: il precursore della grandine nella microfisica.
        "graupel": (
            f"{run_base}/GRAU_GSP/{surface_file}",
            "graupel.grib",
        ),
        # Raffica massima: il downburst.
        "vmax_10m": (
            f"{run_base}/VMAX_10M/{height_file_10}",
            "vmax_10m.grib",
        ),
        # Omega a 500 e 850 hPa. A 700 hPa e' gia' scaricato dal ramo
        # frontale; qui servono gli altri due perche' il nucleo di una cella
        # non sta sempre allo stesso livello.
        "omega500": (
            f"{run_base}/OMEGA/{pressure_file_500}",
            "omega500.grib",
        ),
        "omega850": (
            f"{run_base}/OMEGA/{common}_isobaricInhPa-850.grib",
            "omega850.grib",
        ),
        # --- INNESCO DAL TERRENO ---
        # Flussi di calore al suolo: il loro rapporto e' il numero di Bowen,
        # cioe' come si divide l'energia solare fra scaldare l'aria e
        # evaporare acqua. E' l'effetto atmosferico dell'umidita' del suolo, e
        # costa cinque volte meno che scaricare l'umidita' del suolo stessa.
        "ashfl": (
            f"{run_base}/ASHFL_S/{surface_file}",
            "ashfl.grib",
        ),
        "alhfl": (
            f"{run_base}/ALHFL_S/{surface_file}",
            "alhfl.grib",
        ),
        # Costanti nel tempo: maschera terra-mare per la brezza, orografia
        # per la risalita forzata dal rilievo.
        "fr_land": (
            f"{run_base}/FR_LAND/{surface_file}",
            "fr_land.grib",
        ),
        "hsurf": (
            f"{run_base}/HSURF/{surface_file}",
            "hsurf_storm.grib",
        ),
    }
    optional_fields = {
        "t700", "u700", "v700", "q700", "u500", "v500", "fi500",
        "lpi", "uh_max", "wshear_u", "wshear_v", "cape_con", "td_2m",
        "hzerocl", "graupel", "vmax_10m", "omega500", "omega850",
        "ashfl", "alhfl", "fr_land", "hsurf",
    }
    if os.path.exists(HAZARD_TEMP_DIR):
        shutil.rmtree(HAZARD_TEMP_DIR)
    os.makedirs(HAZARD_TEMP_DIR)
    paths = {}
    print(
        "2b. Scarico ML-CAPE/CIN, T/U/V/QV a 700 hPa e U/V/FI a 500 hPa reali "
        "ICON-2I…",
        flush=True,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            for name, (url, filename) in requests_to_make.items():
                destination = os.path.join(HAZARD_TEMP_DIR, filename)
                futures[executor.submit(download_grib_file, url, destination)] = (
                    name,
                    destination,
                )
            for future in as_completed(futures):
                name, destination = futures[future]
                try:
                    future.result()
                except Exception as error:
                    if name in optional_fields:
                        print(f"   {name} non disponibile: {error}", flush=True)
                        continue
                    raise
                paths[name] = destination
                print(
                    f"   {name}: {os.path.getsize(destination) / 1048576:.1f} MB",
                    flush=True,
                )
        fields = IconRunFields(paths)
        cape_cin_hours = sorted(
            fields.hours.get("cape_ml", set())
            & fields.hours.get("cin_ml", set())
        )
        if len(cape_cin_hours) < 70:
            fields.close()
            raise ValueError(
                f"solo {len(cape_cin_hours)} scadenze CAPE/CIN disponibili"
            )
        print(
            f"   Diagnostica convettiva pronta: {len(cape_cin_hours)} ore.",
            flush=True,
        )
        return fields
    except Exception as error:
        print(
            "   Diagnostica convettiva non disponibile; "
            f"pubblico il layer come assente: {error}",
            flush=True,
        )
        return None


def get_latest_run_files():
    print("1. Cerco dati su MeteoHub...", flush=True)
    try:
        r = requests.get(API_LIST_URL, timeout=30)
        r.raise_for_status()
        items = r.json()
    except Exception as e:
        print(f"Errore connessione API: {e}")
        return None, []

    runs = {}
    for item in items:
        if isinstance(item, dict) and 'date' in item and 'run' in item:
            key = f"{item['date']} {item['run']}"
            runs.setdefault(key, []).append(item['filename'])

    if not runs:
        return None, []
    latest_key = sorted(runs.keys())[-1]
    run_dt = datetime.strptime(latest_key, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    return run_dt, sorted(runs[latest_key])


def calculate_rh_numpy(temp_k, dew_k):
    T = temp_k - 273.15
    Td = dew_k - 273.15
    a = 17.625
    b = 243.04

    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.exp((a * Td) / (b + Td))
        denominator = np.exp((a * T) / (b + T))
        rh = 100 * (numerator / denominator)

    return np.clip(rh, 0, 100)


def calculate_heat_index_celsius(temp_c, humidity):
    temp_f = temp_c * 9.0 / 5.0 + 32.0
    rh = np.clip(humidity, 0.0, 100.0)
    simple = 0.5 * (temp_f + 61.0 + (temp_f - 68.0) * 1.2 + rh * 0.094)
    preliminary = (simple + temp_f) / 2.0

    heat_index_f = (
        -42.379
        + 2.04901523 * temp_f
        + 10.14333127 * rh
        - 0.22475541 * temp_f * rh
        - 0.00683783 * temp_f**2
        - 0.05481717 * rh**2
        + 0.00122874 * temp_f**2 * rh
        + 0.00085282 * temp_f * rh**2
        - 0.00000199 * temp_f**2 * rh**2
    )

    low_humidity = (rh < 13.0) & (temp_f >= 80.0) & (temp_f <= 112.0)
    low_adjustment = ((13.0 - rh) / 4.0) * np.sqrt(
        np.maximum(0.0, (17.0 - np.abs(temp_f - 95.0)) / 17.0)
    )
    heat_index_f = np.where(low_humidity, heat_index_f - low_adjustment, heat_index_f)

    high_humidity = (rh > 85.0) & (temp_f >= 80.0) & (temp_f <= 87.0)
    high_adjustment = ((rh - 85.0) / 10.0) * ((87.0 - temp_f) / 5.0)
    heat_index_f = np.where(high_humidity, heat_index_f + high_adjustment, heat_index_f)
    heat_index_f = np.where(preliminary >= 80.0, heat_index_f, temp_f)
    return (heat_index_f - 32.0) * 5.0 / 9.0


def calculate_wind_chill_celsius(temp_c, wind_kmh):
    wind_factor = np.power(wind_kmh, 0.16)
    return (
        13.12
        + 0.6215 * temp_c
        - 11.37 * wind_factor
        + 0.3965 * temp_c * wind_factor
    )


def calculate_feels_like(temp_c, humidity, u_wind, v_wind):
    result = np.array(temp_c, dtype=float, copy=True)
    speed_kmh = np.hypot(u_wind, v_wind) * 3.6
    finite_temp = np.isfinite(temp_c)

    cold = finite_temp & np.isfinite(speed_kmh) & (temp_c <= 10.0) & (speed_kmh > 4.8)
    wind_chill = calculate_wind_chill_celsius(temp_c, speed_kmh)
    result = np.where(cold, wind_chill, result)

    hot = finite_temp & np.isfinite(humidity) & (temp_c >= 26.7)
    heat_index = calculate_heat_index_celsius(temp_c, humidity)
    result = np.where(hot, heat_index, result)
    return np.where(finite_temp, result, np.nan)


def extract_raw_grid(ds, mask, var_names):
    try:
        var_key = next((k for k in var_names if k in ds), None)
        if not var_key:
            return None
        d_masked = ds[var_key].where(mask, drop=True)
        return d_masked.values
    except Exception:
        return None


def try_open_cloud_dataset(grib_path):
    candidates = [
        {'filter_by_keys': {'shortName': 'tcc'}},
        {'filter_by_keys': {'shortName': 'tcdc'}},
        {'filter_by_keys': {'shortName': 'clct'}},
        {'filter_by_keys': {'shortName': 'cc'}},
    ]
    candidates += [
        {'filter_by_keys': {'typeOfLevel': 'entireAtmosphere'}},
        {'filter_by_keys': {'typeOfLevel': 'atmosphere'}},
        {'filter_by_keys': {'typeOfLevel': 'surface'}},
    ]

    for bk in candidates:
        try:
            ds = xr.open_dataset(grib_path, engine='cfgrib', backend_kwargs=bk)
            if ds is not None and len(ds.data_vars) > 0:
                return ds
        except Exception:
            continue

    return None


def normalize_cloud_to_percent(cloud_arr):
    c = np.asarray(cloud_arr, dtype=float)
    if c.size == 0:
        return c

    mx = float(np.nanmax(c)) if np.isfinite(np.nanmax(c)) else 0.0
    if mx <= 1.01:
        c = c * 100.0

    return np.clip(c, 0.0, 100.0)


def compute_downsample_factor(ny, nx, max_pixels=MAX_PIXELS):
    pixels = int(ny) * int(nx)
    if pixels <= max_pixels:
        return 1
    f = int(np.ceil(np.sqrt(pixels / max_pixels)))
    return max(1, f)


def downsample_2d(arr2d, f):
    if f <= 1:
        return arr2d
    return arr2d[::f, ::f]


def downsample_1d(arr1d, f):
    if f <= 1:
        return arr1d
    return arr1d[::f]



def select_step(ds, step_value, index=0):
    """Select a forecast step by value, falling back to positional index only when needed."""
    if ds is None or 'step' not in ds.sizes:
        return ds
    try:
        return ds.sel(step=step_value)
    except Exception:
        if ds.sizes.get('step', 1) > 1:
            return ds.isel(step=index)
        return ds


def finite_or_none(arr, shape=None):
    if arr is None:
        return None
    out = np.asarray(arr, dtype=float)
    if shape is not None and out.shape != shape:
        return None
    return out


def clean_for_json(arr, decimals):
    rounded = np.round(np.asarray(arr, dtype=float), decimals).flatten()
    return [None if not np.isfinite(float(v)) else float(v) for v in rounded]


def interpolate_native_field(payload, target_latitudes, target_longitudes):
    """Interpolate a regular native-grid diagnostic onto the surface grid."""
    if not payload:
        return None
    values = np.asarray(payload.get("values"), dtype=float)
    latitudes = np.asarray(payload.get("latitudes"), dtype=float)
    longitudes = np.asarray(payload.get("longitudes"), dtype=float)
    if (
        values.ndim != 2
        or latitudes.ndim != 1
        or longitudes.ndim != 1
        or values.shape != (latitudes.size, longitudes.size)
    ):
        return None
    field = xr.DataArray(
        values,
        coords={"latitude": latitudes, "longitude": longitudes},
        dims=("latitude", "longitude"),
    )
    if latitudes[0] > latitudes[-1]:
        field = field.sortby("latitude")
    if longitudes[0] > longitudes[-1]:
        field = field.sortby("longitude")
    interpolated = field.interp(
        latitude=xr.DataArray(
            np.asarray(target_latitudes, dtype=float), dims=("latitude",)
        ),
        longitude=xr.DataArray(
            np.asarray(target_longitudes, dtype=float), dims=("longitude",)
        ),
    )
    result = np.asarray(interpolated.values, dtype=float)
    expected = (len(target_latitudes), len(target_longitudes))
    return result if result.shape == expected else None


def iso_z(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def temperature_celsius(values):
    """Normalize an ICON temperature field to Celsius without guessing gaps."""
    if values is None:
        return None
    result = np.asarray(values, dtype=float)
    finite = result[np.isfinite(result)]
    if finite.size and float(np.nanmedian(finite)) > 150.0:
        result = result - 273.15
    return result


def process_data():
    run_dt, file_list = get_latest_run_files()
    if not file_list:
        print("Nessun dato trovato.")
        sys.exit(0)

    print(f"2. Elaboro Run: {run_dt} ({len(file_list)} files)", flush=True)

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    catalog = []
    step_errors = []
    front_qc_hours = []
    hazard_qc_hours = []
    front_analysis_summary = {}
    front_pipeline_diagnostics = {}
    bulletin_history = {}
    meteogram_archive = None
    icon_front_analyzer = prepare_icon_front_analyzer(run_dt)
    if icon_front_analyzer is None:
        raise RuntimeError(
            "analisi frontale ICON-2I obbligatoria non disponibile; "
            "mantengo l'ultima pubblicazione valida"
        )
    icon_hazard_fields = prepare_icon_hazard_fields(run_dt)
    for idx, filename in enumerate(file_list):
        print(f"   [{idx+1:02d}] DL {filename}...", end=" ", flush=True)

        # Un singolo tentativo rendeva l'intero aggiornamento ostaggio di una
        # disconnessione passeggera di MeteoHub: se la lista contiene un solo
        # file, un "read timed out" faceva terminare la pipeline senza dati.
        # Ritento la stessa richiesta, con attesa crescente. La lettura ha un
        # limite piu' largo perche' questi GRIB pesano decine di megabyte e il
        # timeout scatta sull'inattivita', non sulla durata totale.
        downloaded = False
        for attempt in range(1, 4):
            try:
                with requests.get(
                    f"{API_DOWNLOAD_URL}/{filename}",
                    stream=True,
                    timeout=(30, 180),
                ) as r:
                    r.raise_for_status()
                    with open(TEMP_FILE, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            f.write(chunk)
                downloaded = True
                print("OK", end=" ", flush=True)
                break
            except Exception as e:
                if attempt == 3:
                    print(f"KO ({e})", flush=True)
                else:
                    print(f"retry {attempt}/3 ({e})...", end=" ", flush=True)
                    time.sleep(attempt * 5)
        if not downloaded:
            continue

        if os.path.exists(f"{TEMP_FILE}.idx"):
            os.remove(f"{TEMP_FILE}.idx")

        # Apertura con i TUOI blocchi try...except separati (sicurissimi)
        try:
            ds_wind = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
            
            ds_thermo = None
            try: ds_thermo = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
            except: pass

            ds_press = None
            try: ds_press = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}})
            except: pass

            ds_rain = None
            try: ds_rain = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'stepType': 'accum'}})
            except: pass

            ds_cloud = try_open_cloud_dataset(TEMP_FILE)
            
        except Exception as e:
            print(f" Skip (Grib Error: {e})")
            continue

        steps = range(ds_wind.sizes.get('step', 1))

        for i in steps:
            step_hours = None
            try:
                if ds_wind.sizes.get('step', 1) > 1:
                    dw_step = ds_wind.isel(step=i)
                    raw_step = ds_wind.step.values[i]
                else:
                    dw_step = ds_wind
                    raw_step = ds_wind.step.values

                step_hours = int(raw_step / np.timedelta64(1, 'h')) if isinstance(raw_step, np.timedelta64) else int(raw_step)

                # MASCHERA
                dw_step = dw_step.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
                mask = (
                    (dw_step.latitude >= LAT_MIN - 1.0e-6)
                    & (dw_step.latitude <= LAT_MAX + 1.0e-6)
                    & (dw_step.longitude >= LON_MIN - 1.0e-6)
                    & (dw_step.longitude <= LON_MAX + 1.0e-6)
                )

                cut_w = dw_step.where(mask, drop=True)
                if cut_w.latitude.size == 0:
                    continue

                # VENTO
                u_key = next((k for k in ['u10', 'u'] if k in cut_w), None)
                v_key = next((k for k in ['v10', 'v'] if k in cut_w), None)
                if not u_key or not v_key: continue

                u_val = np.asarray(cut_w[u_key].values, dtype=float)
                v_val = np.asarray(cut_w[v_key].values, dtype=float)

                lat = cut_w.latitude.values
                lon = cut_w.longitude.values
                if lat.ndim > 1: lat = lat[:, 0]
                if lon.ndim > 1: lon = lon[0, :]

                ny, nx = u_val.shape
                f = compute_downsample_factor(ny, nx, MAX_PIXELS)
                if f > 1:
                    u_val, v_val = downsample_2d(u_val, f), downsample_2d(v_val, f)
                    lat, lon = downsample_1d(lat, f), downsample_1d(lon, f)
                    ny, nx = u_val.shape

                la1, lo1 = float(lat[0]), float(lon[0])
                dx = float(abs(lon[1] - lon[0])) if nx > 1 else 0.0
                dy = float(abs(lat[0] - lat[1])) if ny > 1 else 0.0
                lo2 = lo1 + (nx - 1) * dx
                la2 = la1 - (ny - 1) * dy
                if meteogram_archive is None:
                    meteogram_archive = MeteogramArchive(
                        lat,
                        lon,
                        run_time=iso_z(run_dt),
                    )

                # 1) TEMP e RH
                temp_c = np.full_like(u_val, np.nan, dtype=float)
                rh_val = np.full_like(u_val, np.nan, dtype=float)

                if ds_thermo is not None:
                    dt_step = select_step(ds_thermo, raw_step, i)
                    dt_step = dt_step.sortby('latitude', ascending=False).sortby('longitude', ascending=True)

                    t_raw = extract_raw_grid(dt_step, mask, ['t2m', 't'])
                    d_raw = extract_raw_grid(dt_step, mask, ['d2m', '2d'])

                    if t_raw is not None and t_raw.ndim == 2:
                        if f > 1: t_raw = downsample_2d(t_raw, f)
                        if t_raw.shape == u_val.shape:
                            temp_c = t_raw - 273.15
                            if d_raw is not None and d_raw.ndim == 2:
                                if f > 1: d_raw = downsample_2d(d_raw, f)
                                if d_raw.shape == u_val.shape:
                                    rh_val = calculate_rh_numpy(t_raw, d_raw)

                # Temperatura percepita: Heat Index con caldo, Wind Chill con freddo.
                feels_like = calculate_feels_like(temp_c, rh_val, u_val, v_val)

                # 2) PRESSIONE
                press = np.full_like(u_val, np.nan, dtype=float)
                if ds_press is not None:
                    dp_step = select_step(ds_press, raw_step, i)
                    dp_step = dp_step.sortby('latitude', ascending=False).sortby('longitude', ascending=True)

                    p_raw = extract_raw_grid(dp_step, mask, ['pmsl', 'prmsl', 'msl'])
                    if p_raw is not None and p_raw.ndim == 2:
                        if f > 1: p_raw = downsample_2d(p_raw, f)
                        if p_raw.shape == u_val.shape:
                            p_raw = np.asarray(p_raw, dtype=float)
                            press = (p_raw / 100.0) if np.nanmax(p_raw) > 80000 else p_raw

                # Non inventare pressione standard: se manca resta null nel JSON.

                # 3) PIOGGIA
                rain = np.full_like(u_val, np.nan, dtype=float)
                if ds_rain is not None:
                    dr_step = select_step(ds_rain, raw_step, i)
                    dr_step = dr_step.sortby('latitude', ascending=False).sortby('longitude', ascending=True)

                    r_raw = extract_raw_grid(dr_step, mask, ['tp', 'tot_prec'])
                    if r_raw is not None and r_raw.ndim == 2:
                        if f > 1: r_raw = downsample_2d(r_raw, f)
                        if r_raw.shape == u_val.shape:
                            rain = np.asarray(r_raw, dtype=float)
                            # ICON total precipitation is normally encoded as
                            # an accumulation from the start of the run.
                            # Convert it to the amount arriving in this step:
                            # hazard algorithms must not keep using rain that
                            # fell many hours earlier.
                            rain_steps = np.atleast_1d(ds_rain.step.values)
                            rain_index = next(
                                (
                                    position
                                    for position, value in enumerate(rain_steps)
                                    if value == raw_step
                                ),
                                i if i < len(rain_steps) else -1,
                            )
                            if rain_index > 0:
                                previous_step_value = rain_steps[rain_index - 1]
                                previous_step = select_step(
                                    ds_rain,
                                    previous_step_value,
                                    rain_index - 1,
                                )
                                previous_step = previous_step.sortby(
                                    'latitude', ascending=False
                                ).sortby('longitude', ascending=True)
                                previous_raw = extract_raw_grid(
                                    previous_step, mask, ['tp', 'tot_prec']
                                )
                                if previous_raw is not None and previous_raw.ndim == 2:
                                    if f > 1:
                                        previous_raw = downsample_2d(previous_raw, f)
                                    if previous_raw.shape == u_val.shape:
                                        rain = np.maximum(
                                            rain - np.asarray(previous_raw, dtype=float),
                                            0.0,
                                        )

                # 4) NUVOLOSITÀ
                cloud = np.full_like(u_val, np.nan, dtype=float)
                if ds_cloud is not None:
                    dc_step = select_step(ds_cloud, raw_step, i)
                    dc_step = dc_step.sortby('latitude', ascending=False).sortby('longitude', ascending=True)

                    c_raw = extract_raw_grid(dc_step, mask, ['tcc', 'tcdc', 'clct', 'cc', 'tcc_total', 'totalCloudCover'])
                    if c_raw is not None and c_raw.ndim == 2:
                        if f > 1: c_raw = downsample_2d(c_raw, f)
                        if c_raw.shape == u_val.shape:
                            cloud = normalize_cloud_to_percent(c_raw)

                # EXPORT JSON
                valid_dt = run_dt + timedelta(hours=step_hours)
                iso_date = iso_z(valid_dt)
                fronts = {"type": "FeatureCollection", "features": []}
                front_method = None
                front_source = None
                front_level = None
                front_valid_time = None

                if icon_front_analyzer is not None:
                    try:
                        fronts = icon_front_analyzer.analyze(step_hours)
                        front_method = ICON_FRONT_METHOD
                        front_source = "ICON-2I"
                        front_level = "850 hPa"
                        front_valid_time = iso_date
                    except Exception as front_error:
                        print(
                            f" front-{step_hours}h:{front_error}",
                            end="",
                            flush=True,
                        )

                front_properties = (
                    fronts.get("properties", {})
                    if isinstance(fronts, dict) else {}
                )
                front_analysis_status = front_properties.get(
                    "analysisStatus", "unavailable"
                )
                front_analysis_message = front_properties.get(
                    "analysisMessage",
                    "Analisi frontale non disponibile per questa ora.",
                )

                header = {
                    "nx": nx, "ny": ny, "lo1": lo1, "la1": la1, "lo2": lo2, "la2": la2,
                    "dx": dx, "dy": dy, "runTime": iso_z(run_dt), "validTime": iso_date, "refTime": iso_date, "leadHours": step_hours,
                    "feelsLikeMethod": FEELS_LIKE_METHOD,
                    "frontMethod": front_method,
                    "frontSource": front_source,
                    "frontLevel": front_level,
                    "frontValidTime": front_valid_time,
                    "frontAnalysisStatus": front_analysis_status,
                    "frontAnalysisMessage": front_analysis_message,
                    "frontFusion": bool(front_properties.get("mlFusion")),
                    "rainAccumulation": "forecast-step",
                }

                # --- DIAGNOSTICA CONVETTIVA REALE ---
                # CAPE e CIN provengono dai campi nativi ICON-2I.  La
                # convergenza deriva da u/v a 10 m e la distanza è misurata
                # dalle linee frontali effettivamente pubblicate.
                convection_prob = None
                convection_summary = {
                    "status": "unavailable",
                    "maximum": None,
                    "p95": None,
                    "validCellPct": 0.0,
                    "areaAbove40Pct": None,
                    "areaAbove70Pct": None,
                    "areaExactly80Pct": None,
                }
                convection_message = (
                    "ML-CAPE o ML-CIN ICON-2I non disponibile per questa scadenza."
                )
                cape_ml = None
                cin_ml = None
                convergence_10m = None
                nearest_front_km = None
                omega_700 = None
                deep_layer_shear = None
                mid_level_rh = None
                wind_gust_10m = None
                shear_source = None
                if icon_hazard_fields is not None:
                    try:
                        cape_ml = icon_hazard_fields.field(
                            "cape_ml", step_hours, lat, lon
                        )
                        cin_ml = icon_hazard_fields.field(
                            "cin_ml", step_hours, lat, lon
                        )
                        if cape_ml is None or cin_ml is None:
                            raise ValueError("scadenza CAPE/CIN assente")
                        convergence_10m = horizontal_convergence(
                            u_val, v_val, lat, lon, smoothing_km=10.0
                        )
                        nearest_front_km = front_distance_km(lat, lon, fronts)
                        omega_payload = (
                            icon_front_analyzer.diagnostic_field(
                                step_hours, "omega700"
                            )
                            if icon_front_analyzer is not None else None
                        )
                        omega_700 = interpolate_native_field(
                            omega_payload, lat, lon
                        )
                        # Deep-layer (0-6 km) bulk shear: 500-hPa wind relative
                        # to the 10 m wind.  Optional - absent 500-hPa fields
                        # simply drop the shear ingredient.
                        # ICON-2I pubblica lo shear 0-6 km gia' calcolato sui
                        # livelli veri del modello: quando c'e' si usa quello,
                        # perche' la differenza 500 hPa - 10 m approssima lo
                        # stesso strato ignorando tutto quello che c'e' in
                        # mezzo. Il vecchio calcolo resta come ripiego.
                        wshear_u = icon_hazard_fields.field(
                            "wshear_u", step_hours, lat, lon
                        )
                        wshear_v = icon_hazard_fields.field(
                            "wshear_v", step_hours, lat, lon
                        )
                        if wshear_u is not None and wshear_v is not None:
                            deep_layer_shear = bulk_shear(wshear_u, wshear_v)
                            shear_source = "WSHEAR 0-6 km ICON-2I"
                        else:
                            u500 = icon_hazard_fields.field(
                                "u500", step_hours, lat, lon
                            )
                            v500 = icon_hazard_fields.field(
                                "v500", step_hours, lat, lon
                            )
                            if u500 is not None and v500 is not None:
                                deep_layer_shear = np.hypot(
                                    u500 - u_val, v500 - v_val
                                )
                                shear_source = "500 hPa meno 10 m (ripiego)"
                        # Mid-tropospheric moisture: 700-hPa relative humidity
                        # from ICON T and specific humidity.  Optional.
                        mid_level_rh = None
                        t700_field = icon_hazard_fields.field(
                            "t700", step_hours, lat, lon
                        )
                        q700_field = icon_hazard_fields.field(
                            "q700", step_hours, lat, lon
                        )
                        if t700_field is not None and q700_field is not None:
                            mid_level_rh = (
                                relative_humidity_from_specific_humidity(
                                    t700_field, q700_field, 700.0
                                )
                            )
                        wind_gust_10m = icon_hazard_fields.field(
                            "vmax_10m", step_hours, lat, lon
                        )
                        convection_message = (
                            "Ingredienti fisici ICON-2I acquisiti; la probabilità "
                            "finale viene calcolata dall'algoritmo temporali unico."
                        )
                    except Exception as convection_error:
                        convection_prob = None
                        convection_message = (
                            "Diagnostica convettiva non disponibile: "
                            f"{type(convection_error).__name__}:"
                            f"{convection_error}"
                        )
                        print(
                            f" convection-{step_hours}h:{convection_error}",
                            end="",
                            flush=True,
                        )

                # Nebbia/visibilità usa esclusivamente i campi di superficie
                # reali disponibili.
                xr_rh = xr.DataArray(rh_val)
                xr_cloud = xr.DataArray(cloud)
                fog_type = classify_fog_type(
                    xr_rh, xr.DataArray(np.hypot(u_val, v_val)), xr_cloud
                )
                fog_probability = calculate_fog_probability(
                    xr_rh,
                    xr.DataArray(np.hypot(u_val, v_val)),
                    xr_cloud,
                    fog_type=fog_type,
                )
                visibility = estimate_visibility(
                    xr_rh, fog_type, fog_probability=fog_probability
                )
                hail_threat = None

                # Gelicidio: warm nose osservato sui livelli reali
                # 925/850/700 hPa, superficie sottozero e precipitazione
                # effettivamente in arrivo nel passo.
                freezing_rain = None
                t925 = None
                t850 = None
                t700 = None
                freezing_message = (
                    "Profilo termico 925/850/700 hPa non disponibile."
                )
                try:
                    t925 = interpolate_native_field(
                        icon_front_analyzer.diagnostic_field(step_hours, "t925"),
                        lat,
                        lon,
                    )
                    t850 = interpolate_native_field(
                        icon_front_analyzer.diagnostic_field(step_hours, "t850"),
                        lat,
                        lon,
                    )
                    t700 = (
                        icon_hazard_fields.field("t700", step_hours, lat, lon)
                        if icon_hazard_fields is not None else None
                    )
                    if t925 is None or t850 is None or t700 is None:
                        raise ValueError("uno o più livelli termici assenti")
                    freezing_rain = np.asarray(
                        detect_freezing_rain(
                            t925,
                            t850,
                            t700,
                            temp_c,
                            rain,
                        ),
                        dtype=float,
                    )
                    if not np.isfinite(freezing_rain).any():
                        raise ValueError("nessuna cella con profilo completo")
                    freezing_message = (
                        "Warm nose ICON-2I 925/850/700 hPa, T2m sottozero "
                        "e precipitazione del passo."
                    )
                except Exception as freezing_error:
                    freezing_rain = None
                    freezing_message = (
                        "Rischio gelicidio non disponibile: "
                        f"{type(freezing_error).__name__}:{freezing_error}"
                    )

                # Foehn: vento realmente perpendicolare alla catena a 700 hPa,
                # differenza barica nord-sud misurata su PMSL e aria secca
                # sottovento. Il campo è confinato all'arco alpino.
                foehn = None
                foehn_message = "Vento 700 hPa o gradiente alpino non disponibile."
                try:
                    u700 = (
                        icon_hazard_fields.field("u700", step_hours, lat, lon)
                        if icon_hazard_fields is not None else None
                    )
                    v700 = (
                        icon_hazard_fields.field("v700", step_hours, lat, lon)
                        if icon_hazard_fields is not None else None
                    )
                    if u700 is None or v700 is None:
                        raise ValueError("vento 700 hPa assente")
                    pressure_difference = cross_alpine_pressure_difference(
                        press,
                        lat,
                    )
                    foehn = np.asarray(
                        detect_foehn(
                            u700,
                            v700,
                            rh_val,
                            north_minus_south_pressure_hpa=pressure_difference,
                            domain_mask=alpine_domain_mask(lat, lon),
                        ),
                        dtype=float,
                    )
                    if not np.isfinite(foehn).any():
                        raise ValueError("nessuna cella con diagnostica completa")
                    foehn_message = (
                        "Vento trasversale ICON-2I a 700 hPa, differenza PMSL "
                        "nord-sud e UR sottovento, limitati all'arco alpino."
                    )
                except Exception as foehn_error:
                    foehn = None
                    foehn_message = (
                        "Indice foehn non disponibile: "
                        f"{type(foehn_error).__name__}:{foehn_error}"
                    )

                # Un solo algoritmo temporalesco alimenta mappa, bollettino e
                # meteogrammi. Il payload pubblico resta ridotto; qui si conserva
                # temporaneamente la probabilità nativa per le analisi puntuali.
                storm_payload = None
                storm_native = {}
                storm_available = False
                try:
                    storm_payload = build_storm_payload(
                        icon_hazard_fields,
                        step_hours,
                        lat,
                        lon,
                        cape_ml,
                        deep_layer_shear,
                        omega_700,
                        temp_c,
                        u_val,
                        v_val,
                        convergence_10m,
                        cin_ml,
                        rh_val,
                        mid_level_rh,
                        nearest_front_km,
                        include_native=True,
                    )
                    if storm_payload is not None:
                        storm_native = storm_payload.pop("_native", {})
                        native_probability = storm_native.get("probability")
                        if native_probability is not None:
                            convection_prob = np.asarray(
                                native_probability, dtype=float
                            )
                            convection_summary = summarize_convection(
                                convection_prob, mask=np.isfinite(temp_c)
                            )
                            convection_message = (
                                "Algoritmo temporali unico: LPI e updraft fusi con "
                                "CAPE/CIN, umidità, innesco, fronti, shear e "
                                "coerenza temporale."
                            )
                except Exception as storm_error:
                    storm_payload = None
                    storm_native = {}
                    convection_prob = None
                    convection_message = (
                        "Algoritmo temporali non disponibile: "
                        f"{type(storm_error).__name__}:{storm_error}"
                    )
                    print(
                        f" storm-{step_hours}h:{storm_error}",
                        end="",
                        flush=True,
                    )

                previous = bulletin_history.get(step_hours - 3, {})
                bulletin_inputs = build_bulletin_inputs(
                    valid_time=iso_date,
                    fronts=fronts,
                    convection_probability=(
                        convection_prob
                        if convection_prob is not None
                        else np.full_like(temp_c, np.nan)
                    ),
                    temperature=temp_c,
                    precipitation=rain,
                    cloud=cloud,
                    pressure=press,
                    u_wind=u_val,
                    v_wind=v_val,
                    hail_threat=hail_threat,
                    mask=np.isfinite(temp_c),
                    previous_temperature_mean=previous.get("temperatureMean"),
                    previous_pressure_mean=previous.get("pressureMean"),
                    trend_hours=3 if previous else None,
                    area="dominio completo ICON-2I",
                )
                nlg_details = generate_bulletin_details(bulletin_inputs)
                nlg_bulletin = nlg_details["text"]
                bulletin_history[step_hours] = {
                    "temperatureMean": bulletin_inputs.temperature.get("mean"),
                    "pressureMean": bulletin_inputs.pressure.get("mean"),
                }

                header.update(
                    {
                        "convectionMethod": (
                            CONVECTION_METHOD
                            if convection_prob is not None else None
                        ),
                        "convectionStatus": convection_summary["status"],
                        "convectionMessage": convection_message,
                        "convectionSummary": convection_summary,
                        "convectionCAPE": "ML-CAPE",
                        "nlgMethod": NLG_METHOD,
                        "hazardAvailability": {
                            "convection": convection_prob is not None,
                            "fogVisibility": True,
                            "hail": False,
                            "freezingRain": freezing_rain is not None,
                            "foehn": foehn is not None,
                        },
                        "freezingRainMethod": (
                            "icon2i-warm-nose-925-850-700-v1"
                            if freezing_rain is not None else None
                        ),
                        "freezingRainMessage": freezing_message,
                        "foehnMethod": (
                            "icon2i-cross-alpine-700-pmsl-rh-v1"
                            if foehn is not None else None
                        ),
                        "foehnMessage": foehn_message,
                    }
                )

                # --- ISOIPSE 500 hPa: altezza di geopotenziale (m) ---
                # FI e' il geopotenziale (m2/s2); l'altezza di geopotenziale e'
                # FI/g. Opzionale: se il campo manca il layer isoipse resta
                # semplicemente assente.
                geopot500 = None
                if icon_hazard_fields is not None:
                    try:
                        fi500 = icon_hazard_fields.field(
                            "fi500", step_hours, lat, lon
                        )
                        if fi500 is not None:
                            geopot500 = np.asarray(fi500, dtype=float) / 9.80665
                    except Exception as geopot_error:
                        geopot500 = None
                        print(
                            f" geopot500-{step_hours}h:{geopot_error}",
                            end="",
                            flush=True,
                        )

                step_data = {
                    "meta": header,
                    "wind_u": { "header": {**header, "parameterCategory": 2, "parameterNumber": 2}, "data": np.round(u_val, 1).flatten().tolist() },
                    "wind_v": { "header": {**header, "parameterCategory": 2, "parameterNumber": 3}, "data": np.round(v_val, 1).flatten().tolist() },
                    "temp": clean_for_json(temp_c, 1),
                    # Il browser ricava la temperatura percepita dalla stessa
                    # formula, evitando una grande matrice duplicata.
                    "feels_like": None,
                    "rain": clean_for_json(rain, 2),
                    "press": clean_for_json(press, 1),
                    "geopot500": (
                        clean_for_json(geopot500, 0)
                        if geopot500 is not None else None
                    ),
                    "rh": clean_for_json(rh_val, 0),
                    "cloud": clean_for_json(cloud, 0),
                    "convection_prob": (
                        clean_for_json(convection_prob, 1)
                        if convection_prob is not None else None
                    ),
                    "hail_threat": None,
                    "visibility": clean_for_json(visibility.values, 0),
                    "freezing_rain": (
                        clean_for_json(freezing_rain, 0)
                        if freezing_rain is not None else None
                    ),
                    "foehn": (
                        clean_for_json(foehn, 0)
                        if foehn is not None else None
                    ),
                    "nlg_bulletin": nlg_bulletin,
                    "nlg_bulletin_details": nlg_details,
                    "fronts": fronts
                }

                out_name = f"step_{step_hours}.json.gz"
                write_json_gzip_atomic(f"{TEMP_DIR}/{out_name}", step_data)

                # Sezione temporali: il payload è già stato calcolato prima del
                # bollettino, affinché ogni prodotto usi lo stesso algoritmo.
                if storm_payload is not None:
                    try:
                        storm_payload["runTime"] = iso_z(run_dt)
                        storm_payload["validTime"] = iso_date
                        # Geometria della griglia ridotta: il punto
                        # pubblicato rappresenta un blocco, quindi il suo
                        # centro e' spostato di mezza cella originale.
                        storm_payload["nx"] = -(-nx // STORM_COARSEN)
                        storm_payload["ny"] = -(-ny // STORM_COARSEN)
                        storm_payload["lo1"] = lo1 + dx * (STORM_COARSEN - 1) / 2.0
                        storm_payload["la1"] = la1 - dy * (STORM_COARSEN - 1) / 2.0
                        storm_payload["dx"] = dx * STORM_COARSEN
                        storm_payload["dy"] = dy * STORM_COARSEN
                        write_json_gzip_atomic(
                            f"{TEMP_DIR}/storm_{step_hours}.json.gz",
                            storm_payload,
                        )
                        storm_available = True
                    except Exception as storm_write_error:
                        print(
                            f" storm-write-{step_hours}h:{storm_write_error}",
                            end="",
                            flush=True,
                        )

                # Campi a 850 hPa (theta-e, T, vento) per il layer di
                # ispezione dei fronti: file separato, scaricato dal sito
                # solo quando l'utente attiva la vista 850 hPa.
                upper = None
                if icon_front_analyzer is not None:
                    try:
                        upper = icon_front_analyzer.upper_air(step_hours)
                        if upper is not None:
                            upper["runTime"] = iso_z(run_dt)
                            upper["validTime"] = iso_date
                            write_json_gzip_atomic(
                                f"{TEMP_DIR}/upper_{step_hours}.json.gz", upper
                            )
                    except Exception as upper_error:
                        print(f" upper-{step_hours}h:{upper_error}", end="", flush=True)

                if not any(x['hour'] == step_hours for x in catalog):
                    meteogram_archive.add(
                        step_hours,
                        iso_date,
                        {
                            "temperature2m": temp_c,
                            "feelsLike": feels_like,
                            "rainStep": rain,
                            "pressureMsl": press,
                            "relativeHumidity2m": rh_val,
                            "cloudCover": cloud,
                            "windU10": u_val,
                            "windV10": v_val,
                            "windGust10": wind_gust_10m,
                            "convectionProbability": convection_prob,
                            "stormConfidence": storm_native.get("confidence"),
                            "stormContradiction": storm_native.get("contradiction"),
                            "capeMl": cape_ml,
                            "cinMl": normalize_cin(cin_ml) if cin_ml is not None else None,
                            "omega700": omega_700,
                            "frontDistanceKm": nearest_front_km,
                            "visibility": visibility.values,
                            "fogProbability": np.asarray(fog_probability) * 100.0,
                            "freezingRainRisk": freezing_rain,
                            "foehnIndex": foehn,
                        },
                    )
                    catalog.append({ "file": out_name, "label": f"{valid_dt.strftime('%d/%m %H:00')} UTC", "hour": step_hours, "runTime": iso_z(run_dt), "validTime": iso_date, "leadHours": step_hours, "storm": storm_available })
                    front_qc_hours.append({
                        "leadHours": step_hours,
                        "validTime": iso_date,
                        "analysisStatus": front_analysis_status,
                        "analysisMessage": front_analysis_message,
                        "frontCount": len(fronts.get("features", [])),
                        "fronts": fronts.get("features", []),
                    })
                    hazard_qc_hours.append({
                        "leadHours": step_hours,
                        "validTime": iso_date,
                        "convectionMethod": (
                            CONVECTION_METHOD
                            if convection_prob is not None else None
                        ),
                        "convectionMessage": convection_message,
                        "convectionSummary": convection_summary,
                        "frontCount": len(fronts.get("features", [])),
                        "capeMaximum": (
                            round(float(np.nanmax(cape_ml)), 1)
                            if cape_ml is not None and np.isfinite(cape_ml).any()
                            else None
                        ),
                        "cinMinimum": (
                            round(float(np.nanmin(normalize_cin(cin_ml))), 1)
                            if cin_ml is not None
                            and np.isfinite(normalize_cin(cin_ml)).any()
                            else None
                        ),
                        "cinMissingPct": (
                            round(
                                float(
                                    np.mean(~np.isfinite(normalize_cin(cin_ml)))
                                    * 100.0
                                ),
                                2,
                            )
                            if cin_ml is not None else None
                        ),
                        "convergenceP99": (
                            float(np.round(
                                np.nanpercentile(convergence_10m, 99), 7
                            ))
                            if convergence_10m is not None
                            and np.isfinite(convergence_10m).any()
                            else None
                        ),
                        "freezingRainMessage": freezing_message,
                        "freezingRainCells": (
                            int(np.count_nonzero(freezing_rain >= 1.0))
                            if freezing_rain is not None else None
                        ),
                        "foehnMessage": foehn_message,
                        "foehnCells": (
                            int(np.count_nonzero(foehn >= 1.0))
                            if foehn is not None else None
                        ),
                    })

            except Exception as e:
                step_errors.append((step_hours, str(e)))
                print(
                    f" step-{step_hours if step_hours is not None else '?'}:"
                    f"{type(e).__name__}:{e}",
                    end="",
                    flush=True,
                )
                continue

        print(" -> Done")

    if icon_front_analyzer is not None:
        front_analysis_summary = dict(
            icon_front_analyzer.analysis_summary or {}
        )
        front_pipeline_diagnostics = {
            str(hour): dict(values)
            for hour, values in icon_front_analyzer._pipeline_diag.items()
        }
        icon_front_analyzer.close()
    if icon_hazard_fields is not None:
        icon_hazard_fields.close()
    if os.path.exists(FRONT_TEMP_DIR):
        shutil.rmtree(FRONT_TEMP_DIR)
    if os.path.exists(HAZARD_TEMP_DIR):
        shutil.rmtree(HAZARD_TEMP_DIR)
    if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
    if os.path.exists(f"{TEMP_FILE}.idx"): os.remove(f"{TEMP_FILE}.idx")

    if catalog:
        catalog.sort(key=lambda x: x['hour'])
        expected_hours = set(icon_front_analyzer.available_hours)
        actual_hours = {int(item["hour"]) for item in catalog}
        missing_hours = sorted(expected_hours - actual_hours)
        if step_errors or missing_hours:
            preview = "; ".join(
                f"+{hour}h {message}" for hour, message in step_errors[:5]
            )
            raise RuntimeError(
                f"output ICON incompleto: {len(step_errors)} errori, "
                f"ore mancanti {missing_hours}; {preview}"
            )
        front_qc_hours.sort(key=lambda item: item["leadHours"])
        write_json_atomic(
            f"{TEMP_DIR}/front_qc.json",
            {
                "schemaVersion": 1,
                "runTime": iso_z(run_dt),
                "model": "ICON-2I",
                "method": ICON_FRONT_METHOD,
                "analysisSummary": front_analysis_summary,
                "pipelineByHour": front_pipeline_diagnostics,
                "hours": front_qc_hours,
            },
        )
        hazard_qc_hours.sort(key=lambda item: item["leadHours"])
        convection_qc_failures = []
        for hour_qc in hazard_qc_hours:
            summary = hour_qc.get("convectionSummary") or {}
            high_area = summary.get("areaAbove70Pct")
            fixed_80_area = summary.get("areaExactly80Pct")
            if high_area is not None and high_area > MAX_CONVECTIVE_HIGH_AREA_PCT:
                convection_qc_failures.append(
                    f"+{hour_qc['leadHours']}h area>=70%: {high_area}%"
                )
            if fixed_80_area is not None and fixed_80_area > MAX_FIXED_80_AREA_PCT:
                convection_qc_failures.append(
                    f"+{hour_qc['leadHours']}h area=80%: {fixed_80_area}%"
                )
        if convection_qc_failures:
            raise RuntimeError(
                "QC convettivo anti-saturazione fallito; mantengo il run "
                "pubblicato precedente: " + "; ".join(convection_qc_failures[:8])
            )
        write_json_atomic(
            f"{TEMP_DIR}/hazard_qc.json",
            {
                "schemaVersion": 1,
                "runTime": iso_z(run_dt),
                "model": "ICON-2I",
                "convectionMethod": CONVECTION_METHOD,
                "antiSaturationThresholds": {
                    "maximumAreaAbove70Pct": MAX_CONVECTIVE_HIGH_AREA_PCT,
                    "maximumAreaExactly80Pct": MAX_FIXED_80_AREA_PCT,
                },
                "nlgMethod": NLG_METHOD,
                "hours": hazard_qc_hours,
            },
        )
        write_json_atomic(f"{TEMP_DIR}/catalog.json", catalog)
        if meteogram_archive is None:
            raise RuntimeError("archivio meteogrammi non inizializzato")
        meteogram_manifest = meteogram_archive.write(
            os.path.join(TEMP_DIR, "meteograms")
        )
        if len(meteogram_manifest["hours"]) != len(catalog):
            raise RuntimeError(
                "archivio meteogrammi incompleto: "
                f"{len(meteogram_manifest['hours'])}/{len(catalog)} scadenze"
            )

        write_observations(TEMP_DIR)

        if os.path.exists(FINAL_DIR): shutil.rmtree(FINAL_DIR)
        shutil.move(TEMP_DIR, FINAL_DIR)
        print("\nELABORAZIONE COMPLETATA CON SUCCESSO.")
    else:
        print("\nNESSUN DATO VALIDO ESTRATTO.")
        sys.exit(1)

if __name__ == "__main__":
    process_data()
