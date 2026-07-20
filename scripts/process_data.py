import requests
import xarray as xr
import numpy as np
import json
import os
import sys
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from front_analysis import SynopticFrontAnalyzer, corroborate_with_reference

# --- CONFIGURAZIONE ---
DATASET_ID = "ICON_2I_SURFACE_PRESSURE_LEVELS"
API_LIST_URL = f"https://meteohub.agenziaitaliameteo.it/api/datasets/{DATASET_ID}/opendata"
API_DOWNLOAD_URL = "https://meteohub.agenziaitaliameteo.it/api/opendata"

FINAL_DIR = "data_weather"
TEMP_DIR = "temp_processing"
TEMP_FILE = "temp.grib2"
FRONT_TEMP_DIR = "temp_front_processing"
NWP_DIRECTORY_ID = "ICON-2I_SURFACE_PRESSURE_LEVELS"
NWP_DIRECT_BASE = "https://meteohub.agenziaitaliameteo.it/nwp"
ICON_FRONT_METHOD = "theta-e-850-tfp-wind-icon2i-v6-ptrough"
# ICON-2I risolve strutture di mesoscala (brezze, canalizzazioni orografiche,
# outflow temporaleschi) che il rilevatore puo' scambiare per fronti sinottici.
# Un fronte vero e' anche visibile - smussato e spostato di poche decine di km -
# nella corsa ECMWF, molto piu' rada: usiamo quella come guida di conferma e
# scartiamo i candidati ICON-2I privi di riscontro.
FRONT_CORROBORATION_BASE_KM = 180.0
FRONT_CORROBORATION_PER_HOUR_KM = 1.5
FRONT_CORROBORATION_MAX_KM = 320.0
SYNOPTIC_FRONT_CATALOG = os.path.join(
    "data_weather_ecmwf",
    "fronts_catalog.json",
)

# ============================================================
# ITALIA (inclusi Sicilia + Sardegna)
# ============================================================
LAT_MIN, LAT_MAX = 35.0, 48.5
LON_MIN, LON_MAX = 6.0, 19.5

# ============================================================
# CONTROLLO DIMENSIONE GRIGLIA
# FIX 1: Portato a 600_000 per mantenere la risoluzione nativa a 2.2km
# ============================================================
MAX_PIXELS = 600_000  
FEELS_LIKE_METHOD = "heat-index-wind-chill-v1"


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


def prepare_icon_front_analyzer(run_dt):
    """Download ICON-2I T/QV/U/V at 850 hPa and build an hourly analyzer."""
    run_tag = run_dt.strftime("%Y%m%d%H")
    common = f"ICON_2I_SURFACE_PRESSURE_LEVELS_{run_tag}"
    run_base = f"{NWP_DIRECT_BASE}/{NWP_DIRECTORY_ID}/{run_tag}"
    pressure_file = f"{common}_isobaricInhPa-850.grib"
    surface_file = f"{common}_surface-0.grib"
    mean_sea_file = f"{common}_meanSea-0.grib"
    requests_to_make = {
        "temperature": (f"{run_base}/T/{pressure_file}", "t850.grib"),
        "humidity": (f"{run_base}/QV/{pressure_file}", "q850.grib"),
        "u_wind": (f"{run_base}/U/{pressure_file}", "u850.grib"),
        "v_wind": (f"{run_base}/V/{pressure_file}", "v850.grib"),
        "orography": (f"{run_base}/HSURF/{surface_file}", "hsurf.grib"),
        "pressure": (f"{run_base}/PMSL/{mean_sea_file}", "pmsl.grib"),
    }
    # La pressione arricchisce l'analisi (firma della saccatura) ma non e'
    # indispensabile: un suo download fallito non blocca i fronti.
    optional_fields = {"pressure"}

    if os.path.exists(FRONT_TEMP_DIR):
        shutil.rmtree(FRONT_TEMP_DIR)
    os.makedirs(FRONT_TEMP_DIR)
    paths = {}
    print(
        "2a. Scarico ICON-2I T/QV/U/V a 850 hPa per 73 ore…",
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

        analyzer = SynopticFrontAnalyzer(
            paths["temperature"],
            paths["humidity"],
            paths["u_wind"],
            paths["v_wind"],
            paths["orography"],
            pressure_path=paths.get("pressure"),
            downsample=4,
            bounds=(3.0, 22.0, 33.7, 48.9),
            method=ICON_FRONT_METHOD,
            source="ICON-2I",
            tendency_window_hours=3,
        )
        if len(analyzer.available_hours) < 70:
            analyzer.close()
            raise ValueError(
                f"solo {len(analyzer.available_hours)} scadenze ICON-2I disponibili"
            )
        print(
            f"   Analisi ICON-2I pronta: {len(analyzer.available_hours)} ore.",
            flush=True,
        )
        return analyzer
    except Exception as error:
        print(f"   ICON-2I 850 hPa non disponibile: {error}", flush=True)
        return None


def load_synoptic_front_catalog():
    """Load the small ECMWF 850-hPa guide produced earlier in the workflow."""
    try:
        with open(SYNOPTIC_FRONT_CATALOG, encoding="utf-8") as source:
            items = json.load(source)
        prepared = []
        for item in items:
            valid_time = datetime.fromisoformat(
                item["validTime"].replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            prepared.append((valid_time, item))
        print(
            f"2a. Guida fronti ECMWF disponibile ({len(prepared)} scadenze).",
            flush=True,
        )
        return prepared
    except Exception as error:
        print(f"2a. Guida fronti non disponibile: {error}", flush=True)
        return []


def nearest_synoptic_front(front_catalog, valid_time):
    if not front_catalog:
        return None
    candidate_time, candidate = min(
        front_catalog,
        key=lambda pair: abs((pair[0] - valid_time).total_seconds()),
    )
    difference_hours = abs((candidate_time - valid_time).total_seconds()) / 3600.0
    return candidate if difference_hours <= 3.1 else None


def front_corroboration_radius_km(lead_hours):
    """Allowed ICON-vs-ECMWF front displacement: grows with lead time, capped."""
    radius = FRONT_CORROBORATION_BASE_KM + FRONT_CORROBORATION_PER_HOUR_KM * lead_hours
    return float(min(FRONT_CORROBORATION_MAX_KM, radius))


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


def iso_z(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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
    icon_front_analyzer = prepare_icon_front_analyzer(run_dt)
    # Serve sia come guida di conferma per i fronti ICON-2I sia come fallback
    # quando l'analisi ICON-2I non e' disponibile.
    front_catalog = load_synoptic_front_catalog()

    for idx, filename in enumerate(file_list):
        print(f"   [{idx+1:02d}] DL {filename}...", end=" ", flush=True)

        try:
            with requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(TEMP_FILE, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            print("OK", end=" ", flush=True)
        except Exception as e:
            print(f"KO ({e})", flush=True)
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
                    (dw_step.latitude >= LAT_MIN) & (dw_step.latitude <= LAT_MAX) &
                    (dw_step.longitude >= LON_MIN) & (dw_step.longitude <= LON_MAX)
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
                front_entry = None
                fronts = {"type": "FeatureCollection", "features": []}
                front_method = None
                front_source = None
                front_level = None
                front_valid_time = None

                if icon_front_analyzer is not None:
                    try:
                        fronts = icon_front_analyzer.analyze(step_hours)
                        guide_entry = nearest_synoptic_front(front_catalog, valid_dt)
                        if guide_entry is not None:
                            fronts = corroborate_with_reference(
                                fronts,
                                guide_entry.get("fronts"),
                                max_distance_km=front_corroboration_radius_km(step_hours),
                            )
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
                else:
                    front_entry = nearest_synoptic_front(front_catalog, valid_dt)
                    if front_entry is not None:
                        fronts = front_entry.get("fronts", fronts)
                        front_method = front_entry.get("method")
                        front_source = front_entry.get("source")
                        front_level = front_entry.get("level")
                        front_valid_time = front_entry.get("validTime")

                header = {
                    "nx": nx, "ny": ny, "lo1": lo1, "la1": la1, "lo2": lo2, "la2": la2,
                    "dx": dx, "dy": dy, "runTime": iso_z(run_dt), "validTime": iso_date, "refTime": iso_date, "leadHours": step_hours,
                    "feelsLikeMethod": FEELS_LIKE_METHOD,
                    "frontMethod": front_method,
                    "frontSource": front_source,
                    "frontLevel": front_level,
                    "frontValidTime": front_valid_time,
                }

                step_data = {
                    "meta": header,
                    "wind_u": { "header": {**header, "parameterCategory": 2, "parameterNumber": 2}, "data": np.round(u_val, 1).flatten().tolist() },
                    "wind_v": { "header": {**header, "parameterCategory": 2, "parameterNumber": 3}, "data": np.round(v_val, 1).flatten().tolist() },
                    "temp": clean_for_json(temp_c, 1),
                    "feels_like": clean_for_json(feels_like, 1),
                    "rain": clean_for_json(rain, 2),
                    "press": clean_for_json(press, 1),
                    "rh": clean_for_json(rh_val, 0),
                    "cloud": clean_for_json(cloud, 0),
                    "fronts": fronts
                }

                out_name = f"step_{step_hours}.json"
                # FIX 3: Minificazione del file
                with open(f"{TEMP_DIR}/{out_name}", 'w') as jf:
                    json.dump(step_data, jf, separators=(',', ':'), allow_nan=False)

                # Campi a 850 hPa (theta-e, T, vento) per il layer di
                # ispezione dei fronti: file separato, scaricato dal sito
                # solo quando l'utente attiva la vista 850 hPa.
                if icon_front_analyzer is not None:
                    try:
                        upper = icon_front_analyzer.upper_air(step_hours)
                        if upper is not None:
                            upper["runTime"] = iso_z(run_dt)
                            upper["validTime"] = iso_date
                            with open(f"{TEMP_DIR}/upper_{step_hours}.json", 'w') as uf:
                                json.dump(upper, uf, separators=(',', ':'), allow_nan=False)
                    except Exception as upper_error:
                        print(f" upper-{step_hours}h:{upper_error}", end="", flush=True)

                if not any(x['hour'] == step_hours for x in catalog):
                    catalog.append({ "file": out_name, "label": f"{valid_dt.strftime('%d/%m %H:00')} UTC", "hour": step_hours, "runTime": iso_z(run_dt), "validTime": iso_date, "leadHours": step_hours })

            except Exception as e:
                print(f"!", end="", flush=True)
                continue

        print(" -> Done")

    if icon_front_analyzer is not None:
        icon_front_analyzer.close()
    if os.path.exists(FRONT_TEMP_DIR):
        shutil.rmtree(FRONT_TEMP_DIR)
    if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
    if os.path.exists(f"{TEMP_FILE}.idx"): os.remove(f"{TEMP_FILE}.idx")

    if catalog:
        catalog.sort(key=lambda x: x['hour'])
        with open(f"{TEMP_DIR}/catalog.json", 'w') as f:
            json.dump(catalog, f, separators=(',', ':'), allow_nan=False)

        if os.path.exists(FINAL_DIR): shutil.rmtree(FINAL_DIR)
        shutil.move(TEMP_DIR, FINAL_DIR)
        print("\nELABORAZIONE COMPLETATA CON SUCCESSO.")
    else:
        print("\nNESSUN DATO VALIDO ESTRATTO.")
        sys.exit(1)

if __name__ == "__main__":
    process_data()
