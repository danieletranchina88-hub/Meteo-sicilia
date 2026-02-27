import requests
import xarray as xr
import numpy as np
import json
import os
import sys
import shutil
from datetime import datetime, timedelta

# --- CONFIGURAZIONE ---
DATASET_ID = "ICON_2I_SURFACE_PRESSURE_LEVELS"
API_LIST_URL = f"https://meteohub.agenziaitaliameteo.it/api/datasets/{DATASET_ID}/opendata"
API_DOWNLOAD_URL = "https://meteohub.agenziaitaliameteo.it/api/opendata"

FINAL_DIR = "data_weather"
TEMP_DIR = "temp_processing"
TEMP_FILE = "temp.grib2"

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

# PARAMETRI OROGRAFIA
PATH_DEM_ITALIA = "topografia_italia_1km.nc"
LAPSE_RATE = 0.0065

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
    run_dt = datetime.strptime(latest_key, "%Y-%m-%d %H:%M")
    return run_dt, runs[latest_key][:48]


def calculate_rh_numpy(temp_k, dew_k):
    T = temp_k - 273.15
    Td = dew_k - 273.15
    a = 17.625
    b = 243.04

    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.exp((a * Td) / (b + Td))
        denominator = np.exp((a * T) / (b + T))
        rh = 100 * (numerator / denominator)

    return np.nan_to_num(np.clip(rh, 0, 100))


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
    c = np.nan_to_num(cloud_arr)
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


def process_data():
    run_dt, file_list = get_latest_run_files()
    if not file_list:
        print("Nessun dato trovato.")
        sys.exit(0)

    print(f"2. Elaboro Run: {run_dt} ({len(file_list)} files)", flush=True)

    # Caricamento DEM (Topografia Reale) se presente
    real_dem = None
    if os.path.exists(PATH_DEM_ITALIA):
        try:
            real_dem = xr.open_dataset(PATH_DEM_ITALIA)
            print("   DEM Topografico trovato.")
        except:
            pass

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    catalog = []

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
            
            # Recupero altitudine modello (opzionale)
            ds_hsurf = None
            try: ds_hsurf = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys':{'shortName':'gh','typeOfLevel':'surface'}})
            except: pass

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

                u_val = np.nan_to_num(cut_w[u_key].values)
                v_val = np.nan_to_num(cut_w[v_key].values)

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
                temp_c = np.zeros_like(u_val)
                rh_val = np.zeros_like(u_val)

                if ds_thermo is not None:
                    dt_step = ds_thermo.isel(step=i) if ds_thermo.sizes.get('step', 1) > 1 else ds_thermo
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

                # FIX 2: CALCOLO PERCEPITA SICURO
                feels_like = np.copy(temp_c)
                try:
                    if real_dem is not None and ds_hsurf is not None:
                        dh_step = ds_hsurf.isel(step=i) if ds_hsurf.sizes.get('step', 1) > 1 else ds_hsurf
                        dh_step = dh_step.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
                        h_model = extract_raw_grid(dh_step, mask, ['gh'])
                        if h_model is not None:
                            if f > 1: h_model = downsample_2d(h_model, f)
                            h_real = real_dem.interp(latitude=cut_w.latitude, longitude=cut_w.longitude).elevation.values
                            if h_real.shape == u_val.shape:
                                feels_like = temp_c + (h_model - np.nan_to_num(h_real)) * LAPSE_RATE
                except Exception:
                    pass # Se fallisce, feels_like rimane uguale a temp_c

                # 2) PRESSIONE
                press = np.zeros_like(u_val)
                if ds_press is not None:
                    dp_step = ds_press.isel(step=i) if ds_press.sizes.get('step', 1) > 1 else ds_press
                    dp_step = dp_step.sortby('latitude', ascending=False).sortby('longitude', ascending=True)

                    p_raw = extract_raw_grid(dp_step, mask, ['pmsl', 'prmsl', 'msl'])
                    if p_raw is not None and p_raw.ndim == 2:
                        if f > 1: p_raw = downsample_2d(p_raw, f)
                        if p_raw.shape == u_val.shape:
                            p_clean = np.nan_to_num(p_raw)
                            press = (p_clean / 100.0) if np.max(p_clean) > 80000 else p_clean

                if np.max(press) < 800: press.fill(1013.0)

                # 3) PIOGGIA
                rain = np.zeros_like(u_val)
                if ds_rain is not None:
                    dr_step = ds_rain.isel(step=i) if ds_rain.sizes.get('step', 1) > 1 else ds_rain
                    dr_step = dr_step.sortby('latitude', ascending=False).sortby('longitude', ascending=True)

                    r_raw = extract_raw_grid(dr_step, mask, ['tp', 'tot_prec'])
                    if r_raw is not None and r_raw.ndim == 2:
                        if f > 1: r_raw = downsample_2d(r_raw, f)
                        if r_raw.shape == u_val.shape:
                            rain = np.nan_to_num(r_raw)

                # 4) NUVOLOSITÀ
                cloud = np.zeros_like(u_val)
                if ds_cloud is not None:
                    dc_step = ds_cloud.isel(step=i) if ds_cloud.sizes.get('step', 1) > 1 else ds_cloud
                    dc_step = dc_step.sortby('latitude', ascending=False).sortby('longitude', ascending=True)

                    c_raw = extract_raw_grid(dc_step, mask, ['tcc', 'tcdc', 'clct', 'cc', 'tcc_total', 'totalCloudCover'])
                    if c_raw is not None and c_raw.ndim == 2:
                        if f > 1: c_raw = downsample_2d(c_raw, f)
                        if c_raw.shape == u_val.shape:
                            cloud = normalize_cloud_to_percent(c_raw)

                # EXPORT JSON
                valid_dt = run_dt + timedelta(hours=step_hours)
                iso_date = valid_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

                header = {
                    "nx": nx, "ny": ny, "lo1": lo1, "la1": la1, "lo2": lo2, "la2": la2,
                    "dx": dx, "dy": dy, "refTime": iso_date
                }

                step_data = {
                    "meta": header,
                    "wind_u": { "header": {**header, "parameterCategory": 2, "parameterNumber": 2}, "data": np.round(u_val, 1).flatten().tolist() },
                    "wind_v": { "header": {**header, "parameterCategory": 2, "parameterNumber": 3}, "data": np.round(v_val, 1).flatten().tolist() },
                    "temp": np.round(temp_c, 1).flatten().tolist(),
                    "feels_like": np.round(feels_like, 1).flatten().tolist(), # Inserito qui!
                    "rain": np.round(rain, 2).flatten().tolist(),
                    "press": np.round(press, 1).flatten().tolist(),
                    "rh": np.round(rh_val, 0).flatten().tolist(),
                    "cloud": np.round(cloud, 0).flatten().tolist()
                }

                out_name = f"step_{step_hours}.json"
                # FIX 3: Minificazione del file
                with open(f"{TEMP_DIR}/{out_name}", 'w') as jf:
                    json.dump(step_data, jf, separators=(',', ':'))

                if not any(x['hour'] == step_hours for x in catalog):
                    catalog.append({ "file": out_name, "label": f"{valid_dt.strftime('%d/%m %H:00')}", "hour": step_hours })

            except Exception as e:
                print(f"!", end="", flush=True)
                continue

        print(" -> Done")

    if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
    if os.path.exists(f"{TEMP_FILE}.idx"): os.remove(f"{TEMP_FILE}.idx")

    if catalog:
        catalog.sort(key=lambda x: x['hour'])
        with open(f"{TEMP_DIR}/catalog.json", 'w') as f:
            json.dump(catalog, f, separators=(',', ':'))

        if os.path.exists(FINAL_DIR): shutil.rmtree(FINAL_DIR)
        shutil.move(TEMP_DIR, FINAL_DIR)
        print("\nELABORAZIONE COMPLETATA CON SUCCESSO.")
    else:
        print("\nNESSUN DATO VALIDO ESTRATTO.")
        sys.exit(1)

if __name__ == "__main__":
    process_data()
