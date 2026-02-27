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

# ITALIA BOUNDS
LAT_MIN, LAT_MAX = 35.0, 48.5
LON_MIN, LON_MAX = 6.0, 19.5

# RISOLUZIONE MASSIMA (Sbloccata per 2.2km)
MAX_PIXELS = 600_000 

# PARAMETRI SCIENTIFICI
PATH_DEM_ITALIA = "topografia_italia_1km.nc" 
LAPSE_RATE = 0.0065  # 0.65°C ogni 100 metri

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
    a, b = 17.625, 243.04
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.exp((a * Td) / (b + Td))
        denominator = np.exp((a * T) / (b + T))
        rh = 100 * (numerator / denominator)
    return np.nan_to_num(np.clip(rh, 0, 100))

def extract_raw_grid(ds, mask, var_names):
    try:
        var_key = next((k for k in var_names if k in ds), None)
        if not var_key: return None
        return ds[var_key].where(mask, drop=True).values
    except: return None

def try_open_cloud_dataset(grib_path):
    candidates = [
        {'filter_by_keys': {'shortName': 'tcc'}},
        {'filter_by_keys': {'shortName': 'clct'}},
        {'filter_by_keys': {'typeOfLevel': 'entireAtmosphere'}},
    ]
    for bk in candidates:
        try:
            ds = xr.open_dataset(grib_path, engine='cfgrib', backend_kwargs=bk)
            if ds is not None and len(ds.data_vars) > 0: return ds
        except: continue
    return None

def normalize_cloud_to_percent(cloud_arr):
    c = np.nan_to_num(cloud_arr)
    if c.size == 0: return c
    mx = float(np.nanmax(c)) if np.isfinite(np.nanmax(c)) else 0.0
    if mx <= 1.01: c = c * 100.0
    return np.clip(c, 0.0, 100.0)

def compute_downsample_factor(ny, nx, max_pixels=MAX_PIXELS):
    pixels = int(ny) * int(nx)
    if pixels <= max_pixels: return 1
    return int(np.ceil(np.sqrt(pixels / max_pixels)))

def process_data():
    run_dt, file_list = get_latest_run_files()
    if not file_list: sys.exit(0)

    # Caricamento DEM
    real_dem = None
    if os.path.exists(PATH_DEM_ITALIA):
        real_dem = xr.open_dataset(PATH_DEM_ITALIA)

    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    catalog = []

    for idx, filename in enumerate(file_list):
        print(f"[{idx+1:02d}] DL {filename}...", end=" ", flush=True)
        try:
            r = requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True)
            with open(TEMP_FILE, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
            print("OK", flush=True)
        except: continue

        try:
            # Apertura dataset con i TUOI filtri originali
            ds_wind = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
            ds_thermo = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
            ds_press = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}})
            ds_rain = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'stepType': 'accum'}})
            ds_cloud = try_open_cloud_dataset(TEMP_FILE)
            
            # Recupero HSURF (Altezza modello) per Lapse Rate
            try: ds_hsurf = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys':{'shortName':'gh','typeOfLevel':'surface'}})
            except: ds_hsurf = None

            dw_step = ds_wind.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
            mask = (dw_step.latitude >= LAT_MIN) & (dw_step.latitude <= LAT_MAX) & (dw_step.longitude >= LON_MIN) & (dw_step.longitude <= LON_MAX)
            cut_w = dw_step.where(mask, drop=True)
            
            # Meta griglia
            lat, lon = cut_w.latitude.values, cut_w.longitude.values
            if lat.ndim > 1: lat = lat[:, 0]
            if lon.ndim > 1: lon = lon[0, :]
            ny, nx = len(lat), len(lon)

            # Estrazione dati
            u_val = np.nan_to_num(extract_raw_grid(ds_wind, mask, ['u10', 'u']))
            v_val = np.nan_to_num(extract_raw_grid(ds_wind, mask, ['v10', 'v']))
            t_raw = extract_raw_grid(ds_thermo, mask, ['t2m', 't'])
            d_raw = extract_raw_grid(ds_thermo, mask, ['d2m', '2d'])
            p_raw = extract_raw_grid(ds_press, mask, ['pmsl', 'prmsl', 'msl'])
            r_raw = extract_raw_grid(ds_rain, mask, ['tp', 'tot_prec'])
            c_raw = extract_raw_grid(ds_cloud, mask, ['tcc', 'tcdc', 'clct']) if ds_cloud else None

            # Calcoli
            temp_c = t_raw - 273.15
            rh_val = calculate_rh_numpy(t_raw, d_raw)
            press = (p_raw / 100.0) if np.nanmax(p_raw) > 80000 else p_raw
            rain = np.nan_to_num(r_raw)
            cloud = normalize_cloud_to_percent(c_raw) if c_raw is not None else np.zeros_like(temp_c)

            # --- CORREZIONE OROGRAFICA (feels_like) ---
            feels_like = temp_c.copy()
            if ds_hsurf is not None and real_dem is not None:
                h_model = ds_hsurf.gh.where(mask, drop=True).values
                h_real = real_dem.interp(latitude=cut_w.latitude, longitude=cut_w.longitude).elevation.values
                feels_like = temp_c + (h_model - h_real) * LAPSE_RATE

            # Export
            step_hours = int(cut_w.step.values / np.timedelta64(1, 'h'))
            valid_dt = run_dt + timedelta(hours=step_hours)
            
            header = {"nx": nx, "ny": ny, "lo1": float(lon[0]), "la1": float(lat[0]), "lo2": float(lon[-1]), "la2": float(lat[-1]), "dx": float(abs(lon[1]-lon[0])), "dy": float(abs(lat[0]-lat[1]))}

            step_data = {
                "meta": header,
                "wind_u": {"data": np.round(u_val, 1).flatten().tolist()},
                "wind_v": {"data": np.round(v_val, 1).flatten().tolist()},
                "temp": np.round(temp_c, 1).flatten().tolist(),
                "feels_like": np.round(feels_like, 1).flatten().tolist(),
                "rain": np.round(rain, 2).flatten().tolist(),
                "press": np.round(press, 1).flatten().tolist(),
                "rh": np.round(rh_val, 0).flatten().tolist(),
                "cloud": np.round(cloud, 0).flatten().tolist()
            }

            out_name = f"step_{step_hours}.json"
            with open(f"{TEMP_DIR}/{out_name}", 'w') as jf:
                json.dump(step_data, jf, separators=(',', ':'))

            catalog.append({"file": out_name, "label": valid_dt.strftime('%d/%m %H:00'), "hour": step_hours})

        except Exception as e:
            print(f" Errore elaborazione: {e}")
            continue

    if catalog:
        with open(f"{TEMP_DIR}/catalog.json", 'w') as f:
            json.dump(sorted(catalog, key=lambda x: x['hour']), f, separators=(',', ':'))
        if os.path.exists(FINAL_DIR): shutil.rmtree(FINAL_DIR)
        shutil.move(TEMP_DIR, FINAL_DIR)
        print("COMPLETATO.")

if __name__ == "__main__":
    process_data()
