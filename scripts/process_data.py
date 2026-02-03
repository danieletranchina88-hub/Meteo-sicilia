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

# Coordinate Box Sicilia
LAT_MIN, LAT_MAX = 35.0, 39.5
LON_MIN, LON_MAX = 11.0, 16.5

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
            if key not in runs: runs[key] = []
            runs[key].append(item['filename'])
            
    if not runs: return None, []
    
    # Prende l'ultimo run disponibile
    latest_key = sorted(runs.keys())[-1]
    run_dt = datetime.strptime(latest_key, "%Y-%m-%d %H:%M")
    
    return run_dt, runs[latest_key][:48]

def calculate_rh(temp_k, dew_k):
    T = temp_k - 273.15
    Td = dew_k - 273.15
    a = 17.625
    b = 243.04
    numerator = np.exp((a * Td) / (b + Td))
    denominator = np.exp((a * T) / (b + T))
    rh = 100 * (numerator / denominator)
    return np.clip(rh, 0, 100)

def process_data():
    run_dt, file_list = get_latest_run_files()
    if not file_list:
        print("Nessun dato trovato.")
        sys.exit(0) # Exit code 0 (successo parziale) per non bloccare la pipeline se mancano dati momentanei
        
    print(f"2. Elaboro Run: {run_dt} ({len(file_list)} files)", flush=True)
    
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    catalog = []

    for idx, filename in enumerate(file_list):
        print(f"   [{idx+1:02d}] DL {filename}...", end=" ", flush=True)
        
        # --- DOWNLOAD ---
        try:
            with requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(TEMP_FILE, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
            print("OK", end=" ", flush=True)
        except Exception as e:
            print(f"KO (Download fallito: {e})", flush=True)
            continue

        if os.path.exists(f"{TEMP_FILE}.idx"): os.remove(f"{TEMP_FILE}.idx")

        # --- APERTURA DATASET ---
        try:
            # Opzione read_keys per forzare la lettura corretta delle coordinate
            ds_wind = xr.open_dataset(TEMP_FILE, engine='cfgrib', 
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
        except Exception as e:
            print(f"\n   [ERR] Impossibile aprire GRIB Vento: {e}")
            continue

        # Apertura opzionali
        ds_thermo = None
        try: ds_thermo = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
        except: pass

        ds_press = None
        try: ds_press = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}})
        except: 
            try: ds_press = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
            except: pass

        ds_rain = None
        try: ds_rain = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'stepType': 'accum'}})
        except: pass

        # --- ELABORAZIONE STEP ---
        # Gestione sicura del range step
        if 'step' in ds_wind.dims:
            steps = range(ds_wind.sizes['step'])
            is_multi_step = True
        else:
            steps = [0]
            is_multi_step = False
        
        for i in steps:
            try:
                if is_multi_step:
                    d_w_raw = ds_wind.isel(step=i)
                    raw_step = ds_wind.step.values[i]
                else:
                    d_w_raw = ds_wind
                    raw_step = ds_wind.step.values

                # Calcolo ore
                step_hours = int(raw_step / np.timedelta64(1, 'h')) if isinstance(raw_step, np.timedelta64) else int(raw_step)
                
                # --- RITAGLIO ---
                # Rinomina coordinate se necessario (lat -> latitude)
                if 'lat' in d_w_raw.coords and 'latitude' not in d_w_raw.coords:
                    d_w_raw = d_w_raw.rename({'lat': 'latitude', 'lon': 'longitude'})

                # Sort solo se le coordinate sono 1D
                if d_w_raw.latitude.ndim == 1:
                    d_w = d_w_raw.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
                else:
                    d_w = d_w_raw

                mask = ((d_w.latitude >= LAT_MIN) & (d_w.latitude <= LAT_MAX) & (d_w.longitude >= LON_MIN) & (d_w.longitude <= LON_MAX))
                cut_w = d_w.where(mask, drop=True)
                
                # CHECK SALVAVITA: Se il ritaglio è vuoto (es. file non copre la Sicilia)
                if cut_w.latitude.size == 0 or cut_w.longitude.size == 0:
                    print(f"\n   [WARN] Step {step_hours}h: Area vuota (coordinate fuori range?)", end="")
                    continue

                u_key = next((k for k in ['u10','u'] if k in cut_w), None)
                v_key = next((k for k in ['v10','v'] if k in cut_w), None)
                
                if not u_key or not v_key: 
                    print(f"\n   [WARN] Step {step_hours}h: U/V mancanti", end="")
                    continue

                u = np.nan_to_num(cut_w[u_key].values)
                v = np.nan_to_num(cut_w[v_key].values)

                # Info Griglia (Sicura)
                lat = cut_w.latitude.values
                lon = cut_w.longitude.values
                
                if lat.ndim > 1: lat = lat[:,0] # Gestione griglie 2D proiettate (se necessario)
                if lon.ndim > 1: lon = lon[0,:]

                ny, nx = u.shape
                # Check dimensioni
                if len(lat) < 2 or len(lon) < 2:
                     print(f"\n   [WARN] Griglia troppo piccola ({len(lat)}x{len(lon)})", end="")
                     continue

                la1, lo1 = float(lat[0]), float(lon[0])
                dx = float(abs(lon[1] - lon[0]))
                dy = float(abs(lat[0] - lat[1]))
                lo2 = lo1 + (nx - 1) * dx
                la2 = la1 - (ny - 1) * dy

                # --- ALTRI PARAMETRI ---
                temp_c = np.zeros_like(u)
                rh_val = np.zeros_like(u)
                if ds_thermo:
                    d_t = ds_thermo.where(mask, drop=True)
                    if is_multi_step and 'step' in ds_thermo.dims: d_t = d_t.isel(step=i)
                    t_key = next((k for k in ['t2m','t'] if k in d_t), None)
                    d_key = next((k for k in ['d2m','2d','d'] if k in d_t), None)
                    if t_key:
                        tk = d_t[t_key].values
                        temp_c = tk - 273.15
                        if d_key: rh_val = calculate_rh(tk, d_t[d_key].values)

                rain = np.zeros_like(u)
                if ds_rain:
                    d_r = ds_rain.where(mask, drop=True)
                    if is_multi_step and 'step' in ds_rain.dims: d_r = d_r.isel(step=i)
                    r_key = next((k for k in ['tp', 'tot_prec', 'apcp'] if k in d_r), None)
                    if r_key: rain = np.nan_to_num(d_r[r_key].values)

                press = np.zeros_like(u)
                if ds_press:
                    d_p = ds_press.where(mask, drop=True)
                    if is_multi_step and 'step' in ds_press.dims: d_p = d_p.isel(step=i)
                    p_key = next((k for k in ['prmsl', 'msl', 'sp', 'pres'] if k in d_p), None)
                    if p_key:
                        raw_p = np.nan_to_num(d_p[p_key].values)
                        if np.max(raw_p) > 80000: press = raw_p / 100.0
                        else: press = raw_p
                    if np.max(press) < 500: press.fill(1013.0)
                else: press.fill(1013.0)

                # --- EXPORT ---
                valid_dt = run_dt + timedelta(hours=step_hours)
                iso_date = valid_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                
                header = {"nx": nx, "ny": ny, "lo1": lo1, "la1": la1, "lo2": lo2, "la2": la2, "dx": dx, "dy": dy, "refTime": iso_date}
                
                step_data = {
                    "meta": header,
                    "wind_u": {"header":{**header,"parameterCategory":2,"parameterNumber":2}, "data": np.round(u,1).flatten().tolist()},
                    "wind_v": {"header":{**header,"parameterCategory":2,"parameterNumber":3}, "data": np.round(v,1).flatten().tolist()},
                    "temp": np.round(temp_c,1).flatten().tolist(),
                    "rain": np.round(rain,2).flatten().tolist(),
                    "press": np.round(press,1).flatten().tolist(),
                    "rh": np.round(rh_val,0).flatten().tolist()
                }
                
                out_name = f"step_{step_hours}.json"
                with open(f"{TEMP_DIR}/{out_name}", 'w') as jf: json.dump(step_data, jf)
                
                if not any(x['hour'] == step_hours for x in catalog):
                    catalog.append({"file": out_name, "label": f"{valid_dt.strftime('%d/%m %H:00')}", "hour": step_hours})

            except Exception as e_step:
                print(f"\n   [ERR Step {i}] {e_step}", end="", flush=True)
                # print(traceback.format_exc()) # Decommentare per debug profondo
                continue
        
        print(" -> Done")

    if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
    if os.path.exists(f"{TEMP_FILE}.idx"): os.remove(f"{TEMP_FILE}.idx")

    if catalog:
        catalog.sort(key=lambda x: x['hour'])
        with open(f"{TEMP_DIR}/catalog.json", 'w') as f: json.dump(catalog, f)
        if os.path.exists(FINAL_DIR): shutil.rmtree(FINAL_DIR)
        shutil.move(TEMP_DIR, FINAL_DIR)
        print("COMPLETATO.")
    else:
        print("\nNESSUN DATO GENERATO (CATALOG VUOTO).")
        sys.exit(1)

if __name__ == "__main__":
    process_data()
    
