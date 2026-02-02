import requests
import xarray as xr
import numpy as np
import json
import os
import sys
import shutil
import time
from datetime import datetime, timedelta

# --- CONFIGURAZIONE ---
DATASET_ID = "ICON_2I_SURFACE_PRESSURE_LEVELS"
API_LIST_URL = f"https://meteohub.agenziaitaliameteo.it/api/datasets/{DATASET_ID}/opendata"
API_DOWNLOAD_URL = "https://meteohub.agenziaitaliameteo.it/api/opendata"

FINAL_DIR = "data_weather"
TEMP_DIR = "temp_processing"

LAT_MIN, LAT_MAX = 35.0, 39.5
LON_MIN, LON_MAX = 11.0, 16.5

def get_latest_run_files():
    print("1. MeteoHub Connection...", flush=True)
    try:
        r = requests.get(API_LIST_URL, timeout=30)
        r.raise_for_status()
        items = r.json()
    except Exception as e:
        print(f"   Err API: {e}", flush=True)
        return None, []

    runs = {}
    for item in items:
        if isinstance(item, dict) and 'date' in item and 'run' in item:
            key = f"{item['date']} {item['run']}"
            if key not in runs: runs[key] = []
            runs[key].append(item['filename'])
    
    if not runs: return None, []
    latest_key = sorted(runs.keys())[-1]
    # Scarica SOLO 24 ORE per fare in fretta e testare
    return datetime.strptime(latest_key, "%Y-%m-%d %H:%M"), runs[latest_key][:24]

def process_data():
    # --- TEST DI SCRITTURA FORZATO ---
    # Creiamo un file dummy per vedere se GitHub ci lascia salvare
    if not os.path.exists(FINAL_DIR): os.makedirs(FINAL_DIR)
    with open(f"{FINAL_DIR}/debug_status.txt", "w") as f:
        f.write(f"Ultimo aggiornamento tentato: {datetime.now()}\nSe leggi questo, i permessi di scrittura FUNZIONANO.")
    print("   [TEST] File debug creato.", flush=True)
    # ---------------------------------

    run_dt, file_list = get_latest_run_files()
    if not file_list:
        print("!!! NESSUN DATO DAL SERVER.", flush=True)
        sys.exit(1) # Rosso se manca la rete

    print(f"2. Trovati {len(file_list)} file. Elaborazione...", flush=True)
    
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    catalog = []
    success_count = 0

    for idx, filename in enumerate(file_list):
        print(f"   DL {filename}...", end=" ", flush=True)
        local_path = "temp.grib2"
        try:
            with requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
            print("OK", end=" ", flush=True)
        except:
            print("KO", flush=True)
            continue

        try:
            ds_wind = xr.open_dataset(local_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
            # Temp/Rain opzionali
            try: ds_temp = xr.open_dataset(local_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
            except: ds_temp = None
            try: ds_rain = xr.open_dataset(local_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
            except: ds_rain = None
        except:
            print(" -> Err GRIB", flush=True)
            continue

        steps = range(ds_wind.sizes.get('step', 1))
        for i in steps:
            try:
                raw_step = ds_wind.step.values[i]
                if isinstance(raw_step, np.timedelta64): step_hours = int(raw_step / np.timedelta64(1, 'h'))
                else: step_hours = int(raw_step)

                d_w = ds_wind.isel(step=i) if 'step' in ds_wind.dims else ds_wind
                d_w = d_w.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
                mask = ((d_w.latitude >= LAT_MIN) & (d_w.latitude <= LAT_MAX) & (d_w.longitude >= LON_MIN) & (d_w.longitude <= LON_MAX))
                cut_w = d_w.where(mask, drop=True)

                u_key = next((k for k in ['u10','u'] if k in cut_w), None)
                v_key = next((k for k in ['v10','v'] if k in cut_w), None)
                
                if u_key and v_key:
                    u = np.nan_to_num(cut_w[u_key].values)
                    v = np.nan_to_num(cut_w[v_key].values)
                    
                    lat = cut_w.latitude.values
                    lon = cut_w.longitude.values
                    ny, nx = u.shape
                    la1, lo1, dx, dy = float(lat[0]), float(lon[0]), float(abs(lon[1]-lon[0])), float(abs(lat[0]-lat[1]))

                    temp, rain = np.zeros_like(u), np.zeros_like(u)
                    if ds_temp:
                        d_t = ds_temp.isel(step=i) if 'step' in ds_temp.dims else ds_temp
                        d_t = d_t.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
                        cut_t = d_t.where(mask, drop=True)
                        t_key = next((k for k in ['t2m','t'] if k in cut_t), None)
                        if t_key: temp = cut_t[t_key].values - 273.15
                    
                    if ds_rain:
                        d_r = ds_rain.isel(step=i) if 'step' in ds_rain.dims else ds_rain
                        d_r = d_r.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
                        cut_r = d_r.where(mask, drop=True)
                        r_key = next((k for k in ['tp','tot_prec'] if k in cut_r), None)
                        if r_key: rain = np.nan_to_num(cut_r[r_key].values)

                    valid_dt = run_dt + timedelta(hours=step_hours)
                    iso_date = valid_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                    header = { "parameterCategory": 2, "parameterNumber": 2, "nx": nx, "ny": ny, "lo1": lo1, "la1": la1, "dx": dx, "dy": dy, "refTime": iso_date }
                    
                    step_data = {
                        "meta": header,
                        "wind_u": { "header": header, "data": np.round(u, 1).flatten().tolist() },
                        "wind_v": { "header": header, "data": np.round(v, 1).flatten().tolist() },
                        "temp": np.round(temp, 1).flatten().tolist(),
                        "rain": np.round(rain, 2).flatten().tolist()
                    }

                    out_name = f"step_{step_hours}.json"
                    with open(f"{TEMP_DIR}/{out_name}", 'w') as jf: json.dump(step_data, jf)
                    
                    if not any(x['hour'] == step_hours for x in catalog):
                        catalog.append({"file": out_name, "label": f"{valid_dt.strftime('%d/%m %H:00')}", "hour": step_hours})
                    success_count += 1
            except: continue
        print(" -> OK", flush=True)

    if success_count > 0:
        print(f"\n3. Salvataggio {success_count} file...", flush=True)
        catalog.sort(key=lambda x: x['hour'])
        with open(f"{TEMP_DIR}/catalog.json", 'w') as f: json.dump(catalog, f)
        
        # SOSTITUZIONE FINALE
        if os.path.exists(FINAL_DIR): shutil.rmtree(FINAL_DIR)
        shutil.move(TEMP_DIR, FINAL_DIR)
        print("   Cartella aggiornata.", flush=True)
    else:
        print("\n!!! ERRORE CRITICO: Nessun dato estratto.", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    process_data()
    
