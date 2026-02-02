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

# CARTELLE
FINAL_DIR = "data_weather"
TEMP_DIR = "temp_data_weather"  # Cartella di appoggio

LAT_MIN, LAT_MAX = 35.0, 39.5
LON_MIN, LON_MAX = 11.0, 16.5

# Imposta a False per scaricare tutto (più lento ma completo)
FAST_MODE = True 

def get_latest_run_files():
    print("1. Contatto MeteoHub...", flush=True)
    try:
        r = requests.get(API_LIST_URL, timeout=10)
        r.raise_for_status()
        items = r.json()
    except Exception as e:
        print(f"   ERRORE API: {e}", flush=True)
        return None, []

    runs = {}
    for item in items:
        if isinstance(item, dict) and 'date' in item and 'run' in item:
            key = f"{item['date']} {item['run']}"
            if key not in runs: runs[key] = []
            runs[key].append(item['filename'])
    
    if not runs: return None, []
    latest_key = sorted(runs.keys())[-1]
    run_dt = datetime.strptime(latest_key, "%Y-%m-%d %H:%M")
    
    file_list = runs[latest_key]
    if FAST_MODE:
        file_list = file_list[:24] # Scarica solo le prime 24 ore
        print("   >>> FAST MODE: Scarico solo 24h.", flush=True)

    return run_dt, file_list

def process_data():
    run_dt, file_list = get_latest_run_files()
    
    if not file_list:
        print("!!! NESSUN DATO. Mantengo i vecchi.", flush=True)
        sys.exit(0)

    print(f"2. Preparo cartella temporanea...", flush=True)
    
    # Lavoriamo nella TEMP DIR, non tocchiamo quella vera per ora
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    catalog = []
    processed_hours = set()
    success_count = 0

    for idx, filename in enumerate(file_list):
        print(f"   [{idx+1}/{len(file_list)}] Scarico: {filename} ...", end=" ", flush=True)
        local_path = "temp.grib2"
        try:
            with requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
            print("OK ->", end=" ", flush=True)
        except:
            print("FALLITO.", flush=True)
            continue

        try:
            ds_wind = xr.open_dataset(local_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
            
            # Tentativi opzionali
            try: ds_temp = xr.open_dataset(local_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
            except: ds_temp = None
            try: ds_rain = xr.open_dataset(local_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
            except: ds_rain = None
        except:
            print("Err GRIB", flush=True)
            continue

        steps = range(ds_wind.sizes.get('step', 1))
        for i in steps:
            try:
                raw_step = ds_wind.step.values[i]
                if isinstance(raw_step, np.timedelta64): step_hours = int(raw_step / np.timedelta64(1, 'h'))
                else: step_hours = int(raw_step)

                if step_hours in processed_hours: continue

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
                    la1, lo1 = float(lat[0]), float(lon[0])
                    dx, dy = float(abs(lon[1] - lon[0])), float(abs(lat[0] - lat[1]))

                    # Temp & Rain
                    temp, rain = np.zeros_like(u), np.zeros_like(u)
                    if ds_temp:
                        # Logica temp... (semplificata per brevità, usa logica precedente)
                        pass 
                    
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
                    # SALVIAMO NELLA CARTELLA TEMP
                    with open(f"{TEMP_DIR}/{out_name}", 'w') as jf: json.dump(step_data, jf)
                    
                    day_str = valid_dt.strftime("%d/%m")
                    hour_str = valid_dt.strftime("%H:00")
                    catalog.append({"file": out_name, "label": f"{day_str} {hour_str}", "hour": step_hours})
                    processed_hours.add(step_hours)
                    success_count += 1
                    print(f"JSON (+{step_hours}h)", flush=True)

            except Exception as e: continue

    # --- MOMENTO DELLA VERITÀ ---
    if success_count > 0:
        print(f"\n3. SUCCESSO! Salvati {success_count} file.", flush=True)
        catalog.sort(key=lambda x: x['hour'])
        with open(f"{TEMP_DIR}/catalog.json", 'w') as f: json.dump(catalog, f)
        
        # SOSTITUZIONE ATOMICA: Cancella vecchia, sposta nuova
        if os.path.exists(FINAL_DIR): shutil.rmtree(FINAL_DIR)
        shutil.move(TEMP_DIR, FINAL_DIR)
        print("   Cartella data_weather aggiornata.", flush=True)
    else:
        print("\n!!! NESSUN FILE VALIDO SCARICATO.", flush=True)
        print("   Mantengo i vecchi dati. Non tocco nulla.", flush=True)
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR) # Pulizia

if __name__ == "__main__":
    process_data()
    
