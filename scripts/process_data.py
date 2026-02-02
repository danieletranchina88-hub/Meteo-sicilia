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
OUTPUT_DIR = "data_weather"

LAT_MIN, LAT_MAX = 35.0, 39.5
LON_MIN, LON_MAX = 11.0, 16.5

def get_latest_run_files():
    print("1. Controllo disponibilità dati...")
    try:
        r = requests.get(API_LIST_URL, timeout=30)
        r.raise_for_status()
        items = r.json()
    except Exception as e:
        print(f"   ERRORE API: {e}")
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
    return run_dt, runs[latest_key]

def process_data():
    # --- MODIFICA FONDAMENTALE DI SICUREZZA ---
    # Prima controlliamo se ci sono dati, POI cancelliamo la cartella.
    run_dt, file_list = get_latest_run_files()
    
    if not file_list:
        print("!!! ATTENZIONE: Nessun dato trovato o API offline.")
        print("!!! ABORTO OPERAZIONE: Mantengo i vecchi dati per non rompere il sito.")
        sys.exit(0) # Esce senza errore (verde) ma non tocca nulla

    print(f"2. Dati trovati! Run: {run_dt}. Inizio pulizia e download...")
    
    # Ora è sicuro cancellare
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    catalog = []
    processed_hours = set()

    for idx, filename in enumerate(file_list):
        print(f"   Analisi GRIB: {filename}")
        local_path = "temp.grib2"
        try:
            with requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
        except: continue

        try:
            ds_wind = xr.open_dataset(local_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
            try: ds_temp = xr.open_dataset(local_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
            except: ds_temp = None
            try: ds_rain = xr.open_dataset(local_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
            except: ds_rain = None
        except: continue

        steps = range(ds_wind.sizes.get('step', 1))

        for i in steps:
            try:
                raw_step = ds_wind.step.values[i]
                if isinstance(raw_step, np.timedelta64): step_hours = int(raw_step / np.timedelta64(1, 'h'))
                else: step_hours = int(raw_step)

                if step_hours in processed_hours: continue

                d_w = ds_wind.isel(step=i) if 'step' in ds_wind.dims else ds_wind
                d_w = d_w.sortby('latitude', ascending=False).sortby('longitude', ascending=True)

                mask = ((d_w.latitude >= LAT_MIN) & (d_w.latitude <= LAT_MAX) & 
                        (d_w.longitude >= LON_MIN) & (d_w.longitude <= LON_MAX))
                cut_w = d_w.where(mask, drop=True)

                u_key = next((k for k in ['u10','u','10u'] if k in cut_w), None)
                v_key = next((k for k in ['v10','v','10v'] if k in cut_w), None)
                if not u_key or not v_key: continue

                u = np.nan_to_num(cut_w[u_key].values)
                v = np.nan_to_num(cut_w[v_key].values)

                # Geometria
                lat = cut_w.latitude.values
                lon = cut_w.longitude.values
                ny, nx = u.shape
                la1, lo1 = float(lat[0]), float(lon[0])
                dx = float(abs(lon[1] - lon[0]))
                dy = float(abs(lat[0] - lat[1]))

                # Dati opzionali
                temp, rain = np.zeros_like(u), np.zeros_like(u)
                if ds_temp:
                    d_t = ds_temp.isel(step=i) if 'step' in ds_temp.dims else ds_temp
                    d_t = d_t.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
                    cut_t = d_t.where(mask, drop=True)
                    t_key = next((k for k in ['t2m','2t','t'] if k in cut_t), None)
                    if t_key: temp = cut_t[t_key].values - 273.15
                
                if ds_rain:
                    d_r = ds_rain.isel(step=i) if 'step' in ds_rain.dims else ds_rain
                    d_r = d_r.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
                    cut_r = d_r.where(mask, drop=True)
                    r_key = next((k for k in ['tp','tot_prec'] if k in cut_r), None)
                    if r_key: rain = np.nan_to_num(cut_r[r_key].values)

                valid_dt = run_dt + timedelta(hours=step_hours)
                iso_date = valid_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

                # HEADER GRIB2 STANDARD (Essenziale per Leaflet-Velocity)
                header_u = {
                    "parameterCategory": 2, "parameterNumber": 2,
                    "nx": nx, "ny": ny, 
                    "lo1": lo1, "la1": la1, "dx": dx, "dy": dy, 
                    "refTime": iso_date
                }
                header_v = {
                    "parameterCategory": 2, "parameterNumber": 3,
                    "nx": nx, "ny": ny, 
                    "lo1": lo1, "la1": la1, "dx": dx, "dy": dy, 
                    "refTime": iso_date
                }

                step_data = {
                    "meta": header_u,
                    "wind_u": { "header": header_u, "data": np.round(u, 1).flatten().tolist() },
                    "wind_v": { "header": header_v, "data": np.round(v, 1).flatten().tolist() },
                    "temp": np.round(temp, 1).flatten().tolist(),
                    "rain": np.round(rain, 2).flatten().tolist()
                }

                out_name = f"step_{step_hours}.json"
                with open(f"{OUTPUT_DIR}/{out_name}", 'w') as jf: json.dump(step_data, jf)
                
                day_str = valid_dt.strftime("%d/%m")
                hour_str = valid_dt.strftime("%H:00")
                catalog.append({"file": out_name, "label": f"{day_str} {hour_str}", "hour": step_hours})
                processed_hours.add(step_hours)
                print(f"   Step +{step_hours}h OK")

            except Exception as e: continue

    catalog.sort(key=lambda x: x['hour'])
    # Salviamo il catalogo solo se abbiamo scaricato qualcosa
    if len(catalog) > 0:
        with open(f"{OUTPUT_DIR}/catalog.json", 'w') as f: json.dump(catalog, f)
        print("Finito con successo.")
    else:
        print("Nessun dato valido estratto. Non salvo catalogo vuoto.")

if __name__ == "__main__":
    process_data()
    
