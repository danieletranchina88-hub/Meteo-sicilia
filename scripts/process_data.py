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
MAX_PIXELS = 600_000 

def get_latest_run_files():
    print("--- 1. RICERCA DATI ---")
    try:
        r = requests.get(API_LIST_URL, timeout=30)
        r.raise_for_status()
        items = r.json()
        runs = {}
        for item in items:
            if isinstance(item, dict) and 'date' in item and 'run' in item:
                key = f"{item['date']} {item['run']}"
                runs.setdefault(key, []).append(item['filename'])
        
        if not runs:
            print("ERRORE: Nessuna run trovata nell'API.")
            return None, []
            
        latest_key = sorted(runs.keys())[-1]
        print(f"Ultima run disponibile: {latest_key}")
        run_dt = datetime.strptime(latest_key, "%Y-%m-%d %H:%M")
        return run_dt, sorted(runs[latest_key])[:48] # Prendiamo le prime 48 ore
    except Exception as e:
        print(f"ERRORE CONNESSIONE API: {e}")
        return None, []

def extract_raw_grid(ds, mask, var_names):
    """Funzione di estrazione sicura per evitare crash se la variabile manca"""
    for name in var_names:
        if name in ds:
            try:
                return ds[name].where(mask, drop=True).values
            except:
                continue
    return None

def process_data():
    run_dt, file_list = get_latest_run_files()
    if not file_list:
        print("Operazione annullata: File non trovati.")
        return

    print(f"--- 2. ELABORAZIONE ({len(file_list)} file) ---")

    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    catalog = []

    for idx, filename in enumerate(file_list):
        print(f"[{idx+1}/{len(file_list)}] Download {filename}...", end=" ", flush=True)
        try:
            r = requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True, timeout=60)
            r.raise_for_status()
            with open(TEMP_FILE, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
            print("OK.")
        except Exception as e:
            print(f"FALLITO: {e}")
            continue

        try:
            # Apriamo il file per vedere cosa contiene (Wind come riferimento)
            ds_wind = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
            
            # Ordiniamo e tagliamo subito sull'Italia
            ds_wind = ds_wind.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
            mask = (ds_wind.latitude >= LAT_MIN) & (ds_wind.latitude <= LAT_MAX) & (ds_wind.longitude >= LON_MIN) & (ds_wind.longitude <= LON_MAX)
            cut_ref = ds_wind.where(mask, drop=True)
            
            if cut_ref.latitude.size == 0:
                print("   Skipping: File fuori dall'area geografica impostata.")
                continue

            # Estrazione variabili con i tuoi nomi originali
            u_vals = np.nan_to_num(extract_raw_grid(ds_wind, mask, ['u10', 'u']))
            v_vals = np.nan_to_num(extract_raw_grid(ds_wind, mask, ['v10', 'v']))
            
            # Temperatura e Umidità
            ds_t = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
            t_raw = extract_raw_grid(ds_t, mask, ['t2m', 't'])
            d_raw = extract_raw_grid(ds_t, mask, ['d2m', '2d'])
            
            # Pressione
            ds_p = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}})
            p_raw = extract_raw_grid(ds_p, mask, ['pmsl', 'prmsl', 'msl'])

            # Preparazione dati finali
            temp_c = t_raw - 273.15
            press = (p_raw / 100.0) if np.nanmax(p_raw) > 80000 else p_raw
            
            # Meta
            ny, nx = temp_c.shape
            header = {
                "nx": nx, "ny": ny,
                "lo1": float(cut_ref.longitude[0]), "la1": float(cut_ref.latitude[0]),
                "dx": float(abs(cut_ref.longitude[1] - cut_ref.longitude[0])),
                "dy": float(abs(cut_ref.latitude[0] - cut_ref.latitude[1]))
            }

            step_hours = int(cut_ref.step.values / np.timedelta64(1, 'h'))
            valid_dt = run_dt + timedelta(hours=step_hours)

            # --- SALVATAGGIO JSON COMPATTO ---
            step_data = {
                "meta": header,
                "wind_u": {"data": np.round(u_vals, 1).flatten().tolist()},
                "wind_v": {"data": np.round(v_vals, 1).flatten().tolist()},
                "temp": np.round(temp_c, 1).flatten().tolist(),
                "feels_like": np.round(temp_c, 1).flatten().tolist(), # Iniziamo col mettere temp qui
                "press": np.round(press, 1).flatten().tolist()
            }

            out_name = f"step_{step_hours}.json"
            with open(f"{TEMP_DIR}/{out_name}", 'w') as jf:
                json.dump(step_data, jf, separators=(',', ':'))

            catalog.append({"file": out_name, "label": valid_dt.strftime('%d/%m %H:00'), "hour": step_hours})
            print(f"   Successo: salvato {out_name}")

        except Exception as e:
            print(f"   ERRORE ELABORAZIONE FILE: {e}")
            continue

    if catalog:
        print("--- 3. COMPLETAMENTO ---")
        catalog.sort(key=lambda x: x['hour'])
        with open(f"{TEMP_DIR}/catalog.json", 'w') as f:
            json.dump(catalog, f, separators=(',', ':'))

        if os.path.exists(FINAL_DIR): shutil.rmtree(FINAL_DIR)
        shutil.move(TEMP_DIR, FINAL_DIR)
        print(f"Dati aggiornati con successo in {FINAL_DIR}.")
    else:
        print("ERRORE FINALE: Nessun file JSON è stato generato.")

if __name__ == "__main__":
    process_data()
