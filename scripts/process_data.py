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

# --- CONFIGURAZIONE SCIENTIFICA ---
# Per un lavoro perfetto, dovresti avere un file .nc con l'altitudine reale dell'Italia.
# Se non lo hai, il codice userà l'orografia standard del modello (meno precisa).
PATH_DEM_ITALIA = "topografia_italia_1km.nc" 
LAPSE_RATE = 0.0065  # 0.65°C ogni 100 metri

LAT_MIN, LAT_MAX = 35.0, 48.5
LON_MIN, LON_MAX = 6.0, 19.5
MAX_PIXELS = 600_000 

def get_latest_run_files():
    print("1. Ricerca dati ICON-2I...", flush=True)
    try:
        r = requests.get(API_LIST_URL, timeout=30)
        items = r.json()
    except: return None, []
    
    runs = {}
    for item in items:
        if isinstance(item, dict) and 'date' in item and 'run' in item:
            key = f"{item['date']} {item['run']}"
            runs.setdefault(key, []).append(item['filename'])
    if not runs: return None, []
    latest_key = sorted(runs.keys())[-1]
    return datetime.strptime(latest_key, "%Y-%m-%d %H:%M"), runs[latest_key][:48]

def process_data():
    run_dt, file_list = get_latest_run_files()
    if not file_list: sys.exit(0)

    # Caricamento DEM (Topografia Reale) se presente
    real_dem = None
    if os.path.exists(PATH_DEM_ITALIA):
        print(f"-> Caricamento DEM Topografico: {PATH_DEM_ITALIA}")
        real_dem = xr.open_dataset(PATH_DEM_ITALIA)

    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    catalog = []

    for idx, filename in enumerate(file_list):
        print(f"[{idx+1:02d}] Elaborazione {filename}...", flush=True)
        try:
            r = requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True)
            with open(TEMP_FILE, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
        except: continue

        try:
            # Estrazione parametri base
            ds_wind = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel':'heightAboveGround','level':10}})
            ds_temp = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel':'heightAboveGround','level':2}})
            
            # --- NOVITÀ: Estrazione Altezza Modello (Orografia ICON) ---
            # Questo serve per sapere a che altezza il modello "pensa" che sia il terreno
            try:
                ds_hsurf = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys':{'shortName':'gh','typeOfLevel':'surface'}})
            except: ds_hsurf = None

            # ... (Logica per Pressione, Pioggia, Nuvole come prima) ...
            
            # Taglio e Maschera
            mask = (ds_temp.latitude >= LAT_MIN) & (ds_temp.latitude <= LAT_MAX) & (ds_temp.longitude >= LON_MIN) & (ds_temp.longitude <= LON_MAX)
            cut_t = ds_temp.where(mask, drop=True)
            
            # Dati Temperatura Originali
            t_vals = cut_t.t2m.values - 273.15
            
            # --- APPLICAZIONE GRADIENTE TERMICO (Scientifico) ---
            if ds_hsurf is not None and real_dem is not None:
                print("   -> Applicazione correzione orografica...")
                h_model = ds_hsurf.gh.where(mask, drop=True).values
                # Interpola il DEM reale sulla griglia del modello
                h_real = real_dem.interp(latitude=cut_t.latitude, longitude=cut_t.longitude).elevation.values
                
                # Formula: T_corretta = T_modello + (H_modello - H_reale) * 0.0065
                # Se la montagna reale è più alta del modello, la T scende.
                t_vals = t_vals + (h_model - h_real) * LAPSE_RATE

            # --- SALVATAGGIO ---
            # (Inclusa la minificazione del JSON come prima)
            step_hours = int(ds_temp.step.values / np.timedelta64(1, 'h'))
            valid_dt = run_dt + timedelta(hours=step_hours)
            
            step_data = {
                "meta": {"nx": len(cut_t.longitude), "ny": len(cut_t.latitude), "lo1": float(cut_t.longitude[0]), "la1": float(cut_t.latitude[0])},
                "temp": np.round(t_vals, 1).flatten().tolist(),
                # ... (altri dati) ...
            }

            out_name = f"step_{step_hours}.json"
            with open(f"{TEMP_DIR}/{out_name}", 'w') as jf:
                json.dump(step_data, jf, separators=(',', ':'))

            catalog.append({"file": out_name, "label": valid_dt.strftime('%d/%m %H:00'), "hour": step_hours})

        except Exception as e:
            print(f" Errore: {e}")
            continue

    # Cleanup e Catalogo
    with open(f"{TEMP_DIR}/catalog.json", 'w') as f: json.dump(sorted(catalog, key=lambda x:x['hour']), f, separators=(',', ':'))
    if os.path.exists(FINAL_DIR): shutil.rmtree(FINAL_DIR)
    shutil.move(TEMP_DIR, FINAL_DIR)
    print("FINITO.")

if __name__ == "__main__": process_data()
