import requests
import xarray as xr
import numpy as np
import json
import os
import sys
import shutil
import re
from datetime import datetime, timedelta

# --- CONFIGURAZIONE ---
DATASET_ID = "ICON_2I_SURFACE_PRESSURE_LEVELS"
# Endpoint per la lista dei file
API_LIST_URL = f"https://meteohub.agenziaitaliameteo.it/api/datasets/{DATASET_ID}/opendata"
# Endpoint per scaricare il singolo file
API_DOWNLOAD_URL = "https://meteohub.agenziaitaliameteo.it/api/opendata"
OUTPUT_DIR = "data_weather"

# Limiti Sicilia
LAT_MIN, LAT_MAX = 36.0, 39.0
LON_MIN, LON_MAX = 11.5, 16.0

# Quante ore di previsione scaricare?
FORECAST_HOURS = 12 

def get_available_files():
    """Scarica la lista di TUTTI i file disponibili dal server."""
    print(f"Recupero lista file da: {API_LIST_URL}")
    try:
        r = requests.get(API_LIST_URL, timeout=15)
        r.raise_for_status()
        files = r.json()
        print(f"Trovati {len(files)} elementi nel dataset.")
        # Debug: stampiamo il primo elemento per capire com'è fatto
        if len(files) > 0:
            print(f"Esempio struttura file: {files[0]}")
        return files
    except Exception as e:
        print(f"Errore recupero lista file: {e}")
        return []

def get_filename_str(item):
    """Estrae il nome del file (stringa) dall'elemento della lista."""
    if isinstance(item, str):
        return item
    elif isinstance(item, dict):
        # MeteoHub spesso usa 'name' o 'filename' dentro un oggetto JSON
        if 'name' in item: return item['name']
        if 'filename' in item: return item['filename']
        if 'file' in item: return item['file']
    
    # Se non riusciamo a decifrarlo, lo stampiamo per debug
    print(f"ATTENZIONE: Impossibile trovare il nome in questo oggetto: {item}")
    return str(item) # Fallback disperato

def parse_filename(filename):
    """
    Cerca di estrarre data e ora dal nome del file.
    Regex flessibile per diversi formati.
    """
    # Cerca una sequenza di 10 cifre (YYYYMMDDHH) seguita da _ e 3 cifre (step)
    # Esempio: ...2026020112_000...
    match = re.search(r'(\d{10})_(\d{3})', filename)
    if match:
        run_str = match.group(1) # es. 2026020112
        step_str = match.group(2) # es. 000
        return run_str, int(step_str)
    return None, None

def process_all():
    # 1. Preparazione cartella
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # 2. Ottieni lista file dal server
    all_items = get_available_files()
    if not all_items:
        print("Nessun file trovato. Esco.")
        sys.exit(1)

    # 3. Raggruppa i file per RUN
    runs = {}
    valid_files_count = 0
    
    for item in all_items:
        fname = get_filename_str(item)
        
        run_id, step = parse_filename(fname)
        if run_id:
            if run_id not in runs: runs[run_id] = []
            runs[run_id].append((step, fname))
            valid_files_count += 1

    if not runs:
        print("ERRORE: Non sono riuscito a leggere le date dai nomi dei file.")
        sys.exit(1)

    print(f"Analisi completata: trovate {len(runs)} run diverse per un totale di {valid_files_count} file validi.")

    # 4. Trova la RUN più recente
    latest_run_id = sorted(runs.keys())[-1]
    print(f"=== INIZIO ELABORAZIONE RUN: {latest_run_id} ===")
    
    # Prendi i file di questa run, ordinati per step
    files_to_process = sorted(runs[latest_run_id], key=lambda x: x[0])
    
    catalog = [] 

    # 5. Elabora i primi N step
    count = 0
    for step, filename in files_to_process:
        if count >= FORECAST_HOURS: break
        
        print(f"[{count+1}/{FORECAST_HOURS}] Scaricamento step +{step}h : {filename}")
        
        # Scarica
        url = f"{API_DOWNLOAD_URL}/{filename}"
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open("temp.grib2", 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        except Exception as e:
            print(f"Errore download {filename}: {e}")
            continue
        
        # Elabora GRIB
        print("   -> Apertura GRIB...")
        try:
            # Filtro typeOfLevel per evitare conflitti
            ds = xr.open_dataset("temp.grib2", engine='cfgrib', 
                                 backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround'}})
            
            # --- RITAGLIO ---
            mask = ((ds.latitude >= LAT_MIN) & (ds.latitude <= LAT_MAX) & 
                    (ds.longitude >= LON_MIN) & (ds.longitude <= LON_MAX))
            ds_cut = ds.where(mask, drop=True)

            # --- ESTRAZIONE DATI ---
            # Normalizzazione nomi variabili
            u = np.nan_to_num(ds_cut['u10'].values if 'u10' in ds_cut else ds_cut['u'].values)
            v = np.nan_to_num(ds_cut['v10'].values if 'v10' in ds_cut else ds_cut['v'].values)
            
            # Temp (K -> C)
            if 't2m' in ds_cut: temp = ds_cut['t2m'].values - 273.15
            elif 't' in ds_cut: temp = ds_cut['t'].values - 273.15 # A volte è solo 't'
            else: temp = np.zeros_like(u)

            # Pioggia (Total Precipitation)
            precip = ds_cut['tp'].values if 'tp' in ds_cut else np.zeros_like(u)

            # Pressione (Pa -> hPa)
            press = (ds_cut['msl'].values / 100.0) if 'msl' in ds_cut else np.zeros_like(u)

            # --- JSON ---
            lat = ds_cut.latitude.values
            lon = ds_cut.longitude.values
            ny, nx = u.shape
            
            dx = float((lon.max()-lon.min())/(nx-1)) if nx > 1 else 0.02
            dy = float((lat.max()-lat.min())/(ny-1)) if ny > 1 else 0.02

            step_data = {
                "meta": {
                    "run": latest_run_id,
                    "step": step,
                    "nx": nx, "ny": ny,
                    "la1": float(lat.max()), "lo1": float(lon.min()),
                    "dx": dx, "dy": dy
                },
                "wind_u": np.round(u, 1).flatten().tolist(),
                "wind_v": np.round(v, 1).flatten().tolist(),
                "temp": np.round(temp, 1).flatten().tolist(),
                "rain": np.round(precip, 2).flatten().tolist(),
                "press": np.round(press, 0).flatten().tolist()
            }
            
            out_name = f"step_{count}.json"
            with open(f"{OUTPUT_DIR}/{out_name}", 'w') as jf:
                json.dump(step_data, jf)
            
            # Data validità
            run_dt = datetime.strptime(latest_run_id, "%Y%m%d%H")
            valid_dt = run_dt + timedelta(hours=step)
            
            catalog.append({
                "file": out_name,
                "label": valid_dt.strftime("%d/%m %H:00"),
                "hour": step
            })
            
            count += 1
            ds.close()

        except Exception as e:
            print(f"   -> ERRORE elaborazione file {filename}: {e}")
            continue

    # Salva catalogo
    with open(f"{OUTPUT_DIR}/catalog.json", 'w') as f:
        json.dump(catalog, f)
    
    print(f"Finito. Generati {count} step nel catalogo.")

if __name__ == "__main__":
    process_all()
    
