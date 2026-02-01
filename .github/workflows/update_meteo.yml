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
# NOTA: L'endpoint per la lista è diverso da quello di download
API_LIST_URL = f"https://meteohub.agenziaitaliameteo.it/api/datasets/{DATASET_ID}/opendata"
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
        print(f"Trovati {len(files)} file totali nel dataset.")
        return files
    except Exception as e:
        print(f"Errore recupero lista file: {e}")
        return []

def parse_filename(filename):
    """
    Cerca di estrarre data e ora dal nome del file.
    Formati tipici MeteoHub: 
    - ..._YYYYMMDDHH_fff.grib2 (DataRun + OraPrevisione)
    """
    # Regex per trovare una data YYYYMMDDHH (es. 2026020112) e l'ora previsione (es. 000, 001)
    # Cerchiamo pattern tipo: 2026020112_000 o simili
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
    all_files = get_available_files()
    if not all_files:
        print("Nessun file trovato. Esco.")
        sys.exit(1)

    # 3. Raggruppa i file per RUN (Data + Ora di lancio modello)
    runs = {}
    for fname in all_files:
        run_id, step = parse_filename(fname)
        if run_id:
            if run_id not in runs: runs[run_id] = []
            runs[run_id].append((step, fname))

    if not runs:
        print("Impossibile decifrare i nomi dei file. Formato sconosciuto.")
        print("Esempio file trovato:", all_files[0] if all_files else "Nessuno")
        sys.exit(1)

    # 4. Trova la RUN più recente
    latest_run_id = sorted(runs.keys())[-1]
    print(f"Run più recente trovata: {latest_run_id}")
    
    # Prendi i file di questa run, ordinati per step (ora 0, 1, 2...)
    files_to_process = sorted(runs[latest_run_id], key=lambda x: x[0])
    
    catalog = [] 

    # 5. Elabora i primi N step
    count = 0
    for step, filename in files_to_process:
        if count >= FORECAST_HOURS: break
        
        print(f"Processing step +{step}h : {filename}")
        
        # Scarica
        url = f"{API_DOWNLOAD_URL}/{filename}"
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open("temp.grib2", 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        except Exception as e:
            print(f"Errore download {filename}: {e}")
            continue
        
        # Elabora GRIB
        try:
            # Filtro typeOfLevel per evitare conflitti di messaggi GRIB
            ds = xr.open_dataset("temp.grib2", engine='cfgrib', 
                                 backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround'}})
            
            # --- RITAGLIO ---
            mask = ((ds.latitude >= LAT_MIN) & (ds.latitude <= LAT_MAX) & 
                    (ds.longitude >= LON_MIN) & (ds.longitude <= LON_MAX))
            ds_cut = ds.where(mask, drop=True)

            # --- ESTRAZIONE DATI ---
            # Verifica nomi variabili (u10/u, v10/v, t2m, tp, msl)
            u = np.nan_to_num(ds_cut['u10'].values if 'u10' in ds_cut else ds_cut['u'].values)
            v = np.nan_to_num(ds_cut['v10'].values if 'v10' in ds_cut else ds_cut['v'].values)
            
            # Temp (K -> C)
            if 't2m' in ds_cut: temp = ds_cut['t2m'].values - 273.15
            else: temp = np.zeros_like(u)

            # Pioggia
            precip = ds_cut['tp'].values if 'tp' in ds_cut else np.zeros_like(u)

            # Pressione (Pa -> hPa)
            press = (ds_cut['msl'].values / 100.0) if 'msl' in ds_cut else np.zeros_like(u)

            # --- JSON ---
            lat = ds_cut.latitude.values
            lon = ds_cut.longitude.values
            ny, nx = u.shape
            
            # Risoluzione approssimativa
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
                # Arrotondiamo per file leggeri
                "wind_u": np.round(u, 1).flatten().tolist(),
                "wind_v": np.round(v, 1).flatten().tolist(),
                "temp": np.round(temp, 1).flatten().tolist(),
                "rain": np.round(precip, 2).flatten().tolist(),
                "press": np.round(press, 0).flatten().tolist()
            }
            
            out_name = f"step_{count}.json"
            with open(f"{OUTPUT_DIR}/{out_name}", 'w') as jf:
                json.dump(step_data, jf)
            
            # Parsing data run per etichetta (YYYYMMDDHH)
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
            print(f"Errore elaborazione {filename}: {e}")
            # Se fallisce un file, proviamo col prossimo
            continue

    # Salva catalogo
    with open(f"{OUTPUT_DIR}/catalog.json", 'w') as f:
        json.dump(catalog, f)
    
    print(f"Finito. Generati {count} step.")

if __name__ == "__main__":
    process_all()
    
