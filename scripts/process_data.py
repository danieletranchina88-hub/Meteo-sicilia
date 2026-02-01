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

# Coordinate Sicilia
LAT_MIN, LAT_MAX = 36.0, 39.0
LON_MIN, LON_MAX = 11.5, 16.0

def get_latest_run_files():
    """Trova TUTTI i file appartenenti all'ultima run disponibile."""
    print(f"1. Recupero lista file da MeteoHub...")
    try:
        r = requests.get(API_LIST_URL, timeout=15)
        r.raise_for_status()
        items = r.json()
    except Exception as e:
        print(f"Errore API: {e}")
        return None, []

    # Raggruppa per data/ora run
    runs = {}
    for item in items:
        if isinstance(item, dict) and 'date' in item and 'run' in item:
            run_key = f"{item['date']} {item['run']}"
            if run_key not in runs: runs[run_key] = []
            runs[run_key].append(item['filename'])

    if not runs:
        print("Nessuna run valida trovata.")
        return None, []

    # Prende la run più recente
    latest_key = sorted(runs.keys())[-1]
    print(f"2. Run più recente trovata: {latest_key}")
    print(f"   -> Ci sono {len(runs[latest_key])} file in questa run.")
    
    return datetime.strptime(latest_key, "%Y-%m-%d %H:%M"), runs[latest_key]

def process_data():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    run_dt, file_list = get_latest_run_files()
    if not file_list:
        sys.exit(1)

    # --- FASE DI RICERCA DEL FILE GIUSTO ---
    # Dobbiamo trovare quale dei file contiene il VENTO (u, v)
    
    target_ds = None
    target_u_var = None
    target_v_var = None
    
    print("3. Inizio scansione file per cercare il vento...")
    
    for filename in file_list:
        print(f"   Analisi file: {filename} ...")
        
        # Scarica file temporaneo
        url = f"{API_DOWNLOAD_URL}/{filename}"
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open("temp_scan.grib2", 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
        except:
            print("   -> Errore download, salto.")
            continue

        # Prova ad aprire cercando variabili Vento (livello 10m)
        try:
            # Filtro stretto per wind 10m
            ds = xr.open_dataset("temp_scan.grib2", engine='cfgrib', 
                                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
            
            # Controlla se ci sono u e v
            keys = list(ds.data_vars)
            u = next((k for k in ['u10', 'u', '10u'] if k in keys), None)
            v = next((k for k in ['v10', 'v', '10v'] if k in keys), None)
            
            if u and v:
                print(f"   >>> TROVATO! Questo file contiene il vento: {u}, {v}")
                target_ds = ds
                target_u_var = u
                target_v_var = v
                break # Trovato! Usciamo dal ciclo
            else:
                print(f"   -> Niente vento qui. (Variabili trovate: {keys})")
                ds.close()
                
        except Exception as e:
            # Se fallisce con level=10, proviamo senza filtro livello
            try:
                ds = xr.open_dataset("temp_scan.grib2", engine='cfgrib',
                                   backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround'}})
                keys = list(ds.data_vars)
                u = next((k for k in ['u10', 'u', '10u'] if k in keys), None)
                v = next((k for k in ['v10', 'v', '10v'] if k in keys), None)
                if u and v:
                    print(f"   >>> TROVATO (Senza filtro livello)! {u}, {v}")
                    target_ds = ds
                    target_u_var = u
                    target_v_var = v
                    break
                ds.close()
            except:
                pass
            print(f"   -> File non compatibile o senza vento.")

    if target_ds is None:
        print("ERRORE FATALE: Ho controllato tutti i file ma nessuno contiene u/v (Vento).")
        sys.exit(1)

    # --- ELABORAZIONE ---
    print(f"4. Elaborazione dati meteo per Sicilia...")
    catalog = []
    
    # Se il file ha la dimensione 'step' (più orari), usiamo quella. Altrimenti assumiamo 1 step.
    steps = range(min(target_ds.sizes.get('step', 1), 24))

    for i in steps:
        try:
            ds_step = target_ds.isel(step=i) if 'step' in target_ds.dims else target_ds
            
            # Calcolo Data Validità
            step_hours = int(target_ds.step.values[i] / 3.6e12) if 'step' in target_ds.dims else 0
            valid_dt = run_dt + timedelta(hours=step_hours)

            # --- RITAGLIO ---
            mask = ((ds_step.latitude >= LAT_MIN) & (ds_step.latitude <= LAT_MAX) & 
                    (ds_step.longitude >= LON_MIN) & (ds_step.longitude <= LON_MAX))
            ds_cut = ds_step.where(mask, drop=True)

            # Estrazione Vento
            u = np.nan_to_num(ds_cut[target_u_var].values)
            v = np.nan_to_num(ds_cut[target_v_var].values)
            
            # Temperatura (se c'è nel file vento bene, altrimenti mettiamo placeholder)
            # Spesso u,v,t sono nello stesso file
            t_var = next((k for k in ['t2m', 't', '2t'] if k in ds_cut), None)
            if t_var:
                temp = ds_cut[t_var].values - 273.15
            else:
                temp = np.zeros_like(u) # Se manca la temp, mappa grigia ma vento ok

            # Griglia
            lat = ds_cut.latitude.values
            lon = ds_cut.longitude.values
            ny, nx = u.shape
            
            dx = float((lon.max()-lon.min())/(nx-1)) if nx > 1 else 0.02
            dy = float((lat.max()-lat.min())/(ny-1)) if ny > 1 else 0.02

            # JSON Output
            step_data = {
                "meta": {
                    "run": run_dt.strftime("%Y%m%d%H"),
                    "step": step_hours,
                    "nx": nx, "ny": ny,
                    "la1": float(lat.max()), "lo1": float(lon.min()),
                    "dx": dx, "dy": dy
                },
                "wind_u": np.round(u, 1).flatten().tolist(),
                "wind_v": np.round(v, 1).flatten().tolist(),
                "temp": np.round(temp, 1).flatten().tolist(),
                "rain": [0], 
                "press": [0]
            }
            
            out_name = f"step_{i}.json"
            with open(f"{OUTPUT_DIR}/{out_name}", 'w') as jf:
                json.dump(step_data, jf)
            
            catalog.append({
                "file": out_name,
                "label": valid_dt.strftime("%d/%m %H:00"),
                "hour": step_hours
            })
            print(f"   -> Generato step +{step_hours}h")

        except Exception as e:
            print(f"Errore generazione step {i}: {e}")
            continue

    if catalog:
        with open(f"{OUTPUT_DIR}/catalog.json", 'w') as f:
            json.dump(catalog, f)
        print("5. SUCCESSO! Dati pronti.")
    else:
        print("Nessun dato generato.")
        sys.exit(1)

if __name__ == "__main__":
    process_data()
    
