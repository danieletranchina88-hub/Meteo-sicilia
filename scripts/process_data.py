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

def get_latest_file_info():
    """Scarica la lista e trova il file della run più recente."""
    print(f"Recupero lista file da: {API_LIST_URL}")
    try:
        r = requests.get(API_LIST_URL, timeout=15)
        r.raise_for_status()
        items = r.json()
    except Exception as e:
        print(f"Errore API: {e}")
        return None, None

    valid_runs = []
    for item in items:
        # Leggiamo date e run direttamente dal JSON
        if isinstance(item, dict) and 'date' in item and 'run' in item:
            try:
                run_str = f"{item['date']} {item['run']}"
                dt = datetime.strptime(run_str, "%Y-%m-%d %H:%M")
                valid_runs.append((dt, item['filename']))
            except:
                continue

    if not valid_runs:
        print("Nessuna run valida trovata.")
        return None, None

    # Ordina e prendi l'ultimo
    valid_runs.sort(key=lambda x: x[0])
    return valid_runs[-1]

def find_wind_variables(ds):
    """Cerca intelligentemente le variabili del vento nel dataset."""
    keys = list(ds.data_vars)
    print(f"   >>> Variabili trovate nel GRIB: {keys}")  # DEBUG FONDAMENTALE
    
    # Possibili nomi per U e V
    u_candidates = ['u10', '10u', 'u', 'U_10M', 'ut']
    v_candidates = ['v10', '10v', 'v', 'V_10M', 'vt']
    
    u_var = next((k for k in u_candidates if k in keys), None)
    v_var = next((k for k in v_candidates if k in keys), None)
    
    return u_var, v_var

def process_data():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    run_dt, filename = get_latest_file_info()
    if not filename:
        sys.exit(1)
        
    print(f"Run rilevata: {run_dt} - File: {filename}")

    # Scarica
    grib_path = "dataset.grib2"
    url = f"{API_DOWNLOAD_URL}/{filename}"
    print(f"Download in corso...")
    
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(grib_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
    except Exception as e:
        print(f"Errore download: {e}")
        sys.exit(1)

    # --- TENTATIVI DI APERTURA GRIB ---
    # I file GRIB sono complessi, proviamo diversi filtri per trovare il vento
    ds = None
    
    # Tentativo 1: heightAboveGround (Standard per vento 10m)
    print("Tentativo apertura: filter_by_keys={'typeOfLevel': 'heightAboveGround'}")
    try:
        ds = xr.open_dataset(grib_path, engine='cfgrib', 
                             backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround'}})
        u_var, v_var = find_wind_variables(ds)
        if not u_var: 
            print("   -> Variabili vento non trovate qui. Chiudo e riprovo.")
            ds.close()
            ds = None
    except Exception as e:
        print(f"   -> Fallito: {e}")

    # Tentativo 2: Nessun filtro (apre tutto ciò che può, ma rischia conflitti)
    if ds is None:
        print("Tentativo apertura: Senza filtri (potrebbe dare warning)")
        try:
            ds = xr.open_dataset(grib_path, engine='cfgrib')
            u_var, v_var = find_wind_variables(ds)
        except Exception as e:
            print(f"   -> Fallito: {e}")

    if ds is None or not u_var or not v_var:
        print("ERRORE FATALE: Impossibile trovare variabili vento (u, v) nel file.")
        # Se fallisce qui, nei log vedremo la lista delle variabili stampata sopra
        # e potremo correggere i nomi manualmente.
        sys.exit(1)

    print(f"USO VARIABILI VENTO: U='{u_var}', V='{v_var}'")

    # --- ELABORAZIONE ---
    catalog = []
    steps = range(min(ds.sizes.get('step', 1), 24)) # Processa max 24 ore

    for i in steps:
        try:
            # Selezione temporale
            ds_step = ds.isel(step=i) if 'step' in ds.dims else ds
            
            # Calcolo ora validità
            step_hours = int(ds.step.values[i] / 3.6e12) if 'step' in ds.dims else 0
            valid_dt = run_dt + timedelta(hours=step_hours)
            
            # --- RITAGLIO SICILIA ---
            # Gestione coordinate (a volte lat/lon non sono dimensioni)
            if 'latitude' in ds_step.coords:
                mask = ((ds_step.latitude >= LAT_MIN) & (ds_step.latitude <= LAT_MAX) & 
                        (ds_step.longitude >= LON_MIN) & (ds_step.longitude <= LON_MAX))
                ds_cut = ds_step.where(mask, drop=True)
            else:
                ds_cut = ds_step # Fallback se coordinate non standard

            # Estrazione dati
            u = np.nan_to_num(ds_cut[u_var].values)
            v = np.nan_to_num(ds_cut[v_var].values)
            
            # Temperatura (cerca vari nomi)
            t_var = next((k for k in ['t2m', 't', '2t'] if k in ds_cut), None)
            temp = (ds_cut[t_var].values - 273.15) if t_var else np.zeros_like(u)

            # Metadata griglia
            lat = ds_cut.latitude.values
            lon = ds_cut.longitude.values
            ny, nx = u.shape
            
            dx = float((lon.max()-lon.min())/(nx-1)) if nx > 1 else 0.02
            dy = float((lat.max()-lat.min())/(ny-1)) if ny > 1 else 0.02

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
                "rain": [0], # Placeholder per ora
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
            print(f"Step +{step_hours}h OK")

        except Exception as e:
            print(f"Errore step {i}: {e}")
            continue

    if catalog:
        with open(f"{OUTPUT_DIR}/catalog.json", 'w') as f:
            json.dump(catalog, f)
        print("Successo! Catalog generato.")

if __name__ == "__main__":
    process_data()
    
