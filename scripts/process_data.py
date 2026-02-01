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

LAT_MIN, LAT_MAX = 36.0, 39.0
LON_MIN, LON_MAX = 11.5, 16.0

def get_run_files():
    """Trova i file della run più recente raggruppati per tipologia."""
    print("1. Cerco ultima run disponibile...")
    try:
        r = requests.get(API_LIST_URL, timeout=15)
        r.raise_for_status()
        items = r.json()
    except Exception as e:
        print(f"Errore API: {e}")
        return None, None

    runs = {}
    for item in items:
        if isinstance(item, dict) and 'date' in item and 'run' in item:
            key = f"{item['date']} {item['run']}"
            if key not in runs: runs[key] = []
            runs[key].append(item['filename'])

    if not runs: return None, None
    
    latest_key = sorted(runs.keys())[-1]
    run_dt = datetime.strptime(latest_key, "%Y-%m-%d %H:%M")
    print(f"   -> Run trovata: {latest_key}")
    
    return run_dt, runs[latest_key]

def download_file(filename):
    if os.path.exists(filename): return True
    print(f"   -> Download: {filename} ...")
    try:
        with requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
        return True
    except:
        return False

def process_data():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    run_dt, file_list = get_run_files()
    if not file_list: sys.exit(1)

    # --- FASE 1: Identificazione e Download dei File necessari ---
    # Dobbiamo trovare quale file ha cosa. Scarichiamo e controlliamo.
    
    ds_wind = None
    ds_temp = None
    ds_rain = None
    
    vars_found = {'u': False, 't': False, 'rain': False}

    print("2. Analisi e unione file...")
    
    # Mappatura variabili temporanea
    datasets = []

    for fname in file_list:
        if not download_file(fname): continue
        
        try:
            # Apriamo il file (senza filtri stretti per vedere cosa c'è)
            ds = xr.open_dataset(fname, engine='cfgrib', 
                                 backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround'}})
            keys = list(ds.data_vars)
            
            # Identifica VENTO
            if not vars_found['u']:
                if any(k in keys for k in ['u10', 'u', '10u']):
                    print(f"   [OK] File Vento: {fname}")
                    ds_wind = ds
                    vars_found['u'] = True
            
            # Identifica TEMP (spesso nello stesso del vento, ma controlliamo)
            if not vars_found['t']:
                if any(k in keys for k in ['t2m', 't', '2t']):
                    print(f"   [OK] File Temp: {fname}")
                    ds_temp = ds
                    vars_found['t'] = True

            # Per la pioggia serve filtro diverso (surface)
            if not vars_found['rain']:
                try:
                    ds_r = xr.open_dataset(fname, engine='cfgrib', 
                                           backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
                    keys_r = list(ds_r.data_vars)
                    if any(k in keys_r for k in ['tp', 'tot_prec']):
                        print(f"   [OK] File Pioggia: {fname}")
                        ds_rain = ds_r
                        vars_found['rain'] = True
                except:
                    pass

        except Exception as e:
            continue

    if not ds_wind:
        print("ERRORE: Vento non trovato in nessun file.")
        sys.exit(1)

    # Se mancano temp o rain, usiamo il vento come placeholder (con zeri)
    if not ds_temp: ds_temp = ds_wind
    if not ds_rain: ds_rain = ds_wind

    # --- FASE 2: Unione e Export ---
    catalog = []
    # Usiamo gli step del vento come riferimento
    steps = range(min(ds_wind.sizes.get('step', 1), 24))

    print(f"3. Generazione {len(steps)} step orari...")

    for i in steps:
        try:
            # Sincronizzazione step (assumiamo che tutti i file abbiano gli stessi step orari)
            # Nota: Potrebbe servire logica più complessa se gli step differiscono
            step_idx = i
            
            # --- VENTO ---
            d_w = ds_wind.isel(step=step_idx) if 'step' in ds_wind.dims else ds_wind
            mask = ((d_w.latitude >= LAT_MIN) & (d_w.latitude <= LAT_MAX) & 
                    (d_w.longitude >= LON_MIN) & (d_w.longitude <= LON_MAX))
            cut_w = d_w.where(mask, drop=True)
            
            u_var = next((k for k in ['u10', 'u', '10u'] if k in cut_w), None)
            v_var = next((k for k in ['v10', 'v', '10v'] if k in cut_w), None)
            u = np.nan_to_num(cut_w[u_var].values)
            v = np.nan_to_num(cut_w[v_var].values)

            # --- TEMP ---
            # Se ds_temp è diverso, dobbiamo ritagliare anche lui
            d_t = ds_temp.isel(step=step_idx) if 'step' in ds_temp.dims else ds_temp
            # Importante: Interpola o assumi stessa griglia. Qui assumiamo stessa griglia ritagliata.
            cut_t = d_t.where(mask, drop=True)
            t_var = next((k for k in ['t2m', 't', '2t'] if k in cut_t), None)
            if t_var:
                temp = cut_t[t_var].values - 273.15
            else:
                temp = np.zeros_like(u)

            # --- RAIN ---
            d_r = ds_rain.isel(step=step_idx) if 'step' in ds_rain.dims else ds_rain
            cut_r = d_r.where(mask, drop=True)
            r_var = next((k for k in ['tp', 'tot_prec'] if k in cut_r), None)
            if r_var:
                rain = np.nan_to_num(cut_r[r_var].values)
            else:
                rain = np.zeros_like(u)

            # --- Export ---
            lat = cut_w.latitude.values
            lon = cut_w.longitude.values
            ny, nx = u.shape
            
            # Fix per griglie non quadrate
            dx = float((lon.max()-lon.min())/(nx-1)) if nx > 1 else 0.02
            dy = float((lat.max()-lat.min())/(ny-1)) if ny > 1 else 0.02
            
            step_hours = int(ds_wind.step.values[i] / 3.6e12) if 'step' in ds_wind.dims else 0
            valid_dt = run_dt + timedelta(hours=step_hours)

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
                "rain": np.round(rain, 2).flatten().tolist(), # Pioggia con 2 decimali
                "press": [0] # Pressione facoltativa
            }

            out_name = f"step_{i}.json"
            with open(f"{OUTPUT_DIR}/{out_name}", 'w') as jf:
                json.dump(step_data, jf)
            
            catalog.append({
                "file": out_name,
                "label": valid_dt.strftime("%d/%m %H:00"),
                "hour": step_hours
            })

        except Exception as e:
            print(f"Errore step {i}: {e}")
            continue

    with open(f"{OUTPUT_DIR}/catalog.json", 'w') as f:
        json.dump(catalog, f)
    print("Finito.")

if __name__ == "__main__":
    process_data()
    
