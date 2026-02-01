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

def get_latest_run_file():
    print("1. Cerco ultima run...")
    try:
        r = requests.get(API_LIST_URL, timeout=15)
        r.raise_for_status()
        items = r.json()
    except Exception as e:
        print(f"Errore API: {e}")
        return None, None

    # Trova l'ultimo file in assoluto basato su data e ora
    valid_items = []
    for item in items:
        if isinstance(item, dict) and 'date' in item and 'run' in item:
            try:
                dt = datetime.strptime(f"{item['date']} {item['run']}", "%Y-%m-%d %H:%M")
                valid_items.append((dt, item['filename']))
            except: continue
    
    if not valid_items: return None, None
    
    # Ordina e prendi l'ultimo
    valid_items.sort(key=lambda x: x[0])
    last_run_dt, last_filename = valid_items[-1]
    
    print(f"   Run trovata: {last_run_dt} -> File: {last_filename}")
    return last_run_dt, last_filename

def process_data():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    run_dt, filename = get_latest_run_file()
    if not filename: sys.exit(1)

    # --- DOWNLOAD ---
    print("2. Download file unico...")
    local_path = "meteo_data.grib2"
    try:
        with requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
    except Exception as e:
        print(f"Errore download: {e}")
        sys.exit(1)

    # --- ESTRAZIONE CHIRURGICA ---
    print("3. Estrazione Variabili (Separata)...")

    # A. VENTO (Livello 10m)
    print("   -> Apro Vento (10m)...")
    try:
        ds_wind = xr.open_dataset(local_path, engine='cfgrib', 
                                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
    except Exception as e:
        print(f"      ERRORE VENTO: {e}")
        sys.exit(1) # Senza vento è inutile

    # B. TEMPERATURA (Livello 2m)
    print("   -> Apro Temperatura (2m)...")
    try:
        ds_temp = xr.open_dataset(local_path, engine='cfgrib', 
                                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
    except:
        print("      Warn: Temp non trovata, userò zeri.")
        ds_temp = None

    # C. PIOGGIA (Superficie)
    print("   -> Apro Pioggia (Surface)...")
    try:
        ds_rain = xr.open_dataset(local_path, engine='cfgrib', 
                                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
    except:
        print("      Warn: Pioggia non trovata, userò zeri.")
        ds_rain = None

    # --- UNIONE E EXPORT ---
    print("4. Generazione JSON...")
    
    steps = range(min(ds_wind.sizes.get('step', 1), 24))
    catalog = []

    for i in steps:
        try:
            # 1. Vento (Master)
            d_w = ds_wind.isel(step=i) if 'step' in ds_wind.dims else ds_wind
            mask = ((d_w.latitude >= LAT_MIN) & (d_w.latitude <= LAT_MAX) & 
                    (d_w.longitude >= LON_MIN) & (d_w.longitude <= LON_MAX))
            cut_w = d_w.where(mask, drop=True)
            
            # Cerca variabili u/v
            keys_w = list(cut_w.data_vars)
            u_key = next((k for k in ['u10','u','10u'] if k in keys_w), None)
            v_key = next((k for k in ['v10','v','10v'] if k in keys_w), None)
            
            if not u_key: continue

            u = np.nan_to_num(cut_w[u_key].values)
            v = np.nan_to_num(cut_w[v_key].values)

            # 2. Temp
            if ds_temp:
                d_t = ds_temp.isel(step=i) if 'step' in ds_temp.dims else ds_temp
                cut_t = d_t.where(mask, drop=True) # Assumiamo stessa griglia
                keys_t = list(cut_t.data_vars)
                t_key = next((k for k in ['t2m','2t','t'] if k in keys_t), None)
                temp = (cut_t[t_key].values - 273.15) if t_key else np.zeros_like(u)
            else:
                temp = np.zeros_like(u)

            # 3. Pioggia
            if ds_rain:
                d_r = ds_rain.isel(step=i) if 'step' in ds_rain.dims else ds_rain
                cut_r = d_r.where(mask, drop=True)
                keys_r = list(cut_r.data_vars)
                r_key = next((k for k in ['tp','tot_prec'] if k in keys_r), None)
                rain = np.nan_to_num(cut_r[r_key].values) if r_key else np.zeros_like(u)
            else:
                rain = np.zeros_like(u)

            # Metadata
            lat = cut_w.latitude.values
            lon = cut_w.longitude.values
            ny, nx = u.shape
            dx = float((lon.max()-lon.min())/(nx-1)) if nx > 1 else 0.02
            dy = float((lat.max()-lat.min())/(ny-1)) if ny > 1 else 0.02
            
            step_hours = int(ds_wind.step.values[i] / 3.6e12) if 'step' in ds_wind.dims else 0
            valid_dt = run_dt + timedelta(hours=step_hours)

            step_data = {
                "meta": {
                    "run": run_dt.strftime("%Y%m%d%H"),
                    "step": step_hours, "nx": nx, "ny": ny,
                    "la1": float(lat.max()), "lo1": float(lon.min()), "dx": dx, "dy": dy
                },
                "wind_u": np.round(u, 1).flatten().tolist(),
                "wind_v": np.round(v, 1).flatten().tolist(),
                "temp": np.round(temp, 1).flatten().tolist(),
                "rain": np.round(rain, 2).flatten().tolist(),
                "press": [0]
            }
            
            out_name = f"step_{i}.json"
            with open(f"{OUTPUT_DIR}/{out_name}", 'w') as jf: json.dump(step_data, jf)
            catalog.append({"file": out_name, "label": valid_dt.strftime("%d/%m %H:00"), "hour": step_hours})
            print(f"   OK Step +{step_hours}h")

        except Exception as e:
            print(f"Errore step {i}: {e}")
            continue

    with open(f"{OUTPUT_DIR}/catalog.json", 'w') as f: json.dump(catalog, f)
    print("Finito.")

if __name__ == "__main__":
    process_data()
    
