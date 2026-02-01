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

def get_latest_run_files():
    print("1. Cerco file ultima run...")
    try:
        r = requests.get(API_LIST_URL, timeout=15)
        r.raise_for_status()
        items = r.json()
    except Exception as e:
        print(f"Errore API: {e}")
        return None, None

    # Raggruppa i file per Data/Ora Run
    runs = {}
    for item in items:
        if isinstance(item, dict) and 'date' in item and 'run' in item:
            key = f"{item['date']} {item['run']}"
            if key not in runs: runs[key] = []
            runs[key].append(item['filename'])

    if not runs: return None, None
    
    # Prendi la run più recente
    latest_key = sorted(runs.keys())[-1]
    run_dt = datetime.strptime(latest_key, "%Y-%m-%d %H:%M")
    print(f"   Run trovata: {latest_key} ({len(runs[latest_key])} file)")
    
    return run_dt, runs[latest_key]

def check_file_content(filename):
    """Scarica un file e controlla quali variabili contiene."""
    local_path = "temp_check.grib2"
    url = f"{API_DOWNLOAD_URL}/{filename}"
    
    try:
        # Download rapido
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
        
        # Tentativo 1: Vento/Temp (heightAboveGround)
        try:
            ds = xr.open_dataset(local_path, engine='cfgrib', 
                               backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround'}})
            vars = list(ds.data_vars)
            ds.close()
            return vars, "heightAboveGround"
        except:
            pass

        # Tentativo 2: Pioggia/Pressione (surface)
        try:
            ds = xr.open_dataset(local_path, engine='cfgrib', 
                               backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
            vars = list(ds.data_vars)
            ds.close()
            return vars, "surface"
        except:
            pass
            
        return [], None
    except Exception as e:
        print(f"Errore check {filename}: {e}")
        return [], None

def process_data():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    run_dt, file_list = get_latest_run_files()
    if not file_list: sys.exit(1)

    # --- FASE 1: Identifica quale file ha cosa ---
    file_map = {'wind': None, 'temp': None, 'rain': None}
    
    print("2. Scansione contenuto file...")
    for fname in file_list:
        if all(file_map.values()): break # Abbiamo trovato tutto
        
        vars, level_type = check_file_content(fname)
        print(f"   -> {fname[:20]}... contiene: {vars}")
        
        # Cerca Vento
        if not file_map['wind'] and any(k in vars for k in ['u10','u','10u','v10','v','10v']):
            file_map['wind'] = (fname, level_type)
            print("      [TROVATO] Vento")
        
        # Cerca Temp (spesso è t2m o 2t)
        if not file_map['temp'] and any(k in vars for k in ['t2m','2t','t']):
            file_map['temp'] = (fname, level_type)
            print("      [TROVATO] Temperatura")
            
        # Cerca Pioggia (tp o tot_prec)
        if not file_map['rain'] and any(k in vars for k in ['tp','tot_prec','unknown']): 
            file_map['rain'] = (fname, level_type)
            print("      [TROVATO] Pioggia")

    if not file_map['wind']:
        print("ERRORE FATALE: Non ho trovato nessun file con il vento (u/v).")
        # Fallback: se troviamo almeno la temp, usiamo quella come vento fake (solo per non crashare)
        # Ma l'utente vuole il vento, quindi è meglio uscire.
        sys.exit(1)

    # --- FASE 2: Estrazione e Unione ---
    print("3. Elaborazione e Unione dati...")
    
    # Carichiamo i dataset necessari
    ds_wind = xr.open_dataset(f"{API_DOWNLOAD_URL}/{file_map['wind'][0]}", engine='cfgrib', cache=False,
                              backend_kwargs={'filter_by_keys': {'typeOfLevel': file_map['wind'][1]}}) if file_map['wind'] else None
    
    ds_temp = xr.open_dataset(f"{API_DOWNLOAD_URL}/{file_map['temp'][0]}", engine='cfgrib', cache=False,
                              backend_kwargs={'filter_by_keys': {'typeOfLevel': file_map['temp'][1]}}) if file_map['temp'] else None
    
    ds_rain = xr.open_dataset(f"{API_DOWNLOAD_URL}/{file_map['rain'][0]}", engine='cfgrib', cache=False,
                              backend_kwargs={'filter_by_keys': {'typeOfLevel': file_map['rain'][1]}}) if file_map['rain'] else None

    # Usiamo il vento come riferimento temporale
    steps = range(min(ds_wind.sizes.get('step', 1), 24))
    catalog = []

    for i in steps:
        try:
            # --- VENTO ---
            d_w = ds_wind.isel(step=i) if 'step' in ds_wind.dims else ds_wind
            mask = ((d_w.latitude >= LAT_MIN) & (d_w.latitude <= LAT_MAX) & 
                    (d_w.longitude >= LON_MIN) & (d_w.longitude <= LON_MAX))
            cut_w = d_w.where(mask, drop=True)
            
            u_name = next(k for k in ['u10','u','10u'] if k in cut_w)
            v_name = next(k for k in ['v10','v','10v'] if k in cut_w)
            u = np.nan_to_num(cut_w[u_name].values)
            v = np.nan_to_num(cut_w[v_name].values)

            # --- TEMP ---
            if ds_temp:
                d_t = ds_temp.isel(step=i) if 'step' in ds_temp.dims else ds_temp
                # Assumiamo stessa griglia per semplicità (ritaglio identico)
                cut_t = d_t.where(mask, drop=True)
                t_name = next((k for k in ['t2m','2t','t'] if k in cut_t), None)
                temp = (cut_t[t_name].values - 273.15) if t_name else np.zeros_like(u)
            else:
                temp = np.zeros_like(u)

            # --- PIOGGIA ---
            if ds_rain:
                d_r = ds_rain.isel(step=i) if 'step' in ds_rain.dims else ds_rain
                cut_r = d_r.where(mask, drop=True)
                r_name = next((k for k in ['tp','tot_prec'] if k in cut_r), None)
                rain = np.nan_to_num(cut_r[r_name].values) if r_name else np.zeros_like(u)
            else:
                rain = np.zeros_like(u)

            # --- JSON ---
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
            print(f"   Generato step +{step_hours}h")

        except Exception as e:
            print(f"Errore generazione step {i}: {e}")
            continue

    with open(f"{OUTPUT_DIR}/catalog.json", 'w') as f: json.dump(catalog, f)
    print("Finito.")

if __name__ == "__main__":
    process_data()
    
