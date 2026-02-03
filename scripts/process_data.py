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

LAT_MIN, LAT_MAX = 35.0, 39.5
LON_MIN, LON_MAX = 11.0, 16.5

def get_latest_run_files():
    print("1. Cerco dati...", flush=True)
    try:
        r = requests.get(API_LIST_URL, timeout=30)
        r.raise_for_status()
        items = r.json()
    except: return None, []
    runs = {}
    for item in items:
        if isinstance(item, dict) and 'date' in item and 'run' in item:
            key = f"{item['date']} {item['run']}"
            if key not in runs: runs[key] = []
            runs[key].append(item['filename'])
    if not runs: return None, []
    latest_key = sorted(runs.keys())[-1]
    run_dt = datetime.strptime(latest_key, "%Y-%m-%d %H:%M")
    return run_dt, runs[latest_key][:48]

def process_data():
    run_dt, file_list = get_latest_run_files()
    if not file_list: sys.exit(0)
    print(f"2. Elaboro {len(file_list)} files...", flush=True)
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    catalog = []

    for idx, filename in enumerate(file_list):
        print(f"   DL {filename}...", end=" ", flush=True)
        try:
            with requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(TEMP_FILE, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
            print("OK", end=" ", flush=True)
        except:
            print("KO", flush=True)
            continue

        # Pulizia file indice cfgrib
        for f in os.listdir("."):
            if f.endswith(".idx"): os.remove(f)

        try:
            # Vento (10m)
            ds_wind = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
            # Temp e Dew Point (2m)
            try: ds_2m = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
            except: ds_2m = None
            # Pioggia e Nuvolosità (Surface)
            try: ds_surf = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
            except: ds_surf = None
            # Pressione (Mean Sea Level)
            try: ds_msl = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}})
            except: ds_msl = None
        except Exception as e:
            print(f"Errore: {e}")
            continue

        steps = range(ds_wind.sizes.get('step', 1))
        for i in steps:
            try:
                raw_step = ds_wind.step.values[i]
                step_hours = int(raw_step / np.timedelta64(1, 'h')) if isinstance(raw_step, np.timedelta64) else int(raw_step)
                
                d_w = ds_wind.isel(step=i).sortby('latitude', ascending=False).sortby('longitude', ascending=True)
                mask = ((d_w.latitude >= LAT_MIN) & (d_w.latitude <= LAT_MAX) & (d_w.longitude >= LON_MIN) & (d_w.longitude <= LON_MAX))
                cut_w = d_w.where(mask, drop=True)
                
                u = np.nan_to_num(cut_w.u10.values)
                v = np.nan_to_num(cut_w.v10.values)
                
                lat, lon = cut_w.latitude.values, cut_w.longitude.values
                ny, nx = u.shape
                la1, lo1 = float(lat[0]), float(lon[0])
                dx, dy = float(abs(lon[1] - lon[0])), float(abs(lat[0] - lat[1]))
                lo2, la2 = lo1 + (nx - 1) * dx, la1 - (ny - 1) * dy

                # Inizializzazione dati aggiuntivi
                temp = np.zeros_like(u)
                dew_point = np.zeros_like(u)
                rain = np.zeros_like(u)
                clouds = np.zeros_like(u)
                press = np.full_like(u, 1013.2) # Default pressione standard

                if ds_2m:
                    d_2 = ds_2m.isel(step=i).where(mask, drop=True)
                    if 't2m' in d_2: temp = d_2.t2m.values - 273.15
                    if 'd2m' in d_2: dew_point = d_2.d2m.values - 273.15

                if ds_surf:
                    d_s = ds_surf.isel(step=i).where(mask, drop=True)
                    r_key = next((k for k in ['tp', 'tot_prec'] if k in d_s), None)
                    if r_key: rain = np.nan_to_num(d_s[r_key].values)
                    c_key = next((k for k in ['tcc', 'clct'] if k in d_s), None)
                    if c_key: clouds = np.nan_to_num(d_s[c_key].values)

                # Logica Pressione
                p_src = ds_msl if ds_msl else ds_surf
                if p_src:
                    d_p = p_src.isel(step=i).where(mask, drop=True)
                    p_key = next((k for k in ['prmsl', 'msl', 'sp', 'pres'] if k in d_p), None)
                    if p_key:
                        p_val = d_p[p_key].values
                        press = p_val / 100.0 if np.max(p_val) > 2000 else p_val

                valid_dt = run_dt + timedelta(hours=step_hours)
                iso_date = valid_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                
                header = {"nx": nx, "ny": ny, "lo1": lo1, "la1": la1, "lo2": lo2, "la2": la2, "dx": dx, "dy": dy, "refTime": iso_date}
                step_data = {
                    "meta": header,
                    "wind_u": {"header":{**header,"parameterCategory":2,"parameterNumber":2}, "data": np.round(u,1).flatten().tolist()},
                    "wind_v": {"header":{**header,"parameterCategory":2,"parameterNumber":3}, "data": np.round(v,1).flatten().tolist()},
                    "temp": np.round(temp,1).flatten().tolist(),
                    "dew_point": np.round(dew_point,1).flatten().tolist(),
                    "rain": np.round(rain,2).flatten().tolist(),
                    "clouds": np.round(clouds,1).flatten().tolist(),
                    "press": np.round(press,1).flatten().tolist()
                }
                out_name = f"step_{step_hours}.json"
                with open(f"{TEMP_DIR}/{out_name}", 'w') as jf: json.dump(step_data, jf)
                if not any(x['hour'] == step_hours for x in catalog): 
                    catalog.append({"file": out_name, "label": f"{valid_dt.strftime('%d/%m %H:00')}", "hour": step_hours})
            except: continue
        print(" -> OK")

    if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
    if catalog:
        catalog.sort(key=lambda x: x['hour'])
        with open(f"{TEMP_DIR}/catalog.json", 'w') as f: json.dump(catalog, f)
        if os.path.exists(FINAL_DIR): shutil.rmtree(FINAL_DIR)
        shutil.move(TEMP_DIR, FINAL_DIR)
        print("COMPLETATO.")
    else: sys.exit(1)

if __name__ == "__main__":
    process_data()
                
