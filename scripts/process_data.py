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

# Coordinate Sicilia
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
        except: continue

        # Pulizia file temporanei di cfgrib
        for f in os.listdir("."):
            if f.endswith(".idx"): os.remove(f)

        try:
            # Caricamento con riordinamento esplicito delle latitudini (Fix Ribaltamento)
            ds_wind = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}}).sortby('latitude', ascending=False)
            ds_2m = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}}).sortby('latitude', ascending=False)
            ds_surf = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}}).sortby('latitude', ascending=False)
            
            # Pressione: proviamo prima Mean Sea Level, poi Surface
            try: ds_p = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}}).sortby('latitude', ascending=False)
            except: ds_p = ds_surf
        except: continue

        steps = range(ds_wind.sizes.get('step', 1))
        for i in steps:
            try:
                # Maschera geografica
                mask = ((ds_wind.latitude >= LAT_MIN) & (ds_wind.latitude <= LAT_MAX) & (ds_wind.longitude >= LON_MIN) & (ds_wind.longitude <= LON_MAX))
                
                # Taglio e allineamento
                d_w = ds_wind.isel(step=i).where(mask, drop=True)
                d_2 = ds_2m.isel(step=i).where(mask, drop=True)
                d_s = ds_surf.isel(step=i).where(mask, drop=True)
                d_p = ds_p.isel(step=i).where(mask, drop=True)

                u = np.nan_to_num(d_w.u10.values)
                v = np.nan_to_num(d_w.v10.values)
                
                # Fix Nomi Variabili (Safe-Search)
                temp = (d_2.t2m.values - 273.15) if 't2m' in d_2 else (d_2.t.values - 273.15)
                
                # Pioggia: ICON-2I usa spesso 'tp' o 'tot_prec'
                r_key = next((k for k in ['tp', 'tot_prec', 'apcp', 'prate'] if k in d_s), None)
                rain = np.nan_to_num(d_s[r_key].values) if r_key else np.zeros_like(u)

                # Nubi
                c_key = next((k for k in ['tcc', 'clct'] if k in d_s), None)
                clouds = np.nan_to_num(d_s[c_key].values) if c_key else np.zeros_like(u)

                # Dew Point
                d_key = next((k for k in ['d2m', '2d'] if k in d_2), None)
                dew_point = (d_2[d_key].values - 273.15) if d_key else np.zeros_like(u)

                # Pressione (Fix Pressione)
                p_key = next((k for k in ['prmsl', 'msl', 'sp', 'pres'] if k in d_p), None)
                press = d_p[p_key].values if p_key else np.full_like(u, 101325.0)
                if np.max(press) > 2000: press = press / 100.0

                # Meta-dati per l'header
                lat, lon = d_w.latitude.values, d_w.longitude.values
                ny, nx = u.shape
                la1, lo1 = float(lat[0]), float(lon[0])
                dx, dy = float(abs(lon[1] - lon[0])), float(abs(lat[0] - lat[1]))
                lo2, la2 = lo1 + (nx - 1) * dx, la1 - (ny - 1) * dy

                valid_dt = run_dt + timedelta(hours=int(ds_wind.step.values[i] / np.timedelta64(1, 'h')))
                header = {"nx": nx, "ny": ny, "lo1": lo1, "la1": la1, "lo2": lo2, "la2": la2, "dx": dx, "dy": dy, "refTime": valid_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")}
                
                step_data = {
                    "meta": header,
                    "wind_u": {"header":header, "data": np.round(u,1).flatten().tolist()},
                    "wind_v": {"header":header, "data": np.round(v,1).flatten().tolist()},
                    "temp": np.round(temp,1).flatten().tolist(),
                    "dew_point": np.round(dew_point,1).flatten().tolist(),
                    "rain": np.round(rain,2).flatten().tolist(),
                    "clouds": np.round(clouds,1).flatten().tolist(),
                    "press": np.round(press,1).flatten().tolist()
                }
                
                out_name = f"step_{i}.json"
                with open(f"{TEMP_DIR}/{out_name}", 'w') as jf: json.dump(step_data, jf)
                catalog.append({"file": out_name, "label": valid_dt.strftime('%d/%m %H:00'), "hour": i})
            except: continue
        print("OK")

    if catalog:
        with open(f"{TEMP_DIR}/catalog.json", 'w') as f: json.dump(catalog, f)
        if os.path.exists(FINAL_DIR): shutil.rmtree(FINAL_DIR)
        shutil.move(TEMP_DIR, FINAL_DIR)
    if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)

if __name__ == "__main__":
    process_data()
    
