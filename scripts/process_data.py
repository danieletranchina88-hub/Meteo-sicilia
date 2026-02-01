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
LAT_MIN, LAT_MAX = 35.0, 39.5
LON_MIN, LON_MAX = 11.0, 16.5

def get_latest_run_file():
    print("1. Cerco ultima run...")
    try:
        r = requests.get(API_LIST_URL, timeout=15)
        r.raise_for_status()
        items = r.json()
    except Exception as e:
        print(f"Errore API: {e}")
        return None, None

    valid_items = []
    for item in items:
        if isinstance(item, dict) and 'date' in item and 'run' in item:
            try:
                dt = datetime.strptime(f"{item['date']} {item['run']}", "%Y-%m-%d %H:%M")
                valid_items.append((dt, item['filename']))
            except: continue
    
    if not valid_items: return None, None
    valid_items.sort(key=lambda x: x[0])
    return valid_items[-1]

def process_data():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    run_dt, filename = get_latest_run_file()
    if not filename: sys.exit(1)

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

    print("3. Apertura Dataset...")
    try:
        ds_wind = xr.open_dataset(local_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
        ds_temp = xr.open_dataset(local_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
        ds_rain = xr.open_dataset(local_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
    except Exception as e:
        print(f"Errore apertura GRIB: {e}")
        sys.exit(1)

    print("4. Generazione JSON (fino a 72 ore)...")
    
    # --- MODIFICA 72 ORE ---
    # Proviamo a prendere fino a 72 step se disponibili
    max_steps = min(ds_wind.sizes.get('step', 1), 75) 
    steps = range(max_steps)
    
    catalog = []

    for i in steps:
        try:
            d_w = ds_wind.isel(step=i) if 'step' in ds_wind.dims else ds_wind
            d_t = ds_temp.isel(step=i) if 'step' in ds_temp.dims else ds_temp
            d_r = ds_rain.isel(step=i) if 'step' in ds_rain.dims else ds_rain

            # Fix Orientamento
            d_w = d_w.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
            d_t = d_t.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
            d_r = d_r.sortby('latitude', ascending=False).sortby('longitude', ascending=True)

            mask = ((d_w.latitude >= LAT_MIN) & (d_w.latitude <= LAT_MAX) & 
                    (d_w.longitude >= LON_MIN) & (d_w.longitude <= LON_MAX))
            
            cut_w = d_w.where(mask, drop=True)
            cut_t = d_t.where(mask, drop=True)
            cut_r = d_r.where(mask, drop=True)

            u_key = next((k for k in ['u10','u','10u'] if k in cut_w), None)
            v_key = next((k for k in ['v10','v','10v'] if k in cut_w), None)
            u = np.nan_to_num(cut_w[u_key].values)
            v = np.nan_to_num(cut_w[v_key].values)

            t_key = next((k for k in ['t2m','2t','t'] if k in cut_t), None)
            temp = (cut_t[t_key].values - 273.15) if t_key else np.zeros_like(u)

            r_key = next((k for k in ['tp','tot_prec'] if k in cut_r), None)
            rain = np.nan_to_num(cut_r[r_key].values) if r_key else np.zeros_like(u)

            lat = cut_w.latitude.values
            lon = cut_w.longitude.values
            ny, nx = u.shape
            
            dx = float(np.abs(lon[1] - lon
            
