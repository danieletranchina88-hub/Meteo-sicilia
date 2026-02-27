import requests
import xarray as xr
import numpy as np
import json
import os
import shutil
from datetime import datetime, timedelta

# --- CONFIGURAZIONE ---
DATASET_ID = "ICON_2I_SURFACE_PRESSURE_LEVELS"
API_LIST_URL = f"https://meteohub.agenziaitaliameteo.it/api/datasets/{DATASET_ID}/opendata"
API_DOWNLOAD_URL = "https://meteohub.agenziaitaliameteo.it/api/opendata"
FINAL_DIR = "data_weather"
TEMP_DIR = "temp_processing"
TEMP_FILE = "temp.grib2"

LAT_MIN, LAT_MAX = 35.0, 48.5
LON_MIN, LON_MAX = 6.0, 19.5
MAX_PIXELS = 600_000 
LAPSE_RATE = 0.0065 # 0.65°C ogni 100m
PATH_DEM = "topografia_italia_1km.nc" # Opzionale

def safe_extract(ds, names, mask):
    for n in names:
        if n in ds:
            return ds[n].where(mask, drop=True).values
    return None

def process_data():
    print("--- RICERCA RUN ---")
    try:
        r = requests.get(API_LIST_URL, timeout=30)
        items = r.json()
        latest_key = sorted([f"{i['date']} {i['run']}" for i in items if 'date' in i])[-1]
        run_dt = datetime.strptime(latest_key, "%Y-%m-%d %H:%M")
        file_list = sorted([i['filename'] for i in items if f"{i['date']} {i['run']}" == latest_key])[:48]
    except Exception as e:
        print(f"Errore API: {e}")
        return

    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    # Caricamento DEM per 3D
    dem = None
    if os.path.exists(PATH_DEM):
        dem = xr.open_dataset(PATH_DEM)

    catalog = []

    for idx, filename in enumerate(file_list):
        print(f"[{idx+1}] Scarico {filename}...")
        try:
            with requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True) as r:
                with open(TEMP_FILE, 'wb') as f: shutil.copyfileobj(r.raw, f)
            
            # 1. Apertura selettiva (evita file vuoti)
            ds_w = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
            ds_t = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
            ds_p = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}})
            
            # Maschera Italia
            mask = (ds_t.latitude >= LAT_MIN) & (ds_t.latitude <= LAT_MAX) & (ds_t.longitude >= LON_MIN) & (ds_t.longitude <= LON_MAX)
            cut_t = ds_t.sortby('latitude', ascending=False).where(mask, drop=True)
            
            # Estrazione dati
            temp = cut_t.t2m.values - 273.15
            u = ds_w.u10.where(mask, drop=True).values
            v = ds_w.v10.where(mask, drop=True).values
            press = ds_p.pmsl.where(mask, drop=True).values / 100.0
            
            # Correzione 3D (Lapse Rate)
            # 
            feels_like = temp.copy()
            if dem is not None:
                try:
                    ds_h = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys':{'shortName':'gh','typeOfLevel':'surface'}})
                    h_mod = ds_h.gh.where(mask, drop=True).values
                    h_real = dem.interp(latitude=cut_t.latitude, longitude=cut_t.longitude).elevation.values
                    feels_like = temp + (h_mod - h_real) * LAPSE_RATE
                except: pass

            # Header Meta
            header = {
                "nx": int(temp.shape[1]), "ny": int(temp.shape[0]),
                "lo1": float(cut_t.longitude[0]), "la1": float(cut_t.latitude[0]),
                "dx": float(abs(cut_t.longitude[1] - cut_t.longitude[0])),
                "dy": float(abs(cut_t.latitude[0] - cut_t.latitude[1]))
            }

            step_h = int(ds_t.step.values / np.timedelta64(1, 'h'))
            out_name = f"step_{step_h}.json"
            
            step_data = {
                "meta": header,
                "temp": np.round(temp, 1).flatten().tolist(),
                "feels_like": np.round(feels_like, 1).flatten().tolist(),
                "wind_u": {"data": np.round(u, 1).flatten().tolist()},
                "wind_v": {"data": np.round(v, 1).flatten().tolist()},
                "press": np.round(press, 1).flatten().tolist()
            }

            with open(f"{TEMP_DIR}/{out_name}", 'w') as jf:
                json.dump(step_data, jf, separators=(',', ':'))

            catalog.append({"file": out_name, "label": (run_dt + timedelta(hours=step_h)).strftime('%d/%m %H:00'), "hour": step_h})
            print(f"   Successo -> {out_name}")

        except Exception as e:
            print(f"   Salto file: {e}")
            continue

    if catalog:
        with open(f"{TEMP_DIR}/catalog.json", 'w') as f:
            json.dump(sorted(catalog, key=lambda x:x['hour']), f, separators=(',', ':'))
        if os.path.exists(FINAL_DIR): shutil.rmtree(FINAL_DIR)
        shutil.move(TEMP_DIR, FINAL_DIR)

if __name__ == "__main__":
    process_data()
