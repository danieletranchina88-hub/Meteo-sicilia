import requests
import xarray as xr
import numpy as np
import json
import os
import sys
import shutil
from datetime import datetime, timedelta

# CONFIGURAZIONE
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

        # --- APERTURA DATASET ---
        try:
            # VENTO
            ds_wind = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
            
            # TEMP
            try: ds_temp = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
            except: ds_temp = None
            
            # PIOGGIA
            try: ds_rain = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'stepType': 'accum'}})
            except:
                try: ds_rain = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
                except: ds_rain = None

        except Exception as e: 
            print(f"Skip file: {e}")
            continue

        steps = range(ds_wind.sizes.get('step', 1))
        
        for i in steps:
            try:
                raw_step = ds_wind.step.values[i]
                if isinstance(raw_step, np.timedelta64): step_hours = int(raw_step / np.timedelta64(1, 'h'))
                else: step_hours = int(raw_step)

                d_w = ds_wind.isel(step=i) if 'step' in ds_wind.dims else ds_wind
                d_w = d_w.sortby('latitude', ascending=False).sortby('longitude', ascending=True)

                mask = ((d_w.latitude >= LAT_MIN) & (d_w.latitude <= LAT_MAX) & (d_w.longitude >= LON_MIN) & (d_w.longitude <= LON_MAX))
                cut_w = d_w.where(mask, drop=True)

                u_key = next((k for k in ['u10','u'] if k in cut_w), None)
                v_key = next((k for k in ['v10','v'] if k in cut_w), None)
                
                if u_key and v_key:
                    u = np.nan_to_num(cut_w[u_key].values)
                    v = np.nan_to_num(cut_w[v_key].values)
                    
                    # Geometria
                    lat = cut_w.latitude.values
                    lon = cut_w.longitude.values
                    ny, nx = u.shape
                    la1, lo1 = float(lat[0]), float(lon[0])
                    dx = float(abs(lon[1] - lon[0]))
                    dy = float(abs(lat[0] - lat[1]))
                    lo2 = lo1 + (nx - 1) * dx
                    la2 = la1 - (ny - 1) * dy

                    # TEMP
                    temp = np.zeros_like(u)
                    if ds_temp:
                        d_t = ds_temp.isel(step=i) if 'step' in ds_temp.dims else ds_temp
                        d_t = d_t.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
                        cut_t = d_t.where(mask, drop=True)
                        t_key = next((k for k in ['t2m','t'] if k in cut_t), None)
                        if t_key: temp = cut_t[t_key].values - 273.15
                    
                    # RAIN
                    rain = np.zeros_like(u)
                    if ds_rain:
                        d_r = ds_rain.isel(step=i) if 'step' in ds_rain.dims else ds_rain
                        d_r = d_r.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
                        cut_r = d_r.where(mask, drop=True)
                        r_key = next((k for k in ['tp', 'tot_prec', 'apcp'] if k in cut_r), None)
                        if r_key: rain = np.nan_to_num(cut_r[r_key].values)

                    # PRESSIONE - LOGICA INTELLIGENTE
                    press = np.zeros_like(u)
                    press_source = "None"

                    # 1. Prova Mean Sea Level
                    try:
                        ds_p1 = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}})
                        d_p = ds_p1.isel(step=i) if 'step' in ds_p1.dims else ds_p1
                        d_p = d_p.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
                        cut_p = d_p.where(mask, drop=True)
                        p_key = next((k for k in ['prmsl', 'msl'] if k in cut_p), None)
                        if p_key:
                            val = np.nan_to_num(cut_p[p_key].values)
                            if np.max(val) > 100: # Se il valore è valido (>100)
                                press = val
                                press_source = "MeanSea"
                    except: pass

                    # 2. Se MeanSea ha fallito (è ancora 0), prova Surface Pressure
                    if np.max(press) < 100:
                        try:
                            ds_p2 = xr.open_dataset(TEMP_FILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'shortName': 'sp'}})
                            d_p = ds_p2.isel(step=i) if 'step' in ds_p2.dims else ds_p2
                            d_p = d_p.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
                            cut_p = d_p.where(mask, drop=True)
                            val = np.nan_to_num(cut_p['sp'].values)
                            press = val
                            press_source = "Surface"
                        except: pass
                    
                    # Conversione Pa -> hPa
                    if np.max(press) > 2000: press = press / 100.0

                    # LOG DEBUG
                    if i == 0:
                        print(f" [DEBUG: Press Source={press_source}, MaxVal={np.max(press):.1f}]", end="")

                    valid_dt = run_dt + timedelta(hours=step_hours)
                    iso_date = valid_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

                    header_common = {
                        "nx": nx, "ny": ny, "lo1": lo1, "la1": la1, "lo2": lo2, "la2": la2,
                        "dx": dx, "dy": dy, "scanMode": 0, "refTime": iso_date
                    }
                    
                    step_data = {
                        "meta": header_common,
                        "wind_u": { "header": {**header_common, "parameterCategory": 2, "parameterNumber": 2}, "data": np.round(u, 1).flatten().tolist() },
                        "wind_v": { "header": {**header_common, "parameterCategory": 2, "parameterNumber": 3}, "data": np.round(v, 1).flatten().tolist() },
                        "temp": np.round(temp, 1).flatten().tolist(),
                        "rain": np.round(rain, 2).flatten().tolist(),
                        "press": np.round(press, 1).flatten().tolist()
                    }

                    out_name = f"step_{step_hours}.json"
                    with open(f"{TEMP_DIR}/{out_name}", 'w') as jf: json.dump(step_data, jf)
                    
                    if not any(x['hour'] == step_hours for x in catalog):
                        catalog.append({"file": out_name, "label": f"{valid_dt.strftime('%d/%m %H:00')}", "hour": step_hours})
            except Exception as e: continue
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
    
