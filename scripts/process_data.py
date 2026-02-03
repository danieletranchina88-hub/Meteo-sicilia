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

# Coordinate Box Sicilia
LAT_MIN, LAT_MAX = 35.0, 39.5
LON_MIN, LON_MAX = 11.0, 16.5

def get_latest_run_files():
    print("1. Cerco dati su MeteoHub...", flush=True)
    try:
        r = requests.get(API_LIST_URL, timeout=30)
        r.raise_for_status()
        items = r.json()
    except Exception as e:
        print(f"Errore connessione API: {e}")
        return None, []
    
    runs = {}
    for item in items:
        if isinstance(item, dict) and 'date' in item and 'run' in item:
            key = f"{item['date']} {item['run']}"
            if key not in runs: runs[key] = []
            runs[key].append(item['filename'])
            
    if not runs: return None, []
    
    # Prende l'ultimo run disponibile
    latest_key = sorted(runs.keys())[-1]
    run_dt = datetime.strptime(latest_key, "%Y-%m-%d %H:%M")
    
    # Restituisce i primi 48 file (48 ore di previsione)
    return run_dt, runs[latest_key][:48]

def calculate_rh(temp_k, dew_k):
    """
    Calcola Umidità Relativa (%) da Temp e DewPoint (Kelvin)
    Formula: August-Roche-Magnus
    """
    T = temp_k - 273.15
    Td = dew_k - 273.15
    
    # Coefficienti (per acqua, range -40 a +50 C)
    a = 17.625
    b = 243.04
    
    # Formula vettoriale (numpy optimized)
    # RH = 100 * (EXP(a*Td / (b+Td)) / EXP(a*T / (b+T)))
    numerator = np.exp((a * Td) / (b + Td))
    denominator = np.exp((a * T) / (b + T))
    
    rh = 100 * (numerator / denominator)
    return np.clip(rh, 0, 100) # Limita tra 0 e 100

def process_data():
    run_dt, file_list = get_latest_run_files()
    if not file_list:
        print("Nessun dato trovato.")
        sys.exit(0)
        
    print(f"2. Elaboro Run: {run_dt} ({len(file_list)} files)", flush=True)
    
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    catalog = []

    for idx, filename in enumerate(file_list):
        print(f"   [{idx+1:02d}] DL {filename}...", end=" ", flush=True)
        
        # --- DOWNLOAD ---
        try:
            with requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(TEMP_FILE, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
            print("OK", end=" ", flush=True)
        except:
            print("KO (Download fallito)", flush=True)
            continue

        # Pulizia indici precedenti
        if os.path.exists(f"{TEMP_FILE}.idx"): os.remove(f"{TEMP_FILE}.idx")

        # --- APERTURA DATASET (Modularizzata per evitare conflitti) ---
        
        # 1. Vento (10m)
        try:
            ds_wind = xr.open_dataset(TEMP_FILE, engine='cfgrib', 
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
        except:
            print("Skip (No Wind)")
            continue

        # 2. Termodinamica (2m) -> Serve per Temp e RH (Cerca Temp e DewPoint)
        ds_thermo = None
        try:
            ds_thermo = xr.open_dataset(TEMP_FILE, engine='cfgrib', 
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
        except: pass

        # 3. Pressione (Mean Sea Level - Prioritaria)
        ds_press = None
        press_src = "None"
        try:
            # Cerca pressione ridotta al livello del mare (MSL)
            ds_press = xr.open_dataset(TEMP_FILE, engine='cfgrib', 
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}})
            press_src = "MSL"
        except:
            # Fallback: Pressione Superficiale (meno precisa per le isobare ma ok)
            try:
                ds_press = xr.open_dataset(TEMP_FILE, engine='cfgrib', 
                    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
                press_src = "Surface"
            except: pass

        # 4. Pioggia
        ds_rain = None
        try:
            ds_rain = xr.open_dataset(TEMP_FILE, engine='cfgrib', 
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'stepType': 'accum'}})
        except: pass


        # --- ELABORAZIONE STEP ---
        steps = range(ds_wind.sizes.get('step', 1))
        
        for i in steps:
            try:
                # Gestione asse temporale singolo o multiplo
                if ds_wind.sizes.get('step', 1) > 1:
                    d_w_raw = ds_wind.isel(step=i)
                    raw_step = ds_wind.step.values[i]
                else:
                    d_w_raw = ds_wind
                    raw_step = ds_wind.step.values

                step_hours = int(raw_step / np.timedelta64(1, 'h')) if isinstance(raw_step, np.timedelta64) else int(raw_step)
                
                # Ritaglio Area (Sicilia)
                d_w = d_w_raw.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
                mask = ((d_w.latitude >= LAT_MIN) & (d_w.latitude <= LAT_MAX) & (d_w.longitude >= LON_MIN) & (d_w.longitude <= LON_MAX))
                cut_w = d_w.where(mask, drop=True)
                
                # Estrazione Vento (U, V)
                u_key = next((k for k in ['u10','u'] if k in cut_w), None)
                v_key = next((k for k in ['v10','v'] if k in cut_w), None)
                
                if not u_key or not v_key: continue

                u = np.nan_to_num(cut_w[u_key].values)
                v = np.nan_to_num(cut_w[v_key].values)

                # Info Griglia
                lat = cut_w.latitude.values
                lon = cut_w.longitude.values
                ny, nx = u.shape
                la1, lo1 = float(lat[0]), float(lon[0])
                dx = float(abs(lon[1] - lon[0]))
                dy = float(abs(lat[0] - lat[1]))
                lo2 = lo1 + (nx - 1) * dx
                la2 = la1 - (ny - 1) * dy

                # --- TEMP & RH ---
                temp_c = np.zeros_like(u)
                rh_val = np.zeros_like(u)
                
                if ds_thermo:
                    d_t = ds_thermo.where(mask, drop=True)
                    if ds_thermo.sizes.get('step', 1) > 1: d_t = d_t.isel(step=i)
                    d_t = d_t.sortby('latitude', ascending=False).sortby('longitude', ascending=True)

                    t_key = next((k for k in ['t2m','t'] if k in d_t), None)
                    d_key = next((k for k in ['d2m','2d','d'] if k in d_t), None) # DewPoint

                    if t_key:
                        tk = d_t[t_key].values
                        temp_c = tk - 273.15
                        
                        # Calcolo RH se abbiamo Dew Point
                        if d_key:
                            dk = d_t[d_key].values
                            rh_val = calculate_rh(tk, dk)

                # --- PIOGGIA ---
                rain = np.zeros_like(u)
                if ds_rain:
                    d_r = ds_rain.where(mask, drop=True)
                    if ds_rain.sizes.get('step', 1) > 1: d_r = d_r.isel(step=i)
                    d_r = d_r.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
                    
                    r_key = next((k for k in ['tp', 'tot_prec', 'apcp'] if k in d_r), None)
                    if r_key: rain = np.nan_to_num(d_r[r_key].values)

                # --- PRESSIONE (Logica corretta) ---
                press = np.zeros_like(u)
                if ds_press:
                    d_p = ds_press.where(mask, drop=True)
                    if ds_press.sizes.get('step', 1) > 1: d_p = d_p.isel(step=i)
                    d_p = d_p.sortby('latitude', ascending=False).sortby('longitude', ascending=True)

                    # Chiavi comuni: prmsl (MSL), sp (Surface), pres
                    p_key = next((k for k in ['prmsl', 'msl', 'sp', 'pres'] if k in d_p), None)
                    
                    if p_key:
                        raw_p = np.nan_to_num(d_p[p_key].values)
                        max_p = np.max(raw_p)

                        # Conversione Pascal -> hPa
                        # Se il valore massimo è > 80000 (es. 101325), è in Pascal
                        if max_p > 80000:
                            press = raw_p / 100.0
                        else:
                            press = raw_p
                        
                        # Fallback sicurezza: se tutto 0 o troppo basso, setta a 1013 (ma ora non dovrebbe servire)
                        if np.max(press) < 800:
                             press.fill(1013.0)
                    else:
                        press.fill(1013.0)
                else:
                    press.fill(1013.0)

                # --- EXPORT ---
                valid_dt = run_dt + timedelta(hours=step_hours)
                iso_date = valid_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                
                header = {"nx": nx, "ny": ny, "lo1": lo1, "la1": la1, "lo2": lo2, "la2": la2, "dx": dx, "dy": dy, "refTime": iso_date}
                
                # Creazione oggetto JSON
                step_data = {
                    "meta": header,
                    "wind_u": {"header":{**header,"parameterCategory":2,"parameterNumber":2}, "data": np.round(u,1).flatten().tolist()},
                    "wind_v": {"header":{**header,"parameterCategory":2,"parameterNumber":3}, "data": np.round(v,1).flatten().tolist()},
                    "temp": np.round(temp_c,1).flatten().tolist(),
                    "rain": np.round(rain,2).flatten().tolist(),
                    "press": np.round(press,1).flatten().tolist(), # Ora in hPa reali
                    "rh": np.round(rh_val,0).flatten().tolist()    # Nuova variabile RH
                }
                
                out_name = f"step_{step_hours}.json"
                with open(f"{TEMP_DIR}/{out_name}", 'w') as jf: json.dump(step_data, jf)
                
                if not any(x['hour'] == step_hours for x in catalog):
                    catalog.append({"file": out_name, "label": f"{valid_dt.strftime('%d/%m %H:00')}", "hour": step_hours})

            except Exception as e_step:
                print(f"X", end="", flush=True)
                continue
        
        print(" -> Done")

    # Finalizzazione
    if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
    if os.path.exists(f"{TEMP_FILE}.idx"): os.remove(f"{TEMP_FILE}.idx")

    if catalog:
        catalog.sort(key=lambda x: x['hour'])
        with open(f"{TEMP_DIR}/catalog.json", 'w') as f: json.dump(catalog, f)
        if os.path.exists(FINAL_DIR): shutil.rmtree(FINAL_DIR)
        shutil.move(TEMP_DIR, FINAL_DIR)
        print("COMPLETATO.")
    else:
        sys.exit(1)

if __name__ == "__main__":
    process_data()
