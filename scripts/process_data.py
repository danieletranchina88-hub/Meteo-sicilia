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
BASE_URL = "https://meteohub.agenziaitaliameteo.it/api"
OUTPUT_DIR = "data_weather"  # Cartella dove salveremo i JSON spezzettati

# Limiti Sicilia
LAT_MIN, LAT_MAX = 36.0, 39.0
LON_MIN, LON_MAX = 11.5, 16.0

# Quante ore di previsione scaricare? (Mettiamo 6-12 per non intasare GitHub all'inizio)
FORECAST_HOURS = 12 

def get_run_info():
    """Trova la run più recente (00 o 12) e la data."""
    # Cerchiamo oggi e ieri
    for days_back in [0, 1]:
        date_check = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        for run_time in ["12:00", "00:00"]:
            print(f"Check run: {date_check} {run_time}...")
            # Verifica se esistono file per questa run
            try:
                params = {"reftime": date_check, "run": run_time}
                r = requests.get(f"{BASE_URL}/opendata/{DATASET_ID}/download", params=params, timeout=10)
                if r.status_code == 200 and len(r.json()) > 5:
                    return date_check, run_time, r.json()
            except:
                continue
    return None, None, None

def process_all():
    # 1. Pulisci o crea cartella dati
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # 2. Ottieni lista file
    date_str, run_time, file_list = get_run_info()
    if not file_list:
        print("Nessuna run trovata.")
        sys.exit(1)

    print(f"Elaborazione Run: {date_str} {run_time}")
    file_list.sort() # Ordina per orario previsione

    catalog = [] # Lista per lo slider temporale

    # 3. Ciclo sulle prime N ore
    count = 0
    for filename in file_list:
        if count >= FORECAST_HOURS: break
        
        # Scarica file temporaneo
        print(f"Scaricamento {filename}...")
        url = f"{BASE_URL}/opendata/{filename}"
        with requests.get(url, stream=True) as r:
            with open("temp.grib2", 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        
        try:
            # Apri GRIB
            # Filtriamo i messaggi per evitare conflitti, poi uniamo
            # Nota: ICON mette tutto in messages diversi. Carichiamo tutto e filtriamo con xarray
            ds = xr.open_dataset("temp.grib2", engine='cfgrib', 
                                 backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround'}})
            
            # --- RITAGLIO SICILIA ---
            mask = ((ds.latitude >= LAT_MIN) & (ds.latitude <= LAT_MAX) & 
                    (ds.longitude >= LON_MIN) & (ds.longitude <= LON_MAX))
            ds_cut = ds.where(mask, drop=True)

            # --- ESTRAZIONE VARIABILI ---
            # Mappatura nomi variabili (dipende dal GRIB, adattiamo dinamicamente)
            data_step = {}
            
            # 1. VENTO (u, v a 10m)
            u = np.nan_to_num(ds_cut['u10'].values if 'u10' in ds_cut else ds_cut['u'].values)
            v = np.nan_to_num(ds_cut['v10'].values if 'v10' in ds_cut else ds_cut['v'].values)
            
            # 2. TEMPERATURA (t2m) -> Convertiamo Kelvin a Celsius
            if 't2m' in ds_cut:
                temp_c = ds_cut['t2m'].values - 273.15
            else:
                temp_c = np.zeros_like(u)

            # 3. PRECIPITAZIONE (tp - Total Precipitation)
            # Nota: Spesso è accumulata. Per ora prendiamo il valore grezzo.
            precip = ds_cut['tp'].values if 'tp' in ds_cut else np.zeros_like(u)

            # 4. PRESSIONE (msl - Mean Sea Level) -> Pascal a hPa
            press = (ds_cut['msl'].values / 100.0) if 'msl' in ds_cut else np.zeros_like(u)

            # --- SALVATAGGIO JSON COMPATTO ---
            # Salviamo un JSON unico per questo step temporale
            # Usiamo una griglia semplificata per il frontend
            
            lat = ds_cut.latitude.values
            lon = ds_cut.longitude.values
            ny, nx = u.shape
            
            # Dati meteo per il "Picker" e per le mappe
            # Per risparmiare spazio, arrotondiamo i float
            step_data = {
                "meta": {
                    "date": date_str,
                    "run": run_time,
                    "forecast_hour": count,
                    "nx": nx, "ny": ny,
                    "la1": float(lat.max()), "lo1": float(lon.min()),
                    "dx": float((lon.max()-lon.min())/(nx-1)),
                    "dy": float((lat.max()-lat.min())/(ny-1))
                },
                "wind_u": np.round(u, 1).flatten().tolist(),
                "wind_v": np.round(v, 1).flatten().tolist(),
                "temp": np.round(temp_c, 1).flatten().tolist(),
                "rain": np.round(precip, 2).flatten().tolist(),
                "press": np.round(press, 0).flatten().tolist()
            }
            
            # Nome file basato sull'ora: 0.json, 1.json...
            out_file = f"{OUTPUT_DIR}/step_{count}.json"
            with open(out_file, 'w') as jf:
                json.dump(step_data, jf)
            
            # Aggiungi al catalogo
            # Calcola l'orario reale di validità
            valid_time = datetime.strptime(f"{date_str} {run_time}", "%Y-%m-%d %H:%M") + timedelta(hours=count)
            catalog.append({
                "file": f"step_{count}.json",
                "label": valid_time.strftime("%d/%m %H:00"),
                "hour": count
            })

            count += 1
            ds.close()

        except Exception as e:
            print(f"Errore processamento file {filename}: {e}")
            continue

    # Salva il catalogo (il menu dello slider)
    with open(f"{OUTPUT_DIR}/catalog.json", 'w') as f:
        json.dump(catalog, f)
    
    print("Elaborazione completata.")

if __name__ == "__main__":
    process_all()
    
