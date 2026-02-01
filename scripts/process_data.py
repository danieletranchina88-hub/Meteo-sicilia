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
LAT_MIN, LAT_MAX = 36.0, 39.0
LON_MIN, LON_MAX = 11.5, 16.0

# Quante ore vogliamo processare dal file?
MAX_HOURS = 24

def get_latest_file_info():
    """Scarica la lista e trova il file della run più recente."""
    print(f"Recupero lista file da: {API_LIST_URL}")
    try:
        r = requests.get(API_LIST_URL, timeout=15)
        r.raise_for_status()
        items = r.json()
        print(f"Trovati {len(items)} file nel dataset.")
    except Exception as e:
        print(f"Errore connessione API: {e}")
        return None, None

    # Cerchiamo l'elemento più recente basandoci su 'date' e 'run'
    valid_runs = []
    for item in items:
        # Controllo che l'oggetto abbia i campi necessari
        if isinstance(item, dict) and 'date' in item and 'run' in item and 'filename' in item:
            try:
                # Esempio: date="2026-02-01", run="00:00"
                # Creiamo un oggetto datetime per poterli ordinare cronologicamente
                run_str = f"{item['date']} {item['run']}" 
                dt = datetime.strptime(run_str, "%Y-%m-%d %H:%M")
                valid_runs.append((dt, item))
            except Exception as e:
                print(f"Saltato elemento illeggibile: {item} ({e})")
                continue

    if not valid_runs:
        print("Nessuna run valida trovata nel JSON.")
        return None, None

    # Ordiniamo dal più vecchio al più recente e prendiamo l'ultimo
    valid_runs.sort(key=lambda x: x[0])
    latest_dt, latest_item = valid_runs[-1]
    
    print(f"Run più recente trovata: {latest_dt} (File: {latest_item['filename']})")
    return latest_dt, latest_item['filename']

def process_data():
    # 1. Setup cartelle
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # 2. Trova il file giusto
    run_dt, filename = get_latest_file_info()
    if not filename:
        sys.exit(1)

    # 3. Scarica il file GRIB completo
    grib_path = "dataset_completo.grib2"
    url = f"{API_DOWNLOAD_URL}/{filename}"
    print(f"Scaricamento file unico: {url}")
    
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(grib_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                    f.write(chunk)
        print("Download completato.")
    except Exception as e:
        print(f"Errore durante il download: {e}")
        sys.exit(1)

    # 4. Elaborazione con XArray
    # Nota: ICON mette variabili su livelli diversi. 
    # Apriamo filtrando per 'heightAboveGround' che contiene VENTO (10m) e TEMP (2m).
    # Se mancano Pioggia/Pressione pazienza per ora, l'importante è sbloccare il vento.
    print("Apertura GRIB e slicing...")
    
    try:
        ds = xr.open_dataset(
            grib_path, 
            engine='cfgrib', 
            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround'}}
        )
    except Exception as e:
        print(f"Errore apertura GRIB: {e}")
        sys.exit(1)

    # Verifica dimensioni
    # Se il file contiene più step temporali, 'step' sarà una dimensione
    steps = [0] # Default se c'è un solo step
    if 'step' in ds.dims:
        steps = range(min(ds.dims['step'], MAX_HOURS))
        print(f"Trovati {ds.dims['step']} step temporali. Ne processo {len(steps)}.")
    
    catalog = []

    # Ciclo sugli step temporali (ore)
    for i in steps:
        try:
            # Seleziona lo step corrente
            ds_step = ds.isel(step=i) if 'step' in ds.dims else ds

            # Calcolo ora validità
            # step è in nanosecondi (timedelta64), lo convertiamo in ore
            step_hours = int(ds.step.values[i] / 3600000000000) if 'step' in ds.dims else 0
            valid_dt = run_dt + timedelta(hours=step_hours)
            
            print(f" -> Elaborazione step +{step_hours}h ({valid_dt})")

            # --- RITAGLIO SICILIA ---
            mask = ((ds_step.latitude >= LAT_MIN) & (ds_step.latitude <= LAT_MAX) & 
                    (ds_step.longitude >= LON_MIN) & (ds_step.longitude <= LON_MAX))
            ds_cut = ds_step.where(mask, drop=True)

            # --- ESTRAZIONE VARIABILI ---
            # Vento (u, v)
            # Spesso i nomi sono 'u', 'v', 'u10', 'v10', '10u', '10v'
            u_var = next((x for x in ['u10', 'u', '10u'] if x in ds_cut), None)
            v_var = next((x for x in ['v10', 'v', '10v'] if x in ds_cut), None)
            
            if u_var is None or v_var is None:
                print("    WARN: Variabili vento non trovate in questo step.")
                continue

            u = np.nan_to_num(ds_cut[u_var].values)
            v = np.nan_to_num(ds_cut[v_var].values)
            
            # Temperatura (t2m o 2t)
            t_var = next((x for x in ['t2m', '2t', 't'] if x in ds_cut), None)
            temp = (ds_cut[t_var].values - 273.15) if t_var else np.zeros_like(u)

            # Nota: Pressione e Pioggia sono solitamente 'surface', non 'heightAboveGround'.
            # Per evitare crash complessi multi-messaggio, per ora mettiamo 0.
            # Li aggiungeremo in una versione v2 più avanzata.
            press = np.zeros_like(u)
            rain = np.zeros_like(u)

            # --- EXPORT JSON ---
            lat = ds_cut.latitude.values
            lon = ds_cut.longitude.values
            ny, nx = u.shape
            
            dx = float((lon.max()-lon.min())/(nx-1)) if nx > 1 else 0.02
            dy = float((lat.max()-lat.min())/(ny-1)) if ny > 1 else 0.02

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
                "rain": np.round(rain, 1).flatten().tolist(),
                "press": np.round(press, 0).flatten().tolist()
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
            print(f"Errore nello step {i}: {e}")
            continue

    # Salva catalogo
    if catalog:
        with open(f"{OUTPUT_DIR}/catalog.json", 'w') as f:
            json.dump(catalog, f)
        print(f"Finito! Generati {len(catalog)} file JSON.")
    else:
        print("Nessun dato generato.")
        sys.exit(1)

if __name__ == "__main__":
    process_data()
    
