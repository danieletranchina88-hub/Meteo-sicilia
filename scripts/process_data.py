import requests
import xarray as xr
import numpy as np
import json
import os
from datetime import datetime, timedelta

# --- CONFIGURAZIONE ---
DATASET_ID = "ICON_2I_SURFACE_PRESSURE_LEVELS"
BASE_URL = "https://meteohub.agenziaitaliameteo.it/api"
GRIB_FILE = "model_data.grib2"
OUTPUT_JSON = "sicilia_wind.json"

# Bounding Box Sicilia (Lat/Lon)
LAT_MIN, LAT_MAX = 36.0, 39.0
LON_MIN, LON_MAX = 11.5, 16.0

def get_latest_run_url():
    """Cerca l'ultima run disponibile (00 o 12 UTC)"""
    today = datetime.utcnow().date()
    date_str = today.strftime("%Y-%m-%d")
    
    # Prova run 00 di oggi
    url = f"{BASE_URL}/opendata/{DATASET_ID}/download?reftime={date_str}&run=00"
    print(f"Controllo run odierna 00 UTC: {url}")
    
    try:
        r = requests.head(url)
        if r.status_code == 200:
            return url
    except:
        pass

    # Se fallisce, prendi la 12 di ieri
    yesterday_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    print("Run odierna non pronta. Uso run di ieri 12 UTC.")
    return f"{BASE_URL}/opendata/{DATASET_ID}/download?reftime={yesterday_str}&run=12"

def download_and_process():
    url = get_latest_run_url()
    print(f"Scaricando: {url}")
    
    # Download
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(GRIB_FILE, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("Elaborazione GRIB...")
    
    # Apre il GRIB filtrando solo Vento (u, v) a 10 metri
    try:
        ds = xr.open_dataset(
            GRIB_FILE, 
            engine='cfgrib',
            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}}
        )
    except Exception as e:
        print(f"Errore lettura GRIB: {e}")
        return

    # Rinomina variabili standard
    var_map = {}
    if 'u10' in ds: var_map['u10'] = 'u'
    if 'v10' in ds: var_map['v10'] = 'v'
    if var_map: ds = ds.rename(var_map)

    # Ritaglio Sicilia
    mask = (
        (ds.latitude >= LAT_MIN) & (ds.latitude <= LAT_MAX) &
        (ds.longitude >= LON_MIN) & (ds.longitude <= LON_MAX)
    )
    ds_sicily = ds.where(mask, drop=True)

    # Preparazione JSON
    u = np.nan_to_num(ds_sicily['u'].values)
    v = np.nan_to_num(ds_sicily['v'].values)
    lat = ds_sicily.latitude.values
    lon = ds_sicily.longitude.values
    
    ny, nx = u.shape
    dy = (lat.max() - lat.min()) / (ny - 1) if ny > 1 else 0.02
    dx = (lon.max() - lon.min()) / (nx - 1) if nx > 1 else 0.02

    output = [
        {
            "header": {
                "parameterUnit": "m.s-1",
                "parameterNumber": 2,
                "parameterNumberName": "Eastward current",
                "dx": float(dx), "dy": float(dy),
                "la1": float(lat.max()), "lo1": float(lon.min()),
                "nx": nx, "ny": ny,
                "refTime": datetime.now().isoformat()
            },
            "data": u.flatten().tolist()
        },
        {
            "header": {
                "parameterUnit": "m.s-1",
                "parameterNumber": 3,
                "parameterNumberName": "Northward current",
                "dx": float(dx), "dy": float(dy),
                "la1": float(lat.max()), "lo1": float(lon.min()),
                "nx": nx, "ny": ny,
                "refTime": datetime.now().isoformat()
            },
            "data": v.flatten().tolist()
        }
    ]

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f)
    
    print(f"Successo! Creato {OUTPUT_JSON}")

if __name__ == "__main__":
    download_and_process()
  
