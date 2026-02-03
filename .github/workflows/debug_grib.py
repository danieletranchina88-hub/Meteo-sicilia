import requests
import xarray as xr
import os
import sys
import cfgrib

# --- CONFIGURAZIONE ---
DATASET_ID = "ICON_2I_SURFACE_PRESSURE_LEVELS"
API_LIST_URL = f"https://meteohub.agenziaitaliameteo.it/api/datasets/{DATASET_ID}/opendata"
API_DOWNLOAD_URL = "https://meteohub.agenziaitaliameteo.it/api/opendata"
TEMP_FILE = "test_debug.grib2"

def inspect_grib():
    print("1. Cerco un file di esempio...", flush=True)
    try:
        r = requests.get(API_LIST_URL, timeout=30)
        items = r.json()
        # Prendo il primo file disponibile dell'ultimo run
        last_item = items[-1]
        filename = last_item['filename']
        print(f"   Scarico: {filename}")
        
        with requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True) as r:
            with open(TEMP_FILE, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
    except Exception as e:
        print(f"Errore download: {e}")
        return

    print("\n2. ANALISI CONTENUTO GRIB (backend: cfgrib)")
    print("="*60)

    # cfgrib.open_datasets apre TUTTI i "messaggi" GRIB separati
    try:
        datasets = cfgrib.open_datasets(TEMP_FILE)
    except Exception as e:
        print(f"Errore apertura GRIB: {e}")
        return

    found_press = False
    found_rh = False

    for i, ds in enumerate(datasets):
        print(f"\n--- GRUPPO {i+1} ---")
        
        # Stampiamo le coordinate per capire il livello (heightAboveGround, meanSea, surface...)
        print("COORDINATE/FILTRI:")
        for coord in ds.coords:
            if coord not in ['latitude', 'longitude', 'step', 'time', 'valid_time']:
                val = ds[coord].values
                print(f"  - {coord}: {val}")

        print("VARIABILI TROVATE:")
        for var in ds.data_vars:
            attrs = ds[var].attrs
            long_name = attrs.get('long_name', 'N/A')
            units = attrs.get('units', 'N/A')
            print(f"  >>> KEY: '{var}' | Name: {long_name} | Units: {units}")
            
            if 'press' in long_name.lower() or 'press' in var.lower():
                found_press = True
            if 'dew' in long_name.lower() or 'humidity' in long_name.lower():
                found_rh = True

    print("\n" + "="*60)
    print(f"DIAGNOSI VELOCE:")
    print(f"- Pressione individuata? {'SI' if found_press else 'NO'}")
    print(f"- Dati per Umidità (DewPoint o RH)? {'SI' if found_rh else 'NO'}")
    
    if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
    if os.path.exists(f"{TEMP_FILE}.idx"): os.remove(f"{TEMP_FILE}.idx")

if __name__ == "__main__":
    inspect_grib()
  
