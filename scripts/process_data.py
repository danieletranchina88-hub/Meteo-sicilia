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
        if not items:
            print("Nessun file trovato nelle API.")
            return
            
        # Prendo l'ultimo file disponibile
        last_item = items[-1]
        filename = last_item['filename']
        print(f"   Scarico: {filename}")
        
        with requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True) as r:
            with open(TEMP_FILE, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
    except Exception as e:
        print(f"Errore download: {e}")
        return

    print("\n2. ANALISI CONTENUTO GRIB")
    print("="*60)

    # Proviamo ad aprire il file e vedere cosa contiene
    try:
        # filter_by_keys={} vuoto prova a leggere tutto, ma grib spesso separa i messaggi.
        # open_datasets (plurale) ci dà una lista di tutti i "messaggi" grib separati.
        datasets = cfgrib.open_datasets(TEMP_FILE)
    except Exception as e:
        print(f"Errore apertura GRIB generale: {e}")
        return

    found_press = False
    
    for i, ds in enumerate(datasets):
        print(f"\n--- GRUPPO DATI {i+1} ---")
        
        # Stampiamo le coordinate per capire il livello
        print("COORDINATE/FILTRI DEL GRUPPO:")
        coords_info = []
        for coord in ds.coords:
            if coord not in ['latitude', 'longitude', 'step', 'time', 'valid_time']:
                val = ds[coord].values
                coords_info.append(f"{coord}={val}")
        print("  " + ", ".join(coords_info))

        print("VARIABILI DISPONIBILI:")
        for var in ds.data_vars:
            attrs = ds[var].attrs
            long_name = attrs.get('long_name', 'N/A')
            units = attrs.get('units', 'N/A')
            grib_name = attrs.get('GRIB_shortName', var)
            
            print(f"  >>> NOME VARIABILE: '{var}' (GribName: {grib_name})")
            print(f"      Descrizione: {long_name}")
            print(f"      Unità: {units}")
            
            if 'pres' in long_name.lower() or 'mean sea' in long_name.lower():
                found_press = True
                print("      [!!!] QUESTA SEMBRA PRESSIONE")

    print("\n" + "="*60)
    
    if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
    if os.path.exists(f"{TEMP_FILE}.idx"): os.remove(f"{TEMP_FILE}.idx")

if __name__ == "__main__":
    try:
        inspect_grib()
    except Exception as e:
        print(f"Crash finale: {e}")
        
