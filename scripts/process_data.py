import xarray as xr
import numpy as np
import json
import os
import requests
from datetime import datetime, timedelta

# --- CONFIGURAZIONE SICILIA ---
# Ritaglio preciso sulla Sicilia per non appesantire il JSON
LAT_MIN, LAT_MAX = 35.0, 39.0
LON_MIN, LON_MAX = 11.5, 16.0

# Cartella di output per i JSON
OUTPUT_DIR = "data_weather"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- MAPPA VARIABILI (GRIB shortName -> JSON key) ---
# Qui avviene la magia: chiediamo al modello ICON-2I esattamente questi dati.
VARIABLES_MAP = {
    # Dati Base
    "t_2m": "temp",        # Temperatura 2m
    "u_10m": "wind_u",     # Vento Zonale (per particelle)
    "v_10m": "wind_v",     # Vento Meridionale (per particelle)
    "tot_prec": "rain",    # Pioggia Totale
    
    # --- DATI PRO (LIVELLO ESPERTO) ---
    "clct": "clouds",      # Copertura Nuvolosa Totale (0-100%)
    "vmax_10m": "gust",    # Raffiche di Vento (m/s)
    "r_2m": "hum",         # Umidità Relativa (%)
    "cape_ml": "cape"      # Energia Temporali (J/kg)
}

# URL Base del modello DWD ICON-D2 (Aggiornato ogni 3 ore circa)
# Nota: Questo scarica l'ultima run disponibile.
BASE_URL = "https://opendata.dwd.de/weather/nwp/icon-d2/grib"

def get_latest_run_url():
    """Trova l'URL dell'ultima run disponibile (00, 03, 06, 09, etc.)"""
    # Per semplicità, in questo esempio puntiamo alla run delle 00 o 12 di oggi.
    # In produzione, dovresti fare lo scraping per trovare l'ultima cartella.
    now = datetime.utcnow()
    run = "00" if now.hour < 12 else "12" 
    date_str = now.strftime("%Y%m%d")
    return f"{BASE_URL}/{run}", run, date_str

def download_file(url, filename):
    """Scarica il file se non esiste"""
    if os.path.exists(filename):
        return True
    print(f"Scarico: {url}")
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            print(f"Errore {r.status_code} su {url}")
            return False
    except Exception as e:
        print(f"Eccezione: {e}")
        return False

def process_grib():
    """Processa i dati e crea i JSON"""
    print("--- INIZIO ELABORAZIONE DATI METEO PRO ---")
    
    # 1. Setup Base URL
    base_url, run_hour, date_str = get_latest_run_url()
    
    # 2. Scarichiamo un file "merged" o vari file. 
    # ICON-D2 è diviso per variabili. Per questo script dimostrativo, 
    # assumiamo di aver scaricato i file GRIB necessari o usiamo un approccio semplificato.
    # NOTA: Se usi MeteoHub, sostituisci questa parte con il tuo path locale dei file GRIB.
    
    # Simuliamo il caricamento di un Dataset (sostituisci con il tuo file GRIB reale)
    # ds = xr.open_dataset('tuo_file_meteo_sicilia.grib2', engine='cfgrib')
    
    # SE NON HAI IL FILE, ecco come xarray può aprire file remoti o locali se li hai scaricati.
    # Qui simuliamo la logica di estrazione assumendo che tu abbia un dataset `ds` pronto.
    # Se scarichi da MeteoHub, avrai probabilmente un file tipo "icon_sicily.grib2"
    
    grib_file = "icon_d2_sicily.grib2" 
    
    # (Inserire qui la logica di download specifica se non hai i file locali)
    # Se il file non c'è, ferma tutto (o scaricalo dai tuoi link MeteoHub)
    if not os.path.exists(grib_file):
        print(f"⚠️ ATTENZIONE: File {grib_file} non trovato. Inserisci il file GRIB nella cartella.")
        print("Il file deve contenere le variabili: t_2m, u/v_10m, tot_prec, clct, vmax_10m, r_2m, cape_ml")
        return

    try:
        # Apriamo il GRIB filtrando per le chiavi che ci servono
        # filter_by_keys aiuta a gestire file con più tipi di livelli
        ds = xr.open_dataset(grib_file, engine='cfgrib', 
                             backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
        # Nota: CAPE e Nuvole potrebbero essere su livelli diversi (es. atmosphere), 
        # potresti dover aprire il file più volte o senza filtri e poi selezionare.
        
    except Exception as e:
        print(f"Errore apertura GRIB: {e}")
        return

    # 3. Ritaglio sulla Sicilia (Cruciale per performance)
    ds_sicily = ds.sel(latitude=slice(LAT_MIN, LAT_MAX), longitude=slice(LON_MIN, LON_MAX))
    
    # Estraiamo coordinate per i metadati
    lats = ds_sicily.latitude.values
    lons = ds_sicily.longitude.values
    
    # Meta dati per il frontend (necessari per disegnare la griglia)
    meta = {
        'nx': int(ds_sicily.sizes['longitude']),
        'ny': int(ds_sicily.sizes['latitude']),
        'la1': float(lats.max()), # Latitudine Top-Left
        'lo1': float(lons.min()), # Longitudine Top-Left
        'dx': float(lons[0, 1] - lons[0, 0]), # Passo griglia X
        'dy': float(lats[0, 0] - lats[1, 0])  # Passo griglia Y
    }

    catalog = []

    # 4. Iteriamo sui timesteps (ore)
    # Se il file ha dimensione tempo 'step' o 'time'
    steps = ds_sicily.step if 'step' in ds_sicily else [0]
    
    for i, step in enumerate(steps):
        # Selezione frame temporale
        frame = ds_sicily.isel(step=i)
        
        # Dizionario dati per questo frame
        frame_data = {'meta': meta}
        
        # Estrazione e Ottimizzazione Variabili
        for grib_name, json_key in VARIABLES_MAP.items():
            if grib_name in frame:
                val = frame[grib_name].values
                
                # Sostituzione NaN con 0
                val = np.nan_to_num(val)
                
                # OTTIMIZZAZIONE DATI PRO
                if json_key == 'temp':
                    val = val - 273.15 # Kelvin -> Celsius
                    val = np.round(val, 1)
                elif json_key == 'rain':
                    val = np.round(val, 1) # mm
                elif json_key in ['wind_u', 'wind_v', 'gust']:
                    val = np.round(val, 1) # m/s
                elif json_key in ['clouds', 'hum']:
                    val = np.round(val, 0).astype(int) # % intero
                elif json_key == 'cape':
                    val = np.round(val, 0).astype(int) # J/kg intero
                
                # Aggiungiamo i dati "flat" (lista semplice)
                frame_data[json_key] = val.flatten().tolist()
                
                # Per il vento, calcoliamo min/max per calibrazione particelle
                if json_key == 'wind_u':
                    frame_data['wind_u'] = {'data': val.flatten().tolist(), 'min': float(val.min()), 'max': float(val.max())}
                if json_key == 'wind_v':
                    frame_data['wind_v'] = {'data': val.flatten().tolist(), 'min': float(val.min()), 'max': float(val.max())}

        # Salvataggio JSON Frame
        filename = f"frame_{i:03d}.json"
        with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
            json.dump(frame_data, f)
            
        # Aggiunta al catalogo
        time_label = (datetime.now() + timedelta(hours=i)).strftime("%d/%m %H:00")
        catalog.append({'file': filename, 'label': time_label})
        
        print(f"Generato frame {i}: {time_label} con dati PRO (CAPE, Nuvole, Raffiche...)")

    # 5. Salvataggio Catalogo
    with open(os.path.join(OUTPUT_DIR, "catalog.json"), 'w') as f:
        json.dump(catalog, f)

    print("--- COMPLETATO: DATI METEO PRO PRONTI ---")

if __name__ == "__main__":
    process_grib()
    
