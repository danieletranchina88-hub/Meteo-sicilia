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
            if key not in runs:
                runs[key] = []
            runs[key].append(item['filename'])

    if not runs:
        return None, []
    latest_key = sorted(runs.keys())[-1]
    run_dt = datetime.strptime(latest_key, "%Y-%m-%d %H:%M")
    return run_dt, runs[latest_key][:48]


def calculate_rh_numpy(temp_k, dew_k):
    """Calcola RH usando Numpy puro per evitare errori di allineamento Xarray"""
    # Formula August-Roche-Magnus
    T = temp_k - 273.15
    Td = dew_k - 273.15
    a = 17.625
    b = 243.04

    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.exp((a * Td) / (b + Td))
        denominator = np.exp((a * T) / (b + T))
        rh = 100 * (numerator / denominator)

    return np.nan_to_num(np.clip(rh, 0, 100))


def extract_raw_grid(ds, mask, var_names):
    """Estrae i valori crudi (numpy) ignorando coordinate temporali per evitare crash"""
    try:
        var_key = next((k for k in var_names if k in ds), None)
        if not var_key:
            return None
        d_masked = ds[var_key].where(mask, drop=True)
        return d_masked.values
    except Exception:
        return None


def try_open_cloud_dataset(grib_path):
    """
    Prova ad aprire un dataset che contenga la copertura nuvolosa.
    In GRIB può essere tcc/tcdc/clct ecc con vari typeOfLevel.
    """
    # Tentativi “mirati” per shortName (spesso funziona anche senza typeOfLevel)
    candidates = [
        {'filter_by_keys': {'shortName': 'tcc'}},
        {'filter_by_keys': {'shortName': 'tcdc'}},
        {'filter_by_keys': {'shortName': 'clct'}},
        {'filter_by_keys': {'shortName': 'cc'}},
    ]

    # Tentativi per typeOfLevel comuni (dipende da come è impacchettato il GRIB)
    candidates += [
        {'filter_by_keys': {'typeOfLevel': 'entireAtmosphere'}},
        {'filter_by_keys': {'typeOfLevel': 'atmosphere'}},
        {'filter_by_keys': {'typeOfLevel': 'surface'}},
        {'filter_by_keys': {'typeOfLevel': 'cloudBase'}},
        {'filter_by_keys': {'typeOfLevel': 'cloudTop'}},
    ]

    for bk in candidates:
        try:
            ds = xr.open_dataset(grib_path, engine='cfgrib', backend_kwargs=bk)
            if ds is not None and len(ds.data_vars) > 0:
                return ds
        except Exception:
            continue

    return None


def normalize_cloud_to_percent(cloud_arr):
    """
    Normalizza copertura nuvolosa in percentuale 0-100.
    - Se arriva in frazione 0-1 => *100
    - Se già in 0-100 => ok
    """
    c = np.nan_to_num(cloud_arr)
    if c.size == 0:
        return c

    # Heuristica robusta: se max <= 1.01 assume frazione
    mx = float(np.nanmax(c)) if np.isfinite(np.nanmax(c)) else 0.0
    if mx <= 1.01:
        c = c * 100.0

    # clamp
    c = np.clip(c, 0.0, 100.0)
    return c


def process_data():
    run_dt, file_list = get_latest_run_files()
    if not file_list:
        print("Nessun dato trovato.")
        sys.exit(0)

    print(f"2. Elaboro Run: {run_dt} ({len(file_list)} files)", flush=True)

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    catalog = []

    for idx, filename in enumerate(file_list):
        print(f"   [{idx+1:02d}] DL {filename}...", end=" ", flush=True)

        # --- DOWNLOAD ---
        try:
            with requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(TEMP_FILE, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            print("OK", end=" ", flush=True)
        except Exception as e:
            print(f"KO ({e})", flush=True)
            continue

        # pulizia idx
        if os.path.exists(f"{TEMP_FILE}.idx"):
            os.remove(f"{TEMP_FILE}.idx")

        # --- APERTURA DATASET (Setup Multi-livello) ---
        try:
            # 1. Vento (Master grid) - Level 10 heightAboveGround
            ds_wind = xr.open_dataset(
                TEMP_FILE,
                engine='cfgrib',
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}}
            )

            # 2. Termodinamica (Temp/Dew) - Level 2 heightAboveGround
            ds_thermo = None
            try:
                ds_thermo = xr.open_dataset(
                    TEMP_FILE,
                    engine='cfgrib',
                    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}}
                )
            except Exception:
                pass

            # 3. Pressione - meanSea (pmsl / prmsl / msl)
            ds_press = None
            try:
                ds_press = xr.open_dataset(
                    TEMP_FILE,
                    engine='cfgrib',
                    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}}
                )
            except Exception:
                pass

            # 4. Pioggia - surface accum
            ds_rain = None
            try:
                ds_rain = xr.open_dataset(
                    TEMP_FILE,
                    engine='cfgrib',
                    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'stepType': 'accum'}}
                )
            except Exception:
                pass

            # 5. NUVOLOSITÀ - tentativi multipli
            ds_cloud = try_open_cloud_dataset(TEMP_FILE)

        except Exception as e:
            print(f" Skip (Grib Error: {e})")
            continue

        # --- LOOP TEMPORALE ---
        steps = range(ds_wind.sizes.get('step', 1))

        for i in steps:
            try:
                # Gestione singolo step o multi-step
                if ds_wind.sizes.get('step', 1) > 1:
                    dw_step = ds_wind.isel(step=i)
                    raw_step = ds_wind.step.values[i]
                else:
                    dw_step = ds_wind
                    raw_step = ds_wind.step.values

                step_hours = int(raw_step / np.timedelta64(1, 'h')) if isinstance(raw_step, np.timedelta64) else int(raw_step)

                # --- MASCHERA GEOGRAFICA ---
                dw_step = dw_step.sortby('latitude', ascending=False).sortby('longitude', ascending=True)
                mask = (
                    (dw_step.latitude >= LAT_MIN) & (dw_step.latitude <= LAT_MAX) &
                    (dw_step.longitude >= LON_MIN) & (dw_step.longitude <= LON_MAX)
                )

                cut_w = dw_step.where(mask, drop=True)
                if cut_w.latitude.size == 0:
                    continue

                # --- VENTO ---
                u_key = next((k for k in ['u10', 'u'] if k in cut_w), None)
                v_key = next((k for k in ['v10', 'v'] if k in cut_w), None)
                if not u_key or not v_key:
                    continue

                u_val = np.nan_to_num(cut_w[u_key].values)
                v_val = np.nan_to_num(cut_w[v_key].values)

                # GRID METADATA
                lat = cut_w.latitude.values
                lon = cut_w.longitude.values
                if lat.ndim > 1:
                    lat = lat[:, 0]
                if lon.ndim > 1:
                    lon = lon[0, :]

                ny, nx = u_val.shape
                la1, lo1 = float(lat[0]), float(lon[0])
                dx, dy = float(abs(lon[1] - lon[0])), float(abs(lat[0] - lat[1]))
                lo2, la2 = lo1 + (nx - 1) * dx, la1 - (ny - 1) * dy

                # --- 1) TEMP e RH ---
                temp_c = np.zeros_like(u_val)
                rh_val = np.zeros_like(u_val)

                if ds_thermo is not None:
                    dt_step = ds_thermo.isel(step=i) if ds_thermo.sizes.get('step', 1) > 1 else ds_thermo
                    dt_step = dt_step.sortby('latitude', ascending=False).sortby('longitude', ascending=True)

                    t_raw = extract_raw_grid(dt_step, mask, ['t2m', 't'])
                    d_raw = extract_raw_grid(dt_step, mask, ['d2m', '2d'])

                    if t_raw is not None and t_raw.shape == u_val.shape:
                        temp_c = t_raw - 273.15
                        if d_raw is not None and d_raw.shape == u_val.shape:
                            rh_val = calculate_rh_numpy(t_raw, d_raw)

                # --- 2) PRESSIONE (pmsl) ---
                press = np.zeros_like(u_val)
                if ds_press is not None:
                    dp_step = ds_press.isel(step=i) if ds_press.sizes.get('step', 1) > 1 else ds_press
                    dp_step = dp_step.sortby('latitude', ascending=False).sortby('longitude', ascending=True)

                    p_raw = extract_raw_grid(dp_step, mask, ['pmsl', 'prmsl', 'msl'])
                    if p_raw is not None and p_raw.shape == u_val.shape:
                        p_clean = np.nan_to_num(p_raw)
                        if np.max(p_clean) > 80000:
                            press = p_clean / 100.0  # Pa -> hPa
                        else:
                            press = p_clean

                if np.max(press) < 800:
                    press.fill(1013.0)

                # --- 3) PIOGGIA ---
                rain = np.zeros_like(u_val)
                if ds_rain is not None:
                    dr_step = ds_rain.isel(step=i) if ds_rain.sizes.get('step', 1) > 1 else ds_rain
                    dr_step = dr_step.sortby('latitude', ascending=False).sortby('longitude', ascending=True)

                    r_raw = extract_raw_grid(dr_step, mask, ['tp', 'tot_prec'])
                    if r_raw is not None and r_raw.shape == u_val.shape:
                        rain = np.nan_to_num(r_raw)

                # --- 4) NUVOLOSITÀ TOTALE (cloud cover %) ---
                cloud = np.zeros_like(u_val)
                if ds_cloud is not None:
                    dc_step = ds_cloud.isel(step=i) if ds_cloud.sizes.get('step', 1) > 1 else ds_cloud
                    dc_step = dc_step.sortby('latitude', ascending=False).sortby('longitude', ascending=True)

                    # nomi variabili tipici
                    c_raw = extract_raw_grid(dc_step, mask, [
                        'tcc', 'tcdc', 'clct', 'cc', 'tcc_total', 'totalCloudCover'
                    ])
                    if c_raw is not None and c_raw.shape == u_val.shape:
                        cloud = normalize_cloud_to_percent(c_raw)

                # --- EXPORT JSON ---
                valid_dt = run_dt + timedelta(hours=step_hours)
                iso_date = valid_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

                header = {
                    "nx": nx, "ny": ny,
                    "lo1": lo1, "la1": la1,
                    "lo2": lo2, "la2": la2,
                    "dx": dx, "dy": dy,
                    "refTime": iso_date
                }

                step_data = {
                    "meta": header,
                    "wind_u": {"header": {**header, "parameterCategory": 2, "parameterNumber": 2}, "data": np.round(u_val, 1).flatten().tolist()},
                    "wind_v": {"header": {**header, "parameterCategory": 2, "parameterNumber": 3}, "data": np.round(v_val, 1).flatten().tolist()},
                    "temp": np.round(temp_c, 1).flatten().tolist(),
                    "rain": np.round(rain, 2).flatten().tolist(),
                    "press": np.round(press, 1).flatten().tolist(),   # hPa
                    "rh": np.round(rh_val, 0).flatten().tolist(),      # %
                    "cloud": np.round(cloud, 0).flatten().tolist()     # % (0-100)
                }

                out_name = f"step_{step_hours}.json"
                with open(f"{TEMP_DIR}/{out_name}", 'w') as jf:
                    json.dump(step_data, jf)

                if not any(x['hour'] == step_hours for x in catalog):
                    catalog.append({"file": out_name, "label": f"{valid_dt.strftime('%d/%m %H:00')}", "hour": step_hours})

            except Exception:
                print("!", end="", flush=True)
                continue

        print(" -> Done")

    # Cleanup finale
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)
    if os.path.exists(f"{TEMP_FILE}.idx"):
        os.remove(f"{TEMP_FILE}.idx")

    if catalog:
        catalog.sort(key=lambda x: x['hour'])
        with open(f"{TEMP_DIR}/catalog.json", 'w') as f:
            json.dump(catalog, f)

        if os.path.exists(FINAL_DIR):
            shutil.rmtree(FINAL_DIR)
        shutil.move(TEMP_DIR, FINAL_DIR)
        print("\nELABORAZIONE COMPLETATA CON SUCCESSO.")
    else:
        print("\nNESSUN DATI VALIDO ESTRATTO.")
        sys.exit(1)


if __name__ == "__main__":
    process_data()