import requests
import xarray as xr
import numpy as np
import json
import os
import sys
import shutil
import glob
from datetime import datetime, timedelta

# --- CONFIGURAZIONE ---
DATASET_ID = "ICON_2I_SURFACE_PRESSURE_LEVELS"
API_LIST_URL = f"https://meteohub.agenziaitaliameteo.it/api/datasets/{DATASET_ID}/opendata"
API_DOWNLOAD_URL = "https://meteohub.agenziaitaliameteo.it/api/opendata"

FINAL_DIR = "data_weather"
TEMP_DIR = "temp_processing"
TEMP_FILE = "temp.grib2"

LAT_MIN, LAT_MAX = 35.0, 39.5
LON_MIN, LON_MAX = 11.0, 16.5

def cleanup_idx_files():
    # cfgrib crea file .idx nella cartella corrente
    try:
        for f in glob.glob("*.idx"):
            os.remove(f)
    except:
        pass

def safe_open_dataset(grib_path, filter_by_keys):
    """
    Apertura robusta: se non trova nulla, ritorna None.
    """
    cleanup_idx_files()
    try:
        ds = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": filter_by_keys}
        )
        return ds
    except:
        return None

def pick_var_key(ds, candidates):
    """
    Ritorna la prima chiave presente nel dataset fra i candidati, altrimenti None.
    """
    if ds is None:
        return None
    for k in candidates:
        if k in ds:
            return k
    return None

def get_latest_run_files():
    print("1. Cerco dati...", flush=True)
    try:
        r = requests.get(API_LIST_URL, timeout=30)
        r.raise_for_status()
        items = r.json()
    except:
        return None, []

    runs = {}
    for item in items:
        if isinstance(item, dict) and "date" in item and "run" in item:
            key = f"{item['date']} {item['run']}"
            if key not in runs:
                runs[key] = []
            runs[key].append(item["filename"])

    if not runs:
        return None, []

    latest_key = sorted(runs.keys())[-1]
    run_dt = datetime.strptime(latest_key, "%Y-%m-%d %H:%M")
    return run_dt, runs[latest_key][:48]

def process_data():
    run_dt, file_list = get_latest_run_files()
    if not file_list:
        sys.exit(0)

    print(f"2. Elaboro {len(file_list)} files...", flush=True)

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    catalog = []

    for idx, filename in enumerate(file_list):
        print(f"   DL {filename}...", end=" ", flush=True)

        try:
            with requests.get(f"{API_DOWNLOAD_URL}/{filename}", stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(TEMP_FILE, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            print("OK", end=" ", flush=True)
        except:
            print("KO", flush=True)
            continue

        # --- APERTURE DATASET ---
        # VENTO 10m (obbligatorio)
        ds_wind = safe_open_dataset(TEMP_FILE, {"typeOfLevel": "heightAboveGround", "level": 10})
        if ds_wind is None:
            print(" -> KO (wind missing)", flush=True)
            continue

        # TEMPERATURA 2m (opzionale)
        ds_temp = safe_open_dataset(TEMP_FILE, {"typeOfLevel": "heightAboveGround", "level": 2})

        # UMIDITÀ RELATIVA 2m (opzionale) - spesso in dataset separato, quindi prova più filtri
        ds_rh = (
            safe_open_dataset(TEMP_FILE, {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "r"}) or
            safe_open_dataset(TEMP_FILE, {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "2r"}) or
            safe_open_dataset(TEMP_FILE, {"typeOfLevel": "heightAboveGround", "level": 2})
        )

        # DEWPOINT 2m (opzionale)
        ds_dew = (
            safe_open_dataset(TEMP_FILE, {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "2d"}) or
            safe_open_dataset(TEMP_FILE, {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "d2m"}) or
            safe_open_dataset(TEMP_FILE, {"typeOfLevel": "heightAboveGround", "level": 2})
        )

        # PIOGGIA accumulata (opzionale) - come avevi tu
        ds_rain = safe_open_dataset(TEMP_FILE, {"typeOfLevel": "surface", "stepType": "accum"})

        # NUVOLOSITÀ totale (opzionale): spesso shortName tcc
        ds_cloud = (
            safe_open_dataset(TEMP_FILE, {"typeOfLevel": "surface", "shortName": "tcc"}) or
            safe_open_dataset(TEMP_FILE, {"typeOfLevel": "surface", "shortName": "tcdc"}) or
            safe_open_dataset(TEMP_FILE, {"shortName": "tcc"}) or
            safe_open_dataset(TEMP_FILE, {"typeOfLevel": "surface"})
        )

        # RAFFICHE (opzionale): spesso 'gust' o simili
        ds_gust = (
            safe_open_dataset(TEMP_FILE, {"typeOfLevel": "surface", "shortName": "gust"}) or
            safe_open_dataset(TEMP_FILE, {"shortName": "gust"}) or
            safe_open_dataset(TEMP_FILE, {"typeOfLevel": "surface"})
        )

        # --- RICERCA AVANZATA PRESSIONE (come avevi tu) ---
        press_data_full = None
        press_src = "None"

        # 1. Mean Sea Level (msl, prmsl)
        if press_data_full is None:
            try:
                ds_p = safe_open_dataset(TEMP_FILE, {"typeOfLevel": "meanSea"})
                if ds_p is not None:
                    for k in ["prmsl", "msl", "pres", "press"]:
                        if k in ds_p:
                            press_data_full = ds_p
                            press_src = f"MeanSea_{k}"
                            break
            except:
                pass

        # 2. Surface (sp, pres)
        if press_data_full is None:
            try:
                ds_p = safe_open_dataset(TEMP_FILE, {"typeOfLevel": "surface"})
                if ds_p is not None:
                    for k in ["sp", "pres", "pressure", "aps"]:
                        if k in ds_p:
                            press_data_full = ds_p
                            press_src = f"Surface_{k}"
                            break
            except:
                pass

        # 3. Generic shortName pres
        if press_data_full is None:
            try:
                ds_p = safe_open_dataset(TEMP_FILE, {"shortName": "pres"})
                if ds_p is not None:
                    press_data_full = ds_p
                    press_src = "Generic_Pres"
            except:
                pass

        # --- Loop sugli step ---
        steps = range(ds_wind.sizes.get("step", 1))

        for i in steps:
            try:
                raw_step = ds_wind.step.values[i]
                step_hours = int(raw_step / np.timedelta64(1, "h")) if isinstance(raw_step, np.timedelta64) else int(raw_step)

                # Taglio Sicilia (basato su griglia del vento, così resta coerente)
                d_w = (
                    ds_wind.isel(step=i)
                    .sortby("latitude", ascending=False)
                    .sortby("longitude", ascending=True)
                )
                mask = (
                    (d_w.latitude >= LAT_MIN) & (d_w.latitude <= LAT_MAX) &
                    (d_w.longitude >= LON_MIN) & (d_w.longitude <= LON_MAX)
                )
                cut_w = d_w.where(mask, drop=True)

                u_key = pick_var_key(cut_w, ["u10", "u"])
                v_key = pick_var_key(cut_w, ["v10", "v"])
                if not (u_key and v_key):
                    continue

                u = np.nan_to_num(cut_w[u_key].values)
                v = np.nan_to_num(cut_w[v_key].values)

                lat, lon = cut_w.latitude.values, cut_w.longitude.values
                ny, nx = u.shape
                la1, lo1 = float(lat[0]), float(lon[0])
                dx, dy = float(abs(lon[1] - lon[0])), float(abs(lat[0] - lat[1]))
                lo2, la2 = lo1 + (nx - 1) * dx, la1 - (ny - 1) * dy

                # --- TEMP 2m (°C) ---
                temp = np.zeros_like(u)
                if ds_temp is not None:
                    try:
                        d_t = (
                            ds_temp.isel(step=i)
                            .sortby("latitude", ascending=False)
                            .sortby("longitude", ascending=True)
                            .where(mask, drop=True)
                        )
                        t_key = pick_var_key(d_t, ["t2m", "t"])
                        if t_key:
                            temp = np.nan_to_num(d_t[t_key].values - 273.15)
                    except:
                        pass

                # --- RAIN (accum) ---
                rain = np.zeros_like(u)
                if ds_rain is not None:
                    try:
                        d_r = (
                            ds_rain.isel(step=i)
                            .sortby("latitude", ascending=False)
                            .sortby("longitude", ascending=True)
                            .where(mask, drop=True)
                        )
                        r_key = pick_var_key(d_r, ["tp", "tot_prec", "apcp"])
                        if r_key:
                            rain = np.nan_to_num(d_r[r_key].values)
                    except:
                        pass

                # --- PRESSIONE (hPa) ---
                press = np.zeros_like(u)
                if press_data_full is not None:
                    try:
                        d_p = (
                            press_data_full.isel(step=i)
                            .sortby("latitude", ascending=False)
                            .sortby("longitude", ascending=True)
                            .where(mask, drop=True)
                        )
                        p_key = pick_var_key(d_p, ["prmsl", "msl", "sp", "pres", "pressure"])
                        if p_key:
                            press = np.nan_to_num(d_p[p_key].values)
                    except:
                        pass

                # Convert Pascal -> hPa se necessario
                if np.max(press) > 2000:
                    press = press / 100.0

                # Se ancora 0, fallback 1013
                if np.max(press) < 500:
                    press.fill(1013.0)

                # --- UMIDITÀ RELATIVA 2m (%) ---
                rh = np.zeros_like(u)
                if ds_rh is not None:
                    try:
                        d_h = (
                            ds_rh.isel(step=i)
                            .sortby("latitude", ascending=False)
                            .sortby("longitude", ascending=True)
                            .where(mask, drop=True)
                        )
                        rh_key = pick_var_key(d_h, ["r2", "2r", "r", "rh", "relhum"])
                        if rh_key:
                            rh_raw = np.nan_to_num(d_h[rh_key].values)
                            # alcune griglie sono 0..1, altre 0..100
                            if np.nanmax(rh_raw) <= 1.5:
                                rh = rh_raw * 100.0
                            else:
                                rh = rh_raw
                    except:
                        pass

                # --- DEWPOINT 2m (°C) ---
                dewpoint = np.zeros_like(u)
                if ds_dew is not None:
                    try:
                        d_d = (
                            ds_dew.isel(step=i)
                            .sortby("latitude", ascending=False)
                            .sortby("longitude", ascending=True)
                            .where(mask, drop=True)
                        )
                        d_key = pick_var_key(d_d, ["2d", "d2m", "td2m", "dewpoint"])
                        if d_key:
                            dp = np.nan_to_num(d_d[d_key].values)
                            # spesso in Kelvin
                            if np.nanmean(dp) > 150:
                                dewpoint = dp - 273.15
                            else:
                                dewpoint = dp
                    except:
                        pass

                # --- NUVOLOSITÀ totale (%) ---
                cloud = np.zeros_like(u)
                if ds_cloud is not None:
                    try:
                        d_c = (
                            ds_cloud.isel(step=i)
                            .sortby("latitude", ascending=False)
                            .sortby("longitude", ascending=True)
                            .where(mask, drop=True)
                        )
                        c_key = pick_var_key(d_c, ["tcc", "tcdc", "cc", "clct"])
                        if c_key:
                            c_raw = np.nan_to_num(d_c[c_key].values)
                            # 0..1 oppure 0..100
                            if np.nanmax(c_raw) <= 1.5:
                                cloud = c_raw * 100.0
                            else:
                                cloud = c_raw
                    except:
                        pass

                # --- RAFFICHE (m/s) ---
                gust = np.zeros_like(u)
                if ds_gust is not None:
                    try:
                        d_g = (
                            ds_gust.isel(step=i)
                            .sortby("latitude", ascending=False)
                            .sortby("longitude", ascending=True)
                            .where(mask, drop=True)
                        )
                        g_key = pick_var_key(d_g, ["gust", "10fg", "fg10", "fg"])
                        if g_key:
                            gust = np.nan_to_num(d_g[g_key].values)
                    except:
                        pass

                # DEBUG (solo primo frame)
                if i == 0:
                    print(f" [DEBUG: PressSrc={press_src}, Pmax={np.max(press):.1f}, RHmax={np.max(rh):.1f}, Cmax={np.max(cloud):.1f}]", end="")

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
                    "wind_u": {"header": {**header, "parameterCategory": 2, "parameterNumber": 2}, "data": np.round(u, 1).flatten().tolist()},
                    "wind_v": {"header": {**header, "parameterCategory": 2, "parameterNumber": 3}, "data": np.round(v, 1).flatten().tolist()},
                    "temp": np.round(temp, 1).flatten().tolist(),
                    "rain": np.round(rain, 2).flatten().tolist(),
                    "press": np.round(press, 1).flatten().tolist(),

                    # --- NUOVI CAMPI ---
                    "rh": np.round(rh, 1).flatten().tolist(),
                    "dewpoint": np.round(dewpoint, 1).flatten().tolist(),
                    "cloud": np.round(cloud, 1).flatten().tolist(),
                    "gust": np.round(gust, 1).flatten().tolist()
                }

                out_name = f"step_{step_hours}.json"
                with open(f"{TEMP_DIR}/{out_name}", "w") as jf:
                    json.dump(step_data, jf)

                if not any(x["hour"] == step_hours for x in catalog):
                    catalog.append({"file": out_name, "label": f"{valid_dt.strftime('%d/%m %H:00')}", "hour": step_hours})

            except Exception:
                continue

        print(" -> OK")

    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)

    if catalog:
        catalog.sort(key=lambda x: x["hour"])
        with open(f"{TEMP_DIR}/catalog.json", "w") as f:
            json.dump(catalog, f)

        if os.path.exists(FINAL_DIR):
            shutil.rmtree(FINAL_DIR)
        shutil.move(TEMP_DIR, FINAL_DIR)
        print("COMPLETATO.")
    else:
        sys.exit(1)

if __name__ == "__main__":
    process_data()
