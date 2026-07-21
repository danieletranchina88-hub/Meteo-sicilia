import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import xarray as xr
from ecmwf.opendata import Client

FINAL_DIR = "data_weather_ecmwf"
TEMP_DIR = "temp_processing_ecmwf"
GRIB_FILE = "temp_ecmwf.grib2"

LAT_MIN, LAT_MAX = 35.0, 48.5
LON_MIN, LON_MAX = 6.0, 19.5

# Cinque giorni con passo di sei ore: abbastanza esteso per il confronto
# sinottico, ma leggero da pubblicare tramite GitHub Pages.
FORECAST_STEPS = list(range(0, 121, 6))
PARAMETERS = ["2t", "2d", "10u", "10v", "msl", "tp", "tcc"]
FEELS_LIKE_METHOD = "heat-index-wind-chill-v1"


def iso_z(value):
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_for_json(values, decimals):
    rounded = np.round(np.asarray(values, dtype=float), decimals).ravel()
    return [None if not np.isfinite(item) else float(item) for item in rounded]


def calculate_relative_humidity(temp_kelvin, dew_kelvin):
    temp_c = temp_kelvin - 273.15
    dew_c = dew_kelvin - 273.15
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        numerator = np.exp((17.625 * dew_c) / (243.04 + dew_c))
        denominator = np.exp((17.625 * temp_c) / (243.04 + temp_c))
        humidity = 100.0 * numerator / denominator
    return np.clip(humidity, 0.0, 100.0)


def calculate_heat_index_celsius(temp_c, humidity):
    temp_f = temp_c * 9.0 / 5.0 + 32.0
    rh = np.clip(humidity, 0.0, 100.0)
    simple = 0.5 * (temp_f + 61.0 + (temp_f - 68.0) * 1.2 + rh * 0.094)
    preliminary = (simple + temp_f) / 2.0

    heat_index_f = (
        -42.379
        + 2.04901523 * temp_f
        + 10.14333127 * rh
        - 0.22475541 * temp_f * rh
        - 0.00683783 * temp_f**2
        - 0.05481717 * rh**2
        + 0.00122874 * temp_f**2 * rh
        + 0.00085282 * temp_f * rh**2
        - 0.00000199 * temp_f**2 * rh**2
    )

    low_humidity = (rh < 13.0) & (temp_f >= 80.0) & (temp_f <= 112.0)
    low_adjustment = ((13.0 - rh) / 4.0) * np.sqrt(
        np.maximum(0.0, (17.0 - np.abs(temp_f - 95.0)) / 17.0)
    )
    heat_index_f = np.where(low_humidity, heat_index_f - low_adjustment, heat_index_f)

    high_humidity = (rh > 85.0) & (temp_f >= 80.0) & (temp_f <= 87.0)
    high_adjustment = ((rh - 85.0) / 10.0) * ((87.0 - temp_f) / 5.0)
    heat_index_f = np.where(high_humidity, heat_index_f + high_adjustment, heat_index_f)
    heat_index_f = np.where(preliminary >= 80.0, heat_index_f, temp_f)
    return (heat_index_f - 32.0) * 5.0 / 9.0


def calculate_wind_chill_celsius(temp_c, wind_kmh):
    wind_factor = np.power(wind_kmh, 0.16)
    return (
        13.12
        + 0.6215 * temp_c
        - 11.37 * wind_factor
        + 0.3965 * temp_c * wind_factor
    )


def calculate_feels_like(temp_c, humidity, u_wind, v_wind):
    result = np.array(temp_c, dtype=float, copy=True)
    speed_kmh = np.hypot(u_wind, v_wind) * 3.6
    finite_temp = np.isfinite(temp_c)

    cold = finite_temp & np.isfinite(speed_kmh) & (temp_c <= 10.0) & (speed_kmh > 4.8)
    wind_chill = calculate_wind_chill_celsius(temp_c, speed_kmh)
    result = np.where(cold, wind_chill, result)

    hot = finite_temp & np.isfinite(humidity) & (temp_c >= 26.7)
    heat_index = calculate_heat_index_celsius(temp_c, humidity)
    result = np.where(hot, heat_index, result)
    return np.where(finite_temp, result, np.nan)


def download_latest_forecast():
    last_error = None
    for source in ("aws", "ecmwf"):
        try:
            if os.path.exists(GRIB_FILE):
                os.remove(GRIB_FILE)
            print(f"Scarico ECMWF IFS dalla sorgente {source}...", flush=True)
            client = Client(source=source)
            result = client.retrieve(
                stream="oper",
                type="fc",
                step=FORECAST_STEPS,
                param=PARAMETERS,
                target=GRIB_FILE,
            )
            run_time = result.datetime
            if run_time.tzinfo is None:
                run_time = run_time.replace(tzinfo=timezone.utc)
            return run_time.astimezone(timezone.utc)
        except Exception as error:
            last_error = error
            print(f"Sorgente {source} non disponibile: {error}", flush=True)

    raise RuntimeError(f"Download ECMWF non riuscito: {last_error}")


def open_parameter(short_name):
    dataset = xr.open_dataset(
        GRIB_FILE,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"shortName": short_name},
            "indexpath": "",
        },
    )
    if not dataset.data_vars:
        dataset.close()
        raise ValueError(f"Parametro {short_name} assente")
    return dataset


def select_forecast_step(data_array, step_hours):
    if "step" not in data_array.dims:
        return data_array
    target = np.timedelta64(int(step_hours), "h")
    try:
        return data_array.sel(step=target)
    except Exception:
        steps = np.asarray(data_array.step.values)
        hours = steps / np.timedelta64(1, "h")
        index = int(np.argmin(np.abs(hours.astype(float) - float(step_hours))))
        return data_array.isel(step=index)


def crop_field(dataset, step_hours):
    variable_name = next(iter(dataset.data_vars))
    field = select_forecast_step(dataset[variable_name], step_hours).squeeze(drop=True)
    if "latitude" not in field.coords or "longitude" not in field.coords:
        raise ValueError(f"Coordinate mancanti per {variable_name}")

    field = field.sortby("latitude", ascending=False).sortby("longitude", ascending=True)
    field = field.sel(
        latitude=slice(LAT_MAX, LAT_MIN),
        longitude=slice(LON_MIN, LON_MAX),
    )
    values = np.asarray(field.values, dtype=float).squeeze()
    if values.ndim != 2 or values.size == 0:
        raise ValueError(f"Griglia non valida per {variable_name}")
    return field, values


def process_data():
    datasets = {}
    try:
        run_time = download_latest_forecast()
        print(f"Run ECMWF selezionato: {iso_z(run_time)}", flush=True)

        for short_name in PARAMETERS:
            datasets[short_name] = open_parameter(short_name)

        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR)

        catalog = []
        for step_hours in FORECAST_STEPS:
            print(f"Elaboro ECMWF +{step_hours:03d}h...", flush=True)

            reference, temp_kelvin = crop_field(datasets["2t"], step_hours)
            _, dew_kelvin = crop_field(datasets["2d"], step_hours)
            _, u_wind = crop_field(datasets["10u"], step_hours)
            _, v_wind = crop_field(datasets["10v"], step_hours)
            _, pressure_pa = crop_field(datasets["msl"], step_hours)
            rain_field, total_precipitation = crop_field(datasets["tp"], step_hours)
            _, total_cloud = crop_field(datasets["tcc"], step_hours)

            shape = temp_kelvin.shape
            arrays = [dew_kelvin, u_wind, v_wind, pressure_pa, total_precipitation, total_cloud]
            if any(array.shape != shape for array in arrays):
                raise ValueError(f"Griglie ECMWF non coerenti allo step {step_hours}")

            latitude = np.asarray(reference.latitude.values, dtype=float)
            longitude = np.asarray(reference.longitude.values, dtype=float)
            ny, nx = shape
            la1 = float(latitude[0])
            lo1 = float(longitude[0])
            dx = float(abs(longitude[1] - longitude[0])) if nx > 1 else 0.25
            dy = float(abs(latitude[0] - latitude[1])) if ny > 1 else 0.25

            temp_c = temp_kelvin - 273.15
            humidity = calculate_relative_humidity(temp_kelvin, dew_kelvin)
            feels_like = calculate_feels_like(temp_c, humidity, u_wind, v_wind)
            pressure_hpa = pressure_pa / 100.0

            precipitation_units = str(rain_field.attrs.get("units", "m")).lower()
            rain_mm = total_precipitation * 1000.0 if precipitation_units in {"m", "metres", "meter"} else total_precipitation
            rain_mm = np.maximum(rain_mm, 0.0)

            if np.nanmax(total_cloud) <= 1.01:
                total_cloud = total_cloud * 100.0
            total_cloud = np.clip(total_cloud, 0.0, 100.0)

            valid_time = run_time + timedelta(hours=step_hours)
            header = {
                "nx": int(nx),
                "ny": int(ny),
                "lo1": lo1,
                "la1": la1,
                "lo2": lo1 + (nx - 1) * dx,
                "la2": la1 - (ny - 1) * dy,
                "dx": dx,
                "dy": dy,
                "model": "ECMWF IFS",
                "resolution": "0.25 degree",
                "runTime": iso_z(run_time),
                "validTime": iso_z(valid_time),
                "refTime": iso_z(valid_time),
                "leadHours": int(step_hours),
                "rainAccumulation": "from-run",
                "feelsLikeMethod": FEELS_LIKE_METHOD,
                "frontMethod": None,
                "frontSource": None,
                "frontLevel": None,
                "frontValidTime": None,
            }

            step_data = {
                "meta": header,
                "wind_u": {
                    "header": {**header, "parameterCategory": 2, "parameterNumber": 2},
                    "data": clean_for_json(u_wind, 2),
                },
                "wind_v": {
                    "header": {**header, "parameterCategory": 2, "parameterNumber": 3},
                    "data": clean_for_json(v_wind, 2),
                },
                "temp": clean_for_json(temp_c, 1),
                "feels_like": clean_for_json(feels_like, 1),
                "rain": clean_for_json(rain_mm, 2),
                "press": clean_for_json(pressure_hpa, 1),
                "rh": clean_for_json(humidity, 0),
                "cloud": clean_for_json(total_cloud, 0),
                # ECMWF remains an independent surface model.  Synoptic
                # fronts are produced exclusively by the hourly ICON-2I
                # physical analysis, never copied or compared across models.
                "fronts": {"type": "FeatureCollection", "features": []},
            }

            file_name = f"step_{step_hours}.json"
            with open(os.path.join(TEMP_DIR, file_name), "w", encoding="utf-8") as output:
                json.dump(step_data, output, separators=(",", ":"), allow_nan=False)

            catalog.append(
                {
                    "file": file_name,
                    "label": valid_time.strftime("%d/%m %H:00 UTC"),
                    "hour": int(step_hours),
                    "runTime": iso_z(run_time),
                    "validTime": iso_z(valid_time),
                    "leadHours": int(step_hours),
                    "model": "ECMWF IFS",
                }
            )
        with open(os.path.join(TEMP_DIR, "catalog.json"), "w", encoding="utf-8") as output:
            json.dump(catalog, output, separators=(",", ":"), allow_nan=False)

        if os.path.exists(FINAL_DIR):
            shutil.rmtree(FINAL_DIR)
        shutil.move(TEMP_DIR, FINAL_DIR)
        print("Elaborazione ECMWF completata.", flush=True)

    except Exception as error:
        print(f"Errore elaborazione ECMWF: {error}", file=sys.stderr, flush=True)
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        sys.exit(1)
    finally:
        for dataset in datasets.values():
            try:
                dataset.close()
            except Exception:
                pass
        if os.path.exists(GRIB_FILE):
            os.remove(GRIB_FILE)


if __name__ == "__main__":
    process_data()
