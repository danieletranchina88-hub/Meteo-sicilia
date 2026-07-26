"""Memory-bounded ICON-2I field store for operational ML inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .features import GRID_BOUNDS, regular_grid


def _hours(dataset):
    if "step" not in dataset.coords:
        return [0]
    result = []
    for value in np.atleast_1d(dataset.step.values):
        if isinstance(value, np.timedelta64):
            result.append(int(value / np.timedelta64(1, "h")))
        else:
            result.append(int(value))
    return result


class Icon2IStore:
    def __init__(self, paths, run_tag, *, bounds=GRID_BOUNDS, resolution=0.20):
        self.run_tag = str(run_tag)
        self.target_latitudes, self.target_longitudes = regular_grid(
            bounds, resolution
        )
        self.datasets = {}
        self.variables = {}
        self._hours = {}
        try:
            for name, path in paths.items():
                dataset = xr.open_dataset(
                    Path(path), engine="cfgrib",
                    backend_kwargs={"indexpath": ""},
                )
                if not dataset.data_vars:
                    raise ValueError(f"{name}: GRIB privo di variabili")
                variable = next(iter(dataset.data_vars))
                self._validate(name, dataset[variable])
                self.datasets[name] = dataset
                self.variables[name] = variable
                self._hours[name] = set(_hours(dataset))
        except Exception:
            self.close()
            raise

    @staticmethod
    def _validate(name, array):
        sample = np.asarray(array.values)
        finite = sample[np.isfinite(sample)]
        if not finite.size:
            raise ValueError(f"{name}: campo non finito")
        median = float(np.nanmedian(finite))
        if name.startswith("t") and not 180.0 < median < 335.0:
            raise ValueError(f"{name}: temperatura non in kelvin")
        if name.startswith("q") and not 0.0 <= median < 0.10:
            raise ValueError(f"{name}: umidità specifica non in kg/kg")
        if name == "pmsl" and not 80_000.0 < median < 110_000.0:
            raise ValueError("PMSL non in pascal")

    def available_hours(self, name):
        return set(self._hours.get(name, set()))

    def field(self, name, hour):
        if int(hour) not in self._hours.get(name, set()):
            raise KeyError(f"{name} non disponibile a +{hour}h")
        data = self.datasets[name][self.variables[name]]
        if "step" in data.dims:
            try:
                data = data.sel(step=np.timedelta64(int(hour), "h"))
            except Exception:
                data = data.isel(step=_hours(self.datasets[name]).index(int(hour)))
        data = data.squeeze(drop=True)
        if "latitude" not in data.coords or "longitude" not in data.coords:
            raise ValueError(f"{name}: coordinate geografiche assenti")
        if data.latitude.ndim != 1 or data.longitude.ndim != 1:
            raise ValueError(f"{name}: griglia non regolare")
        if data.latitude.values[0] > data.latitude.values[-1]:
            data = data.sortby("latitude")
        if data.longitude.values[0] > data.longitude.values[-1]:
            data = data.sortby("longitude")
        interpolated = data.interp(
            latitude=xr.DataArray(self.target_latitudes, dims="latitude"),
            longitude=xr.DataArray(self.target_longitudes, dims="longitude"),
            method="linear",
        )
        values = np.asarray(interpolated.values, float)
        expected = (len(self.target_latitudes), len(self.target_longitudes))
        if values.shape != expected:
            raise ValueError(f"{name}: forma {values.shape}, attesa {expected}")
        return values

    def close(self):
        for dataset in self.datasets.values():
            try:
                dataset.close()
            except Exception:
                pass
        self.datasets.clear()
