"""Small, validated loader for run-wide ICON-2I diagnostic GRIB files."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class FieldSpec:
    minimum: float
    maximum: float
    units: str


FIELD_SPECS = {
    "cape_ml": FieldSpec(-1.0, 12_000.0, "J kg-1"),
    # CIN is normalised later because providers use both signed and magnitude
    # conventions.
    "cin_ml": FieldSpec(-8_000.0, 8_000.0, "J kg-1"),
}


def _step_hours(dataset: xr.Dataset) -> list[int]:
    if "step" not in dataset.coords:
        return [0]
    result = []
    for value in np.atleast_1d(dataset.step.values):
        if isinstance(value, np.timedelta64):
            result.append(int(value / np.timedelta64(1, "h")))
        else:
            result.append(int(value))
    return result


class IconRunFields:
    """Open a set of single-variable run files and interpolate them on demand."""

    def __init__(self, paths: dict[str, str]) -> None:
        self.datasets: dict[str, xr.Dataset] = {}
        self.variables: dict[str, str] = {}
        self.hours: dict[str, set[int]] = {}
        try:
            for name, path in paths.items():
                dataset = xr.open_dataset(
                    path,
                    engine="cfgrib",
                    backend_kwargs={"indexpath": ""},
                )
                if not dataset.data_vars:
                    dataset.close()
                    raise ValueError(f"{name}: GRIB senza variabili")
                variable = next(iter(dataset.data_vars))
                self._validate(name, dataset[variable])
                self.datasets[name] = dataset
                self.variables[name] = variable
                self.hours[name] = set(_step_hours(dataset))
        except Exception:
            self.close()
            raise

    def _validate(self, name: str, data: xr.DataArray) -> None:
        spec = FIELD_SPECS.get(name)
        sample = data
        if "step" in sample.dims and sample.sizes["step"] > 3:
            last = sample.sizes["step"] - 1
            sample = sample.isel(step=[0, last // 2, last])
        for dimension in ("latitude", "longitude"):
            if dimension in sample.dims and sample.sizes[dimension] > 80:
                stride = max(1, sample.sizes[dimension] // 80)
                sample = sample.isel({dimension: slice(None, None, stride)})
        values = np.asarray(sample.values, dtype=float)
        finite = values[np.isfinite(values)]
        if not finite.size:
            raise ValueError(f"{name}: campo interamente non finito")
        if spec is None:
            return
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        if minimum < spec.minimum or maximum > spec.maximum:
            raise ValueError(
                f"{name}: valori fuori scala [{minimum:.1f}, {maximum:.1f}] "
                f"{spec.units}"
            )

    @property
    def available_hours(self) -> list[int]:
        if not self.hours:
            return []
        return sorted(set.intersection(*self.hours.values()))

    def field(
        self,
        name: str,
        hour: int,
        target_latitudes,
        target_longitudes,
    ) -> np.ndarray | None:
        dataset = self.datasets.get(name)
        if dataset is None or hour not in self.hours.get(name, set()):
            return None
        data = dataset[self.variables[name]]
        if "step" in data.dims:
            step = np.timedelta64(int(hour), "h")
            try:
                data = data.sel(step=step)
            except Exception:
                index = _step_hours(dataset).index(int(hour))
                data = data.isel(step=index)
        data = data.squeeze(drop=True)
        if "latitude" not in data.coords or "longitude" not in data.coords:
            raise ValueError(f"{name}: coordinate latitude/longitude assenti")

        latitude = np.asarray(data.latitude.values, dtype=float).squeeze()
        longitude = np.asarray(data.longitude.values, dtype=float).squeeze()
        if latitude.ndim != 1 or longitude.ndim != 1:
            raise ValueError(f"{name}: griglia non regolare latitude/longitude")
        if latitude[0] > latitude[-1]:
            data = data.sortby("latitude")
        if longitude[0] > longitude[-1]:
            data = data.sortby("longitude")

        target_lat = np.asarray(target_latitudes, dtype=float)
        target_lon = np.asarray(target_longitudes, dtype=float)
        interpolated = data.interp(
            latitude=xr.DataArray(target_lat, dims=("latitude",)),
            longitude=xr.DataArray(target_lon, dims=("longitude",)),
            method="linear",
        )
        values = np.asarray(interpolated.values, dtype=float)
        expected = (target_lat.size, target_lon.size)
        if values.shape != expected:
            raise ValueError(
                f"{name}: forma interpolata {values.shape}, attesa {expected}"
            )
        return values

    def close(self) -> None:
        for dataset in self.datasets.values():
            try:
                dataset.close()
            except Exception:
                pass
        self.datasets.clear()
        self.variables.clear()
        self.hours.clear()
