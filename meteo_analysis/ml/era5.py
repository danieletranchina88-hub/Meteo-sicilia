"""Analysis-ready ERA5 reader used only for offline supervised training."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import xarray as xr

from .features import GRID_BOUNDS, regular_grid


class EarthmoverERA5:
    """Read public, anonymous Icechunk ERA5 without downloading global files."""

    def __init__(self, *, bounds=GRID_BOUNDS, resolution=0.20):
        import icechunk

        storage = icechunk.s3_storage(
            bucket="earthmover-icechunk-era5",
            prefix="icechunkV2",
            region="us-east-1",
            anonymous=True,
        )
        repository = icechunk.Repository.open(storage)
        self.session = repository.readonly_session("main")
        self.single = xr.open_zarr(
            # One regional map at a time: the spatial chunk layout avoids
            # decoding long time-series chunks and keeps RAM bounded.
            self.session.store, group="single/spatial",
            consolidated=False, chunks=None,
        )
        self.pressure = xr.open_zarr(
            self.session.store, group="pressure/spatial",
            consolidated=False, chunks=None,
        )
        self.target_latitudes, self.target_longitudes = regular_grid(
            bounds, resolution
        )
        self.bounds = tuple(map(float, bounds))

    def _regional(self, array, timestamp):
        west, east, south, north = self.bounds
        timestamp = np.datetime64(pd.Timestamp(timestamp).tz_localize(None), "ns")
        latitude = np.asarray(array.latitude.values)
        lat_slice = slice(north + 1.0, south - 1.0)
        if latitude[0] < latitude[-1]:
            lat_slice = slice(south - 1.0, north + 1.0)
        data = array.sel(
            valid_time=timestamp,
            latitude=lat_slice,
            longitude=slice(west - 1.0, east + 1.0),
        ).load()
        if data.latitude.values[0] > data.latitude.values[-1]:
            data = data.sortby("latitude")
        result = data.interp(
            latitude=xr.DataArray(self.target_latitudes, dims="latitude"),
            longitude=xr.DataArray(self.target_longitudes, dims="longitude"),
            method="linear",
        )
        return np.asarray(result.values, float)

    def fields(self, timestamp):
        pressure_specs = {
            "t850": ("t", 850), "q850": ("q", 850),
            "u850": ("u", 850), "v850": ("v", 850),
            "t700": ("t", 700), "q700": ("q", 700),
            "u500": ("u", 500), "v500": ("v", 500),
            "fi500": ("z", 500),
        }
        surface_specs = {"u10": "u10", "v10": "v10", "pmsl": "msl"}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                name: executor.submit(
                    self.pressure_field, variable, level, timestamp
                )
                for name, (variable, level) in pressure_specs.items()
            }
            futures.update({
                name: executor.submit(self.single_field, variable, timestamp)
                for name, variable in surface_specs.items()
            })
            result = {name: future.result() for name, future in futures.items()}
        result.update({
            # Not part of the ERA5-transfer feature profile.
            "wshear_u_0_6km": np.zeros((
                self.target_latitudes.size, self.target_longitudes.size
            )),
            "wshear_v_0_6km": np.zeros((
                self.target_latitudes.size, self.target_longitudes.size
            )),
            "hsurf": np.zeros((
                self.target_latitudes.size, self.target_longitudes.size
            )),
            "ruggedness_10km": np.zeros((
                self.target_latitudes.size, self.target_longitudes.size
            )),
        })
        return result

    def pressure_field(self, name, level, timestamp):
        return self._regional(
            self.pressure[name].sel(pressure_level=int(level)), timestamp
        )

    def single_field(self, name, timestamp):
        return self._regional(self.single[name], timestamp)

    def close(self):
        self.single.close()
        self.pressure.close()
