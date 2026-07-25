import numpy as np
import xarray as xr

def classify_fog_type(rh_sfc: xr.DataArray, wind_10m: xr.DataArray, clct: xr.DataArray) -> xr.DataArray:
    """
    Classifica il tipo di nebbia:
    0 = Nessuna nebbia
    1 = Nebbia da radiazione
    2 = Nebbia da avvezione
    """
    fog_type = xr.zeros_like(rh_sfc, dtype=int)
    
    # Condizioni Nebbia da Radiazione: UR > 95%, Vento < 3 m/s, sereno (clct < 20%)
    radiation_fog = (rh_sfc > 95.0) & (wind_10m < 3.0) & (clct < 20.0)
    
    # Condizioni Nebbia da Avvezione: UR > 95%, Vento 2-7 m/s
    advection_fog = (rh_sfc > 95.0) & (wind_10m >= 2.0) & (wind_10m <= 7.0) & (~radiation_fog)
    
    fog_type = xr.where(radiation_fog, 1, fog_type)
    fog_type = xr.where(advection_fog, 2, fog_type)
    
    return fog_type

def estimate_visibility(rh_sfc: xr.DataArray, fog_type: xr.DataArray) -> xr.DataArray:
    """
    Stima grezza della visibilità in metri.
    """
    # Kunkel semplificata:
    # Extinction coefficient beta = a * LWC^b
    # Qui usiamo una stima molto semplice basata su RH
    
    vis = xr.full_like(rh_sfc, 10000.0)  # default 10km
    
    # Se c'è nebbia da radiazione
    vis = xr.where((fog_type == 1) & (rh_sfc > 98), 100.0, vis)
    vis = xr.where((fog_type == 1) & (rh_sfc > 95) & (rh_sfc <= 98), 500.0, vis)
    
    # Nebbia da avvezione
    vis = xr.where(fog_type == 2, 200.0, vis)
    
    return vis
