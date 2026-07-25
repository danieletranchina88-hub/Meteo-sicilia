import numpy as np
import xarray as xr

def detect_freezing_rain(t_925: xr.DataArray, t_850: xr.DataArray, t_700: xr.DataArray, t_2m: xr.DataArray, precip_rate: xr.DataArray) -> xr.DataArray:
    """
    Rileva il rischio di gelicidio (freezing rain) usando il profilo termico e le precipitazioni.
    """
    # 1. Trova il warm nose: temperatura > 0°C (273.15 K) in quota
    warm_nose = (t_925 > 273.15) | (t_850 > 273.15) | (t_700 > 273.15)
    
    # 2. Temperatura al suolo sotto zero
    subfreezing_sfc = t_2m < 273.15
    
    # 3. Precipitazione in atto
    precip_active = precip_rate > 0.1  # mm/h
    
    # Maschera booleana del rischio gelicidio
    risk_mask = warm_nose & subfreezing_sfc & precip_active
    
    risk = xr.where(risk_mask, 1, 0)
    risk.attrs['description'] = 'Freezing Rain Risk'
    return risk
