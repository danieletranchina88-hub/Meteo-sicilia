import numpy as np
import xarray as xr

def detect_foehn(u_700: xr.DataArray, v_700: xr.DataArray, rh_sfc: xr.DataArray) -> xr.DataArray:
    """
    Rileva il Foehn alpino.
    - Flusso perpendicolare alle Alpi > 15 nodi (circa 7.7 m/s) a 700 hPa.
    - UR sottovento < 40%.
    (Semplificato: si assume un flusso da Nord, v_700 < -7.7)
    """
    
    # Flusso settentrionale forte
    strong_north_wind = v_700 < -7.7
    
    # Flusso meridionale forte
    strong_south_wind = v_700 > 7.7
    
    # Aria secca al suolo
    dry_air = rh_sfc < 40.0
    
    foehn_risk = xr.zeros_like(rh_sfc, dtype=int)
    
    # Foehn nord (es. pianura padana)
    foehn_north = strong_north_wind & dry_air
    foehn_risk = xr.where(foehn_north, 1, foehn_risk)
    
    # Foehn sud (es. versante nord alpino)
    foehn_south = strong_south_wind & dry_air
    foehn_risk = xr.where(foehn_south, 2, foehn_risk)
    
    return foehn_risk
