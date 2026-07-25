import numpy as np
import xarray as xr

def calculate_ship(mucape: xr.DataArray, lapse_rate_700_500: xr.DataArray, 
                   z_0c: xr.DataArray, mix_ratio: xr.DataArray) -> xr.DataArray:
    """
    Calcola il Significant Hail Parameter (SHIP) semplificato.
    - mucape: MUCAPE (J/kg)
    - lapse_rate_700_500: Lapse rate 700-500 hPa (C/km)
    - z_0c: Altezza zero termico (m)
    - mix_ratio: Rapporto di mescolanza bassi livelli (g/kg)
    """
    # Formula empirica SHIP = (MUCAPE * mixing_ratio * lapse_rate * -500C_temp) / 42000000 
    # Usiamo una proxy scalata per semplicità basata sulle variabili fornite
    ship = (mucape * mix_ratio * lapse_rate_700_500) / 10000.0
    
    # Mascheriamo i valori dove lo zero termico è troppo basso o troppo alto
    ship = xr.where(z_0c < 1500, 0, ship)
    
    return ship

def calculate_scp(mucape: xr.DataArray, srh_0_3km: xr.DataArray, bulk_shear_0_6km: xr.DataArray) -> xr.DataArray:
    """
    Calcola il Supercell Composite Parameter (SCP).
    - mucape: MUCAPE (J/kg)
    - srh_0_3km: Storm Relative Helicity 0-3km (m^2/s^2)
    - bulk_shear_0_6km: Bulk shear 0-6km (m/s)
    """
    # SCP = (MUCAPE/1000) * (SRH/50) * (Shear/20)
    # limitiamo shear max
    shear_term = np.clip(bulk_shear_0_6km / 20.0, 0, 1.5)
    
    scp = (mucape / 1000.0) * (srh_0_3km / 50.0) * shear_term
    return scp

def evaluate_hail_threat(ship: xr.DataArray, scp: xr.DataArray, dry_air_700hpa_rh: xr.DataArray) -> xr.DataArray:
    """
    Indice di minaccia grandinigena: 0=Basso, 1=Medio, 2=Alto
    """
    threat = xr.zeros_like(ship, dtype=int)
    
    # Intrusione di aria secca (RH < 40% a 700hPa) favorisce downdraft forti
    dry_intrusion = dry_air_700hpa_rh < 40.0
    
    high_mask = (ship > 1.0) & (scp > 2.0) & dry_intrusion
    med_mask = ((ship > 0.5) | (scp > 1.0)) & (~high_mask)
    
    threat = xr.where(high_mask, 2, threat)
    threat = xr.where(med_mask, 1, threat)
    
    return threat
