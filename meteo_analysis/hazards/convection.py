import numpy as np
import xarray as xr

def calculate_convection_probability(mucape: xr.DataArray, cin: xr.DataArray, convergence_10m: xr.DataArray, front_distance_km: xr.DataArray) -> xr.DataArray:
    """
    Calcola la probabilità di innesco convettivo basata su:
    - MUCAPE (J/kg)
    - CIN (J/kg)
    - Convergenza del vento a 10m (s^-1)
    - Distanza dal fronte più vicino (km)
    
    Regola:
    MUCAPE > 800 AND CIN > -50 AND convergenza > 1e-4 AND distanza_fronte < 50 km -> Alta probabilità
    """
    
    # Inizializziamo a zero (bassa probabilità)
    prob = xr.zeros_like(mucape, dtype=float)
    
    # Condizioni per alta probabilità (>70%)
    high_prob_mask = (mucape > 800) & (cin > -50) & (convergence_10m > 1e-4) & (front_distance_km < 50)
    
    # Condizioni per media probabilità (rilassamento parziale)
    medium_prob_mask = (mucape > 400) & (cin > -100) & (convergence_10m > 0.5e-4) & (~high_prob_mask)
    
    prob = xr.where(high_prob_mask, 0.80, prob)
    prob = xr.where(medium_prob_mask, 0.40, prob)
    
    prob.attrs['description'] = 'Convective Initiation Probability'
    prob.attrs['units'] = 'probability'
    
    return prob
