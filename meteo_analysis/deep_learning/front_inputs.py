"""Multi-level, unit-explicit tensor assembly for FrontUNet."""

from __future__ import annotations

import numpy as np

from meteo_analysis.ml.features import _thermodynamics, compute_feature_frame

from .schemas import FRONT_FEATURES

# Davies-Jones (2008), identical to scripts/thermodynamics.py.
_A = (7.101574, -20.68208, 16.11182, 2.574631, -5.205688)
_B = (1.0, -3.552497, 3.781782, -0.6899655, -0.5929340)


def wet_bulb_potential_temperature(theta_e_k):
    theta_e = np.asarray(theta_e_k, dtype=np.float64)
    x = theta_e / 273.15
    numerator = _A[0] + x * (_A[1] + x * (_A[2] + x * (_A[3] + x * _A[4])))
    denominator = _B[0] + x * (_B[1] + x * (_B[2] + x * (_B[3] + x * _B[4])))
    with np.errstate(over="ignore", invalid="ignore"):
        theta_w = theta_e - np.exp(numerator / denominator)
    return np.where(theta_e <= 173.15, theta_e, theta_w)


def build_front_tensor(
    fields,
    latitude,
    longitude,
    *,
    valid_time,
    previous_pmsl_3h=None,
    gradient_history=None,
):
    """Return ``[C,H,W]`` features; unavailable optional levels remain NaN."""
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    shape = (latitude.size, longitude.size)
    arrays = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in fields.items() if value is not None
    }
    base = compute_feature_frame(
        arrays, latitude, longitude, valid_time=valid_time,
        previous_pmsl_3h=previous_pmsl_3h,
        gradient_history=gradient_history,
    )
    channels = {
        name: base[name].to_numpy(dtype=np.float32).reshape(shape)
        for name in FRONT_FEATURES if name in base.columns
    }

    def thermo(level):
        temperature = arrays.get(f"t{level}")
        humidity = arrays.get(f"q{level}")
        if temperature is None or humidity is None:
            return None
        return _thermodynamics(float(level), temperature, humidity)

    for level in (925, 850, 700):
        values = thermo(level)
        if values is not None:
            channels[f"theta_w_{level}_k"] = wet_bulb_potential_temperature(
                values["theta_e"]
            )
            if level != 850:
                channels[f"theta_e_{level}_k"] = values["theta_e"]
    direct = {
        "u925_m_s": "u925", "v925_m_s": "v925",
        "u850_m_s": "u850", "v850_m_s": "v850",
        "u700_m_s": "u700", "v700_m_s": "v700",
        "u10_m_s": "u10", "v10_m_s": "v10",
        "omega700_pa_s": "omega700",
    }
    for channel, field in direct.items():
        if field in arrays:
            channels[channel] = arrays[field]
    channels["pmsl_hpa"] = arrays["pmsl"] / 100.0

    result = []
    for name in FRONT_FEATURES:
        value = np.asarray(
            channels.get(name, np.full(shape, np.nan)), dtype=np.float32
        )
        if value.shape != shape:
            raise ValueError(f"{name}: forma {value.shape}, attesa {shape}")
        result.append(value)
    return np.stack(result)
