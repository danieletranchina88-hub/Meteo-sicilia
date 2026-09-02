"""Physics-aware PyTorch models for offline meteorological research.

The package is deliberately not imported by the operational processor.  A
checkpoint must pass independent validation before an explicit deployment
step can make it visible to the website.
"""

from .schemas import (
    DOWNSCALING_OUTPUTS,
    FRONT_CLASSES,
    FRONT_FEATURES,
    STATIC_DOWNSCALING_FEATURES,
)

__all__ = [
    "DOWNSCALING_OUTPUTS",
    "FRONT_CLASSES",
    "FRONT_FEATURES",
    "STATIC_DOWNSCALING_FEATURES",
]
