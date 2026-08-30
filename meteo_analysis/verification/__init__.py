"""Forecast/observation archive and verification primitives.

Modules are deliberately not imported eagerly: the hourly METAR collector only
needs ``requests`` and must not pull NumPy or the NWP processing stack merely by
loading the package.
"""
