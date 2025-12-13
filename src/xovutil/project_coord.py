"""Coordinate projection helpers for stereographic mapping."""

# ----------------------------------
# project_coord.py
#
# Description: Coordinate projection
#
# ----------------------------------
# Author: Stefano Bertone
# Created: 22-Oct-2018

import numpy as np


def sind(x: np.ndarray | float) -> np.ndarray:
    """Return the sine of angles expressed in degrees."""

    return np.sin(np.deg2rad(x))


def cosd(x: np.ndarray | float) -> np.ndarray:
    """Return the cosine of angles expressed in degrees."""

    return np.cos(np.deg2rad(x))


def project_stereographic(
    lon: np.ndarray,
    lat: np.ndarray,
    lon0: float,
    lat0: float,
    R: float = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Project geographic coordinates to a stereographic plane.

    Parameters
    ----------
    lon : numpy.ndarray
        Input longitudes in degrees.
    lat : numpy.ndarray
        Input latitudes in degrees.
    lon0 : float
        Central longitude of the projection in degrees.
    lat0 : float
        Central latitude of the projection in degrees.
    R : float, optional
        Planetary radius in kilometers, by default ``1``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Projected ``x`` and ``y`` coordinates in kilometers.
    """

    cosd_lat = cosd(lat)
    cosd_lon_lon0 = cosd(lon - lon0)
    sind_lat = sind(lat)

    k = (2. * R) / (1. + sind(lat0) * sind_lat + cosd(lat0) * cosd_lat * cosd_lon_lon0)
    x = k * cosd_lat * sind(lon - lon0)
    y = k * (cosd(lat0) * sind_lat - sind(lat0) * cosd_lat * cosd_lon_lon0)

    return x, y
