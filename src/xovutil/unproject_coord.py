"""Inverse stereographic projection helpers matching :mod:`project_coord`."""

#!/usr/bin/env python3
# ----------------------------------
# unproject_coord.py
#
# Description: Stereographic Coordinate unprojection
#
# Remark: inverse transformation in project_coord.py
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 30-Jan-2019

import numpy as np


def sind(x: np.ndarray | float) -> np.ndarray:
    """Return the sine of angles expressed in degrees."""

    return np.sin(np.deg2rad(x))


def cosd(x: np.ndarray | float) -> np.ndarray:
    """Return the cosine of angles expressed in degrees."""

    return np.cos(np.deg2rad(x))


def unproject_stereographic(
    x: np.ndarray,
    y: np.ndarray,
    lon0: float,
    lat0: float,
    R: float = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert stereographic ``x``/``y`` coordinates back to longitude/latitude.

    Parameters
    ----------
    x : numpy.ndarray
        Projected x coordinates in kilometers.
    y : numpy.ndarray
        Projected y coordinates in kilometers.
    lon0 : float
        Central longitude of the stereographic projection in degrees.
    lat0 : float
        Central latitude of the stereographic projection in degrees.
    R : float, optional
        Planetary radius in kilometers, by default ``1``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Longitude and latitude arrays in degrees matching the input grid.
    """

    rho = np.sqrt(np.power(x, 2) + np.power(y, 2))
    c = 2 * np.arctan2(rho, 2 * R)

    lat = np.rad2deg(np.arcsin(np.cos(c) * sind(lat0) + (cosd(lat0) * y * np.sin(c)) / rho))
    lon = np.mod(
        lon0 + np.rad2deg(np.arctan2(x * np.sin(c), cosd(lat0) * rho * np.cos(c) - sind(lat0) * y * np.sin(c))), 360)

    if (x == 0).any() and (y == 0).any():
        return lon0, lat0
    else:
        return lon, lat
