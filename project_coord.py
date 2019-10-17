#!/usr/bin/env python3
# ----------------------------------
# project_coord.py
#
# Description: Coordinate projection
# 
# ----------------------------------
# Author: Stefano Bertone
# Created: 22-Oct-2018

import numpy as np


def sind(x):
    return np.sin(np.deg2rad(x))


def cosd(x):
    return np.cos(np.deg2rad(x))


def project_stereographic(lon, lat, lon0, lat0, R=1):
    """
    project cylindrical coordinates to stereographic xy from central lon0/lat0
    :param lon: array of input longitudes
    :param lat: array of input latitudes
    :param lon0: center longitude for the projection
    :param lat0: center latitude for the projection
    :param R: planetary radius (km)
    :return: stereographic projection xy coord from center (km)
    """
    k = (2 * R) / (1 + sind(lat0) * sind(lat) + cosd(lat0) * cosd(lat) * cosd(lon - lon0))
    x = k * cosd(lat) * sind(lon - lon0)
    y = k * (cosd(lat0) * sind(lat) - sind(lat0) * cosd(lat) * cosd(lon - lon0))

    return x, y
