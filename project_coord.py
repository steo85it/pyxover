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
    :param lon: array of input longitudes (deg)
    :param lat: array of input latitudes (deg)
    :param lon0: center longitude for the projection (deg)
    :param lat0: center latitude for the projection (deg)
    :param R: planetary radius (km)
    :return: stereographic projection xy coord from center (km)
    """
    #
    # print(lon, lat, lon0, lat0)
    # exit()

    cosd_lat = cosd(lat)
    cosd_lon_lon0 = cosd(lon - lon0)
    sind_lat = sind(lat)

    k = (2. * R) / (1. + sind(lat0) * sind_lat + cosd(lat0) * cosd_lat * cosd_lon_lon0)
    x = k * cosd_lat * sind(lon - lon0)
    y = k * (cosd(lat0) * sind_lat - sind(lat0) * cosd_lat * cosd_lon_lon0)

    return x, y
