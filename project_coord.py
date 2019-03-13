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
    k = (2 * R) / (1 + sind(lat0) * sind(lat) + cosd(lat0) * cosd(lat) * cosd(lon - lon0))
    x = k * cosd(lat) * sind(lon - lon0)
    y = k * (cosd(lat0) * sind(lat) - sind(lat0) * cosd(lat) * cosd(lon - lon0))

    return x, y
