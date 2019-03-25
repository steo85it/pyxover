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


def sind(x):
    return np.sin(np.deg2rad(x))


def cosd(x):
    return np.cos(np.deg2rad(x))


def unproject_stereographic(x, y, lon0, lat0, R=1):
    #    x = np.concatenate(x).ravel()
    #    y = np.concatenate(y).ravel()
    rho = np.sqrt(np.power(x, 2) + np.power(y, 2))
    c = 2 * np.arctan2(rho, 2 * R)

    lat = np.rad2deg(np.arcsin(np.cos(c) * sind(lat0) + (cosd(lat0) * y * np.sin(c)) / rho))
    lon = np.mod(
        lon0 + np.rad2deg(np.arctan2(x * np.sin(c), cosd(lat0) * rho * np.cos(c) - sind(lat0) * y * np.sin(c))), 360)

    if (x == 0).any() and (y == 0).any():
        #    if x == 0 and y == 0:
        return lon0, lat0
    else:
        return lon, lat
