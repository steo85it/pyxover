#!/usr/bin/env python3
# ----------------------------------
# dem_util.py
#
# Description: import and interpolate digital elevation maps
#
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 16-Aug-2019
import os

import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline

import pickleIO
from prOpt import tmpdir


def import_dem(filein):
    # grid point lists

    # open netCDF file
    # nc_file = "/home/sberton2/Downloads/sresa1b_ncar_ccsm3-example.nc"
    nc_file = filein
    dem_xarr = xr.open_dataset(nc_file)

    lats = np.deg2rad(dem_xarr.lat.values)+np.pi/2.
    lons = np.deg2rad(dem_xarr.lon.values)#-np.pi
    data = dem_xarr.z.values

    # Exclude last column because only 0<=lat<pi
    # and 0<=lon<pi are accepted (checked that lon=0 has same values)
    # print(data[:,0]==data[:,-1])
    # kx=ky=1 required not to mess up results!!!!!!!!!! Higher interp orders mess up...
    dem_interp_path = tmpdir+"interp_dem.pkl"
    if not os.path.exists(dem_interp_path):
        interp_spline = RectBivariateSpline(lats[:-1],
                                            lons[:-1],
                                            data[:-1,:-1], kx=1, ky=1)
        pickleIO.save(interp_spline, dem_interp_path)
    else:
        interp_spline = pickleIO.load(dem_interp_path)

    return interp_spline


def get_demz_at(dem_xarr, lattmp, lontmp):
    # lontmp += 180.
    lontmp[lontmp < 0] += 360.
    # print("eval")
    # print(np.sort(lattmp))
    # print(np.sort(lontmp))
    # print(np.sort(np.deg2rad(lontmp)))
    # exit()

    return dem_xarr.ev(np.deg2rad(lattmp)+np.pi/2., np.deg2rad(lontmp))


def get_demz_diff_at(dem_xarr, lattmp, lontmp, axis='lon'):
    lontmp[lontmp < 0] += 360.
    # print(dem_xarr)
    diff_dem_xarr = dem_xarr.differentiate(axis)

    lat_ax = xr.DataArray(lattmp, dims='z')
    lon_ax = xr.DataArray(lontmp, dims='z')

    return diff_dem_xarr.interp(lat=lat_ax, lon=lon_ax).z.to_dataframe().loc[:, 'z'].values