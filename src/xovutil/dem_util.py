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
import time
import logging

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline

from xovutil import pickleIO
# from examples.MLA.options import XovOpt.get("tmpdir"), XovOpt.get("debug")
from config import XovOpt

import pandas as pd


# TODO retrieve outfil = dem_interp_path as it was for LOLA (or just switch LOLA to other routines)
def import_dem(filein, outdir=''):
    # open netCDF file
    # nc_file = "/home/sberton2/Downloads/sresa1b_ncar_ccsm3-example.nc"
    nc_file = filein
    dem_xarr = xr.open_dataset(nc_file)

    # print(filein)

    try:
        lats = np.deg2rad(dem_xarr.lat.values) + np.pi / 2.
        lons = np.deg2rad(dem_xarr.lon.values)  # -np.pi
    except:
        lats = np.deg2rad(dem_xarr.y.values) + np.pi / 2.
        lons = np.deg2rad(dem_xarr.x.values)  # -np.pi
    data = dem_xarr.z.values

    # Exclude last column because only 0<=lat<pi
    # and 0<=lon<pi are accepted (checked that lon=0 has same values)
    # print(data[:,0]==data[:,-1])
    # kx=ky=1 required not to mess up results!!!!!!!!!! Higher interp orders mess up...

    if outdir == '':
        dem_interp_path = XovOpt.get("auxdir") + "interp_dem.pkl"
    else:
        dem_interp_path = f"{outdir}interp_dem.pkl"

    if not os.path.exists(dem_interp_path):
        interp_spline = RectBivariateSpline(lats[:-1],
                                            lons[:-1],
                                            data[:-1, :-1], kx=1, ky=1)
        pickleIO.save(interp_spline, dem_interp_path)
        logging.info(f"Interpolated DEM saved to {dem_interp_path}.")
    else:
        logging.info(f"### Just read, interp_map exists in {dem_interp_path}")
        interp_spline = pickleIO.load(dem_interp_path)

    return interp_spline


def get_demz_at(dem_xarr, lattmp, lontmp):
    # lontmp += 180.
    lontmp[lontmp < 0] += 360.


    return dem_xarr.ev(np.deg2rad(lattmp) + np.pi / 2., np.deg2rad(lontmp))


def get_demz_diff_at(dem_xarr, lattmp, lontmp, axis='lon'):
    lontmp[lontmp < 0] += 360.
    diff_dem_xarr = dem_xarr.differentiate(axis)

    lat_ax = xr.DataArray(lattmp, dims='z')
    lon_ax = xr.DataArray(lontmp, dims='z')

    return diff_dem_xarr.interp(lat=lat_ax, lon=lon_ax).z.to_dataframe().loc[:, 'z'].values


def get_demz_tiff(filin, lon, lat):
    import pyproj

    # Read the data
    da = xr.open_dataset(filin, engine='rasterio')
    # Rasterio works with 1D arrays (but still need to pass whole mesh, flattened)
    # convert lon/lat to xy using intrinsic crs, then generate additional dimension for
    # advanced xarray interpolation
    # print(da.crs)
    p = pyproj.Proj(da.rio.crs)
    xi, yi = p(lon, lat, inverse=False)

    xi = xr.DataArray(xi, dims="z")
    yi = xr.DataArray(yi, dims="z")

    if XovOpt.get("debug"):
        print(da)
        print("x,y len:", len(xi), len(yi))

    # interpolate & extrapolate from dem at ladata xy
    da_interp = da.interp(x=xi, y=yi, kwargs={"fill_value": None})

    return da_interp.band_data.data * 1.e-3  # convert to km for compatibility with grd


def get_demz_grd(filin, lon, lat):
    da = xr.open_dataset(filin)
    # for LDAM_8, rename coordinates
    # da = da.rename({'x':'lon','y':'lat'})

    lon[lon < 0] += 360.
    lon = xr.DataArray(lon, dims="x")
    lat = xr.DataArray(lat, dims="x")

    if XovOpt.get("debug"):
        print(da)
        print("lon,lat len:", len(lon), len(lat))

    da_interp = da.interp(lon=lon, lat=lat)

    return da_interp.z.values


if __name__ == '__main__':

    start = time.time()

    method = 'pygmt'  # 'xarray' #
    number_of_samples = 100000
    filin = '/home/sberton2/tmp/mercury_SP/interp_craters/SP8_with_SP9mask_250km_interpolated_v2s.grd'  # '/home/sberton2/tmp/LDAM_8.GRD' # '/home/sberton2/Works/NASA/Mercury_tides/aux/HDEM_64.GRD' #

    rng = np.random.default_rng()
    lon = rng.random((number_of_samples)) * 360.  # if grd lon is [0,360)
    lat = (rng.random((number_of_samples)) * 2 - 1) * 90.  # if grd lat is [-90,90)

    if method == 'xarray':
        import xarray as xr
        import pyproj

        if filin.split('.')[-1] in ['grd', 'GRD']:  # to read grd/netcdf files
            z = get_demz_grd(filin, lon, lat)
        elif filin.split('.')[-1] in ['tif', 'TIF']:  # to read geotiffs usgs
            z = np.squeeze(get_demz_tiff(filin, lon, lat))

        out = pd.DataFrame([lon, lat, z], index=['lon', 'lat', 'z']).T

    elif method == 'pygmt':
        import \
            pygmt  # needs GMT 6.1.1 installed, plus linking of GMTdir/lib64/libgmt.so to some general xovutil dir (see bottom of https://www.pygmt.org/dev/install.html)

        points = pd.DataFrame([lon, lat], index=['lon', 'lat']).T
        out = pygmt.grdtrack(points, filin, newcolname='z')

    print(out)

    end = time.time()
    print("## Reading/interpolation of", number_of_samples, "samples finished after", str(np.round(end - start, 2)),
          "sec!")
    exit()
