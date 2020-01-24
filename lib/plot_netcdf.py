#!/usr/bin/env python3
# ----------------------------------
# Read DEM, GRD, IMG tests
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-Oct-2018
#
import subprocess

import matplotlib.pyplot as plt
#from planetaryimage import PDS3Image
# from osgeo import gdal
# import rasterio
import pylab as py
import xarray as xr
import numpy as np
from scipy.interpolate import RectBivariateSpline

import pickleIO
from prOpt import auxdir, tmpdir


def import_dem(filein):
    # grid point lists
    from hillshade import hill_shade


    # open netCDF file
    # nc_file = "/home/sberton2/Downloads/sresa1b_ncar_ccsm3-example.nc"
    # nc_file = filein
    # dem_xarr = xr.open_dataset('/home/sberton2/Works/NASA/Mercury_tides/aux/gaussii.grd')
    dem_xarr = xr.open_dataset(tmpdir+filein)

    # fig, [ax0, ax1] = plt.subplots(2, 1)
    #
    # dem_xarr.to_array().plot(ax=ax0)

    # print(dem_xarr)
    # dem_xarr.coords['x'] = dem_xarr.coords['x'].values/dem_xarr.coords['x'].max().values*0.25
    # dem_xarr.coords['y'] = dem_xarr.coords['y'].values/dem_xarr.coords['y'].max().values*0.25
    #
    # print(dem_xarr.z.std().values)
    # print(dem_xarr.z.mean().values)
    # print(dem_xarr.z)
    #
    # dem_xarr.__setitem__('z', (dem_xarr.z - dem_xarr.z.mean().values) / dem_xarr.z.std().values)
    # dem_xarr.__setitem__('z', (dem_xarr.z / dem_xarr.z.max().values))
    # # dem_xarr = dem_xarr.to_dataset(dim='z')
    # print(dem_xarr.z.std())
    # print(dem_xarr.z.mean())

    print(dem_xarr)

    fig, ax1 = plt.subplots(1, 1)

    dem = dem_xarr.to_array()
    # print(dem)
    dem = hill_shade(dem.data[0],terrain=dem.data[0]*5)
    plt.imshow(dem)
    #dem.plot(ax=ax1)
    #ax1.set_title('RMS of residuals (m):')

    fig.savefig(
        tmpdir + filein.split('.')[0]+'.png',dpi=2000)
    # exit()
    #
    # fig, ax1 = plt.subplots(1, 1)
    #
    # dem_xarr.differentiate('x').to_array().plot(ax=ax1,vmin=-500,vmax=500)
    # # ax1.set_title('RMS of residuals (m):')
    #
    # fig.savefig(
    #     tmpdir + filein.split('.')[0]+'_dx.png',dpi=2000)

    # exit()
    return dem_xarr




    # lats = np.deg2rad(dem_xarr.lat.values)+np.pi/2.
    # lons = np.deg2rad(dem_xarr.lon.values)#-np.pi
    # data = dem_xarr.z.values
    #
    # # Exclude last column because only 0<=lat<pi
    # # and 0<=lon<pi are accepted (checked that lon=0 has same values)
    # # print(data[:,0]==data[:,-1])
    # # kx=ky=1 required not to mess up results!!!!!!!!!! Higher interp orders mess up...
    # if True:
    #     interp_spline = RectBivariateSpline(lats[:-1],
    #                                         lons[:-1],
    #                                         data[:-1,:-1], kx=1, ky=1)
    #     pickleIO.save(interp_spline,auxdir+"interp_dem.pkl" )
    # else:
    #     interp_spline = pickleIO.load(auxdir+"interp_dem.pkl")
    #
    # return interp_spline

if __name__ == '__main__':

    # r_dem = subprocess.check_output('mapproject', 'ladata_concat_KX1r2_18.txt', '-Js0/90/1:100000 -R0/360/75/90 -S -C -bo | blockmedian -bi3 -C -r -I.05 -R-630/630/-630/630 -bo3 | surface -bi3 -T0.75 -r -I.05 -R-630/630/-630/630 -Gtopo_KX1r2_18.grd
    #     ['grdtrack', gmt_in, '-G' + dem],
    #     universal_newlines=True, cwd='tmp')
    #
    # _ = import_dem('/home/sberton2/tmp/run-DEM-final.grd') #tmpdir+'outfile.grd')
    dem_A = import_dem('topo_AG_AC_KX1r2_0.grd')
    exit()
    dem_B = import_dem('topo_AG_AC_KX1r2_0.grd')

    dem_diff = dem_A-dem_B

    fig, ax1 = plt.subplots(1, 1)

    dem_diff.to_array().plot(ax=ax1,vmin=-1000,vmax=1000)
    # ax1.set_title('RMS of residuals (m):')

    fig.savefig(
        tmpdir + 'dem_diff.png',dpi=2000)

    exit()
