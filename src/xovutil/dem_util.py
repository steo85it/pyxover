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
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline

from xovutil import pickleIO
from config import XovOpt

import pandas as pd


# TODO retrieve outfil = dem_interp_path as it was for LOLA (or just switch LOLA to other routines)
def import_dem(filein, outdir=''):
    # open netCDF file
    # nc_file = "/home/sberton2/Downloads/sresa1b_ncar_ccsm3-example.nc"
    nc_file = filein
    dem_xarr = xr.open_dataset(nc_file)

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

def get_demslope_tiff(filin, lon, lat):
    import pyproj
    from xrspatial import slope

    # Read the data
    da = xr.open_dataarray(filin, engine='rasterio')  
    slope_da = slope(da.squeeze())
    # Rasterio works with 1D arrays (but still need to pass whole mesh, flattened)
    # convert lon/lat to xy using intrinsic crs, then generate additional dimension for
    # advanced xarray interpolation
    p = pyproj.Proj(da.rio.crs)
    xi, yi = p(lon, lat, inverse=False)

    xi = xr.DataArray(xi, dims="z")
    yi = xr.DataArray(yi, dims="z")

    # interpolate & extrapolate from dem at ladata xy
    slope_interp = slope_da.interp(x=xi, y=yi, kwargs={"fill_value": None})

    return slope_interp.data

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


def _get_geotiff_masks(df):
    geotiff = {'global': f'{XovOpt.get("auxdir")}dem/Mercury_Messenger_USGS_DEM_Global_665m_v2.tif',
               'NP': f'{XovOpt.get("auxdir")}dem/Mercury_Messenger_USGS_DEM_NPole_665m_v2_32bit.tif',
               'SP': f'{XovOpt.get("auxdir")}dem/Mercury_Messenger_USGS_DEM_SPole_665m_v2_32bit.tif'}

    masks = {'NP': (df['LAT'] >= 70),# NP (MLA DEM)
             'global': (df['LAT'] < 70) & (df['LAT'] > -70), # EQUAT (USGS)
             'SP': (df['LAT'] <= -70)} # SP (USGS)
    return geotiff, masks


def get_topoelev(track, lattmp, lontmp):

    if XovOpt.get("apply_topo"):
        # st = time.time()
        
        # if gmt==False don't use grdtrack, but interpolate once using xarray and store interp
        gmt = False

        if XovOpt.get("instrument") in ['BELA', 'CALA', 'MLA']:
            df = pd.DataFrame(zip(lattmp, lontmp), columns=['LAT', 'LON'])  # .reset_index()
            # nice but not broadcasted... slow
            # df['r_dem'] = df.apply(lambda x: get_demz_tiff(geotiff[0],lat=x.LAT,lon=x.LON) if x.LAT > 30
            #                         else get_demz_grd(filin=dem,lon=x.LON,lat=x.LAT), axis=1)

            geotiff, masks = _get_geotiff_masks(df)

            df['r_dem'] = 0
            for name, mask in masks.items():
                if len(df.loc[mask, :]) > 0:
                    df.loc[mask, 'r_dem'] = np.squeeze(get_demz_tiff(filin=geotiff[name],
                                                                     lon=df.loc[mask, 'LON'].values,
                                                                     lat=df.loc[mask, 'LAT'].values).T)

            r_dem = df.r_dem.values
            if np.isnan(r_dem).any():
                print("r_dem is nan")

        elif (not gmt) and (XovOpt.get("instrument") == 'LOLA'):

            if track.dem is None:
                if not XovOpt.get("local"):
                    dem_path = track.slewdir + "/SLDEM2015_512PPD.GRD"
                else:
                    dem_path = XovOpt.get("auxdir") + 'HDEM_64.GRD'  # ''MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD'

                track.dem = import_dem(filein=dem_path, outdir=f"{track.slewdir}/")
            else:
                logging.info("DEM already read")
                pass
        else:
            print("Using grdtrack")

        # GMT case not really used
        if gmt and XovOpt.get("instrument") == 'LOLA':
            gmt_in = 'gmt_' + track.name + '.in'
            if os.path.exists('tmp/' + gmt_in):
                os.remove('tmp/' + gmt_in)

            np.savetxt('tmp/' + gmt_in, list(zip(lontmp, lattmp, track.ladata_df.seqid.values)))

            if XovOpt.get("local") == 0:
                if XovOpt.get("instrument") == 'LOLA':
                    if XovOpt.get("local_dem"):
                        dem = track.slewdir + "/SLDEM2015_512PPD.GRD"
                    else:
                        dem = "/explore/nobackup/projects/pgda/LOLA/data/LOLA_GDR/CYLINDRICAL/raw/LDEM_4.GRD"
                else:
                    dem = '/explore/nobackup/people/emazaric/MESSENGER/data/GDR/MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD'
            #             r_dem = subprocess.check_output(
            #                 ['grdtrack', gmt_in,
            #                  '-G' + dem],
            #                 universal_newlines=True, cwd='tmp')
            #             r_dem = np.fromstring(r_dem, sep=' ').reshape(-1, 3)[:, 2]
            # # np.savetxt('gmt_'+track.name+'.out', r_dem)

            else:
                dem = XovOpt.get("instrument") + 'SLDEM2015_512PPD.GRD'
                # r_dem = np.loadtxt('tmp/gmt_' + track.name + '.out')

            # print(['grdtrack', gmt_in, '-G' + dem,'-R0.0/360.0/-50.0/50.0'])
            if XovOpt.get("local_dem"):
                r_dem = subprocess.check_output(['grdtrack', gmt_in, '-G' + dem],
                                                universal_newlines=True, cwd='tmp')
            else:  # replace -RLON0/LONMAX/LAT0/LATMAX with appropriate bbox
                r_dem = subprocess.check_output(['grdtrack', gmt_in, '-G' + dem, '-R0.0/360.0/-50.0/50.0'],
                                                universal_newlines=True, cwd='tmp')
            if len(r_dem) == 0:
                print("Weird empty grdtrack output, please check")
                exit()

            # r_dem = np.fromstring(r_dem, sep=' ').reshape(-1, 3)[:, 2]
            r_dem = np.fromstring(r_dem, sep=' ').reshape(-1, 4)[:, 2:]

            df_ = pd.DataFrame(r_dem, columns=['seqid', 'elevation']).set_index('seqid')
            new_index = track.ladata_df.seqid.values
            r_dem = np.transpose(df_.reindex(new_index).fillna(0).values).flatten()

        elif gmt and XovOpt.get("instrument") != 'BELA':
            gmt_in = 'gmt_' + track.name + '.in'
            if os.path.exists('tmp/' + gmt_in):
                os.remove('tmp/' + gmt_in)
            np.savetxt('tmp/' + gmt_in, list(zip(lontmp, lattmp)))

            r_dem = subprocess.check_output(['grdtrack', gmt_in, '-G' + dem],
                                            universal_newlines=True, cwd='tmp')
            r_dem = np.fromstring(r_dem, sep=' ').reshape(-1, 3)[:, 2]

        elif not (XovOpt.get("instrument") in ['BELA', 'CALA', 'MLA']):
            # print("## Using weird combination (not BELA).")
            lontmp[lontmp < 0] += 360.

            r_dem = get_demz_at(track.dem, lattmp, lontmp)

            # Works but slower (interpolates each time, could be improved by https://github.com/JiaweiZhuang/xESMF/issues/24)
            # radius_xarr = dem_xarr.interp(lon=xr.DataArray(lontmp, dims='z'), lat= xr.DataArray(lattmp, dims='z')).z.values * 1.e3 #

        # Convert to meters (if DEM given in km)
        r_dem *= 1.e3
    else:
        r_dem = 0.

    # TODO replace with "small_scale_topo/texture_noise" option
    if XovOpt.get("small_scale_topo") and XovOpt.get("instrument") != "LOLA":
        texture_noise = track.apply_texture(np.mod(lattmp, 0.25), np.mod(lontmp, 0.25), grid=False)
    else:
        texture_noise = 0.

    # update Rmerc with r_dem/text (meters)
    radius = XovOpt.get("vecopts")['PLANETRADIUS'] * 1.e3 + r_dem + texture_noise

    return radius


def get_toposlope(track):
    
    df = pd.DataFrame(zip(track.ladata_df['LAT'], track.ladata_df['LON']), columns=['LAT', 'LON'])
    df['slope'] = 0

    if XovOpt.get("instrument") in ['BELA', 'CALA', 'MLA']:
        
        geotiff, masks = _get_geotiff_masks(df)
          
        for name, mask in masks.items():
            if len(df.loc[mask, :]) > 0:
                df.loc[mask,'slope'] = get_demslope_tiff(geotiff[name],
                                                         df.loc[mask, 'LON'].values,
                                                         df.loc[mask, 'LAT'].values)      

    return df.slope.values


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
