# !/usr/bin/env python3
# ----------------------------------
# Fit orbits to dem vs elevation discrepancies
# ----------------------------------
# Author: Stefano Bertone
# Created: 21-Jun-2019
#

import glob
import os
import time

import pandas as pd
import subprocess

import numpy as np
from scipy.constants import c as clight

import pickleIO
from PyAltSim import sim_gtrack
from ground_track import gtrack
from prOpt import vecopts, auxdir
import astro_trans as astr
import spiceypy as spice

import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats

def import_dem(filein):

    # grid point lists

    # open netCDF file
    # nc_file = "/home/sberton2/Downloads/sresa1b_ncar_ccsm3-example.nc"
    nc_file = filein
    dem_xarr = xr.open_dataset(nc_file)

    return dem_xarr


def get_demz_at(dem_xarr, lattmp, lontmp):
    return dem_xarr.interp(lat=lattmp, lon=lontmp).to_dataframe().loc[:, 'z'].values


def get_demz_diff_at(dem_xarr, lattmp, lontmp, axis='lon'):
    lontmp[lontmp < 0] += 360.
    print(dem_xarr)
    diff_dem_xarr = dem_xarr.differentiate(axis)

    lat_ax = xr.DataArray(lattmp, dims='z')
    lon_ax = xr.DataArray(lontmp, dims='z')

    return diff_dem_xarr.interp(lat=lat_ax, lon=lon_ax).z.to_dataframe().loc[:, 'z'].values


def plot_dem(dem_xarr):
    fh = dem_xarr
    fig, [ax1, ax2, ax3] = plt.subplots(3, 1)
    fh.z.plot(ax=ax1)
    fh.differentiate('lon').z.plot(robust=True, ax=ax2)
    fh.differentiate('lat').z.plot(robust=True, ax=ax3)
    plt.savefig('dem_map.png')


def rosen(x):
    # """The Rosenbrock function"""
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def get_demres(df,dorb):

    df_ = df.copy()
    # print("dorb =", dorb)
    # print(len(df_))

    track.pertPar['dA'] = dorb[0]
    track.pertPar['dC'] = dorb[1]
    track.pertPar['dR'] = dorb[2]

    # get time of flight in ns from probe one-way range in km
    # df_['TOF'] = df_['rng'] * 2. * 1.e3 / clight
    # preparing df for geoloc
    df_['seqid'] = df_.index
    df_ = df_.rename(columns={"epo_tx": "ET_TX"})
    df_ = df_.reset_index(drop=True)
    # copy to self
    track.ladata_df = df_[['ET_TX', 'TOF', 'orbID', 'seqid']]
    # retrieve spice data for geoloc
    if not hasattr(track, 'SpObj'):
        # create interp for track
        track.interpolate()
    else:
        track.SpObj = pickleIO.load(auxdir + 'spaux_' + track.name + '.pkl')
    old_tof = track.ladata_df.loc[:, 'TOF'].values
    rng_apr = old_tof * clight / 2.
    # read just lat, lon, elev from geoloc (reads ET and TOF and updates LON, LAT, R in df)
    track.geoloc()
    # print('pre',len(track.ladata_df))
    track.ladata_df = mad_clean(track.ladata_df,'R')
    # print('post',len(track.ladata_df))

    lontmp, lattmp, rtmp = np.transpose(track.ladata_df[['LON', 'LAT', 'R']].values)
    r_bc = rtmp + vecopts['PLANETRADIUS'] * 1.e3

    radius = read_dem(lattmp, lontmp)

    # use "real" elevation to get bounce point coordinates
    bcxyz_pbf = astr.sph2cart(radius, lattmp, lontmp)
    bcxyz_pbf = np.transpose(np.vstack(bcxyz_pbf))
    # get S/C body fixed position (useful to update ranges, has to be computed on reduced df)
    scxyz_tx_pbf = track.get_sc_pos_bf(track.ladata_df)
    # compute range btw probe@TX and bounce point@BC (no BC epoch needed, all coord planet fixed)
    rngvec = (bcxyz_pbf - scxyz_tx_pbf)
    # compute correction for off-nadir observation
    offndr = np.arccos(np.einsum('ij,ij->i', rngvec, -scxyz_tx_pbf) /
                       np.linalg.norm(rngvec, axis=1) /
                       np.linalg.norm(scxyz_tx_pbf, axis=1))
    # compute residual between "real" elevation and geoloc (based on a priori TOF)
    dr = (rtmp - radius) * np.cos(offndr)
    # print(np.mean(dr), np.std(dr))
    # print(np.abs(dr).max())
    elev_rms = np.sqrt(np.mean(dr ** 2))
    # print("elev_rms: ", elev_rms)
    return elev_rms, dr


def read_dem(lattmp, lontmp):
    # use lon and lat to get "real" elevation from map
    gmt_in = 'gmt_' + track.name + '.in'
    if os.path.exists('tmp/' + gmt_in):
        os.remove('tmp/' + gmt_in)
    np.savetxt('tmp/' + gmt_in, list(zip(lontmp, lattmp)))
    if local:
        dem = auxdir + 'MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD'
    else:
        dem = '/att/nobackup/emazaric/MESSENGER/data/GDR/MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD'
    r_dem = subprocess.check_output(
        ['grdtrack', gmt_in, '-G' + dem],
        universal_newlines=True, cwd='tmp')
    r_dem = np.fromstring(r_dem, sep=' ').reshape(-1, 3)[:, 2]
    radius = r_dem * 1.e3
    return radius


def mad_clean(df,column):
    # print("enter mad_clean")
    # print(column)
    # print(df[column])
    # exit()
    # global sorted, std_median, dforb
    mad = df[column].mad(axis=0)
    sorted = np.sort(abs(df[column].values - df[column].median(axis=0)) / mad)
    # print(mu)
    std_median = sorted[round(0.68 * len(sorted))]
    # print(len(sorted), 3 * std_median)
    # print(len(sorted[sorted > 3 * std_median]))
    # sorted = sorted[sorted < 3 * std_median]
    # print(mad, std_median)
    # print(df[column].abs().mean(axis=0))
    # print("pre-clean:",len(dforb))
    df = df[
        abs(df[column].values - df[column].median(axis=0)) / mad < 3 * std_median].reset_index()
    return df


if __name__ == '__main__':

    start = time.time()

    local = 0
    debug = 0
    numer_sol = 0

    epo = '1212'
    spk = ['KX']  # ['KX', 'AGr']
    vecopts['ALTIM_BORESIGHT'] = [0.0022105, 0.0029215, 0.9999932892]  # out[2]
    
    if local:
      spice.furnsh(auxdir + 'mymeta')  # 'aux/mymeta')
    else:
      spice.furnsh(['/att/nobackup/emazaric/MESSENGER/data/furnsh/furnsh.MESSENGER.def'])

    if local:
        dem_xarr = import_dem(
            "/home/sberton2/Works/NASA/Mercury_tides/MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_4ppd_HgM008frame.GRD")
    else:
        dem_xarr = import_dem(
            "/att/nobackup/emazaric/MESSENGER/data/GDR/MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD")

    for ex in spk:

        if False:
            sol_df = pd.read_pickle("tmp/sol_demfit_" + ex + ".pkl")
            sol_df = sol_df.filter(regex='^[d,r].*$').apply(pd.to_numeric, errors='ignore', downcast='float')

            print(sol_df)
            plt.clf()
            plt.figure(1)
            sol_df[['rms_pre', 'rms_post']].plot(kind='hist')
            plt.legend()
            plt.savefig("tmp/dr_fit_rms_" + ex + ".png")

            print("Correction avg:", sol_df.mean(axis=0))
            print("Correction rms:", np.sqrt(np.mean(sol_df.values ** 2, axis=0)))
            exit()

        if local:
            path = '/home/sberton2/Works/NASA/Mercury_tides/out/mladata/' + epo + '_' + ex + '/gtrack_' + epo[
                                                                                                    :2] + '/' + '*.pkl'
        else:
            path = "/att/nobackup/sberton2/MLA/out/mladata/" + ex + "/gtrack_*/*.pkl"

        allFiles = glob.glob(os.path.join(path))
        allFiles = np.sort(allFiles)
        print("Processing ", ex)
        print(os.path.join(path))
        print("nfiles: ", len(allFiles))
        # print(allFiles)

        prel_track = gtrack(vecopts)

        sol_df = []
        # Prepare list of tracks to geolocalise
        for fil in allFiles[:]:
            prel_track = prel_track.load(fil)
            df_ = prel_track.ladata_df.iloc[:]

            if len(df_ > 0):

                track = sim_gtrack(vecopts, orbID=df_.orbID.values[0])
                track.SpObj = prel_track.SpObj

                r_dem = read_dem(df_.LAT.values, df_.LON.values)

                df_['altdiff_dem'] = df_.R.values - r_dem
                # print(len(df_.R.values),len(r_dem[np.abs(r_dem)>0]))

                orb_unique = np.sort(list(set(df_.orbID.values)), axis=0)
                print(orb_unique)

                for orb in orb_unique:
                    dforb = df_[df_.orbID == orb].iloc[:]
                    print("pre-clean:",len(dforb),dforb.altdiff_dem.abs().max())

                    dforb = mad_clean(dforb,'altdiff_dem')
                    print("post-clean:",len(dforb),dforb.altdiff_dem.abs().max())
                    # exit()

                    # print(dforb.filter(regex='dLON/d[A,C,R]$'))

                    # dforb['dR/dLON'] = \
                    # print(dem_xarr)
                    dforb['dR/dLON'] = get_demz_diff_at(dem_xarr, dforb.LAT.values, dforb.LON.values, 'lon')
                    dforb['dR/dLAT'] = get_demz_diff_at(dem_xarr, dforb.LAT.values, dforb.LON.values, 'lat')
                    # print(dforb.filter(regex='dR/dL[O,A]'))
                    #
                    # print(dforb.filter(regex='dR/d[A,C,R]$').values)

                    dforb_full = (dforb.filter(regex='dR/d[A,C,R]$') +
                                  dforb[['dR/dLON']].values * dforb.filter(regex='dLON/d[A,C,R]$').values +
                                  dforb[['dR/dLAT']].values * dforb.filter(regex='dLAT/d[A,C,R]$').values)
                    # print(dforb_full.filter(regex='dR/d[A,C,R]$').values)
                    # exit()

                    Amat = dforb_full.filter(regex='dR/d[A,C,R]$').values[:,:]
                    bvec = dforb.altdiff_dem[:]
                    # print(bvec)
                    # print(Amat)
                    # print(max(bvec),min(bvec))
                    # # # print(pd.concat([bvec,Amat],axis=1))
                    # # # print(bvec)
                    # exit()

                    solstd = np.linalg.lstsq(Amat, bvec, rcond=None)
                    print("Sol for ",orb,":",solstd[0],"+-",solstd[3])

                    # compute residuals pre-fit
                    dorb = [0,0,0]
                    elev_rms_pre, dr_pre = get_demres(dforb.copy(),dorb)
                    # print(elev_rms_pre, dr_pre)

                    # apply corrections and compute post-fit residuals
                    dorb = -1.*solstd[0]
                    elev_rms_post, dr_post = get_demres(dforb.copy(),dorb)
                    # print(elev_rms_post, dr_post)
                    #
                    # print(np.max(np.abs(dr_pre)),np.max(np.abs(dr_post)))
                    dr_pre = mad_clean(pd.DataFrame(dr_pre,columns=['a']),'a').a.values
                    dr_post = mad_clean(pd.DataFrame(dr_post,columns=['a']),'a').a.values
                    sol_df.append(np.concatenate([[orb], solstd[0], solstd[3],
                                                  [np.sqrt(np.mean(dr_pre ** 2, axis=0))],[np.sqrt(np.mean(dr_post ** 2, axis=0))]]))

                    print("pre-fit rms for orb#",orb," :",elev_rms_pre, np.sqrt(np.mean(dr_pre ** 2, axis=0)))
                    # print(dr_pre)

                    plt.clf()
                    plt.figure(1)
                    plt.hist(dr_pre, density=True,label="pre-fit")
                    # print(min(dr_pre), max(dr_pre))

                    mean = np.mean(dr_pre)
                    variance = np.var(dr_pre)
                    sigma = np.sqrt(variance)
                    x = np.linspace(min(dr_pre), max(dr_pre), 100)
                    plt.plot(x, stats.norm.pdf(x, mean, sigma),label="pre-fit")

                    print("post-fit rms for orb#",orb," :",elev_rms_post, np.sqrt(np.mean(dr_post ** 2, axis=0)))

                    plt.hist(dr_post, density=True,label="post-fit")
                    plt.xlim((min(dr_post), max(dr_post)))

                    mean = np.mean(dr_post)
                    variance = np.var(dr_post)
                    sigma = np.sqrt(variance)
                    x = np.linspace(min(dr_post), max(dr_post), 100)
                    plt.plot(x, stats.norm.pdf(x, mean, sigma),label="post-fit")
                    plt.legend()
                    plt.savefig("tmp/dr_fit_"+orb+".png")

                if debug:
                    Amat = dforb.filter(regex='dR/d[A,C,R]$')
                    bvec = dforb.altdiff_dem
                    # print(bvec)

                    solstd_old = np.linalg.lstsq(Amat, bvec, rcond=None)
                    print("Diff with old sol (no dR/dLonLat) for ", orb, ":", solstd_old[0] - solstd[0])

                if numer_sol:
                    from scipy.optimize import fmin_powell, minimize

                    dorb = [-47.35363236, 465.27145885,
                            -58.184007]  # range(-10,10,1) #np.array([10.3, 0.7, 10.8, 11.9, 1.2])
                    # print(fmin_powell(get_demres,dorb,retall=True))
                    print(minimize(get_demres, dorb, method='BFGS', jac=False, options={'disp': True}))
                    # exit()
                    # for d in dorb:
                    elev_rms = get_demres(dorb)
                    print("elev - dem rms is", elev_rms, "meters for d=", dorb)
            else:
                print("No data in ", fil)

        sol_df = pd.DataFrame(sol_df, columns=['orb', 'dA', 'dC', 'dR', 'sig_dA', 'sig_dC', 'sig_dR','rms_pre','rms_post'])
        sol_df.to_pickle("tmp/sol_demfit_"+ex+".pkl")

    # stop clock and print runtime
    # -----------------------------
    end = time.time()
    print('----- Runtime = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
