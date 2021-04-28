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
from scipy.interpolate import RectBivariateSpline

from src.pyaltsim.PyAltSim import sim_gtrack
from src.xovutil.dem_util import get_demz_grd, get_demz_tiff
from src.geolocate_altimetry import geoloc
from src.pygeoloc.ground_track import gtrack
# from examples.MLA.options import XovOpt.get("vecopts"), XovOpt.get("auxdir"), XovOpt.get("SpInterp"), XovOpt.get("tmpdir"), XovOpt.get("local"), XovOpt.get("debug"), pert_cloop, XovOpt.get("spauxdir")
from config import XovOpt

from src.xovutil import astro_trans as astr, pickleIO
import spiceypy as spice

import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats, signal

from src.tidal_deform import set_const, tidepart_h2
from src.xovutil.iterables import mergsum

numer_sol = 1
pergram = 0
check_pkl = 0


def import_dem(filein):
    # grid point lists

    # open netCDF file
    # nc_file = "/home/sberton2/Downloads/sresa1b_ncar_ccsm3-example.nc"
    nc_file = filein
    dem_xarr = xr.open_dataset(nc_file)

    lats = np.deg2rad(dem_xarr.lat.values) + np.pi / 2.
    lons = np.deg2rad(dem_xarr.lon.values)  # -np.pi
    data = dem_xarr.z.values


    print("test")
    # Exclude last column because only 0<=lat<pi
    # and 0<=lon<pi are accepted (checked that lon=0 has same values)
    # print(data[:,0]==data[:,-1])
    # kx=ky=1 required not to mess up results!!!!!!!!!! Higher interp orders mess up...
    if True:
        interp_spline = RectBivariateSpline(lats[:-1],
                                            lons[:-1],
                                            data[:-1, :-1], kx=1, ky=1)
        pickleIO.save(interp_spline, XovOpt.get("tmpdir") + "interp_dem.pkl")
    else:
        interp_spline = pickleIO.load(XovOpt.get("tmpdir") + "interp_dem.pkl")

    return interp_spline


def get_demz_at(dem_xarr, lattmp, lontmp):
    # lontmp += 180.
    lontmp[lontmp < 0] += 360.
    # print("eval")
    # print(np.sort(lattmp))
    # print(np.sort(lontmp))
    # print(np.sort(np.deg2rad(lontmp)))
    # exit()

    return dem_xarr.ev(np.deg2rad(lattmp) + np.pi / 2., np.deg2rad(lontmp))


def get_demz_diff_at(dem_xarr, lattmp, lontmp, axis='lon'):
    step = 1.e-3

    if axis == 'lon':
        demz_p = get_demz_at(dem_xarr, lattmp, lontmp + step)
        demz_m = get_demz_at(dem_xarr, lattmp, lontmp - step)
    elif axis == 'lat':
        demz_p = get_demz_at(dem_xarr, lattmp + step, lontmp)
        demz_m = get_demz_at(dem_xarr, lattmp - step, lontmp)
    else:
        print("Problem!!")

    return (demz_p - demz_m) / (2 * step)


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


def get_demres(dorb, track, df, coeff_set=['dA', 'dC', 'dR','dRl','dPt']): #]):  # ,'dA1', 'dC1', 'dR1']): #
    #
    dorb = np.array(dorb)
    dorb[:3] *= 1000.

    # print("in demres", df['altdiff_dem'].max())
    dr, dummy = get_demres_full(dorb, track, df, coeff_set)

    # print("in demres post", df['altdiff_dem'].max(), np.sqrt(np.mean(df['altdiff_dem'] ** 2)), np.max(dr))

    #    print_demfit(dr, df['ET_TX'], dorb, track.name)

    # print(np.mean(dr), np.std(dr))
    # print(np.abs(dr).max())
    elev_rms = np.sqrt(np.mean(dr ** 2))
    # print("elev_rms: ", elev_rms)
    # exit()

    # print(dorb, elev_rms)

    return elev_rms  # , dr, track.ladata_df.ET_TX.values - track.t0_orb


def get_demres_full(dorb, track, df, dem_file,
                    coeff_set=['dA', 'dC', 'dR']):  # ,'dA1', 'dC1', 'dR1']): #,'dRl','dPt']):
    track.pertPar = {'dA': 0.,
                     'dC': 0.,
                     'dR': 0.,
                     'dRl': 0.,
                     'dPt': 0.,
                     'dRA': [0., 0., 0.],
                     'dDEC': [0., 0., 0.],
                     'dPM': [0., 0., 0.],
                     'dL': 0.}

    _ = {}
    # get cloop sim perturbations from prOpt
    #    [_.update(v) for k, v in pert_cloop.items()]
    #    track.pert_cloop = _.copy()
    # randomize and assign pert for closed loop sim IF orbit in pert_tracks
    #    np.random.seed(int(track.name))
    #    rand_pert_orb = np.random.randn(len(pert_cloop_orb))
    #    track.pert_cloop = dict(zip(track.pert_cloop.keys(),list(pert_cloop_orb.values()) * rand_pert_orb))
    #    track.pertPar = mergsum(track.pert_cloop.copy(), track.pertPar.copy())

    # print("dorb", dorb)
    df_ = df.copy()
    for idx, key in enumerate(coeff_set):
        if key in track.pertPar:
            track.pertPar[key] += dorb[idx]
            # print("exist", key, idx, track.pertPar[key])
        else:
            track.pertPar[key] = dorb[idx]
            # print("new", key, idx, track.pertPar[key])

    # get time of flight in ns from probe one-way range in km
    # df_['TOF'] = df_['rng'] * 2. * 1.e3 / clight
    # preparing df for geoloc
    df_['seqid'] = df_.index
    df_ = df_.rename(columns={"epo_tx": "ET_TX"})
    df_ = df_.reset_index(drop=True)
    # copy to self
    track.ladata_df = df_[['ET_TX', 'TOF', 'orbID', 'seqid']]
    track.t0_orb = track.ladata_df['ET_TX'].values[0]
    # retrieve spice data for geoloc
    # print(track.ladata_df)

    if not hasattr(track, 'SpObj') and XovOpt.get("SpInterp") == 2:
        # create interp for track
        track.interpolate()
    elif XovOpt.get("SpInterp") != 0:
        track.SpObj = pickleIO.load(XovOpt.get("auxdir") + XovOpt.get("spauxdir") + 'spaux_' + track.name + '.pkl')
    else:
        track.SpObj = None

    old_tof = track.ladata_df.loc[:, 'TOF'].values
    rng_apr = old_tof * clight / 2.
    # read just lat, lon, elev from geoloc (reads ET and TOF and updates LON, LAT, R in df)
    # track.geoloc(get_partials=False)
    tmp_pertPar = track.pertPar
    tmp_pertPar = {k:(np.array(v) if isinstance(v, list) else v) for k, v in tmp_pertPar.items()}
    # print(track.pertPar, tmp_pertPar)
    # tmp_pertPar = mergsum(track.pertPar, tmp_pertPar)
    # track.pertPar = tmp_pertPar
    # print("call geoloc",track.pertPar)
    # exit()
    # print("in demres_full", df_['altdiff_dem'].max())
    geoloc_out, et_bc, dr_tidal, dummy = geoloc(df_, XovOpt.get("vecopts"), tmp_pertPar, track.SpObj, t0=track.t0_orb)
    # print(np.transpose(geoloc_out))
    # df_['LON'] = geoloc_out[:, 0]
    # df_['LAT'] = geoloc_out[:, 1]
    # print(len(results[0]))
    Rbase = XovOpt.get("vecopts")['PLANETRADIUS'] * 1.e3
    # df_['R'] = geoloc_out[:, 2] - Rbase

    # Clean data --> commented out for consistency
    # print('pre',len(track.ladata_df))
    # track.ladata_df = mad_clean(track.ladata_df,'R')
    # print('post',len(track.ladata_df))
    # exit()
    # print(track.ladata_df[['LON', 'LAT', 'R']].values)
    # exit()
    lontmp, lattmp, rtmp = np.transpose(geoloc_out)
    rtmp = rtmp - XovOpt.get("vecopts")['PLANETRADIUS'] * 1.e3
    # r_bc = rtmp + vecopts['PLANETRADIUS'] * 1.e3

    # Call GrdTrk
    # ------------
    # gmt_in = 'gmt_' + track.name + '.in'
    # s = time.time()
    # radius = read_dem(gmt_in, lattmp, lontmp)
    # e = time.time()
    # print("time gmt",e-s)

    ## time and use xarray (a bit slower, maybe have to use better interp routine)
    # s = time.time()
    lontmp[lontmp < 0] += 360.
    # Works but slower (interpolates each time, could be improved by https://github.com/JiaweiZhuang/xESMF/issues/24)
    # radius_xarr = dem_xarr.interp(lon=xr.DataArray(lontmp, dims='z'), lat= xr.DataArray(lattmp, dims='z')).z.values * 1.e3 #
    # radius_xarr = get_demz_at(dem_xarr, lattmp, lontmp) * 1.e3

    filin = dem_file
    if filin.split('.')[-1] == 'GRD': # to read grd/netcdf files
        radius = get_demz_grd(filin, lontmp, lattmp) * 1.e3 #
    elif filin.split('.')[-1] == 'tif': #'TIF': # to read geotiffs usgs
        radius = np.squeeze(get_demz_tiff(filin, lontmp, lattmp)) * 1.e3

    # print(radius)
    # exit()
    # e = time.time()
    # print("time xarr",e-s)
    # print(radius)
    # print(radius_xarr)
    # print("diff dems", np.max(radius-radius_xarr))
    # radius = radius_xarr

    # df_['RmDEM'] = df_['R'].values - radius

    # use "real" elevation to get bounce point coordinates
    # bcxyz_pbf = astr.sph2cart(radius, lattmp, lontmp)
    # bcxyz_pbf = np.transpose(np.vstack(bcxyz_pbf))
    # # get S/C body fixed position (useful to update ranges, has to be computed on reduced df)
    # scxyz_tx_pbf = track.get_sc_pos_bf(track.ladata_df)
    # # compute range btw probe@TX and bounce point@BC (no BC epoch needed, all coord planet fixed)
    # rngvec = (bcxyz_pbf - scxyz_tx_pbf)
    # # print("rngvec", np.min(np.linalg.norm(rngvec, axis=1)))
    # # print("scxyz_tx_pbf", np.min(np.linalg.norm(scxyz_tx_pbf, axis=1)))
    # # print(np.mean(rngvec), np.mean(scxyz_tx_pbf),np.mean(bcxyz_pbf))
    # rngvec_normed = rngvec/np.linalg.norm(rngvec, axis=1)[:, np.newaxis]
    # scxyz_tx_pbf_normed = np.array(scxyz_tx_pbf)/np.linalg.norm(scxyz_tx_pbf, axis=1)[:, np.newaxis]
    # # print(np.max(np.abs(np.einsum('ij,ij->i', rngvec_normed, -scxyz_tx_pbf_normed))))
    # # compute correction for off-nadir observation
    # if np.max(np.abs(np.einsum('ij,ij->i', rngvec_normed, -scxyz_tx_pbf_normed))) <= 1:
    #     offndr = np.arccos(np.einsum('ij,ij->i', rngvec_normed, -scxyz_tx_pbf_normed))
    # else:
    #     offndr = 0.

    # plt.clf()
    # # plt.plot(radius[:])
    # # plt.plot(rtmp[:])
    # plt.plot((radius-rtmp)[:])
    # plt.savefig("tmp/tst.png")

    # print(np.vstack([rtmp, radius, np.cos(offndr)]))

    # print(np.mean(offndr),np.mean(rtmp),np.mean(radius))
    # compute residual between "real" elevation and geoloc (based on a priori TOF)
    dr = (rtmp - radius)  # * np.cos(offndr)

    # df_['RattmDEM'] = dr
    # print(df_[['ET_TX','LON','LAT','R','RmDEM','RattmDEM']])
    #
    # test_erwan = pd.read_csv('/home/sberton2/Works/NASA/Mercury_tides/PyXover_diralt/tmp/tst_erwan.txt',
    #             sep='\s+', header=None)
    # test_erwan.columns = ['dum','ET_TX','LON','LAT','R','RmDEM']
    # # test_erwan.LON *= 0
    # test_erwan.drop('dum',inplace=True,axis=1)
    # test_erwan.loc[:,'R'] = test_erwan.loc[:,'R'].values * 1.e3
    # test_erwan.loc[:,'RmDEM'] = test_erwan.loc[:,'RmDEM'].values * 1.e3
    # test_erwan.loc[:,'RattmDEM'] = test_erwan.loc[:,'RmDEM'].values
    # test_erwan.loc[:,'ET_TX'] = test_erwan.loc[:,'ET_TX'].round(3)
    # df_.loc[:,'ET_TX'] = df_.loc[:,'ET_TX'].round(3)
    # print(test_erwan)
    #
    # check = df_[['ET_TX','LON','LAT','R','RmDEM','RattmDEM']].set_index('ET_TX')-test_erwan.set_index('ET_TX')
    # print(check)
    # check = check.fillna(0)
    #
    # plt.clf()
    # # check = check.loc[check.RmDEM.abs() < 10]
    # # print(check)
    # check.reset_index().plot(x='ET_TX',y=['RmDEM','RattmDEM'])
    # plt.ylim([-30,30])
    # plt.savefig("tmp/check.png")
    #
    # exit()
    # print("in demres_full geoloc out", np.max(dr))

    return dr, dr_tidal


# def read_dem(gmt_in, lattmp, lontmp):
#     # use lon and lat to get "real" elevation from map
#     if os.path.exists('tmp/' + gmt_in):
#         os.remove('tmp/' + gmt_in)
#     np.savetxt('tmp/' + gmt_in, list(zip(lontmp, lattmp)))
#     if local:
#         dem = auxdir + 'HDEM_64.GRD'  # 'MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km.GRD' # 'MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD'
#     else:
#         dem = '/att/nobackup/emazaric/MESSENGER/data/GDR/HDEM_64.GRD'
#     # print(subprocess.check_output(['grdinfo', dem]))
#     r_dem = subprocess.check_output(
#         ['grdtrack', gmt_in, '-G' + dem],
#         universal_newlines=True, cwd='tmp')
#     # print(r_dem)
#     # exit()
#     r_dem = np.fromstring(r_dem, sep=' ').reshape(-1, 3)[:, 2]
#     radius = r_dem * 1.e3
#     return radius


def mad_clean(df, column='altdiff_dem_data'):
    # print("enter mad_clean")
    # print(column)
    # print(df[column])
    # exit()
    # global sorted, std_median, dforb
    if len(df[column].values) > 1:
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


def lomb(data_df: pd.DataFrame,
         min_period: float,
         max_period: float):
    """
    LombâScargle periodogram implementation
    :param data: list
    :param high_frequency: float
    :param low_frequency: float
    :return lomb-scargle pgram and frequency values
    """

    time_stamps = data_df.Dt.values
    samples = data_df.altdiff_dem.values

    window = signal.tukey(len(samples))
    samples *= window

    # nout = 10000  # number of frequency-space points at which to calculate the signal strength (output)
    periods = np.linspace(min_period, max_period, len(data_df) * 100)
    freqs = 1.0 / periods
    frequency_range = 2 * np.pi * freqs
    # frequency_range = np.linspace(low_frequency, high_frequency, len(data_df))
    result = signal.lombscargle(time_stamps, samples, frequency_range, precenter=True, normalize=True)
    return result, frequency_range


def plot_pergram(dforb, filename):
    n_per_orb_min = 0.005
    n_per_orb_max = 1
    orb_period = 0.5 * 86400  # 12h to seconds
    twopi = 2 * np.pi
    w = twopi / orb_period

    # extract data from df
    tmp_df = pd.DataFrame(dforb.Dt)
    tmp_df.columns = ['Dt']
    tmp_df['altdiff_dem'] = dforb.altdiff_dem.values  # + # 1*np.sin(100*tmp_df.Dt.values*2*np.pi/(0.5*86400)) #
    # generate periodogram
    pgram, freq = lomb(tmp_df.reset_index(), n_per_orb_min, n_per_orb_max)

    plt.clf()
    fig, [ax1, ax2] = plt.subplots(2, 1)
    # data
    tmp_df.plot(x='Dt', y='altdiff_dem', ax=ax1, style='.', label='data')
    ax1.xaxis.set_ticks_position('top')
    window = signal.tukey(len(tmp_df))
    ax1.plot(tmp_df.Dt.values, window * tmp_df.altdiff_dem.values, label='windata')
    # periodogram
    ax2.plot(freq / w, pgram, label='lbscft')
    ax2.set_xscale('log')
    ax2.set_xlabel('n_per_orbit (12h)')
    plt.legend()
    # ax2.axvline(41.3*2*np.pi/(0.5*86400), lw=2, color='red', alpha=0.4)
    # freq axes (Hz)
    ax3 = ax2.twiny()
    ax3.set_xlim(ax2.get_xlim()[0] * w, ax2.get_xlim()[1] * w)
    ax3.set_xscale('log')
    ax3.tick_params(direction='in', length=6, width=2, colors='r', which='both',
                    grid_color='r', grid_alpha=0.5)
    plt.savefig(filename)

    return


def fit_track_to_dem(df_in,dem_file):
    # import warnings
    # warnings.filterwarnings("ignore", message="SettingWithCopyWarning ")
    # warnings.filterwarnings("ignore", message="RankWarning: The fit may be poorly conditioned ")

    df_ = df_in.copy()

    # coeff_set_re = '^dR/d[A,C,R]'
    # n_per_orbit = [400]  # 100,800] #

    XovOpt.get("vecopts")['ALTIM_BORESIGHT'] = [0.0022105, 0.0029215, 0.9999932892]  # out[2]

    # if local:
    #     dem_xarr = import_dem(
    #         "/home/sberton2/Works/NASA/Mercury_tides/aux/MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km.GRD") #HDEM_64.GRD")
    # else:
    #     dem_xarr = import_dem(
    #         "/att/nobackup/emazaric/MESSENGER/data/GDR/MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD")

    if len(df_) > 0:

        track = sim_gtrack(XovOpt.get("vecopts"), orbID=df_.orbID.values[0])
        # # track.SpObj = prel_track.SpObj
        #
        # gmt_in = 'gmt_' + track.name + '.in'
        # r_dem = read_dem(gmt_in, df_.LAT.values, df_.LON.values)
        #
        # df_.loc[:,'altdiff_dem'] = df_.R.values - r_dem
        # print(len(df_.R.values),len(r_dem[np.abs(r_dem)>0]))

        orb_unique = np.sort(list(set(df_.orbID.values)), axis=0)
        # print(orb_unique)

        # for orb in orb_unique:
        #     dforb = df_[df_.orbID == orb]
        #     dforb.loc[:,"Dt"] = dforb.ET_TX.values - dforb.ET_TX.iloc[0]
        #     # store t0 orbit
        #     track.t0_orb = dforb.ET_TX.iloc[0]
        #     # print("ET0", track.t0_orb)
        #
        #     # if pergram:
        #     #     plot_pergram(dforb, "tmp/dr_pref_res_" + orb + "_" + spk + ".png")
        #     #
        #     # plt.clf()
        #     # fig, ax1 = plt.subplots(1, 1)
        #     # dforb.plot(x='ET_TX', y='altdiff_dem', label="pre-clean", ax=ax1)
        #     # print("pre-clean:", len(dforb), dforb.altdiff_dem.abs().max())
        #     # dforb = mad_clean(dforb, 'altdiff_dem')
        #     # dforb.plot(x='ET_TX', y='altdiff_dem', label="post-clean", ax=ax1)
        #     # print("post-clean:", len(dforb), dforb.altdiff_dem.abs().max())
        #     #
        #     # plt.savefig("tmp/clean_" + orb + "_" + spk + "_" + test_name + ".png")
        #     #
        #     # dforb['dR/dLON'] = get_demz_diff_at(dem_xarr, dforb.LAT.values, dforb.LON.values, 'lon')
        #     # dforb['dR/dLAT'] = get_demz_diff_at(dem_xarr, dforb.LAT.values, dforb.LON.values, 'lat')
        #     #
        #     # dforb_full = (dforb.filter(regex='dR/d[A,C,R]$') +
        #     #               dforb[['dR/dLON']].values * dforb.filter(regex='dLON/d[A,C,R]$').values +
        #     #               dforb[['dR/dLAT']].values * dforb.filter(regex='dLAT/d[A,C,R]$').values)
        #     # Add linear terms
        #     # _ = pd.DataFrame(np.multiply(dforb_full.filter(regex='dR/d[A,C,R]$').values,dforb.Dt.values[..., np.newaxis]),columns=['dR/dA_1','dR/dC_1','dR/dR_1'])
        #     # dforb_full = pd.concat([dforb_full,_],axis=1)
        #
        #     # Add sin and cos terms to bias coeff
        #     # orb_period = 0.5 * 86400  # 12h to seconds
        #     # twopi = 2 * np.pi
        #     # w = twopi / orb_period
        #     #
        #     # for cpr in n_per_orbit:
        #     #     cpr_str = str(cpr)
        #     #     coskwt = np.cos(cpr * w * dforb.Dt.values)
        #     #     sinkwt = np.sin(cpr * w * dforb.Dt.values)
        #     #     cos_coeffs = pd.DataFrame(
        #     #         np.multiply(dforb_full.filter(regex='dR/d[A,C,R]$').values, coskwt[..., np.newaxis]),
        #     #         columns=['dR/dAc'+cpr_str, 'dR/dCc'+cpr_str, 'dR/dRc'+cpr_str])
        #     #     sin_coeffs = pd.DataFrame(
        #     #         np.multiply(dforb_full.filter(regex='dR/d[A,C,R]$').values, sinkwt[..., np.newaxis]),
        #     #         columns=['dR/dAs'+cpr_str, 'dR/dCs'+cpr_str, 'dR/dRs'+cpr_str])
        #     #     dforb_full = pd.concat([dforb_full, cos_coeffs, sin_coeffs], axis=1)
        #
        #     # Amat = dforb_full.filter(regex=coeff_set_re).values[:, :]
        #     # bvec = dforb.altdiff_dem[:]
        #     #
        #     # solstd = np.linalg.lstsq(Amat, bvec, rcond=None)
        #     #
        #     # if debug:
        #     #     print("Sol for ", orb, ":", solstd[0], "+-", solstd[3])
        #
        #     # exit()
        #     # numerical check of partials
        #     # dorb = [0,0,0,0,0,0,0,0,0]
        #     # elev_rms_pre, dr_min = get_demres(dforb.copy(),dorb)
        #     # dorb = [0,0,0,0,0.1,0,0,0,0]
        #     # elev_rms_pre, dr_plu = get_demres(dforb.copy(),dorb)
        #     # print((dr_plu-dr_min)/0.1)
        #     # exit()
        #
        #     if False:
        #         # compute residuals pre-fit
        #         coeff_set = dforb_full.filter(regex=coeff_set_re).columns.values
        #         coeff_set = [str.split('/')[-1] for str in coeff_set]
        #         dorb = np.zeros(len(coeff_set))
        #         # print(coeff_set,dorb)
        #         elev_rms_pre, dr_pre, dt_pre = get_demres(track, dforb.copy(), dorb)
        #         # print(elev_rms_pre, dr_pre)
        #
        #         # iter over orbit corrections until convergence
        #         maxiter = 0
        #         tol = 1
        #         for i in range(maxiter):
        #             try:
        #                 # apply corrections and compute post-fit residuals
        #                 dorb = -1. * solstd[0]
        #
        #                 elev_rms_post, dr_post, dt_post = get_demres(track, dforb.copy(), dorb, coeff_set=coeff_set)
        #
        #                 # print("iter", i, elev_rms_post, dr_post)
        #                 #
        #                 # print(len(Amat), len(bvec), len(dr_post), len(dr_pre))
        #
        #                 bvec = dr_post
        #                 solstd = np.linalg.lstsq(Amat, bvec, rcond=None)
        #
        #                 if debug:
        #                     print("post-fit rms for orb#", orb, " iter ", i, " :", elev_rms_post,
        #                           np.sqrt(np.mean(dr_post ** 2, axis=0)))
        #                     print("Sol2 for ", orb, ":", solstd[0], "+-", solstd[3])
        #
        #                 if max(abs(solstd[0])) < tol:
        #                     # if (max(abs(avgerr-avgerr_old))<tlcbnc):
        #                     print("Final pars values", track.pertPar)
        #                     # print("post-fit rms for orb#", orb, " iter ", i, " :", elev_rms_post)
        #                     break
        #                 elif i == maxiter - 1:
        #                     print("Max iter reached for orb ", orb, i)
        #             except:
        #                 print("Failed for orbit ", orb, "at iter", i, "rms: ", elev_rms_pre, elev_rms_post)
        #                 break
        #
        #             plt.clf()
        #             fig, [ax1, ax2] = plt.subplots(2, 1)
        #             ax1.xaxis.set_ticks_position('top')
        #             ax1.plot(dt_pre, dr_pre, label='data_pre')
        #             ax1.plot(dt_post, dr_post, label='data_post')
        #
        #             # Clean up residuals for plotting/histo
        #             dr_pre = mad_clean(pd.DataFrame(dr_pre, columns=['a']), 'a').a.values
        #             dr_post = mad_clean(pd.DataFrame(dr_post, columns=['a']), 'a').a.values
        #             # sol_df.append(np.concatenate([[orb], list([v for x, v in track.pertPar.items() if x in coeff_set]),
        #             #                               solstd[3],
        #             #                               [np.sqrt(np.mean(dr_pre ** 2, axis=0))],
        #             #                               [np.sqrt(np.mean(dr_post ** 2, axis=0))]]))
        #
        #             print("pre-fit rms for orb#", orb, " :", elev_rms_pre, np.sqrt(np.mean(dr_pre ** 2, axis=0)))
        #             # print(dr_pre)
        #
        #             ax2.hist(dr_pre, density=True, bins=50, label="pre-fit")
        #             # print(min(dr_pre), max(dr_pre))
        #
        #             mean = np.mean(dr_pre)
        #             variance = np.var(dr_pre)
        #             sigma = np.sqrt(variance)
        #             x = np.linspace(min(dr_pre), max(dr_pre), 100)
        #             ax2.plot(x, stats.norm.pdf(x, mean, sigma), label="pre-fit")
        #
        #             print("post-fit rms for orb#", orb, " :", elev_rms_post, np.sqrt(np.mean(dr_post ** 2, axis=0)))
        #
        #             ax2.hist(dr_post, density=True, bins=50, label="post-fit")
        #             ax2.set_xlim((min(dr_post), max(dr_post)))
        #
        #             mean = np.mean(dr_post)
        #             variance = np.var(dr_post)
        #             sigma = np.sqrt(variance)
        #             x = np.linspace(min(dr_post), max(dr_post), 100)
        #             ax2.plot(x, stats.norm.pdf(x, mean, sigma), label="post-fit")
        #             plt.title('pre-fit:' + str(np.mean(dr_pre).round(2)) + ', ' + str(np.std(dr_pre).round(2)) +
        #                       ' post-fit:' + str(np.mean(dr_post).round(2)) + ', ' + str(np.std(dr_post).round(2)))
        #
        #             plt.legend()
        #             plt.savefig("tmp/dr_fit_" + orb + "_" + spk + "_" + test_name + ".png")

        # if debug:
        #     Amat = dforb.filter(regex='dR/d[A,C,R]$')
        #     bvec = dforb.altdiff_dem
        #     # print(bvec)
        #
        #     solstd_old = np.linalg.lstsq(Amat, bvec, rcond=None)
        #     print("Diff with old sol (no dR/dLonLat) for ", orb, ":", solstd_old[0] - solstd[0])

        if numer_sol:
            from scipy.optimize import fmin_powell, minimize

            print("Numerical sol:")

            # if local:
            #     dem_xarr = import_dem(
            #         "/home/sberton2/Works/NASA/Mercury_tides/aux/HDEM_64.GRD")  # MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_4ppd_HgM008frame.GRD")
            # else:
            #     dem_xarr = import_dem(
            #         "/att/nobackup/emazaric/MESSENGER/data/GDR/HDEM_64.GRD")

            # Preliminary data cleaning
            dorb = np.array([0., 0., 0.]) #, 0., 0.])  # ,0,0,0]  # range(-10,10,1) #np.array([10.3, 0.7, 10.8, 11.9, 1.2])
            dr, dummy = get_demres_full(dorb, track, df_, dem_file)
            # print("dr", dr)
            print(track.name, "pre-clean (len, max, rms): ", len(dr), np.max(dr), np.round(np.sqrt(np.mean(dr ** 2)),2))

            df_.loc[:, 'dr_dem'] = dr
            df_ = df_[df_['dr_dem'] < 1.e3]
            df_ = mad_clean(df_, 'dr_dem')
            # df_=df_.iloc[:-1000,:]
            # exit()
            dr_pre = df_.loc[:, 'dr_dem'].values
            dt_pre = df_['ET_TX'].values

            dr_pre, dummy = get_demres_full(dorb, track, df_, dem_file)
            print(track.name, "pre-fit (len, max, rms): ", len(dr_pre), np.round(np.max(dr_pre),2), np.round(np.sqrt(np.mean(dr_pre ** 2)),2))

            # Fit orbit corrections to DEM

            sol = minimize(get_demres, dorb, args=(track,df_,dem_file), method='SLSQP', #'L-BFGS-B', #'Nelder-Mead', #
                           bounds=[(-0.5,0.5),(-0.5,0.5),(-0.1,0.1)], #,(-0.5,0.5),(-0.5,0.5)], #,(-1e-2,1e-2),(-1e-2,1e-2),(-1e-2,1e-2)],
                          jac='2-point',
                          options={'disp': False, 'eps': 0.0005, 'ftol': 1.e-6})
            # print(sol)

            # Fake sol
            # sol = pd.Series()
            # sol.x = dorb  # np.array([ 35.76667513, -23.1126592,    5.82315721])*1.e-3 #[ 0.01443999, -0.01408257, -0.01278677]
            # sol.fun = np.sqrt(np.mean(dr_pre ** 2))

            # Convert solution to meters
            dorb = sol.x
            dorb[:3] *= 1000.
            dr_post, dr_tidal = get_demres_full(dorb, track, df_, dem_file)
            df_.loc[:, 'altdiff_dem_data'] = dr_post
            df_.loc[:, 'dR_tid'] = dr_tidal
            df_ = mad_clean(df_, 'altdiff_dem_data')
            dt_post = df_['ET_TX'].values
            dr_post = df_['altdiff_dem_data'].values
            print(track.name, "post-fit (len, max, rms): ", len(dr_post), np.round(np.max(dr_post),2), np.round(np.sqrt(np.mean(dr_post ** 2)),2))
            print("rms for",track.name,"is", sol.fun, "meters for dACR =", np.round(dorb,2), " meters.")

            # compute partials dr/dACR
            # df_ = get_lstsq_partials(dem_xarr, df_, dorb, track)
            # print(df_)

        #            print_demfit(dr_post, dt_post, dorb, track.name, dr_pre, dt_pre)
        df_ = df_.rename(columns={'altdiff_dem_data':'dr_post','dr_dem':'dr_pre'})
        # print(pd.DataFrame([dr_pre,dr_post]).T)
        print(df_)
        # exit()

        return dr_post, dr_pre, dorb, df_

    else:
        print("No data in ", fil)


def get_lstsq_partials(dem_xarr, df_, dorb, track):
    step = 10
    dr_dorb = []
    dorb_tmp = dorb
    for idx, parnam in enumerate(['dA', 'dC', 'dR']):
        dorb_tmp = dorb
        dorb_tmp[idx] += step
        dr_p, dummy = get_demres_full(dorb_tmp, track, df_, dem_xarr)
        dorb_tmp = dorb
        dorb_tmp[idx] -= step
        dr_m, dummy = get_demres_full(dorb_tmp, track, df_, dem_xarr)
        dr_dorb.append((dr_p - dr_m) / (2 * step))
    # print(dr_dorb)
    dr_dorb.append(tidepart_h2(track.XovOpt.get("vecopts"), \
                               np.transpose(astr.sph2cart(
                                   df_['R'].values + 2440 * 1.e3,
                                   df_['LAT'].values, df_['LON'].values)), \
                               df_['ET_TX'].values + 0.5 * df_['TOF'].values, track.SpObj)[0])
    dr_dorb = np.vstack(dr_dorb).T
    # print(dr_dorb)
    # Amat = dr_dorb
    # bvec = df_.altdiff_dem_data[:]
    # solstd = np.linalg.lstsq(Amat, bvec, rcond=None)
    # print(solstd)
    # dorb += solstd[0][:3]
    #
    # dr_post, dr_tidal = get_demres_full(dorb, track, df_, dem_xarr)
    # df_.loc[:,'altdiff_dem_data'] = dr_post
    # df_.loc[:,'dR_tid'] = dr_tidal
    # dt_post = df_['ET_TX'].values
    # print("post-fit2: ", np.max(dr_post),np.sqrt(np.mean(dr_post ** 2)))
    # print("elev - dem rms is", dr_post, "meters for dACR =", dorb," meters (post-least squares).")
    df_ = pd.concat([df_, pd.DataFrame(dr_dorb, columns=['dr/dA', 'dr/dC', 'dr/dR', 'dr/dh2'])], axis=1)
    return df_


def lstsq_demfit(df_, dem_xarr, coeff_set_re):
    df_['dR/dLON'] = get_demz_diff_at(dem_xarr, df_.LAT.values, df_.LON.values, 'lon')
    df_['dR/dLAT'] = get_demz_diff_at(dem_xarr, df_.LAT.values, df_.LON.values, 'lat')
    dforb_full = (df_.filter(regex='dR/d[A,C,R]$') +
                  df_[['dR/dLON']].values * df_.filter(regex='dLON/d[A,C,R]$').values +
                  df_[['dR/dLAT']].values * df_.filter(regex='dLAT/d[A,C,R]$').values)

    # # Add linear terms
    # _ = pd.DataFrame(np.multiply(dforb_full.filter(regex='dR/d[A,C,R]$').values, dforb.Dt.values[..., np.newaxis]),
    #                  columns=['dR/dA_1', 'dR/dC_1', 'dR/dR_1'])
    # dforb_full = pd.concat([dforb_full, _], axis=1)
    # # Add sin and cos terms to bias coeff
    # orb_period = 0.5 * 86400  # 12h to seconds
    # twopi = 2 * np.pi
    # w = twopi / orb_period
    # for cpr in n_per_orbit:
    #     cpr_str = str(cpr)
    #     coskwt = np.cos(cpr * w * df_.Dt.values)
    #     sinkwt = np.sin(cpr * w * df_.Dt.values)
    #     cos_coeffs = pd.DataFrame(
    #         np.multiply(dforb_full.filter(regex='dR/d[A,C,R]$').values, coskwt[..., np.newaxis]),
    #         columns=['dR/dAc' + cpr_str, 'dR/dCc' + cpr_str, 'dR/dRc' + cpr_str])
    #     sin_coeffs = pd.DataFrame(
    #         np.multiply(dforb_full.filter(regex='dR/d[A,C,R]$').values, sinkwt[..., np.newaxis]),
    #         columns=['dR/dAs' + cpr_str, 'dR/dCs' + cpr_str, 'dR/dRs' + cpr_str])
    #     dforb_full = pd.concat([dforb_full, cos_coeffs, sin_coeffs], axis=1)

    coeff_set_re = 'dR/'
    Amat = dforb_full.filter(regex=coeff_set_re).values[:, :]
    bvec = df_.altdiff_dem[:]
    solstd = np.linalg.lstsq(Amat, bvec, rcond=None)
    print(solstd[0])

    return solstd[0]


def print_demfit(dr_post, dt_post, dorb='', orb='xxx', dr_pre='', dt_pre=''):
    plt.clf()
    fig, [ax1, ax2] = plt.subplots(2, 1)
    dorb_plot = dorb
    dorb_plot[3:] *= 1000

    np.set_printoptions(suppress=True)

    if isinstance(dr_pre, (list, tuple, np.ndarray)):
        ax1.set_title('sol (ACR,m:mm/s):' + str(np.around(dorb, 1)) + ',  \n RMS (pre/post, m):' +
                      str(np.around(np.sqrt(np.mean(dr_pre ** 2)), 1)) + '/' +
                      str(np.around(np.sqrt(np.mean(dr_post ** 2)), 1)))
        ax1.plot(dt_pre, dr_pre, label='data_pre')
        ax2.hist(dr_pre, density=True, bins=50, label="pre-fit")
        # print(min(dr_pre), max(dr_pre))
        mean = np.mean(dr_pre)
        variance = np.var(dr_pre)
        sigma = np.sqrt(variance)
        x = np.linspace(min(dr_pre), max(dr_pre), 100)
        ax2.plot(x, stats.norm.pdf(x, mean, sigma), label="pre-fit")
        ax2.legend()
        #
        # ax2.set_title('pre-fit:' + str(np.mean(dr_pre).round(2)) + ', ' + str(np.std(dr_pre).round(2)) +
        #           ' post-fit:' + str(np.mean(dr_post).round(2)) + ', ' + str(np.std(dr_post).round(2)))
    else:
        ax1.set_title('sol (ACR,m:mm/s):' + str(np.around(dorb, 1)) + ',  \n RMS:' +
                      str(np.around(np.sqrt(np.mean(dr_post ** 2)), 1)))

    ax1.xaxis.set_ticks_position('none')
    ax1.plot(dt_post, dr_post, label='data_post')
    ax2.hist(dr_post, density=True, bins=50, label="post-fit")
    ax2.set_xlim((min(dr_post), max(dr_post)))
    mean = np.mean(dr_post)
    variance = np.var(dr_post)
    sigma = np.sqrt(variance)
    x = np.linspace(min(dr_post), max(dr_post), 100)
    ax2.plot(x, stats.norm.pdf(x, mean, sigma), label="post-fit")

    plt.legend()
    plt.savefig(XovOpt.get("tmpdir") + "dr_fit_" + orb + ".png")


if __name__ == '__main__':

    start = time.time()

    XovOpt.get("vecopts")['ALTIM_BORESIGHT'] = [0.0022105, 0.0029215, 0.9999932892]  # out[2]

    #    if local:
    #      spice.furnsh(auxdir + 'mymeta')  # 'aux/mymeta')
    #    else:
    #      spice.furnsh(['/att/nobackup/emazaric/MESSENGER/data/furnsh/furnsh.MESSENGER.def'])

    if XovOpt.get("local"):
        dem_xarr = import_dem(
            "/home/sberton2/Works/NASA/Mercury_tides/aux/HDEM_64.GRD")  # MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_4ppd_HgM008frame.GRD")
    else:
        dem_xarr = import_dem(
            "/att/nobackup/emazaric/MESSENGER/data/GDR/MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD")

    for ex in spk:

        if check_pkl:
            sol_df = pd.read_pickle("tmp/sol_demfit_" + ex + "_" + test_name + ".pkl")
            sol_df = sol_df.filter(regex='^[d,r].*$').apply(pd.to_numeric, errors='coerce', downcast='float').fillna(0)

            print(sol_df)
            plt.clf()
            plt.figure(1)
            sol_df[['rms_pre', 'rms_post']].plot(kind='hist')
            plt.xlabel("RMS (m)")
            plt.legend()
            plt.title('pre-fit:' + str(np.sqrt(np.mean(sol_df.values ** 2, axis=0))[-2]) +
                      ' post-fit:' + str(np.sqrt(np.mean(sol_df.values ** 2, axis=0))[-1]))
            plt.savefig("tmp/dr_fit_rms_" + ex + "_" + test_name + ".png")

            print("Correction avg:", sol_df.mean(axis=0))
            print("Correction rms:", np.sqrt(np.mean(sol_df.values ** 2, axis=0)))
            exit()

        if XovOpt.get("local"):
            path = '/home/sberton2/Works/NASA/Mercury_tides/out/mladata/' + epo + '_' + ex + '/gtrack_' + epo[
                                                                                                          :2] + '/' + '*.pkl'
        else:
            path = "/att/nobackup/sberton2/MLA/out/mladata/" + ex + "/gtrack_*/*.pkl"

        allFiles = glob.glob(os.path.join(path))
        allFiles = np.sort(allFiles)[1:2]
        print("Processing ", ex)
        print(os.path.join(path))
        print("nfiles: ", len(allFiles))
        # print(allFiles)

        prel_track = gtrack(XovOpt.get("vecopts"))

        sol_df = []
        # Prepare list of tracks to geolocalise
        for fil in allFiles[:]:
            prel_track = prel_track.load(fil)
            df_ = prel_track.ladata_df

            dr_post = fit_track_to_dem(df_)

        # # print(track.pertPar)
        # sol_df = pd.DataFrame(sol_df)
        # sol_df.columns=['orb'] + coeff_set + ['sig_'+str for str in coeff_set] + ['rms_pre', 'rms_post']
        #
        # # print("Final param solution: ",sol_df)
        # sol_df.to_pickle("tmp/sol_demfit_" + ex + "_"+test_name+".pkl")

    # stop clock and print runtime
    # -----------------------------
    end = time.time()
    print('----- Runtime = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
