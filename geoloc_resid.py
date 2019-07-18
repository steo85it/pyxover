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
from scipy import stats, signal


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


def get_demres(df, dorb, coeff_set=['dA', 'dC', 'dR']):
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
    # retrieve spice data for geoloc
    if not hasattr(track, 'SpObj'):
        # create interp for track
        track.interpolate()
    else:
        track.SpObj = pickleIO.load(auxdir + 'spaux_' + track.name + '.pkl')

    old_tof = track.ladata_df.loc[:, 'TOF'].values
    rng_apr = old_tof * clight / 2.
    # read just lat, lon, elev from geoloc (reads ET and TOF and updates LON, LAT, R in df)
    track.geoloc(get_partials=False)

    # Clean data --> commented out for consistency
    # print('pre',len(track.ladata_df))
    # track.ladata_df = mad_clean(track.ladata_df,'R')
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

    return elev_rms, dr, track.ladata_df.ET_TX.values - track.t0_orb


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


def mad_clean(df, column):
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


def lomb(data_df: pd.DataFrame,
         min_period: float,
         max_period: float):
    """
    Lomb–Scargle periodogram implementation
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
    periods = np.linspace(min_period, max_period, len(data_df)*100)
    freqs = 1.0 / periods
    frequency_range = 2 * np.pi * freqs
    # frequency_range = np.linspace(low_frequency, high_frequency, len(data_df))
    result = signal.lombscargle(time_stamps, samples, frequency_range, precenter=True, normalize=True)
    return result, frequency_range


def plot_pergram(dforb,filename):

    n_per_orb_min = 1
    n_per_orb_max = 86400
    orb_period = 0.5*86400 # 12h to seconds
    twopi = 2*np.pi
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
    tmp_df.plot(x='Dt', y='altdiff_dem', ax=ax1, style='.',label='data')
    ax1.xaxis.set_ticks_position('top')
    window = signal.tukey(len(tmp_df))
    ax1.plot(tmp_df.Dt.values,window*tmp_df.altdiff_dem.values,label='windata')
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

if __name__ == '__main__':

    start = time.time()

    local = 1
    debug = 0
    numer_sol = 0
    pergram = 0
    check_pkl = 0
    test_name = 'ct_40'
    coeff_set_re = '^dR/d[A,C,R]'
    n_per_orbit = [40] # 100,800] #

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

        if check_pkl:
            sol_df = pd.read_pickle("tmp/sol_demfit_" + ex + "_"+test_name+".pkl")
            sol_df = sol_df.filter(regex='^[d,r].*$').apply(pd.to_numeric, errors='coerce', downcast='float').fillna(0)

            print(sol_df)
            plt.clf()
            plt.figure(1)
            sol_df[['rms_pre', 'rms_post']].plot(kind='hist')
            plt.xlabel("RMS (m)")
            plt.legend()
            plt.title('pre-fit:' + str(np.sqrt(np.mean(sol_df.values ** 2, axis=0))[-2]) +
                      ' post-fit:' + str(np.sqrt(np.mean(sol_df.values ** 2, axis=0))[-1]))
            plt.savefig("tmp/dr_fit_rms_" + ex + "_"+test_name+".png")

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
            df_ = prel_track.ladata_df

            if len(df_ > 0):

                track = sim_gtrack(vecopts, orbID=df_.orbID.values[0])
                track.SpObj = prel_track.SpObj

                r_dem = read_dem(df_.LAT.values, df_.LON.values)

                df_['altdiff_dem'] = df_.R.values - r_dem
                # print(len(df_.R.values),len(r_dem[np.abs(r_dem)>0]))

                orb_unique = np.sort(list(set(df_.orbID.values)), axis=0)
                print(orb_unique)

                for orb in orb_unique:
                    dforb = df_[df_.orbID == orb]
                    dforb["Dt"] = dforb.ET_TX.values - dforb.ET_TX.iloc[0]
                    # store t0 orbit
                    track.t0_orb = dforb.ET_TX.iloc[0]
                    print("ET0", track.t0_orb)

                    if pergram:
                        plot_pergram(dforb,"tmp/dr_pref_res_" + orb + "_" + ex + ".png")

                    print("pre-clean:", len(dforb), dforb.altdiff_dem.abs().max())
                    dforb = mad_clean(dforb, 'altdiff_dem')
                    print("post-clean:", len(dforb), dforb.altdiff_dem.abs().max())

                    dforb['dR/dLON'] = get_demz_diff_at(dem_xarr, dforb.LAT.values, dforb.LON.values, 'lon')
                    dforb['dR/dLAT'] = get_demz_diff_at(dem_xarr, dforb.LAT.values, dforb.LON.values, 'lat')

                    dforb_full = (dforb.filter(regex='dR/d[A,C,R]$') +
                                  dforb[['dR/dLON']].values * dforb.filter(regex='dLON/d[A,C,R]$').values +
                                  dforb[['dR/dLAT']].values * dforb.filter(regex='dLAT/d[A,C,R]$').values)
                    # Add linear terms
                    # _ = pd.DataFrame(np.multiply(dforb_full.filter(regex='dR/d[A,C,R]$').values,dforb.Dt.values[..., np.newaxis]),columns=['dR/dA_1','dR/dC_1','dR/dR_1'])
                    # dforb_full = pd.concat([dforb_full,_],axis=1)

                    # Add sin and cos terms to bias coeff
                    orb_period = 0.5 * 86400  # 12h to seconds
                    twopi = 2 * np.pi
                    w = twopi / orb_period

                    for cpr in n_per_orbit:
                        cpr_str = str(cpr)
                        coskwt = np.cos(cpr * w * dforb.Dt.values)
                        sinkwt = np.sin(cpr * w * dforb.Dt.values)
                        cos_coeffs = pd.DataFrame(
                            np.multiply(dforb_full.filter(regex='dR/d[A,C,R]$').values, coskwt[..., np.newaxis]),
                            columns=['dR/dAc'+cpr_str, 'dR/dCc'+cpr_str, 'dR/dRc'+cpr_str])
                        sin_coeffs = pd.DataFrame(
                            np.multiply(dforb_full.filter(regex='dR/d[A,C,R]$').values, sinkwt[..., np.newaxis]),
                            columns=['dR/dAs'+cpr_str, 'dR/dCs'+cpr_str, 'dR/dRs'+cpr_str])
                        dforb_full = pd.concat([dforb_full, cos_coeffs, sin_coeffs], axis=1)

                    Amat = dforb_full.filter(regex=coeff_set_re).values[:, :]
                    bvec = dforb.altdiff_dem[:]
                    # print(bvec)
                    # print(Amat)
                    # exit()

                    solstd = np.linalg.lstsq(Amat, bvec, rcond=None)

                    if debug:
                        print("Sol for ", orb, ":", solstd[0], "+-", solstd[3])

                    # exit()
                    # numerical check of partials
                    # dorb = [0,0,0,0,0,0,0,0,0]
                    # elev_rms_pre, dr_min = get_demres(dforb.copy(),dorb)
                    # dorb = [0,0,0,0,0.1,0,0,0,0]
                    # elev_rms_pre, dr_plu = get_demres(dforb.copy(),dorb)
                    # print((dr_plu-dr_min)/0.1)
                    # exit()

                    # compute residuals pre-fit
                    coeff_set = dforb_full.filter(regex=coeff_set_re).columns.values
                    coeff_set = [str.split('/')[-1] for str in coeff_set]
                    dorb = np.zeros(len(coeff_set))
                    # print(coeff_set,dorb)
                    elev_rms_pre, dr_pre, dt_pre = get_demres(dforb.copy(), dorb,coeff_set=coeff_set)
                    # print(elev_rms_pre, dr_pre)

                    # iter over orbit corrections until convergence
                    maxiter = 100
                    tol = 1
                    for i in range(maxiter):
                        try:
                            # apply corrections and compute post-fit residuals
                            dorb = -1. * solstd[0]
                            elev_rms_post, dr_post, dt_post = get_demres(dforb.copy(), dorb, coeff_set=coeff_set)

                            # print(elev_rms_post, dr_post)
                            #
                            # print(len(Amat), len(bvec), len(dr_post), len(dr_pre))
                            bvec = dr_post
                            solstd = np.linalg.lstsq(Amat, bvec, rcond=None)

                            if debug:
                                print("post-fit rms for orb#", orb, " iter ", i, " :", elev_rms_post,
                                      np.sqrt(np.mean(dr_post ** 2, axis=0)))
                                print("Sol2 for ", orb, ":", solstd[0], "+-", solstd[3])

                            if max(abs(solstd[0])) < tol:
                                # if (max(abs(avgerr-avgerr_old))<tlcbnc):
                                print("Final pars values",track.pertPar)
                                # print("post-fit rms for orb#", orb, " iter ", i, " :", elev_rms_post)
                                break
                            elif i == maxiter - 1:
                                print("Max iter reached for orb ", orb, i)
                        except:
                            print("Failed for orbit ",orb,"at iter",i,"rms: ",elev_rms_pre, elev_rms_post)
                            break

                    plt.clf()
                    fig, [ax1, ax2] = plt.subplots(2, 1)
                    ax1.xaxis.set_ticks_position('top')
                    ax1.plot(dt_pre, dr_pre, label='data_pre')
                    ax1.plot(dt_post, dr_post, label='data_post')

                    # Clean up residuals for plotting/histo
                    dr_pre = mad_clean(pd.DataFrame(dr_pre, columns=['a']), 'a').a.values
                    dr_post = mad_clean(pd.DataFrame(dr_post, columns=['a']), 'a').a.values
                    sol_df.append(np.concatenate([[orb], list([v for x, v in track.pertPar.items() if x in coeff_set]),
                                                  solstd[3],
                                                  [np.sqrt(np.mean(dr_pre ** 2, axis=0))],
                                                  [np.sqrt(np.mean(dr_post ** 2, axis=0))]]))

                    print("pre-fit rms for orb#", orb, " :", elev_rms_pre, np.sqrt(np.mean(dr_pre ** 2, axis=0)))
                    # print(dr_pre)

                    ax2.hist(dr_pre, density=True, bins=50, label="pre-fit")
                    # print(min(dr_pre), max(dr_pre))

                    mean = np.mean(dr_pre)
                    variance = np.var(dr_pre)
                    sigma = np.sqrt(variance)
                    x = np.linspace(min(dr_pre), max(dr_pre), 100)
                    ax2.plot(x, stats.norm.pdf(x, mean, sigma), label="pre-fit")

                    print("post-fit rms for orb#", orb, " :", elev_rms_post, np.sqrt(np.mean(dr_post ** 2, axis=0)))

                    ax2.hist(dr_post, density=True, bins=50, label="post-fit")
                    ax2.set_xlim((min(dr_post), max(dr_post)))

                    mean = np.mean(dr_post)
                    variance = np.var(dr_post)
                    sigma = np.sqrt(variance)
                    x = np.linspace(min(dr_post), max(dr_post), 100)
                    ax2.plot(x, stats.norm.pdf(x, mean, sigma), label="post-fit")
                    plt.title('pre-fit:'+str(np.mean(dr_pre).round(2))+', '+str(np.std(dr_pre).round(2))+
                              ' post-fit:'+str(np.mean(dr_post).round(2))+', '+str(np.std(dr_post).round(2)))

                    plt.legend()
                    plt.savefig("tmp/dr_fit_" + orb + "_" + ex + "_"+test_name+".png")

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

        # print(track.pertPar)
        sol_df = pd.DataFrame(sol_df)
        sol_df.columns=['orb'] + coeff_set + ['sig_'+str for str in coeff_set] + ['rms_pre', 'rms_post']

        # print("Final param solution: ",sol_df)
        sol_df.to_pickle("tmp/sol_demfit_" + ex + "_"+test_name+".pkl")

    # stop clock and print runtime
    # -----------------------------
    end = time.time()
    print('----- Runtime = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
