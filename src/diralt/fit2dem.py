#!/usr/bin/env python3
# ----------------------------------
# Check tides from altimetry data
# ----------------------------------
# Author: Stefano Bertone
# Created: 21-Jun-2019
#

import glob
import multiprocessing as mp
import os
import sys
import time

import sys

import numpy as np
import pandas as pd
from scipy.constants import c as clight
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt

from src.pyaltsim.PyAltSim import sim_gtrack
from src.xovutil.dem_util import get_demz_tiff, get_demz_grd
# from geoloc_resid import lomb, fit_track_to_dem, import_dem, get_demz_at, read_dem
from src.pygeoloc.ground_track import gtrack
# from examples.MLA.options import XovOpt.get("vecopts"), XovOpt.get("auxdir"), XovOpt.get("SpInterp"), XovOpt.get("outdir"), XovOpt.get("local"), XovOpt.get("parallel"), XovOpt.get("tmpdir")
from config import XovOpt

from src.xovutil import astro_trans as astr, pickleIO
import spiceypy as spice
from scipy.interpolate import RectBivariateSpline
from scipy.sparse import csr_matrix

from src.pyaltsim import perlin2d
from src.diralt.geoloc_resid import lomb, fit_track_to_dem, import_dem, get_demz_at, lstsq_demfit, get_demres_full
from src.pygeoloc.ground_track import gtrack
# from examples.MLA.options import XovOpt.get("outdir"), XovOpt.get("tmpdir")
# from examples.MLA.options import XovOpt.get("vecopts"), XovOpt.get("auxdir"), XovOpt.get("SpInterp")
from config import XovOpt


#
# def import_dem(filein):
#     # grid point lists
#
#     # open netCDF file
#     # nc_file = "/home/sberton2/Downloads/sresa1b_ncar_ccsm3-example.nc"
#     nc_file = filein
#     dem_xarr = xr.open_dataset(nc_file)
#
#     return dem_xarr


# def get_demz_at(dem_xarr, lattmp, lontmp):
#     return dem_xarr.interp(lat=lattmp, lon=lontmp).to_dataframe().loc[:, 'z'].values
#
#
# def get_demz_diff_at(dem_xarr, lattmp, lontmp, axis='lon'):
#     lontmp[lontmp < 0] += 360.
#     diff_dem_xarr = dem_xarr.differentiate(axis)
#
#     lat_ax = xr.DataArray(lattmp, dims='z')
#     lon_ax = xr.DataArray(lontmp, dims='z')
#
#     return diff_dem_xarr.interp(lat=lat_ax, lon=lon_ax).z.to_dataframe().loc[:, 'z'].values


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


def plot_pergram(dforb, filename):
    from scipy import signal

    n_per_orb_min = 86400
    n_per_orb_max = 86400 * 30 * 6
    orb_period = 0.5 * 86400  # 12h to seconds
    twopi = 2 * np.pi
    w = twopi / orb_period

    # extract data from df
    tmp_df = pd.DataFrame(dforb.Dt)
    tmp_df.columns = ['Dt']
    tmp_df['altdiff_dem'] = dforb.data.values  # + # 1*np.sin(100*tmp_df.Dt.values*2*np.pi/(0.5*86400)) #
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


def generate_topo(track, res_in=0, ampl_in=1):
    np.random.seed(62)
    shape_text = 1024
    res_text = 2 ** res_in
    depth_text = 5
    size_stamp = 0.25
    amplitude = ampl_in
    noise = perlin2d.generate_periodic_fractal_noise_2d(amplitude, (shape_text, shape_text), (res_text, res_text),
                                                        depth_text, persistence=0.65)

    interp_spline = RectBivariateSpline(np.array(range(shape_text)) / shape_text * size_stamp,
                                        np.array(range(shape_text)) / shape_text * size_stamp,
                                        noise)

    track.apply_texture = interp_spline


def extract_dRvsLAT(fil,file_dem):
    track = gtrack(XovOpt.get("vecopts"))

    track = track.load(fil)
    print("Processing track #", track.name)

    if XovOpt.get("SpInterp") == 2:
        np.set_printoptions(suppress=True,
                            formatter={'float_kind': '{:0.3f}'.format})
        track.interpolate()
        # exit()

    df_data = track.ladata_df.copy()
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    # print(df_.ET_TX.round(3))
    # print(df_data.ET_TX.round(3))

    # print(df_data.columns)
    # print(df_data)
    mask = (df_data['LAT'] >= 82) & (df_data['LAT'] <= 84) & (df_data['chn'] == 0) & (df_data['LON'] >= 120) & (df_data['LON'] <= 137)
    df_data = df_data.loc[mask]

    if len(df_data) == 0:
        print("## fit2dem: no data in bounding box")
        return None

    # gmt_in = 'gmt_' + track.name + '.in'
    # print(len(r_dem),len(df_data.LAT.values),len(texture_noise),len(gmt_in))
    # df_data = df_data.loc[df_data['ET_TX'].round(3).isin(df_['ET_TX'].round(3).values)]
    # r_dem = read_dem(gmt_in, df_data.LAT.values, df_data.LON.values)
    lattmp = df_data.LAT.values
    lontmp = df_data.LON.values
    lontmp[lontmp < 0] += 360

    # print(dem_xarr)
    if file_dem.split('.')[-1] == 'GRD': # to read grd/netcdf files
        r_dem = get_demz_grd(file_dem, lontmp, lattmp) * 1.e3 #
    elif file_dem.split('.')[-1] == 'tif': #'TIF': # to read geotiffs usgs
        r_dem = np.squeeze(get_demz_tiff(file_dem, lontmp, lattmp)) * 1.e3
    # print(r_dem)
    # exit()
    # r_dem = get_demz_at(dem_xarr, lattmp, lontmp) * 1.e3
    # print(r_dem)

    df_data.loc[:, 'altdiff_dem_data'] = df_data.R.values - (r_dem)
    dr_apriori = df_data.loc[:, 'altdiff_dem_data']


    # Fit track to DEM
    dr, dr_pre, ACRcorr, df_data = fit_track_to_dem(df_data,dem_file=file_dem)
    # print(np.max(np.abs(dr)))
    # df_data.loc[:, 'altdiff_dem'] = dr
    # exit()
    # df_data.loc[:, 'dr_apriori'] = dr_pre


    # update track.previous iter
    coeff_set_re = ['sol_dR/dA', 'sol_dR/dC', 'sol_dR/dR'] #, 'sol_dR/dRl', 'sol_dR/dPt']
    tmp = pd.DataFrame(ACRcorr).T
    tmp.columns = coeff_set_re
    track.sol_prev_iter = {'orb': tmp, 'glo': ''}

    print("Saving to ",fil)
    track.save(fil)

    # print(df_data)
    # # print(dr_apriori)
    # exit()
    # print(df_data.columns)

    return df_data[['ET_TX', 'orbID', 'LAT', 'LON', 'dR_tid', 'dr_post','dr_pre']], ACRcorr #, 'dr/dA', 'dr/dC', 'dr/dR', 'dr/dh2']]


def create_amat_csr(tid_df):
    tid_df = tid_df.reset_index()

    # parOrb_xy = list(set([part.split('_')[0] for part in sorted(self.xov.parOrb_xy)]))
    # parOrb_xy = list(set([part for part in sorted(self.xov.parOrb_xy)]))
    # print(parOrb_xy)
    # parGlo_xy = sorted(self.xov.parGlo_xy)
    # xovers_df.fillna(0, inplace=True)
    parOrb_xy = ['dr/dA', 'dr/dC', 'dr/dR']
    parGlo_xy = ['dr/dh2']

    # Retrieve all orbits involved in xovers
    orb_unique = set(tid_df.orbID)

    # select cols
    OrbParFull = [x + '_' + y.split('_')[0] for x in orb_unique for y in parOrb_xy]
    Amat_col = list(set(OrbParFull)) + parGlo_xy
    # print(Amat_col)

    dict_ = dict(zip(Amat_col, range(len(Amat_col))))
    print(dict_)

    # Retrieve and re-organize partials w.r.t. observations, parameters and orbits

    # Set-up columns to extract
    # Extract from df to np arrays for orbit A and B, then stack togheter all partials for
    # parameters/observations for each orbit (dR/dp_1,...,dR/dp_n,orb,xovID)
    csr = []

    partder = tid_df.loc[:, parOrb_xy].values
    orb_loc = tid_df.loc[:, 'orbID'].values
    #
    # print([str(x) + '_' + str(y).split('_')[0] for x in orb_loc for y in parOrb_xy][:3])
    col = np.array(
        list(map(dict_.get, [str(x) + '_' + str(y).split('_')[0] for x in orb_loc for y in parOrb_xy])))
    # print("cols",col)

    # row = np.repeat(xovers_df.xOvID.values, len(par_xy_loc))
    row = np.repeat(tid_df.index.values, len(parOrb_xy))
    # print("rows",row)
    val = partder.flatten()
    # print("vals", val)
    # print(tid_df)
    # exit()

    # print(len(orb_loc), len(Amat_col))
    csr.append(csr_matrix((val, (row, col)), dtype=np.float32, shape=(len(orb_loc), len(Amat_col))))
    csr.append(csr_matrix((tid_df['dr/dh2'].values, (tid_df.index.values, np.repeat(dict_['dr/dh2'], len(tid_df)))),
                          dtype=np.float32, shape=(len(orb_loc), len(Amat_col))))
    # print(csr)

    csr = sum(csr)
    print(csr)
    # print("shape csr",np.shape(csr.todense()))
    # print(list([np.array(map({v: k for k, v in dict_.items()}.get, csr.indices)), csr.data]))
    # print(sys.getsizeof(csr))

    # Save A and b matrix/array for least square (Ax = b)
    spA = csr
    b = tid_df.altdiff_dem_data.values
    # print(b)

    sol = lsqr(spA, b, damp=10, show=False, iter_lim=5000, atol=1.e-9, btol=1.e-9, calc_var=True)
    print(sol)

    dict_ = dict(zip(dict_.keys(), sol[0]))
    print(max(dict_, key=dict_.get), dict_[max(dict_, key=dict_.get)])
    print(min(dict_, key=dict_.get), dict_[min(dict_, key=dict_.get)])

    fig, ax1 = plt.subplots(1, 1)

    ax1.hist(sol[0][:-1], density=False, bins=200, label="post-fit")
    ax1.set_xlim(-500, 500)  # (min(sol[0][:-1]), max(sol[0][:-1])))
    plt.legend()
    plt.savefig(XovOpt.get("tmpdir") + "dem_lsqfit_" + exp + ".png")

    print("h2 solution is 0.8 (a priori)+", sol[0][-1], "with error ", sol[-1][-1])

    return sol


if __name__ == '__main__':

    # from examples.MLA.options import XovOptgetexpopt
    from config import XovOpt

    start = time.time()

    if len(sys.argv) > 1:
        args = sys.argv[1:]
        ym = args[0]
    else:
        ym = '*'
        print("Taking all files in, else use as python3 fit2dem.py YYMM")
        # exit()

    if False:
        from datetime import datetime

        tid_df = pd.read_pickle("/explore/nobackup/people/sberton2/MLA/tmp/tid_lat_time.pkl")
        # tid_df.columns = tid_df.columns.map(''.join)
        # tid_df = tid_df.fillna(0).sort_values(by=['LAT']).reset_index()
        # tid_median_df = pd.DataFrame([tid_df.set_index('LAT').filter(regex='????'+f'{day:02}').median(axis=1) for day in range(0,31)]).transpose()
        # tid_median_df.columns = [''+f'{day:02}' for day in range(0,31)]
        tid_median_df = tid_df
        tid_median_df = tid_median_df.fillna(0).drop(columns=['index']).set_index('LAT')
        # print(tid_median_df.loc[:, (tid_median_df != 0).any(axis=0)])

        fig, ax1 = plt.subplots(nrows=1)

        secs = []
        for dt in tid_median_df.columns.values:
            secs.append(datetime.strptime(dt, '%y%m%d%H%M').timestamp())  # Get timezone naive now
        tid_median_df.columns = secs
        print(tid_median_df)
        tmp = tid_median_df.transpose()
        print(tmp)
        # exit()
        # tmp.columns = tid_median_df['LAT'].values
        # print(tmp.drop(['index','LAT']))
        tmp[50.0].plot(ax=ax1)

        # plt.tight_layout()
        # ax1.invert_yaxis()
        #         ylabel='Topog ampl rms (1st octave, m)')
        fig.savefig(
            '/explore/nobackup/people/sberton2/MLA/tmp/tid_median_df.png')  # '/home/sberton2/Works/NASA/Mercury_tides/PyXover/tmp/tid_median_df.png')

        tmp = tmp[50.0].reset_index()
        tmp.columns = ['Dt', 'data']
        plot_pergram(tmp,
                     '/explore/nobackup/people/sberton2/MLA/tmp/tid_pergram.png')  # /home/sberton2/Works/NASA/Mercury_tides/PyXover/tmp/tid_pergram.png')

        exit()

    debug = 0
    numer_sol = 1
    read_pkl = 0
    dem_opt = 'sfs'

    # epo = '1212'
    exp = XovOpt.get("expopt") + '_0' # 'KX1r4_0'  # 'KX0_0'
    rghn = '0res_1amp'

    if read_pkl:

        tid_df = pd.DataFrame()
        if XovOpt.get("local"):
            for f in glob.glob(XovOpt.get("tmpdir") + "tid_" + str(exp) + "_????.pkl"):
                # print(f)
                # print(pd.read_pickle(f))
                tid_df = tid_df.append(pd.read_pickle(f), ignore_index=True)
        else:
            for f in glob.glob(XovOpt.get("tmpdir") + "tid_" + str(exp) + "_12??.pkl"):
                print("Processing", f)
                tid_df = tid_df.append(pd.read_pickle(f), ignore_index=True)

        # print(tid_df)
        # tid_df = pd.concat(tid_df)
        # tid_df.columns = tid_df.columns.map(''.join)
        tid_df = tid_df.fillna(0).sort_values(by=['ET_TX']).reset_index(drop=True)
        #
        # print(tid_df)
        # exit()

        print(tid_df.columns)

        amat = tid_df[['altdiff_dem_data', 'dr/dA',
                       'dr/dC', 'dr/dR', 'dr/dh2']]

        create_amat_csr(tid_df)

    else:

        if XovOpt.get("local"):
            # dem = "/home/sberton2/Downloads/Mercury_Messenger_USGS_DEM_NPole_665m_v2.tif" #
            if dem_opt == 'sfs':
                dem = "/home/sberton2/tmp/run-DEM-final.tif" #
            elif dem_opt == 'hdem':
                dem = XovOpt.get("auxdir") + 'HDEM_64.GRD'  # ''MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD'
        else:
            dem = '/explore/nobackup/people/emazaric/MESSENGER/data/GDR/HDEM_64.GRD'  # MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD'
        # dem_xarr = import_dem(dem)

        if XovOpt.get("local"):
            spice.furnsh(XovOpt.get("auxdir") + 'mymeta')  # 'aux/mymeta')
        else:
            spice.furnsh(['/explore/nobackup/people/emazaric/MESSENGER/data/furnsh/furnsh.MESSENGER.def'])
            # ,
            # '/explore/nobackup/people/sberton2/MLA/aux/spk/Genovaetal_DE432_Mercury_05min.bsp',
            # '/explore/nobackup/people/sberton2/MLA/aux/spk/MSGR_HGM008_INTGCB.bsp'])

        if XovOpt.get("local"):
            path = XovOpt.get("outdir") + 'sim/' + exp + '/' + rghn + '/gtrack_' + str(ym)[:2] + '/gtrack_' + str(ym) + '*.pkl'
        else:
            path = XovOpt.get("outdir") + 'sim/' + exp + '/' + rghn + '/gtrack_' + str(ym)[:2] + '/gtrack_' + str(ym) + '*.pkl'
        # path = '/home/sberton2/Works/NASA/Mercury_tides/out/sim/'+ spk + '/3res_20amp/gtrack_*' + '/' + '*.pkl'

        allFiles = glob.glob(os.path.join(path))
        allFiles = np.sort(allFiles)[:]
        # print("Processing ", ex)
        print(os.path.join(path))
        print("nfiles: ", len(allFiles))
        print(allFiles)

        prel_track = gtrack(XovOpt.get("vecopts"))

        sol_df = []
        tid_df = []
        ACRcorr = []

        fit2dem = False

        if fit2dem:
            # loop over all gtracks
            if XovOpt.get("parallel"):
                # print((mp.cpu_count() - 1))
                pool = mp.Pool(processes=mp.cpu_count())
                tid_df = pool.map(extract_dRvsLAT, allFiles)  # parallel
                pool.close()
                pool.join()
            else:
                #            tid_df = [extract_dRvsLAT(fil) for fil in allFiles[:]]
                for fil in allFiles[:]:
                    try:
                        df, ACR = extract_dRvsLAT(fil, file_dem=dem)
                        tid_df.append(df)
                        ACRcorr.append(ACR)
                        # exit()
                    except:
                        print("Issue with:", fil)
                        pass

            ACRcorr = pd.DataFrame(ACRcorr,columns=['dA','dC','dR'])
            # print(ACRcorr)
            df = pd.concat(tid_df)
            # print(df)

            pd.to_pickle(ACRcorr, XovOpt.get("tmpdir") + 'mla_corr_' + dem_opt + '.pkl')
            pd.to_pickle(df, XovOpt.get("tmpdir") + 'mla_res_' + dem_opt + '.pkl')
        else:
            ACRcorr = pd.read_pickle(XovOpt.get("tmpdir") + 'mla_corr_' + dem_opt + '.pkl')
            df = pd.read_pickle(XovOpt.get("tmpdir") + 'mla_res_' + dem_opt + '.pkl')

        # print(df.columns)
        df = df.loc[:,['LON','LAT','altdiff_dem_data','dr_pre','dr_post']].reset_index()

        fig, axs = plt.subplots(4)
        df.abs().plot(kind='scatter', x='LON', y='LAT', c='dr_pre', s=0.2,colormap='YlOrRd',ax=axs[0],vmax=50)
        df.loc[:,'dr_pre'].abs().hist(ax=axs[1],bins=200)
        axs[1].set_xlim((0,50))


        df.abs().plot(kind='scatter', x='LON', y='LAT', c='dr_post', s=0.2,colormap='YlOrRd',ax=axs[2],vmax=50)
        df.loc[:,'dr_post'].abs().hist(ax=axs[3],bins=200)
        axs[3].set_xlim((0,50))

        plt.savefig(XovOpt.get("tmpdir") + 'diralt_res_' + dem_opt + '.png')
        exit()


        if False:
           tid_df = pd.concat(tid_df)
           # tid_df.columns = tid_df.columns.map(''.join)
           tid_df = tid_df.fillna(0).sort_values(by=['ET_TX'])  # .reset_index()

           print(tid_df.columns)

           # amat = tid_df[['altdiff_dem_data', 'dr/dA',
           #    'dr/dC', 'dr/dR', 'dr/dh2']]

           # create_amat_csr(tid_df)

           tid_df.to_pickle(tmpdir + "tid_" + str(exp) + "_" + str(ym) + ".pkl")

           fig, ax1 = plt.subplots(nrows=1)

           # create data
           x = tid_df.dR_tid  # altdiff_dem
           # if y = sim with tides and no errors -> horiz line,
           # else if tides not included in model, and no other errors, 1:1 bisec
           y = tid_df.altdiff_dem_data * -1.

           # Big bins
           plt.hist2d(x, y, bins=(50, 50), cmap=plt.cm.jet)
           fig.savefig(
               tmpdir + 'tid_histo_' + exp + '_' + str(
                   ym) + '.png')  # '/home/sberton2/Works/NASA/Mercury_tides/PyXover/tmp/tid_median_df.png')

    end = time.time()
    print('----- Runtime TidTest tot = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')

