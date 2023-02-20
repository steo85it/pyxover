#!/usr/bin/env python3
# ----------------------------------
# PyXover
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-Oct-2018
#
from fileinput import filename
import warnings

from pyaltsim.prepro import prepro_ilmNG, prepro_BELA_sim
from xovutil.dem_util import get_demz_at, import_dem, get_demz_tiff
from xovutil.icrf2pbf import icrf2pbf
from xovutil.orient_setup import orient_setup

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import shutil
import glob
import time

import numpy as np
import pandas as pd
from scipy.constants import c as clight
from scipy.interpolate import RectBivariateSpline
import multiprocessing as mp
import subprocess

import spiceypy as spice
import matplotlib.pyplot as plt

# mylib
from config import XovOpt

from xovutil import astro_trans as astr, pickleIO
from pygeoloc.ground_track import gtrack
from geolocate_altimetry import get_sc_ssb, get_sc_pla
from pyaltsim import perlin2d

########################################
# start clock
start = time.time()


##############################################
class sim_gtrack(gtrack):
    def __init__(self, opts, orbID):
        gtrack.__init__(self, opts)
        self.orbID = orbID
        self.name = str(orbID)
        self.outdir = None
        self.slewdir = None
        # self.ran_noise = None
        self.rdr_df = None

    def setup(self, df):
        df_ = df.copy()

        # get time of flight in ns from probe one-way range in km
        df_['TOF'] = df_['rng'] * 2. * 1.e3 / clight
        # preparing df for geoloc
        df_['seqid'] = df_.index
        df_ = df_.rename(columns={"epo_tx": "ET_TX"})
        df_ = df_.reset_index(drop=True)
        # copy to self
        self.ladata_df = df_[['ET_TX', 'TOF', 'orbID', 'seqid']]

        # retrieve spice data for geoloc from interp, if desired
        if XovOpt.get("SpInterp") > 0:
            if not os.path.exists(XovOpt.get("auxdir") + XovOpt.get("spauxdir") +
                               'spaux_' + self.name + '.pkl') or \
                    XovOpt.get("SpInterp") == 2:
                # create interp for track
                self.interpolate()
            else:
                self.SpObj = pickleIO.load(XovOpt.get("auxdir") + XovOpt.get("spauxdir") +
                                       'spaux_' + self.name + '.pkl')

        # actual processing
        self.lt_topo_corr(df=df_)

        # add range noise
        if XovOpt.get("range_noise"):
            mean = 0.
            std = 0.2
            self.add_range_noise(df_,mean,std)

        self.setup_rdr()

    #@staticmethod
    def add_range_noise(self, df_, mean=0., std=0.2):
        """
        Add range noise (normal distribution) to simulated time of flight (seconds)
        :param df_: input altimetry dataframe
        :param mean: mean value for normal distribution (meters)
        :param std: standard deviation for normal distribution (meters)
        """
        np.random.seed(int(self.name))
        tof_noise = (std * np.random.randn(len(df_)) + mean) / clight
        df_.loc[:, 'TOF'] += tof_noise
        if XovOpt.get("debug"):
            plt.plot(df_.loc[:, 'ET_TX'], df_.TOF, 'bo', df_.loc[:, 'ET_TX'], df_.TOF - tof_noise, 'k')
            plt.savefig('tmp/noise.png')

    def lt_topo_corr(self, df, itmax=50, tol=5.e-2):
        """
        iterate from a priori rough TOF @ ET_TX to account for light-time and
        terrain roughness and topography
        et, rng0 -> lon0, lat0, z0 (using geoloc)
        lon0, lat0 -> DEM elevation (using GMT), texture from stamp
        dz -> difference btw z0 from geoloc and "real" elevation z at lat0/lon0
        update range and tof -> new_rng = old_rng + dz

        :param df: table with tof, et (+all apriori data)
        :param itmax: max iters allowed
        :param tol: tolerance for convergence
        """
        # self.ladata_df[["TOF"]] = self.ladata_df.loc[:,"TOF"] + 200./clight

        # a priori values for internal FULL df
        df.loc[:, 'converged'] = False
        df.loc[:, 'offnadir'] = 0

        for it in range(itmax):

            # tof and rng from previous iter as input for new geoloc
            old_tof = self.ladata_df.loc[:, 'TOF'].values
            rng_apr = old_tof * clight / 2.

            # read just lat, lon, elev from geoloc (reads ET and TOF and updates LON, LAT, R in df)
            self.geoloc()
            lontmp, lattmp, rtmp = np.transpose(self.ladata_df[['LON', 'LAT', 'R']].values)
            r_bc = rtmp + XovOpt.get("vecopts")['PLANETRADIUS'] * 1.e3

            # use lon and lat to get "real" elevation from map
            radius = self.get_topoelev(lattmp, lontmp)

            # use "real" elevation to get bounce point coordinates
            bcxyz_pbf = astr.sph2cart(radius, lattmp, lontmp)
            bcxyz_pbf = np.transpose(np.vstack(bcxyz_pbf))

            # get S/C body fixed position (useful to update ranges, has to be computed on reduced df)
            scxyz_tx_pbf = self.get_sc_pos_bf(self.ladata_df)
            # compute range btw probe@TX and bounce point@BC (no BC epoch needed, all coord planet fixed)
            rngvec = (bcxyz_pbf - scxyz_tx_pbf)
            # compute correction for off-nadir observation
            rngvec_normed = rngvec/np.linalg.norm(rngvec, axis=1)[:, np.newaxis]
            scxyz_tx_pbf_normed = np.array(scxyz_tx_pbf)/np.linalg.norm(scxyz_tx_pbf, axis=1)[:, np.newaxis]
            # print(np.max(np.abs(np.einsum('ij,ij->i', rngvec_normed, -scxyz_tx_pbf_normed))))
            # compute correction for off-nadir observation (with check to avoid numerical issues on arccos)
            if np.max(np.abs(np.einsum('ij,ij->i', rngvec_normed, -scxyz_tx_pbf_normed))) <= 1:
                offndr = np.arccos(np.einsum('ij,ij->i', rngvec_normed, -scxyz_tx_pbf_normed))
            else:
                offndr = 0.
            # offndr = np.arccos(np.einsum('ij,ij->i', rngvec, -scxyz_tx_pbf) /
            #                    np.linalg.norm(rngvec, axis=1) /
            #                    np.linalg.norm(scxyz_tx_pbf, axis=1))

            # compute residual between "real" elevation and geoloc (based on a priori TOF)
            dr = (r_bc - radius) * np.cos(offndr)

            # update range
            rng_new = rng_apr + dr

            # update tof
            tof = 2. * rng_new / clight

            self.ladata_df.loc[:, 'TOF'] = tof  # convert to update
            self.ladata_df.loc[:, 'converged'] = abs(dr) < tol
            self.ladata_df.loc[:, 'offnadir'] = np.rad2deg(offndr)

            if it == 0:
                df = self.ladata_df.copy()
            else:
                df.update(self.ladata_df)

            percent_left = 100. - (len(df) - np.count_nonzero(abs(dr) > tol))/len(df)*100.

            if XovOpt.get("debug"):
                print("it = " + str(it))
                print("max resid:", max(abs(dr)), "# > tol:", np.count_nonzero(abs(dr) > tol), percent_left,' %')

            if (max(abs(dr)) < tol):
                # print("Convergence reached")
                # pass all epochs to next step
                self.ladata_df = df.copy()
                break
            elif it > 10 and percent_left < 5:
                print('### altsim: Most data point converged!')
                print("it = " + str(it))
                print("max resid:", max(abs(dr)), "# > tol:", np.count_nonzero(abs(dr) > tol), percent_left,' %')
                print('offnadir max', max(np.rad2deg(offndr)))
                self.ladata_df = df.copy()  # keep non converged but set chn>5 (bad msrmts)
                break
            elif it == itmax - 1:
                print('### altsim: Max number of iterations reached!')
                print("it = " + str(it))
                print("max resid:", max(abs(dr)), "# > tol:", np.count_nonzero(abs(dr) > tol))
                print('offnadir max', max(np.rad2deg(offndr)))
                self.ladata_df = df.copy()  # keep non converged but set chn>5 (bad msrmts)
                break
            else:
                # update global df used in geoloc at next iteration (TOF)
                # df = df[df.loc[:, 'offnadir'] < 5]
                # only operate on non-converged epochs for next iteration
                self.ladata_df = df[df.loc[:, 'converged'] == False].copy()
            # self.ladata_df = df.copy()

    def get_topoelev(self, lattmp, lontmp):

        if XovOpt.get("apply_topo"):
            # st = time.time()

            # if not XovOpt.get("local"):
            #     if XovOpt.get("instrument") == "LOLA":
            #         dem = self.slewdir+"/SLDEM2015_512PPD.GRD"
            #     else:
            #         dem = '/att/nobackup/emazaric/MESSENGER/data/GDR/HDEM_64.GRD' #MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD'
            # else:
            #     dem = XovOpt.get("auxdir") + 'HDEM_64.GRD'  # ''MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD'
            #
            # # if gmt==False don't use grdtrack, but interpolate once using xarray and store interp
            gmt = False

            if XovOpt.get("instrument") == 'BELA':
                if XovOpt.get("local"):
                    geotiff = {'global': f'{XovOpt.get("auxdir")}dem/Mercury_Messenger_USGS_DEM_Global_665m_v2.tif',
                               'NP': f'{XovOpt.get("auxdir")}dem/Mercury_Messenger_USGS_DEM_NPole_665m_v2_32bit.tif',
                               'SP': f'{XovOpt.get("auxdir")}dem/Mercury_Messenger_USGS_DEM_SPole_665m_v2_32bit.tif'}
                else:
                    geotiff = [f'{XovOpt.get("auxdir")}dem/Mercury_Messenger_USGS_DEM_Global_665m_v2.tif',
                               f'{XovOpt.get("auxdir")}dem/Mercury_Messenger_USGS_DEM_SPole_665m_v2_32bit.tif']

                df = pd.DataFrame(zip(lattmp, lontmp), columns=['LAT', 'LON']) #.reset_index()
                # nice but not broadcasted... slow
                # df['r_dem'] = df.apply(lambda x: get_demz_tiff(geotiff[0],lat=x.LAT,lon=x.LON) if x.LAT > 30
                #                         else get_demz_grd(filin=dem,lon=x.LON,lat=x.LAT), axis=1)

                mask_np = (df['LAT'] >= 70)
                mask_equat = (df['LAT'] < 70) & (df['LAT'] > -70)
                mask_sp = (df['LAT'] <= -70)

                df['r_dem'] = 0
                # NP (MLA DEM)
                if len(df.loc[mask_np,:])>0:
                    # df.loc[mask_np, 'r_dem'] = get_demz_grd(filin=dem,lon=df.loc[mask_np,'LON'].values,lat=df.loc[mask_np,'LAT'].values).T
                    df.loc[mask_np, 'r_dem'] = np.squeeze(get_demz_tiff(filin=geotiff['NP'],
                                                                           lon=df.loc[mask_np,'LON'].values,
                                                                           lat=df.loc[mask_np,'LAT'].values).T)
                # EQUAT (USGS)
                if len(df.loc[mask_equat,:])>0:
                    df.loc[mask_equat, 'r_dem'] = np.squeeze(get_demz_tiff(filin=geotiff['global'],
                                                                           lon=df.loc[mask_equat,'LON'].values,
                                                                           lat=df.loc[mask_equat,'LAT'].values).T)
                # SP (USGS)
                if len(df.loc[mask_sp,:])>0:
                    df.loc[mask_sp, 'r_dem'] = np.squeeze(get_demz_tiff(filin=geotiff['SP'],
                                                                        lon=df.loc[mask_sp,'LON'].values,
                                                                        lat=df.loc[mask_sp,'LAT'].values).T)

                r_dem = df.r_dem.values

            elif not gmt:

                if self.dem == None:
                    print(self.dem)
                    self.dem = import_dem(filein=self.dem,outdir=f"{self.slewdir}/")
                else:
                    print("DEM already read")
                    pass
            else:
                print("Using grdtrack")

            if gmt and XovOpt.get("instrument") == 'LOLA':
                gmt_in = 'gmt_' + self.name + '.in'
                if os.path.exists('tmp/' + gmt_in):
                    os.remove('tmp/' + gmt_in)

                np.savetxt('tmp/' + gmt_in, list(zip(lontmp, lattmp, self.ladata_df.seqid.values)))

                if XovOpt.get("local") == 0:
                    if XovOpt.get("instrument") == 'LOLA':
                        if local_dem:
                            dem = self.slewdir + "/SLDEM2015_512PPD.GRD"
                        else:
                            dem = "/att/projrepo/PGDA/LOLA/data/LOLA_GDR/CYLINDRICAL/raw/LDEM_4.GRD"
                    else:
                        dem = '/att/nobackup/emazaric/MESSENGER/data/GDR/MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD'
                #             r_dem = subprocess.check_output(
                #                 ['grdtrack', gmt_in,
                #                  '-G' + dem],
                #                 universal_newlines=True, cwd='tmp')
                #             r_dem = np.fromstring(r_dem, sep=' ').reshape(-1, 3)[:, 2]
                # # np.savetxt('gmt_'+self.name+'.out', r_dem)

                else:
                    dem = XovOpt.get("instrument") + 'SLDEM2015_512PPD.GRD'
                    # r_dem = np.loadtxt('tmp/gmt_' + self.name + '.out')

                # print(['grdtrack', gmt_in, '-G' + dem,'-R0.0/360.0/-50.0/50.0'])
                if local_dem:
                    r_dem = subprocess.check_output(
                        ['grdtrack', gmt_in, '-G' + dem],
                        universal_newlines=True, cwd='tmp')
                else:  # replace -RLON0/LONMAX/LAT0/LATMAX with appropriate bbox
                    r_dem = subprocess.check_output(
                        ['grdtrack', gmt_in, '-G' + dem, '-R0.0/360.0/-50.0/50.0'],
                        universal_newlines=True, cwd='tmp')
                if len(r_dem) == 0:
                    print("Weird empty grdtrack output, please check")
                    exit()

                # r_dem = np.fromstring(r_dem, sep=' ').reshape(-1, 3)[:, 2]
                r_dem = np.fromstring(r_dem, sep=' ').reshape(-1, 4)[:, 2:]

                df_ = pd.DataFrame(r_dem, columns=['seqid', 'elevation']).set_index('seqid')
                new_index = self.ladata_df.seqid.values
                r_dem = np.transpose(df_.reindex(new_index).fillna(0).values).flatten()

            elif gmt and XovOpt.get("instrument") != 'BELA':
                gmt_in = 'gmt_' + self.name + '.in'
                if os.path.exists('tmp/' + gmt_in):
                    os.remove('tmp/' + gmt_in)
                np.savetxt('tmp/' + gmt_in, list(zip(lontmp, lattmp)))

                r_dem = subprocess.check_output(
                ['grdtrack', gmt_in, '-G' + dem],
                universal_newlines=True, cwd='tmp')
                r_dem = np.fromstring(r_dem, sep=' ').reshape(-1, 3)[:, 2]
            elif XovOpt.get("instrument") != 'BELA':
                print("## Using weird combination (not BELA).")
                lontmp[lontmp < 0] += 360.
                r_dem = get_demz_at(self.dem, lattmp, lontmp)

                # Works but slower (interpolates each time, could be improved by https://github.com/JiaweiZhuang/xESMF/issues/24)
                # radius_xarr = dem_xarr.interp(lon=xr.DataArray(lontmp, dims='z'), lat= xr.DataArray(lattmp, dims='z')).z.values * 1.e3 #

            # Convert to meters (if DEM given in km)
            r_dem *= 1.e3

            # TODO replace with "small_scale_topo/texture_noise" option
            if XovOpt.get("small_scale_topo") and XovOpt.get("instrument") != "LOLA":
                texture_noise = self.apply_texture(np.mod(lattmp, 0.25), np.mod(lontmp, 0.25), grid=False)
                # print("texture noise check",texture_noise,r_dem)
            else:
                texture_noise = 0.
                
            # update Rmerc with r_dem/text (meters)
            radius = XovOpt.get("vecopts")['PLANETRADIUS'] * 1.e3 + r_dem + texture_noise
            # print("radius etc",radius,r_dem,texture_noise)
        else:
            radius = XovOpt.get("vecopts")['PLANETRADIUS'] * 1.e3

        return radius


    def get_sc_pos_bf(self, df):
        et_tx = df.loc[:, 'ET_TX'].values
        sc_pos, sc_vel = get_sc_ssb(et_tx, self.SpObj, self.pertPar, self.vecopts)
        scpos_tx_p, _ = get_sc_pla(et_tx, sc_pos, sc_vel, self.SpObj, self.vecopts)
        if XovOpt.get('body') == 'MOON':
            pxform_array = np.frompyfunc(spice.pxform, 3, 1)
            tsipm = pxform_array(XovOpt.get("vecopts")['INERTIALFRAME'],XovOpt.get("vecopts")['PLANETFRAME'],et_tx) 
        else:
            rotpar, upd_rotpar = orient_setup(self.pertPar['dRA'], self.pertPar['dDEC'], self.pertPar['dPM'],
                                          self.pertPar['dL'])
            tsipm = icrf2pbf(et_tx, upd_rotpar)
        scxyz_tx_pbf = np.vstack([np.dot(tsipm[i], scpos_tx_p[i]) for i in range(0, np.size(scpos_tx_p, 0))])
        return scxyz_tx_pbf


    def setup_rdr(self):
        df_ = self.ladata_df.copy()

        # only select nadir data
        # df_ = df_[df_.loc[:,'offnadir']<5]

        mlardr_cols = ['geoc_long', 'geoc_lat', 'altitude', 'EphemerisTime', 'MET', 'frm',
                       'chn', 'Pulswd', 'thrsh', 'gain', '1way_range', 'Emiss', 'TXmJ',
                       'UTC', 'TOF_ns_ET', 'Sat_long', 'Sat_lat', 'Sat_alt', 'Offnad', 'Phase',
                       'Sol_inc', 'SCRNGE', 'seqid']
        self.rdr_df = pd.DataFrame(columns=mlardr_cols)

        # assign "bad chn" to non converged observations
        df_['chn'] = 0
        df_.loc[df_['converged'] == False, 'chn'] = 10

        # update other columns for compatibility with real data format
        df_['TOF_ns_ET'] = np.round(df_['TOF'].values * 1.e9, 10)
        df_['UTC'] = pd.to_datetime(df_['ET_TX'], unit='s',
                                    origin=pd.Timestamp('2000-01-01T12:00:00'))

        df_ = df_.rename(columns={'ET_TX': 'EphemerisTime',
                                  'LON': 'geoc_long', 'LAT': 'geoc_lat', 'R': 'altitude',
                                  })
        df_ = df_.reset_index(drop=True)
        # if XovOpt.get("local"):
        self.rdr_df = pd.concat([self.rdr_df,df_[['EphemerisTime', 'geoc_long', 'geoc_lat', 'altitude',
                                              'UTC', 'TOF_ns_ET', 'chn', 'seqid']]])[mlardr_cols]
        # else:
        #     self.rdr_df = self.rdr_df.append(df_[['EphemerisTime', 'geoc_long', 'geoc_lat', 'altitude',
        #                                           'UTC', 'TOF_ns_ET', 'chn', 'seqid']], sort=True)[mlardr_cols]


##############################################


def sim_track(args):
    # tracks: (sim_gtrack(XovOpt.get("vecopts"), i)
    # df: "simil-illumNG prediction data frame
    # i: i in list(df.groupby('orbID').groups.keys()))
    # outdir_: Input/Output directory?
    track, df, i, outdir_ = args

    if track.XovOpt.get("instrument") == "LOLA":
        track.slewdir = track.XovOpt.get("auxdir") + outdir_.split('/')[-3]
    else:
        assert track.slewdir == None

    filename = outdir_ + f'{XovOpt.get("instrument")}SIMRDR' + track.name + '.TAB'
    if os.path.isfile(filename) == False:
        track.setup(df[df['orbID'] == i])
        track.rdr_df.to_csv(filename, index=False, sep=',', na_rep='NaN')
        print('Simulated observations written to', filename)
    else:
        print('Simulated observations ', filename + ' already exists. Skip.')


def main(args):  # dirnam_in = 'tst', ampl_in=35,res_in=0):
    import datetime as dt

    ampl_in = args[0]   # Ampl directory?
    res_in = args[1]    # Result directory?
    dirnam_in = args[2] # Input directory? 
    epos_in = args[3]   # Month to simulate (format: YYMM)
    opts = args[4]      # Options (dictionnary)

    print("arg")
    print(*args)

    if len(args)<6:
        if XovOpt.get("instrument") != "LOLA":  # if BELA/CALA
            # generate list of epoch within selected month and given sampling rate (fixed to 10 Hz)
            from calendar import monthrange

            days_in_month = monthrange(int('20'+epos_in[:2]), int(epos_in[2:]))

            d_first = dt.datetime(int('20'+epos_in[:2]), int(epos_in[2:]), int('01'),1,00,00) # TODO avoiding issues with 30-Apr 23:59:59 ... extend spk

            # if test, avoid computing tons of files
            if XovOpt.get("unittest"):
                d_last = dt.datetime(int('20'+epos_in[:2]), int(epos_in[2:]), int('02'),5,00,00) # for testing
            else:
                d_last = dt.datetime(int('20'+epos_in[:2]), int(epos_in[2:]), int(days_in_month[-1]),23,59,59)
    else:
        d_first = args[5]
        d_last = args[6]


    # update options (needed when sending to slurm)
    XovOpt.clone(opts)

    # locate data
    data_pth = f'{XovOpt.get("rawdir")}'
    dataset = dirnam_in
    data_pth += dataset

    if XovOpt.get("SpInterp") in [0, 2]:
        # load kernels
        if not XovOpt.get("local"): # WD: Messenger related??
            spice.furnsh([f'{XovOpt.get("auxdir")}furnsh.MESSENGER.def',
                          f'{XovOpt.get("auxdir")}mymeta_pgda'])
        else:
            spice.furnsh(f'{XovOpt.get("auxdir")}mymeta')

    if XovOpt.get("instrument") == "LOLA":
        path_illumng = f'{XovOpt.get("auxdir")}{epos_in}/slewcheck_{ampl_in}/'

    print('dirnam_in', dirnam_in)
    print('epos_in', epos_in)

    if not os.path.exists(XovOpt.get('tmpdir')):
        os.makedirs(XovOpt.get('tmpdir'))

    # if not XovOpt.get("local"):
        # data_pth = '/att/nobackup/sberton2/MLA/data/MLA_'+epos_in[:2]  # /home/sberton2/Works/NASA/Mercury_tides/data/'
        # dataset = ''  # 'small_test/' #'test1/' #'1301/' #
        # data_pth += dataset
        # TODO Avoid/remove explicit paths!!!
        # load kernels
        # if XovOpt.get("instrument") == "BELA":
        #     spice.furnsh(['/att/nobackup/emazaric/MESSENGER/data/furnsh/furnsh.MESSENGER.def'
        #                  ,
        #                  '/att/nobackup/sberton2/MLA/aux/spk/bc_sci_v06.tf',
        #                  '/att/nobackup/sberton2/MLA/aux/spk/bc_mpo_mlt_50037_20260314_20280529_v03.bsp']
        #     )  # 'aux/mymeta')
        # else:
        #     spice.furnsh('/att/nobackup/emazaric/MESSENGER/data/furnsh/furnsh.MESSENGER.def')

    # else:
        # data_pth = '/home/sberton2/Works/NASA/Mercury_tides/data/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
        # dataset = "test/"  # ''  # 'small_test/' #'1301/' #
        # data_pth += dataset
    # load kernels
    # TODO adapt for pgda w/o mentioning paths
    if not XovOpt.get("instrument") == 'LOLA':
        spice.furnsh(XovOpt.get("auxdir") + 'mymeta')  # 'aux/mymeta')

    if XovOpt.get("parallel"):
        # set ncores
        ncores = mp.cpu_count() - 1  # 8
        print('Process launched on ' + str(ncores) + ' CPUs')

    # out = spice.getfov(vecopts['INSTID'][0], 1)
    # updated w.r.t. SPICE from Mike's scicdr2mat.m
    if XovOpt.get("instrument") == 'LOLA':
        print(path_illumng)
        print(path_illumng+'_boresights_LOLA_ch12345_*_laser2_fov_bs'+str(ampl_in)+'.inc')
        XovOpt.get("vecopts")['ALTIM_BORESIGHT'] = np.loadtxt(glob.glob(path_illumng+'_boresights_LOLA_ch12345_*_laser2_fov_bs'+str(ampl_in)+'.inc')[0])
        #             data det1/0.000839737903394d0, -0.00457961230781711d0, 0.999989000131539d0, !day laser 1
        #      &            0.000856137903d0,       -0.004609612308d0,  0.999989004837226d0,  !day laser 2
        #      &            0.000937737903394d0, -0.00453461230781711d0,0.99998900013153902d0, !day
        #      &            0.00076189248005d0,  -0.00431815664221d0, 0.99998900013153902d0/ !2 night + 5 night - 1 night completes square
        #
        #         data det2/0.000383964851735d0, -0.00436155174587546d0, 0.999990433217333d0,
        #      &            0.000400364852d0,       -0.004391551746d0, 0.999989891939858d0,
        #      &            0.000481964851735d0, -0.00431655174587546d0, 0.999989891939858d0,
        #      &            0.00030611942839d0,   -0.00410009608028d0, 0.999989891939858d0/ ! offset 2 squares
        #
        #        data det3/0.000626524101387d0, -0.00504023006379987d0,0.99998664896001d0,
        #      &            0.000642924101d0,       -0.005070230064d0, 0.999986483343407d0,
        #      &            0.000724524101387d0, -0.00499523006379987d0, 0.99998664896000999d0,
        #      &            0.000553524100735d0, -0.00476423006387546d0,0.999989891939858d0/ !2 nightside
        #
        #         data det4/0.001290665162689d0,  -0.00480736878596984d0, 0.999987131025527d0,
        #      &            0.001307065163d0,       -0.004837368786d0, 0.999986961502879d0,
        #      &            0.001388665162689d0,  -0.00476236878596984d0, 0.999986961502879d0,
        #      &            0.001217665531707d0,  -0.00453621720415891d0, 0.999990352590072d0/!5 nightside
        #
        #        data det5/0.001048106282707d0,  -0.00413353888615891d0, 0.999990497919053d0,
        #      &            0.001064506283d0,      -0.004163538886d0, 0.999990352590072d0,
        #      &            0.001146106282707d0,  -0.00408853888615891d0, 0.999990352590072d0,
        #      &            0.00097026085936d0,  -0.00387208322056d0, 0.999990352590072d0/ ! offset 2 squares
    else:
        XovOpt.get("vecopts")['ALTIM_BORESIGHT'] = [0.0022105, 0.0029215, 0.9999932892]  # out[2]
    ###########################

    # Generate list of epochs
    #########################
    if XovOpt.get("new_illumNG") and not XovOpt.get("instrument") in ["BELA","CALA"]:
        # read all MLA datafiles (*.TAB in data_pth) corresponding to the given time period
        data_pth = XovOpt.get("rawdir")
        allFiles = glob.glob(os.path.join(data_pth, 'MLAS??RDR' + epos_in + '*.TAB'))
        print("path+files")
        print(data_pth, epos_in)
        print(allFiles)

        # Prepare list of tracks
        tracknames = ['gtrack_' + fil.split('.')[0][-10:] for fil in allFiles]
        epo_in = []
        for track_id, infil in zip(tracknames, allFiles):
            track = track_id
            track = gtrack(XovOpt.get("vecopts"))
            track.prepro(infil)
            epo_in.extend(track.ladata_df.ET_TX.values)

        epo_in = np.sort(np.array(epo_in))
    # print(epo_in)
    # print(epo_in.shape)
    # print(np.sort(epo_in)[0],np.sort(epo_in)[-1])
    # print(np.sort(epo_in)[-1])

    elif XovOpt.get("instrument") != "LOLA":  # if BELA/CALA
        # generate list of epoch within selected month and given sampling rate (fixed to 10 Hz)

        dj2000 = dt.datetime(2000, 1, 1, 12, 00, 00)

        sec_j2000_first = (d_first - dj2000).total_seconds()
        sec_j2000_last = (d_last - dj2000).total_seconds()
        # print(sec_j2000_first,sec_j2000_last)
        # get vector of epochs J2000 in year-month, with step equal to the laser sampling rate
        epo_tx = np.arange(sec_j2000_first,sec_j2000_last,.1) # WD: create option?

    # pass to illumNG
    if not XovOpt.get("instrument") in ['BELA','CALA']:
        if XovOpt.get("local"):
            if XovOpt.get("new_illumNG"):
                np.savetxt(XovOpt.get("tmpdir")+"epo_mla_" + epos_in + ".in", epo_tx, fmt="%10.2f")
                print("illumNG call")
                if not os.path.exists("illumNG/"):
                    print('*** create and copy required files to ./illumNG')
                    exit()

                shutil.copy(XovOpt.get("tmpdir")+"epo_mla_" + epos_in + ".in", '../_MLA_Stefano/epo.in')
                illumNG_call = subprocess.call(
                    ['sbatch', 'doslurmEM', 'MLA_raytraces.cfg'],
                    universal_newlines=True, cwd="../_MLA_Stefano/")  # illumNG/")
                for f in glob.glob("../_MLA_Stefano/bore*"):
                    shutil.move(f, XovOpt.get("auxdir") + '/illumNG/grd/' + epos_in + "_" + f.split('/')[1])
            path = XovOpt.get("auxdir") + 'illumng/mlatimes_' + epos_in + '/'  # sph/' # use your path
            print('illumng dir', path)
            illumNGf = glob.glob(path + "/bore*")
        else:
            if XovOpt.get("new_illumNG"):
                np.savetxt("tmp/epo_mla_" + epos_in + ".in", epo_in, fmt="%10.5f")
                print("illumNG call")
                if not os.path.exists("illumNG/"):
                    print('*** create and copy required files to ./illumNG')
                    exit()

                shutil.copy("tmp/epo_mla_" + epos_in + ".in", '../_MLA_Stefano/epo.in')
                illumNG_call = subprocess.call(
                    ['sbatch', 'doslurmEM', 'MLA_raytraces.cfg'],
                    universal_newlines=True, cwd="../_MLA_Stefano/")  # illumNG/")
                for f in glob.glob("../_MLA_Stefano/bore*"):
                    shutil.move(f, XovOpt.get("auxdir") + '/illumNG/grd/' + epos_in + "_" + f.split('/')[1])
            if XovOpt.get("instrument") == 'LOLA':
                path = path_illumng
            else:
                path = XovOpt.get("auxdir") + 'illumng/mlatimes_' + epos_in + '/'  # sph/' # use your path
            print('illumng dir', path)
            illumNGf = glob.glob(path + "bore*")

        # else:
        # launch illumNG directly
        df = prepro_ilmNG(illumNGf)
        print('illumNGf', illumNGf)

    else: # if BELA/CALA
        # WD: name to be changed ...
        illumpklf = XovOpt.get("tmpdir")+'bela_illumNG_'+epos_in+'.pkl'

        if XovOpt.get("new_illumNG"):
            start_BELA_prepro = time.time()
            df = prepro_BELA_sim(epo_in=epo_tx)
            end_BELA_prepro = time.time()
            print("BELA prepro (simil illumNG) completed after ", end_BELA_prepro - start_BELA_prepro, "sec")
            df.to_pickle(illumpklf)
        else:
            df = pd.read_pickle(illumpklf)
            print("simil-illumNG prediction read from ", illumpklf)
        # print(df)

    # TODO replace with "small_scale_topo" option
    if XovOpt.get("apply_topo") and XovOpt.get("instrument") != "LOLA":
        # read and interpolate DEM
        # # open netCDF file
        # nc_file = "/home/sberton2/Works/NASA/Mercury_tides/MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_4ppd_HgM008frame.GRD"
        # sim_gtrack.dem_xr = xr.open_dataset(nc_file)

        # prepare surface texture "stamp" and assign the interpolated function as class attribute
	# persistence = 0.65 to fit power law of Steinbrugge 2018 over scales 50m (spot-size) to 200m (spots distance)
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
        sim_gtrack.apply_texture = interp_spline

    # Process tracks
    # tracks = []
    # for i in list(df.groupby('orbID').groups.keys()):
    #     if debug:
    #         print("Processing",i)
    #     tracks.append(sim_gtrack(vecopts, i))
    #
    # print(tracks)
    # print([tr.name for tr in tracks])

    if XovOpt.get("local"):
        outdir_ = XovOpt.get("rawdir") + dirnam_in
    else:
        outdir_ = dirnam_in

    print("outdir = ",outdir_)

    if not os.path.exists(outdir_):
        os.makedirs(outdir_, exist_ok=True)

    # loop over all gtracks
    # initializze objects
    print('orbs = ', list(df.groupby('orbID').groups.keys()))
    args = ((sim_gtrack(XovOpt.to_dict(), i), df, i, outdir_) for i in list(df.groupby('orbID').groups.keys()))

    if XovOpt.get("parallel") and False:  # incompatible with grdtrack call ...
        # print((mp.cpu_count() - 1))
        pool = mp.Pool(processes=ncores)  # mp.cpu_count())
        _ = pool.map(sim_track, args)  # parallel
        pool.close()
        pool.join()
    else:
        _ = [sim_track(arg) for arg in args]  # seq


##############################################
if __name__ == '__main__':

    import sys

    ##############################################
    # launch program and clock
    # -----------------------------
    start = time.time()

    print("Running PyAltsim")

    if len(sys.argv) == 1:

        args = sys.argv[0]

        main(args)
    else:
        print("PyAltSim running with standard args...")
        main()

    # stop clock and print runtime
    # -----------------------------
    end = time.time()
    print('----- Runtime = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
