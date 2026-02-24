#!/usr/bin/env python3
# ----------------------------------
# PyXover
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-Oct-2018
#
from fileinput import filename
import warnings
import logging

from pyaltsim.prepro import load_illumng_predictions, build_bela_sim_inputs
from xovutil.dem_util import get_topoelev, get_toposlope
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
def build_rdr_df(ladata_df):
   """
   Convert geolocated altimetry data into the RDR-like dataframe.
   """
   df_ = ladata_df.copy()

   # only select nadir data
   # df_ = df_[df_.loc[:,'offnadir']<5]

   # mlardr_cols = ['geoc_long', 'geoc_lat', 'altitude', 'EphemerisTime', 'MET', 'frm',
   #                'chn', 'Pulswd', 'thrsh', 'gain', '1way_range', 'Emiss', 'TXmJ',
   #                'UTC', 'TOF_ns_ET', 'Sat_long', 'Sat_lat', 'Sat_alt', 'Offnad', 'Phase',
   #                'Sol_inc', 'SCRNGE', 'seqid']
   mlardr_cols = ['geoc_long', 'geoc_lat', 'altitude', 'EphemerisTime',
                  'chn', 'UTC', 'TOF_ns_ET', 'seqid']

   # assign "bad chn" to non converged observations
   df_['chn'] = 0
   df_.loc[~df_['converged'], 'chn'] = 10

   # update other columns for compatibility with real data format
   df_['TOF_ns_ET'] = np.round(df_['TOF'].values * 1.e9, 10)
   df_['UTC'] = pd.to_datetime(df_['ET_TX'], unit='s',
                               origin=pd.Timestamp('2000-01-01T12:00:00'))

   df_ = df_.rename(columns={'ET_TX': 'EphemerisTime',
                             'LON': 'geoc_long', 'LAT': 'geoc_lat', 'R': 'altitude'})
   df_ = df_.reset_index(drop=True)
   # match legacy "RDR" column order to real data tables
   rdr_df = df_[['EphemerisTime', 'geoc_long', 'geoc_lat', 'altitude',
                'UTC', 'TOF_ns_ET', 'chn', 'seqid']].reindex(columns=mlardr_cols)
   # legacy path (kept for reference): concat/append to a pre-made empty df

   return rdr_df


class sim_gtrack(gtrack):
   def __init__(self, opts, orbID):
      gtrack.__init__(self, opts)
      self.name = str(orbID)
      self.outdir = None
      self.slewdir = None
      self.rdr_df = None

   def setup(self, df):
      df_ = df.copy()

      # get time of flight in ns from probe one-way range in m
      df_['TOF'] = df_['altitude'] * 2. / clight
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
      print("lt_topo_corr(df=df_) done")
      if self.ladata_df.empty:
         print("### PyAltsim.setup: ladata_df is empty")
         return

      # add range noise
      if XovOpt.get("range_noise"):
         if XovOpt.get("instrument") == 'BELA':
            self.add_range_noise_model(df_)
         else:
            self.add_range_noise(df_, XovOpt.get("range_noise_mean_std")[0],
                                 XovOpt.get("range_noise_mean_std")[1])

      self.rdr_df = build_rdr_df(self.ladata_df)

   # @staticmethod
   def add_range_noise(self, df_, mean=0., std=0.2):
      """
      Add range noise (normal distribution) to simulated time of flight (seconds)
      :param df_: input altimetry dataframe
      :param mean: mean value for normal distribution (meters)
      :param std: standard deviation for normal distribution (meters)
      """
      np.random.seed(int(self.name))
      tof_noise = (std * np.random.randn(len(df_)) + mean) / clight
      df_.loc[:, 'TOF'] += tof_noise  # WD: not affected by lt_topo_corr?
      self.ladata_df.loc[:, 'TOF'] += tof_noise
      if XovOpt.get("debug"):
         plt.plot(df_.loc[:, 'ET_TX'], df_.TOF, 'bo', df_.loc[:, 'ET_TX'], df_.TOF - tof_noise, 'k')
         plt.savefig('tmp/noise.png')

   def add_range_noise_model(self, df_):
      """
      Add range noise based on a probably density function of
      the altitude and the slope to simulated time of flight (seconds)
      :param df_: input altimetry dataframe
      :param mean: mean value for normal distribution (meters)
      :param std: standard deviation for normal distribution (meters)
      """

      np.random.seed(int(self.name))
      mod_data = np.load(XovOpt.get("auxdir") + "probability_density.npy")
      mod_alt  = np.linspace(200,1500,14) # [km]
      mod_rerr = np.linspace(-10,10,100)  # [m]
      mod_slp  = np.linspace(0,49,50)     # [degree]
      
      slopes = get_toposlope(self)
   
      index_a = [(np.abs(mod_alt*1e3 - alt)).argmin() for alt in df_['altitude']]
      index_s = [(np.abs(mod_slp - slp)).argmin() for slp in slopes]
      
      range_noise = [np.random.choice(mod_rerr, p=mod_data[i_a,i_s,:]) for i_a, i_s in zip(index_a,index_s)]
      
      tof_noise = np.array(range_noise) / clight
      df_.loc[:, 'TOF'] += tof_noise
      self.ladata_df.loc[:, 'TOF'] += tof_noise
   
   def _compute_offnadir(self, rngvec, scxyz_tx_pbf):
      # compute correction for off-nadir observation (with check to avoid numerical issues on arccos)
      rngvec_normed = rngvec / np.linalg.norm(rngvec, axis=1)[:, np.newaxis]
      scxyz_tx_pbf_normed = np.array(scxyz_tx_pbf) / np.linalg.norm(scxyz_tx_pbf, axis=1)[:, np.newaxis]
      cosang = np.einsum('ij,ij->i', rngvec_normed, -scxyz_tx_pbf_normed)
      if np.max(np.abs(cosang)) <= 1:
         return np.arccos(cosang)
      return 0.

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

      # a priori values for internal FULL df
      df.loc[:, 'converged'] = False
      df.loc[:, 'offnadir'] = 0

      for it in range(itmax):

         # read just lat, lon, elev from geoloc (reads ET and TOF and updates LON, LAT, R in df)
         self.geoloc()
         
         # Remove nan
         self.ladata_df = self.ladata_df.dropna(subset=['LON'])
         if self.ladata_df.empty:
            print("### lt_topo_corr: ladata_df is empty")
            return
         lontmp, lattmp, rtmp = np.transpose(self.ladata_df[['LON', 'LAT', 'R']].values)
         r_bc = rtmp + XovOpt.get("vecopts")['PLANETRADIUS'] * 1.e3

         if np.isnan(lattmp).any():
            print("lattmp is nan")
         if np.isnan(lontmp).any():
            print("lontmp is nan")
         # use lon and lat to get "real" elevation from map
         radius = get_topoelev(self, lattmp, lontmp)
         if np.isnan(radius).any():
            print("radius is nan")

         # use "real" elevation to get bounce point coordinates
         bcxyz_pbf = astr.sph2cart(radius, lattmp, lontmp)
         bcxyz_pbf = np.transpose(np.vstack(bcxyz_pbf))

         # get S/C body fixed position (useful to update ranges, has to be computed on reduced df)
         scxyz_tx_pbf = self.get_sc_pos_bf(self.ladata_df)
         # compute range btw probe@TX and bounce point@BC (no BC epoch needed, all coord planet fixed)
         rngvec = (bcxyz_pbf - scxyz_tx_pbf)
         offndr = self._compute_offnadir(rngvec, scxyz_tx_pbf)
         # offndr = np.arccos(np.einsum('ij,ij->i', rngvec, -scxyz_tx_pbf) /
         #                    np.linalg.norm(rngvec, axis=1) /
         #                    np.linalg.norm(scxyz_tx_pbf, axis=1))

         # compute residual between "real" elevation and geoloc (based on a priori TOF)
         dr = (r_bc - radius) * np.cos(offndr)
         
         # tof and rng from previous iter
         old_tof = self.ladata_df.loc[:, 'TOF'].values
         rng_apr = old_tof * clight / 2.

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
            # df = self.ladata_df.copy()

         percent_left = 100. - (len(df) - np.count_nonzero(abs(dr) > tol)) / len(df) * 100.

         if XovOpt.get("debug"):
            print("it = " + str(it))
            print("max resid:", max(abs(dr)), "# > tol:", np.count_nonzero(abs(dr) > tol), percent_left, ' %')

         if (max(abs(dr)) < tol):
            # pass all epochs to next step
            self.ladata_df = df.copy()
            break
         elif it > 10 and percent_left < 5:
            print('### altsim: Most data point converged!')
            print("it = " + str(it))
            print("max resid:", max(abs(dr)), "# > tol:", np.count_nonzero(abs(dr) > tol), percent_left, ' %')
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
            self.ladata_df = df.loc[~df['converged']].copy()
         # self.ladata_df = df.copy()

   def get_sc_pos_bf(self, df):
      et_tx = df.loc[:, 'ET_TX'].values
      sc_pos, sc_vel = get_sc_ssb(et_tx, self.SpObj, self.pertPar, self.vecopts)
      scpos_tx_p, _ = get_sc_pla(et_tx, sc_pos, sc_vel, self.SpObj, self.vecopts)
      if XovOpt.get('body') in ["MERCURY", "CALLISTO"]:
         rotpar, upd_rotpar = orient_setup(self.pertPar['dRA'], self.pertPar['dDEC'], self.pertPar['dPM'],
                                           self.pertPar['dL'], self.pertPar['dLIB'])
         tsipm = icrf2pbf(et_tx, upd_rotpar)
      else:
         pxform_array = np.frompyfunc(spice.pxform, 3, 1)
         tsipm = pxform_array(XovOpt.get("vecopts")['INERTIALFRAME'], XovOpt.get("vecopts")['PLANETFRAME'], et_tx)

      scxyz_tx_pbf = np.vstack([np.dot(tsipm[i], scpos_tx_p[i]) for i in range(0, np.size(scpos_tx_p, 0))])

      return scxyz_tx_pbf


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
      # try:
      #    track.setup(df[df['orbID'] == i])
      # except:
      #    logging.info('Error when simulating observations to', filename)
      #    print('Error when simulating observations to', filename)
      #    return
      track.rdr_df['altitude'] *=1e-3 # write altitude in km
      track.rdr_df.to_csv(filename, index=False, sep=',', na_rep='NaN')
      logging.info('Simulated observations written to', filename)
      print('Simulated observations written to', filename)
   else:
      logging.info('Simulated observations ', filename + ' already exists. Skip.')
      print('Simulated observations ', filename + ' already exists. Skip.')


def main(args):
   import datetime as dt
   if len(args) == 0:
      print("Usage: PyAltSim.py <ampl_in> <res_in> <dirnam_in> <epos_or_start> <opts> [<d_last> <opts>]")
      return

   ampl_in   = args[0]
   res_in    = args[1]
   dirnam_in = args[2]

   if len(args) < 6:
      epos_in = args[3]  # Month to simulate (format: YYMM)
      opts = args[4]  # Options (dictionnary)
      if XovOpt.get("instrument") != "LOLA":  # if BELA/CALA
         # generate list of epoch within selected month and given sampling rate (fixed to 10 Hz)
         from calendar import monthrange

         days_in_month = monthrange(int('20' + epos_in[:2]), int(epos_in[2:]))

         # TODO avoiding issues with 30-Apr 23:59:59 ... extend spk
         d_first = dt.datetime(int('20' + epos_in[:2]), int(epos_in[2:]), int('01'), 1, 00,00)

         # if test, avoid computing tons of files
         if XovOpt.get("unittest"):
            d_last = dt.datetime(int('20' + epos_in[:2]), int(epos_in[2:]), int('02'), 5, 00, 00)  # for testing
         else:
            d_last = dt.datetime(int('20' + epos_in[:2]), int(epos_in[2:]), int(days_in_month[-1]), 23, 59, 59)
   else:
      d_first = args[3]
      d_last = args[4]
      opts = args[5]  # Options (dictionnary)
      epos_in = d_first.strftime('%y%m%d')

     
   # update options (needed when sending to slurm)
   XovOpt.clone(opts)
   XovOpt.check_consistency()

   if XovOpt.get("small_scale_topo"):
      print(f"Small-scale topgraphy: amplitude:{ampl_in}, resolution{res_in}")
   print(f"Alimetry files created in {dirnam_in}")
   print("XovOpt")
   print("------")
   XovOpt.display()
   print("\n")
   print(f'Simulation of {XovOpt.get("instrument")} data from {d_first} to {d_last}')

   # locate data
   outdir_ = f'{XovOpt.get("rawdir")}' + dirnam_in

   # load kernels
   if (not XovOpt.get("instrument") == "LOLA"): # and (XovOpt.get("SpInterp") in [0, 2]):
      spice.furnsh(f'{XovOpt.get("auxdir")}{XovOpt.get("spice_meta")}')
      if XovOpt.get("spice_spk"):
         spice.furnsh(XovOpt.get("spice_spk"))

   if not os.path.exists(XovOpt.get('tmpdir')):
      os.makedirs(XovOpt.get('tmpdir'))

   if XovOpt.get("parallel"):
      # set ncores
      ncores = mp.cpu_count() - 1  # 8
      print('Process launched on ' + str(ncores) + ' CPUs')

   # out = spice.getfov(vecopts['INSTID'][0], 1)
   # updated w.r.t. SPICE from Mike's scicdr2mat.m
   if XovOpt.get("instrument") == 'LOLA':
      path_illumng = f'{XovOpt.get("auxdir")}{epos_in}/slewcheck_{ampl_in}/'
      print("XovOpt.get(vecopts)['ALTIM_BORESIGHT'] read from file")
      XovOpt.get("vecopts")['ALTIM_BORESIGHT'] = np.loadtxt(
         glob.glob(path_illumng + '_boresights_LOLA_ch*_*_laser2_fov_bs' + str(ampl_in) + '.inc')[0])
   ###########################

   # Preprocessing contract:
   #  - returns a dataframe of simulated/observed shot inputs
   #  - required columns: epo_tx (seconds since J2000), altitude, orbID
   #  - additional geometry columns (lat/lon or x/y/z) are preserved if present
   # Generate list of epochs
   #########################
   if XovOpt.get("new_illumNG") and not XovOpt.get("instrument") in ["BELA", "CALA", "MLA"]:
      # read all MLA datafiles (*.TAB in data_pth) corresponding to the given time period
      data_pth = XovOpt.get("rawdir")
      allFiles = glob.glob(os.path.join(data_pth, 'MLAS??RDR' + epos_in + '*.TAB'))

      # Prepare list of tracks
      tracknames = ['gtrack_' + fil.split('.')[0][-10:] for fil in allFiles]
      epo_in = []
      for track_id, infil in zip(tracknames, allFiles):
         track = track_id
         track = gtrack(XovOpt.get("vecopts"))
         track.prepro(infil)
         epo_in.extend(track.ladata_df.ET_TX.values)

      epo_in = np.sort(np.array(epo_in))

   elif XovOpt.get("instrument") != "LOLA":
      # generate list of epoch within selected month and given sampling rate

      if XovOpt.get("instrument") == "MLA": # given in UTC
         sec_j2000_first = spice.utc2et(d_first.strftime('%Y-%m-%dT%H:%M:%S'))
         sec_j2000_last  = spice.utc2et(d_last.strftime('%Y-%m-%dT%H:%M:%S'))
      else: # already given in ET?
         dj2000 = dt.datetime(2000, 1, 1, 12, 00, 00)
         sec_j2000_first = (d_first - dj2000).total_seconds()
         sec_j2000_last  = (d_last  - dj2000).total_seconds()

      # get vector of epochs J2000 in year-month, with step equal to the laser sampling rate
      epo_tx = np.arange(sec_j2000_first, sec_j2000_last, 1/XovOpt.get("sampling_rate"))

   # pass to illumNG
   if not XovOpt.get("instrument") in ['BELA', 'CALA', 'MLA']:
      if XovOpt.get("local"):
         if XovOpt.get("new_illumNG"):
            np.savetxt(XovOpt.get("tmpdir") + "epo_mla_" + epos_in + ".in", epo_tx, fmt="%10.2f")
            print("illumNG call")
            if not os.path.exists("illumNG/"):
               print('*** create and copy required files to ./illumNG')
               exit()

            shutil.copy(XovOpt.get("tmpdir") + "epo_mla_" + epos_in + ".in", '../_MLA_Stefano/epo.in')
            illumNG_call = subprocess.call(['sbatch', 'doslurmEM', 'MLA_raytraces.cfg'],
                                           universal_newlines=True, cwd="../_MLA_Stefano/")  # illumNG/")
            for f in glob.glob("../_MLA_Stefano/bore*"):
               shutil.move(f, XovOpt.get("auxdir") + '/illumNG/grd/' + epos_in + "_" + f.split('/')[1])
         path = XovOpt.get("auxdir") + 'illumng/mlatimes_' + epos_in + '/'  # sph/' # use your path
         print('illumng dir', path)
         illumNGf = glob.glob(path + "/bore*")
      else:
         if XovOpt.get("new_illumNG"):
            np.savetxt("tmp/epo_mla_" + epos_in + ".in", epo_in, fmt="%10.5f")
            if not os.path.exists("illumNG/"):
               print('*** create and copy required files to ./illumNG')
               exit()

            shutil.copy("tmp/epo_mla_" + epos_in + ".in", '../_MLA_Stefano/epo.in')
            illumNG_call = subprocess.call(['sbatch', 'doslurmEM', 'MLA_raytraces.cfg'],
                                           universal_newlines=True, cwd="../_MLA_Stefano/")  # illumNG/")
            for f in glob.glob("../_MLA_Stefano/bore*"):
               shutil.move(f, XovOpt.get("auxdir") + '/illumNG/grd/' + epos_in + "_" + f.split('/')[1])
         if XovOpt.get("instrument") == 'LOLA':
            path = path_illumng
         else:
            path = XovOpt.get("auxdir") + 'illumng/mlatimes_' + epos_in + '/'  # sph/' # use your path

         illumNGf = glob.glob(path + "bore*")

      # else:
      # launch illumNG directly
      df = load_illumng_predictions(illumNGf)

   else:  # if BELA/CALA
      # WD: name to be changed ...
      illumpklf = XovOpt.get("tmpdir") + 'bela_illumNG_' + epos_in + '.pkl'

      if XovOpt.get("new_illumNG"):
         start_BELA_prepro = time.time()
         df = build_bela_sim_inputs(epo_in=epo_tx)
         end_BELA_prepro = time.time()
         print("BELA prepro (simil illumNG) completed after ", end_BELA_prepro - start_BELA_prepro, "sec")
         df.to_pickle(illumpklf)
      else:
         df = pd.read_pickle(illumpklf)
         print("simil-illumNG prediction read from ", illumpklf)

   if df.empty:
      print('No ranges to process after preprocessing')
      return

   if XovOpt.get("small_scale_topo") and XovOpt.get("instrument") != "LOLA":
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

   if not os.path.exists(outdir_):
      os.makedirs(outdir_, exist_ok=True)

   # loop over all gtracks
   # initialize objects
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
        print("PyAltSim running with no args...")
    else:
        print("PyAltSim running with standard args...")
    main(sys.argv[1:])

    # stop clock and print runtime
    # -----------------------------
    end = time.time()
    print('----- Runtime = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
