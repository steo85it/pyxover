#!/usr/bin/env python3
# ----------------------------------
# PyXover
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-Oct-2018
#
import re
import os
import glob
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd
import multiprocessing as mp
import spiceypy as spice
import numpy as np
import time
import datetime as dt
from tqdm import tqdm

from accumxov.Amat import Amat
from accumxov import AccumXov as xovacc, accum_utils
from pygeoloc.ground_track import gtrack
from config import XovOpt

########################################

def launch_gtrack(args):
   track, outdir_in = args
   track_id = 'gtrack_' + track.name

   if XovOpt.get("new_gtrack") > 0:
      gtrack_out = XovOpt.get("outdir") + outdir_in + '/' + track_id
      if not (os.path.isfile(gtrack_out) or os.path.isfile(gtrack_out)) or XovOpt.get("new_gtrack") == 2:

         if not os.path.exists(XovOpt.get("outdir") + outdir_in):
            os.makedirs(XovOpt.get("outdir") + outdir_in, exist_ok=True)

         track.setup()
         

         if XovOpt.get("debug"):
            pd.set_option('display.max_columns', 500)
            print("track#:", track.name)
            print("max diff R", abs(track.ladata_df.loc[:, 'R'] - (
               track.ladata_df.loc[:, 'altitude'] - XovOpt.get("vecopts")['PLANETRADIUS']) * 1.e3).max())
            # print("R (check if radius included + units)",track.ladata_df.loc[:,'R'].max(),track.ladata_df.loc[:,'altitude'].max())
            print("max diff LON", XovOpt.get("vecopts")['PLANETRADIUS'] * 1.e3 * np.sin(
               np.deg2rad(abs(track.ladata_df.loc[:, 'LON'] - track.ladata_df.loc[:, 'geoc_long']).max())))
            print("max diff LAT", XovOpt.get("vecopts")['PLANETRADIUS'] * 1.e3 * np.sin(
               np.deg2rad(abs(track.ladata_df.loc[:, 'LAT'] - track.ladata_df.loc[:, 'geoc_lat']).max())))
            print("max elev sim", abs(track.ladata_df.loc[:, 'altitude']).max())
         # pd.set_option('display.max_columns', None)

         if len(track.ladata_df) > 0:
            track.save(gtrack_out)
            if not XovOpt.get("local") or XovOpt.get("debug"):
               print('Orbit ' + track_id.split('_')[1] + ' processed and written to ' + gtrack_out + '!')
         else:
            print(f"Orbit {track.name} contains no valid data. No gtrack created.")
            # except:
            #    print('failed to process ' + track_id)
      else:
         # track = track.load('out/'+track_id)
         if not XovOpt.get("local") or XovOpt.get("debug"):
            print('Gtrack file ' + gtrack_out + ' already existed!')


def main(args):

   # read input args
   epo_in = args[0]     # WD: (list of?) epoch from input raw alti file
   indir_in = args[1]   # Location of input raw alti file
   outdir_in = args[2]  # Location of output pickle gtrack
   d_tracks = args[3]   # list of date?
   iter_in = args[4]    # iteration number (to load previous teration info)
   # if len(args) > 4:  # passing a fct to slurm doesn't pass these updated Opt
   opts = args[5]

   # update options (needed when sending to slurm)
   XovOpt.clone(opts)
   XovOpt.check_consistency()
   
   print(f"epo_in: {epo_in}")
   print(f"Alimetry raw files located in {indir_in}")
   print(f"Output gtrack files located in {outdir_in}")
   print(f"List of dates: {d_tracks}")
   print(f"Iteration n°{iter_in}")
   print("XovOpt")
   print("------")
   XovOpt.display()
   print("\n")

   # locate data
   data_pth = f'{XovOpt.get("rawdir")}'
   data_pth += indir_in

   spice.furnsh(f'{XovOpt.get("auxdir")}{XovOpt.get("spice_meta")}')
   # load additional kernels
   if XovOpt.get("spice_spk"):
      print("Additional spice kernels loaded:", XovOpt.get("spice_spk"))
      spice.furnsh(XovOpt.get("spice_spk"))# or, add custom kernels

   # set ncores
   ncores = mp.cpu_count() - 1  # 8

   if XovOpt.get("parallel"):
      print('Process launched on ' + str(ncores) + ' CPUs')

   ##############################################
   # updated w.r.t. SPICE from Mike's scicdr2mat.m
   if XovOpt.get("instrument") == 'LOLA':
      print("XovOpt.get(vecopts)['ALTIM_BORESIGHT'] read from file")
      XovOpt.get("vecopts")['ALTIM_BORESIGHT'] = np.loadtxt(
         glob.glob(f'{XovOpt.get("auxdir")}{epo_in}/slewcheck_0/' + '_boresights_LOLA_ch12345_*_laser2_fov_bs0.inc')[0])

   ###########################

   # -------------------------------
   # File reading and ground-tracks computation
   # -------------------------------

   startInit = time.time()

   # read all MLA datafiles (*.TAB in data_pth) corresponding to the given years
   # for orbitA and orbitB.
   allFiles = glob.glob(os.path.join(data_pth, f'{XovOpt.get("instrument")}*RDR*' + epo_in + '*.*'))
   allFiles = allFiles+glob.glob(os.path.join(data_pth, str.lower(f'{XovOpt.get("instrument")}*rdr*' + epo_in + '*.*')))

   if len(allFiles) == 0:
      print("# No files found in", os.path.join(data_pth, f'{XovOpt.get("instrument")}*RDR*' + epo_in + '*.*'))

   endInit = time.time()
   # Useful?
   print('----- Runtime Init= ' + str(endInit - startInit) + ' sec -----' +
         str((endInit - startInit) / 60.) + ' min -----')

   startPrepro = time.time()

   # -------------------------------------
   # Load all epochs of all raw alti files
   # Get a list of arc boundaries
   # -------------------------------------

   # Prepare list of tracks to geolocalise
   dstr_files = [fil.split('.')[0][-10:] for fil in allFiles[:]]
   d_files = [dt.datetime.strptime(date, '%y%m%d%H%M') for date in dstr_files]
   d_files.sort()

   dj2000 = dt.datetime(2000, 1, 1, 12, 00, 00)
   if len(d_tracks) > 0:
      d_track_start = d_tracks[:-1]
      d_track_end = d_tracks[1:]
   else:
      d_track_start = d_files
      d_track_end = d_files[1:] + [d_files[-1]]

   if XovOpt.get("new_gtrack") > 0:

      # Import solution at previous iteration
      if int(iter_in) > 0:
         tmp = Amat(XovOpt.get("vecopts"))
         # previous_dir = ('_').join(((XovOpt.get("outdir") + ('/').join(outdir_in.split('/')[:-2]))).split('_')[:-1]) \
         #   + '_' + str(iter_in - 1) + '/' + outdir_in.split('/')[-2] + '/'
         # tmp = tmp.load(previous_dir + 'Abmat_' + ('_').join(outdir_in.split('/')[:-1]))
         id = outdir_in.split('/')[0].split('_')[0]
         previous_dir = XovOpt.get("outdir") + id + '_' + str(iter_in - 1) + '/'
         if XovOpt.get("import_abmat") == "":
            Abmat_infile = 'Abmat_' + id +  '_' + str(iter_in - 1)  + '_' + str(iter_in)
         else:
            Abmat_infile = XovOpt.get("import_abmat")
            print(previous_dir)
            print(Abmat_infile)
         tmp = tmp.load(os.path.join(previous_dir + Abmat_infile))
         import_prev_sol = hasattr(tmp, 'sol4_pars')
         if import_prev_sol:
            orb_sol, glo_sol, sol_dict = accum_utils.analyze_sol(tmp, tmp.xov.xovers)
      tracks = []
      for d_start, d_end in tqdm(zip(d_track_start, d_track_end), total=len(d_track_start)):
         track_name = d_start.strftime('%y%m%d%H%M')
         track_id = f'gtrack_{track_name}'
         track = track_id  # WD: what is it for?

         # if 0, don't cut the gtrack
         # t_start = 0
         # t_end = 0
         # Look for which file to use in allFiles

         if (track_name in dstr_files):
            index_file = dstr_files.index(track_name)
            infil = allFiles[index_file]
            t_start = 0
         else:
            d_track = dt.datetime.strptime(track_name, '%y%m%d%H%M')
            duration = [(date - d_track).total_seconds() for date in d_files]
            print(f"Duration: {duration}")
            index = [i for i, x in enumerate(duration) if x < 0]
            print("index")
            print(index)
            d_file = d_files[index[-1]]  # find the closest before
            index_file = dstr_files.index(d_file.strftime('%y%m%d%H%M'))
            infil = allFiles[index_file]
            print("Arc discontinuity:")
            print(f"Use file {infil} for track {track_name}")
            t_start = (d_start - dj2000).total_seconds()

         if (d_end.strftime('%y%m%d%H%M') in dstr_files):
            t_end = 0
         else:
            # -1 to avoid overlap with next gtrack
            t_end = (d_end - dj2000).total_seconds() - 1
            print("Arc discontinuity:")
            print(f"Track {track_name} ends at {t_end}")

         print(f"t_start: {t_start}")
         print(f"t_end: {t_end}")

         track = gtrack(XovOpt.to_dict())
         # try:
         # Read and fill
         # decide hemisphere in prepro (read_fill)
         track.prepro(infil, t_start=t_start, t_end=t_end)
         # except:
         #    print('Issue in preprocessing for '+track_id)

         if int(iter_in) > 0 and import_prev_sol:
            try:
               track.pert_cloop_0 = tmp.pert_cloop_0.loc[str(track.name)].to_dict()
            except:
               if XovOpt.get("debug"):
                  print("No pert_cloop_0 for ", track.name)
               pass

            regex = re.compile(track.name + "_dR/d.*")
            soltmp = [('sol_' + x.split('_')[1], v) for x, v in tmp.sol_dict['sol'].items() if regex.match(x)]

            if len(soltmp) > 0:
               stdtmp = [('std_' + x.split('_')[1], v) for x, v in tmp.sol_dict['std'].items() if regex.match(x)]
               soltmp = pd.DataFrame(np.vstack([('orb', str(track.name)), soltmp, stdtmp])).set_index(0).T

               if XovOpt.get("debug"):
                  print("orbsol prev iter")
                  print(orb_sol.reset_index().orb.values)
                  print(orb_sol.columns)
                  print(str(track.name))
                  print(orb_sol.loc[orb_sol.reset_index().orb.values == str(track.name)])

               track.sol_prev_iter = {'orb': soltmp,'glo': glo_sol}
            else:
               track.sol_prev_iter = {'orb': orb_sol,'glo': glo_sol}
         # if first iter, check if track has been pre-processed by fit2dem and import corrections
         elif int(iter_in) == 0:
            try:
               gtrack_fit2dem = XovOpt.get("outdir") + outdir_in + '/' + track_id + '.pkl'
               fit2dem_res = gtrack(XovOpt.to_dict)
               fit2dem_res = fit2dem_res.load(gtrack_fit2dem).sol_prev_iter
               # if debug:
               print("Solution of fit2dem for file", track_id + ".pkl imported: \n", fit2dem_res['orb'])
               track.sol_prev_iter = fit2dem_res
            except:
               True

         tracks.append(track)

      if XovOpt.get("SpInterp") == 3:
         print('Orbit and attitude data loaded for years 20' +
               str(misycmb[par][0]) + ' and 20' + str(misycmb[par][1]))
         endPrepro = time.time()
         print('----- Runtime Init= ' + str(endPrepro - startPrepro) + ' sec -----' +
               str((endPrepro - startPrepro) / 60.) + ' min -----')
         exit()

   endPrepro = time.time()
   print('----- Runtime Prepro= ' + str(endPrepro - startPrepro) + ' sec -----' +
         str((endPrepro - startPrepro) / 60.) + ' min -----')

   startGeoloc = time.time()

   # WD: if not XovOpt.get("new_gtrack") > 0, tracks is not defined ...
   # WD: fil is not used in launch_gtrack
   args = ((tr, outdir_in) for tr in tracks)

   # loop over all gtracks
   if XovOpt.get("parallel"):
      # print((mp.cpu_count() - 1))
      if XovOpt.get("local"):
         # forks everything, if much memory needed, use the remote option with get_context
         from tqdm.contrib.concurrent import process_map  # or thread_map
         _ = process_map(launch_gtrack, args, max_workers=ncores, total=len(tracks))
      else:
         pool = mp.Pool(processes=ncores)  # mp.cpu_count())
         _ = pool.map(launch_gtrack, args)  # parallel
         pool.close()
         pool.join()
   else:
      for arg in tqdm(args, total=len(tracks)):
         launch_gtrack(arg)  # seq

   endGeoloc = time.time()
   print('----- Runtime Geoloc= ' + str(endGeoloc - startGeoloc) + ' sec -----' +
         str((endGeoloc - startGeoloc) / 60.) + ' min -----')


##############################################
# locate data
if __name__ == '__main__':
   import sys

   ##############################################
   # launch program and clock
   # -----------------------------
   start = time.time()

   args = sys.argv[1:]
   print(args)
   main(args)

   # stop clock and print runtime
   # -----------------------------
   end = time.time()
   print('----- Runtime = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
