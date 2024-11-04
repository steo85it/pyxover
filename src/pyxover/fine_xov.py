import multiprocessing as mp
import time
import numpy as np
import pandas as pd
from memory_profiler import profile

from config import XovOpt
from pyxover.get_xov_latlon import get_xov_latlon
from pyxover.xov_setup import xov

# @profile
def fine_xov_intersection(mla_proj_df, msrm_smpl):
   # Fine search of xovers, based on a rough search, and projected mla_df
   # Only the indices, and the intersection stereographically projected
   # are returned in a dataframe

   start_finexov = time.time()
   # initialize xov object
   xovs_list = mla_proj_df.xovid.unique()
   xovtmp = xov(XovOpt.get("vecopts"))
   # pass measurements sampling to fine_xov computation through xovtmp
   xovtmp.msrm_sampl = msrm_smpl

   # WD: probably less columns to keep
   cols_to_keep = mla_proj_df.loc[:,
                                  mla_proj_df.columns.str.startswith('ET_BC') + mla_proj_df.columns.str.startswith('X_stgprj') +
                                  mla_proj_df.columns.str.startswith('Y_stgprj') + mla_proj_df.columns.str.startswith(
                                     'dR/')].columns
   proj_df_tmp = mla_proj_df[
      ['orbID', 'seqid_mla', 'ET_TX', 'LON', 'LAT', 'R', 'dt', 'offnadir', 'xovid', 'LON_proj', 'LAT_proj'] + list(
         cols_to_keep)]
   print("total memory proj_df_tmp:", proj_df_tmp.memory_usage(deep=True).sum() * 1.e-6)

   fine_xov = []
   for xovi in xovs_list:
      # create a df storing xovi and the rest of the values
      fine_xovi = fine_intersection_proc(xovi, proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi], xovtmp)
      # fine_xovi is None if pb with xover
      fine_xov.append(np.insert(fine_xovi, 0 ,xovi))
   fine_xov_df = pd.DataFrame(fine_xov,columns=['xovi', 'x', 'y', 'mla_idA', 'mla_idB', 'ldA', 'ldB'])

   end_finexov = time.time()
   print("Fine intersection computation finished after", int(end_finexov - start_finexov), "sec or ",
         round((end_finexov - start_finexov) / 60., 2), " min")
   return fine_xov_df

def fine_intersection_proc(xovi, df, all_xov):

   df = df.reset_index(drop=True).reset_index()
   df.rename(columns={'index': 'genID', 'seqid_mla': 'seqid'}, inplace=True)

   # WD: I don't think this should be done here, but rahter at interpolation
   #     if the xover is too close to the gtrack limit
   # check that we got the same number of rows for both tracks
   # TODO WHY is this happening with pandas 2.x.x
   points_per_track = df.groupby('orbID')['genID'].count().values
   try:
      assert len(points_per_track) == 2
      assert  np.diff(points_per_track) == 0
   except:
      # if XovOpt.get("debug"):
      print(f"# Removing weird xover {xovi}, difference between trackA/B = {np.diff(points_per_track)}. Check!")
      print(points_per_track)
      return

   if len(df) > 0:  # and {'LON_proj','LAT_proj'}.issubset(df.columns):
      # all_xov.proj_center = {'lon': df['LON_proj'].values[0], 'lat': df['LAT_proj'].values[0]}
      df.drop(columns=['LON_proj', 'LAT_proj'], inplace=True)
   else:
      if XovOpt.get("debug"):
         print("### Empty proj_df_xovi on xov #", xovi)
         print(df)
      return  # continue

   # populate all_xov with local data for specific xov
   all_xov.ladata_df = df.copy()
   all_xov.tracks = dict(zip(df.orbID.unique(), list(range(2))))
   all_xov.ladata_df['orbID'] = all_xov.ladata_df['orbID'].map(all_xov.tracks)

   msrm_smpl = all_xov.msrm_sampl
    
   # attributes needed in all_xov: msrm_sampl, ladata_df, tracks, proj_center

   try:
      x, y, subldA, subldB, ldA, ldB = all_xov.get_xover_fine([msrm_smpl], [msrm_smpl], msrm_smpl, '')
   except:
      # if XovOpt.get("debug"):
      print("### get_xover_fine issue on xov #", xovi)
      print(all_xov.ladata_df)
      return  # continue

   # You should check if there is indeed an intersection which has beeen found!
   if len(subldA) == 0:
      print("No intersection found on xov", int(xovi))
      return
   # last alti point before xov
   mla_idA = all_xov.ladata_df[all_xov.ladata_df['orbID'] == 0]['seqid'].iloc[int(subldA[0])]
   mla_idB = all_xov.ladata_df[all_xov.ladata_df['orbID'] == 1]['seqid'].iloc[int(subldB[0])]

   return x[0], y[0], mla_idA, mla_idB, ldA[0], ldB[0]

def compute_fine_xov(mla_proj_df, fine_xov_df, n_interp):
   # Using previously computed intersection, compute all the necessary
   # quantitied related to the crossovers

   start_finexov = time.time()
   # initialize xov object
   xovs_list = mla_proj_df.xovid.unique()
   all_xov = xov(XovOpt.get("vecopts"))
   # pass measurements sampling to fine_xov computation through all_xov
   all_xov.msrm_sampl = n_interp

   # if huge_dataset:
   #     for chunk_path in chunk_proj_df_paths:
   #         mla_proj_df = pd.read_pickle(chunk_path)
   #         print(mla_proj_df)
   #         exit()
   # select what to keep
   cols_to_keep = mla_proj_df.loc[:,
                                  mla_proj_df.columns.str.startswith('ET_BC') + mla_proj_df.columns.str.startswith('X_stgprj') +
                                  mla_proj_df.columns.str.startswith('Y_stgprj') + mla_proj_df.columns.str.startswith(
                                     'dR/')].columns
   proj_df_tmp = mla_proj_df[
      ['orbID', 'seqid_mla', 'ET_TX', 'LON', 'LAT', 'R', 'dt', 'offnadir', 'xovid', 'LON_proj', 'LAT_proj'] + list(
         cols_to_keep)]
   print("total memory proj_df_tmp:", proj_df_tmp.memory_usage(deep=True).sum() * 1.e-6)
    
   # WD: Check above what is really useful now that the search is done way before
    
   if XovOpt.get("parallel"):
      # pool = mp.Pool(processes=ncores)  # mp.cpu_count())
      # xov_list = pool.map(fine_xov_proc, args)  # parallel
      # apply_async example
      if XovOpt.get("local"):
         from tqdm import tqdm
         pbar = tqdm(total=len(xovs_list))

         def update(*a):
            pbar.update()

      xov_list = []
      with mp.get_context("spawn").Pool(processes=XovOpt.get("n_proc")) as pool:
         # for idx in range(len(xovs_list)):
         # xov_list.append(pool.apply_async(fine_xov_proc, args=(xovs_list[idx], dfl[idx], all_xov), callback=update))
         for xovi in xovs_list:
            
            if XovOpt.get("local"):
               # xov_list.append(pool.apply_async(fine_xov_proc, args=(
               #     xovi, proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi], all_xov), callback=update))
               xov_list.append(pool.apply_async(fine_compute_xov_proc, args=(
                  xovi, proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi], all_xov,
                  fine_xov_df.loc[fine_xov_df['xovi'] == xovi], n_interp), callback=update))
            else:
               # xov_list.append(pool.apply_async(fine_xov_proc, args=(
               #     xovi, proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi], all_xov)))
               xov_list.append(pool.apply_async(fine_compute_xov_proc, args=(
                  xovi, proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi], all_xov,
                  fine_xov_df.loc[fine_xov_df['xovi'] == xovi], n_interp)))

         pool.close()
         pool.join()

        # blocks until all results are fetched
      tmpl = []
      for idx, r in enumerate(xov_list):
         try:
            tmpl.append(r.get())
         except:
            print("r.get failed on", idx)
      xov_list = tmpl
      print(len(xov_list))
      # xov_list = [r.get() for r in xov_list]

      # launch once in serial mode to get ancillary values
      # fine_xov_proc(0, proj_df_tmp.loc[proj_df_tmp['xovid'] == 0], all_xov)
      fine_compute_xov_proc(0, proj_df_tmp.loc[proj_df_tmp['xovid'] == 0], all_xov,
                            fine_xov_df.loc[fine_xov_df['xovi'] == 0], n_interp)
      # assign xovers to the new all_xov containing ancillary values
      all_xov.xovers = pd.concat(xov_list, axis=0)

   else:
      if XovOpt.get("local"):
         from tqdm import tqdm
         for xovi in tqdm(xovs_list):
            # fine_xov_proc(xovi, proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi], all_xov)
            # xov_tmp.xovers = fine_xov_proc(xovi, proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi], all_xov)
            fine_compute_xov_proc(xovi, proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi], all_xov,
                                  fine_xov_df.loc[fine_xov_df['xovi'] == xovi], n_interp)
      else:
         for xovi in xovs_list:
            # fine_xov_proc(xovi, proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi], all_xov)
            fine_compute_xov_proc(xovi, proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi], all_xov,
                                  fine_xov_df.loc[fine_xov_df['xovi'] == xovi], n_interp)

   # fill xov structure with info for LS solution
   all_xov.parOrb_xy = [x for x in all_xov.xovers.filter(regex='^dR/[a-zA-Z0-9]+_.*$').columns]  # update partials list
   all_xov.parGlo_xy = [(a + b) for a in ['dR/'] for b in list(XovOpt.get("parGlo").keys())]
   all_xov.par_xy = all_xov.parOrb_xy + all_xov.parGlo_xy  # update partials list
   if XovOpt.get("debug"):
      print("Parameters:", all_xov.parOrb_xy, all_xov.parGlo_xy, all_xov.par_xy)
   # update xovers table with LAT and LON
   if len(all_xov.xovers) > 0:
      all_xov = get_xov_latlon(all_xov, mla_proj_df.loc[mla_proj_df.partid == 'none'])
      all_xov.xovers.drop('xovid', axis=1).reset_index(inplace=True, drop=True)
   if XovOpt.get("debug"):
      pd.set_option('display.max_columns', 500)
      pd.set_option('display.max_rows', 500)

   # print(all_xov.xovers)

   end_finexov = time.time()
   print("Fine_xov finished after", int(end_finexov - start_finexov), "sec or ",
         round((end_finexov - start_finexov) / 60., 2), " min and located", len(all_xov.xovers), "out of previous",
         len(xovs_list), "xovers!")
   return all_xov

def fine_compute_xov_proc(xovi, df, all_xov, fine_xov_df, n_interp):

   df = df.reset_index(drop=True).reset_index()
   df.rename(columns={'index': 'genID', 'seqid_mla': 'seqid'}, inplace=True)
    
   if len(df) > 0:  # and {'LON_proj','LAT_proj'}.issubset(df.columns):
      all_xov.proj_center = {'lon': df['LON_proj'].values[0], 'lat': df['LAT_proj'].values[0]}
      df.drop(columns=['LON_proj', 'LAT_proj'], inplace=True)
   else:
      if XovOpt.get("debug"):
         print("### Empty proj_df_xovi on xov #", xovi)
         print(df)
      return  # continue

   # populate all_xov with local data for specific xov
   all_xov.ladata_df = df.copy()
   all_xov.tracks = dict(zip(df.orbID.unique(), list(range(2))))
   all_xov.ladata_df['orbID'] = all_xov.ladata_df['orbID'].map(all_xov.tracks)

   [x, y, mla_idA, mla_idB, ldA, ldB] = fine_xov_df[['x', 'y', 'mla_idA', 'mla_idB', 'ldA', 'ldB']].values[0]
   
   if (np.isnan(mla_idA)):
      print(f"xov {xovi} ignored in compute_fine_xov")
      return
   
   # Compute ldX from seqidX
   # would be nicer, but get_elev would need to be modified
   # ldA = np.where(all_xov.ladata_df[all_xov.ladata_df['orbID'] == 0]['seqid']==np.floor(ind_A))[0]
   # ldB = np.where(all_xov.ladata_df[all_xov.ladata_df['orbID'] == 1]['seqid']==np.floor(ind_B))[0]
   # retrieve point just before xover
   ind_A = np.where((all_xov.ladata_df['orbID'] == 0) & (all_xov.ladata_df['seqid']==np.floor(mla_idA)))[0]
   ind_B = np.where((all_xov.ladata_df['orbID'] == 1) & (all_xov.ladata_df['seqid']==np.floor(mla_idB)))[0]

   ldA = ind_A[0] + (ldA-np.floor(ldA))
   ldB = ind_B[0] + (ldB-np.floor(ldB))

   ldA, ldB, R_A, R_B = all_xov.get_elev('', ldA, ldB, n_interp, x=x, y=y)

   out = np.vstack((x, y, ldA, ldB, R_A, R_B)).T

   if len(out) == 0:
      if XovOpt.get("debug"):
         print("### Empty out on xov #", xovi)
      return  # continue

   # post-processing
   xovtmp = all_xov.postpro_xov_elev(all_xov.ladata_df, out)
   try:
      xovtmp = pd.DataFrame(xovtmp, index=[0])  # TODO possibly avoid, very time consuming
   except:
      print("issue with xovtmp")
      print(xovtmp)
      return

   if len(xovtmp) > 1:
      if XovOpt.get("debug"):
         print("### Bad multi-xov at xov#", xovi)
         print(xovtmp)
      return  # continue

   # Update xovtmp as attribute for partials
   all_xov.xovtmp = xovtmp

   # Compute and store distances between obs and xov coord
   all_xov.set_xov_obs_dist()
   # Compute and store offnadir state for obs around xov
   all_xov.set_xov_offnadir()

   # process partials from gtrack to xov, if required
   if XovOpt.get("partials"):
      all_xov.set_partials(n_interp)
   else:
      # retrieve epoch to, e.g., trace tracks quality (else done inside set_partials)
      all_xov.xovtmp = pd.concat([all_xov.xovtmp, pd.DataFrame(
         np.reshape(all_xov.get_dt(all_xov.ladata_df, all_xov.xovtmp), (len(all_xov.xovtmp), 2)),
         columns=['dtA', 'dtB'])], axis=1)
      all_xov.xovtmp = pd.concat([all_xov.xovtmp, pd.DataFrame(
         np.reshape(all_xov.get_tX(all_xov.ladata_df, all_xov.xovtmp), (len(all_xov.xovtmp), 2)),
         columns=['tA', 'tB'])], axis=1)

   # Remap track names to df
   all_xov.xovtmp['orbA'] = all_xov.xovtmp['orbA'].map({v: k for k, v in all_xov.tracks.items()})
   all_xov.xovtmp['orbB'] = all_xov.xovtmp['orbB'].map({v: k for k, v in all_xov.tracks.items()})

   all_xov.xovtmp['xOvID'] = xovi

   # Update general df (serial only, does not work in parallel since not a shared object)
   if not XovOpt.get("parallel"):
      # all_xov.xovers = all_xov.xovers.append(all_xov.xovtmp, sort=True)
      all_xov.xovers = pd.concat([all_xov.xovers, all_xov.xovtmp], sort=True)

   # print used memory
   if XovOpt.get('debug'):
      import sys
      def sizeof_fmt(num, suffix='B'):
         ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
         for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
               return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
            return "%.1f %s%s" % (num, 'Yi', suffix)

      print("iter over xov:", xovi)
      for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                               key=lambda xm: -xm[1])[:10]:
         print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

   return all_xov.xovtmp