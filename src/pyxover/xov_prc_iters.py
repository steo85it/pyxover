import glob
import os
import time
import logging
import pandas as pd
import numpy as np
from collections import defaultdict

from pygeoloc.ground_track import gtrack

from pyxover.fine_xov import compute_fine_xov, fine_xov_intersection
from pyxover.project_gtracks import project_mla
from pyxover.xov_utils import get_ds_attrib

from accumxov.Amat import Amat
from config import XovOpt
from memory_profiler import profile

## MAIN ##
def xov_prc_iters_run(outdir_in, cmb, old_xovs, gtrack_dirs):
   start = time.time()
   xov_dir = XovOpt.get("outdir") + outdir_in + 'xov/'
   outpath = xov_dir + 'xov_' + str(cmb[0]) + '_' + str(cmb[1]) + '.pkl'
   # Exit process if file already exists and no option to recreate
   if (XovOpt.get("new_xov") != 2) and (os.path.isfile(outpath)):
      print("Fine xov", outpath," already exists. Stop!")
      return

   # Compute fine intersection from old xovers and
   # project the la data around the fine intersection
   mla_proj_df, fine_xov_df = proj_around_intersection(outdir_in, cmb, old_xovs, gtrack_dirs)

    # compute new xovs
   xov_tmp = compute_fine_xov(mla_proj_df, fine_xov_df, XovOpt.get("n_interp"))
    
   # Load auxilliary data from gtracks
   xov_tmp.store_pertubation(gtrack_dirs, cmb)

   # Save to file
   xov_pklname = 'xov_' + str(cmb[0]) + '_' + str(cmb[1]) + '.pkl'  # one can split the df by trackA and save multiple pkl, one for each trackA if preferred
   xov_tmp.save(xov_dir + xov_pklname)

   end = time.time()

   print('Xov for ' + str(cmb) + ' processed and written to ' + xov_dir + xov_pklname +
         '@' + time.strftime("%H:%M:%S", time.gmtime()))

   print("Fine xov determination finished after", int(end - start), "sec or ", round((end - start) / 60., 2), " min!")
   # print(xov_tmp.xovers.columns)
   print(xov_tmp.xovers.dR)
   return xov_tmp

# @profile
def proj_around_intersection(outdir_in, cmb, old_xovs, gtrack_dirs):
   """
   Fine search of the intersection and projection of mla_data around (rough/fine) xover
   If n_interp < msrm_smpl, the fine search is performed on initial projection
   (w/o partials) for memory efficiency
   """
   
   start = time.time()
   proj_dir =  XovOpt.get("outdir") + outdir_in + 'xov/tmp/proj/'

   # create useful dirs recursively
   os.makedirs(proj_dir, exist_ok=True)
    
   # TODO remove check on orbits for this test
   if not XovOpt.get("weekly_sets"):
      old_xovs = old_xovs.loc[
         (old_xovs['orbA'].str.startswith(str(cmb[0]))) & (old_xovs['orbB'].str.startswith(str(cmb[1])))]

   tracks_in_xovs = np.unique(old_xovs[['orbA', 'orbB']].values)
   print("Processing", len(tracks_in_xovs), "tracks, previously resulting in", len(old_xovs), "xovers.")
   # check if tracks to process in this combination
   if len(tracks_in_xovs)==0:
      print("No tracks to be processed. Stop!")
      exit()

   delta_pars, etbcs, pars = get_ds_attrib()
   columns = ['seqid', 'LON', 'LAT', 'orbID', 'ET_BC', 'ET_TX', 'R', 'offnadir','dt'] + pars + etbcs
   
   msrm_smpl = XovOpt.get("msrm_sampl")  # should be even...
   n_interp = XovOpt.get("n_interp")

   # Populate mladata with laser altimeter data for each track
   if not XovOpt.get("import_proj") or n_interp < msrm_smpl:
      mladata = load_mla_df(gtrack_dirs, tracks_in_xovs, columns)
   
   # First projection w/o partials only on a larger set of mla, 
   # to fine search the intersection more efficiently        
   if n_interp < msrm_smpl:

      # Projection w/o partials
      mla_proj_df, part_proj_dict = extract_mla_xov(old_xovs, tracks_in_xovs, mladata, msrm_smpl, False)
      
      if XovOpt.get("import_proj"):
         del mladata
      
      mla_proj_df = project_mla(mla_proj_df, part_proj_dict, False)
      
      # Fine search
      fine_xov_df = fine_xov_intersection(mla_proj_df, msrm_smpl) # to return anyway
      # fine_xov_df[['xovi','mla_idA','mla_idB']] = fine_xov_df[['xovi','mla_idA','mla_idB']].astype('int')
      
      # Free-up memory
      del mla_proj_df
      
      # Update old_xovs based on fine_xov_df
      old_xovs=old_xovs.set_index('xOvID')
      old_xovs.update(fine_xov_df.set_index('xovi'))
      old_xovs=old_xovs.reset_index()

   elif n_interp > msrm_smpl:
      print(f"n_interp ({n_interp}) can't be > msrm_smpl{msrm_smpl}")
      exit()

   # Projection of mla_data around old xovs (w/ partials if needed)
   # Old projection should not be used if one expect the xovers further
   # than "interpolation distance (n_interp)"
   proj_pkl_path = proj_dir + 'xov_' + str(cmb[0]) + '_' + str(cmb[1]) + '_project.pkl.gz'
   if not XovOpt.get("import_proj"): # compute projection
      
      # Extract relevant mladata for the tracks_in_xovs
      mla_proj_df, part_proj_dict = extract_mla_xov(old_xovs, tracks_in_xovs, mladata, n_interp, XovOpt.get("partials"))

      # free-up memory
      mladata.clear()

      mla_proj_df = project_mla(mla_proj_df, part_proj_dict, XovOpt.get("partials"))
      
      # free-up memory
      part_proj_dict.clear()

      # Save intermediate result
      # WD: col = ['R_A', 'R_B', 'dR'] are 0 -> drop them?
      mla_proj_df.to_pickle(proj_pkl_path)
      print("Projected df saved to:", proj_pkl_path)

   elif os.path.exists(proj_pkl_path): # or just retrieve them from file
      mla_proj_df = pd.read_pickle(proj_pkl_path)
      print("mla_proj_df loaded from", proj_pkl_path, ". Done!!")
   else:
      print("No mla_proj_df found at ", proj_pkl_path)
      exit()

   # Fine search with mla projectec with partials
   if n_interp == msrm_smpl: # also > ?
      fine_xov_df = fine_xov_intersection(mla_proj_df, msrm_smpl) # to return anyway

   end = time.time()

   print("Pre-processing finished after", int(end - start), "sec or ", round((end - start) / 60., 2), " min!")
   return mla_proj_df, fine_xov_df

def extract_mla_xov(old_xovs, tracks_in_xovs, mladata, n_interp, partials):

   delta_pars, etbcs, pars = get_ds_attrib()
   # part_proj_dict = dict.fromkeys([x + '_' + y for x in delta_pars.keys() for y in ['p', 'm']], [])
   part_proj_dict = defaultdict(list)
   mla_proj_list = []
   mla_idx = ['mla_idA', 'mla_idB']

   # Extract relevant mladata for the tracks_in_xovs
   for track_id in tracks_in_xovs[:]:
      for idx, orb in enumerate(['orbA', 'orbB']):
         
         # load file if year of track corresponds to folder
         # TODO removed check on orbid for this test
         tmp_ladata = mladata[track_id]  # faster and less I/O

         # xov df extract and expand to neighb obs
         tmp = old_xovs.loc[old_xovs[orb] == track_id][[mla_idx[idx], 'xOvID', 'LON', 'LAT']].astype(
            {mla_idx[idx]: int, 'xOvID': int, 'LON': float, 'LAT': float}).round({'LON': 3, 'LAT': 3}).values

         if len(tmp) == 0:
            logging.debug(f"no xovers for track={track_id} and orb={orb}, xov df extract empty. Continue.")
            continue

         # WD: I don't understand why seqid and xovid are not int at this point...
         xov_extract = pd.DataFrame(tmp, columns=['seqid', 'xovid', 'LON_proj', 'LAT_proj'])
         xov_extract.sort_values(by='seqid', inplace=True)

         # Get geolocated observation with same seqid (unique LA range #)
         rows = tmp_ladata['seqid'].loc[tmp_ladata['seqid'].isin(xov_extract.loc[:, 'seqid'])]  # .index)  #
         # Check if multiple xovers at same mla index
         multiple_counts = xov_extract.loc[:, 'seqid'].value_counts()  # .loc[lambda x: x>1]

         # WD: Is the goal to create a unique genid, with the number of occurences?
         # genid should be unique for this batch of LA ranges (?)
         if len(multiple_counts[multiple_counts > 1]) > 0:
            # print(rows.to_frame())
            rows = rows.reset_index()
            # multiple_counts = multiple_counts.reset_index()
            tmp = pd.merge(rows, multiple_counts, left_on='seqid',
                           right_index=True).drop(['seqid'], axis=1)
            tmp.columns = ["genid", "dupl"]
            rows = tmp.reindex(tmp.index.repeat(tmp.dupl)).set_index('genid')

         xov_extract['index'] = rows.index

         xov_extract = xov_extract[['index', 'seqid', 'xovid', 'LON_proj', 'LAT_proj']]

         tmp = xov_extract.values
         mla_close_to_xov = np.ravel([np.arange(i[0] - n_interp, i[0] + (n_interp + 1), 1) for i in tmp])
         tmp = np.hstack(
            [mla_close_to_xov[:, np.newaxis], np.reshape(np.tile(tmp[:, 1:], (n_interp * 2 + 1)), (-1, 4))])
         # genid: a local id for la_ranges; seqid: global id for la_ranges; xovid: global id for xovers
         # the same genid and seqid can belong to multiple xovers (so that genid can be not unique)
         xov_extract = pd.DataFrame(tmp, columns=['genid', 'seqid', 'xovid', 'LON_proj', 'LAT_proj'])

         ladata_extract = tmp_ladata.loc[tmp_ladata.index.isin(xov_extract.genid)][
            ['seqid', 'LON', 'LAT', 'orbID', 'ET_BC', 'ET_TX', 'R', 'offnadir','dt']]

         mla_proj_df = pd.merge(xov_extract, ladata_extract, left_on='genid', right_index=True)
         mla_proj_df['partid'] = 'none'
            

         if partials:
            # do stuff (unsure if sorted in the same way... merge would be safer)
            tmp_ladata_partials = tmp_ladata.loc[tmp_ladata.index.isin(xov_extract.genid)][
               ['seqid', 'LON', 'LAT', 'orbID'] + pars + etbcs]

            # update dict keys to multiply derivatives by deltas for LON and LAT and get increments
            delta_pars_tmp = dict(zip(['dLON/' + x for x in delta_pars.keys()],
                                      [np.linalg.norm(x) for x in list(delta_pars.values())]))
            delta_pars_tmp.update(dict(zip(['dLAT/' + x for x in delta_pars.keys()],
                                           [np.linalg.norm(x) for x in list(delta_pars.values())])))
            part_deltas_df = pd.Series(delta_pars_tmp) * tmp_ladata_partials.loc[:, pars].astype(float)

            # use increment to extrapolate new LON and LAT values
            newlon_p = part_deltas_df.loc[:, part_deltas_df.columns.str.startswith("dLON")].add(
               tmp_ladata_partials.LON, axis=0)
            newlat_p = part_deltas_df.loc[:, part_deltas_df.columns.str.startswith("dLAT")].add(
               tmp_ladata_partials.LAT, axis=0)
            newlon_m = -1. * (
               part_deltas_df.loc[:, part_deltas_df.columns.str.startswith("dLON")].sub(tmp_ladata_partials.LON,
                                                                                        axis=0))
            newlat_m = -1. * (
               part_deltas_df.loc[:, part_deltas_df.columns.str.startswith("dLAT")].sub(tmp_ladata_partials.LAT,
                                                                                        axis=0))

            # update column names
            upd_cols = [x.split('/')[-1] for x in newlat_p.columns]
            newlon_p.columns = ['LON_' + x + '_p' for x in upd_cols]
            newlat_p.columns = ['LAT_' + x + '_p' for x in upd_cols]
            newlon_m.columns = ['LON_' + x + '_m' for x in upd_cols]
            newlat_m.columns = ['LAT_' + x + '_m' for x in upd_cols]

            # concatenate partials with LON/LAT associated to xovers (improve efficiency?)
            tmp_mla_proj = mla_proj_df[['genid', 'LON_proj', 'LAT_proj']]
            for idx, pder in enumerate(['_' + x + '_' + y for x in delta_pars.keys() for y in ['p', 'm']]):
               if pder[-1] == 'p':
                  tmp = pd.concat([
                     newlon_p['LON' + pder], newlat_p['LAT' + pder],
                     tmp_ladata_partials['ET_BC' + pder], tmp_ladata_partials['dR/' + pder[1:-2]]
                     ], axis=1)
               else:
                  tmp = pd.concat([
                     newlon_m['LON' + pder], newlat_m['LAT' + pder],
                     tmp_ladata_partials['ET_BC' + pder], tmp_ladata_partials['dR/' + pder[1:-2]]
                     ], axis=1)
               # Concatenate all Series into a single DataFrame
               tmp['genid'] = tmp.index
               # Merge the partials DataFrame with the lonlat DataFrame
               tmp = tmp_mla_proj.merge(tmp, on='genid', how='left')
               # REMEMBER: tmp.columns = genid, LON_proj, LAT_proj, LON, LAT, ET_BC, dR/dp
               part_proj_dict[idx].append(tmp.values)

         mla_proj_list.append(mla_proj_df)

   # concatenate lists
   mla_proj_df = pd.concat(mla_proj_list, sort=False).reset_index(drop=True)
   mla_proj_df.rename(columns={"seqid_x": "seqid_xov", "seqid_y": "seqid_mla"},
                      inplace=True)  # remember that seqid_mla is not continuous because of eventual bad obs
   part_proj_dict = {x: np.vstack(y) for x, y in part_proj_dict.items()}
   part_proj_dict = dict(zip([x + '_' + y for x in delta_pars.keys() for y in ['p', 'm']], part_proj_dict.values()))
   
   part_proj_dict.update({'none': mla_proj_df[['genid', 'LON_proj', 'LAT_proj', 'LON', 'LAT']].values})
   
   # save main part
   # mla_proj_df.to_pickle(outdir + outdir_in + 'xov/tmp/proj/xov_' + str(cmb[0]) + '_' + str(
   #     cmb[1]) + '_mla_proj.pkl.gz')

   return mla_proj_df, part_proj_dict
 
def load_mla_df(gtrack_dirs, tracks_in_xovs, columns):
   # Populate mladata with laser altimeter data for each track

   track = gtrack(XovOpt.to_dict())
   mladata = {}

   # WD: Whole gtrack is not needed.ladata_df could be passed as an
   # argument, provided that necessary columns havent been dumped.
   for track_id in tracks_in_xovs[:]:
      track.load_df_from_id(gtrack_dirs[0], track_id)
      if track.ladata_df is None:
         track.load_df_from_id(gtrack_dirs[1], track_id)
      if track.ladata_df is None:
         print(f"*** PyXover: Issue loading ladata from {track_id} from {gtrack_dirs}.")
         exit()
      mladata[track_id] = track.ladata_df[columns]

   return mladata
 
def retrieve_xov(outdir_in, xov_iter, cmb, useful_columns):
   # depending on available input xov, get xovers location from AbMat or from xov_rough
   if xov_iter > 0 or XovOpt.get("import_abmat")[0]:  # len(input_xov)==0:
      # read old abmat file
      if xov_iter > 0:
         outdir_old = outdir_in.replace('_' + str(xov_iter) + '/', '_' + str(xov_iter - 1) + '/')
         abmat = XovOpt.get("outdir") + outdir_old + 'Abmat*.pkl'
      else: # read a user defined abmat file
         abmat = XovOpt.get("import_abmat")[1]

      tmp_Amat = Amat(XovOpt.get("vecopts"))
      tmp = tmp_Amat.load(glob.glob(abmat)[0])
      old_xovs = tmp.xov.xovers[useful_columns]
   else:
      xov_dir = XovOpt.get("outdir") + outdir_in + 'xov/'
      input_xov_path = xov_dir + 'tmp/xovin_' + str(cmb[0]) + '_' + str(cmb[1]) + '.pkl.gz'
      if not XovOpt.get("compute_input_xov"):
         if os.path.exists(input_xov_path):
            input_xov = pd.read_pickle(input_xov_path)
            print("Input xovs read from", input_xov_path, ". Done!")
         else:
            print("No xov file at", input_xov_path)
            exit()
         # reindex and keep only useful columns
         old_xovs = input_xov[useful_columns]
         old_xovs = old_xovs.drop('xOvID', axis=1).rename_axis('xOvID').reset_index()
      print(old_xovs)
      
   return old_xovs

## MAIN ##
if __name__ == '__main__':
   start = time.time()

   cmb = [12,18]

   xov_tmp = xov_prc_iters_run()

   end = time.time()

   print("Process finished after", int(end - start), "sec or ", round((end - start)/60.,2), " min!")
