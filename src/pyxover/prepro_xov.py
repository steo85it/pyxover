import logging
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from pyxover.xov_utils import get_ds_attrib
from xovutil.project_coord import project_stereographic
from pyxover.intersection import intersection
from pyxover.fine_xov import fine_xov_intersection
from pyxover.project_gtracks import project_mla
from pyxover.xov_prc_iters import load_mla_df

# from examples.MLA.options import XovOpt.get("vecopts"), XovOpt.get("cloop_sim"), XovOpt.get("outdir"), XovOpt.get("partials")
from config import XovOpt
# from memory_profiler import profile

# Deprecated module
# @profile

def prepro_mla_xov(old_xovs, msrm_smpl, gtrack_dirs, cmb):
    start_prepro = time.time()

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
    
    # Populate mladata with laser altimeter data for each track
    mladata = load_mla_df(gtrack_dirs, tracks_in_xovs, columns)
    
    # Projection w/o partials
    mla_proj_df, part_proj_dict = extract_mla_xov(old_xovs, tracks_in_xovs, mladata, msrm_smpl, False)
    
    mla_proj_df = project_mla(mla_proj_df, part_proj_dict, False)

    fine_xov_df = fine_xov_intersection(mla_proj_df, msrm_smpl) # to return anyway 
    
    # update ['mla_idA', 'mla_idB'] in old_xovs based on fine_xov_df
    
    n_interp = XovOpt.get("n_interp")

    # Refine xovers before projection
    # WD: In this case, fine search is unecessarily done twice...
    if n_interp < msrm_smpl:
       for index, xov in old_xovs.iterrows():

         # extract mladata around xovers
         tmp_ladataA = mladata[xov['orbA']][(mladata[xov['orbA']]['seqid'] >= xov['mla_idA']-msrm_smpl)
                                           & (mladata[xov['orbA']]['seqid'] <= xov['mla_idA']+msrm_smpl)]
         tmp_ladataB = mladata[xov['orbB']][(mladata[xov['orbB']]['seqid'] >= xov['mla_idB']-msrm_smpl)
                                          & (mladata[xov['orbB']]['seqid'] <= xov['mla_idB']+msrm_smpl)]

         # Project mladata around rough xovers for each tracks
         # The projection is done once more later, but with partials
         la_projA = project_stereographic(tmp_ladataA['LON'],
                                           tmp_ladataA['LAT'],
                                           xov['LON'],
                                           xov['LAT'],
                                           R=XovOpt.get("vecopts")['PLANETRADIUS'])

         la_projB = project_stereographic(tmp_ladataB['LON'],
                                           tmp_ladataB['LAT'],
                                           xov['LON'],
                                           xov['LAT'],
                                           R=XovOpt.get("vecopts")['PLANETRADIUS'])

         la_projA_df = pd.DataFrame(np.asarray(la_projA).T, columns=['X_stgprj', 'Y_stgprj'])
         la_projB_df = pd.DataFrame(np.asarray(la_projB).T, columns=['X_stgprj', 'Y_stgprj'])

         # Search for intersection between the projected ground tracks
         x, y, ind_A, ind_B = intersection(la_projA_df['X_stgprj'],la_projA_df['Y_stgprj'],
                                            la_projB_df['X_stgprj'],la_projB_df['Y_stgprj'])

         # Assign the closest mla data point
         xov['mla_idA'] = tmp_ladataA['seqid'].iloc[0] + np.floor(ind_A[0])
         xov['mla_idB'] = tmp_ladataB['seqid'].iloc[0] + np.floor(ind_B[0])

    # Extract relevant mladata for the tracks_in_xovs
    mla_proj_df, part_proj_dict = extract_mla_xov(old_xovs, tracks_in_xovs, mladata, n_interp, XovOpt.get("partials"))
    
    # free-up memory
    mladata.clear()

    end_prepro = time.time()

    print("Pre-processing finished after ", int(end_prepro - start_prepro), "sec or ",
          round((end_prepro - start_prepro) / 60., 2), " min!")
    
    mla_proj_df = project_mla(mla_proj_df, part_proj_dict, XovOpt.get("partials"))

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
            if True: #track_id[:2] == str(cmb[idx]):
                # track = track.load(outdir + outdir_in + 'gtrack_' + str(cmb[idx]) + '/gtrack_' + track_id + '.pkl')
                tmp_ladata = mladata[track_id]  # faster and less I/O
            else:
                # print("### Track ",track_id,"does not exist in",outdir + outdir_in + 'gtrack_' + str(cmb[idx]))
                # track = gtrack(vecopts) # reinitialize track to avoid error "'NoneType' object has no attribute 'load'"
                continue

            # tmp_ladata = track.ladata_df.copy()  # .reset_index()

            # xov df extract and expand to neighb obs
            tmp = old_xovs.loc[old_xovs[orb] == track_id][[mla_idx[idx], 'xOvID', 'LON', 'LAT']].astype(
                {mla_idx[idx]: int, 'xOvID': int, 'LON': float, 'LAT': float}).round({'LON': 3, 'LAT': 3}).values

            if len(tmp) == 0:
                logging.debug(f"no xovers for track={track_id} and orb={orb}, xov df extract empty. Continue.")
                continue
                # exit()

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