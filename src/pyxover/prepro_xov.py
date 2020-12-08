import time
from collections import defaultdict

import numpy as np
import pandas as pd
from pygeoloc.ground_track import gtrack
from pyxover.xov_utils import get_ds_attrib

from examples.MLA.options import vecopts, cloop_sim, outdir, partials


def prepro_mla_xov(old_xovs, msrm_smpl, outdir_in, cmb):
    start_prepro = time.time()

    # print(tmp.xov.xovers.columns)
    old_xovs = old_xovs.loc[
        (old_xovs['orbA'].str.startswith(str(cmb[0]))) & (old_xovs['orbB'].str.startswith(str(cmb[1])))]
    # print(tmp.xov.xovers.loc[tmp.xov.xovers.xOvID.isin([7249,7526,1212,8678,34,11436,625])][['R_A','R_B','x0','y0']])
    # exit()
    # print(old_xovs.loc[old_xovs['orbA'].isin(['1502130018','1504152013'])])
    # exit()
    # print(old_xovs)

    tracks_in_xovs = np.unique(old_xovs[['orbA', 'orbB']].values)
    print("Processing", len(tracks_in_xovs), "tracks, previously resulting in", len(old_xovs), "xovers.")
    # check if tracks to process in this combination
    if len(tracks_in_xovs)==0:
        print("No tracks to be processed. Stop!")
        exit()

    track = gtrack(vecopts)
    delta_pars, etbcs, pars = get_ds_attrib()
    # part_proj_dict = dict.fromkeys([x + '_' + y for x in delta_pars.keys() for y in ['p', 'm']], [])
    # print(part_proj_dict)
    part_proj_dict = defaultdict(list)
    mla_proj_list = []
    mla_idx = ['mla_idA', 'mla_idB']
    mladata = {}

    for track_id in tracks_in_xovs[:]:
        # if track_id in ['1502130018','1502202222']:
        # print(track_id)
        if cloop_sim:
            trackfil = outdir + outdir_in + 'gtrack_' + track_id[:2] + '/gtrack_' + track_id[:-2] + '*.pkl'
        else:
            trackfil = outdir + outdir_in + 'gtrack_' + track_id[:2] + '/gtrack_' + track_id + '.pkl'
        track = track.load(trackfil)
        mladata[track_id] = track.ladata_df

    for track_id in tracks_in_xovs[:]:
        # if track_id in ['1502130018','1502202222']:

        for idx, orb in enumerate(['orbA', 'orbB']):
            # print(idx,orb)
            # print(outdir + outdir_in + 'gtrack_' + str(cmb[idx]) + '/gtrack_' + track_id + '.pkl')

            # load file if year of track corresponds to folder
            if track_id[:2] == str(cmb[idx]):
                # track = track.load(outdir + outdir_in + 'gtrack_' + str(cmb[idx]) + '/gtrack_' + track_id + '.pkl')
                tmp_ladata = mladata[track_id]  # faster and less I/O
            else:
                # print("### Track ",track_id,"does not exist in",outdir + outdir_in + 'gtrack_' + str(cmb[idx]))
                # track = gtrack(vecopts) # reinitialize track to avoid error "'NoneType' object has no attribute 'load'"
                continue

            # exit()
            # tmp_ladata = track.ladata_df.copy()  # .reset_index()

            # xov df extract and expand to neighb obs
            tmp = old_xovs.loc[old_xovs[orb] == track_id][[mla_idx[idx], 'xOvID', 'LON', 'LAT']].astype(
                {mla_idx[idx]: int, 'xOvID': int, 'LON': float, 'LAT': float}).round({'LON': 3, 'LAT': 3}).values
            # print(tmp)
            # sampl = 10
            xov_extract = pd.DataFrame(tmp, columns=['seqid', 'xovid', 'LON_proj', 'LAT_proj'])
            xov_extract.sort_values(by='seqid', inplace=True)
            # print(xov_extract)
            ########### TEST
            # xov_extract = xov_extract.loc[xov_extract.xovid == 0]
            ###########
            # exit()
            # # get geoloc obs with same seqid
            rows = tmp_ladata['seqid'].loc[tmp_ladata['seqid'].isin(xov_extract.loc[:, 'seqid'])]  # .index)  #
            # check if multiple xovers at same mla index
            multiple_counts = xov_extract.loc[:, 'seqid'].value_counts()  # .loc[lambda x: x>1]

            if len(multiple_counts.loc[lambda x: x > 1]) > 0:
                # print(rows.to_frame())
                tmp = pd.merge(rows.reset_index(), multiple_counts.reset_index(), left_on='seqid',
                               right_on='index').drop(['seqid_x', 'index_y'], axis=1)
                tmp.columns = ["genid", "dupl"]
                rows = tmp.reindex(tmp.index.repeat(tmp.dupl)).set_index('genid')

            xov_extract['index'] = rows.index

            xov_extract = xov_extract[['index', 'seqid', 'xovid', 'LON_proj', 'LAT_proj']]
            # print("xov",xov_extract)

            tmp = xov_extract.values
            mla_close_to_xov = np.ravel([np.arange(i[0] - msrm_smpl, i[0] + (msrm_smpl + 1), 1) for i in tmp])
            tmp = np.hstack(
                [mla_close_to_xov[:, np.newaxis], np.reshape(np.tile(tmp[:, 1:], (msrm_smpl * 2 + 1)), (-1, 4))])
            xov_extract = pd.DataFrame(tmp, columns=['genid', 'seqid', 'xovid', 'LON_proj', 'LAT_proj'])

            ladata_extract = tmp_ladata.loc[tmp_ladata.reset_index()['index'].isin(xov_extract.genid)][
                ['seqid', 'LON', 'LAT', 'orbID', 'ET_BC', 'ET_TX', 'R', 'offnadir','dt']].reset_index()
            # print("ladata",ladata_extract)
            # if idx>0:
            #     exit()
            # print(xov_extract)
            # exit()
            mla_proj_df = pd.merge(xov_extract, ladata_extract, left_on='genid', right_on='index')
            mla_proj_df['partid'] = 'none'
            # print("len mla_df_proj",len(mla_df_proj))

            if partials:
                # do stuff
                tmp_ladata_partials = tmp_ladata.loc[tmp_ladata.reset_index()['index'].isin(xov_extract.genid)][
                    ['seqid', 'LON', 'LAT', 'orbID'] + pars + etbcs]
                # print(tmp_ladata_partials.columns)
                # print(delta_pars)
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

                # tmp_proj_part = tmp_proj.drop(['ET_BC', 'LON', 'LAT'], axis=1)
                for idx, pder in enumerate(['_' + x + '_' + y for x in delta_pars.keys() for y in ['p', 'm']]):
                    if pder[-1] == 'p':
                        tmp = pd.concat([mla_proj_df.set_index('genid')[['LON_proj', 'LAT_proj']],
                                         newlon_p['LON' + pder], newlat_p['LAT' + pder],
                                         tmp_ladata_partials['ET_BC' + pder], tmp_ladata_partials['dR/' + pder[1:-2]]
                                         ], axis=1)
                    else:
                        tmp = pd.concat([mla_proj_df.set_index('genid')[['LON_proj', 'LAT_proj']],
                                         newlon_m['LON' + pder], newlat_m['LAT' + pder],
                                         tmp_ladata_partials['ET_BC' + pder], tmp_ladata_partials['dR/' + pder[1:-2]]],
                                        axis=1)
                    # tmp['partid'] = pder[1:]
                    tmp.reset_index(inplace=True)

                    tmp = tmp.values

                    # REMEMBER: tmp.columns = genid, LON_proj, LAT_proj, LON, LAT, ET_BC, dR/dp

                    # tmp.rename(
                    #     columns={'level_0': 'genid', 'LON' + pder: "LON", 'dR/' + pder[1:-2]: "R", 'LAT' + pder: "LAT",
                    #              'ET_BC' + pder: "ET_BC"}, inplace=True)
                    # print(part_proj_dict[idx])

                    part_proj_dict[idx].append(tmp)

            mla_proj_list.append(mla_proj_df)

    # concatenate lists
    mla_proj_df = pd.concat(mla_proj_list, sort=False).reset_index(drop=True)
    mla_proj_df.rename(columns={"seqid_x": "seqid_xov", "seqid_y": "seqid_mla"},
                       inplace=True)  # remember that seqid_mla is not continuous because of eventual bad obs
    # print(mla_proj_df)
    part_proj_dict = {x: np.vstack(y) for x, y in part_proj_dict.items()}
    part_proj_dict = dict(zip([x + '_' + y for x in delta_pars.keys() for y in ['p', 'm']], part_proj_dict.values()))

    part_proj_dict.update({'none': mla_proj_df[['genid', 'LON_proj', 'LAT_proj', 'LON', 'LAT']].values})

    # free-up memory
    mladata.clear()
    track = None

    # save main part
    # mla_proj_df.to_pickle(outdir + outdir_in + 'xov/tmp/proj/xov_' + str(cmb[0]) + '_' + str(
    #     cmb[1]) + '_mla_proj.pkl.gz')

    # print({x:y.shape for x,y in part_proj_dict.items()})

    end_prepro = time.time()

    print("Pre-processing finished after ", int(end_prepro - start_prepro), "sec or ",
          round((end_prepro - start_prepro) / 60., 2),
          " min!")

    return mla_proj_df, part_proj_dict