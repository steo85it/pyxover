import glob
import multiprocessing as mp
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from Amat import Amat
from ground_track import gtrack
from prOpt import outdir, vecopts, partials, parOrb, parGlo, parallel, local, debug, compute_input_xov
from project_coord import project_stereographic
# from util import get_size
from xov_setup import xov

# from memory_profiler import profile

# set number of processors to use

n_proc = mp.cpu_count() - 1
import_proj = False
import_abmat = (True,outdir+"sim/KX2_0/0res_1amp/Abmat*.pkl")


## MAIN ##
# @profile
def xov_prc_iters_run(outdir_in, xov_iter, cmb, input_xov):
    start = time.time()

    # create useful dirs
    if not os.path.exists(outdir + outdir_in + 'xov/'):
        os.mkdir(outdir + outdir_in + 'xov/')
    if not os.path.exists(outdir + outdir_in + 'xov/tmp/'):
        os.mkdir(outdir + outdir_in + 'xov/tmp/')
    if not os.path.exists(outdir + outdir_in + 'xov/tmp/proj'):
        os.mkdir(outdir + outdir_in + 'xov/tmp/proj')

    if xov_iter == 0:
        msrm_smpl = 10  # same as used for rough_xovs in PyXover (should be automatic)
    else:
        msrm_smpl = 4  # should be even...

    if msrm_smpl % 2 != 0:
        print("*** ERROR: msrm_smpl not an even number:", msrm_smpl)

    # compute projected mla_data around old xovs
    if not import_proj:

        useful_columns = ['LON', 'LAT', 'xOvID', 'orbA', 'orbB', 'mla_idA', 'mla_idB']

        # depending on available input xov, get xovers location from AbMat or from xov_rough
        if xov_iter > 0 or import_abmat[0]:  # len(input_xov)==0:
            # read old abmat file
            if xov_iter > 0:
                outdir_old = outdir_in.replace('_' + str(xov_iter) + '/', '_' + str(xov_iter - 1) + '/')
                abmat = outdir + outdir_old + 'Abmat*.pkl'
            else: # read a user defined abmat file
                abmat = import_abmat[1]

            # print(outdir_old, outdir_in)
            tmp_Amat = Amat(vecopts)
            # print(outdir + outdir_old + 'Abmat*.pkl')
            tmp = tmp_Amat.load(glob.glob(abmat)[0])
            old_xovs = tmp.xov.xovers[useful_columns]
        else:
            input_xov_path = outdir + outdir_in + 'xov/tmp/xovin_' + str(cmb[0]) + '_' + str(cmb[1]) + '.pkl.gz'
            if not compute_input_xov:
                input_xov = pd.read_pickle(input_xov_path)
                print("Input xovs read from", input_xov_path, ". Done!")
            else:
                # save to file (just in case...)
                input_xov.to_pickle(input_xov_path)
            old_xovs = input_xov[useful_columns]
            old_xovs = old_xovs.drop('xOvID', axis=1).rename_axis('xOvID').reset_index()

        # preprocessing of old xov and mla_data
        mla_proj_df, part_proj_dict = prepro_mla_xov(old_xovs, msrm_smpl, outdir_in, cmb)

        # projection of mla_data around old xovs
        mla_proj_df = project_mla(mla_proj_df, part_proj_dict, outdir_in, cmb)

    # or just retrieve them from file
    else:
        proj_pkl_path = outdir + outdir_in + 'xov/tmp/proj/xov_' + str(cmb[0]) + '_' + str(
            cmb[1]) + '_project.pkl.gz'
        mla_proj_df = pd.read_pickle(proj_pkl_path)
        print("mla_proj_df loaded from", proj_pkl_path, ". Done!!")

    # compute new xovs
    xov_tmp = compute_fine_xov(mla_proj_df, msrm_smpl, outdir_in, cmb)

    # Save to file
    if not os.path.exists(outdir + outdir_in + 'xov/'):
        os.mkdir(outdir + outdir_in + 'xov/')
    xov_tmp.save(outdir + outdir_in + 'xov/xov_' + str(cmb[0]) + '_' + str(
        cmb[1]) + '.pkl')  # one can split the df by trackA and save multiple pkl, one for each trackA if preferred

    end = time.time()

    print('Xov for ' + str(cmb) + ' processed and written to ' + outdir + outdir_in + 'xov/xov_' + str(
        cmb[0]) + '_' + str(
        cmb[1]) + '.pkl @' + time.strftime("%H:%M:%S", time.gmtime()))

    print("Fine xov determination finished after", int(end - start), "sec or ", round((end - start) / 60., 2), " min!")

    return xov_tmp


## SUBROUTINES ##
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
                ['seqid', 'LON', 'LAT', 'orbID', 'ET_BC', 'ET_TX', 'R', 'offnadir']].reset_index()
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


def project_mla(mla_proj_df, part_proj_dict, outdir_in, cmb):
    start_proj = time.time()
    # setting standard value (no chunks)
    # chunked_proj_dict = False
    if local:
        max_length_proj = 1.e5
    else:
        max_length_proj = 1.e7

    chunked_proj_dict = len(part_proj_dict['none']) > max_length_proj
    if chunked_proj_dict:
        # n_proc = mp.cpu_count()-1

        n_chunks = max(int(len(part_proj_dict['none']) // max_length_proj), n_proc)
        chunksize = int(len(part_proj_dict['none']) / n_chunks)

        print("Splitting a large array of", len(part_proj_dict['none']), "mla_data in", n_chunks, "chunks of",
              chunksize, "rows")

        proc_chunks = {}
        for id, tmp_proj in part_proj_dict.items():
            for i_proc in range(n_chunks):
                chunkstart = i_proc * chunksize
                # make sure to include the division remainder for the last process
                chunkend = (i_proc + 1) * chunksize if i_proc < n_chunks - 1 else None
                # print(mla_proj_df.columns)

                # proc_chunks.append(mla_proj_df.iloc[slice(chunkstart, chunkend)])
                proc_chunks[id + '_' + str(i_proc)] = tmp_proj[chunkstart: chunkend, :]

        # important check but need to solve the vstack issue (none has 7 columns, partials have 5, or viceversa...)
        # assert len(np.vstack(proc_chunks.values()))  == len(np.vstack(part_proj_dict.values()[:,0]))  # make sure all data is in the chunks
    else:
        proc_chunks = part_proj_dict
    if parallel:
        # n_proc = mp.cpu_count()-5

        # distribute work to the worker processes
        with mp.get_context("spawn").Pool(processes=n_proc) as pool:
            # starts the sub-processes without blocking
            # pass the chunk to each worker process
            proc_results = [pool.apply_async(project_chunk, args=(chunk,)) for chunk in proc_chunks.values()]
            pool.close()
            pool.join()
            # blocks until all results are fetched
            # proc_chunk = np.vstack([proc_chunk[:, 0], np.asarray(chunk_res)]).T

            result_chunks = {x: np.hstack([y, proc_results[idx].get()[:, 1:]]) for idx, (x, y) in
                             enumerate(proc_chunks.items())}
            # result_chunks = dict(zip(part_proj_dict.keys(),[r.get() for r in proc_results]))
            # print(result_chunks)
        # exit()
        # genid, LON_proj, LAT_proj, LON, LAT, ET_BC, dR / dp

        # concatenate results from worker processes and add to df
        # mla_proj_df = mla_proj_df.merge(tmp_proj,on='genid')
    else:
        proc_results = []
        # for idx,chunk in enumerate(part_proj_dict.values()):
        #     proc_results.append(project_stereographic(chunk[:,3], chunk[:,4], chunk[:,1], chunk[:,2],
        #                                         R=vecopts['PLANETRADIUS']))  # (lon, lat, lon0, lat0, R=1)
        # result_chunks = {x:np.hstack([y,np.vstack(proc_results[idx]).T]) for idx, (x, y) in enumerate(part_proj_dict.items())}
        # just compatible with chunks
        for chunk in proc_chunks.values():
            proc_results.append(project_chunk(chunk))
        # blocks until all results are fetched
        # proc_chunk = np.vstack([proc_chunk[:, 0], np.asarray(chunk_res)]).T

        result_chunks = {x: np.hstack([y, proc_results[idx][:, 1:]]) for idx, (x, y) in
                         enumerate(proc_chunks.items())}
    # free memory
    proc_chunks.clear()
    part_proj_dict.clear()
    delta_pars, etbcs, pars = get_ds_attrib()
    # operate on chunks #
    if chunked_proj_dict:
        projchunks = []
        chunk_proj_df_paths = []
        for chunkid in range(n_chunks):
            tmp_chunks = {k: v for (k, v) in result_chunks.items() if str(chunkid) == k.split('_')[-1]}
            # print(tmp_chunks)
            none_proj_df = pd.DataFrame(tmp_chunks['none_' + str(chunkid)][:, -2:], columns=['X_stgprj', 'Y_stgprj'])
            # print(mla_proj_df)

            if partials:
                partials_proj_df = []
                for pder in ['_' + x + '_' + y for x in delta_pars.keys() for y in ['p', 'm']]:
                    tmp = pd.DataFrame(tmp_chunks[pder[1:] + '_' + str(chunkid)][:, -4:],
                                       # columns=['genid','LON_proj','LAT_proj','LON','LAT','ET_BC'+pder,'dR/' + pder[1:-2],'X_stgprj' + pder,'Y_stgprj' + pder])
                                       columns=['ET_BC' + pder, 'dR/' + pder[1:-2], 'X_stgprj' + pder,
                                                'Y_stgprj' + pder])
                    # partials dR/dp are the same for +-, so just save one
                    if pder[-1] == 'm':
                        tmp.drop('dR/' + pder[1:-2], axis='columns', inplace=True)

                    partials_proj_df.append(tmp)

                # print(partials_proj_df)
                ind_mla_proj = [(chunksize) * (chunkid), (chunksize) * (chunkid) + len(none_proj_df)]
                chunk_proj_df = pd.concat([mla_proj_df.iloc[ind_mla_proj[0]:ind_mla_proj[1]].reset_index(drop=True),
                                           none_proj_df] + partials_proj_df, axis=1)
                projchunks.append(chunk_proj_df)

                # save intermediate file
                # chunk_proj_df_path = outdir + outdir_in + 'xov/tmp/proj/xov_' + str(cmb[0]) + '_' + str(
                #     cmb[1]) + '_project_' + str(chunkid) + '.pkl.gz'
                # chunk_proj_df_paths.append(chunk_proj_df_path)
                # chunk_proj_df.to_pickle(chunk_proj_df_path)
                # print("Intermediate file saved to:", outdir + outdir_in + 'xov/tmp/proj/xov_' + str(cmb[0]) + '_' + str(
                #     cmb[1]) + '_project_' + str(chunkid) + '.pkl.gz')

                if chunkid == 0:
                    print("chunk memory proj:", projchunks[chunkid].memory_usage(deep=True).sum() * 1.e-6)
            else:
                projchunks.append(pd.concat([mla_proj_df.iloc[ind_mla_proj[0]:ind_mla_proj[1]].reset_index(drop=True),
                                             none_proj_df], axis=1))

        mla_proj_df = pd.concat(projchunks, axis=0, sort=False).reset_index(drop=True)
        print("total memory proj:", mla_proj_df.memory_usage(deep=True).sum() * 1.e-6)

    else:
        mla_proj_df = pd.concat(
            [mla_proj_df, pd.DataFrame(result_chunks['none'][:, -2:], columns=['X_stgprj', 'Y_stgprj'])], axis=1)
        #
        # split and re-concatenate rows related to partial derivatives
        if partials:
            partials_df_list = []
            for pder in ['_' + x + '_' + y for x in delta_pars.keys() for y in ['p', 'm']]:
                partials_proj_df = pd.DataFrame(result_chunks[pder[1:]],
                                                columns=['genid', 'LON_proj', 'LAT_proj', 'LON', 'LAT', 'ET_BC' + pder,
                                                         'dR/' + pder[1:-2], 'X_stgprj' + pder, 'Y_stgprj' + pder])
                # columns = ['ET_BC' + pder, 'dR/' + pder[1:-2],'X_stgprj' + pder, 'Y_stgprj' + pder])

                # partials dR/dp are the same for +-, so just save one
                if pder[-1] == 'p':
                    partials_df_list.append(partials_proj_df[['X_stgprj' + pder, 'Y_stgprj' + pder, 'ET_BC' + pder,
                                                              'dR/' + pder[
                                                                      1:-2]]].reset_index(
                        drop=True))  # could include genid and xovid from partials to check
                else:
                    partials_df_list.append(
                        partials_proj_df[['X_stgprj' + pder, 'Y_stgprj' + pder, 'ET_BC' + pder]].reset_index(drop=True))

            mla_proj_df = pd.concat([mla_proj_df.reset_index(drop=True)] + partials_df_list, axis=1, sort=False)
    # Save intermediate result
    proj_pkl_path = outdir + outdir_in + 'xov/tmp/proj/xov_' + str(cmb[0]) + '_' + str(
        cmb[1]) + '_project.pkl.gz'
    mla_proj_df.to_pickle(proj_pkl_path)
    print("Projected df saved to:", outdir + outdir_in + 'xov/tmp/proj/xov_' + str(cmb[0]) + '_' + str(
        cmb[1]) + '_project.pkl.gz')
    print("Len mla_proj_df after reordering:", len(mla_proj_df))
    ################################################
    end_proj = time.time()
    print("Projection finished after", int(end_proj - start_proj), "sec or ", round((end_proj - start_proj) / 60., 2),
          " min!")
    ################################################

    return mla_proj_df


def compute_fine_xov(mla_proj_df, msrm_smpl, outdir_in, cmb):
    start_finexov = time.time()
    # initialize xov object
    xovs_list = mla_proj_df.xovid.unique()
    xov_tmp = xov(vecopts)
    # pass measurements sampling to fine_xov computation through xov_tmp
    xov_tmp.msrm_sampl = msrm_smpl

    # store the imposed perturbation (if closed loop simulation) - get from any track uploaded in prepro step
    track = gtrack(vecopts)
    track = track.load(glob.glob(outdir + outdir_in + 'gtrack_' + str(cmb[0]) + '/gtrack_' + str(cmb[0]) + '*.pkl')[0])
    xov_tmp.pert_cloop = {'0': track.pert_cloop}
    xov_tmp.pert_cloop_0 = {'0': track.pert_cloop_0}
    # store the solution from the previous iteration
    xov_tmp.sol_prev_iter = {'0': track.sol_prev_iter}
    # if huge_dataset:
    #     for chunk_path in chunk_proj_df_paths:
    #         mla_proj_df = pd.read_pickle(chunk_path)
    #         print(mla_proj_df)
    #         exit()
    # select what to keep
    cols_to_keep = mla_proj_df.loc[:,
                   mla_proj_df.columns.str.startswith('ET_BC') + mla_proj_df.columns.str.startswith('X_stgprj') +
                   mla_proj_df.columns.str.startswith('Y_stgprj') + mla_proj_df.columns.str.startswith(
                       'dR/')].columns  # ,'X_stgprj','Y_stgprj'))
    proj_df_tmp = mla_proj_df[
        ['orbID', 'seqid_mla', 'ET_TX', 'LON', 'LAT', 'R', 'offnadir', 'xovid', 'LON_proj', 'LAT_proj'] + list(
            cols_to_keep)]
    # print(proj_df_tmp.columns)
    print("total memory proj_df_tmp:", proj_df_tmp.memory_usage(deep=True).sum() * 1.e-6)

    if parallel:
        # pool = mp.Pool(processes=ncores)  # mp.cpu_count())
        # xov_list = pool.map(fine_xov_proc, args)  # parallel
        # apply_async example

        if local:
            from tqdm import tqdm
            pbar = tqdm(total=len(xovs_list))

            def update(*a):
                pbar.update()

        xov_list = []
        with mp.get_context("spawn").Pool(processes=n_proc) as pool:
            # for idx in range(len(xovs_list)):
            # xov_list.append(pool.apply_async(fine_xov_proc, args=(xovs_list[idx], dfl[idx], xov_tmp), callback=update))
            for xovi in xovs_list:
                if local:
                    xov_list.append(pool.apply_async(fine_xov_proc, args=(
                        xovi, proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi], xov_tmp), callback=update))
                else:
                    xov_list.append(pool.apply_async(fine_xov_proc, args=(
                        xovi, proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi], xov_tmp)))

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
        # exit()

        # launch once in serial mode to get ancillary values
        fine_xov_proc(0, proj_df_tmp.loc[proj_df_tmp['xovid'] == 0], xov_tmp)
        # assign xovers to the new xov_tmp containing ancillary values
        xov_tmp.xovers = pd.concat(xov_list, axis=0)

    else:
        if local:
            from tqdm import tqdm
            for xovi in tqdm(xovs_list):
                fine_xov_proc(xovi, proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi], xov_tmp)
        else:
            for xovi in xovs_list:
                fine_xov_proc(xovi, proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi], xov_tmp)

            # print(idx)

    # fill xov structure with info for LS solution
    xov_tmp.parOrb_xy = [x for x in xov_tmp.xovers.filter(regex='^dR/[a-zA-Z0-9]+_.*$').columns]  # update partials list
    xov_tmp.parGlo_xy = [(a + b) for a in ['dR/'] for b in list(parGlo.keys())]
    xov_tmp.par_xy = xov_tmp.parOrb_xy + xov_tmp.parGlo_xy  # update partials list
    if debug:
        print("Parameters:", xov_tmp.parOrb_xy, xov_tmp.parGlo_xy, xov_tmp.par_xy)
    # update xovers table with LAT and LON
    xov_tmp = get_xov_latlon(xov_tmp, mla_proj_df.loc[mla_proj_df.partid == 'none'])
    xov_tmp.xovers.drop('xovid', axis=1).reset_index(inplace=True, drop=True)
    # print(xov_tmp.xovers.columns)
    if debug:
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.max_rows', 500)

        print(xov_tmp.xovers)  # .loc[xov_tmp.xovers['orbA']=='1504030011'])
    end_finexov = time.time()
    print("Fine_xov finished after", int(end_finexov - start_finexov), "sec or ",
          round((end_finexov - start_finexov) / 60., 2), " min and located", len(xov_tmp.xovers), "out of previous",
          len(xovs_list), "xovers!")
    return xov_tmp


## SUBSUB ##
# @profile
def fine_xov_proc(xovi, df, xov_tmp):  # args):

    df = df.reset_index(
        drop=True).reset_index()
    df.rename(columns={'index': 'genID', 'seqid_mla': 'seqid'}, inplace=True)

    if len(df) > 0:  # and {'LON_proj','LAT_proj'}.issubset(df.columns):
        xov_tmp.proj_center = {'lon': df['LON_proj'].values[0], 'lat': df['LAT_proj'].values[0]}
        df.drop(columns=['LON_proj', 'LAT_proj'], inplace=True)
    else:
        if debug:
            print("### Empty proj_df_xovi on xov #", xovi)
            print(df)
        return  # continue

    # populate xov_tmp with local data for specific xov
    xov_tmp.ladata_df = df.copy()
    xov_tmp.tracks = dict(zip(df.orbID.unique(), list(range(2))))
    xov_tmp.ladata_df['orbID'] = xov_tmp.ladata_df['orbID'].map(xov_tmp.tracks)

    msrm_smpl = xov_tmp.msrm_sampl

    try:
        x, y, subldA, subldB, ldA, ldB = xov_tmp.get_xover_fine([msrm_smpl], [msrm_smpl], '')
    except:
        if debug:
            print("### get_xover_fine issue on xov #", xovi)
            print(xov_tmp.ladata_df)
        return  # continue
    # print(x, y, subldA, subldB, ldA, ldB)

    ldA, ldB, R_A, R_B = xov_tmp.get_elev('', subldA, subldB, ldA, ldB, x=x, y=y)
    # print(ldA, ldB, R_A, R_B)
    out = np.vstack((x, y, ldA, ldB, R_A, R_B)).T
    # print(df)
    # print(out)
    if len(out) == 0:
        if debug:
            print("### Empty out on xov #", xovi)
        return  # continue

    # post-processing
    xovtmp = xov_tmp.postpro_xov_elev(xov_tmp.ladata_df, out)
    xovtmp = pd.DataFrame(xovtmp, index=[0])  # TODO possibly avoid, very time consuming

    if len(xovtmp) > 1:
        if debug:
            print("### Bad multi-xov at xov#", xovi)
            print(xovtmp)
        return  # continue

    # Update xovtmp as attribute for partials
    xov_tmp.xovtmp = xovtmp

    # Compute and store distances between obs and xov coord
    xov_tmp.set_xov_obs_dist()
    # Compute and store offnadir state for obs around xov
    xov_tmp.set_xov_offnadir()

    # process partials from gtrack to xov, if required
    if partials:
        xov_tmp.set_partials()
    else:
        # retrieve epoch to, e.g., trace tracks quality (else done inside set_partials)
        xov_tmp.xovtmp = pd.concat([xov_tmp.xovtmp, pd.DataFrame(
            np.reshape(xov_tmp.get_dt(xov_tmp.ladata_df, xov_tmp.xovtmp), (len(xov_tmp.xovtmp), 2)),
            columns=['dtA', 'dtB'])], axis=1)

    # Remap track names to df
    xov_tmp.xovtmp['orbA'] = xov_tmp.xovtmp['orbA'].map({v: k for k, v in xov_tmp.tracks.items()})
    xov_tmp.xovtmp['orbB'] = xov_tmp.xovtmp['orbB'].map({v: k for k, v in xov_tmp.tracks.items()})

    xov_tmp.xovtmp['xOvID'] = xovi

    # Update general df (serial only, does not work in parallel since not a shared object)
    if not parallel:
        xov_tmp.xovers = xov_tmp.xovers.append(xov_tmp.xovtmp, sort=True)

    # import sys
    # def sizeof_fmt(num, suffix='B'):
    #     ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    #     for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
    #         if abs(num) < 1024.0:
    #             return "%3.1f %s%s" % (num, unit, suffix)
    #         num /= 1024.0
    #     return "%.1f %s%s" % (num, 'Yi', suffix)
    #
    # print("iter over xov:",xovi)
    # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
    #                          key=lambda x: -x[1])[:10]:
    #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

    # exit()

    return xov_tmp.xovtmp


## UTILITIES ##
def get_xov_latlon(xov, df):
    """
    Retrieve LAT/LON of xover from geolocalisation table and add to xovers table
    (useful to analyze/plot xovers on map). Updates input xov.xovers.

    :param xov: contains xovers table to merge to
    :param df: mla data table with lat and lon to extract
    """
    # pd.set_option('display.max_columns', 500)

    # only get rows corresponding to former xov location
    df = df.loc[df.seqid_xov == df.seqid_mla][['xovid', 'LON', 'LAT']]
    # will get 2 rows, one for each track in the xover: drop one
    df = df.drop_duplicates(subset=['xovid'], keep='first')
    # assign LAT and LON to their respective xover by id and update
    xov.xovers = pd.merge(xov.xovers, df, left_on='xOvID', right_on='xovid')
    # xov.xovers.drop('xovid',inplace=True)

    return xov


def project_chunk(proc_chunk):
    # chunk_res = project_stereographic(proc_chunk['LON'],
    #                                   proc_chunk['LAT'],
    #                                   proc_chunk['LON_proj'],
    #                                   proc_chunk['LAT_proj'],
    #                                   R=vecopts['PLANETRADIUS'])
    chunk_res = project_stereographic(proc_chunk[:, 3],
                                      proc_chunk[:, 4],
                                      proc_chunk[:, 1],
                                      proc_chunk[:, 2],
                                      R=vecopts['PLANETRADIUS'])
    # print(pd.DataFrame(chunk_res).T)
    # proc_chunk[['x','y']] = pd.DataFrame(chunk_res).T

    # proc_chunk['x'], proc_chunk['y'] = zip(*proc_chunk.apply(
    #     lambda x: project_stereographic(x['LON'], x['LAT'], x['LON_proj'], x['LAT_proj'],
    #                                     R=vecopts['PLANETRADIUS']),
    #     axis=1))

    # chunk_res.index = proc_chunk.index
    # print(proc_chunk[:,-1])
    # print(np.asarray(chunk_res))
    proc_chunk = np.vstack([proc_chunk[:, 0], np.asarray(chunk_res)]).T

    return proc_chunk


def get_ds_attrib():
    # prepare pars keys, lists and dicts
    pars = ['d' + x + '/' + y for x in ['LON', 'LAT', 'R'] for y in
            list(parOrb.keys()) + list(parGlo.keys())]
    delta_pars = {**parOrb, **parGlo}
    etbcs = ['ET_BC_' + x + y for x in list(parOrb.keys()) + list(parGlo.keys()) for y in ['_p', '_m']]
    return delta_pars, etbcs, pars


## MAIN ##
if __name__ == '__main__':
    start = time.time()

    cmb = [13, 14]

    xov_tmp = xov_prc_iters_run()

    end = time.time()

    print("Process finished after", int(end - start), "sec or ", round((end - start) / 60., 2), " min!")
