import multiprocessing as mp
import time

import numpy as np
import pandas as pd
from pyxover.xov_utils import get_ds_attrib
from xovutil.project_coord import project_stereographic

# from examples.MLA.options import XovOpt.get("local"), XovOpt.get("parallel"), XovOpt.get("partials"), XovOpt.get("outdir"), XovOpt.get("vecopts"), XovOpt.get("n_proc")
from config import XovOpt


def project_mla(mla_proj_df, part_proj_dict, outdir_in, cmb):
    start_proj = time.time()
    # setting standard value (no chunks)
    # chunked_proj_dict = False
    if XovOpt.get("local"):
        max_length_proj = 1.e5
    else:
        max_length_proj = 1.e7

    chunked_proj_dict = len(part_proj_dict['none']) > max_length_proj
    if chunked_proj_dict:
        # n_proc = mp.cpu_count()-1

        n_chunks = max(int(len(part_proj_dict['none']) // max_length_proj), XovOpt.get("n_proc"))
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
    if XovOpt.get("parallel"):

        # distribute work to the worker processes
        with mp.get_context("spawn").Pool(processes=XovOpt.get("n_proc")) as pool:
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

            if XovOpt.get("partials"):
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
        if XovOpt.get("partials"):
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
    proj_pkl_path = XovOpt.get("outdir") + outdir_in + 'xov/tmp/proj/xov_' + str(cmb[0]) + '_' + str(
        cmb[1]) + '_project.pkl.gz'
    mla_proj_df.to_pickle(proj_pkl_path)
    print("Projected df saved to:", XovOpt.get("outdir") + outdir_in + 'xov/tmp/proj/xov_' + str(cmb[0]) + '_' + str(
        cmb[1]) + '_project.pkl.gz')
    print("Len mla_proj_df after reordering:", len(mla_proj_df))
    ################################################
    end_proj = time.time()
    print("Projection finished after", int(end_proj - start_proj), "sec or ", round((end_proj - start_proj) / 60., 2),
          " min!")
    ################################################

    return mla_proj_df


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
                                      R=XovOpt.get("vecopts")['PLANETRADIUS'])
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