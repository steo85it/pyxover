import glob
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
from pygeoloc.ground_track import gtrack
from pyxover.get_xov_latlon import get_xov_latlon
from pyxover.xov_setup import xov

# from examples.MLA.options import XovOpt.get("vecopts"), XovOpt.get("outdir"), XovOpt.get("parallel"), XovOpt.get("local"), XovOpt.get("parGlo"), XovOpt.get("debug"), XovOpt.get("partials"), XovOpt.get("n_proc")
from config import XovOpt


def compute_fine_xov(mla_proj_df, msrm_smpl, outdir_in, cmb):
    start_finexov = time.time()
    # initialize xov object
    xovs_list = mla_proj_df.xovid.unique()
    xov_tmp = xov(XovOpt.get("vecopts"))
    # pass measurements sampling to fine_xov computation through xov_tmp
    xov_tmp.msrm_sampl = msrm_smpl

    # store the imposed perturbation (if closed loop simulation) - get from any track uploaded in prepro step
    track = gtrack(XovOpt.to_dict())
    # TODO removed check on orbid for this test
    # print(XovOpt.get("outdir") + outdir_in + 'gtrack_' + str(cmb[0][:2]) + '/gtrack_' + str(cmb[0]) + '*.pkl')
    if XovOpt.get("weekly_sets"):
      track = track.load(glob.glob(XovOpt.get("outdir") + outdir_in + 'gtrack_' + str(cmb[0]) + '/gtrack_' + str(cmb[0]) + '*.pkl')[0])
    else:
      track = track.load(glob.glob(XovOpt.get("outdir") + outdir_in + 'gtrack_' + str(cmb[0][:2]) + '/gtrack_' + str(cmb[0]) + '*.pkl')[0])
    # print(XovOpt.get("outdir") + outdir_in + 'gtrack' + '/gtrack_' + str(cmb[0]) + '*.pkl')
    # track = track.load(glob.glob(XovOpt.get("outdir") + outdir_in + 'gtrack' + '/gtrack_' + str(cmb[0]) + '*.pkl')[0])

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
        ['orbID', 'seqid_mla', 'ET_TX', 'LON', 'LAT', 'R', 'dt', 'offnadir', 'xovid', 'LON_proj', 'LAT_proj'] + list(
            cols_to_keep)]
    # print(proj_df_tmp.columns)
    print("total memory proj_df_tmp:", proj_df_tmp.memory_usage(deep=True).sum() * 1.e-6)

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
            # xov_list.append(pool.apply_async(fine_xov_proc, args=(xovs_list[idx], dfl[idx], xov_tmp), callback=update))
            for xovi in xovs_list:

                if XovOpt.get("local"):
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

        # launch once in serial mode to get ancillary values
        fine_xov_proc(0, proj_df_tmp.loc[proj_df_tmp['xovid'] == 0], xov_tmp)
        # assign xovers to the new xov_tmp containing ancillary values
        xov_tmp.xovers = pd.concat(xov_list, axis=0)

    else:
        if XovOpt.get("local"):
            from tqdm import tqdm
            for xovi in tqdm(xovs_list):
                fine_xov_proc(xovi, proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi], xov_tmp)
        else:
            for xovi in xovs_list:
                fine_xov_proc(xovi, proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi], xov_tmp)

            # print(idx)

    # fill xov structure with info for LS solution
    xov_tmp.parOrb_xy = [x for x in xov_tmp.xovers.filter(regex='^dR/[a-zA-Z0-9]+_.*$').columns]  # update partials list
    xov_tmp.parGlo_xy = [(a + b) for a in ['dR/'] for b in list(XovOpt.get("parGlo").keys())]
    xov_tmp.par_xy = xov_tmp.parOrb_xy + xov_tmp.parGlo_xy  # update partials list
    if XovOpt.get("debug"):
        print("Parameters:", xov_tmp.parOrb_xy, xov_tmp.parGlo_xy, xov_tmp.par_xy)
    # update xovers table with LAT and LON
    xov_tmp = get_xov_latlon(xov_tmp, mla_proj_df.loc[mla_proj_df.partid == 'none'])
    xov_tmp.xovers.drop('xovid', axis=1).reset_index(inplace=True, drop=True)
    # print(xov_tmp.xovers.columns)
    if XovOpt.get("debug"):
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.max_rows', 500)

        print(xov_tmp.xovers)  # .loc[xov_tmp.xovers['orbA']=='1504030011'])
    end_finexov = time.time()
    print("Fine_xov for", str(cmb) ,"finished after", int(end_finexov - start_finexov), "sec or ",
          round((end_finexov - start_finexov) / 60., 2), " min and located", len(xov_tmp.xovers), "out of previous",
          len(xovs_list), "xovers!")
    return xov_tmp


def fine_xov_proc(xovi, df, xov_tmp):  # args):

    df = df.reset_index(
        drop=True).reset_index()
    df.rename(columns={'index': 'genID', 'seqid_mla': 'seqid'}, inplace=True)

    if len(df) > 0:  # and {'LON_proj','LAT_proj'}.issubset(df.columns):
        xov_tmp.proj_center = {'lon': df['LON_proj'].values[0], 'lat': df['LAT_proj'].values[0]}
        df.drop(columns=['LON_proj', 'LAT_proj'], inplace=True)
    else:
        if XovOpt.get("debug"):
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
        if XovOpt.get("debug"):
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
        if XovOpt.get("debug"):
            print("### Empty out on xov #", xovi)
        return  # continue

    # post-processing
    xovtmp = xov_tmp.postpro_xov_elev(xov_tmp.ladata_df, out)
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
    xov_tmp.xovtmp = xovtmp

    # Compute and store distances between obs and xov coord
    #xov_tmp.set_xov_obs_dist()
    # Compute and store offnadir state for obs around xov
    xov_tmp.set_xov_offnadir()

    # process partials from gtrack to xov, if required
    if XovOpt.get("partials"):
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
    if not XovOpt.get("parallel"):
        # xov_tmp.xovers = xov_tmp.xovers.append(xov_tmp.xovtmp, sort=True)
        xov_tmp.xovers = pd.concat([xov_tmp.xovers, xov_tmp.xovtmp], sort=True)

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
