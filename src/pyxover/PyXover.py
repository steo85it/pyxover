#!/usr/bin/env python3
# ----------------------------------
# PyXover
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-Oct-2018
#
import warnings

from src.accumxov.Amat import Amat
from src.pyxover.xov_prc_iters import xov_prc_iters_run

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import glob

import numpy as np
import pandas as pd
import itertools as itert
# from itertools import izip, count
import multiprocessing as mp
# from geopy.distance import vincenty

import spiceypy as spice

# from collections import defaultdict
# import mpl_toolkits.basemap as basemap

import time

# mylib
# from examples.MLA.options import XovOpt.get("new_xov"), XovOpt.get("vecopts"), XovOpt.get("outdir"), XovOpt.get("debug"), XovOpt.get("monthly_sets"), XovOpt.get("new_algo"), XovOpt.get("compute_input_xov"), XovOpt.get("basedir")
# from mapcount import mapcount
from config import XovOpt

from src.pygeoloc.ground_track import gtrack
from src.pyxover.xov_setup import xov


# from xovutil import lflatten
########################################
# # test space
#
# tst = [np.array([ 89.72151033, 103.94256763]), np.array([139.94256763])]
# tst = lflatten(tst)
# print(tst)
# exit()
# vecopts = {'SCID':'-236'}
#
# tmp_ser = Amat(vecopts)
# for f in glob.glob('/home/sberton2/Works/NASA/Mercury_tides/out/xov_130101*_13.pkl'):
#     tmp_ser = tmp_ser.load(f)
#
#     print(tmp_ser.xovers)
#
# # tmp_par = Amat(vecopts)
# # tmp_par = tmp_par.load('out/small_par/Amat_small_dataset.pkl')
#
# # print(tmp_par.spA.to_dense().equals(tmp_ser.spA.to_dense()))
#
# # print(tmp_par.spA)
# # print(tmp_ser.spA)
#
# exit()

# @profile
def launch_xov(
        args):  # pool.map functions have to stay on top level (not inside other functions) to avoid the "cannot be pickled" error
    track_idA = args[0]
    # print(track_id)
    comb = args[1]
    misycmb = args[2]
    par = args[3]
    if XovOpt.get("instrument") == 'BELA':
        mladata_pkl_fn = args[4]
        with open(mladata_pkl_fn, 'rb') as handle:
            #start_unpickle = time.time()
            import pickle
            mladata = pickle.load(handle)
            #end_unpickle = time.time()
            #print("mladata unpickled after",end_unpickle-start_unpickle,"sec")
    else:
        mladata = args[4]
    outdir = args[5]

    if XovOpt.get("new_xov"):  # and track_id=='1301232350':
        # print( "check", track_id, misycmb[par])
        if not os.path.isfile(outdir + 'xov/xov_' + track_idA + '_' + misycmb[par][1] + '.pkl') or XovOpt.get("new_xov") == 2:

            # print("Processing " + track_id + " ...")

            # try:
            #    trackA = track_id
            #    trackA = tracklist[str(track_id)]
            trackA = gtrack(XovOpt.get("vecopts"))

            if XovOpt.get("monthly_sets"):
                trackA = trackA.load(outdir + 'gtrack_' + misycmb[par][0][:2] + '/gtrack_' + track_idA + '.pkl')
            else:
                # trackA = trackA.load(outdir + 'gtrack_' + misycmb[par][0] + '/gtrack_' + track_id + '.pkl')
                trackA.ladata_df = mladata[track_idA]   # faster and less I/O which overloads PGDA

            if not trackA == None and len(trackA.ladata_df) > 0:

                xov_tmp = track_idA
                xov_tmp = xov(XovOpt.get("vecopts"))

                # print(comb)

                xovers_list = []
                xover_found = False
                # loop over all combinations containing track_id
                for track_idA, track_idB in [s for s in comb if track_idA in s[0]]:

                    # if debug:
                    #    print("Processing " + gtrackA + " vs " + gtrackB)

                    if track_idB > track_idA:
                        # try:
                        #        trackB = track_id
                        #        trackB = tracklist[str(gtrackB)]
                        trackB = gtrack(XovOpt.get("vecopts"))

                        if XovOpt.get("monthly_sets"):
                           trackB = trackB.load(outdir + 'gtrack_' + misycmb[par][1][:2] + '/gtrack_' + track_idB + '.pkl')
                        else:
                           # trackB = trackB.load(outdir + 'gtrack_' + misycmb[par][1] + '/gtrack_' + gtrackB + '.pkl')
                           trackB.ladata_df = mladata[track_idB]  # faster and less I/O which overloads PGDA

                        if not trackB == None and len(trackB.ladata_df) > 0:

                            # # TODO remove when recomputing
                            # trackA.ladata_df[['X_NPstgprj', 'Y_NPstgprj']] = trackA.ladata_df[['X_stgprj', 'Y_stgprj']]
                            # trackB.ladata_df[['X_NPstgprj', 'Y_NPstgprj']] = trackB.ladata_df[['X_stgprj', 'Y_stgprj']]
                            # trackA.ladata_df[] = trackA.ladata_df.rename(index=str, columns={"X_stgprj": "X_NPstgprj", "Y_stgprj": "Y_NPstgprj"})
                            # trackB.ladata_df = trackB.ladata_df.rename(index=str, columns={"X_stgprj": "X_NPstgprj", "Y_stgprj": "Y_NPstgprj"})

                            # looping over all track combinations and updating the general df xov_tmp.xovers
                            xover_found = xov_tmp.setup([trackA,trackB])

                    if XovOpt.get("new_algo") and xover_found:
                        xovers_list.append(xov_tmp.xovtmp)
                    # except:
                    #     print(
                    #         'failed to load trackB ' + outdir + 'gtrack_' + gtrackB + '.pkl' + ' to process ' + outdir + 'gtrack_' + track_id + '.pkl')

                if XovOpt.get("new_algo"):
                    xov_tmp.xovers = pd.DataFrame(xovers_list)
                    xov_tmp.xovers.reset_index(drop=True, inplace=True)
                    xov_tmp.xovers['xOvID'] = xov_tmp.xovers.index

                # for each gtrackA, write
                # print([s for s in comb if track_id in s[0]])
                if [s for s in comb if track_idA in s[0]] and len(xov_tmp.xovers) > 0:
                    # get xover LAT and LON
                    xov_tmp.get_xov_latlon(trackA)

                    # Save to file
                    if not os.path.exists(outdir + 'xov/'):
                        os.mkdir(outdir + 'xov/')
                    if XovOpt.get("new_algo"):
                        # Save to temporary folder
                        # if not os.path.exists(outdir + 'xov/tmp/'):
                        #     os.mkdir(outdir + 'xov/tmp/')
                        # xov_tmp.save(outdir + 'xov/tmp/xov_' + track_idA + '_' + misycmb[par][1] + '.pkl')

                        # just pass rough_xovs to next step
                        return xov_tmp.xovers
                    else:
                        xov_tmp.save(outdir + 'xov/xov_' + track_idA + '_' + misycmb[par][1] + '.pkl')
                        print(
                            'Xov for ' + track_idA + ' processed and written to ' + outdir + 'xov/xov_' + track_idA + '_' +
                            misycmb[par][1] + '.pkl @' + time.strftime("%H:%M:%S", time.gmtime()))
                        return track_idA

        # except:
        #     print(
        #         'failed to load trackA ' + outdir + 'gtrack_' + track_id + '.pkl' + ' to process xov from ' + outdir + 'gtrack_' + track_id + '.pkl')

        else:

            #      track = track.load('out/xov_'+gtrackA+'.pkl')
            print('Xov for ' + track_idA + ' already exists in ' + outdir + 'xov_' + track_idA + '_' +
                          misycmb[par][1] + '.pkl @' + time.strftime("%H:%M:%S", time.gmtime()))

    ########################################
# @profile
def main(args):
    # from examples.MLA.options import XovOpt.get("parallel"), XovOpt.get("outdir"), XovOpt.get("auxdir"), XovOpt.get("local"), XovOpt.get("vecopts")
    from config import XovOpt

    print(args)

    # read input args
    print('Number of arguments:', len(args), 'arguments.')
    print('Argument List:', str(args))

    cmb_y_in = args[0]
    indir_in = args[1]
    outdir_in = args[2]
    iter_in = args[-1]

    # locate data
    data_pth = XovOpt.get("basedir") # '/att/nobackup/sberton2/MLA/data/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
    dataset = indir_in  # 'test/' #'small_test/' #'1301/' #
    data_pth += dataset
    # # load kernels
    # spice.furnsh(auxdir + 'mymeta')  # 'aux/mymeta')

    # set ncores
    ncores = mp.cpu_count() - 1  # 8

    if XovOpt.get("parallel"):
        print('Process launched on ' + str(ncores) + ' CPUs')

    ##############################################

    # Setup some useful options
    # vecopts = {'SCID': '-236',
    #            'SCNAME': 'MESSENGER',
    #            'SCFRAME': -236000,
    #            'INSTID': (-236500, -236501),
    #            'INSTNAME': ('MSGR_MLA', 'MSGR_MLA_RECEIVER'),
    #            'PLANETID': '199',
    #            'PLANETNAME': 'MERCURY',
    #            'PLANETRADIUS': 2440.,
    #            'PLANETFRAME': 'IAU_MERCURY',
    #            'OUTPUTTYPE': 1,
    #            'ALTIM_BORESIGHT': '',
    #            'INERTIALFRAME': 'J2000',
    #            'INERTIALCENTER': 'SSB',
    #            'PARTDER': ''}

    # out = spice.getfov(vecopts['INSTID'][0], 1)
    # updated w.r.t. SPICE from Mike's scicdr2mat.m
    if XovOpt.get("instrument") == 'BELA':
        XovOpt.get("vecopts")['ALTIM_BORESIGHT'] = [0.,0.,1.]
    else:
        XovOpt.get("vecopts")['ALTIM_BORESIGHT'] = [0.0022105, 0.0029215, 0.9999932892]  # out[2]
    ###########################

    # print(vecopts['ALTIM_BORESIGHT'])

    # apply pointing corrections
    # vecin = {'ZPT':vecopts['ALTIM_BORESIGHT']}

    # setup all combinations between years
    par = int(cmb_y_in)

    if XovOpt.get("monthly_sets"):
        if XovOpt.get("instrument") == 'BELA':
            misy = ['26','27']
        else:
            misy = ['11', '12', '13', '14', '15']

        months = np.arange(1,13,1)
        misy = [x+f'{y:02}' for x in misy for y in months]
        if XovOpt.get("instrument") != 'BELA':
            misy = ['0801','0810']+misy[2:-8]
    else:
        if XovOpt.get("instrument") == 'BELA':
            misy = ['26','27'] #+str("{:02}".format(i)) for i in range(1,13,1)]
        elif XovOpt.get("instrument") == 'LOLA':
            misy = ['09', '10']
        else:
            misy = ['08','11', '12', '13', '14', '15']

    misycmb = [x for x in itert.combinations_with_replacement(misy, 2)]
    # print(misycmb)
    if XovOpt.get("debug"):
        print("Choose grid element among:",dict(map(reversed, enumerate(misycmb))))
    print(par, misycmb[par]," has been selected!")

    ###########################
    startInit = time.time()

    if iter_in == 0 and XovOpt.get("compute_input_xov"):

        # check if input_xov already exists, if yes don't recreate it (should be conditional...)
        input_xov_path = XovOpt.get("outdir") + outdir_in + 'xov/tmp/xovin_' + str(misycmb[par][0]) + '_' + str(misycmb[par][1]) + '.pkl.gz'
        if os.path.exists(input_xov_path) and XovOpt.get("instrument") == 'BELA':
            print("input xov file already exists in", input_xov_path)
            print("Rerun without computing this cumbersome input, be smart!")
            exit(0)

        # -------------------------------
        # File reading and ground-tracks computation
        # -------------------------------

        # read all MLA datafiles (*.TAB in data_pth) corresponding to the given years
        # for orbitA and orbitB.
        # Geoloc, if active, will process all files in A+B. Xov will only process combinations
        # of orbits from A and B
        # allFilesA = glob.glob(os.path.join(data_pth, 'MLAS??RDR' + misycmb[par][0] + '*.TAB'))
        # allFilesB = glob.glob(os.path.join(data_pth, 'MLAS??RDR' + misycmb[par][1] + '*.TAB'))

        # print(os.path.join(outdir, indir_in + misycmb[par][0][:2] + '/gtrack_'+misycmb[par][0]+'*'))
        # print(glob.glob(os.path.join(outdir, indir_in + misycmb[par][0][:2] + '/gtrack_'+misycmb[par][0]+'*')))

        if XovOpt.get("monthly_sets"):
          allFilesA = glob.glob(os.path.join(XovOpt.get("outdir"), indir_in + misycmb[par][0][:2] + '/gtrack_' + misycmb[par][0] + '*'))
          allFilesB = glob.glob(os.path.join(XovOpt.get("outdir"), indir_in + misycmb[par][1][:2] + '/gtrack_' + misycmb[par][1] + '*'))
        else:
          allFilesA = glob.glob(os.path.join(XovOpt.get("outdir"), indir_in + misycmb[par][0] + '/*'))
          allFilesB = glob.glob(os.path.join(XovOpt.get("outdir"), indir_in + misycmb[par][1] + '/*'))

        # if misycmb[par][0] == misycmb[par][1]:
        #     allFiles = allFilesA
        # else:
        #     allFiles = allFilesA + allFilesB

        # print(allFiles)

        # xovnames = ['xov_' + fil.split('.')[0][-10:] for fil in allFiles]
        # trackxov_list = []

        # Compute all combinations among available orbits, where first orbit is in allFilesA and second orbit in allFilesB (exclude same tracks cmb)
        # comb=np.array(list(itert.combinations([fil.split('.')[0][-10:] for fil in allFiles], 2))) # this computes comb btw ALL files
        comb = list(
            itert.product([fil.split('.')[0].split('_')[-1] for fil in allFilesA], [fil.split('.')[0].split('_')[-1] for fil in allFilesB]))
        comb = np.array([c for c in comb if c[0] != c[1]])

        # if iter>0, don't test all combinations, only those resulting in xovers at previous iter
        # TODO, check wether one could safely save time by only considering xovers with a given weight
        iter = int(outdir_in.split('/')[1].split('_')[-1])
        if iter>0:
            comb = select_useful_comb(comb, iter, outdir_in)

        # print(comb)

        # load all tracks
        # tmp = [gtrack(vecopts) for i in range(len(allFiles))]

        # if False:
        #     tracklist = {}
        #     for idx, fil in enumerate(allFiles):
        #         try:
        #             print(outdir)
        #             _ = tmp[idx].load(outdir + '/gtrack_' + fil.split('.')[0][-10:] + '.pkl')
        #             tracklist[str(_.name)] = _
        #         except:
        #             print('Failed to load' + outdir + '/gtrack_' + fil.split('.')[0][-10:] + '.pkl')

        # read all ladata needed for these combinations
        # print(comb)
        # print(comb.shape,np.ravel(comb).shape,len(set(np.ravel(comb))))
        # print(set(np.ravel(comb)))
        track_obj = gtrack(XovOpt.get("vecopts"))
        mladata = {}
        cols = ['ET_TX', 'TOF', 'orbID', 'seqid', 'ET_BC', 'offnadir', 'LON', 'LAT', 'R',
             'X_stgprj', 'Y_stgprj']

        for track_id in set(np.ravel(comb)):
            track_obj = track_obj.load(XovOpt.get("outdir") + outdir_in + 'gtrack_' + track_id[:2] + '/gtrack_' + track_id + '.pkl')

            # resurrect as soon as got also south part of obs track
            #if instr == 'BELA':
            #     print(track_obj.ladata_df['LAT'].max(axis=0))
            #     print(track_obj.ladata_df['LAT'].min(axis=0))
            #     exit()
            #
            #     # print("presel",len(track_obj.ladata_df))
            #     mladata[track_id] =track_obj.ladata_df.loc[track_obj.ladata_df['LAT']>=0,cols]
            #     # print("postsel",len(mladata[track_id]))
            # else:
            mladata[track_id] =track_obj.ladata_df.loc[:,cols]
        # print(len(mladata))
        # exit()
            # transform to df to get memory
            # print("total memory:",pd.from_dict(mladata).memory_usage(deep=True).sum()*1.e-6)
            # exit()


        endInit = time.time()
        print(
            '----- Runtime Init= ' + str(endInit - startInit) + ' sec -----' + str(
                (endInit - startInit) / 60.) + ' min -----')

        # -------------------------------
        # Xovers setup
        # -------------------------------

        startXov2 = time.time()

        print("Memory of mladata (Mb):",sum([mladata[x].memory_usage(deep=True).sum() for x in mladata])/1.e6)
        # mladata_1 = dict(list(mladata.items())[:len(mladata) // 3])
        # mladata_2 = dict(list(mladata.items())[len(mladata) // 3:2*(len(mladata) // 3)])
        # mladata_3 = dict(list(mladata.items())[2*(len(mladata) // 3):])
        #
        # print(mladata_1.keys())
        # print(comb)
        # # mask = np.all(np.any(comb[..., None] == np.array(list(mladata_1.keys()))[None, None], axis=1), axis=1)
        # # print(mask)
        # # print(comb[mask])
        # mask = np.isin(comb, list(mladata_1.keys()))
        # print(mask)
        # print(np.all(mask,axis=1))
        # comb_1 = comb[np.all(mask,axis=1),:]
        # exit()
        if XovOpt.get("instrument") =='BELA':
            os.makedirs(XovOpt.get("tmpdir"), exist_ok=True)
            mladata_pkl_fn = XovOpt.get("tmpdir") + 'mladata_tmp_'+cmb_y_in+'.pkl'
            with open(mladata_pkl_fn, 'wb') as handle:
                import pickle
                pickle.dump(mladata, handle, protocol=pickle.HIGHEST_PROTOCOL)
            args = ((fil.split('.')[0].split('_')[-1], comb, misycmb, par, mladata_pkl_fn, XovOpt.get("outdir") + outdir_in) for fil in allFilesA)
        else:
            # args = ((fil.split('.')[0][-10:], comb, misycmb, par, mladata, outdir + outdir_in) for fil in allFilesA)
            args = ((fil.split('.')[0].split('_')[-1], comb, misycmb, par, mladata, XovOpt.get("outdir") + outdir_in) for fil in allFilesA)
        print("Looking for (potential) xovers within combinations of",len(allFilesA),"tracks (A) with",len(allFilesB),"tracks (B)...")

        # print(XovOpt.get("outdir")+indir_in[:-7]+'xov/tmp/xovin_'+misycmb[par][0]+'_'+misycmb[par][1]+'.pkl.gz')
        if XovOpt.get("instrument") == 'BELA' and \
                os.path.exists(XovOpt.get("outdir")+indir_in[:-7]+'xov/tmp/xovin_'+misycmb[par][0]+'_'+misycmb[par][1]+'.pkl.gz'):
            print("Xovin exists. Exit.")
            exit()

        # loop over all gtracks
        # parallel = 1
        if XovOpt.get("parallel"):
            # close?join?
            if XovOpt.get("local"):
                # forks everything, if much memory needed, use the remote option with get_context
                from tqdm.contrib.concurrent import process_map  # or thread_map
                result = process_map(launch_xov, args, max_workers=ncores, total=len(allFilesA))
            else:
                # filnams_loop = [fil.split('.')[0][-10:] for fil in allFiles]
                # print(filnams_loop)
                # print((mp.cpu_count() - 1))
                pool = mp.get_context("spawn").Pool(processes=ncores)  # mp.cpu_count())
                # store list of tracks with xovs
                result = pool.map(launch_xov, args)  # parallel
# ######################################
#             ncores = mp.cpu_count() - 1  # 8
#
            # if local:
            #     from tqdm import tqdm
            #     pbar = tqdm(total=len(allFilesA))
            #
            #     def update(*a):
            #         pbar.update()
            #
            # result = []
            # with mp.get_context("spawn").Pool(processes=ncores) as pool:
            #     for fil in allFilesA:
            #         if local:
            #             result.append(pool.apply_async(launch_xov, args=(fil.split('.')[0][-10:], comb, misycmb, par, mladata, outdir + outdir_in), callback=update))
            #         else:
            #             result.append(pool.apply_async(launch_xov, args=(fil.split('.')[0][-10:], comb, misycmb, par, mladata, outdir + outdir_in)))
            #
            #     pool.close()
            #     pool.join()
            # # result.get blocks processing until all results of apply_async are fetched
            # result = [r.get() for r in result]

        else:
            result = []  # seq
            if XovOpt.get("local"):
                from tqdm import tqdm
                for arg in tqdm(args, total=len(allFilesA)):
                    result.append(launch_xov(arg))
            else:
                for arg in args:
                    result.append(launch_xov(arg))
            # print(result)

        # remove None from list
        result = [x for x in result if x is not None]
        if len(result)>0:
            if XovOpt.get("new_algo"):
                rough_xov = pd.concat(result).reset_index()
            else:
                acttracks = np.unique(np.array([x for x in result if x is not None]).flatten())
        else:
            print("### PyXover: no xovers between the available tracks")
            exit()

        endXov2 = time.time()
        print(
            '----- Runtime Xov2 = ' + str(endXov2 - startXov2) + ' sec -----' + str(
                (endXov2 - startXov2) / 60.) + ' min -----')

    else: # xovs will be taken from old iter
        rough_xov = pd.DataFrame()

    # called either with results from xov_rough (iter=0) or with empty df and xovers from old solution
    if XovOpt.get("new_algo"):
        print("Calling a new awesome routine!!")
        xov_prc_iters_run(outdir_in, iter_in, misycmb[par], rough_xov)

    #############################################
    endXov3 = time.time()
    print(
        '----- Runtime Xov rough+fine = ' + str(endXov3 - startInit) + ' sec -----' + str(
            (endXov3 - startInit) / 60.) + ' min -----')
    #############################################    # exit()


def select_useful_comb(comb, iter, outdir_in):
    outdir_old = outdir_in.replace('_' + str(iter) + '/', '_' + str(iter - 1) + '/')
    print(outdir_old, outdir_in)
    tmp = Amat(XovOpt.get("vecopts"))
    tmp = tmp.load(glob.glob(XovOpt.get("outdir") + outdir_old + 'Abmat*.pkl')[0])

    old_xov_orb = (tmp.xov.xovers['orbA'].map(str) + tmp.xov.xovers['orbB']).values

    if len(comb)>0:
        comb_new = np.sum(comb.astype(object),axis=1)
        common_index = np.intersect1d(comb_new,old_xov_orb,return_indices=True)[1]
        comb_new = comb[common_index]
    else:
        comb_new = np.array([])

    # slow
    # old_xov_orb = tmp.xov.xovers[['orbA', 'orbB']].values
    # intersetingRows = [(old_xov_orb == irow).all(axis=1).any() for irow in comb]
    # comb = comb[intersetingRows]

    print("Based on previous xovs:", len(comb_new), "tracks combs selected out of", len(comb))

    return comb_new


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
