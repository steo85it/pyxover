#!/usr/bin/env python3
# ----------------------------------
# PyXover
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-Oct-2018
#
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import glob

# from itertools import izip, count
import multiprocessing as mp
# from geopy.distance import vincenty

import spiceypy as spice
import numpy as np

# from collections import defaultdict
# import mpl_toolkits.basemap as basemap

import time

# mylib
# from mapcount import mapcount
from ground_track import gtrack
from prOpt import parallel, SpInterp, new_gtrack, outdir, auxdir, local, vecopts, debug

# from util import lflatten
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

########################################


def launch_gtrack(args):
    track_id, infil, outdir_in = args
    track = gtrack(vecopts)

    if new_gtrack:
        gtrack_out = outdir + outdir_in + '/' + track_id + '.pkl'
        if os.path.isfile(gtrack_out) == False or new_gtrack == 2:

            if not os.path.exists(outdir + outdir_in):
                os.makedirs(outdir + outdir_in, exist_ok=True)

            # try:
            track.setup(infil)
            #
            if debug:
                print("track#:", track.name)
                print("max diff R",abs(track.ladata_df.loc[:,'R']-track.df_input.loc[:,'altitude']).max())
                print("R",track.ladata_df.loc[:,'R'].max(),track.df_input.loc[:,'altitude'].max())
                print("max diff LON", vecopts['PLANETRADIUS']*1.e3*np.sin(np.deg2rad(abs(track.ladata_df.loc[:, 'LON'] - track.df_input.loc[:, 'geoc_long']).max())))
                print("max diff LAT", vecopts['PLANETRADIUS']*1.e3*np.sin(np.deg2rad(abs(track.ladata_df.loc[:, 'LAT'] - track.df_input.loc[:, 'geoc_lat']).max())))
                print("max elev sim", abs(track.df_input.loc[:,'altitude']).max())
            #exit()
            track.save(gtrack_out)
            print('Orbit ' + track_id.split('_')[1] + ' processed and written to ' + gtrack_out + '!')
        # except:
        #    print('failed to process ' + track_id)
        else:
            # track = track.load('out/'+track_id+'.pkl')
            print('Gtrack file ' + gtrack_out + ' already existed!')


def main(args):

    print(args)

    # read input args
    print('Number of arguments:', len(args), 'arguments.')
    print('Argument List:', str(args))

    epo_in = args[0]
    indir_in = args[1]
    outdir_in = args[2]

    # locate data
    if local == 0:
        data_pth = '/att/nobackup/sberton2/MLA/data/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
        dataset = indir_in  # 'test/' #'small_test/' #'1301/' #
        data_pth += dataset

        # load kernels
        spice.furnsh('/att/nobackup/emazaric/MESSENGER/data/furnsh/furnsh.MESSENGER.def')  # 'aux/mymeta')
    else:
        data_pth = '/home/sberton2/Works/NASA/Mercury_tides/data/'
        # data_pth = '/home/sberton2/Works/NASA/Mercury_tides/data/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
        dataset = indir_in  # 'SIM_1301/mlatimes/0res_35amp_tst/' #'1301' #SIM_1301/sphere/' #35-1024-1-8-5/'  #35-1024-32-4-5/' #  'small_dataset/' #''# "test1/"  #''  #
        data_pth += dataset
        #outdir += outdir_in  # 'sim_mlatimes/0res_35amp/'

        # load kernels
        spice.furnsh(auxdir + 'mymeta')  # 'aux/mymeta')

    # set ncores
    ncores = mp.cpu_count() - 1  # 8

    if parallel:
        print('Process launched on ' + str(ncores) + ' CPUs')

    ##############################################
    # updated w.r.t. SPICE from Mike's scicdr2mat.m
    vecopts['ALTIM_BORESIGHT'] = [0.0022105, 0.0029215, 0.9999932892]  # out[2]
    ###########################

    # -------------------------------
    # File reading and ground-tracks computation
    # -------------------------------

    startInit = time.time()

    # read all MLA datafiles (*.TAB in data_pth) corresponding to the given years
    # for orbitA and orbitB.
    allFiles = glob.glob(os.path.join(data_pth, 'MLAS??RDR' + epo_in + '*.TAB'))

    endInit = time.time()
    print(
        '----- Runtime Init= ' + str(endInit - startInit) + ' sec -----' + str(
            (endInit - startInit) / 60.) + ' min -----')

    startPrepro = time.time()

    # Prepare list of tracks to geolocalise
    tracknames = ['gtrack_' + fil.split('.')[0][-10:] for fil in allFiles]

    if new_gtrack:
        # epo_in=[]
        for track_id, infil in zip(tracknames, allFiles):
            track = track_id
            track = gtrack(vecopts)
            # try:
            track.prepro(infil)
            # except:
            #    print('Issue in preprocessing for '+track_id)
            # epo_in.extend(track.ladata_df.ET_TX.values)

        # epo_in = np.array(epo_in)
        # print(epo_in)
        # print(epo_in.shape)
        # print(np.sort(epo_in)[0])
        # print(np.sort(epo_in)[-1])
        # np.savetxt("tmp/epo_mla_1301.in", epo_in, fmt="%10.5f")
        # exit()

        if SpInterp == 3:
            print(
                'Orbit and attitude data loaded for years 20' + str(misycmb[par][0]) + ' and 20' + str(misycmb[par][1]))
            endPrepro = time.time()
            print('----- Runtime Init= ' + str(endPrepro - startPrepro) + ' sec -----' + str(
                (endPrepro - startPrepro) / 60.) + ' min -----')
            exit()

    endPrepro = time.time()
    print('----- Runtime Prepro= ' + str(endPrepro - startPrepro) + ' sec -----' + str(
        (endPrepro - startPrepro) / 60.) + ' min -----')

    startGeoloc = time.time()

    args = ((tr, fil, outdir_in) for (tr, fil) in zip(tracknames,allFiles))

    # loop over all gtracks
    if parallel:
        # print((mp.cpu_count() - 1))
        pool = mp.Pool(processes=ncores)  # mp.cpu_count())
        _ = pool.map(launch_gtrack, args)  # parallel
        pool.close()
        pool.join()
    else:
        _ = [launch_gtrack(arg) for arg in args]  # seq

    endGeoloc = time.time()
    print('----- Runtime Geoloc= ' + str(endGeoloc - startGeoloc) + ' sec -----' + str(
        (endGeoloc - startGeoloc) / 60.) + ' min -----')


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