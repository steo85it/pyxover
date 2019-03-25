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
import sys
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
# from mapcount import mapcount
from prOpt import parallel, SpInterp, new_gtrack, new_xov, outdir, auxdir, local
from ground_track import gtrack
from xov_setup import xov

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
# start clock
start = time.time()

# read input args
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

# locate data
if local == 0:
    data_pth = '/att/nobackup/sberton2/MLA/MLA_RDR/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
    dataset = ''  # 'test/' #'small_test/' #'1301/' #
    data_pth += dataset
    # load kernels
    spice.furnsh('/att/nobackup/emazaric/MESSENGER/data/furnsh/furnsh.MESSENGER.def')  # 'aux/mymeta')
else:
    data_pth = '/home/sberton2/Works/NASA/Mercury_tides/data/'
    # data_pth = '/home/sberton2/Works/NASA/Mercury_tides/data/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
    dataset = '1301/'  # 'small_dataset/' #''# "test1/"  #''  #
    data_pth += dataset
    # load kernels
    spice.furnsh(auxdir + 'mymeta')  # 'aux/mymeta')

# set ncores
ncores = mp.cpu_count() - 1  # 8

if parallel:
    print('Process launched on ' + str(ncores) + ' CPUs')

##############################################

# Setup some useful options
vecopts = {'SCID': '-236',
           'SCNAME': 'MESSENGER',
           'SCFRAME': -236000,
           'INSTID': (-236500, -236501),
           'INSTNAME': ('MSGR_MLA', 'MSGR_MLA_RECEIVER'),
           'PLANETID': '199',
           'PLANETNAME': 'MERCURY',
           'PLANETRADIUS': 2440.,
           'PLANETFRAME': 'IAU_MERCURY',
           'OUTPUTTYPE': 1,
           'ALTIM_BORESIGHT': '',
           'INERTIALFRAME': 'J2000',
           'INERTIALCENTER': 'SSB',
           'PARTDER': ''}

out = spice.getfov(vecopts['INSTID'][0], 1)
# updated w.r.t. SPICE from Mike's scicdr2mat.m
vecopts['ALTIM_BORESIGHT'] = [0.0022105, 0.0029215, 0.9999932892]  # out[2]
###########################

# print(vecopts['ALTIM_BORESIGHT'])

# apply pointing corrections
# vecin = {'ZPT':vecopts['ALTIM_BORESIGHT']}

# setup all combinations between years
par = int(sys.argv[1])
misy = ['11', '12', '13', '14', '15']
misycmb = [x for x in itert.combinations_with_replacement(misy, 2)]
print(misycmb)
print(par, misycmb[par])
# exit()

# -------------------------------
# File reading and ground-tracks computation
# -------------------------------

startInit = time.time()

# read all MLA datafiles (*.TAB in data_pth) corresponding to the given years
# for orbitA and orbitB.
# Geoloc, if active, will process all files in A+B. Xov will only process combinations
# of orbits from A and B
allFilesA = glob.glob(os.path.join(data_pth, 'MLASCIRDR' + misycmb[par][0] + '*.TAB'))
allFilesB = glob.glob(os.path.join(data_pth, 'MLASCIRDR' + misycmb[par][1] + '*.TAB'))

if misycmb[par][0] == misycmb[par][1]:
    allFiles = allFilesA
else:
    allFiles = allFilesA + allFilesB

print(allFiles)

endInit = time.time()
print(
    '----- Runtime Init= ' + str(endInit - startInit) + ' sec -----' + str((endInit - startInit) / 60.) + ' min -----')

startPrepro = time.time()

# Prepare list of tracks to geolocalise
tracknames = ['gtrack_' + fil.split('.')[0][-10:] for fil in allFiles]

if new_gtrack and parallel and SpInterp > 1:
    for track_id, infil in zip(tracknames, allFiles):
        track = track_id
        track = gtrack(vecopts)
        #    try:
        track.prepro(infil)
    #    except:
    #      print('Issue in preprocessing for '+track_id)

    if SpInterp == 3:
        print('Orbit and attitude data loaded for years 20' + str(misycmb[par][0]) + ' and 20' + str(misycmb[par][1]))
        endPrepro = time.time()
        print('----- Runtime Init= ' + str(endPrepro - startPrepro) + ' sec -----' + str(
            (endPrepro - startPrepro) / 60.) + ' min -----')
        exit()

endPrepro = time.time()
print('----- Runtime Prepro= ' + str(endPrepro - startPrepro) + ' sec -----' + str(
    (endPrepro - startPrepro) / 60.) + ' min -----')

startGeoloc = time.time()


def launch_gtrack(args):
    track_id, infil = args

    track = gtrack(vecopts)

    if new_gtrack:
        if os.path.isfile(outdir + track_id + '.pkl') == False or new_gtrack == 2:
            try:
                track.setup(infil)
                track.save(outdir + track_id + '.pkl')
                print('Orbit ' + track_id.split('_')[1] + ' processed and written to ' + outdir + track_id + '.pkl !')
            except:
                print('failed to process ' + track_id)
        else:
            # track = track.load('out/'+track_id+'.pkl')
            print('Gtrack file ' + 'out/' + track_id + '.pkl already existed!')


# loop over all gtracks
if parallel:
    # print((mp.cpu_count() - 1))
    args = zip(tracknames, allFiles)
    pool = mp.Pool(processes=ncores)  # mp.cpu_count())
    _ = pool.map(launch_gtrack, args)  # parallel
    pool.close()
    pool.join()
else:
    _ = [launch_gtrack(args) for args in zip(tracknames, allFiles)]  # seq

endGeoloc = time.time()
print('----- Runtime Geoloc= ' + str(endGeoloc - startGeoloc) + ' sec -----' + str(
    (endGeoloc - startGeoloc) / 60.) + ' min -----')

# -------------------------------
# Xovers setup
# -------------------------------

startXov2 = time.time()

xovnames = ['xov_' + fil.split('.')[0][-10:] for fil in allFiles]
trackxov_list = []

# Compute all combinations among available orbits, where first orbit is in allFilesA and second orbit in allFilesB (exclude same tracks cmb)
# comb=np.array(list(itert.combinations([fil.split('.')[0][-10:] for fil in allFiles], 2))) # this computes comb btw ALL files
comb = list(
    itert.product([fil.split('.')[0][-10:] for fil in allFilesA], [fil.split('.')[0][-10:] for fil in allFilesB]))
comb = np.array([c for c in comb if c[0] != c[1]])

# load all tracks
tmp = [gtrack(vecopts) for i in range(len(allFiles))]

if False:
    tracklist = {}
    for idx, fil in enumerate(allFiles):
        try:
            _ = tmp[idx].load(outdir + '/gtrack_' + fil.split('.')[0][-10:] + '.pkl')
            tracklist[str(_.name)] = _
        except:
            print('Failed to load' + outdir + '/gtrack_' + fil.split('.')[0][-10:] + '.pkl')


def launch_xov(track_id):
    # track_id = tracklist[seqId].name
    # print(track_id)

    if new_xov:  # and track_id=='1301232350':
        if not os.path.isfile(outdir + 'xov_' + track_id + '_' + misycmb[par][1] + '.pkl') or new_xov == 2:

            # print("Processing " + track_id + " ...")

            # try:
            #    trackA = track_id
            #    trackA = tracklist[str(track_id)]
            trackA = gtrack(vecopts)
            trackA = trackA.load(outdir + 'gtrack_' + track_id + '.pkl')
            if not trackA == None:

                xov_tmp = track_id
                xov_tmp = xov(vecopts)

                # print(comb)

                # loop over all combinations containing track_id
                for gtrackA, gtrackB in [s for s in comb if track_id in s[0]]:

                    # if debug:
                    #    print("Processing " + gtrackA + " vs " + gtrackB)

                    if gtrackB > gtrackA:
                        # try:
                        #        trackB = track_id
                        #        trackB = tracklist[str(gtrackB)]
                        trackB = gtrack(vecopts)
                        trackB = trackB.load(outdir + 'gtrack_' + gtrackB + '.pkl')
                        if not trackB == None:
                            xov_tmp.setup(pd.concat([trackA.ladata_df, trackB.ladata_df]).reset_index(drop=True))
                    # except:
                    #     print(
                    #         'failed to load trackB ' + outdir + 'gtrack_' + gtrackB + '.pkl' + ' to process ' + outdir + 'gtrack_' + track_id + '.pkl')

                # for each gtrackA, write
                # print([s for s in comb if track_id in s[0]])
                if [s for s in comb if track_id in s[0]] and len(xov_tmp.xovers) > 0:
                    xov_tmp.save(outdir + 'xov_' + gtrackA + '_' + misycmb[par][1] + '.pkl')
                    # print(xov_tmp.xovers)
                    trackxov_list.append(gtrackA)
                    print('Xov for ' + track_id + ' processed and written to ' + outdir + 'xov_' + gtrackA + '_' +
                          misycmb[par][1] + '.pkl !')
                    return gtrackA

        # except:
        #     print(
        #         'failed to load trackA ' + outdir + 'gtrack_' + track_id + '.pkl' + ' to process xov from ' + outdir + 'gtrack_' + track_id + '.pkl')

        else:

            #      track = track.load('out/xov_'+gtrackA+'.pkl')
            print('Xov for ' + track_id + ' already exists in ' + outdir + 'xov_' + track_id + '.pkl !')


# loop over all gtracks
if parallel:
    filnams_loop = [fil.split('.')[0][-10:] for fil in allFiles]
    # print(filnams_loop)
    # print((mp.cpu_count() - 1))
    pool = mp.Pool(processes=ncores)  # mp.cpu_count())
    # store list of tracks with xovs
    acttracks = pool.map(launch_xov, filnams_loop)  # parallel
    acttracks = np.unique(np.array([x for x in acttracks if x is not None]).flatten())
    pool.close()
    pool.join()
else:
    _ = [launch_xov(fil.split('.')[0][-10:]) for fil in allFiles]  # seq

endXov2 = time.time()
print(
    '----- Runtime Xov2 = ' + str(endXov2 - startXov2) + ' sec -----' + str((endXov2 - startXov2) / 60.) + ' min -----')

# # -------------------------------
# # Amat setup
# # -------------------------------
# startXovPart = time.time()
#
# # Combine all xovers and setup Amat
# xov_ = xov(vecopts)
#
# if parallel and new_xov:
#     trackxov_list = acttracks
# elif ~new_xov:
#     trackxov_list = [fil.split('.')[0][-10:] for fil in allFiles]
#
# print('blablabla')
# print([xov_.load(outdir + 'xov_' + x + '_' + y + '.pkl') for x in trackxov_list for y in misy])
# print([outdir + 'xov_' + x + '_' + y + '.pkl' for x in trackxov_list for y in misy])
# exit()
#
# xov_list = [xov_.load(outdir + 'xov_' + x + '.pkl') for x in trackxov_list]
# xov_cmb = xov(vecopts)
# xov_cmb.combine(xov_list)
# print("blabla")
#
# exit()
#
# if debug:
#     pd.set_option('display.max_columns', 500)
#     print(xov_cmb.xovers)
#
# xovi_amat = Amat(vecopts)
# xovi_amat.setup(xov_cmb)
# xovi_amat.save(outdir + 'Amat_' + dataset.split('/')[0] + '_' + arg + '.pkl')
#
# print('sparse density = ' + str(xovi_amat.spA.density))
#
# if debug:  # check if correctly saved
#     tmp = Amat(vecopts)
#     tmp = tmp.load(outdir + 'Amat_' + dataset.split('/')[0] + '_' + arg + '.pkl')
#     print(tmp.A)
#
# endXovPart = time.time()
# print('----- Runtime Amat = ' + str(endXovPart - startXovPart) + ' sec -----' + str(
#     (endXovPart - startXovPart) / 60.) + ' min -----')
#
# # exit()

##############################################
# stop clock and print runtime
# -----------------------------
end = time.time()
print('----- Runtime = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
