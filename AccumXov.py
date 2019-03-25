#!/usr/bin/env python3
# ----------------------------------
# AccumXov
# ----------------------------------
# Author: Stefano Bertone
# Created: 04-Mar-2019
#
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import sys
import glob

import pandas as pd
# from itertools import izip, count
# from geopy.distance import vincenty

# from collections import defaultdict
# import mpl_toolkits.basemap as basemap

import time
from scipy.sparse.linalg import lsqr

# mylib
# from mapcount import mapcount
from prOpt import debug, outdir, local
from xov_setup import xov
from Amat import Amat

########################################
# test space

# # start clock
# start = time.time()
#
# # read input args
# print('Number of arguments:', len(sys.argv), 'arguments.')
# print('Argument List:', str(sys.argv))
#
# # locate data
# if local == 0:
#     data_pth = '/att/nobackup/sberton2/MLA/MLA_RDR/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
#     dataset = ''  # 'small_test/' #'test1/' #'1301/' #
#     data_pth += dataset
#     # load kernels
# else:
#     data_pth = '/home/sberton2/Works/NASA/Mercury_tides/data/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
#     dataset = "test/"  # ''  # 'small_test/' #'1301/' #
#     data_pth += dataset
#     # load kernels
#
# vecopts = {'SCID': '-236'}
#
# # tmp_ser = Amat(vecopts)
# # tmp_ser = tmp_ser.load('out/small_ser/Amat_small_dataset.pkl')
#
# # Combine all xovers and setup Amat
# xov_ = xov(vecopts)
#
# allFiles = glob.glob(os.path.join(data_pth, 'MLASCIRDR' + '*.TAB'))
# tracknames = [fil.split('.')[0][-10:] for fil in allFiles]
# misy = ['11' ,'12','13','14','15']
# misycmb = [x + '_' + y for x in tracknames for y in misy]
#
# xov_list = [xov_.load(outdir + 'xov_' + x + '.pkl') for x in misycmb]
#
# orb_unique = [x.xovers['orbA'].tolist() for x in xov_list if len(x.xovers) > 0]
# orb_unique.extend([x.xovers['orbB'].tolist() for x in xov_list if len(x.xovers) > 0])
# orb_unique = list(set([y for x in orb_unique for y in x]))
#
# xov_cmb = xov(vecopts)
# xov_cmb.combine(xov_list)
#
# # simplify and downsize
# df_orig = xov_cmb.xovers[['orbA', 'orbB', 'xOvID']]
# df_float = xov_cmb.xovers.filter(regex='^dR.*$').apply(pd.to_numeric, errors='ignore', downcast='float')
#
# xov_cmb.xovers = pd.concat([df_orig, df_float], axis=1)
#
# xovers_df = xov_cmb.xovers
# parOrb_xy = sorted(xov_cmb.parOrb_xy)
# parGlo_xy = sorted(xov_cmb.parGlo_xy)
# # self.par_xy
#
# # print(parOrb_xy,parGlo_xy)
#
# # Retrieve all orbits involved in xovers
# orb_unique = xov_cmb.tracks
#
# OrbParFull = [x + '_' + y for x in orb_unique for y in parOrb_xy]
# Amat_col = OrbParFull + parGlo_xy
#
# # print(len(Amat_col))
# dict_ = dict(zip(Amat_col, range(len(Amat_col))))
# print(dict_)
#
# #############################
# #tmp_par = Amat(vecopts)
# #tmp_par = tmp_par.load(outdir + 'Abmat_.pkl')
#
# #x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = tmp_par.sol
#
# #print(x)
# # print(list({v: k for k, v in dict_.items()}))
# #print(list(dict_.keys())[:len(x)])
# #print([x[i] for i in list({v: k for k, v in dict_.items()})])
# #print(list(dict(zip(list(dict_.keys()), [x[i] for i in list({v: k for k, v in dict_.items()})]))))
# #print(istop)
# #print(itn)
# #print(r1norm)
# #print(var)
#
# # print(tmp_par.spA.to_dense().equals(tmp_ser.spA.to_dense()))
#
# # print(tmp_par.spA)
# # print(tmp_ser.spA)
#
# #exit()

########################################
# start clock
start = time.time()

# read input args
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

# locate data
# locate data
if local == 0:
    data_pth = '/att/nobackup/sberton2/MLA/MLA_RDR/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
    dataset = ''  # 'small_test/' #'test1/' #'1301/' #
    data_pth += dataset
    # load kernels
else:
    data_pth = outdir #'/home/sberton2/Works/NASA/Mercury_tides/data/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
    dataset = ''  # "test/"  # 'small_test/' #'1301/' #
    data_pth += dataset
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

# -------------------------------
# Amat setup
# -------------------------------
startXovPart = time.time()
pd.set_option('display.max_columns', 500)

# Combine all xovers and setup Amat
xov_ = xov(vecopts)

allFiles = glob.glob(os.path.join(data_pth, 'MLASIMRDR' + '*.TAB'))
tracknames = [fil.split('.')[0][-10:] for fil in allFiles]
misy = ['11', '12', '13', '14', '15']
misycmb = [x + '_' + y for x in tracknames for y in misy]

xov_list = [xov_.load(outdir + 'xov_' + x + '.pkl') for x in misycmb]

orb_unique = [x.xovers['orbA'].tolist() for x in xov_list if len(x.xovers) > 0]
orb_unique.extend([x.xovers['orbB'].tolist() for x in xov_list if len(x.xovers) > 0])
orb_unique = list(set([y for x in orb_unique for y in x]))

xov_cmb = xov(vecopts)
xov_cmb.combine(xov_list)

# simplify and downsize
df_orig = xov_cmb.xovers[['orbA', 'orbB', 'xOvID']]
df_float = xov_cmb.xovers.filter(regex='^dR.*$').apply(pd.to_numeric, errors='ignore', downcast='float')

xov_cmb.xovers = pd.concat([df_orig, df_float], axis=1)
xov_cmb.xovers.info(memory_usage='deep')

if (False):
    pd.set_option('display.max_columns', 500)
    print(xov_cmb.xovers)

xovi_amat = Amat(vecopts)
xovi_amat.setup(xov_cmb)

# print(xov_cmb.tracks)
# print(xov_cmb.parOrb_xy)
end_setup = time.time()
print('----- Runtime = ' + str(end_setup - start) + ' sec -----' + str((end_setup - start) / 60.) + ' min -----')

# Solve
# print(xovi_amat.spA.shape)
# print(xovi_amat.b.shape)
xovi_amat.sol = lsqr(xovi_amat.spA, xovi_amat.b)

# Save to pkl
xovi_amat.save(outdir + 'Abmat_.pkl')

end = time.time()
print('----- Runtime = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')

exit()

xovi_amat.save(outdir + 'Abmat_' + dataset.split('/')[0] + '_' + arg + '.pkl')

print('sparse density = ' + str(xovi_amat.spA.density))

if (debug):  # check if correctly saved
    tmp = Amat(vecopts)
    tmp = tmp.load(outdir + 'Amat_' + dataset.split('/')[0] + '_' + arg + '.pkl')
    print(tmp.A)

endXovPart = time.time()
print('----- Runtime Amat = ' + str(endXovPart - startXovPart) + ' sec -----' + str(
    (endXovPart - startXovPart) / 60.) + ' min -----')

# exit()

##############################################
# stop clock and print runtime
# -----------------------------
end = time.time()
print('----- Runtime = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')

# exit()
#
# x = 0
# # Set up Amat
# for i in misycmb:
#     x = len(xov_.load(outdir + 'xov_' + i + '.pkl').xovers)
#     print(x)
#     if (x > 0):
#         xov_ = xov_.load(outdir + 'xov_' + i + '.pkl')
#         break
#
# print(xov_.xovers.columns)
# parOrb_xy = xov_.parOrb_xy
# OrbParFull = [x + '_' + y for x in tracknames for y in parOrb_xy]
# Amat_col = OrbParFull + xov_.parGlo_xy
# Amat_df = pd.DataFrame(columns=Amat_col)
#
# print(Amat_df)
#
# xov_list = [xov_.load(outdir + 'xov_' + x + '.pkl') for x in misycmb[0]]
# print(xov_list[0].xovers)
# # xov_cmb = xov(vecopts)
# # xov_cmb.combine(xov_list)
# # print(xov_cmb.xovers)
#
# xovi_amat = Amat(vecopts)
# xovi_amat.setup(xov_list[0], Amat_col)
#
# exit()
#
# xov_list = [xov_.load(outdir + 'xov_' + x + '.pkl') for x in misycmb]
#
# end = time.time()
# print("done in " + str(end - startXovPart) + "!")  # [len(x.xovers) for x in xov_list])
