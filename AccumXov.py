#!/usr/bin/env python3
# ----------------------------------
# AccumXov
# ----------------------------------
# Author: Stefano Bertone
# Created: 04-Mar-2019
#
import re
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
#
# #exit()

########################################

def prepro(dataset):
    # read input args
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))

    # locate data
    # locate data
    if local == 0:
        data_pth = outdir
        data_pth += dataset
        # load kernels
    else:
        data_pth = outdir
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
    return data_pth, vecopts

def load_combine(xov_pth,vecopts):
    # -------------------------------
    # Amat setup
    # -------------------------------
    pd.set_option('display.max_columns', 500)

    # Combine all xovers and setup Amat
    xov_ = xov(vecopts)

    if local == 0:
        data_pth = '/att/nobackup/sberton2/MLA/MLA_RDR/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
        dataset = ''  # 'small_test/' #'test1/' #'1301/' #
        data_pth += dataset
        # load kernels
    else:
        data_pth = '/home/sberton2/Works/NASA/Mercury_tides/data/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
        dataset = 'SIM_1301/mlatimes/0res_1amp_tst'
        data_pth += dataset

    allFiles = glob.glob(os.path.join(data_pth, 'MLASIMRDR' + '*.TAB'))
    tracknames = [fil.split('.')[0][-10:] for fil in allFiles]
    misy = ['11', '12', '13', '14', '15']
    misycmb = [x + '_' + y for x in tracknames for y in misy]
    print(misycmb)

    print([xov_pth + 'xov_' + x + '.pkl' for x in misycmb])
    xov_list = [xov_.load(xov_pth + 'xov_' + x + '.pkl') for x in misycmb]
    #print(xov_list[0].xovers)

    orb_unique = [x.xovers['orbA'].tolist() for x in xov_list if len(x.xovers) > 0]
    orb_unique.extend([x.xovers['orbB'].tolist() for x in xov_list if len(x.xovers) > 0])
    orb_unique = list(set([y for x in orb_unique for y in x]))

    xov_cmb = xov(vecopts)
    xov_cmb.combine(xov_list)

    return xov_cmb

def get_stats(xov_lst,resval,amplval):
    import seaborn.apionly as sns

    #print(xov_cmb.xovers)

    dR_avg = []
    dR_std = []
    dR_max = []
    dR_min = []
    dR_RMS = []

    for xov in xov_lst:

        #print(xov.)

        if len(xov.xovers)>0:
            xov.xovers['dist_avg'] = xov.xovers.filter(regex='^dist_.*$').mean(axis=1)
            xov.xovers['dist_max'] = xov.xovers.filter(regex='^dist_.*$').max(axis=1)
            xov.xovers['dist_minA'] = xov.xovers.filter(regex='^dist_A.*$').min(axis=1)
            xov.xovers['dist_minB'] = xov.xovers.filter(regex='^dist_B.*$').min(axis=1)
            xov.xovers['dist_min_avg'] = xov.xovers.filter(regex='^dist_min.*$').mean(axis=1)

            #print(len(xov.xovers))
            #print(len(xov.xovers[xov.xovers.dist_max > 5]))
            xov.xovers = xov.xovers[xov.xovers.dist_max < 5]

            #print(xov.xovers[['dist_max','dist_avg','dist_minA','dist_minB','dist_min_avg','dR']])
            # print(xov.xovers['dist_minA'].min(axis=0))
            # print(xov.xovers['dist_minB'].min(axis=0))
            # print(xov.xovers[['dist_max','dR']].abs())
            xov.xovers[['dist_max','dR']].abs().plot(kind='scatter', x='dist_max', y='dR')
            plt.savefig('tmp/dR_vs_dist_32_4.png')
            plt.clf()

            xov.xovers['dist_avg'].plot()
            plt.savefig('tmp/dist.png')
            plt.close()

            dR_avg.append(xov.xovers.dR.mean(axis=0))
            dR_std.append(xov.xovers.dR.std(axis=0))
            dR_max.append(xov.xovers.dR.max(axis=0))
            dR_min.append(xov.xovers.dR.min(axis=0))

            dR_RMS.append(np.sqrt(np.mean(xov.xovers.dR.values**2,axis=0)))
            #print(dR_RMS)

    df_ = pd.DataFrame(list(zip(resval,amplval,dR_RMS)),columns=['res','ampl','RMS'])
    # create pivot table, days will be columns, hours will be rows
    piv = pd.pivot_table(df_, values="RMS",index=["ampl"], columns=["res"], fill_value=0)
    #plot pivot table as heatmap using seaborn

    fig, [ax0,ax1] = plt.subplots(nrows=2)
    ax0.errorbar(range(len(dR_avg)),dR_avg, yerr=dR_std, fmt='-o')
    ax0.set(xlabel='Exp', ylabel='dR_avg (m)')
    ax1 = sns.heatmap(piv, square=False, annot=True, robust=True,
                     cbar_kws={'label': 'RMS (m)'} )
    ax1.set(xlabel='Topog resolution (1st octave, km)',
            ylabel='Topog ampl rms (1st octave, m)')
    fig.savefig('tmp/tst.png')
    plt.clf()
    plt.close()
    #ax0.set_title('variable, symmetric error')
    print(dR_avg,dR_std,dR_max,dR_min)

def prepare_Amat(vecopts,par_list=''):
    # simplify and downsize
    if par_list=='':
        par_list = xov_cmb.xovers.columns.filter(regex='^dR.*$')

    df_orig = xov_cmb.xovers[par_list]
    df_float = xov_cmb.xovers.filter(regex='^dR.*$').apply(pd.to_numeric, errors='ignore', downcast='float')

    xov_cmb.xovers = pd.concat([df_orig, df_float], axis=1)
    xov_cmb.xovers.info(memory_usage='deep')

    if (True):
        pd.set_option('display.max_columns', 500)
        print(xov_cmb.xovers)

    xovi_amat = Amat(vecopts)
    xovi_amat.setup(xov_cmb)

    return xovi_amat

def solve(xovi_amat,dataset):
    # Solve
    print(len(xovi_amat.parNames))
    sol4_pars = []
    sol4_pars = ['1301010742_dR/dA_A','1301011542_dR/dA_B','dR/dRA','dR/dh2']
    #print([xovi_amat.parNames[p] for p in sol4_pars])
    if sol4_pars != []:
        spA_sol4 = xovi_amat.spA[:,[xovi_amat.parNames[p] for p in sol4_pars]]
    else:
        spA_sol4 = xovi_amat.spA

    #print('dense A', spA_sol4.todense())
    #tst = np.array([1.e-3 for i in range(len(xovi_amat.b))])
    #xovi_amat.b = tst
    print('max B', xovi_amat.b.max(),xovi_amat.b.mean())

    #print(np.linalg.inv((spA_sol4.transpose()*spA_sol4).todense()))
    print(np.linalg.pinv((spA_sol4.transpose()*spA_sol4).todense()))
    # Compute the covariance matrix

    print('sol dense',np.linalg.lstsq(spA_sol4.todense(), xovi_amat.b, rcond=1))
    # print(xovi_amat.spA.shape)
    # print(xovi_amat.b.shape)
    xovi_amat.sol = lsqr(spA_sol4, xovi_amat.b,show=True,iter_lim=1000,atol=1.e-9,btol=1.e-9,calc_var=True)

    print("sol4pars:", sol4_pars)
    print('xovi_amat.sol',xovi_amat.sol)

    # Save to pkl
    xovi_amat.save(outdir + 'Abmat_' + dataset.split('/')[0] + '.pkl')

    print('sparse density = ' + str(xovi_amat.spA.density))

    if (debug):  # check if correctly saved
        tmp = Amat(vecopts)
        tmp = tmp.load(outdir + 'Amat_' + dataset.split('/')[0] + '.pkl')
        print(tmp.A)

def main(arg):
    print(arg)
    datasets = arg #['sim_mlatimes/0res_35amp']

    _ = [x.split('/')[-2] for x in datasets]
    resval = [10./2**int(''.join(filter(str.isdigit, strng.split('_')[0]))) for strng in _]
    amplval = [int(''.join(filter(str.isdigit, strng.split('_')[1]))) for strng in _]

    xov_cmb_lst = []

    for ds in datasets:
        data_pth, vecopts = prepro(ds)  # "test/"  # 'small_test/' #'1301/' #)
        print(data_pth)
        #exit()

        xov_cmb = load_combine(data_pth,vecopts)
        xov_cmb_lst.append(xov_cmb)

    get_stats(xov_cmb_lst,resval,amplval)

    #par_list = ['orbA', 'orbB', 'xOvID']
    #xovi_amat = prepare_Amat(xov_cmb,vecopts,par_list)

    #solve(xovi_amat, dataset)

if __name__ == '__main__':

    import sys

    ##############################################
    # launch program and clock
    # -----------------------------
    startT = time.time()

    print(sys.argv)
    main(sys.argv[1:])

    ##############################################
    # stop clock and print runtime
    # -----------------------------
    endT = time.time()
    print('----- Runtime Amat = ' + str(endT - startT) + ' sec -----' + str(
        (endT - startT) / 60.) + ' min -----')