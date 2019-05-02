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
from prOpt import debug, outdir, local, sim, parOrb, parGlo
from xov_setup import xov
from Amat import Amat

########################################
# test space
#
# #exit()

########################################

def prepro(dataset):
    # read input args
    # print('Number of arguments:', len(sys.argv), 'arguments.')
    # print('Argument List:', str(sys.argv))

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

def load_combine(xov_pth,vecopts,dataset='sim'):
    # -------------------------------
    # Amat setup
    # -------------------------------
    pd.set_option('display.max_columns', 500)

    # Combine all xovers and setup Amat
    xov_ = xov(vecopts)

    # if local == 0:
    #     data_pth = '/att/nobackup/sberton2/MLA/MLA_RDR/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
    #     dataset = ''  # 'small_test/' #'test1/' #'1301/' #
    #     data_pth += dataset
    #     # load kernels
    # else:
    #     data_pth = '/home/sberton2/Works/NASA/Mercury_tides/data/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
    #     #### TODO needs to be updated automat if sim
    #     if dataset=='real':
    #         dataset = "" # '1301' #'SIM_1301/mlatimes/0res_1amp_tst' #
    #     else:
    #         dataset = 'SIM_1301/mlatimes/0res_1amp_tst' #
    #     data_pth += dataset
    #
    # allFiles = glob.glob(os.path.join(data_pth, 'MLAS??RDR' + '*.TAB'))
    # # print(allFiles)
    # tracknames = [fil.split('.')[0][-10:] for fil in allFiles]
    # misy = ['11', '12', '13', '14', '15']
    # misycmb = [x + '_' + y for x in tracknames for y in misy]
    # print(misycmb)

    allFiles = glob.glob(os.path.join(xov_pth, 'xov_*.pkl'))

    # print([xov_pth + 'xov_' + x + '.pkl' for x in misycmb])
    xov_list = [xov_.load(x) for x in allFiles]
    print(len(xov_list))


    orb_unique = [x.xovers['orbA'].tolist() for x in xov_list if len(x.xovers) > 0]
    orb_unique.extend([x.xovers['orbB'].tolist() for x in xov_list if len(x.xovers) > 0])
    orb_unique = list(set([y for x in orb_unique for y in x]))

    xov_cmb = xov(vecopts)
    xov_cmb.combine(xov_list)
    #print(len(xov_cmb.xovers))

    return xov_cmb

def get_stats(xov_lst,resval,amplval):
    import seaborn.apionly as sns
    import scipy.stats as stats

    print('resval,amplval', resval, amplval)
    #print(xov_cmb.xovers)

    dR_avg = []
    dR_std = []
    dR_max = []
    dR_min = []
    dR_RMS = []

    for idx, xov in enumerate(xov_lst):

        #print(idx, xov)

        if len(xov.xovers)>0:
            xov.xovers['dist_avg'] = xov.xovers.filter(regex='^dist_.*$').mean(axis=1)
            xov.xovers['dist_max'] = xov.xovers.filter(regex='^dist_.*$').max(axis=1)
            xov.xovers['dist_minA'] = xov.xovers.filter(regex='^dist_A.*$').min(axis=1)
            xov.xovers['dist_minB'] = xov.xovers.filter(regex='^dist_B.*$').min(axis=1)
            xov.xovers['dist_min_avg'] = xov.xovers.filter(regex='^dist_min.*$').mean(axis=1)

            # remove data if xover distance from measurements larger than 5km (interpolation error)
            # plus remove outliers with median method
            xov.xovers = xov.xovers[xov.xovers.dist_max < 5]
            print(len(xov.xovers[xov.xovers.dist_max > 5]),
                  'xovers removed by dist from obs > 5km')
            if sim == 0:
                mean_dR, std_dR = xov.remove_outliers('dR')

            #print(xov.xovers[['dist_max','dist_avg','dist_minA','dist_minB','dist_min_avg','dR']])
            # checks = ['dist_minA','dist_minB','dist_max','dist_min_avg','dist_avg','dR']
            # for c in checks:
            #     print(c,xov.xovers[c].mean(axis=0),xov.xovers[c].max(axis=0),xov.xovers[c].min(axis=0))
            # _ = xov.xovers.dR.values**2
            # print('RMS',np.sqrt(np.mean(_[~np.isnan(_)],axis=0)))

            # print('dR:',xov.xovers['dR'].mean(axis=0),xov.xovers['dR'].max(axis=0),xov.xovers['dR'].min(axis=0))
            # print(xov.xovers[['dist_max','dR']].abs())

            if debug:
                # the histogram of the data
                num_bins = 1000
                n, bins, patches = plt.hist(xov.xovers.dR, bins='auto', density=True, facecolor='blue', alpha=0.5)

                # add a 'best fit' line
                y = stats.norm.pdf(bins, mean_dR, std_dR)
                plt.plot(bins, y, 'r--')
                plt.xlabel('dR (m)')
                plt.ylabel('Probability')
                plt.title(r'Histogram of dR: $\mu=' + str(mean_dR) + ', \sigma=' + str(std_dR) + '$')

                # Tweak spacing to prevent clipping of ylabel
                plt.subplots_adjust(left=0.15)
                plt.savefig('tmp/histo_dR_' + str(idx) + '.png')
                plt.clf()

                xov.xovers[['dist_max', 'dR']].abs().plot(kind='scatter', x='dist_max', y='dR')
                plt.savefig('tmp/dR_vs_dist_' + str(idx) + '.png')
                plt.clf()

                xov.xovers['dist_avg'].plot()
                plt.savefig('tmp/dist_' + str(idx) + '.png')
                plt.close()

            dR_avg.append(xov.xovers.dR.mean(axis=0))
            dR_std.append(xov.xovers.dR.std(axis=0))
            dR_max.append(xov.xovers.dR.max(axis=0))
            dR_min.append(xov.xovers.dR.min(axis=0))

            _ = xov.xovers.dR.values ** 2
            dR_RMS.append(np.sqrt(np.mean(_[~np.isnan(_)], axis=0)))
            # print(np.count_nonzero(np.isnan(xov.xovers.dR.values**2)))

            print("xov_xovers_value_count:")
            print(xov.xovers['orbA'].value_counts()[:5])
            print(xov.xovers['orbB'].value_counts()[:5])

    #print(len(resval),len(amplval),len(dR_RMS))
    df_ = pd.DataFrame(list(zip(resval,amplval,dR_RMS)), columns=['res','ampl','RMS'])
    #print(df_)
    # create pivot table, days will be columns, hours will be rows
    piv = pd.pivot_table(df_, values="RMS",index=["ampl"], columns=["res"], fill_value=0)
    #plot pivot table as heatmap using seaborn

    fig, [ax0,ax1] = plt.subplots(nrows=2)
    ax0.errorbar(range(len(dR_avg)),dR_avg, yerr=dR_std, fmt='-o')
    ax0.set(xlabel='Exp', ylabel='dR_avg (m)')
    ax1 = sns.heatmap(piv, square=False, annot=True, robust=True,
                      cbar_kws={'label': 'RMS (m)'}, xticklabels=piv.columns.values.round(2), fmt='.4g')
    ax1.set(xlabel='Topog scale (1st octave, km)',
            ylabel='Topog ampl rms (1st octave, m)')
    fig.savefig('tmp/tst.png')
    plt.clf()
    plt.close()

    #ax0.set_title('variable, symmetric error')
    #print(dR_avg,dR_std,dR_max,dR_min)


def prepare_Amat(xov, vecopts, par_list=''):
    xovtmp = xov.xovers.copy()

    # xov.xovers = xov.xovers[xov.xovers.orbA=='1301042351']
    # xov.xovers.append(xovtmp[xovtmp.orbA=='1301011544'])
    # xov.xovers.append(xovtmp[xovtmp.orbB=='1301042351'])
    # xov.xovers.append(xovtmp[xovtmp.orbB=='1301011544'])

    # exit()

    # remove data if xover distance from measurements larger than 5km (interpolation error, if dist cols exist)
    # plus remove outliers with median method
    if xov.xovers.filter(regex='^dist_.*$').empty==False:
        xov.xovers['dist_max'] = xov.xovers.filter(regex='^dist_.*$').max(axis=1)
        xov.xovers = xov.xovers[xov.xovers.dist_max < 5]

    if sim == 0:
        mean_dR, std_dR = xov.remove_outliers('dR')

    print(xov.xovers.dR.max())
    # exit()

    # simplify and downsize
    if par_list=='':
        par_list = xov.xovers.columns.filter(regex='^dR.*$')

    df_orig = xov.xovers[par_list]
    df_float = xov.xovers.filter(regex='^dR.*$').apply(pd.to_numeric, errors='ignore', downcast='float')

    xov.xovers = pd.concat([df_orig, df_float], axis=1)
    xov.xovers.info(memory_usage='deep')

    if debug:
        pd.set_option('display.max_columns', 500)
        print(xov.xovers)

    xovi_amat = Amat(vecopts)
    xovi_amat.setup(xov)

    xov.xovers = xovtmp.copy()

    return xovi_amat

def solve(xovi_amat,dataset):
    # Solve
    # select subset of parameters
    sol4_orb = [None] # ['1501040322','1411031307'] # ['1301250752'] # '1301011544','1301042351']
    sol4_orbpar = [] #['dR0'] # 'dR/dA', 'dR/dC', 'dR/dR'] # ['dR/dA0','dR/dC0'] #['dR/dA'] #
    sol4_glo = [] #'dR/dRA','dR/dL','dR/dh2'] #['dR/dL','dR/dRA','dR/dDEC'] #'dR/dL','dR/dh2','dR/dRA','dR/dDEC'] # ['dR/dL']

    sol4_pars = solve4setup(sol4_glo, sol4_orb, sol4_orbpar, xovi_amat.parNames.keys())
    print(sol4_pars)

    xovi_amat.sol4_pars = sol4_pars

    # sol4_pars = ['1301011544_dR/dA',
    #              '1301042351_dR/dA']  # 1301011544_dR/dRl','1301042351_dR/dRl','1301011544_dR/dPt','1301042351_dR/dPt'] #,'1301011544_dR/dC','1301042351_dR/dC','1301011544_dR/dR','1301042351_dR/dR'] #,'1301012343_dR/dA','1301011544_dR/dC','1301011544_dR/dR'] #,'dR/dh2']
    # print([xovi_amat.parNames[p] for p in sol4_pars])
    if sol4_pars != []:
        print('pars', [xovi_amat.parNames[p] for p in sol4_pars])
        spA_sol4 = xovi_amat.spA[:,[xovi_amat.parNames[p] for p in sol4_pars]]
        # set b=0 for rows not involving chosen set of parameters
        nnz_per_row = spA_sol4.getnnz(axis=1)
        xovi_amat.b[np.where(nnz_per_row == 0)[0]] = 0
    else:
        spA_sol4 = xovi_amat.spA

    print("sol4pars:", sol4_pars)

    if debug and len(sol4_pars)<50:
        print('dense A', spA_sol4.todense())
        print('B', xovi_amat.b)
        print('max B', xovi_amat.b.max(),xovi_amat.b.mean())
        # Compute the covariance matrix
        print(np.linalg.pinv((spA_sol4.transpose()*spA_sol4).todense()))
        # compute sol
        print('sol dense',np.linalg.lstsq(spA_sol4.todense(), xovi_amat.b, rcond=1))

    # Compute LSQR solution
    xovi_amat.sol = lsqr(spA_sol4, xovi_amat.b,show=True,iter_lim=5000,atol=1.e-9,btol=1.e-9,calc_var=True)

    # Save to pkl
    xovi_amat.save(outdir + 'Abmat_' + dataset.split('/')[0] + '.pkl')

    #print('sparse density = ' + str(xovi_amat.spA.density))

    if debug:  # check if correctly saved
        vecopts = {}
        tmp = Amat(vecopts)
        tmp = tmp.load(outdir + 'Abmat_' + dataset.split('/')[0] + '.pkl')
        print(tmp.spA)
        print(tmp.sol)

def solve4setup(sol4_glo, sol4_orb, sol4_orbpar, track_names):
    if sol4_orb == []:
        sol4_orb = set([i.split('_')[0] for i in track_names])
        sol4_orb = [x for x in sol4_orb if x.isdigit()]
    elif sol4_orb == [None]:
        sol4_orb = []

    if sol4_orbpar == []:
        sol4_orbpar = list(parOrb.keys())

    sol4_orb = [x + '_' + 'dR/'+y for x in sol4_orb for y in sol4_orbpar]

    if sol4_glo == []:
        sol4_glo = list(parGlo.keys())
        sol4_glo = ['dR/'+x for x in sol4_glo]

    sol4_pars = sol4_orb + sol4_glo

    print('solving for:',sol4_pars)

    return sol4_pars


def analyze_sol(xovi_amat,xov):
    # print('xovi_amat.sol',xovi_amat.sol)

    # print(parOrb)
    # print([x.split('/')[1] in parOrb.keys() for x in xovi_amat.sol4_pars])
    # print(np.sum([x.split('/')[1] in parOrb.keys() for x in xovi_amat.sol4_pars]))
    # exit()

    _ = np.hstack((np.reshape(xovi_amat.sol4_pars, (-1, 1)), np.reshape(xovi_amat.sol[0], (-1, 1)),
                   np.reshape(xovi_amat.sol[-1], (-1, 1))))

    # Extract solution for global parameters
    glb_sol = pd.DataFrame(_[[x.split('/')[1] in list(parGlo.keys()) for x in _[:,0]]],columns=['par','sol','std'])

    # Extract solution for orbit parameters
    if np.sum([x.split('/')[1] in parOrb.keys() for x in xovi_amat.sol4_pars]) > 0:
        df_ = pd.DataFrame(_, columns=['key', 'sol', 'std'])
        df_[['orb', 'par']] = df_['key'].str.split('_', expand=True)
        df_.drop('key', axis=1, inplace=True)
        # df_[['orb','par']] = df_[['par','orb']].where(df_['par'] == None, df_[['orb','par']].values)
        df_ = df_.replace(to_replace='None', value=np.nan).dropna()
        table = pd.pivot_table(df_, values=['sol','std'], index=['orb'], columns=['par'], aggfunc=np.sum)
    # print(table)

        if any(xov.xovers.filter(like='dist', axis=1)):
            xov.xovers['dist_max'] = xov.xovers.filter(regex='^dist_.*$').max(axis=1)
            xov.xovers = xov.xovers[xov.xovers.dist_max < 5]
            _ = xov.xovers[['orbA','orbB']].apply(pd.Series.value_counts).sum(axis=1)
            table['num_obs'] = _

            df1 = xov.xovers.groupby(['orbA'], sort=False)['dist_max'].max().reset_index()
            df2 = xov.xovers.groupby(['orbB'], sort=False)['dist_max'].max().reset_index()

            merged_Frame = pd.merge(df1, df2, left_on='orbA', right_on='orbB',how='outer')
            merged_Frame['orbA'] = merged_Frame['orbA'].fillna(merged_Frame['orbB'])
            merged_Frame['orbB'] = merged_Frame['orbB'].fillna(merged_Frame['orbA'])
            merged_Frame['dist_max'] = merged_Frame[['dist_max_x','dist_max_y']].mean(axis=1)
            merged_Frame = merged_Frame[['orbA','dist_max']]
            merged_Frame.columns = ['orb','dist_max']

            table.columns = ['_'.join(col).strip() for col in table.columns.values]
            table = pd.merge(table.reset_index(),merged_Frame,on='orb')
    # print([isinstance(i,tuple) for i in table.columns])
    # print(['_'.join(i) for i in table.columns if isinstance(i,tuple)])
    print('-- Solutions -- ')
    if np.sum([x.split('/')[1] in parOrb.keys() for x in xovi_amat.sol4_pars]) > 0:
        print('Orbit parameters: ')
        print('-- -- -- -- ')
        print(table)
        print('-- -- -- -- ')
    print('-- -- -- -- ')
    print('Global parameters: ')
    print('-- -- -- -- ')
    print(glb_sol)
    print('-- -- -- -- ')

    _ = xov.remove_outliers('dR')
    print(xov.xovers.columns)
    print(xov.xovers.loc[xov.xovers.orbA == '1301022341' ])
    #print(xov.xovers.loc[xov.xovers.orbA == '1301022341' ].sort_values(by='R_A', ascending=True)[:10])
    if np.sum([x.split('/')[1] in parOrb.keys() for x in xovi_amat.sol4_pars]) > 0:
        print(table.reindex(table.sort_values(by='sol_dR/dR', ascending=False).index)[:10])
        print(table.loc[table.orb=='1301022341',:])

    # table[['sol_dR/dA','std_dR/dA','num_obs']] = table[['sol_dR/dA','std_dR/dA','num_obs_']].apply(pd.to_numeric, errors='coerce')
    # table['sol_dR/dA'] = table['sol_dR/dA'] #+ 100
    # fig, ax = plt.subplots()
    # table.plot(x='dist_max', y='sol_dR/dA', yerr='std_dR/dA', style='x')
    #
    # # fig, ax = plt.subplots()
    # # print(df_.dtypes)
    # # df_[['orb','sol']] = df_[['orb','sol']].apply(pd.to_numeric, errors='coerce')
    # # print(df_.dtypes)
    # # df_.groupby('par').plot(x='orb', y='sol', legend=False)
    #
    # plt.savefig('tmp/plotsol.png')

def main(arg):
    print(arg)
    datasets = arg[0]  # ['sim_mlatimes/0res_35amp']
    data_sim = arg[1]

    if data_sim == 'sim':
        _ = [x.split('/')[-2] for x in datasets]
        resval = [10. / 2 ** int(''.join(filter(str.isdigit, strng.split('_')[0]))) for strng in _]
        amplval = [int(''.join(filter(str.isdigit, strng.split('_')[1]))) for strng in _]
    else:
        resval = [0]
        amplval = [0]

    xov_cmb_lst = []

    for ds in datasets:
        data_pth, vecopts = prepro(ds)  # "test/"  # 'small_test/' #'1301/' #)
        print(data_pth)

        xov_cmb = load_combine(data_pth, vecopts)

        # count occurrences for each orbit ID
        _ = xov_cmb.xovers[['orbA','orbB']].apply(pd.Series.value_counts).sum(axis=1)
        print(_)
        print(_.dtypes)
        # exit()

        # solve dataset
        par_list = ['orbA', 'orbB', 'xOvID']
        xovi_amat = prepare_Amat(xov_cmb, vecopts, par_list)
        solve(xovi_amat, ds)
        analyze_sol(xovi_amat,xov_cmb)

        # append to list for stats
        xov_cmb_lst.append(xov_cmb)

    print(len(xov_cmb_lst[0].xovers))

    get_stats(xov_cmb_lst,resval,amplval)

if __name__ == '__main__':

    import sys

    ##############################################
    # launch program and clock
    # -----------------------------
    startT = time.time()

    #print(sys.argv)
    main(sys.argv[1:])

    ##############################################
    # stop clock and print runtime
    # -----------------------------
    endT = time.time()
    print('----- Runtime Amat = ' + str(endT - startT) + ' sec -----' + str(
        (endT - startT) / 60.) + ' min -----')
