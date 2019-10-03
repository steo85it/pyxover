#!/usr/bin/env python3
# ----------------------------------
# AccumXov
# ----------------------------------
# Author: Stefano Bertone
# Created: 04-Mar-2019
#
import warnings
from functools import reduce

import seaborn as sns
from matplotlib import pyplot as plt

from util import mergsum, update_in_alist
from lib.xovres2weights import run

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
from scipy.sparse.linalg import lsqr, lsmr

# mylib
# from mapcount import mapcount
from prOpt import debug, outdir, tmpdir, local, sim_altdata, parOrb, parGlo, partials, sol4_glo, sol4_orb, sol4_orbpar, \
    mean_constr
from xov_setup import xov
from Amat import Amat

sim_altdata = 0

remove_max_dist = True
remove_3sigma_median = False

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

    # modify this selection to use sub-sample of xov only!!
    #------------------------------------------------------
    allFiles = glob.glob(os.path.join(xov_pth, 'xov/xov_*.pkl'))

    # print([xov_pth + 'xov_' + x + '.pkl' for x in misycmb])
    xov_list = [xov_.load(x) for x in allFiles]

    orb_unique = [x.xovers['orbA'].tolist() for x in xov_list if len(x.xovers) > 0]
    orb_unique.extend([x.xovers['orbB'].tolist() for x in xov_list if len(x.xovers) > 0])
    orb_unique = list(set([y for x in orb_unique for y in x]))

    xov_cmb = xov(vecopts)
    xov_cmb.combine(xov_list)

    # save cloop perturbations to xov_cmb
    pertdict = [x.pert_cloop for x in xov_list if hasattr(x, 'pert_cloop')]
    if pertdict != []:
        xov_cmb.pert_cloop = pd.concat([pd.DataFrame(l) for l in pertdict],axis=1).T
    else:
        xov_cmb.pert_cloop = pd.DataFrame()
    #print(len(xov_cmb.xovers))

    return xov_cmb

def get_stats(xov_lst,resval,amplval):
    import seaborn.apionly as sns

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
            if False and remove_max_dist:
                xov.xovers = xov.xovers[xov.xovers.dist_max < 0.4]
                print(len(xov.xovers[xov.xovers.dist_max > 0.4]),
                      'xovers removed by dist from obs > 0.4 km')
            if False and sim_altdata == 0:
                mean_dR, std_dR, worse_tracks = xov.remove_outliers('dR',remove_bad=remove_3sigma_median)

            #print(xov.xovers[['dist_max','dist_avg','dist_minA','dist_minB','dist_min_avg','dR']])
            # checks = ['dist_minA','dist_minB','dist_max','dist_min_avg','dist_avg','dR']
            # for c in checks:
            #     print(c,xov.xovers[c].mean(axis=0),xov.xovers[c].max(axis=0),xov.xovers[c].min(axis=0))
            # _ = xov.xovers.dR.values**2
            # print('RMS',np.sqrt(np.mean(_[~np.isnan(_)],axis=0)))

            # print('dR:',xov.xovers['dR'].mean(axis=0),xov.xovers['dR'].max(axis=0),xov.xovers['dR'].min(axis=0))
            # print(xov.xovers[['dist_max','dR']].abs())

            if debug:
                plt_histo_dR(idx, mean_dR, std_dR, xov.xovers)

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
            print(xov.xovers['orbA'].value_counts().add(xov.xovers['orbB'].value_counts(), fill_value=0).sort_values(ascending=False))
            # print(xov.xovers['orbA'].value_counts()[:5])
            # print(xov.xovers['orbA'].value_counts()[-5:])
            # print(xov.xovers['orbB'].value_counts()[:5])
            # print(xov.xovers['orbB'].value_counts()[-5:])

    #print(len(resval),len(amplval),len(dR_RMS))
    df_ = pd.DataFrame(list(zip(resval,amplval,dR_RMS)), columns=['res','ampl','RMS'])
    #print(df_)
    # create pivot table, days will be columns, hours will be rows
    piv = pd.pivot_table(df_, values="RMS",index=["ampl"], columns=["res"], fill_value=0)
    #plot pivot table as heatmap using seaborn

    fig, ax0 = plt.subplots(nrows=1)
    # ax0.errorbar(range(len(dR_avg)),dR_avg, yerr=dR_std, fmt='-o')
    # ax0.set(xlabel='Exp', ylabel='dR_avg (m)')
    ax0.set_aspect(aspect=0.6)
    ax0 = sns.heatmap(piv, square=False, annot=True, robust=True,
                      cbar_kws={'label': 'RMS (m)','orientation': 'horizontal'}, xticklabels=piv.columns.values.round(2), fmt='.4g')
    ax0.set(xlabel='Topog scale (1st octave, km)',
            ylabel='Topog ampl rms (1st octave, m)')
    fig.savefig(tmpdir+'tst.png')
    plt.clf()
    plt.close()

    #ax0.set_title('variable, symmetric error')
    #print(dR_avg,dR_std,dR_max,dR_min)


def plt_histo_dR(idx, mean_dR, std_dR, xov, xov_ref=''):
    import scipy.stats as stats

    # the histogram of the data
    num_bins = 'auto'
    n, bins, patches = plt.hist(xov.dR.astype(np.float), bins=num_bins, density=True, facecolor='blue', alpha=0.7) #, range=[-100,100])
    # add a 'best fit' line
    # y = stats.norm.pdf(bins, mean_dR, std_dR)
    # plt.plot(bins, y, 'b--')
    if isinstance(xov_ref, pd.DataFrame):
        n, bins, patches = plt.hist(xov_ref.dR.astype(np.float), bins=num_bins, density=True, facecolor='red',
                                    alpha=0.3)  # , range=[-100,100])
    plt.xlabel('dR (m)')
    plt.ylabel('Probability')
    plt.title(r'Histogram of dR: $\mu=' + str(mean_dR) + ', \sigma=' + str(std_dR) + '$')
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig(tmpdir+'/histo_dR_' + str(idx) + '.png')
    plt.clf()

def plt_geo_dR(empty_geomap_df, sol, xov):
    # dR absolute value taken
    xov.xovers['dR_orig'] = xov.xovers.dR
    xov.xovers['dR'] = xov.xovers.dR.abs()
    mladR = xov.xovers.round({'LON': 0, 'LAT': 0, 'dR': 3}).groupby(['LON', 'LAT']).dR.median().reset_index()
    # print(mladR)
    fig, ax1 = plt.subplots(nrows=1)
    # ax0.errorbar(range(len(dR_avg)),dR_avg, yerr=dR_std, fmt='-o')
    # ax0.set(xlabel='Exp', ylabel='dR_avg (m)')
    # ax1 = sns.heatmap(mlacount, square=False, annot=True, robust=True)
    # cmap = sns.palplot(sns.light_palette("green"), as_cmap=True) #sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    piv = pd.pivot_table(mladR, values="dR", index=["LAT"], columns=["LON"], fill_value=0)
    # plot pivot table as heatmap using seaborn
    piv = (piv + empty_geomap_df).fillna(0)
    # print(piv)
    # exit()
    sns.heatmap(piv, xticklabels=10, yticklabels=10)
    plt.tight_layout()
    ax1.invert_yaxis()
    #         ylabel='Topog ampl rms (1st octave, m)')
    fig.savefig(tmpdir+'/mla_dR_' + sol + '.png')
    plt.clf()
    plt.close()

def prepare_Amat(xov, vecopts, par_list=''):

    # xov.xovers = xov.xovers[xov.xovers.orbA=='1301042351']
    # xov.xovers.append(xovtmp[xovtmp.orbA=='1301011544'])
    # xov.xovers.append(xovtmp[xovtmp.orbB=='1301042351'])
    # xov.xovers.append(xovtmp[xovtmp.orbB=='1301011544'])

    # exit()

    clean_xov(xov, par_list)

    xovtmp = xov.xovers.copy()

    # simplify and downsize
    if par_list == '':
        par_list = xov.xovers.columns.filter(regex='^dR.*$')
    df_orig = xov.xovers[par_list]
    df_float = xov.xovers.filter(regex='^dR.*$').apply(pd.to_numeric, errors='ignore', downcast='float')
    xov.xovers = pd.concat([df_orig, df_float], axis=1)
    xov.xovers.info(memory_usage='deep')
    if debug:
        pd.set_option('display.max_columns', 500)
        print(xov.xovers)

    xov.xovers = xovtmp.copy()

    xovi_amat = Amat(vecopts)
    xovi_amat.setup(xov)

    return xovi_amat


def clean_xov(xov, par_list=[]):
    # remove data if xover distance from measurements larger than 5km (interpolation error, if dist cols exist)
    # plus remove outliers with median method
    tmp = xov.xovers.copy()

    # print(tmp[['orbA', 'orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(ascending=False))

    if xov.xovers.filter(regex='^dist_.*$').empty == False:
        xov.xovers['dist_max'] = xov.xovers.filter(regex='^dist_.*$').max(axis=1)

        tmp['dist_minA'] = xov.xovers.filter(regex='^dist_A.*$').min(axis=1)
        tmp['dist_minB'] = xov.xovers.filter(regex='^dist_B.*$').min(axis=1)
        tmp['dist_min_mean'] = tmp.filter(regex='^dist_min.*$').mean(axis=1)
        xov.xovers['dist_min_mean'] = tmp['dist_min_mean'].copy()
        analyze_dist_vs_dR(xov)

        if remove_max_dist:
            print(len(xov.xovers[xov.xovers.dist_max < 0.4]),
              'xovers removed by dist from obs > 0.4km out of ', len(xov.xovers))
            xov.xovers = xov.xovers[xov.xovers.dist_max < 0.4]
            #xov.xovers = xov.xovers[xov.xovers.dist_min_mean < 1]

    if sim_altdata == 0:
        mean_dR, std_dR, worse_tracks = xov.remove_outliers('dR',remove_bad=remove_3sigma_median)
        analyze_dist_vs_dR(xov)

        print("REMOVING ALL XOV dR>200m", len(xov.xovers))
        xov.xovers = xov.xovers[xov.xovers.dR.abs() < 200]
        print('xovers removed by dR > 200m : ', len(xov.xovers))

    # print(xov.xovers[['orbA', 'orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(ascending=False))
    # exit()
    # xov.xovers = tmp.copy()

    return xov


def analyze_dist_vs_dR(xov):
    tmp = xov.xovers.copy()
    tmp['dist_minA'] = xov.xovers.filter(regex='^dist_A.*$').min(axis=1)
    tmp['dist_minB'] = xov.xovers.filter(regex='^dist_B.*$').min(axis=1)
    tmp['dist_min_mean'] = tmp.filter(regex='^dist_min.*$').mean(axis=1)
    tmp.drop(['dist_minA', 'dist_minB'], inplace=True, axis='columns')
    tmp['abs_dR'] = abs(tmp['dR'])
    # print(tmp.nlargest(5, ['abs_dR']))
    # print(tmp.nsmallest(5, ['abs_dR']))
    # exit()
    tmp['tracks'] = tmp.filter(regex='orb?').apply(lambda x: '{}-{}'.format(x[0], x[1]), axis=1)
    # print('dists_df')
    # print(tmp.set_index('tracks').filter(regex='^dist_.*$'))
    print(tmp.set_index('tracks').filter(regex=r'(^dist_.*$|^abs_dR$)').nlargest(5,'abs_dR'))
    print(tmp.set_index('tracks').filter(regex=r'(^dist_.*$|^abs_dR$)').nsmallest(5,'abs_dR'))
    print(tmp.set_index('tracks').filter(regex=r'(^dist_.*$|^abs_dR$)').nlargest(5,'dist_min_mean'))

    # fig, ax = plt.subplots(nrows=1)
    # tmp.plot(x="dist_min_mean", y='abs_dR', ax=ax)
    #
    # # ax.set_xticks(orb_sol.index.values)
    # # ax.locator_params(nbins=10, axis='x')
    # # ax.set_ylabel('sol (m)')
    # # ax.set_ylim(-300,300)
    # plt.savefig(tmpdir + 'tst_dR_dist_' + '.png')
    # plt.close()

    print("corrs", tmp[['dist_min_mean', 'dist_max', 'abs_dR']].corr()) #(method='spearman','kendall'))

def solve(xovi_amat,dataset,previous_iter=None):
    from scipy.sparse import csr_matrix
    from prOpt import par_constr, sol4_orb

    # Solve
    sol4_glo = ['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dL'] #,'dR/dh2'] # [None] # uncomment when on pgda, since prOpt badly read
    sol4_pars = solve4setup(sol4_glo, sol4_orb, sol4_orbpar, xovi_amat.parNames.keys())
    # print(xovi_amat.parNames)
    # for key, value in sorted(xovi_amat.parNames.items(), key=lambda x: x[0]):
    #     print("{} : {}".format(key, value))

    xovi_amat.sol4_pars = sol4_pars

    if sol4_pars != []:
        # select columns of design matrix corresponding to chosen parameters to solve for
        # print([xovi_amat.parNames[p] for p in sol4_pars])
        spA_sol4 = xovi_amat.spA[:,[xovi_amat.parNames[p] for p in sol4_pars]]
        # set b=0 for rows not involving chosen set of parameters
        nnz_per_row = spA_sol4.getnnz(axis=1)
        xovi_amat.b[np.where(nnz_per_row == 0)[0]] = 0
    else:
        spA_sol4 = xovi_amat.spA

    print("sol4pars:", np.array(sol4_pars))

    if debug and len(sol4_pars)<50:
        print('dense A', spA_sol4.todense())
        print('B', xovi_amat.b)
        print('max B', xovi_amat.b.max(),xovi_amat.b.mean())

        # Compute the covariance matrix
        print(np.linalg.pinv((spA_sol4.transpose()*spA_sol4).todense()))
        # compute sol
        print('sol dense',np.linalg.lstsq(spA_sol4.todense(), xovi_amat.b, rcond=1))

    A = spA_sol4.transpose()*spA_sol4
    b = spA_sol4.transpose()*(csr_matrix(xovi_amat.b).transpose())

    #set up observation weights (according to local roughness and dist of obs from xover point)
    regbas_weights = run(xovi_amat.xov).reset_index()
    # print(regbas_weights)

    # take sqrt of inverse of roughness value at min dist of xover from neighb obs as weight
    val = np.sqrt(1./np.abs(regbas_weights.rough_at_mindist.values))
    # print(np.max(np.abs(val)))
    # val /= np.max(np.abs(val))
    row = col = regbas_weights.index.values

    obs_weights = csr_matrix((np.ones(len(val)), (row, col)), dtype=np.float32, shape=(len(regbas_weights), len(regbas_weights)))
    #obs_weights = csr_matrix((val, (row, col)), dtype=np.float32, shape=(len(regbas_weights), len(regbas_weights)))

    # Cholesky decomposition of diagonal matrix == square root of diagonal
    L = obs_weights

    # apply weights
    spA_sol4 = L * spA_sol4

    # select constrains for processed parameters (TODO should go in sol4pars)
    mod_par = [your_key.split('_')[1] if len(your_key.split('_'))>1 else your_key for your_key in sol4_pars ]
    par_constr = { your_key: par_constr[your_key] for your_key in mod_par }

    csr = []
    for constrain in par_constr.items():

        # print(constrain)
        # print(constrain[0])
        # print(len(sol4_pars))
        parindex = np.array([[idx,constrain[1]] for idx,p in enumerate(sol4_pars) if constrain[0] in p])

        # Constrain tightly to 0 those parameters with few observations
        nobs_tracks = xovi_amat.xov.xovers[['orbA', 'orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(
            ascending=False)
        to_constrain = [idx for idx, p in enumerate(sol4_pars) if p.split('_')[0] in nobs_tracks[nobs_tracks < 10].index]
        for p in parindex:
            if p[0] in to_constrain:
                p[1] *= 1.e10
        #print(parindex)
        val = parindex[:,1]
        row = col = parindex[:,0]

        csr.append(
                csr_matrix((1/val, (row, col)), dtype=np.float32, shape=(len(sol4_pars), len(sol4_pars))))
    penalty_matrix = sum(csr)

    # Add constrain to mean value of parameters
    penalty_par_mean = np.identity(len(sol4_pars)) - 1/len(sol4_pars) * np.ones((len(sol4_pars), len(sol4_pars)))
    penalty_matrix += 1/mean_constr * csr_matrix(penalty_par_mean)

    # Choleski decompose matrix and append to design matrix
    Q = np.linalg.cholesky(penalty_matrix.todense())
    if previous_iter != None:
        b_penal = np.hstack([L*xovi_amat.b, np.zeros(len(sol4_pars))]) #np.ravel(np.dot(Q,previous_iter.sol[0]))])
    else:
        b_penal = np.hstack([L*xovi_amat.b, np.zeros(len(sol4_pars))])
    import scipy
    spA_sol4_penal = scipy.sparse.vstack([spA_sol4,csr_matrix(Q)])

    print("Pre-sol: len(A,b)=",len(b_penal),spA_sol4_penal.shape)

    xovi_amat.sol = lsqr(spA_sol4_penal, b_penal,show=False,iter_lim=10000,atol=1.e-9,btol=1.e-9,calc_var=True)
    # print(xovi_amat.sol[0])

    # _ = lsmr(A+penalty_matrix,b.toarray(),show=False, maxiter=5000)
    # # print(np.diag(np.linalg.pinv(((A+penalty_matrix).transpose() * (A+penalty_matrix)).todense())))
    # # print(xovi_amat.sol)
    # xovi_amat.sol = (_[0], *xovi_amat.sol[1:])

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
    elif sol4_glo == [None]:
        sol4_glo = []

    sol4_pars = sorted(sol4_orb) + sorted(sol4_glo)

    print('solving for:',np.array(sol4_pars))

    return sol4_pars


def analyze_sol(xovi_amat,xov):
    # print('xovi_amat.sol',xovi_amat.sol)

    # print(parOrb)
    # print([x.split('/')[1] in parOrb.keys() for x in xovi_amat.sol4_pars])
    # print(np.sum([x.split('/')[1] in parOrb.keys() for x in xovi_amat.sol4_pars]))
    # exit()

    print(len(np.reshape(xovi_amat.sol4_pars, (-1, 1))),len(np.reshape(xovi_amat.sol[0], (-1, 1))),
                              len(np.reshape(xovi_amat.sol[-1], (-1, 1))) )
    
    _ = np.hstack((np.reshape(xovi_amat.sol4_pars, (-1, 1)), np.reshape(xovi_amat.sol[0], (-1, 1)),
                   np.reshape(xovi_amat.sol[-1], (-1, 1))))
    sol_dict = {'sol': dict(zip(_[:,0],_[:,1].astype(float))), 'std': dict(zip(_[:,0],_[:,2].astype(float))) }

    # print(_)
    # print(np.hstack((np.reshape(xovi_amat.sol4_pars, (-1, 1)), np.reshape([xovi_amat.parNames[p] for p in xovi_amat.sol4_pars], (-1, 1)), np.reshape(xovi_amat.sol[0], (-1, 1)),
    #                np.reshape(xovi_amat.sol[-1], (-1, 1)))))
    # exit()

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

        if False and any(xov.xovers.filter(like='dist', axis=1)):
            xov.xovers['dist_max'] = xov.xovers.filter(regex='^dist_.*$').max(axis=1)
            tmp = xov.xovers.copy()
            tmp['dist_minA'] = xov.xovers.filter(regex='^dist_A.*$').min(axis=1)
            tmp['dist_minB'] = xov.xovers.filter(regex='^dist_B.*$').min(axis=1)
            tmp['dist_min_mean'] = tmp.filter(regex='^dist_min.*$').mean(axis=1)
            xov.xovers['dist_min_mean'] = tmp['dist_min_mean'].copy()

            if remove_max_dist:
                xov.xovers = xov.xovers[xov.xovers.dist_max < 0.4]
                #xov.xovers = xov.xovers[xov.xovers.dist_min_mean < 1]

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

        orb_sol = table

    else:
        orb_sol = pd.DataFrame()

    # print([isinstance(i,tuple) for i in table.columns])
    # print(['_'.join(i) for i in table.columns if isinstance(i,tuple)])

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
    return orb_sol, glb_sol, sol_dict


def print_sol(orb_sol, glb_sol, xov, xovi_amat):

    print('-- Solutions -- ')
    if np.sum([x.split('/')[1] in parOrb.keys() for x in xovi_amat.sol4_pars]) > 0:
        print('Orbit parameters: ')
        print('-- -- -- -- ')
        print(orb_sol)
        print('-- -- -- -- ')
    print('-- -- -- -- ')
    print('Global parameters: ')
    print('-- -- -- -- ')
    print(glb_sol)
    print('-- -- -- -- ')

    if debug:
        _ = xov.remove_outliers('dR',remove_bad=remove_3sigma_median)
        print(xov.xovers.columns)
        print(xov.xovers.loc[xov.xovers.orbA == '1301022345'])
        # print(xov.xovers.loc[xov.xovers.orbA == '1301022341' ].sort_values(by='R_A', ascending=True)[:10])
        if np.sum([x.split('/')[1] in parOrb.keys() for x in xovi_amat.sol4_pars]) > 0:
            print(orb_sol.reindex(orb_sol.sort_values(by='sol_dR/dR', ascending=False).index)[:10])
            print(orb_sol.loc[orb_sol.orb == '1301022345', :])


def main(arg):

    startT = time.time()

    print(arg)
    datasets = arg[0]  # ['sim_mlatimes/0res_35amp']
    data_sim = arg[1]
    ext_iter = arg[2]

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
        xov_cmb.nobs_x_track = xov_cmb.xovers[['orbA','orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(ascending=False)
        # print(_)
        # print(_.dtypes)
        # exit()

        if partials == 1:

            # retrieve old solution
            if int(ext_iter) > 0:
                previous_iter = Amat(vecopts)
                # tmp = tmp.load((data_pth + 'Abmat_' + ds.split('/')[0] + '_' + ds.split('/')[1][:-1] + str(ext_iter) + '_' + ds.split('/')[2]) + '.pkl')
                previous_iter = previous_iter.load(('_').join((outdir + ('/').join(ds.split('/')[:-2])).split('_')[:-1]) +
                                '_' + str(ext_iter - 1) + '/' +
                               ds.split('/')[-2] + '/Abmat_' + ('_').join(ds.split('/')[:-1]) + '.pkl')
            else:
                previous_iter = None

            # solve dataset
            par_list = ['orbA', 'orbB', 'xOvID']
            xovi_amat = prepare_Amat(xov_cmb, vecopts, par_list)
            solve(xovi_amat, ds, previous_iter)

            # Save to pkl
            orb_sol, glb_sol, sol_dict = analyze_sol(xovi_amat,xov_cmb)
            xovi_amat.sol_dict = sol_dict
            print("Sol for iter ", str(ext_iter))
            print_sol(orb_sol, glb_sol, xov, xovi_amat)

            # Cumulate with solution from previous iter
            if int(ext_iter) > 0:
                # sum the values with same keys
                updated_sol = mergsum(xovi_amat.sol_dict['sol'],previous_iter.sol_dict['sol'])
                xovi_amat.sol_dict = {'sol': updated_sol, 'std' : xovi_amat.sol_dict['std']}
                xovi_amat.sol = (list(xovi_amat.sol_dict['sol'].values()), *xovi_amat.sol[1:])
                orb_sol, glb_sol, sol_dict = analyze_sol(xovi_amat, xov_cmb)
                print("Cumulated solution")
                print_sol(orb_sol, glb_sol, xov, xovi_amat)
            if len(ds.split('/'))>2:
                xovi_amat.save(('_').join((data_pth + 'Abmat_' + ds.split('/')[0] + '_' +
                                           ds.split('/')[1]).split('_')[:-1])+'_'+ str(ext_iter+1) +
                               '_' + ds.split('/')[2] + '.pkl')
            else:
                xovi_amat.save((data_pth + 'Abmat_' + ds.split('/')[0] + '_' + ds.split('/')[1] + '_')[:-1] + str(ext_iter+1) + '.pkl')

        else:
            # clean only
            clean_xov(xov_cmb, '')
            # plot histo and geo_dist
            tstname = [x.split('/')[-3] for x in datasets][0]
            mean_dR, std_dR, worst_tracks = xov_cmb.remove_outliers('dR',remove_bad=remove_3sigma_median)
            plt_histo_dR(tstname, mean_dR, std_dR,
                         xov_cmb.xovers)  # [tmp.xov.xovers.orbA.str.contains('14', regex=False)])

            empty_geomap_df = pd.DataFrame(0, index=np.arange(0, 91),
                                           columns=np.arange(-180, 181))
            plt_geo_dR(empty_geomap_df, tstname, xov_cmb)


        # append to list for stats
        xov_cmb_lst.append(xov_cmb)

    print("len xov_cmb ", len(xov_cmb_lst[0].xovers))

    if local:
        get_stats(xov_cmb_lst,resval,amplval)

    endT = time.time()
    print('----- Runtime Amat = ' + str(endT - startT) + ' sec -----' + str(
        (endT - startT) / 60.) + ' min -----')

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
