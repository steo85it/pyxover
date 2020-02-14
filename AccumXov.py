#!/usr/bin/env python3
# ----------------------------------
# AccumXov
# ----------------------------------
# Author: Stefano Bertone
# Created: 04-Mar-2019
#
import re
import warnings
# from functools import reduce
import itertools

import seaborn as sns

from accum_utils import get_xov_cov_tracks
from util import mergsum, update_in_alist, rms
from lib.xovres2weights import get_interpolation_weight
# from xov_utils import get_tracks_rms

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import identity, csr_matrix, diags
import scipy
# from itertools import izip, count
# from geopy.distance import vincenty

# from collections import defaultdict
# import mpl_toolkits.basemap as basemap

import time
from scipy.sparse.linalg import lsqr, lsmr
import scipy.sparse.linalg as spla
import scipy.linalg as la

# mylib
# from mapcount import mapcount
from prOpt import debug, outdir, tmpdir, local, sim_altdata, parOrb, parGlo, partials, sol4_glo, sol4_orb, sol4_orbpar, \
    mean_constr, pert_cloop, OrbRep
from xov_setup import xov
from Amat import Amat

sim_altdata = 0

remove_max_dist = False
remove_3sigma_median = False
remove_dR200 = False
# only applied if the above ones are false
clean_part = True
huber_threshold = 30
distmax_threshold = 0.2
offnad_threshold = 2
h2_limit_on = False
# rescaling factor for weight matrix, based on average error on xovers at Mercury
# dimension of meters (to get s0/s dimensionless)
# could be updated by checking chi2 or by VCE
sigma_0 = 1.e-2 * 2. # * 0.85 # 0.16 #

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
    xov_list = [xov_.load(x) for x in allFiles[:]]

    # orb_unique = [x.xovers['orbA'].tolist() for x in xov_list if len(x.xovers) > 0]
    # orb_unique.extend([x.xovers['orbB'].tolist() for x in xov_list if len(x.xovers) > 0])
    # orb_unique = list(set([y for x in orb_unique for y in x]))

    xov_cmb = xov(vecopts)
    xov_cmb.combine(xov_list)

    # save cloop perturbations to xov_cmb
    pertdict = [x.pert_cloop for x in xov_list if hasattr(x, 'pert_cloop')]
    if pertdict != []:
        xov_cmb.pert_cloop = pd.concat([pd.DataFrame(l) for l in pertdict],axis=1,sort=True).T
    else:
        xov_cmb.pert_cloop = pd.DataFrame()

    # save initial cloop perturbations to xov_cmb
    pertdict = [x.pert_cloop_0 for x in xov_list if hasattr(x, 'pert_cloop_0')]
    if len([v for x in pertdict for k,v in x.items() if v]) > 0 and sol4_orbpar != [None]:
        pertdict = {k: v for x in pertdict for k, v in x.items() if v is not None}
        xov_cmb.pert_cloop_0 = pd.DataFrame(pertdict).T
        xov_cmb.pert_cloop_0.drop_duplicates(inplace=True)
    else:
        xov_cmb.pert_cloop_0 = pd.DataFrame()

    return xov_cmb

def get_stats(amat):
    # import seaborn.apionly as sns

    # print('resval,amplval', resval, amplval)
    # #print(xov_cmb.xovers)
    #
    # dR_avg = []
    # dR_std = []
    # dR_max = []
    # dR_min = []
    # dR_RMS = []
    #
    # for idx, xov in enumerate(xov_lst):
    #
    #     #print(idx, xov)
    #
    #     if len(xov.xovers)>0:
    #         xov.xovers['dist_avg'] = xov.xovers.filter(regex='^dist_[A,B].*$').mean(axis=1)
    #         xov.xovers['dist_max'] = xov.xovers.filter(regex='^dist_[A,B].*$').max(axis=1)
    #         xov.xovers['dist_minA'] = xov.xovers.filter(regex='^dist_A.*$').min(axis=1)
    #         xov.xovers['dist_minB'] = xov.xovers.filter(regex='^dist_B.*$').min(axis=1)
    #         xov.xovers['dist_min_avg'] = xov.xovers.filter(regex='^dist_min.*$').mean(axis=1)
    #
    #         # remove data if xover distance from measurements larger than 5km (interpolation error)
    #         # plus remove outliers with median method
    #         if remove_max_dist:
    #             if debug:
    #                 print(xov.xovers.filter(regex='^dist_.*$'))
    #             print(len(xov.xovers[xov.xovers.dist_max > 0.4]),
    #                   'xovers removed by dist from obs > 0.4 km out of ',
    #                   len(xov.xovers),", or ",
    #                   len(xov.xovers[xov.xovers.dist_max > 0.4])/len(xov.xovers)*100.,'%')
    #             xov.xovers = xov.xovers[xov.xovers.dist_max < 0.4]
    #         if sim_altdata == 0:
    #             mean_dR, std_dR, worse_tracks = xov.remove_outliers('dR',remove_bad=remove_3sigma_median)
    #
    #         # weight residuals
    #         if not remove_max_dist and not remove_3sigma_median and not remove_dR200:
    #             xov.xovers['dR'] *= amat.weights
    #         # exit()
    #         #print(xov.xovers[['dist_max','dist_avg','dist_minA','dist_minB','dist_min_avg','dR']])
    #         # checks = ['dist_minA','dist_minB','dist_max','dist_min_avg','dist_avg','dR']
    #         # for c in checks:
    #         #     print(c,xov.xovers[c].mean(axis=0),xov.xovers[c].max(axis=0),xov.xovers[c].min(axis=0))
    #         # _ = xov.xovers.dR.values**2
    #         # print('RMS',np.sqrt(np.mean(_[~np.isnan(_)],axis=0)))
    #
    #         # print('dR:',xov.xovers['dR'].mean(axis=0),xov.xovers['dR'].max(axis=0),xov.xovers['dR'].min(axis=0))
    #         # print(xov.xovers[['dist_max','dR']].abs())
    #         #TODO update, not very useful
    #         if debug and local and False:
    #             plt_histo_dR(idx, mean_dR, std_dR, xov.xovers)
    #
    #             xov.xovers[['dist_max', 'dR']].abs().plot(kind='scatter', x='dist_max', y='dR')
    #             plt.savefig('tmp/dR_vs_dist_' + str(idx) + '.png')
    #             plt.clf()
    #
    #             xov.xovers['dist_avg'].plot()
    #             plt.savefig('tmp/dist_' + str(idx) + '.png')
    #             plt.close()
    #
    #         dR_avg.append(xov.xovers.dR.mean(axis=0))
    #         dR_std.append(xov.xovers.dR.std(axis=0))
    #         dR_max.append(xov.xovers.dR.max(axis=0))
    #         dR_min.append(xov.xovers.dR.min(axis=0))
    # xover residuals
    w = amat.xov.xovers['dR'].values
    nobs = len(w)
    npar = len(amat.sol_dict['sol'].values())

    lTP = w.reshape(1, -1) @ amat.weights
    lTPl = lTP @ w.reshape(-1, 1)

    xsol = []
    xstd = []
    for filt in amat.sol4_pars:
        filtered_dict = {k: v for (k, v) in amat.sol_dict['sol'].items() if filt in k}
        xsol.append(list(filtered_dict.values())[0])
        filtered_dict = {k: v for (k, v) in amat.sol_dict['std'].items() if filt in k}
        xstd.append(list(filtered_dict.values())[0])

    print("check sol4pars",amat.parNames,amat.sol4_pars,amat.parNames==amat.sol4_pars)

    if amat.sol4_pars != []:
        # select columns of design matrix corresponding to chosen parameters to solve for
        # print([xovi_amat.parNames[p] for p in sol4_pars])
        amat.spA = amat.spA[:, [amat.parNames[p] for p in amat.parNames.keys()]]
        # set b=0 for rows not involving chosen set of parameters
        nnz_per_row = amat.spA.getnnz(axis=1)
        amat.b[np.where(nnz_per_row == 0)[0]] = 0

    xT = np.array(xsol).reshape(1, -1)
    ATP = amat.spA.T * amat.weights
    ATPb = ATP * amat.b

    # when iterations have inconsistent npars, some are imported from previous sols for
    # consistency reasons. Here we remove these values from the computation of vTPv
    # else matrix shapes don't match
    if len(ATPb) != len(xT):
        print("removing stuff...")
        missing = [x for x in amat.sol4_pars if x not in amat.parNames]
        missing = [amat.sol4_pars.index(x) for x in missing]
        xT = np.delete(xT, missing)
    ##################
    print(ATPb.shape)
    print(xT.shape)
    vTPv = lTPl - xT@ATPb
    print("pre-RMS=",np.sqrt(lTPl/(nobs-npar))," post-RMS=",np.sqrt(vTPv/(nobs-npar)))

    # degrees if freedom in case of constrained least square
    # Atmp = (ATP@amat.spA + amat.penalty_mat)
    # trR = np.diagonal(np.linalg.pinv(Atmp.todense())@ATP@amat.spA).sum()
    # dof = nobs - trR
    # # print("check deg of freedom",npar,trR)
    # m0 = np.sqrt(vTPv/dof)

    # alternative method to compute chi2 for constrained least-square (not involving inverse)
    xTlP = xT@amat.penalty_mat
    xTlPx = xTlP@xT.T
    m0 = np.sqrt((vTPv+xTlPx)/nobs)

    print("Weighted a-posteriori RMS is ", m0, " - chi2 = ", m0/sigma_0)

    if local and debug:
        plt.figure()  # figsize=(8, 3))
        num_bins = 200 # 'auto'  #
        n, bins, patches = plt.hist((amat.weights@(np.abs(w).reshape(-1, 1))).astype(np.float), bins=num_bins, cumulative=-1, range=[0.1,50.])
        # n, bins, patches = plt.hist(w.astype(np.float), bins=num_bins, cumulative=True)
        # plt.xlabel('roughness@baseline700 (m/m)')
        plt.savefig(tmpdir + '/histo_residuals.png')
        plt.clf()

    # _ = xov.xovers.dR.values ** 2
    # dR_RMS.append(np.sqrt(np.mean(_[~np.isnan(_)], axis=0)))
    # print(np.count_nonzero(np.isnan(xov.xovers.dR.values**2)))

    if False and debug:
        print("xov_xovers_value_count:")
        nobs_tracks = xov.xovers[['orbA', 'orbB']].apply(pd.Series.value_counts).sum(
            axis=1).sort_values(
            ascending=False)
        print(nobs_tracks)

    # df_ = pd.DataFrame(list(zip(resval,amplval,dR_weighted_RMS)), columns=['res','ampl','RMS'])
    # print("Total RMS: ", df_.RMS.values)
    # # create pivot table, days will be columns, hours will be rows
    # piv = pd.pivot_table(df_, values="RMS",index=["ampl"], columns=["res"], fill_value=0)
    # #plot pivot table as heatmap using seaborn
    #
    # if local == 1 and debug:
    #   fig, ax0 = plt.subplots(nrows=1)
    #   ax0.set_aspect(aspect=0.6)
    #   ax0 = sns.heatmap(piv, square=False, annot=True, robust=True,
    #                   cbar_kws={'label': 'RMS (m)','orientation': 'horizontal'}, xticklabels=piv.columns.values.round(2), fmt='.4g')
    #   ax0.set(xlabel='Topog scale (1st octave, km)',
    #         ylabel='Topog ampl rms (1st octave, m)')
    #   fig.savefig(tmpdir+'tst.png')
    #   plt.clf()
    #   plt.close()

def plt_histo_dR(idx, mean_dR, std_dR, xov, xov_ref=''):
    import scipy.stats as stats

    xlim = 50

    plt.figure(figsize=(8,3))
    plt.xlim(-1.*xlim, xlim)
    # the histogram of the data
    num_bins = 200 # 'auto'
    n, bins, patches = plt.hist(xov.dR.astype(np.float), bins=num_bins, density=True, facecolor='blue',
    alpha=0.7, range=[-1.*xlim, xlim])
    # add a 'best fit' line
    # y = stats.norm.pdf(bins, mean_dR, std_dR)
    # plt.plot(bins, y, 'b--')
    if isinstance(xov_ref, pd.DataFrame):
        n, bins, patches = plt.hist(xov_ref.dR.astype(np.float), bins=num_bins, density=True, facecolor='red',
                                    alpha=0.3, range=[-1.*xlim, xlim])
    plt.xlabel('dR (m)')
    plt.ylabel('Probability')
    plt.title(r'Histogram of dR: $\mu=' + str(mean_dR) + ', \sigma=' + str(std_dR) + '$')
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig(tmpdir+'/histo_dR_' + str(idx) + '.png')
    plt.clf()

def plt_geo_dR(sol, xov):
    # dR absolute value taken
    xov.xovers['dR_orig'] = xov.xovers.dR
    xov.xovers['dR'] = xov.xovers.dR.abs()
    # print(xov.xovers.LON.max(),xov.xovers.LON.min())
    mladR = xov.xovers.round({'LON': 0, 'LAT': 0, 'dR': 3}).groupby(['LON', 'LAT']).dR.median().reset_index()
    # print(mladR.LON.max(),mladR.LON.min())
    # exit()
    fig, ax1 = plt.subplots(nrows=1)
    # ax0.errorbar(range(len(dR_avg)),dR_avg, yerr=dR_std, fmt='-o')
    # ax0.set(xlabel='Exp', ylabel='dR_avg (m)')
    # ax1 = sns.heatmap(mlacount, square=False, annot=True, robust=True)
    # cmap = sns.palplot(sns.light_palette("green"), as_cmap=True) #sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    piv = pd.pivot_table(mladR, values="dR", index=["LAT"], columns=["LON"], fill_value=0)
    # plot pivot table as heatmap using seaborn

    #piv = (piv + empty_geomap_df).fillna(0)
    # print(piv)
    # exit()
    sns.heatmap(piv, xticklabels=10, yticklabels=10)
    plt.tight_layout()
    ax1.invert_yaxis()
    #         ylabel='Topog ampl rms (1st octave, m)')
    fig.savefig(tmpdir+'/mla_dR_' + sol + '.png')
    plt.clf()
    plt.close()

# @profile
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
    df_float = xov.xovers.filter(regex='^dR.*$').apply(pd.to_numeric, errors='ignore') #, downcast='float')
    xov.xovers = pd.concat([df_orig, df_float], axis=1)
    xov.xovers.info(memory_usage='deep')
    if debug:
        pd.set_option('display.max_columns', 500)
        print(xov.xovers)

    if OrbRep in ['lin','quad']:
        xovtmp = xov.upd_orbrep(xovtmp)
        # print(xovi_amat.xov.xovers)
        xov.parOrb_xy = xovtmp.filter(regex='^dR/[a-zA-Z0-9]+_.*$').columns.values

    xovi_amat = Amat(vecopts)

    xov.xovers = xovtmp.copy()

    xovi_amat.setup(xov)

    return xovi_amat


def clean_xov(xov, par_list=[]):
    # remove data if xover distance from measurements larger than 5km (interpolation error, if dist cols exist)
    # plus remove outliers with median method
    tmp = xov.xovers.copy()

    # print(tmp[['orbA', 'orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(ascending=False))

    if xov.xovers.filter(regex='^dist_[A,B].*$').empty == False:
        xov.xovers['dist_max'] = xov.xovers.filter(regex='^dist_[A,B].*$').max(axis=1)

        tmp['dist_minA'] = xov.xovers.filter(regex='^dist_A.*$').min(axis=1)
        tmp['dist_minB'] = xov.xovers.filter(regex='^dist_B.*$').min(axis=1)
        tmp['dist_min_mean'] = tmp.filter(regex='^dist_min[A,B].*$').mean(axis=1)
        xov.xovers['dist_min_mean'] = tmp['dist_min_mean'].copy()
        analyze_dist_vs_dR(xov)

        if remove_max_dist:
            print(len(xov.xovers[xov.xovers.dist_max < 0.4]),
              'xovers removed by dist from obs > 0.4km out of ', len(xov.xovers))
            xov.xovers = xov.xovers[xov.xovers.dist_max < 0.4]
            #xov.xovers = xov.xovers[xov.xovers.dist_min_mean < 1]

    # if sim_altdata == 0:
    #     mean_dR, std_dR, worse_tracks = xov.remove_outliers('dR',remove_bad=remove_3sigma_median)
    #     analyze_dist_vs_dR(xov)
    #
    #     if remove_dR200:
    #         print("REMOVING ALL XOV dR>200m", len(xov.xovers))
    #         xov.xovers = xov.xovers[xov.xovers.dR.abs() < 200]
    #         print('xovers after cleaning by dR > 200m : ', len(xov.xovers))

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
    ## This filter takes a huge amount of time!!!
    # tmp['tracks'] = tmp.filter(regex='orb?').apply(lambda x: '{}-{}'.format(x[0], x[1]), axis=1)
    # # print('dists_df')
    # # print(tmp.set_index('tracks').filter(regex='^dist_.*$'))
    # print(tmp.set_index('tracks').filter(regex=r'(^dist_.*$|^abs_dR$)').nlargest(5,'abs_dR'))
    # print(tmp.set_index('tracks').filter(regex=r'(^dist_.*$|^abs_dR$)').nsmallest(5,'abs_dR'))
    # print(tmp.set_index('tracks').filter(regex=r'(^dist_.*$|^abs_dR$)').nlargest(5,'dist_min_mean'))
    #
    # # fig, ax = plt.subplots(nrows=1)
    # # tmp.plot(x="dist_min_mean", y='abs_dR', ax=ax)
    # #
    # # # ax.set_xticks(orb_sol.index.values)
    # # # ax.locator_params(nbins=10, axis='x')
    # # # ax.set_ylabel('sol (m)')
    # # # ax.set_ylim(-300,300)
    # # plt.savefig(tmpdir + 'tst_dR_dist_' + '.png')
    # # plt.close()
    #
    # print("corrs", tmp[['dist_min_mean', 'dist_max', 'abs_dR']].corr()) #(method='spearman','kendall'))

# @profile
def solve(xovi_amat,dataset, previous_iter=None):
    from scipy.sparse import csr_matrix, identity
    from prOpt import par_constr, mean_constr, sol4_orb, sol4_glo

    # Solve
    if not local:
        sol4_glo = ['dR/dRA', 'dR/dDEC', 'dR/dPM','dR/dL'] #,'dR/dh2'] # [None] # used on pgda, since prOpt badly read
    sol4_pars = solve4setup(sol4_glo, sol4_orb, sol4_orbpar, xovi_amat.parNames.keys())
    # print(xovi_amat.parNames)
    # for key, value in sorted(xovi_amat.parNames.items(), key=lambda x: x[0]):
    #     print("{} : {}".format(key, value))

    if OrbRep  in ['lin','quad']:
        # xovi_amat.xov.xovers = xovi_amat.xov.upd_orbrep(xovi_amat.xov.xovers)
        # print(xovi_amat.xov.xovers)
        regex = re.compile(".*_dR/d[A,C,R]$")
        const_pars = [x for x in sol4_pars if not regex.match(x)]
        sol4_pars = [x+str(y) for y in range(2) for x in list(filter(regex.match, sol4_pars))]
        sol4_pars.extend(const_pars)

    xovi_amat.sol4_pars = sol4_pars
    # exit()

    if sol4_pars != []:
        # select columns of design matrix corresponding to chosen parameters to solve for
        # print([xovi_amat.parNames[p] for p in sol4_pars])
        spA_sol4 = xovi_amat.spA[:,[xovi_amat.parNames[p] for p in sol4_pars]]
        # set b=0 for rows not involving chosen set of parameters
        nnz_per_row = spA_sol4.getnnz(axis=1)
        xovi_amat.b[np.where(nnz_per_row == 0)[0]] = 0
    else:
        spA_sol4 = xovi_amat.spA

    # print("sol4pars:", np.array(sol4_pars))
    # print(spA_sol4)

    # screening of partial derivatives (downweights data)
    nglbpars = len([i for i in sol4_glo if i])
    if nglbpars>0 and clean_part:
        xovi_amat.b, spA_sol4 = clean_partials(xovi_amat.b, spA_sol4, threshold = 1.e6,nglbpars=nglbpars)
        # pass

    # WEIGHTING TODO refactor to separate method

    # compute huber weights (1 if x<huber_threshold, (huber_threshold/abs(dR))**2 if abs(dR)>huber_threshold)
    if not remove_max_dist and not remove_3sigma_median and not remove_dR200:
        tmp = xovi_amat.xov.xovers.dR.abs().values
        huber_weights = np.where(tmp>huber_threshold, (huber_threshold/tmp)**1, 1.)

    if debug and not remove_max_dist and not remove_3sigma_median and not remove_dR200:
        print("Apply Huber weights (resid)")
        print(tmp[tmp>huber_threshold])
        print(np.sort(huber_weights[huber_weights<1.]),np.mean(huber_weights))

    # same but w.r.t. distance
    if not remove_max_dist and not remove_3sigma_median and not remove_dR200:
        tmp = xovi_amat.xov.xovers.dist_max.values
        huber_weights_dist = np.where(tmp>distmax_threshold, (distmax_threshold/tmp)**2, 1.)

    if debug and not remove_max_dist and not remove_3sigma_median and not remove_dR200:
        print("Apply Huber weights (dist)")
        print(tmp[tmp>distmax_threshold])
        print(np.sort(huber_weights_dist[huber_weights_dist<1.]),np.mean(huber_weights_dist))

    # same but w.r.t. offnadir
    if not remove_max_dist and not remove_3sigma_median and not remove_dR200:
        tmp = np.nan_to_num(xovi_amat.xov.xovers.filter(regex='offnad').values)
        tmp = np.max(np.abs(tmp),axis=1)
        huber_weights_offnad = np.where(tmp>offnad_threshold, (offnad_threshold/tmp)**1, 1.)

    if debug and not remove_max_dist and not remove_3sigma_median and not remove_dR200:
        print("Apply Huber weights (offnad)")
        print(tmp[tmp>offnad_threshold])
        print(len(huber_weights_offnad[huber_weights_offnad<1.]),len(huber_weights_offnad))
        print(np.sort(huber_weights_offnad[huber_weights_offnad<1.]),np.mean(huber_weights_offnad))

    # combine weights
    if not remove_max_dist and not remove_3sigma_median and not remove_dR200:
        tmp=huber_weights*huber_weights_dist*huber_weights_offnad
        huber_penal = tmp
        # should use weights or measurement error threshold, but using huber-threshold-like criteria for now
        # to mimic what I was doing without weights
        xovi_amat.xov.xovers['huber'] = huber_penal

    # get quality of tracks and apply huber weights
    if not remove_max_dist and not remove_3sigma_median and not remove_dR200:
        tmp = xovi_amat.xov.xovers.copy()[['xOvID','LON', 'LAT', 'dtA', 'dR', 'orbA', 'orbB', 'huber']]#.astype('float16')
        print("pre xovcov types",tmp.dtypes)
        weights_xov_tracks = get_xov_cov_tracks(df=tmp,plot_stuff=True)

        # the histogram of weight distribution
        if False and local and debug:
            tmp = weights_xov_tracks.diagonal()

            plt.figure() #figsize=(8, 3))
            num_bins = 100 # 'auto'  # 40  # v
            n, bins, patches = plt.hist(tmp.astype(np.float), bins=num_bins)
            plt.xlabel('dR (m)')
            plt.ylabel('# tracks')
            plt.savefig(tmpdir + '/histo_tracks_weights.png')
            plt.clf()

        # xovi_amat.xov.xovers['huber'] *= huber_weights_track

        if debug and False:
            tmp['track_weights'] = weights_xov_tracks.diagonal()
            tmp = tmp[['orbA', 'orbB', 'dR', 'track_weights']]
            print(tmp[tmp.dR.abs() < 0.5].sort_values(by='track_weights'))

#######
        # additional for h2 tests
        if h2_limit_on:
            # cut based on residuals
            limit_h2 = 20.
            tmp = xovi_amat.xov.xovers.dR.abs().values
            tmp = np.where(tmp > limit_h2, (limit_h2 / tmp) ** 4, 1.)
            huber_penal *= tmp
            # cut based on mean min separation
            limit_h2_sep = 10.*1.e-3 # km based
            tmp = xovi_amat.xov.xovers.dist_min_mean.values
            tmp = np.where(tmp > limit_h2_sep, (limit_h2_sep / tmp) ** 4, 1.)
            huber_penal *= tmp

            if debug and local:
                num_bins = 100 #'auto'
                plt.clf()
                n, bins, patches = plt.hist(np.where(huber_penal < 1., huber_penal, 1.).astype(np.float), bins=num_bins,cumulative=True)
                plt.xlabel('huber_penal')
                plt.savefig(tmpdir + '/histo_huber_h2.png')
                plt.clf()

#######
    # interp_weights = get_weight_regrough(xovi_amat.xov).reset_index()  ### old way using residuals to extract roughness
    #
    # get interpolation error based on roughness map (if available at given latitude) + minimal distance
    interp_weights = get_interpolation_weight(xovi_amat.xov).reset_index()
    val = interp_weights['weight'].values # np.ones(len(interp_weights['weight'].values)) #
    # print("interp error values", np.sort(val))

    # apply huber weights
    if not remove_max_dist and not remove_3sigma_median and not remove_dR200:
        val *= huber_penal
        # print("after huber", np.sort(val), np.mean(val))
        # val *= huber_weights_dist
        # print(val)
        # exit()
    # print(np.max(np.abs(val)))
    # val /= np.max(np.abs(val))
    row = col = interp_weights.index.values

    # obs_weights = csr_matrix((np.ones(len(val)), (row, col)), dtype=np.float32, shape=(len(interp_weights), len(interp_weights)))
    obs_weights = csr_matrix((val, (row, col)), dtype=np.float32, shape=(len(interp_weights), len(interp_weights)))

    # combine with off-diag terms from tracks
    #========================================
    # obs_weights = diags(weights_xov_tracks.diagonal()*obs_weights) # to apply only the diagonal
    obs_weights = weights_xov_tracks.multiply(obs_weights)

    if debug and local:
        print("tracks weights", weights_xov_tracks.diagonal().mean(), np.sort(weights_xov_tracks.diagonal()))
        tmp = obs_weights.diagonal()
        tmp = np.where(tmp>1.e-9,tmp,0.)
        print(np.sort(tmp),np.median(tmp),np.mean(tmp))
        # plot histo
        plt.figure() #figsize=(8,3))
        # plt.xlim(-1.*xlim, xlim)
        # the histogram of the data
        num_bins = 200 #'auto'
        n, bins, patches = plt.hist(tmp, bins=num_bins, range=[1.e-4,4.e-2], cumulative=-1) #, density=True, facecolor='blue',
        # alpha=0.7, range=[-1.*xlim, xlim])
        plt.xlabel('obs weights')
        plt.ylabel('# tracks')
        plt.title('Resid+distance+offnadir+interp+weights: $\mu=' + str(np.mean(tmp)) + ', \sigma=' + str(np.std(tmp)) + '$')
        # # Tweak spacing to prevent clipping of ylabel
        plt.subplots_adjust(left=0.15)
        plt.savefig(tmpdir+'/data_weights.png')
        plt.clf()
        # exit()

    # Combine and store weights
    xovi_amat.weights = obs_weights
    xovi_amat.xov.xovers['weights'] = xovi_amat.weights.diagonal()

    ## DIRECT SOLUTION FOR DEBUG AND SMALL PROBLEMS (e.g., global only)
    if len(sol4_pars)<50 and debug:

        print('B', xovi_amat.b)
        print('maxB', np.abs(xovi_amat.b).max(),np.abs(xovi_amat.b).mean())
        print('maxA',np.abs(spA_sol4.todense()).max(),
              # np.shape(spA_sol4.todense()[np.abs(spA_sol4.todense())>seuil_dRdL]),
              np.shape(spA_sol4.todense()))
        # print("values", spA_sol4.todense()[np.abs(spA_sol4.todense())>seuil_dRdL])
        # print("Their indices are ", len(np.nonzero(np.abs(spA_sol4.todense()) > seuil_dRdL)[0]), np.nonzero(np.abs(spA_sol4.todense()) > seuil_dRdL)[0])
        # print("Their values are ", spA_sol4.todense()[np.nonzero(np.abs(spA_sol4.todense()) > seuil_dRdL)[0]].T)
        # print("Their values are ", xovi_amat.b[np.nonzero(np.abs(spA_sol4.todense()) > seuil_dRdL)[0]].T)
        # exclude = np.nonzero(np.abs(spA_sol4.todense()) > seuil_dRdL)[0]
        # if len(exclude) > 0:
        #     print("Partials screened by ", seuil_dRdL, "remove ", np.round(len(exclude)/len(xovi_amat.b)*100,2), "% of obs")
        spAdense = spA_sol4.todense()
        bvec = xovi_amat.b
        # spAdense = np.delete(spAdense, exclude, 0)
        # bvec = np.delete(bvec, exclude, 0)
        #
        # keep = list(set(spA_sol4.nonzero()[0].tolist())^set(exclude))
        # spA_sol4 = spA_sol4[keep,:]
        # xovi_amat.b = bvec
        print("The new values A are ", spAdense)
        print("The new values b are ", bvec)
        # exit()
        # spAdense = spA_sol4.todense()
        # spAdense[np.abs(spAdense) > 200] = 1

        # plt.clf()
        # fig, ax = plt.subplots()
        # # ax.plot(spA_sol4.todense()<2000)
        # ax.plot(spAdense, label=[xovi_amat.parNames[p] for p in sol4_pars])
        # ax.legend()
        # ax.plot(bvec)
        # plt.savefig(tmpdir+'b_and_A.png')

    # analysis of residuals vs h2 partials
    if h2_limit_on and debug and local:
        print(xovi_amat.xov.xovers.columns)
        tmp = xovi_amat.xov.xovers[['dR','dR/dh2','LON','LAT','weights']]
        # print("truc0",tmp['weights'].abs().min(),tmp['weights'].abs().max())
        tmp = tmp.loc[(tmp.dR.abs() < limit_h2) & (tmp['dR/dh2'].abs() > 0.3) & (tmp['weights'].abs() > 0.5*sigma_0)]
        # print("truc",tmp)

        w = np.abs(tmp[['dR']].abs().values)
        dw_dh2 = np.abs(tmp[['dR/dh2']].values) #np.abs(spA_sol4[:,-1].toarray())
        # import statsmodels.api as sm
        # result = sm.OLS(dw_dh2, w).fit()
        print("lenw",len(tmp))

        fig, ax = plt.subplots(1)
        ax.scatter(x=w, y=dw_dh2) #, color = rgb)
        # ax.set_xlim(0,2.5)
        # n, bins, patches = plt.plot(x=w,y=dw_dh2)
        # plt.semilogy()
        # plt.legend()
        # plt.ylabel('# of obs')
        # plt.xlabel('meters/[par]')
        plt.savefig(tmpdir+"discr_vs_dwdh2.png")

        plt.clf()
        piv = pd.pivot_table(tmp.round({'LON':0,'LAT':0}), values="dR/dh2", index=["LAT"], columns=["LON"],
                             fill_value=None, aggfunc=rms)
        ax = sns.heatmap(piv, xticklabels=10, yticklabels=10, cmap="YlGnBu") #, square=False, annot=True)
        plt.tight_layout()
        ax.invert_yaxis()
        plt.savefig(tmpdir+"geo_dwdh2.png")
        # exit()

    # analysis of partial derivatives to check power in obs & param
    if debug and local:

        tmp = spla.norm(spA_sol4,axis=0)
        print("partials analysis",tmp.shape)
        print(sol4_pars)
        print(tmp[-5:])

        dw_dh2 = spA_sol4[:,-1].toarray()
        dw_dra = spA_sol4[:,-2].toarray()
        dw_dpm = spA_sol4[:,-4].toarray()
        dw_dl = spA_sol4[:,-3].toarray()
        dw_ddec = spA_sol4[:,-5].toarray()

        plt.clf()
        plt.figure() #figsize=(8, 3))
        xlim = 1.
        # plt.xlim(-1. * xlim, xlim)
        # the histogram of the data
        num_bins = 200  #'auto' #

        parnam = ['PM','L','DEC','RA','h2']

        for idx,par in enumerate([dw_dpm,dw_dl,dw_ddec,dw_dra,dw_dh2]):
            tmp = np.abs(par)
            n, bins, patches = plt.hist(tmp, bins=num_bins, density=False,
                                        alpha=0.8,label=parnam[idx],weights=obs_weights.diagonal())
        plt.ylim(bottom=sigma_0)
        plt.semilogy()
        plt.legend()
        plt.ylabel('# of obs')
        plt.xlabel('meters/[par]')
        plt.savefig(tmpdir+"partials_histo_weighted.png")

        plt.clf()
        for idx,par in enumerate([dw_dpm,dw_dl,dw_ddec,dw_dra,dw_dh2]):
            tmp = np.abs(par)
            n, bins, patches = plt.hist(tmp, bins=num_bins, density=False,
                                        alpha=0.8,label=parnam[idx])
        plt.semilogy()
        plt.legend()
        plt.ylabel('# of obs')
        plt.xlabel('meters/[par]')
        plt.savefig(tmpdir+"partials_histo.png")
        # exit()

    # svd analysis of parameters (eigenvalues and eigenvectors)
    if debug and local: #len(sol4_pars) < 50 and debug:

        # Compute the covariance matrix
        # print("full sparse",np.linalg.pinv((spA_sol4.transpose()*spA_sol4).todense()))
        # print("screened dense", np.linalg.pinv(spAdense.transpose()*spAdense))
        # tmp = prev.xov.xovers['dR'].values.reshape(1, -1) @ np.diag(prev.xov.xovers['weights'].values)
        # m_0 = np.sqrt(
        #     tmp @ prev.xov.xovers['dR'].values.reshape(-1, 1) / (len(prev.xov.xovers['dR'].values) - len(sol4_glo)))
        # ATP = spAdense.transpose() * obs_weights
        # ATPA = ATP * spAdense
        # PA = obs_weights * spAdense
        N = (spA_sol4.transpose()*obs_weights*spA_sol4).todense() #ATPA
        print(sol4_pars)

        # project arc par on global pars
        ATA = csr_matrix(N[:-5,:-5])
        print(ATA.shape)
        ATB = csr_matrix(N[:-5,-5:])
        BTA = csr_matrix(N[-5:,:-5])
        BTB = csr_matrix(N[-5:,-5:])

        tmp = np.linalg.pinv(ATA.todense())*ATB
        N_proj = BTB - BTA*tmp

        # check eigenvector and values in the problem
        for idx,mat in enumerate([BTB.todense(),N_proj]):
            M = la.cholesky(mat)
            U, S, Vh = la.svd(M)
            print('S')
            print(S)
            plt.clf()
            plt.semilogy(S)
            plt.savefig(tmpdir+"test_lambdaS_"+str(idx)+".png")
            print('Vh transp')
            print(Vh.T)
            plt.clf()
            plt.imshow(Vh.T,cmap='bwr')
            plt.colorbar()
            plt.savefig(tmpdir+"test_svd_"+str(idx)+".png")

            print("Pars:",list(xovi_amat.parNames.keys())[-5:])
            for i in range(5):
                print("Norm of Vh",np.round(np.linalg.norm(Vh.T[:,:i+1],axis=1)*100.,1),"% up to lambda= ",S[i])
        # exit()

        if False:
            ell = csr_matrix(np.diag(np.abs(bvec)))
            print(ell)
            posterr = np.linalg.pinv(ATP * ell * PA)
            posterr = np.sqrt(posterr.diagonal())
            print("posterr")
            print(posterr)
        # print(np.linalg.pinv(posterr))
        # posterr = Ninv * (spAdense.transpose() * (obs_weights * (ell * (obs_weights * (spAdense * N)))))
        # print(posterr)
        # exit()
        # Ninv = np.linalg.pinv(spAdense.transpose() * spAdense)

        # check eigenvalues and vectors: https://andreask.cs.illinois.edu/cs598apk-f15/demos/02-tools-for-low-rank/Rank-Revealing%20QR.html

        # compute sol

        # factorL = np.linalg.norm([0.00993822, \
        #             -0.00104581, \
        #             -0.00010280, \
        #             -0.00002364, \
        #             -0.00000532])
            print('sol dense',np.linalg.lstsq(spAdense[:], bvec[:], rcond=1)[0])#/factorL)
            print('to_be_recovered', pert_cloop['glo'])

        # exit()

    # # Compute the covariance matrix
    # print(np.linalg.pinv((spA_sol4.transpose()*spA_sol4).todense()))
    # # compute sol
    # print('sol dense',np.linalg.lstsq(spA_sol4.todense(), xovi_amat.b, rcond=1))
    #
    # A = spA_sol4.transpose()*spA_sol4
    # b = spA_sol4.transpose()*(csr_matrix(xovi_amat.b).transpose())

    #### CONSTRAINS AND SOLUTION

    # apply weights
    #print("len trucs")
    #print(len(sol4_pars))

    spA_sol4 = xovi_amat.weights * spA_sol4

    # select constrains for processed parameters (TODO should go in sol4pars)
    mod_par = [your_key.split('_')[1] if len(your_key.split('_'))>1 else your_key for your_key in sol4_pars ]
    if OrbRep in ['lin','quad'] :
        for par in ['dA','dC','dR']:
            if par in sol4_orbpar:
                par_constr['dR/'+par+'0'] = par_constr.pop('dR/'+par)

    par_constr = { your_key: par_constr[your_key] for your_key in mod_par }

    csr = []
    for constrain in par_constr.items():

        regex = re.compile(".*" + constrain[0] + "$")
        parindex = np.array([[idx,float(constrain[1])] for idx,p in enumerate(sol4_pars) if regex.match(p)])

        # Constrain tightly to 0 those parameters with few observations (or with few GOOD observations)
        if not remove_max_dist and not remove_3sigma_median and not remove_dR200:
            # TODO should use weights or measurement error threshold, but using huber-threshold-like criteria for now
            # TODO to mimic what I was doing without weights
            # nobs_tracks = xovi_amat.xov.xovers.loc[xovi_amat.xov.xovers.huber > 0.5][['orbA', 'orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(
            # ascending=False)
            n_goodobs_tracks = xovi_amat.xov.xovers.loc[xovi_amat.weights.diagonal() > 0.5*sigma_0][['orbA', 'orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(
            ascending=False)
        else:
            n_goodobs_tracks = xovi_amat.xov.xovers[['orbA', 'orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(
            ascending=False)

        to_constrain = [idx for idx, p in enumerate(sol4_pars) if p.split('_')[0] in n_goodobs_tracks[n_goodobs_tracks < 20].index]

        for p in parindex:
            if p[0] in to_constrain:
                # else a very loose "general" constraint could free it up
                # (if constraint on p is int, could mess up and give 0 => Nan)
                if p[1] > 1.:
                    p[1] = 1.e-4
                else:
                    p[1] *= 1.e-4

        val = parindex[:,1]
        row = col = parindex[:,0]

        csr.append(
                csr_matrix((np.power(sigma_0/val,2), (row, col)), dtype=np.float32, shape=(len(sol4_pars), len(sol4_pars))))
    # combine all constraints
    penalty_matrix = sum(csr)
    # print(penalty_matrix)

    if True:

        csr_avg = []
        for constrain in mean_constr.items():
            regex = re.compile(".*"+constrain[0]+"0{0,1}$")

            if list(filter(regex.match, sol4_pars)):
                parindex = np.array([[idx,float(constrain[1])] for idx,p in enumerate(sol4_pars) if regex.match(p)])
                # print("matching", list(filter(regex.match, sol4_pars)))
                # # exit()
                # # df_sol.columns = [x[:-1] if x in list(filter(regex.match, df_sol.columns)) else x for x in
                # #                   df_sol.columns.values]
                # print(parindex)
                # print(len(parindex))
                if len(parindex)>0:

                    # Constrain tightly to 0 those parameters with few observations
                    # nobs_tracks = xovi_amat.xov.xovers[['orbA', 'orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(
                    #     ascending=False)
                    # to_constrain = [idx for idx, p in enumerate(sol4_pars) if
                    #                 p.split('_')[0] in nobs_tracks[nobs_tracks < 10].index]
                    # for p in parindex:
                    #     if p[0] in to_constrain:
                    #         p[1] *= 1.e10

                    # print(constrain)
                    # print(parindex)

                    rowcols_nodiag = np.array(list(set(itertools.permutations(parindex[:,0], 2))))
                    rowcols_diag = np.array(list([(x,x) for x in parindex[:,0]]))
                    vals = - 1/len(parindex[:,0]) * np.ones(len(parindex[:,0])*len(parindex[:,0])-len(parindex[:,0]))
                    csr_avg.append(csr_matrix((vals*np.power(sigma_0/constrain[1],2),
                                               (rowcols_nodiag[:,0],rowcols_nodiag[:,1])),
                                              dtype=np.float32, shape=(len(sol4_pars), len(sol4_pars))))
                    vals = (1 - 1/len(parindex[:,0])) * np.ones(len(parindex[:,0]))
                    csr_avg.append(csr_matrix((vals*np.power(sigma_0/constrain[1],2),
                                               (rowcols_diag[:,0],rowcols_diag[:,1])),
                                              dtype=np.float32, shape=(len(sol4_pars), len(sol4_pars))))
        # print(sum(csr_avg))
        # print(penalty_matrix)

        penalty_matrix = penalty_matrix + sum(csr_avg)

    # store penalty into amat
    xovi_amat.penalty_mat = penalty_matrix
        # exit()

    # Choleski decompose matrix and append to design matrix
    Q = np.linalg.cholesky(penalty_matrix.todense())

    # print(np.shape(xovi_amat.weights))
    # print(len(np.ravel(np.dot(Q,previous_iter.sol_iter[0]))))
    # print(np.shape(xovi_amat.b))
    if previous_iter.sol_dict_iter != None:
        # get previous solution reordered as sol4_pars (and hence as Q)
        prev_sol_ord = [previous_iter.sol_dict_iter['sol'][key] if
                        key in previous_iter.sol_dict_iter['sol'] else 0. for key in sol4_pars]
        b_penal = np.hstack([xovi_amat.weights*xovi_amat.b, np.ravel(np.dot(Q,prev_sol_ord))]) #np.zeros(len(sol4_pars))]) #
    else:
        b_penal = np.hstack([xovi_amat.weights*xovi_amat.b, np.zeros(len(sol4_pars))])
    # import scipy
    spA_sol4_penal = scipy.sparse.vstack([spA_sol4,csr_matrix(Q)])
    xovi_amat.spA_penal = spA_sol4_penal
    # print("Pre-sol: len(A,b)=",len(b_penal),spA_sol4_penal.shape)
    #print([xovi_amat.parNames[p] for p in sol4_pars])

    # exit()
    # print("Pre-sol-2: len(A,b)=",spA_sol4_penal.shape,len(b_penal))
    # exit()

    xovi_amat.sol = lsqr(spA_sol4_penal, b_penal,damp=0,show=True,iter_lim=100000,atol=1.e-8/sigma_0,btol=1.e-8/sigma_0,calc_var=True)
    # xovi_amat.sol = lsqr(xovi_amat.spA, xovi_amat.b,damp=0,show=True,iter_lim=100000,atol=1.e-8,btol=1.e-8,calc_var=True)
    # print("sol sparse: ",xovi_amat.sol[0])
    # print('to_be_recovered', pert_cloop['glo'])
    print("Solution iters terminated with ", xovi_amat.sol[1])
    if xovi_amat.sol[1] != 2:
       exit(2)

    if not remove_max_dist and not remove_3sigma_median and not remove_dR200:
        tmp = obs_weights.diagonal()
        print("Fully weighted obs (>0.5*sigma0): ", len(tmp[tmp>0.5*sigma_0]), "or ",len(tmp[tmp>0.5*sigma_0])/len(tmp)*100.,"%")
        print("Slightly downweighted obs: ", len(tmp[(tmp<0.5*sigma_0)*(tmp>0.05*sigma_0)]), "or ",len(tmp[(tmp<0.5*sigma_0)*(tmp>0.05*sigma_0)])/len(tmp)*100.,"%")
        print("Brutally downweighted obs (<0.05*sigma0): ", len(tmp[(tmp<0.05*sigma_0)]), "or ",len(tmp[(tmp<0.05*sigma_0)])/len(tmp)*100.,"%")

    # exit()

    # _ = lsmr(A+penalty_matrix,b.toarray(),show=False, maxiter=5000)
    # # print(np.diag(np.linalg.pinv(((A+penalty_matrix).transpose() * (A+penalty_matrix)).todense())))
    # # print(xovi_amat.sol)
    # xovi_amat.sol = (_[0], *xovi_amat.sol[1:])

# @profile
def clean_partials(b, spA, nglbpars, threshold = 1.e6):
    # spA = spA[:99264,-4:]

    if debug:
        print("## clean_partials - size pre:", len(b), spA.shape)
        # print(sol4_glo)
        # print(len(sol4_glo))
        # print(spA[:,2].data)
        plt.clf()
        fig, axlst = plt.subplots(nglbpars)
        # exit()
        # ax.plot(spA_sol4.todense()<2000)
        if nglbpars>1:
            for idx in range(nglbpars):
                axlst[idx].plot(spA[:, -nglbpars + idx].todense(), label=sol4_glo[idx])
                # i.plot(spA[:, :-4].sum(axis=1).A1, label=sol4_glo[idx])
                axlst[idx].legend(loc='upper right')
        else:
            axlst.plot(spA[:, -nglbpars + 0].todense(), label=sol4_glo[0])
            # i.plot(spA[:, :-4].sum(axis=1).A1, label=sol4_glo[idx])
            axlst.legend(loc='upper right')
        # i.plot(b)
        plt.savefig(tmpdir + 'b_and_A_pre.png')
    # exit()

    Nexcluded = 0
    # print(sol4_glo)
    # print(spla.norm(spA[:,-5:],axis=0))
    for i in range(len(sol4_glo)):

        data = spA.tocsc()[:, -i - 1].data
        median_residuals = np.abs(data - np.median(data, axis=0))
        sorted = np.sort(median_residuals)
        std_median = sorted[round(0.68 * len(sorted))]

        std_mean = np.std(data, axis=0)
        if std_mean>10:
            print("## Check partials for outliers", i, std_median, std_mean, std_median/std_mean)

        exclude = np.argwhere(median_residuals >= 20 * std_mean).T[0]
        row2index = dict(zip(range(len(data)),list(set(spA.tocsc()[:, -i - 1].nonzero()[0].tolist()))))
        exclude = [row2index[i] for i in exclude]

        # remove bad rows, only non-zero columns to keep sparsity
        J = identity(spA.shape[0], format='csr')
        row = col = exclude
        J = J + csr_matrix((np.ones(len(row))*(-1+1.e-20), (row, col)), dtype=np.float32, shape=(spA.shape[0],spA.shape[0]))
        spA = J * spA
        b[exclude] = 1e-20

        Nexcluded += len(exclude)

        # keep = list(set(spA.nonzero()[0].tolist()) ^ set(exclude))
        # print("bad= ", i, np.median(data, axis=0), 4 * std_median, len(median_residuals), np.max(median_residuals),
        #       len(exclude) / len(median_residuals) * 100., "% ")
        # print(spA[exclude, -i - 1])
        # print(np.array(keep))
        # print("## clean_partials removed ", i, 4 * std_median, np.round((len(b) - len(keep)) / len(b) * 100., 2),
        #       "% observations")
        # b = b[keep]
        # spA = spA[keep, :]

        # print("post= ", i, np.max(spA[:, -i - 1].data))

    print("## clean_partials - size excluded:", Nexcluded, Nexcluded/len(b)*100.,"%") #, len(keep))
    if debug:
        plt.clf()
        fig, axlst = plt.subplots(nglbpars)
        if nglbpars>1:
            for idx in range(nglbpars):
                axlst[idx].plot(spA[:, -nglbpars + idx].todense(), label=sol4_glo[idx])
                # i.plot(spA[:, :-4].sum(axis=1).A1, label=sol4_glo[idx])
                axlst[idx].legend(loc='upper right')
        else:
            axlst.plot(spA[:, -nglbpars + 0].todense(), label=sol4_glo[0])
            # i.plot(spA[:, :-4].sum(axis=1).A1, label=sol4_glo[idx])
            axlst.legend(loc='upper right')
        plt.savefig(tmpdir + 'b_and_A_post.png')

    # print(spla.norm(spA[:,-5:],axis=0))
    # exit()

    return b, spA


def solve4setup(sol4_glo, sol4_orb, sol4_orbpar, track_names):

    if sol4_orb == [] and sol4_orbpar != [None]:
        sol4_orb = set([i.split('_')[0] for i in track_names])
        sol4_orb = [x for x in sol4_orb if x.isdigit()]
        if sol4_orbpar == []:
            sol4_orbpar = list(parOrb.keys())
        sol4_orb = [x + '_' + 'dR/' + y for x in sol4_orb for y in sol4_orbpar]
    elif sol4_orb == [None] or sol4_orbpar == [None]:
        sol4_orb = []
    else:
        sol4_orb = [x + '_' + 'dR/' + y for x in sol4_orb for y in sol4_orbpar]

    if sol4_glo == []:
        sol4_glo = list(parGlo.keys())
        sol4_glo = ['dR/'+x for x in sol4_glo]
    elif sol4_glo == [None]:
        sol4_glo = []

    sol4_pars = sorted(sol4_orb) + sorted(sol4_glo)

    # print(len(sorted(sol4_orb)))
    # print('solving for:',np.array(sol4_pars))

    return sol4_pars


def analyze_sol(xovi_amat,xov):
    # print('xovi_amat.sol',xovi_amat.sol)

    # print(parOrb)
    # print([x.split('/')[1] in parOrb.keys() for x in xovi_amat.sol4_pars])
    # print(np.sum([x.split('/')[1] in parOrb.keys() for x in xovi_amat.sol4_pars]))
    # exit()

    print(len(np.reshape(xovi_amat.sol4_pars, (-1, 1))),len(np.reshape(xovi_amat.sol[0], (-1, 1))),
                              len(np.reshape(xovi_amat.sol[-1], (-1, 1))) )

    # Ordering is important here, don't use set or other "order changing" functions
    _ = np.hstack((np.reshape(xovi_amat.sol4_pars, (-1, 1)),
                   np.reshape(xovi_amat.sol[0], (-1, 1)),
                   np.reshape(xovi_amat.sol[-1], (-1, 1))
                   ))
    sol_dict = {'sol': dict(zip(_[:,0],_[:,1].astype(float))), 'std': dict(zip(_[:,0],_[:,2].astype(float))) }

    if debug:
        print("sol_dict_analyze_sol")
        print(sol_dict)
    # print(np.hstack((np.reshape(xovi_amat.sol4_pars, (-1, 1)), np.reshape([xovi_amat.parNames[p] for p in xovi_amat.sol4_pars], (-1, 1)), np.reshape(xovi_amat.sol[0], (-1, 1)),
    #                np.reshape(xovi_amat.sol[-1], (-1, 1)))))
    # exit()

    # Extract solution for global parameters
    glb_sol = pd.DataFrame(_[[x.split('/')[1] in list(parGlo.keys()) for x in _[:,0]]],columns=['par','sol','std'])
    partemplate = set([x.split('/')[1] for x in sol_dict['sol'].keys()])
    # regex = re.compile(".*"+str(list(partemplate))+"$")

    # print(sol_dict)

    # Extract solution for orbit parameters
    parOrbKeys = list(parOrb.keys())
    solved4 = list(partemplate)

    if OrbRep in ['lin','quad']:
        parOrbKeys = [x+str(y) for x in parOrbKeys for y in [0,1,2]]
    # solved4orb = list(filter(regex.match, list(parOrb.keys())))
    solved4orb = list(set(parOrbKeys)&set(solved4))

    if len(solved4orb) > 0:
        df_ = pd.DataFrame(_, columns=['key', 'sol', 'std'])
        df_[['orb', 'par']] = df_['key'].str.split('_', expand=True)
        df_ = df_.astype({'sol': 'float64', 'std': 'float64'})
        df_.drop('key', axis=1, inplace=True)
        # df_[['orb','par']] = df_[['par','orb']].where(df_['par'] == None, df_[['orb','par']].values)
        df_ = df_.replace(to_replace='None', value=np.nan).dropna()
        table = pd.pivot_table(df_, values=['sol','std'], index=['orb'], columns=['par'], aggfunc=np.sum)

        if any(xov.xovers.filter(like='dist', axis=1)):
            xov.xovers['dist_max'] = xov.xovers.filter(regex='^dist_[A,B].*$').max(axis=1)
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
        if debug:
            print(orb_sol)

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

    # rescale with sigma0
    print(glb_sol['std'].values)

    # sol_dict['std'] *= sigma_0
    glb_sol['std'] = glb_sol['std'].astype('float').values/sigma_0
    for col in orb_sol.filter(regex='std_*').columns:
        orb_sol[col] = orb_sol[col].astype('float').values/sigma_0

    return orb_sol, glb_sol, sol_dict


def print_sol(orb_sol, glb_sol, xov, xovi_amat):

    partemplate = set([x.split('/')[1] for x in xovi_amat.sol_dict['sol'].keys()])

    regex = re.compile('.*_dR/d([ACR]|(Rl)|(Pt))[0,1,2]?$')
    soltmp = [(x.split('_')[0], 'sol_' + x.split('_')[1], v) for x, v in xovi_amat.sol_dict['sol'].items() if regex.match(x)]

    print('-- Solutions -- ')
    if len(soltmp) > 0:
        stdtmp = [(x.split('_')[0], 'std_' + x.split('_')[1], v) for x, v in xovi_amat.sol_dict['std'].items() if
                  regex.match(x)]
        soltmp = pd.DataFrame(np.vstack([soltmp, stdtmp]))
        soltmp[2] = soltmp[2].astype(float)
        soltmp = pd.pivot_table(soltmp, index=[0], columns=[1], values=[2])
        print('Orbit parameters: ')
        print('-- -- -- -- ')
        print(soltmp)
        print('-- -- -- -- ')
    print('-- -- -- -- ')
    print('Global parameters: ')
    print('-- -- -- -- ')
    print(glb_sol)
    print('-- -- -- -- ')

    if len(pert_cloop['glo'])>0:
        print('to_be_recovered (sim mode, dRl, dPt, dRA, dDEC, dL in arcsec; dPM in arcsec/Julian year)')
        print(pert_cloop['glo'])

    if debug and False:
        _ = xov.remove_outliers('dR',remove_bad=remove_3sigma_median)
        print(xov.xovers.columns)
        print(xov.xovers.loc[xov.xovers.orbA == '1301022345'])
        # print(xov.xovers.loc[xov.xovers.orbA == '1301022341' ].sort_values(by='R_A', ascending=True)[:10])
        if np.sum([x.split('/')[1] in parOrb.keys() for x in xovi_amat.sol4_pars]) > 0:
            print(orb_sol.reindex(orb_sol.sort_values(by='sol_dR/dR', ascending=False).index)[:10])
            print(orb_sol.loc[orb_sol.orb == '1301022345', :])

def main(arg):
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
        #
        # # count occurrences for each orbit ID
        # xov_cmb.nobs_x_track = xov_cmb.xovers[['orbA','orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(ascending=False)
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
                parsk = list(xov_cmb.pert_cloop.to_dict().keys())
                trackk = list(xov_cmb.pert_cloop.to_dict()[parsk[0]].keys())

                fit2dem_sol = np.ravel([list(x.values()) for x in xov_cmb.pert_cloop.to_dict().values()])
                fit2dem_keys = [tr+'_dR/'+par for par in parsk for tr in trackk]
                fit2dem_dict = dict(zip(fit2dem_keys,fit2dem_sol))

                previous_iter = Amat(vecopts)
                previous_iter.sol_dict = {'sol':fit2dem_dict,'std':dict(zip(fit2dem_keys,np.zeros(len(fit2dem_keys))))}
                # previous_iter.sol_dict_iter = previous_iter.sol_dict
                # previous_iter = None

            # solve dataset
            par_list = ['orbA', 'orbB', 'xOvID']
            xovi_amat = prepare_Amat(xov_cmb, vecopts, par_list)

            solve(xovi_amat, dataset=ds, previous_iter=previous_iter)
            # Save to pkl
            orb_sol, glb_sol, sol_dict = analyze_sol(xovi_amat,xov_cmb)

            # remove corrections OF SINGLE ITER if "unreasonable" (larger than 100 meters in any direction, or 50 meters/day, or 20 arcsec)
            sol_dict_iter = sol_dict
            print("Length of original sol", len(sol_dict_iter['sol']))

            sol_dict_iter_clean = []
            std_dict_iter_clean = []
            regex = re.compile(".*_dR/d[A,C,R,Rl,Pt]0{0,1}$")
            tracks = list(set([x.split('_')[0] for x, v in sol_dict_iter['sol'].items() if regex.match(x)]))
            bad_count = 0
            for tr in tracks:
                regex = re.compile("^" + tr + "*")
                soltmp = dict([(x, v) for x, v in sol_dict_iter['sol'].items() if regex.match(x)])
                stdtmp = dict([(x, v) for x, v in sol_dict_iter['std'].items() if regex.match(x)])
                regex = re.compile(".*_dR/d[A,C,R]0{0,1}$")
                max_orb_corr = np.max(np.abs([v if regex.match(x) else 0 for x, v in soltmp.items()]))
                regex = re.compile(".*_dR/d[A,C,R]1$")
                max_orb_drift_corr = np.max(np.abs([v if regex.match(x) else 0 for x, v in soltmp.items()]))
                regex = re.compile(".*_dR/d{Rl,Pt}$")
                max_att_corr = np.max(np.abs([v if regex.match(x) else 0 for x, v in soltmp.items()]))
                # ok to put limit on CUMULATED corrections
                # soltmp = track.sol_prev_iter['orb'].filter(regex='sol_dR/.*')
                # max_orb_corr = soltmp.filter(regex="sol_dR/d[A,C,R]0*").abs().max(axis=1).values[0]
                # max_orb_drift_corr = soltmp.filter(regex='sol_dR/d[A,C,R]1').abs().max(axis=1).values[0]
                # max_att_corr = soltmp.filter(regex='sol_dR/d{Rl,Pt}').abs().max(axis=1).values[0]
                # print("max_orb_corr,max_orb_drift_corr,max_att_corr")
                # print(max_orb_corr, max_orb_drift_corr, max_att_corr)

                if max_orb_corr > 2000 or max_orb_drift_corr > 500 or max_att_corr > 20.:
                    print("Solution fixed for track", tr, 'with max_orb_corr,max_orb_drift_corr,max_att_corr:',max_orb_corr, max_orb_drift_corr, max_att_corr)
                    sol_dict_iter_clean.append(dict.fromkeys(soltmp, 0.))
                    bad_count += 1
                else:
                    # pass
                    sol_dict_iter_clean.append(soltmp)
                # keep std also for bad orbits
                std_dict_iter_clean.append(stdtmp)

            # add back global parameters
            if len(sol4_glo)>0:
                sol_dict_iter_clean.append(dict([(x,v) for x, v in sol_dict_iter['sol'].items() if x in sol4_glo]))
                std_dict_iter_clean.append(dict([(x,v) for x, v in sol_dict_iter['std'].items() if x in sol4_glo]))

            sol_dict_iter_clean = {k: v for d in sol_dict_iter_clean for k, v in d.items()}
            std_dict_iter_clean = {k: v for d in std_dict_iter_clean for k, v in d.items()}
            sol_dict_iter_clean = dict(zip(['sol','std'],[sol_dict_iter_clean,std_dict_iter_clean]))
            print("New length of cleaned sol",len(sol_dict_iter_clean['sol'])-bad_count)

            xovi_amat.sol_dict = sol_dict_iter_clean
            # exit()

            print("Sol for iter ", str(ext_iter))
            print_sol(orb_sol, glb_sol, xov, xovi_amat)
            # # print(orb_sol.filter(regex="sol_.*"))
            # print("#####\n Average corrections:")
            # print(orb_sol.filter(regex="sol_.*").astype(float).mean(axis=0))
            # print("#####\n Std corrections:")
            # print(orb_sol.filter(regex="sol_.*").astype(float).std(axis=0))
            # print("#####")
            # print("Max corrections:")
            # print(orb_sol.filter(regex="sol_.*").astype(float).abs().max(axis=0))
            # print("#####")

            # store improvments from current iteration
            xovi_amat.sol_dict_iter = xovi_amat.sol_dict.copy()
            xovi_amat.sol_iter = (list(xovi_amat.sol_dict['sol'].values()), *xovi_amat.sol[1:-1], list(xovi_amat.sol_dict['std'].values()))

            # Cumulate with solution from previous iter
            if (int(ext_iter) > 0) | (previous_iter != None):
                # sum the values with same keys
                updated_sol = mergsum(xovi_amat.sol_dict['sol'],previous_iter.sol_dict['sol'])
                updated_std = mergsum(xovi_amat.sol_dict['std'],previous_iter.sol_dict['std'].fromkeys(previous_iter.sol_dict['std'], 0.))
                xovi_amat.sol4_pars = list(updated_sol.keys())

                xovi_amat.sol_dict = {'sol': updated_sol, 'std' : updated_std}
                # use dict to update amat.sol, keep std
                xovi_amat.sol = (list(xovi_amat.sol_dict['sol'].values()), *xovi_amat.sol[1:-1], list(xovi_amat.sol_dict['std'].values()))
                # print(xovi_amat.sol)
                # print(xovi_amat.sol_dict)
                # exit()
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
            if debug:
                plt_histo_dR(tstname, mean_dR, std_dR,
                             xov_cmb.xovers)  # [tmp.xov.xovers.orbA.str.contains('14', regex=False)])

                empty_geomap_df = pd.DataFrame(0, index=np.arange(0, 91),
                                               columns=np.arange(-180, 181))
                plt_geo_dR(tstname, xov_cmb)


        # append to list for stats
        xov_cmb_lst.append(xov_cmb)

    print("len xov_cmb ", len(xov_cmb_lst[0].xovers))

    if partials:
        get_stats(amat=xovi_amat)

    print("len xov_cmb post getstats", len(xov_cmb_lst[0].xovers))


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
