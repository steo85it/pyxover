#!/usr/bin/env python3
# ----------------------------------
# accum_utils.py
#
# Description: various methods to be applied to AccumXov
#
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 05-Feb-2020
import logging
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, diags, issparse

# from AccumXov import sigma_0, remove_3sigma_median, remove_max_dist
# from accumxov.accum_opt import sigma_0, remove_3sigma_median
# from examples.MLA.options import XovOpt.get("tmpdir"), XovOpt.get("full_covar"), XovOpt.get("debug"), XovOpt.get("local"), pert_cloop, XovOpt.get("parOrb"), XovOpt.get("parGlo"), XovOpt.get("OrbRep"), XovOpt.get("vecopts"), XovOpt.get("outdir")
from accumxov.accum_opt import AccOpt
from config import XovOpt

import matplotlib.pyplot as plt

from pyxover.xov_setup import xov
from pyxover.xov_utils import get_tracks_rms
from xovutil.iterables import multiply_sparse_get_diag

# @profile
def get_xov_cov_tracks(df, plot_stuff=False):
    tracks_rms_df = get_tracks_rms(df, plot_xov_tseries=plot_stuff)
    # pd.set_option("display.max_rows", 999)
    # print(tracks_rms_df.reindex(tracks_rms_df.pre.abs().sort_values().index))
    # pd.reset_option("display.max_rows")

    # print(xovi_amat.xov.xovers)
    tmp = df[['xOvID', 'orbA', 'orbB']].astype('int32')
    # get unique tracksID in dataset
    unique_orb = np.sort((tmp['orbA'].append(tmp['orbB'])).unique())
    # map to pseudo-pivot csr indexes
    tracks_map = dict(zip(unique_orb, range(len(unique_orb))))
    tmp['mapA'] = tmp['orbA'].map(tracks_map)
    tmp['mapB'] = tmp['orbB'].map(tracks_map)
    # generate sparse matrix of ones for each track, then sum to have 2 "ones-elements" for each xov
    csrA = csr_matrix((np.ones(len(tmp['xOvID'].values)), (tmp['xOvID'].values, tmp['mapA'].values)),
                      dtype=np.float32, shape=(len(tmp['xOvID'].values), len(unique_orb)))
    csrB = csr_matrix((np.ones(len(tmp['xOvID'].values)), (tmp['xOvID'].values, tmp['mapB'].values)),
                      dtype=np.float32, shape=(len(tmp['xOvID'].values), len(unique_orb)))
    A_tracks = (csrA + csrB)
    # print(A_tracks)

    # tmp_orbA = tmp.pivot_table(index='xOvID', columns='orbA', aggfunc='count', fill_value=0)
    # tmp_orbB = tmp.pivot_table(index='xOvID', columns='orbB', aggfunc='count', fill_value=0)
    # tmp_orbA.columns = tmp_orbA.columns.droplevel(0)
    # tmp_orbB.columns = tmp_orbB.columns.droplevel(0)
    # A_tracks = tmp_orbA.combine(tmp_orbB, np.subtract).fillna(0)
    # print(A_tracks)
    # print(A_tracks.columns)
    # print(tracks_rms_df.sort_values(by='track'))
    # A_tracks = csr_matrix(A_tracks.values)

    # reorder tracks (!!!) and extract variances
    huber_threshold_track = 5
    # tmp = tracks_rms_df.sort_values(by='track').pre.abs().values
    # makes sense to use the bias and not the rms (actually the average value would also be fine...) !! remember ABS!!
    tracks_rms_df.sort_values(by='track', inplace=True)
    tmp = tracks_rms_df.bias.abs().values
    huber_weights_track = np.where(tmp > huber_threshold_track, (huber_threshold_track / tmp) ** 1, 1.)
    # tracks_rms_df['huber'] = huber_weights_track
    # print(tracks_rms_df)
    # exit()

    if plot_stuff and XovOpt.get("local"):  # and debug:
        # plot histo
        plt.figure()  # figsize=(8, 3))
        # plt.xlim(-1.*xlim, xlim)
        # the histogram of the data
        num_bins = 'auto'
        n, bins, patches = plt.hist(huber_weights_track.astype(float),
                                    bins=num_bins)  # , cumulative=True)  # , density=True, facecolor='blue',
        # alpha=0.7, range=[-1.*xlim, xlim])
        plt.xlabel('weight (1/m)')
        plt.ylabel('# tracks')
        # plt.title(r'Histogram of dR: $\mu=' + str(mean_dR) + ', \sigma=' + str(std_dR) + '$')
        # # Tweak spacing to prevent clipping of ylabel
        plt.subplots_adjust(left=0.15)
        plt.savefig(XovOpt.get("tmpdir") + '/huber_weights_track.png')
        plt.clf()
        # exit()

    cov_tracks = huber_weights_track.round(2).astype('float16')
    # if cov_tracks element == 0, add 0.001 to avoid inf because of rounding)
    cov_tracks[cov_tracks == 0] += 1.e-3
    # from weights to covariances (inverse)
    np.reciprocal(cov_tracks, out=cov_tracks)
    # convert to sparse (on diagonal)
    cov_tracks = diags(cov_tracks, 0)

    # project variance of individual tracks on xovers
    # print(A_tracks.getnnz() / np.prod(A_tracks.shape))

    cov_xov_tracks = cov_tracks * A_tracks.transpose()  # .round(2)
    # print(cov_xov_tracks)
    # print(cov_xov_tracks.getnnz(), np.prod(cov_xov_tracks.shape), cov_xov_tracks.getnnz() / np.prod(cov_xov_tracks.shape))
    if XovOpt.get("full_covar"):
        cov_xov_tracks = A_tracks * cov_xov_tracks
    else:
        cov_xov_tracks = diags(multiply_sparse_get_diag(A_tracks, cov_xov_tracks))

    np.reciprocal(cov_xov_tracks.data, out=cov_xov_tracks.data)

    return cov_xov_tracks


def get_vce_factor(Ninv, Cinv, x, b=None, A=None, sapr=1., kind='obs'):
    """
    compute vce factor for subset of data or constraint
    see eq. 17-21 of https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1002/jgre.20118
    Lemoine 2013, JGRE
    # from scipy.sparse import csr_matrix, issparse
    # import numpy as np
    :param Ninv: full covariance matrix (inverse of full normal matrix), (npar,npar)
    :param Cinv: weight matrix (inverse of apriori covariance), (nele,nele), nele= npar if constraint, nobs if data
    :param x: solution vector (for iter if kind=obs, total if constraint), (nele,), nele= npar if constraint, nobs if data
    :param b: residuals vector (kind=obs only), (nobs,)
    :param A: partials matrix (kind=obs only), (nobs,npar)
    :param sapr: sigma a priori of the subset, scalar
    :param kind: 'obs' if computing weights for a subset of data, whatever else for a constraint (influences arguments)
    :return: new sigma^2 (inverse of estimated vce weight) associated to the subset of data or constraint
    """

    # A and w should be csr sparse matrices (else multiplication doesn't work, should replace by @)
    if not issparse(A) and A != None:
        A = csr_matrix(A)
    if not issparse(Cinv):
        Cinv = csr_matrix(Cinv)

    # nelem is nobs for a subset of data or nparam for a constraint
    nelem = Cinv.shape[0]
    # a priori squared sigma (inverse of weight associated)
    s2apr = sapr ** 2

    if kind == 'obs':
        ri = (b - A * x)
        Ni = A.T * Cinv * A
    else:
        ri = x
        Ni = Cinv

    # numerator (basically the quantity to minimize)
    rTw = csr_matrix(ri.T) * Cinv
    # print(csr_matrix(ri.T),rTw * ri, Cinv,Cinv.max(),Cinv.min())
    rTwr = (rTw * ri)[0]
    # basically a modified dof for the subset
    redundancy = nelem - (1. / s2apr) * np.trace(Ni @ Ninv)
    print("kind, sqrt(rTwr),redundancy,chi2:", kind, np.sqrt(rTwr), redundancy, np.trace(Ni.todense()),
          rTwr / redundancy)

    # the new sigma^2 associated to the subset of data or constraint
    return rTwr / redundancy


def downsize_xovers(xov_df, max_xovers=1.e5):
    # remove very large dR (>1km)
    xov_df = xov_df.loc[xov_df['dR'].abs() < 1.e3]
    print(xov_df.columns)
    # print(tmp.weights)
    # exit()
    hilat_xov = xov_df.loc[xov_df.LAT >= 60]
    print(hilat_xov[['dR', 'weights', 'huber']].abs().max())
    print(hilat_xov[['dR', 'weights', 'huber']].abs().min())
    print(hilat_xov[['dR', 'weights', 'huber']].abs().mean())
    print(hilat_xov[['dR', 'weights', 'huber']].abs().median())

    # select approx number of xovers to keep and derive proportion to keep at hi-lats
    to_keep = 1. - max_xovers / len(hilat_xov)
    to_keep_hilat = hilat_xov.loc[hilat_xov['weights'] > hilat_xov['weights'].quantile(to_keep)].xOvID.values
    # by default, keep 90% of xovers at low-lats
    lolat_xov = xov_df.loc[xov_df.LAT < 60]
    to_keep_lolat = lolat_xov.loc[lolat_xov['weights'] > lolat_xov['weights'].quantile(0.3)].xOvID.values

    # select very good xovers at LAT>60N OR decent xovers at low latitudes
    selected = xov_df.loc[(xov_df.xOvID.isin(to_keep_hilat)) | (xov_df.xOvID.isin(to_keep_lolat))]
    print(len(selected))
    print(selected[['dR', 'weights', 'huber', 'dist_min_mean']].abs().max())
    print(selected[['dR', 'weights', 'huber', 'dist_min_mean']].abs().min())
    print(selected[['dR', 'weights', 'huber', 'dist_min_mean']].abs().median())
    print(selected[['dR', 'weights', 'huber', 'dist_min_mean']].abs().mean())
    print("Downsized xovers to the 'best'", len(selected), "xovers out of", len(xov_df), ". Done!")

    return selected


def subsample_xovers(xov_df, size_samples=1.e5, rand_seed=0):
    from sklearn.utils import resample

    # use xovers index to identify in sampling
    data = xov_df.index.values
    # round to 10 deg latitude to assign stratification
    lats = np.round(xov_df.LAT.values / 10., 0) * 10.

    # prepare bootstrap sample (no replace cause I just want 1 occurrence per element)
    boot = resample(data, replace=False, n_samples=size_samples, random_state=rand_seed, stratify=lats)

    # check common elements among extractions (approx 12-15% for 500K out of 3.5 mln)
    # sets = []
    # for rand_seed in range(20):
    #
    #     boot = resample(data, replace=False, n_samples=size_samples, random_state=rand_seed, stratify=lats)
    #     sets.append(boot)
    # for idx,list2 in enumerate(sets[:]):
    #     list1_as_set = set(sets[1])
    #     intersection = list1_as_set.intersection(list2)
    #     intersection_as_list = list(intersection)
    #     print(np.array(intersection_as_list))
    #     print(100.*len(intersection_as_list)/len(list2),"% for list ", idx+1)
    # print(boot)
    # print(len(boot))

    return xov_df.iloc[boot]


def get_stats(amat):
    # xover residuals
    w = amat.xov.xovers['dR'].values
    nobs = len(w)
    npar = len(amat.sol_dict['sol'].values())

    lTP = np.hstack(w).reshape(1, -1) @ amat.weights  # hstack converts "object" to "float" for product
    lTPl = lTP @ np.hstack(w).reshape(-1, 1)

    xsol = []
    xstd = []
    # print(amat.sol_dict['sol'])
    print(amat.sol4_pars)
    print(amat.sol_dict['sol'].items())
    for filt in amat.sol4_pars:
        # print(np.array(filt))
        filtered_dict = {k: v for (k, v) in amat.sol_dict['sol'].items() if filt in k}
        # print(filtered_dict)
        if len(list(filtered_dict.values())) > 0:
            xsol.append(list(filtered_dict.values())[0])
        else:
            print(np.array(filt), "not found")
            xsol.append(0.)
        filtered_dict = {k: v for (k, v) in amat.sol_dict['std'].items() if filt in k}
        if len(list(filtered_dict.values())) > 0:
            xstd.append(list(filtered_dict.values())[0])
        else:
            xstd.append(0.)

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
    if len(ATPb) != xT.shape[1]:
        print("removing stuff...", len(ATPb), (xT.shape[1]))
        missing = [x for x in amat.sol4_pars if x not in amat.parNames]
        missing = [amat.sol4_pars.index(x) for x in missing]
        xT = np.delete(xT, missing)
        # remove parameters not in sol4_pars
        tmp_sol4pars = [amat.parNames[x] for x in amat.sol4_pars if x in amat.parNames.keys()]
        ATPb = ATPb[tmp_sol4pars]
        ATP = ATP[tmp_sol4pars, :]
        spA_tmp = amat.spA[:, tmp_sol4pars]
    else:
        spA_tmp = amat.spA
    ##################
    vTPv = lTPl - xT @ ATPb
    print(nobs, npar)
    print("pre-RMS=", np.sqrt(lTPl / (nobs - npar)), " post-RMS=", np.sqrt(vTPv / (nobs - npar)))

    # degrees of freedom in case of constrained least square
    # Atmp = (ATP@spA_tmp + amat.penalty_mat)
    # trR = np.diagonal(np.linalg.pinv(Atmp.todense())@ATP@spA_tmp).sum()
    # dof = nobs - trR
    dof = nobs - npar
    m0 = np.linalg.norm(np.sqrt(vTPv / dof))
    # print("test-m0", m0)

    # alternative method to compute chi2 for constrained least-square (not involving inverse)
    # print("xT",xT)
    # print("pen",amat.penalty_mat)
    # print("weig",amat.weights)
    # xTlP = xT @ amat.penalty_mat
    # xTlPx = xTlP @ xT.T
    # print("check wrmse", vTPv,xTlPx)
    # m0 = np.linalg.norm(np.sqrt((vTPv + xTlPx) / nobs))
    amat.resid_wrmse = m0

    print("Weighted a-posteriori RMS is ", m0.round(4), " - chi2 = ", (m0 / AccOpt.get("sigma_0")).round(4))

    if XovOpt.get("local") and XovOpt.get("debug"):
        plt.figure()  # figsize=(8, 3))
        num_bins = 200  # 'auto'  #
        n, bins, patches = plt.hist((amat.weights @ (np.abs(w).reshape(-1, 1))).astype(float), bins=num_bins,
                                    cumulative=-1, range=[0.1, 50.])
        # n, bins, patches = plt.hist(w.astype(float), bins=num_bins, cumulative=True)
        # plt.xlabel('roughness@baseline700 (m/m)')
        plt.savefig(XovOpt.get("tmpdir") + '/histo_residuals.png')
        plt.clf()

    if False and XovOpt.get("debug"):
        print("xov_xovers_value_count:")
        nobs_tracks = xov.xovers[['orbA', 'orbB']].apply(pd.Series.value_counts).sum(
            axis=1).sort_values(
            ascending=False)
        print(nobs_tracks)


def analyze_dist_vs_dR(xov):
    tmp = xov.xovers.copy()
    tmp['dist_minA'] = xov.xovers.filter(regex='^dist_A.*$').min(axis=1)
    tmp['dist_minB'] = xov.xovers.filter(regex='^dist_B.*$').min(axis=1)
    tmp['dist_min_mean'] = tmp.filter(regex='^dist_min.*$').mean(axis=1)
    tmp.drop(['dist_minA', 'dist_minB'], inplace=True, axis='columns')
    tmp['abs_dR'] = abs(tmp['dR'])


def print_sol(orb_sol, glb_sol, xov, xovi_amat):
    partemplate = set([x.split('/')[1] for x in xovi_amat.sol_dict['sol'].keys()])

    regex = re.compile(".*_dR/d([ACR]|(Rl)|(Pt))[0,1,2,'C','S']?$")
    soltmp = [(x.split('_')[0], 'sol_' + x.split('_')[1], v) for x, v in xovi_amat.sol_dict['sol'].items() if
              regex.match(x)]

    print('-- Solutions -- ')
    if len(soltmp) > 0:
        # formal errors based on inv(ATA) according to scipy.linalg.sparse.lsqr
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

    if len(XovOpt.get("pert_cloop")['glo']) > 0:
        print('to_be_recovered (sim mode, dRl, dPt, dRA, dDEC, dL in arcsec; dPM in arcsec/Julian year)')
        print(XovOpt.get("pert_cloop")['glo'])

    if XovOpt.get("debug") and False:
        _ = xov.remove_outliers('dR', remove_bad=remove_3sigma_median)
        print(xov.xovers.columns)
        print(xov.xovers.loc[xov.xovers.orbA == '1301022345'])
        # print(xov.xovers.loc[xov.xovers.orbA == '1301022341' ].sort_values(by='R_A', ascending=True)[:10])
        if np.sum([x.split('/')[1] in XovOpt.get("parOrb").keys() for x in xovi_amat.sol4_pars]) > 0:
            print(orb_sol.reindex(orb_sol.sort_values(by='sol_dR/dR', ascending=False).index)[:10])
            print(orb_sol.loc[orb_sol.orb == '1301022345', :])


def solve4setup(sol4_glo, sol4_orb, sol4_orbpar, track_names):
    if sol4_orb == [] and sol4_orbpar != [None]:
        sol4_orb = set([i.split('_')[0] for i in track_names])
        sol4_orb = [x for x in sol4_orb if x.isdigit()]
        if sol4_orbpar == []:
            sol4_orbpar = list(XovOpt.get("parOrb").keys())
        sol4_orb = [x + '_' + 'dR/' + y for x in sol4_orb for y in sol4_orbpar]
    elif sol4_orb == [None] or sol4_orbpar == [None]:
        sol4_orb = []
    else:
        sol4_orb = [x + '_' + 'dR/' + y for x in sol4_orb for y in sol4_orbpar]

    if sol4_glo == []:
        sol4_glo = list(XovOpt.get("parGlo").keys())
        sol4_glo = ['dR/' + x for x in sol4_glo]
    elif sol4_glo == [None]:
        sol4_glo = []

    sol4_pars = sorted(sol4_orb) + sorted(sol4_glo)

    if len(sol4_pars) > 0:
        logging.info(f'solving for {len(sol4_pars)} parameters.')
        # print('solving for:',np.array(sol4_pars))
    else:
        logging.error(f'** Solving for {len(sol4_pars)} parameters. Is this correct? Recheck options.')
        XovOpt.display()
        exit()

    return sol4_pars


def analyze_sol(xovi_amat, xov, mode='full'):
    # from accumxov.accum_opt import remove_max_dist

    # print('xovi_amat.sol',xovi_amat.sol)

    # print(parOrb)
    # print([x.split('/')[1] in parOrb.keys() for x in xovi_amat.sol4_pars])
    # print(np.sum([x.split('/')[1] in parOrb.keys() for x in xovi_amat.sol4_pars]))
    # exit()

    # print(len(np.reshape(xovi_amat.sol4_pars, (-1, 1))),len(np.reshape(xovi_amat.sol[0], (-1, 1))),
    #                          len(np.reshape(xovi_amat.sol[-1], (-1, 1))) )
    # check wheter list contains full list of pars or just those from this iter
    if mode == 'full':
        sol4_pars = xovi_amat.sol4_pars
    else:
        sol4_pars = xovi_amat.sol4_pars_iter

    # Ordering is important here, don't use set or other "order changing" functions
    _ = np.hstack((np.reshape(sol4_pars, (-1, 1)),
                   np.reshape(xovi_amat.sol[0], (-1, 1)),
                   np.reshape(xovi_amat.sol[-1], (-1, 1))
                   ))
    sol_dict = {'sol': dict(zip(_[:, 0], _[:, 1].astype(float))),
                'std': dict(zip(_[:, 0], np.sqrt(_[:, 2].astype(float))))}

    if XovOpt.get("debug"):
        print("sol_dict_analyze_sol")
        print(sol_dict)

    # Extract solution for global parameters
    glb_sol = pd.DataFrame(_[[x.split('/')[1] in list(XovOpt.get("parGlo").keys()) for x in _[:, 0]]],
                           columns=['par', 'sol', 'std'])
    partemplate = set([x.split('/')[1] for x in sol_dict['sol'].keys()])
    # regex = re.compile(".*"+str(list(partemplate))+"$")

    # print(sol_dict)

    # Extract solution for orbit parameters
    parOrbKeys = list(XovOpt.get("parOrb").keys())
    solved4 = list(partemplate)

    if XovOpt.get("OrbRep") in ['lin', 'quad', 'per']:
        parOrbKeys = [x + str(y) for x in parOrbKeys for y in [0, 1, 2, 'C', 'S']]
    # solved4orb = list(filter(regex.match, list(parOrb.keys())))
    solved4orb = list(set(parOrbKeys) & set(solved4))

    if len(solved4orb) > 0:
        df_ = pd.DataFrame(_, columns=['key', 'sol', 'std'])
        df_[['orb', 'par']] = df_['key'].str.split('_', expand=True)
        df_ = df_.astype({'sol': 'float64', 'std': 'float64'})
        df_.drop('key', axis=1, inplace=True)
        # df_[['orb','par']] = df_[['par','orb']].where(df_['par'] == None, df_[['orb','par']].values)
        df_ = df_.replace(to_replace='None', value=np.nan).dropna()
        table = pd.pivot_table(df_, values=['sol', 'std'], index=['orb'], columns=['par'], aggfunc=np.sum)

        if any(xov.xovers.filter(like='dist', axis=1)):
            xov.xovers['dist_max'] = xov.xovers.filter(regex='^dist_[A,B].*$').max(axis=1)
            tmp = xov.xovers.copy()
            tmp['dist_minA'] = xov.xovers.filter(regex='^dist_A.*$').min(axis=1)
            tmp['dist_minB'] = xov.xovers.filter(regex='^dist_B.*$').min(axis=1)
            tmp['dist_min_mean'] = tmp.filter(regex='^dist_min.*$').mean(axis=1)
            xov.xovers['dist_min_mean'] = tmp['dist_min_mean'].copy()

            if AccOpt.get("remove_max_dist"):
                xov.xovers = xov.xovers[xov.xovers.dist_max < 0.4]
                # xov.xovers = xov.xovers[xov.xovers.dist_min_mean < 1]

            _ = xov.xovers[['orbA', 'orbB']].apply(pd.Series.value_counts).sum(axis=1)
            table['num_obs'] = _

            df1 = xov.xovers.groupby(['orbA'], sort=False)['dist_max'].max().reset_index()
            df2 = xov.xovers.groupby(['orbB'], sort=False)['dist_max'].max().reset_index()

            merged_Frame = pd.merge(df1, df2, left_on='orbA', right_on='orbB', how='outer')
            merged_Frame['orbA'] = merged_Frame['orbA'].fillna(merged_Frame['orbB'])
            merged_Frame['orbB'] = merged_Frame['orbB'].fillna(merged_Frame['orbA'])
            merged_Frame['dist_max'] = merged_Frame[['dist_max_x', 'dist_max_y']].mean(axis=1)
            merged_Frame = merged_Frame[['orbA', 'dist_max']]
            merged_Frame.columns = ['orb', 'dist_max']

            table.columns = ['_'.join(col).strip() for col in table.columns.values]
            try:
                table = pd.merge(table.reset_index(), merged_Frame, on='orb')
            except:
                table = pd.concat([table.reset_index(), merged_Frame])

        orb_sol = table
        if XovOpt.get("debug"):
            print(orb_sol)

    else:
        orb_sol = pd.DataFrame()

    print("Max values:")
    print(orb_sol.filter(regex='sol_dR/').max(axis=0))
    print("Min values:")
    print(orb_sol.filter(regex='sol_dR/').min(axis=0))
    print("Median values:")
    print(orb_sol.filter(regex='sol_dR/').median(axis=0))
    print("Mean values:")
    print(orb_sol.filter(regex='sol_dR/').mean(axis=0))
    # exit()

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
    glb_sol['std'] = glb_sol['std'].astype('float').values / AccOpt.get("sigma_0")
    for col in orb_sol.filter(regex='std_*').columns:
        orb_sol[col] = orb_sol[col].astype('float').values / AccOpt.get("sigma_0")

    return orb_sol, glb_sol, sol_dict


def load_previous_iter_if_any(ds, ext_iter, xov_cmb):
    from accumxov.Amat import Amat

    # retrieve old solution
    if int(ext_iter) > 0:
        previous_iter = Amat(XovOpt.get("vecopts"))
        # tmp = tmp.load((data_pth + 'Abmat_' + ds.split('/')[0] + '_' + ds.split('/')[1][:-1] + str(ext_iter) + '_' + ds.split('/')[2]) + '.pkl')
        previous_iter = previous_iter.load(
            ('_').join((XovOpt.get("outdir") + ('/').join(ds.split('/')[:-2])).split('_')[:-1]) +
            '_' + str(ext_iter - 1) + '/' +
            ds.split('/')[-2] + '/Abmat_' + ('_').join(ds.split('/')[:-1]) + '.pkl')
        print("initial sol dict=", len(previous_iter.sol_dict['sol']))

        # print(previous_iter.sol_dict)
        # exit()
    # if pre-processing took place (else, if perturbing simulation, also pert_cloop_orb should contain something)
    elif xov_cmb.pert_cloop.shape[1] > 0:  # and len(pert_cloop_orb) == 0:
        parsk = list(xov_cmb.pert_cloop.to_dict().keys())
        trackk = list(xov_cmb.pert_cloop.to_dict()[parsk[0]].keys())

        # pertpar_list = list(pert_cloop_orb.keys())
        pertpar_list = xov_cmb.pert_cloop.columns
        fit2dem_sol = np.ravel([list(x.values()) for x in xov_cmb.pert_cloop.filter(pertpar_list).to_dict().values()])
        fit2dem_keys = [tr + '_dR/' + par for par in parsk for tr in trackk]
        fit2dem_dict = dict(zip(fit2dem_keys, fit2dem_sol))

        previous_iter = Amat(XovOpt.get("vecopts"))
        previous_iter.sol_dict = {'sol': fit2dem_dict, 'std': dict(zip(fit2dem_keys, np.zeros(len(fit2dem_keys))))}
    # not first iter, nor pre-processing
    else:
        previous_iter = None  # Amat(vecopts)
        # previous_iter.sol_dict_iter = previous_iter.sol_dict
    return previous_iter
