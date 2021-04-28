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

from src.accumxov.accum_opt import remove_max_dist, remove_3sigma_median, remove_dR200, downsize, clean_part, huber_threshold, \
    distmax_threshold, offnad_threshold, h2_limit_on, sigma_0, convergence_criteria, sampling,compute_vce
from src.accumxov.accum_utils import get_xov_cov_tracks, get_vce_factor, downsize_xovers, get_stats, print_sol, solve4setup, \
    analyze_sol, subsample_xovers, load_previous_iter_if_any
from src.xovutil.iterables import mergsum
from src.xovutil.stat import rms
from src.xovutil.xovres2weights import get_interpolation_weight
# from xov_utils import get_tracks_rms
from src.pyxover.xov_utils import load_combine, clean_xov, clean_partials

warnings.filterwarnings("ignore", category=RuntimeWarning)

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
from scipy.sparse.linalg import lsqr
import scipy.sparse.linalg as spla
import scipy.linalg as la

# mylib
# from mapcount import mapcount
# from examples.MLA.options import XovOpt.get("debug"), XovOpt.get("outdir"), XovOpt.get("tmpdir"), XovOpt.get("local"), XovOpt.get("partials"), XovOpt.get("sol4_glo"), XovOpt.get("sol4_orbpar"), \
#     pert_cloop, XovOpt.get("OrbRep"), XovOpt.get("vecopts")
from config import XovOpt

from src.pyxover.xov_setup import xov
from src.accumxov.Amat import Amat

########################################
# test space
#
# #exit()

########################################

######## SUBROUTINES ##########

def prepro(dataset):
    # read input args
    # print('Number of arguments:', len(sys.argv), 'arguments.')
    # print('Argument List:', str(sys.argv))

    # locate data
    # locate data
    if XovOpt.get("local") == 0:
        data_pth = XovOpt.get("outdir")
        data_pth += dataset
        # load kernels
    else:
        data_pth = XovOpt.get("outdir")
        data_pth += dataset
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
    return data_pth, XovOpt.get("vecopts")

# #@profile
def prepare_Amat(xov, vecopts, par_list=''):

    # xov.xovers = xov.xovers[xov.xovers.orbA=='1301042351']
    # xov.xovers.append(xovtmp[xovtmp.orbA=='1301011544'])
    # xov.xovers.append(xovtmp[xovtmp.orbB=='1301042351'])
    # xov.xovers.append(xovtmp[xovtmp.orbB=='1301011544'])

    # exit()

    clean_xov(xov, par_list)

    xovtmp = xov.xovers.copy()

    if XovOpt.get("partials"):
        # simplify and downsize
        if par_list == '':
            par_list = xov.xovers.columns.filter(regex='^dR.*$')
        df_orig = xov.xovers[par_list]
        df_float = xov.xovers.filter(regex='^dR.*$').apply(pd.to_numeric, errors='ignore') #, downcast='float')
        xov.xovers = pd.concat([df_orig, df_float], axis=1)
        xov.xovers.info(memory_usage='deep')
        if XovOpt.get("debug"):
            pd.set_option('display.max_columns', 500)
            print(xov.xovers)

        if XovOpt.get("OrbRep") in ['lin', 'quad']:
            xovtmp = xov.upd_orbrep(xovtmp)
            # print(xovi_amat.xov.xovers)
            xov.parOrb_xy = xovtmp.filter(regex='^dR/[a-zA-Z0-9]+_.*$').columns.values

        xovi_amat = Amat(vecopts)
        xov.xovers = xovtmp.copy()

        xovi_amat.setup(xov)

    else:

        xovi_amat = Amat(vecopts)

        xovi_amat.xov = xov
        xovi_amat.xov.xovers = xovtmp.copy()

    return xovi_amat

# #@profile
def prepro_weights_constr(xovi_amat, previous_iter=None):
    from scipy.sparse import csr_matrix
    # from examples.MLA.options import XovOpt.get("par_constr"), XovOpt.get("mean_constr"), XovOpt.get("sol4_orb"), XovOpt.get("sol4_glo")
    from config import XovOpt

    # Solve
    # if not local:
    #     sol4_glo = ['dR/dRA', 'dR/dDEC', 'dR/dPM','dR/dL','dR/dh2'] # [None] # used on pgda, since prOpt badly read
    # test: if converged, also solve for h2
    # if previous_iter.converged:
    #     print("Adding h2 to sol4_glo as solution converged...")
    #     sol4_glo.extend(['dR/dh2'])

    sol4_pars = solve4setup(XovOpt.get("sol4_glo"), XovOpt.get("sol4_orb"), XovOpt.get("sol4_orbpar"), xovi_amat.parNames.keys())
    # print(xovi_amat.parNames)
    # for key, value in sorted(xovi_amat.parNames.items(), key=lambda x: x[0]):
    #     print("{} : {}".format(key, value))

    if XovOpt.get("OrbRep")  in ['lin', 'quad']:
        # xovi_amat.xov.xovers = xovi_amat.xov.upd_orbrep(xovi_amat.xov.xovers)
        # print(xovi_amat.xov.xovers)
        regex = re.compile(".*_dR/d[A,C,R]$")
        const_pars = [x for x in sol4_pars if not regex.match(x)]
        sol4_pars = [x+str(y) for y in range(2) for x in list(filter(regex.match, sol4_pars))]
        sol4_pars.extend(const_pars)

    # Initialize list of solved for parameters (first one will be updated with full list
    # from previous iterations if needed)
    xovi_amat.sol4_pars = sol4_pars
    xovi_amat.sol4_pars_iter = sol4_pars

    # exit()

    if sol4_pars != []:
        if XovOpt.get("debug"):
            print(sol4_pars)
            print(xovi_amat.parNames)
            print([xovi_amat.parNames[p] for p in sol4_pars])
        # select columns of design matrix corresponding to chosen parameters to solve for
        spA_sol4 = xovi_amat.spA[:,[xovi_amat.parNames[p] for p in sol4_pars]]
        # set b=0 for rows not involving chosen set of parameters
        nnz_per_row = spA_sol4.getnnz(axis=1)
        xovi_amat.b[np.where(nnz_per_row == 0)[0]] = 0
    else:
        spA_sol4 = xovi_amat.spA

    # print("sol4pars:", np.array(sol4_pars))
    # print(spA_sol4)

    # screening of partial derivatives (downweights data)
    nglbpars = len([i for i in XovOpt.get("sol4_glo") if i])
    if nglbpars>0 and clean_part:
        xovi_amat.b, spA_sol4 = clean_partials(xovi_amat.b, spA_sol4, threshold = 1.e6, glbpars=XovOpt.get("sol4_glo"))
        # pass

    # WEIGHTING TODO refactor to separate method
    # after convergence of residuals RMS at 1%, fix weights and bring parameters to convergence
    if previous_iter != None and previous_iter.converged:
        print("Weights are fixed as solution converged to 5%")
        # associate weights of old solution to corresponding xov of current one (id by tracks, supposing uniqueness)
        if True: # only if issues with duplicates
            previous_iter.xov.xovers = previous_iter.xov.xovers.drop(columns=['xOvID', 'xovid'],errors='ignore').drop_duplicates().reset_index().rename(columns={"index": "xOvID"})
        tmp_prev_trk = pd.DataFrame(previous_iter.xov.xovers['orbA']+previous_iter.xov.xovers['orbB'],columns = ['trksid'])
        tmp_prev_trk['weights'] = previous_iter.xov.xovers.weights.values
        if True: # only if issues with duplicates
            tmp_prev_trk.drop_duplicates(inplace=True)
        else:
            tmp_prev_trk['weights'] = previous_iter.weights.diagonal().T
            print("len(tmp_prev_trk)",len(tmp_prev_trk))
            print("len(xovi_amat.xov.xovers)",len(xovi_amat.xov.xovers))

        tmp_xov_trk = pd.DataFrame(xovi_amat.xov.xovers['orbA']+xovi_amat.xov.xovers['orbB'],columns = ['trksid'])

        obs_weights = pd.merge(tmp_xov_trk, tmp_prev_trk, how='left', on=['trksid'])
        obs_weights = diags(obs_weights['weights'].fillna(0).values)

        # unsafe if list of xov is different or ordered differently
        # obs_weights = previous_iter.weights
    else:
        # compute huber weights (1 if x<huber_threshold, (huber_threshold/abs(dR))**2 if abs(dR)>huber_threshold)
        if not remove_max_dist and not remove_3sigma_median and not remove_dR200:
            tmp = xovi_amat.xov.xovers.dR.abs().values
            huber_weights = np.where(tmp > huber_threshold, (huber_threshold / tmp) ** 1, 1.)

        if XovOpt.get("debug") and not remove_max_dist and not remove_3sigma_median and not remove_dR200:
            print("Apply Huber weights (resid)")
            print(tmp[tmp > huber_threshold])
            print(np.sort(huber_weights[huber_weights<1.]),np.mean(huber_weights))

        # same but w.r.t. distance
        if not remove_max_dist and not remove_3sigma_median and not remove_dR200:
            tmp = xovi_amat.xov.xovers.dist_max.values
            huber_weights_dist = np.where(tmp > distmax_threshold, (distmax_threshold / tmp) ** 2, 1.)

        if XovOpt.get("debug") and not remove_max_dist and not remove_3sigma_median and not remove_dR200:
            print("Apply Huber weights (dist)")
            print(tmp[tmp > distmax_threshold])
            print(np.sort(huber_weights_dist[huber_weights_dist<1.]),np.mean(huber_weights_dist))

        # same but w.r.t. offnadir
        if not remove_max_dist and not remove_3sigma_median and not remove_dR200:
            tmp = np.nan_to_num(xovi_amat.xov.xovers.filter(regex='offnad').values)
            tmp = np.max(np.abs(tmp),axis=1)
            huber_weights_offnad = np.where(tmp > offnad_threshold, (offnad_threshold / tmp) ** 1, 1.)

        if XovOpt.get("debug") and not remove_max_dist and not remove_3sigma_median and not remove_dR200:
            print("Apply Huber weights (offnad)")
            print(tmp[tmp > offnad_threshold])
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
            if XovOpt.get("debug"):
                print("pre xovcov types",tmp.dtypes)

            weights_xov_tracks = get_xov_cov_tracks(df=tmp,plot_stuff=True)
            xovi_amat.xov.xovers['huber_trks'] = weights_xov_tracks.diagonal()


            # the histogram of weight distribution
            if False and XovOpt.get("local") and XovOpt.get("debug"):
                tmp = weights_xov_tracks.diagonal()

                plt.figure() #figsize=(8, 3))
                num_bins = 100 # 'auto'  # 40  # v
                n, bins, patches = plt.hist(tmp.astype(np.float), bins=num_bins)
                plt.xlabel('dR (m)')
                plt.ylabel('# tracks')
                plt.savefig(XovOpt.get("tmpdir") + '/histo_tracks_weights.png')
                plt.clf()

            # xovi_amat.xov.xovers['huber'] *= huber_weights_track

            if XovOpt.get("debug") and False:
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

                if XovOpt.get("debug") and XovOpt.get("local"):
                    num_bins = 100 #'auto'
                    plt.clf()
                    n, bins, patches = plt.hist(np.where(huber_penal < 1., huber_penal, 1.).astype(np.float), bins=num_bins,cumulative=True)
                    plt.xlabel('huber_penal')
                    plt.savefig(XovOpt.get("tmpdir") + '/histo_huber_h2.png')
                    plt.clf()

    #######

        # interp_weights = get_weight_regrough(xovi_amat.xov).reset_index()  ### old way using residuals to extract roughness
        #
        # get interpolation error based on roughness map (if available at given latitude) + minimal distance
        interp_weights = get_interpolation_weight(xovi_amat.xov).reset_index()

        val = interp_weights['weight'].values # np.ones(len(interp_weights['weight'].values)) #
        # print("interp error values", np.sort(val))
        xovi_amat.xov.xovers['interp_weight'] = val

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
        print("Observations weights re-evaluated, solution has not converged yet")

        if XovOpt.get("debug") and XovOpt.get("local"):
            print("tracks weights", weights_xov_tracks.diagonal().mean(), np.sort(weights_xov_tracks.diagonal()))
            tmp = obs_weights.diagonal()
            tmp = np.where(tmp>1.e-9,tmp,0.)
            print(np.sort(tmp),np.median(tmp),np.mean(tmp))
            # plot histo
            plt.figure() #figsize=(8,3))
            # plt.xlim(-1.*xlim, xlim)
            # the histogram of the data
            num_bins = 200 #'auto'
            n, bins, patches = plt.hist(tmp, bins=num_bins, range=[1.e-4,4.e-2]) #, cumulative=-1) #, density=True, facecolor='blue',
            # alpha=0.7, range=[-1.*xlim, xlim])
            plt.xlabel('obs weights')
            plt.ylabel('# tracks')
            plt.title('Resid+distance+offnadir+interp+weights: $\mu=' + str(np.mean(tmp)) + ', \sigma=' + str(np.std(tmp)) + '$')
            # # Tweak spacing to prevent clipping of ylabel
            plt.subplots_adjust(left=0.15)
            plt.savefig(XovOpt.get("tmpdir") + '/data_weights.png')
            plt.clf()
            # exit()

    # Combine and store weights
    xovi_amat.weights = obs_weights
    print(len(xovi_amat.xov.xovers))
    print(obs_weights.shape)
    xovi_amat.xov.xovers['weights'] = xovi_amat.weights.diagonal()

    ## DIRECT SOLUTION FOR DEBUG AND SMALL PROBLEMS (e.g., global only)
    if len(sol4_pars)<50 and XovOpt.get("debug"):

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
    if h2_limit_on and XovOpt.get("debug") and XovOpt.get("local"):
        print(xovi_amat.xov.xovers.columns)
        tmp = xovi_amat.xov.xovers[['dR','dR/dh2','LON','LAT','weights']]
        # print("truc0",tmp['weights'].abs().min(),tmp['weights'].abs().max())
        tmp = tmp.loc[(tmp.dR.abs() < limit_h2) & (tmp['dR/dh2'].abs() > 0.3) & (tmp['weights'].abs() > 0.5 * sigma_0)]
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
        plt.savefig(XovOpt.get("tmpdir") + "discr_vs_dwdh2.png")

        plt.clf()
        piv = pd.pivot_table(tmp.round({'LON':0,'LAT':0}), values="dR/dh2", index=["LAT"], columns=["LON"],
                             fill_value=None, aggfunc=rms)
        ax = sns.heatmap(piv, xticklabels=10, yticklabels=10, cmap="YlGnBu") #, square=False, annot=True)
        plt.tight_layout()
        ax.invert_yaxis()
        plt.savefig(XovOpt.get("tmpdir") + "geo_dwdh2.png")
        # exit()

    # analysis of partial derivatives to check power in obs & param
    if XovOpt.get("debug") and XovOpt.get("local"):

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
        plt.savefig(XovOpt.get("tmpdir") + "partials_histo_weighted.png")

        plt.clf()
        for idx,par in enumerate([dw_dpm,dw_dl,dw_ddec,dw_dra,dw_dh2]):
            tmp = np.abs(par)
            n, bins, patches = plt.hist(tmp, bins=num_bins, density=False,
                                        alpha=0.8,label=parnam[idx])
        plt.semilogy()
        plt.legend()
        plt.ylabel('# of obs')
        plt.xlabel('meters/[par]')
        plt.savefig(XovOpt.get("tmpdir") + "partials_histo.png")
        # exit()

    # svd analysis of parameters (eigenvalues and eigenvectors)
    if XovOpt.get("debug") and XovOpt.get("local"): #len(sol4_pars) < 50 and debug:

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
            plt.savefig(XovOpt.get("tmpdir") + "test_lambdaS_" + str(idx) + ".png")
            print('Vh transp')
            print(Vh.T)
            plt.clf()
            plt.imshow(Vh.T,cmap='bwr')
            plt.colorbar()
            plt.savefig(XovOpt.get("tmpdir") + "test_svd_" + str(idx) + ".png")

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
    if XovOpt.get("OrbRep") in ['lin', 'quad'] :
        for par in ['dA','dC','dR']:
            if par in XovOpt.get("sol4_orbpar"):
                XovOpt.get("par_constr")['dR/' + par + '0'] = XovOpt.get("par_constr").pop('dR/' + par)

    par_constr = {your_key: XovOpt.get("par_constr")[your_key] for your_key in mod_par}

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
            weights_mean = np.mean(xovi_amat.weights.diagonal())
            # n_goodobs_tracks = xovi_amat.xov.xovers.loc[xovi_amat.weights.diagonal() > 0.5*sigma_0][['orbA', 'orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(
            n_goodobs_tracks = xovi_amat.xov.xovers.loc[xovi_amat.weights.diagonal() > 0.1*weights_mean][['orbA', 'orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(
            ascending=False)
        else:
            n_goodobs_tracks = xovi_amat.xov.xovers[['orbA', 'orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(
            ascending=False)

        to_constrain = [idx for idx, p in enumerate(sol4_pars) if p.split('_')[0] in n_goodobs_tracks[n_goodobs_tracks < 10].index if p.split('_')[0][:2]!='08'] # exclude flybys from this, else orbits are never improved
        # to_constrain = [idx for idx, p in enumerate(sol4_pars) if p.split('_')[0] in n_goodobs_tracks[n_goodobs_tracks < 1].index if p.split('_')[0][:2]!='08'] # exclude flybys from this, else orbits are never improved

        xovi_amat.to_constrain = to_constrain
        if XovOpt.get("debug"):
            print("number of constrained pars", len(to_constrain))
            print(to_constrain)
            print([dict(zip(xovi_amat.parNames.values(), xovi_amat.parNames.keys()))[x] for x in to_constrain])
            # exit()

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
                csr_matrix((np.power(sigma_0 / val, 2), (row, col)), dtype=np.float32, shape=(len(sol4_pars), len(sol4_pars))))
    # combine all constraints
    penalty_matrix = sum(csr)
    # print(penalty_matrix)

    # does not work with only one set of orbit parameters solved
    # if len(sol4_orb)>1 and True:
    if True:
        csr_avg = []
        for constrain in XovOpt.get("mean_constr").items():
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
                    csr_avg.append(csr_matrix((vals*np.power(sigma_0 / constrain[1], 2),
                                               (rowcols_nodiag[:,0],rowcols_nodiag[:,1])),
                                              dtype=np.float32, shape=(len(sol4_pars), len(sol4_pars))))
                    vals = (1 - 1/len(parindex[:,0])) * np.ones(len(parindex[:,0]))
                    csr_avg.append(csr_matrix((vals*np.power(sigma_0 / constrain[1], 2),
                                               (rowcols_diag[:,0],rowcols_diag[:,1])),
                                              dtype=np.float32, shape=(len(sol4_pars), len(sol4_pars))))
        # print(sum(csr_avg))
        # print(penalty_matrix)

        # penalty_matrix = penalty_matrix + sum(csr_avg)
        penalty_matrix_avg = sum(csr_avg)
        xovi_amat.penalty_mat_avg = penalty_matrix_avg

    # store penalties into amat
    xovi_amat.penalty_mat = penalty_matrix

    # store cleaned and weighted spA
    xovi_amat.spA_weight_clean = spA_sol4

    if not remove_max_dist and not remove_3sigma_median and not remove_dR200:
        tmp = obs_weights.diagonal()
        avg_weight = np.mean(tmp)
        print("Weights (avg, med, std):", avg_weight, np.median(tmp), np.std(tmp))
        print("Fully weighted obs (>0.5*mean(weight)): ", len(tmp[tmp>0.5*avg_weight]), "or ",len(tmp[tmp>0.5*avg_weight])/len(tmp)*100.,"%")
        print("Slightly downweighted obs: ", len(tmp[(tmp<0.5*avg_weight)*(tmp>0.05*avg_weight)]), "or ",len(tmp[(tmp<0.5*avg_weight)*(tmp>0.05*avg_weight)])/len(tmp)*100.,"%")
        print("Brutally downweighted obs (<0.05*sigma0): ", len(tmp[(tmp<0.05*avg_weight)]), "or ",len(tmp[(tmp<0.05*avg_weight)])/len(tmp)*100.,"%")

def compute_vce_weights(amat):

    xsol = []
    for filt in amat.sol4_pars_iter:
        filtered_dict = {k: v for (k, v) in amat.sol_dict['sol'].items() if filt in k}
        xsol.append(list(filtered_dict.values())[0])
    xsol = np.array(xsol)

    xsol_iter = []
    for filt in amat.sol4_pars_iter:
        filtered_dict = {k: v for (k, v) in amat.sol_dict_iter['sol'].items() if filt in k}
        if len(list(filtered_dict.values()))>0:
            xsol_iter.append(list(filtered_dict.values())[0])
    xsol_iter = np.array(xsol_iter)

    lP = amat.penalty_mat
    # print("len xsol", len(xsol),len(xsol_iter),amat.penalty_mat.shape)

    s2_obs_apr = 1./amat.vce[0]
    s2_constr_apr = 1./amat.vce[1]
    s2_constr_avg_apr = 1./amat.vce[2]

    # compute total N**-1
    # select only columns with solved for parameters
    Amat = amat.spA[:,[amat.parNames[p] for p in amat.sol4_pars if p in amat.parNames.keys()]]

    ATP = Amat.T * amat.weights
    ATPA = ATP * Amat

    N = ((1./s2_obs_apr)*ATPA + (1./s2_constr_apr)*lP + (1./s2_constr_avg_apr)*amat.penalty_mat_avg).todense()
    Ninv = np.linalg.pinv(N,hermitian=True,rcond=1.e-20) #

    if not np.allclose(N, N@(Ninv@N)):
        print('### N is almost singular!! Help!!!')

    # TODO this should be handled differently!!!
    # if len(xsol)!=len(xsol_iter):
    #     print("### updating xsol=xsol_iter for VCE")
    #     xsol = xsol_iter
    #     print('then xsol=\n',xsol)
    # print("len of full and iter sol",len(amat.sol_dict['sol']),len(amat.sol_dict_iter['sol']))
    # print('xTx=', xsol.T@xsol,'xTx_iter=', xsol_iter.T@xsol_iter)

    # print("Computing VCE weights:")
    s2_obs_new = get_vce_factor(b=amat.b,A=Amat,x=xsol_iter,Cinv=amat.weights,Ninv=Ninv,sapr=1./np.sqrt(amat.vce[0]),kind='obs')
    s2_constr_new = get_vce_factor(b=0.,A=0.,x=xsol,Cinv=lP,Ninv=Ninv,sapr=1./np.sqrt(amat.vce[1]),kind='constr')
    s2_constr_avg_new = get_vce_factor(b=0.,A=0.,x=xsol,Cinv=amat.penalty_mat_avg,Ninv=Ninv,sapr=1./np.sqrt(amat.vce[2]),kind='constr_avg')

    # print("avg weight:", 1./s2_constr_avg)

    # print("total redundancy:", redundancy_obs+redundancy_constr, nobs)
    # print("old weights (obs, constr, avg)", 1./s2_obs_apr, 1./s2_constr_apr, 1./s2_constr_avg_apr)
    # print("new weights (obs, constr, avg)", 1./s2_obs_new, 1./s2_constr_new, 1./s2_constr_avg_new)
    # exit()

    # new weights have to be > 0
    assert s2_obs_new > 0
    assert s2_constr_new > 0
    assert s2_constr_avg_new > 0

    # exit()

    return s2_obs_new, s2_constr_new, s2_constr_avg_new

######## MAIN ##########
def main(arg):

    ##############################################
    # launch program and clock
    # -----------------------------
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
        #
        # # count occurrences for each orbit ID
        # xov_cmb.nobs_x_track = xov_cmb.xovers[['orbA','orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(ascending=False)
        # print(_)
        # print(_.dtypes)
        # exit()

        if XovOpt.get("partials"):
            # load previous iter from disk (orbs, sols, etc) if available
            previous_iter = load_previous_iter_if_any(ds, ext_iter, xov_cmb)
            if ext_iter > 0 and previous_iter.converged and 'dR/dh2' not in XovOpt.get("sol4_glo"):
                print("Adding h2 to sol4_glo as solution converged...")
                XovOpt.get("sol4_glo").extend(['dR/dh2'])

            # solve dataset
            par_list = ['orbA', 'orbB', 'xOvID']
            xovi_amat = prepare_Amat(xov_cmb, vecopts, par_list)

            if (downsize or sampling) and ext_iter==0:
                # downsize dataset by removing worst weighted data (mostly at high latitudes)
                # also removes very bad xovers with dR > 1km
                max_xovers = 8.5e5 # 3.e6
                if downsize and len(xovi_amat.xov.xovers)>max_xovers:
                    # actually preparing weights and constraints for the solution (weights are needed for downsampling)
                    prepro_weights_constr(xovi_amat, previous_iter=previous_iter)
                    # downsize
                    xovi_amat.xov.xovers = downsize_xovers(xovi_amat.xov.xovers,max_xovers=max_xovers)

                    # reset weights and sol4pars variables for modified dataset
                    xovi_amat.xov.combine([xovi_amat.xov])
                    xovi_amat = prepare_Amat(xovi_amat.xov, vecopts, par_list)

                # subsample with replacement for bootstrap test
                if sampling and ext_iter==0:
                    # get seed as experiment name
                    rand_seed = int(datasets[0].split('/')[-3].split('_')[0][-1])
                    xovi_amat.xov.xovers = subsample_xovers(xovi_amat.xov.xovers, size_samples=5.e5, rand_seed=rand_seed)

                    # reset weights and sol4pars variables for modified dataset
                    xovi_amat.xov.combine([xovi_amat.xov])
                    xovi_amat = prepare_Amat(xovi_amat.xov, vecopts, par_list)

            # actually preparing weights and constraints for the solution
            prepro_weights_constr(xovi_amat, previous_iter=previous_iter)

            if previous_iter != None and previous_iter.vce != None:
                xovi_amat.vce = previous_iter.vce
            else:
                weight_obs = 1.e-3
                weight_constr = 5.
                weight_constr_avg = 1.e-2

                xovi_amat.vce = [weight_obs, weight_constr, weight_constr_avg]
                # xovi_amat.vce = [0.0002247404434024504, 5.0025679108113685, 0.0010878786212904351]

            keep_iterating_vce = True
            for i in (i for i in range(10) if keep_iterating_vce):

                print("iter+weights obs/constr", i, xovi_amat.vce)

                ######################### MAKE NEW SOLVE ROUTINE (rename the one above as prepro_weights_constr())
                weight_obs = xovi_amat.vce[0]
                sqrt_weight_obs = np.sqrt(xovi_amat.vce[0])
                weight_constr = xovi_amat.vce[1]
                weight_constr_avg = xovi_amat.vce[2]
                # sqrt_weight_constr = np.sqrt(xovi_amat.vce[1])
                # sqrt_weight_constr = np.sqrt(xovi_amat.vce[2])
                # Choleski decompose matrix and append to design matrix (weight_constr applied except for constrain on avg)
                Q = np.linalg.cholesky((weight_constr*xovi_amat.penalty_mat+weight_constr_avg*xovi_amat.penalty_mat_avg).todense())

                # add penalisation to residuals
                if previous_iter != None and previous_iter.sol_dict != None:
                    # get previous solution reordered as sol4_pars_iter (and hence as Q) - contains the full solution but only for the
                    # parameters also solved in this iteration (and hence consistent with Q, else it crashes).
                    # should this rather be sol_dict?? Do we want to constrain the correction amplitude at each iter or the full correction?
                    if i == 0: # no need to update this at each iter
                        prev_sol_ord = [previous_iter.sol_dict['sol'][key] if
                                    key in previous_iter.sol_dict['sol'] else 0. for key in xovi_amat.sol4_pars_iter]

                    b_penal = np.hstack([sqrt_weight_obs * xovi_amat.weights * xovi_amat.b,
                                         -1. * np.ravel(
                                             np.dot(Q, prev_sol_ord))])
                                        # np.zeros(len(xovi_amat.sol4_pars_iter)))]) #
                else:
                    b_penal = np.hstack([sqrt_weight_obs * xovi_amat.weights * xovi_amat.b,
                                         -1. * np.zeros(len(xovi_amat.sol4_pars_iter))])

                # add penalisation to partials matrix
                spA_sol4_penal = scipy.sparse.vstack([sqrt_weight_obs * xovi_amat.spA_weight_clean, 1. * csr_matrix(Q)])

                # save penalised matrices
                xovi_amat.spA_penal = spA_sol4_penal
                xovi_amat.b_penal = b_penal

                # solve using lsqr
                xovi_amat.sol = lsqr(xovi_amat.spA_penal, xovi_amat.b_penal, damp=0, show=False, iter_lim=100000, atol=1.e-8 / sigma_0,
                                     btol=1.e-8 / sigma_0, calc_var=True)
                # xovi_amat.sol = lsqr(xovi_amat.spA, xovi_amat.b,damp=0,show=True,iter_lim=100000,atol=1.e-8,btol=1.e-8,calc_var=True)

                # print("sol sparse: ",xovi_amat.sol[0])
                # print('to_be_recovered', pert_cloop['glo'])
                print("Solution iters terminated with ", xovi_amat.sol[1])
                if xovi_amat.sol[1] != 2:
                    exit(2)
                #############################################

                # Save to pkl
                orb_sol, glb_sol, sol_dict = analyze_sol(xovi_amat, xov_cmb, mode='iter')

                # check std of orbital parameters for systematics
                if XovOpt.get("debug"):
                    testA = pd.DataFrame.from_dict(sol_dict).filter(like='dR/dA', axis=0).loc[:,'std']
                    testC = pd.DataFrame.from_dict(sol_dict).filter(like='dR/dC', axis=0).loc[:,'std']
                    testR = pd.DataFrame.from_dict(sol_dict).filter(like='dR/dR', axis=0).loc[:,'std']
                    plt.figure(figsize=(8, 3))
                    testA.plot()
                    testC.plot()
                    testR.plot()
                    # plt.xlabel('dR (m)')
                    # plt.ylabel('Probability')
                    # plt.title(r'Histogram of dR: $\mu=' + str(mean_dR) + ', \sigma=' + str(std_dR) + '$')
                    # # Tweak spacing to prevent clipping of ylabel
                    # plt.subplots_adjust(left=0.15)
                    plt.savefig(XovOpt.get("tmpdir") + '/orbpart_vs_time.png')
                    plt.clf()

                # remove corrections OF SINGLE ITER if "unreasonable" (larger than 100 meters in any direction, or 50 meters/day, or 20 arcsec)
                sol_dict_iter = sol_dict
                # print("Length of original sol", len(sol_dict_iter['sol']))

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

                    if max_orb_corr > 250 or max_orb_drift_corr > 50 or max_att_corr > 2.:
                        # print("Solution fixed for track", tr, 'with max_orb_corr,max_orb_drift_corr,max_att_corr:',max_orb_corr, max_orb_drift_corr, max_att_corr)
                        sol_dict_iter_clean.append(dict.fromkeys(soltmp, 0.))
                        bad_count += 1
                    else:
                        # pass
                        sol_dict_iter_clean.append(soltmp)
                    # keep std also for bad orbits
                    std_dict_iter_clean.append(stdtmp)

                # add back global parameters
                if len(XovOpt.get("sol4_glo"))>0:
                    sol_dict_iter_clean.append(dict([(x,v) for x, v in sol_dict_iter['sol'].items() if x in XovOpt.get("sol4_glo")]))
                    std_dict_iter_clean.append(dict([(x,v) for x, v in sol_dict_iter['std'].items() if x in XovOpt.get("sol4_glo")]))

                sol_dict_iter_clean = {k: v for d in sol_dict_iter_clean for k, v in d.items()}
                std_dict_iter_clean = {k: v for d in std_dict_iter_clean for k, v in d.items()}
                sol_dict_iter_clean = dict(zip(['sol','std'],[sol_dict_iter_clean,std_dict_iter_clean]))
                print("New length of cleaned sol",len(sol_dict_iter_clean['sol'])-bad_count)

                xovi_amat.sol_dict = sol_dict_iter_clean
                # exit()

                if XovOpt.get("debug"):
                    pd.set_option('display.max_rows', None)
                    pd.set_option('display.max_columns', None)
                    pd.set_option('display.width', None)
                    pd.set_option('display.max_colwidth', -1)

                get_stats(amat=xovi_amat)

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

                # Cumulate with solution from previous iter (or from pre-processing)
                if previous_iter != None: #(int(ext_iter) > 0) and (previous_iter != None):
                    if previous_iter.sol_dict != None:
                        # def dict2np(x):
                        #     return np.array(list(x.values()))
                        # print("test xTx update -soldictiter- pre=",np.sqrt(dict2np(xovi_amat.sol_dict_iter['sol']).T@dict2np(xovi_amat.sol_dict_iter['sol'])))
                        # print("test xTx update -prevsol- pre=",np.sqrt(dict2np(previous_iter.sol_dict['sol']).T@dict2np(previous_iter.sol_dict['sol'])))
                        # sum the values with same keys
                        updated_sol = mergsum(xovi_amat.sol_dict_iter['sol'],previous_iter.sol_dict['sol'])
                        updated_std = mergsum(xovi_amat.sol_dict_iter['std'],previous_iter.sol_dict['std'].fromkeys(previous_iter.sol_dict['std'], 0.))
                        # print("test xTx update -soldictiter- post=",np.sqrt(dict2np(xovi_amat.sol_dict_iter['sol']).T@dict2np(xovi_amat.sol_dict_iter['sol'])))
                        # print("test xTx update -prevsol- post=",np.sqrt(dict2np(previous_iter.sol_dict['sol']).T@dict2np(previous_iter.sol_dict['sol'])))
                        # print("test xTx update post=",np.sqrt(dict2np(updated_sol).T@dict2np(updated_sol)))

                        # save total list of parameters (previous iters + current)
                        xovi_amat.sol4_pars = list(updated_sol.keys())
                        xovi_amat.sol_dict = {'sol': updated_sol, 'std' : updated_std}
                        # use dict to update amat.sol, keep std
                        xovi_amat.sol = (list(xovi_amat.sol_dict['sol'].values()), *xovi_amat.sol[1:-1], list(xovi_amat.sol_dict['std'].values()))
                        orb_sol, glb_sol, sol_dict = analyze_sol(xovi_amat, xov_cmb, mode='full')
                        print("Cumulated solution")
                        print_sol(orb_sol, glb_sol, xov, xovi_amat)
                    else:
                        print("previous_iter.sol_dict=",previous_iter.sol_dict)

                # VCE
                if compute_vce:
                    sigma2_obs, sigma2_constr, sigma2_constr_avg = compute_vce_weights(amat=xovi_amat)

                    keep_iterating_vce = (np.abs(weight_obs-1./sigma2_obs)/weight_obs > 0.01)+\
                                         (np.abs(weight_constr-1./sigma2_constr)/weight_constr > 0.01)+\
                                         (np.abs(weight_constr_avg-1./sigma2_constr_avg)/weight_constr_avg > 0.01)
                    # print(keep_iterating_vce)
                    if keep_iterating_vce:
                        print("vce iter,weight (obs,constr):", i, xovi_amat.vce,keep_iterating_vce)
                        print("w_obs updated by",(np.abs(weight_obs-1./sigma2_obs)/weight_obs*100.),'% and w_constr by',\
                                         (np.abs(weight_constr-1./sigma2_constr)/weight_constr*100.),'% and w_constr_avg by',\
                                         (np.abs(weight_constr_avg-1./sigma2_constr_avg)/weight_constr_avg*100.),'%')
                        # update weights
                        weight_obs =  1./sigma2_obs
                        weight_constr = 1./sigma2_constr
                        weight_constr_avg = 1./sigma2_constr_avg
                        xovi_amat.vce = (weight_obs, weight_constr, weight_constr_avg)
                    else:
                        print("vce iter,weight (obs,constr):", i, xovi_amat.vce,keep_iterating_vce)
                        print("w_obs updated by",(np.abs(weight_obs-1./sigma2_obs)/weight_obs*100.),'% and w_constr by',\
                                         (np.abs(weight_constr-1./sigma2_constr)/weight_constr*100.),'% and w_constr_avg by',\
                                         (np.abs(weight_constr_avg-1./sigma2_constr_avg)/weight_constr_avg*100.),'%. Stop!')
                else:
                    keep_iterating_vce=False

        else:
            # TODO also compute weights and store them (no reason not to do so w/o partials)
            # create Amat
            xovi_amat = prepare_Amat(xov_cmb, vecopts)
            # print(xovi_amat.xov.xovers.columns)
            # exit()

            # clean only
            # clean_xov(xov_cmb, '')
            # # plot histo and geo_dist
            # tstname = [x.split('/')[-3] for x in datasets][0]
            # mean_dR, std_dR, worst_tracks = xov_cmb.remove_outliers('dR',remove_bad=remove_3sigma_median)
            # if debug:
            #     plt_histo_dR(tstname, mean_dR, std_dR,
            #                  xov_cmb.xovers)  # [tmp.xov.xovers.orbA.str.contains('14', regex=False)])
            #
            #     empty_geomap_df = pd.DataFrame(0, index=np.arange(0, 91),
            #                                    columns=np.arange(-180, 181))
            #     plt_geo_dR(tstname, xov_cmb)
        # exit()

        # append to list for stats
        xov_cmb_lst.append(xov_cmb)

        # print("len xov_cmb ", len(xov_cmb_lst[0].xovers))

    #print("len xov_cmb post getstats", len(xov_cmb_lst[0].xovers))

    # set as converged if relative improvement of residuals RMSE lower than convergence criteria
    if ext_iter > 0 and XovOpt.get("partials"):
        # print(xovi_amat.resid_wrmse,previous_iter.resid_wrmse)
        relative_improvement = np.abs((xovi_amat.resid_wrmse - previous_iter.resid_wrmse)/xovi_amat.resid_wrmse)
        print("Relative improvement at ", (relative_improvement*100.).round(2), "% at iteration", ext_iter)
        if relative_improvement <= convergence_criteria:
            print("Solution converged at iter",ext_iter)
            xovi_amat.converged = True
        elif  previous_iter.converged == True:
            print("Solution already converged...")
            xovi_amat.converged = True
        else:
            xovi_amat.converged = False

    # TODO not sure wether it can also be saved when just computing residuals
    # if partials:
    if len(ds.split('/')) > 2:
        xovi_amat.save(('_').join((data_pth + 'Abmat_' + ds.split('/')[0] + '_' +
                                   ds.split('/')[1]).split('_')[:-1]) + '_' + str(ext_iter + 1) +
                       '_' + ds.split('/')[2] + '.pkl')
    else:
        xovi_amat.save(
            (data_pth + 'Abmat_' + ds.split('/')[0] + '_' + ds.split('/')[1] + '_')[:-1] + str(ext_iter + 1) + '.pkl')

    # print(xovi_amat.xov.xovers)

    print("AccumXov ended succesfully!")
    ##############################################
    # stop clock and print runtime
    # -----------------------------
    endT = time.time()
    print('----- Runtime Amat = ' + str(endT - startT) + ' sec -----' + str(
        (endT - startT) / 60.) + ' min -----')
    return xovi_amat

########################
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
