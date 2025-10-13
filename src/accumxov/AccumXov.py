#!/usr/bin/env python3
# ----------------------------------
# AccumXov
# ----------------------------------
# Author: Stefano Bertone
# Created: 04-Mar-2019
#
import re
import warnings
import itertools

from accumxov.accum_opt import AccOpt
from config import XovOpt

from accumxov.accum_utils import get_xov_cov_tracks, get_vce_factor, downsize_xovers, \
    get_stats, print_sol, solve4setup, analyze_sol, subsample_xovers, clean_xov, \
    load_previous_iter_if_any, blockwise_cholesky
from accumxov.diagonal_estimators import estimate_diag_inv_AtA, stochastic_diag_estimate_N
from xovutil.iterables import mergsum
from xovutil.xovres2weights import get_interpolation_weight
from pyxover.xov_utils import load_combine, clean_partials
from accumxov.accum_plots import plot_obs_weights, plot_partials, \
   plot_weight_distribution, plot_huber_penal, plot_res_h2_partials, plot_orbitstd

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from scipy.sparse import csr_matrix, diags, issparse
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

from memory_profiler import profile

from pyxover.xov_setup import xov
from accumxov.Amat import Amat

######## SUBROUTINES ##########

# #@profile
def prepare_Amat(xov, vecopts, par_list=''):

   clean_xov(xov, par_list)

   xovtmp = xov.xovers.copy()

   if XovOpt.get("partials"):
      # simplify and downsize
      if par_list == '':
         par_list = xov.xovers.columns.filter(regex='^dR.*$')
      df_orig = xov.xovers[par_list]
      df_float = xov.xovers.filter(regex='^dR.*$').apply(pd.to_numeric, errors='ignore')  # , downcast='float')
      xov.xovers = pd.concat([df_orig, df_float], axis=1)
      rows, cols = xov.xovers.shape
      # xov.xovers.info(memory_usage='deep')
      mem_usage = xov.xovers.memory_usage(deep=True).sum()
      print(f"Memory usage: {mem_usage / 1_048_576:.2f} MB, for {rows} Entries and {cols} Columns")

      if XovOpt.get("debug"):
         pd.set_option('display.max_columns', 500)
         print(xov.xovers)

      if XovOpt.get("OrbRep") in ['lin', 'quad', 'per']:
         xovtmp = xov.upd_orbrep(xovtmp)
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

   # Solve
   # if not local:
   #     sol4_glo = ['dR/dRA', 'dR/dDEC', 'dR/dPM','dR/dL','dR/dh2'] # [None] # used on pgda, since prOpt badly read
   # test: if converged, also solve for h2
   # if previous_iter.converged:
   #     print("Adding h2 to sol4_glo as solution converged...")
   #     sol4_glo.extend(['dR/dh2'])

   sol4_pars = solve4setup(XovOpt.get("sol4_glo"), XovOpt.get("sol4_orb"), XovOpt.get("sol4_orbpar"),
                           xovi_amat.parNames.keys())

   if XovOpt.get("OrbRep") in ['lin', 'quad', 'per']:
      # xovi_amat.xov.xovers = xovi_amat.xov.upd_orbrep(xovi_amat.xov.xovers)
      regex = re.compile(".*_dR/d[A,C,R]$")
      const_pars = [x for x in sol4_pars if not regex.match(x)]
      if XovOpt.get("OrbRep") in ['lin']:
         sol4_pars = [x + str(y) for y in ['0', '1'] for x in list(filter(regex.match, sol4_pars))]
      if XovOpt.get("OrbRep") in ['quad']:
         sol4_pars = [x + str(y) for y in ['0', '1', '2'] for x in list(filter(regex.match, sol4_pars))]
      elif XovOpt.get("OrbRep") in ['per']:
         sol4_pars = [x + str(y) for y in ['C', 'S'] for x in list(filter(regex.match, sol4_pars))]

      sol4_pars.extend(const_pars)

   # Initialize list of solved for parameters (first one will be updated with full list
   # from previous iterations if needed)
   xovi_amat.sol4_pars = sol4_pars
   xovi_amat.sol4_pars_iter = sol4_pars

   if sol4_pars != []:
      if XovOpt.get("debug"):
         print(sol4_pars)
         print(xovi_amat.parNames)
         print([xovi_amat.parNames[p] for p in sol4_pars])
      # select columns of design matrix corresponding to chosen parameters to solve for
      spA_sol4 = xovi_amat.spA[:, [xovi_amat.parNames[p] for p in sol4_pars]]
      # set b=0 for rows not involving chosen set of parameters
      nnz_per_row = spA_sol4.getnnz(axis=1)
      xovi_amat.b[np.where(nnz_per_row == 0)[0]] = 0
   else:
      spA_sol4 = xovi_amat.spA

   # screening of partial derivatives (downweights data)
   nglbpars = len([i for i in XovOpt.get("sol4_glo") if i])
   if nglbpars > 0 and AccOpt.get("clean_part"):
      time1 = time.perf_counter()
      xovi_amat.b, spA_sol4 = clean_partials(xovi_amat.b, spA_sol4, threshold=1.e6, glbpars=XovOpt.get("sol4_glo"))
      time2 = time.perf_counter()
      print(f"Partials derivatives screnning lasted {round(time2-time1)}s")
      # pass

   # WEIGHTING
   if AccOpt.get("use_advanced_weighting"):
      obs_weights = advanced_weighting(xovi_amat.xov.xovers, previous_iter)

   else: # if Erwan's lidar test
      # WD: Not working anyway?
      print("!!! DANGEROUS MOD TO KEEP SIMU WEIGHTS !!!")
      obs_weights = diags(xovi_amat.xov.xovers['weights'].values, 0)

   xovi_amat.weights = obs_weights
   xovi_amat.xov.xovers['weights'] = xovi_amat.weights.diagonal()

   ## DIRECT SOLUTION FOR DEBUG AND SMALL PROBLEMS (e.g., global only)
   if XovOpt.get("debug"):
      if len(sol4_pars) < 50:
         print('B', xovi_amat.b)
         print('maxB', np.abs(xovi_amat.b).max(), np.abs(xovi_amat.b).mean())
         print('maxA', np.abs(spA_sol4.todense()).max(),
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
      if AccOpt.get("h2_limit_on") and XovOpt.get("local"):
         plot_res_h2_partials(xovi_amat.xov.xovers[['dR', 'dR/dh2', 'LON', 'LAT', 'weights']])

      # analysis of partial derivatives to check power in obs & param
      if XovOpt.get("local"):
         tmp = spla.norm(spA_sol4, axis=0)
         print("partials analysis", tmp.shape)
         print(sol4_pars)
         print(tmp[-5:])
         plot_partials(spA_sol4, obs_weights)

      # svd analysis of parameters (eigenvalues and eigenvectors)
      if XovOpt.get("local"):  # len(sol4_pars) < 50 and debug:
         print(sol4_pars)
         svd_parameter_analysis(spA_sol4, obs_weights, xovi_amat.parNames.keys())

         if False:
            # ATP = spAdense.transpose() * obs_weights
            # ATPA = ATP * spAdense
            # PA = obs_weights * spAdense
            ell = csr_matrix(np.diag(np.abs(bvec)))
            print(ell)
            posterr = np.linalg.pinv(ATP * ell * PA)
            posterr = np.sqrt(posterr.diagonal())
            print("posterr")
            print(posterr)
            # print(np.linalg.pinv(posterr))
            # posterr = Ninv * (spAdense.transpose() * (obs_weights * (ell * (obs_weights * (spAdense * N)))))
            # Ninv = np.linalg.pinv(spAdense.transpose() * spAdense)

            # check eigenvalues and vectors: https://andreask.cs.illinois.edu/cs598apk-f15/demos/02-tools-for-low-rank/Rank-Revealing%20QR.html

            # compute sol

            # factorL = np.linalg.norm([0.00993822, \
            #             -0.00104581, \
            #             -0.00010280, \
            #             -0.00002364, \
            #             -0.00000532])
            print('sol dense', np.linalg.lstsq(spAdense[:], bvec[:], rcond=1)[0])  # /factorL)
            print('to_be_recovered', pert_cloop['glo'])

   # # Compute the covariance matrix
   # print(np.linalg.pinv((spA_sol4.transpose()*spA_sol4).todense()))
   # # compute sol
   # print('sol dense',np.linalg.lstsq(spA_sol4.todense(), xovi_amat.b, rcond=1))
   #
   # A = spA_sol4.transpose()*spA_sol4
   # b = spA_sol4.transpose()*(csr_matrix(xovi_amat.b).transpose())

   # store cleaned
   xovi_amat.spA_sol4 = spA_sol4
   
   if not AccOpt.get("remove_max_dist") and not AccOpt.get("remove_3sigma_median") and not AccOpt.get("remove_dR200"):
     tmp = xovi_amat.weights.diagonal()
     avg_weight = np.mean(tmp)
     print("Weights (avg, med, std):", avg_weight, np.median(tmp), np.std(tmp))
     print("Fully weighted obs (>0.5*mean(weight)): ", len(tmp[tmp > 0.5 * avg_weight]), "or ",
           len(tmp[tmp > 0.5 * avg_weight]) / len(tmp) * 100., "%")
     print("Slightly downweighted obs: ", len(tmp[(tmp < 0.5 * avg_weight) * (tmp > 0.05 * avg_weight)]), "or ",
           len(tmp[(tmp < 0.5 * avg_weight) * (tmp > 0.05 * avg_weight)]) / len(tmp) * 100., "%")
     print("Brutally downweighted obs (<0.05*sigma0): ", len(tmp[(tmp < 0.05 * avg_weight)]), "or ",
           len(tmp[(tmp < 0.05 * avg_weight)]) / len(tmp) * 100., "%")

def advanced_weighting(xovers, previous_iter):
   
   # after convergence of residuals RMS at 1%, fix weights and bring parameters to convergence
   if previous_iter != None and previous_iter.converged:
      print("Weights are fixed as solution converged to 5%")
      # associate weights of old solution to corresponding xov of current one (id by tracks, supposing uniqueness)
      if True:  # only if issues with duplicates
         previous_iter.xov.xovers = previous_iter.xov.xovers.drop(columns=['xOvID', 'xovid'],
                                                                  errors='ignore').drop_duplicates().reset_index().rename(
                                                                     columns={"index": "xOvID"})
      tmp_prev_trk = pd.DataFrame(previous_iter.xov.xovers['orbA'] + previous_iter.xov.xovers['orbB'],
                                  columns=['trksid'])
      tmp_prev_trk['weights'] = previous_iter.xov.xovers.weights.values
      if True:  # only if issues with duplicates
         tmp_prev_trk.drop_duplicates(inplace=True)
      else:
         tmp_prev_trk['weights'] = previous_iter.weights.diagonal().T
         print("len(tmp_prev_trk)", len(tmp_prev_trk))
         print("len(xovers)", len(xovers))

      tmp_xov_trk = pd.DataFrame(xovers['orbA'] + xovers['orbB'], columns=['trksid'])

      obs_weights = pd.merge(tmp_xov_trk, tmp_prev_trk, how='left', on=['trksid'])
      obs_weights = diags(obs_weights['weights'].fillna(0).values)

      # unsafe if list of xov is different or ordered differently
      # obs_weights = previous_iter.weights

   else:
      # compute huber weights (1 if x<AccOpt.get("huber_threshold"), (AccOpt.get("huber_threshold")/abs(dR))**2 if abs(dR)>AccOpt.get("huber_threshold"))
      if (not AccOpt.get("remove_max_dist") and not AccOpt.get("remove_3sigma_median") and
             not AccOpt.get("remove_dR200")):
            time1 = time.perf_counter()
            tmp = xovers.dR.abs().values
            huber_weights = np.where(tmp > AccOpt.get("huber_threshold"),
                                     (AccOpt.get("huber_threshold") / tmp) ** 1, 1.)

            if XovOpt.get("debug"):
               print("Apply Huber weights (resid)")
               print(tmp[tmp > AccOpt.get("huber_threshold")])
               print(np.sort(huber_weights[huber_weights < 1.]), np.mean(huber_weights))

            # same but w.r.t. distance
            tmp = xovers.dist_max.values
            huber_weights_dist = np.where(tmp > AccOpt.get("distmax_threshold"),
                                          (AccOpt.get("distmax_threshold") / tmp) ** 2, 1.)

            if XovOpt.get("debug"):
               print("Apply Huber weights (dist)")
               print(tmp[tmp > AccOpt.get("distmax_threshold")])
               print(np.sort(huber_weights_dist[huber_weights_dist < 1.]), np.mean(huber_weights_dist))

            # same but w.r.t. offnadir
            tmp = np.nan_to_num(xovers.filter(regex='offnad').values)
            tmp = np.max(np.abs(tmp), axis=1)
            huber_weights_offnad = np.where(tmp > AccOpt.get("offnad_threshold"),
                                            (AccOpt.get("offnad_threshold") / tmp) ** 1, 1.)

            if XovOpt.get("debug"):
               print("Apply Huber weights (offnad)")
               print(tmp[tmp > AccOpt.get("offnad_threshold")])
               print(len(huber_weights_offnad[huber_weights_offnad < 1.]), len(huber_weights_offnad))
               print(np.sort(huber_weights_offnad[huber_weights_offnad < 1.]), np.mean(huber_weights_offnad))

            # combine weights
            tmp = huber_weights * huber_weights_dist * huber_weights_offnad
            huber_penal = tmp
            # should use weights or measurement error threshold, but using huber-threshold-like criteria for now
            # to mimic what I was doing without weights
            xovers['huber'] = huber_penal
            
            time2 = time.perf_counter()
            print(f"Huber weights computation lasted {round(time2-time1)}s")

            # get quality of tracks and apply huber weights
            # WD: comment below
            if AccOpt.get("apply_xov_cov_tracks"):
               tmp = xovers.copy()[
                  ['xOvID', 'LON', 'LAT', 'dtA', 'dR', 'orbA', 'orbB', 'huber']]  # .astype('float16')
               
               if XovOpt.get("debug"):
                  print("pre xovcov types", tmp.dtypes)

               weights_xov_tracks = get_xov_cov_tracks(df=tmp, plot_stuff=False)
               xovers['huber_trks'] = weights_xov_tracks.diagonal()

               if XovOpt.get("debug"):
                  # the histogram of weight distribution
                  if False and XovOpt.get("local"):
                     plot_weight_distribution( weights_xov_tracks.diagonal())
                     
                  if False:
                     tmp['track_weights'] = weights_xov_tracks.diagonal()
                     tmp = tmp[['orbA', 'orbB', 'dR', 'track_weights']]
                     print(tmp[tmp.dR.abs() < 0.5].sort_values(by='track_weights'))

               #######
            # additional for h2 tests
            if AccOpt.get("h2_limit_on"):
               # cut based on residuals
               limit_h2 = 20.
               tmp = xovers.dR.abs().values
               tmp = np.where(tmp > limit_h2, (limit_h2 / tmp) ** 4, 1.)
               huber_penal *= tmp
               # cut based on mean min separation
               limit_h2_sep = 10. * 1.e-3  # km based
               tmp = xovers.dist_min_mean.values
               tmp = np.where(tmp > limit_h2_sep, (limit_h2_sep / tmp) ** 4, 1.)
               huber_penal *= tmp

               if XovOpt.get("debug") and XovOpt.get("local"):
                  plot_huber_penal(huber_penal)

      #######

      # interp_weights = get_weight_regrough(xovers).reset_index()  ### old way using residuals to extract roughness
      #
      # get interpolation error based on roughness map (if available at given latitude) + minimal distance
      interp_weights = get_interpolation_weight(xovers).reset_index()

      val = interp_weights['weight'].values  # np.ones(len(interp_weights['weight'].values)) #
      # print("interp error values", np.sort(val))
      xovers['interp_weight'] = val

      # apply huber weights
      if (not AccOpt.get("remove_max_dist") and not AccOpt.get("remove_3sigma_median") and
             not AccOpt.get("remove_dR200")):
            # print(val)
            # print(huber_penal)
            # print(len(val), len(huber_penal))
            # print(val.dtype, huber_penal.dtype)
            # print(val.shape)
            # print(huber_penal.shape)
            val *= np.hstack(huber_penal) # hstack converts array of objects (not sure why) to array of floats
            # print("after huber", np.sort(val), np.mean(val))
            # val *= huber_weights_dist

      # val /= np.max(np.abs(val))
      row = col = interp_weights.index.values
         
      # add weight based on instrument
      # w2 = 0.01
      # val *= np.where(xovers.orbA.str.startswith('1'), w2, 1.)
      # val *= np.where(xovers.orbB.str.startswith('1'), w2, 1.)
      # print("instrument downweight factor", w2)
      # val *= np.where(xovers.orbA.str.startswith('1') and xovers.orbB.str.startswith('2'), 1., 1.)
      # val *= np.where(xovers.orbA.str.startswith('2') and xovers.orbB.str.startswith('1'), 1., 1.) # useless?
      # val *= np.where(xovers.orbA.str.startswith('2') and xovers.orbB.str.startswith('2'), 1., 1.)

      # obs_weights = csr_matrix((np.ones(len(val)), (row, col)), dtype=np.float32, shape=(len(interp_weights), len(interp_weights)))
      obs_weights = csr_matrix((val, (row, col)), dtype=np.float32,
                               shape=(len(interp_weights), len(interp_weights)))

      # combine with off-diag terms from tracks
      # ========================================
      if AccOpt.get("apply_xov_cov_tracks"):
         # obs_weights = diags(weights_xov_tracks.diagonal()*obs_weights) # to apply only the diagonal
         obs_weights = weights_xov_tracks.multiply(obs_weights)
         print("Observations weights re-evaluated, solution has not converged yet")

         if XovOpt.get("debug") and XovOpt.get("local"):
            print("tracks weights", weights_xov_tracks.diagonal().mean(), np.sort(weights_xov_tracks.diagonal()))
            tmp = obs_weights.diagonal()
            tmp = np.where(tmp > 1.e-9, tmp, 0.)
            print(np.sort(tmp), np.median(tmp), np.mean(tmp))
            plot_obs_weights(tmp)
   
   return obs_weights

def compute_penalty_matrices(xovi_amat, tracks_to_remove):
   
   if XovOpt.get("OrbRep") in ['lin', 'quad', 'per']: #WD: Not sure what this is for
      for par in ['dA', 'dC', 'dR']:
         if par in XovOpt.get("sol4_orbpar"):
            XovOpt.get("par_constr")['dR/' + par + '0'] = XovOpt.get("par_constr").pop('dR/' + par)

   # select constrains for processed parameters (TODO should go in sol4pars)
   if False: # track constraints are not already in regex
      mod_par = [your_key.split('_')[1] if len(your_key.split('_')) > 1 else your_key for your_key in xovi_amat.sol4_pars]
      par_constr = {your_key: XovOpt.get("par_constr")[your_key] for your_key in mod_par}
      # par_constr = {".*" + your_key + "$": XovOpt.get("par_constr")[your_key] for your_key in mod_par}
      penalty_mat = [compute_penalty_mat_abs(xovi_amat, par_constr, xovi_amat.sol4_pars, tracks_to_remove)]
   else: # track constraints are already regex
      mod_par = [f"{your_key[:1]}.*{your_key.split('_')[1]}" if len(your_key.split('_')) > 1 else your_key for your_key in xovi_amat.sol4_pars]
      # par_constr = {your_key: XovOpt.get("par_constr")[your_key] for your_key in mod_par}
      
      # Currently, constraints on global parameters are not applied
      penalty_mat = []
      # par_constr = {your_key: XovOpt.get("par_constr")[your_key] for your_key in mod_par if not your_key.startswith('2')}
      par_constr = {your_key: XovOpt.get("par_constr")[your_key] for your_key in mod_par if your_key.startswith('1')}
      if par_constr:
         penalty_mat.append(compute_penalty_mat_abs(xovi_amat, par_constr, xovi_amat.sol4_pars, tracks_to_remove))

      par_constr = {your_key: XovOpt.get("par_constr")[your_key] for your_key in mod_par if your_key.startswith('2')}
      # par_constr = {your_key: XovOpt.get("par_constr")[your_key] for your_key in mod_par if not your_key.startswith('1')}
      if par_constr:
         penalty_mat.append(compute_penalty_mat_abs(xovi_amat, par_constr, xovi_amat.sol4_pars, tracks_to_remove))
         
      # par_constr = {your_key: XovOpt.get("par_constr")[your_key] for your_key in mod_par if your_key.startswith('d')}
      # if par_constr:
      #    penalty_mat.append(compute_penalty_mat_abs(xovi_amat, par_constr, xovi_amat.sol4_pars, tracks_to_remove))
   
   if len(XovOpt.get("mean_constr")) > 0:
      penalty_mat.append(compute_penalty_mat_avg(xovi_amat.sol4_pars))

   return penalty_mat

def compute_penalty_mat_abs(xovi_amat, par_constr, sol4_pars, tracks_to_remove):
   # penalty_mat from xovi_amat are set
   # xovi_amat.weights, xovi_amat.xov are used
   
   print("Compute penalty matrix with constraints", par_constr)
   
   if XovOpt.get("OrbRep") in ['lin', 'quad', 'per']:
        for par in ['dA', 'dC', 'dR']:
            if par in XovOpt.get("sol4_orbpar"):
                XovOpt.get("par_constr")['dR/' + par + '0'] = XovOpt.get("par_constr").pop('dR/' + par)

   csr = []
   if par_constr: # If constraints:
      # Constrain tightly to 0 those parameters with few observations (or with few GOOD observations)
   
      # WD: hardcoded year of MESSENGER flyby??
      # WD: Is it there to tightly constrain parameters while leaving the corresponding observation?
      #     It may damage other tracks is the track is not good enough
      to_tightly_constrain = [idx for idx, p in enumerate(sol4_pars) if
                      p.split('_')[0] in tracks_to_remove.index if
                      p.split('_')[0][:2] != '08']  # exclude flybys from this, else orbits are never improved
      # to_tightly_constrain = [idx for idx, p in enumerate(sol4_pars) if p.split('_')[0] in n_goodobs_tracks[n_goodobs_tracks < 1].index if p.split('_')[0][:2]!='08'] # exclude flybys from this, else orbits are never improved

      if XovOpt.get("debug"):
         print("number of constrained pars", len(to_tightly_constrain))
         print(to_tightly_constrain)
         print([dict(zip(xovi_amat.parNames.values(), xovi_amat.parNames.keys()))[x] for x in to_tightly_constrain])

      for constrain in par_constr.items():
         regex = re.compile(constrain[0])
         parindex = np.array([[idx, float(constrain[1])] for idx, p in enumerate(sol4_pars) if regex.match(p)])
         for p in parindex:
            if p[0] in to_tightly_constrain:
               # else a very loose "general" constraint could free it up
               # (if constraint on p is int, could mess up and give 0 => Nan)
               if p[1] > 1.:
                  p[1] = 1.e-4
               else:
                  p[1] *= 1.e-4

         val = parindex[:, 1]
         row = col = parindex[:, 0]

         csr.append(csr_matrix((np.power(AccOpt.get("sigma_0") / val, 2), (row, col)),
                               dtype=np.float32, shape=(len(sol4_pars), len(sol4_pars))))
   # combine all constraints and store penalties into amat
   return sum(csr)

def get_bad_tracks(xovi_amat):

   # Constrain tightly to 0 those parameters with few observations (or with few GOOD observations)
   if not AccOpt.get("remove_max_dist") and not AccOpt.get("remove_3sigma_median") and not AccOpt.get(
      "remove_dR200"):
      # TODO should use weights or measurement error threshold, but using huber-threshold-like criteria for now
      # TODO to mimic what I was doing without weights
      # nobs_tracks = xovi_amat.xov.xovers.loc[xovi_amat.xov.xovers.huber > 0.5][['orbA', 'orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(
      # ascending=False)
      weights_mean = np.mean(xovi_amat.weights.diagonal())
      # n_goodobs_tracks = xovi_amat.xov.xovers.loc[xovi_amat.weights.diagonal() > 0.5*AccOpt.get("sigma_0")][['orbA', 'orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(
      n_goodobs_tracks = xovi_amat.xov.xovers.loc[xovi_amat.weights.diagonal() > 0.1 * weights_mean][
         ['orbA', 'orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(ascending=False)
   else:
      n_goodobs_tracks = xovi_amat.xov.xovers[['orbA', 'orbB']].apply(pd.Series.value_counts).sum(
         axis=1).sort_values(ascending=False)

   return n_goodobs_tracks[n_goodobs_tracks < 10]
   
def compute_penalty_mat_avg(sol4_pars):
   # penalty_mat_avg from xovi_amat is set
   # xovi_amat.weights, xovi_amat.xov are used
   
   ###
   # M = num_params
   # G = np.ones((1, M)) / M 
   # d = np.zeros(1) 

   # A_constr = lambda_avg * csr_matrix(G)
   # b_constr = lambda_avg * d
 
   # A_srif = vstack([A, A_constr])  # (N+1)×M
   # b_srif = np.hstack([b, b_constr])  # length N+1
   ###
 
   # does not work with only one set of orbit parameters solved
   # if len(sol4_orb)>1 and True:
      csr_avg = []
      start = time.time()
      print("Loop over mean_constr")
      for constrain in XovOpt.get("mean_constr").items():
         regex = re.compile(".*" + constrain[0] + "0{0,1}$")

         if list(filter(regex.match, sol4_pars)):
            parindex = np.array([[idx, float(constrain[1])] for idx, p in enumerate(sol4_pars) if regex.match(p)])
            if len(parindex) > 0:
               # Constrain tightly to 0 those parameters with few observations
               # nobs_tracks = xovi_amat.xov.xovers[['orbA', 'orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(
               #     ascending=False)
               # to_tightly_constrain = [idx for idx, p in enumerate(sol4_pars) if
               #                 p.split('_')[0] in nobs_tracks[nobs_tracks < 10].index]
               # for p in parindex:
               #     if p[0] in to_tightly_constrain:
               #         p[1] *= 1.e10

               rowcols_nodiag = np.array(list(set(itertools.permutations(parindex[:, 0], 2))))
               rowcols_diag = np.array(list([(x, x) for x in parindex[:, 0]]))
               vals = - 1 / len(parindex[:, 0]) * np.ones(
                  len(parindex[:, 0]) * len(parindex[:, 0]) - len(parindex[:, 0]))
               csr_avg.append(csr_matrix((vals * np.power(AccOpt.get("sigma_0") / constrain[1], 2),
                                          (rowcols_nodiag[:, 0], rowcols_nodiag[:, 1])),
                                         dtype=np.float32, shape=(len(sol4_pars), len(sol4_pars))))
               vals = (1 - 1 / len(parindex[:, 0])) * np.ones(len(parindex[:, 0]))
               csr_avg.append(csr_matrix((vals * np.power(AccOpt.get("sigma_0") / constrain[1], 2),
                                          (rowcols_diag[:, 0], rowcols_diag[:, 1])),
                                         dtype=np.float32, shape=(len(sol4_pars), len(sol4_pars))))

      end = time.time()
      print("End loop over mean_constr after", int(end - start), "sec or ", round((end - start) / 60., 2), " min!")
      return sum(csr_avg)

def svd_parameter_analysis(spA_sol4, obs_weights, parNames):
   # Compute the covariance matrix
   # print("full sparse",np.linalg.pinv((spA_sol4.transpose()*spA_sol4).todense()))
   # print("screened dense", np.linalg.pinv(spAdense.transpose()*spAdense))
   # tmp = prev.xov.xovers['dR'].values.reshape(1, -1) @ np.diag(prev.xov.xovers['weights'].values)
   # m_0 = np.sqrt(
   #     tmp @ prev.xov.xovers['dR'].values.reshape(-1, 1) / (len(prev.xov.xovers['dR'].values) - len(sol4_glo)))
   N = (spA_sol4.transpose() * obs_weights * spA_sol4).todense()  # ATPA

   # project arc par on global pars
   ATA = csr_matrix(N[:-5, :-5])
   print(ATA.shape)
   ATB = csr_matrix(N[:-5, -5:])
   BTA = csr_matrix(N[-5:, :-5])
   BTB = csr_matrix(N[-5:, -5:])

   tmp = np.linalg.pinv(ATA.todense()) * ATB
   N_proj = BTB - BTA * tmp

   # check eigenvector and values in the problem
   for idx, mat in enumerate([BTB.todense(), N_proj]):
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
      plt.imshow(Vh.T, cmap='bwr')
      plt.colorbar()
      plt.savefig(XovOpt.get("tmpdir") + "test_svd_" + str(idx) + ".png")

      print("Pars:", list(parNames)[-5:])
      for i in range(5):
         print("Norm of Vh", np.round(np.linalg.norm(Vh.T[:, :i + 1], axis=1) * 100., 1),
               "% up to lambda= ",S[i])
                                                                     
def compute_vce_weights(amat, L=None, N=None, spA_penal=None, Ndiag=False):
   # VCE for observations NEQs is well approximated by Ndiag=True.
   # VCE for constraint NEQs require more care (because diagonal).
   # Stochastic trace estimators with approximate solver are not very efficient.
   # Making use of Cholesky factor improve the convergence.
   # Combining Cholesky factor with stochastic trace estimators is efficient (albeit less accurate)

   xsol = np.array([
      next(v for k, v in amat.sol_dict['sol'].items() if filt in k)
      for filt in amat.sol4_pars_iter
      ])

   s2_obs_apr = [1. / v for v in amat.vce_obs]
   s2_constr_apr = [1. / v for v in amat.vce_pen]
   
   if N is None and (Ndiag or L is None):
      N = spA_penal.T * spA_penal

   if Ndiag: # N^-1=1/diag(N) seems enough for VCE_obs
      if issparse(N):
         Ninv = diags(1/N.diagonal()).todense()
      else:
         Ninv = np.diag(1/np.diag(N))
   elif (L is None):
      # compute total N^{-1}
      if issparse(N):
         N = N.todense()              
      Ninv = np.linalg.pinv(N, hermitian=True, rcond=1.e-20)
      end = time.time()
      print("Ninv computation finished after", int(end - start), "sec or ", round((end - start) / 60., 2), " min!")

      if not np.allclose(N, N @ (Ninv @ N)):
         print('### N is almost singular!! Help!!!')
   else:
      Ninv = None

   start = time.time()

   # amat.weights.diagonal()[mask_obs] works because it is truyl diagonal
   if Ndiag:
      s2_obs_new = [get_vce_factor(x=xsol, Cinv=diags(amat.weights.diagonal()[mask_obs]), L=None, Ninv=Ninv,
                                   b=amat.b[mask_obs], A=amat.spA_sol4[mask_obs, :],
                                   s2apr=s2_obs, kind='obs', nelem=sum(mask_obs))
                    for s2_obs, mask_obs in zip(s2_obs_apr,amat.obs_blocks)]
   else:
      s2_obs_new = [get_vce_factor(x=xsol, Cinv=diags(amat.weights.diagonal()[mask_obs]), L=L, N=N, Ninv=Ninv,
                                   b=amat.b[mask_obs], A=amat.spA_sol4[mask_obs, :],
                                   s2apr=s2_obs, kind='obs', nelem=sum(mask_obs),stoch=True)
                    for s2_obs, mask_obs in zip(s2_obs_apr,amat.obs_blocks)]

   end = time.time()
   print(f"Building s2_obs_new: {int(end - start)} sec")
   for s in s2_obs_new: # new weights have to be > 0
      assert s > 0

   start = time.time()
   # WD: works only if penalty_mat is a NEQ
   s2_constr_new = [get_vce_factor(x=xsol, Cinv=penalty_mat, L=L, Ninv=Ninv, N=N,
                                   s2apr=s2_constr, kind='constr',
                                   nelem=penalty_mat.nnz, stoch=True)
                    for s2_constr, penalty_mat in zip(s2_constr_apr,amat.penalty_mat)]
   end = time.time()
   print(f"Building s2_constr_new: {int(end - start)} sec")
   for s in s2_constr_new: # new weights have to be > 0
      assert s > 0

   return s2_obs_new, s2_constr_new

def remove_tracks(xovi_amat, tracks_to_remove):
   # remove tracks and related observations
   # works only for one block one penalty matrix

   par_to_keep = [idx for idx, val in enumerate(xovi_amat.sol4_pars) if
                    not any(val.startswith(t) for t in tracks_to_remove)]
   
   obs_to_keep = (~xovi_amat.xov.xovers['orbA'].isin(tracks_to_remove) & ~xovi_amat.xov.xovers['orbB'].isin(tracks_to_remove))
   
   weight_d = xovi_amat.weights.diagonal()
   xovi_amat.b = xovi_amat.b[obs_to_keep]
   xovi_amat.weights = diags(weight_d[obs_to_keep])
   xovi_amat.spA_sol4 = xovi_amat.spA_sol4[:, par_to_keep]
   xovi_amat.spA_sol4 = xovi_amat.spA_sol4[obs_to_keep,:]
   xovi_amat.obs_blocks = [obs[obs_to_keep] for obs in xovi_amat.obs_blocks]
   xovi_amat.penalty_mat = [pen[np.ix_(par_to_keep, par_to_keep)] for pen in xovi_amat.penalty_mat]

   xovi_amat.sol4_pars = [xovi_amat.sol4_pars[par] for par in par_to_keep]
   xovi_amat.sol4_pars_iter = xovi_amat.sol4_pars    

# @profile
def compute_solution(xovi_amat, previous_iter, xov_cmb):
   # xovi_amat attributes which are changes:
   # spA_penal, b_penal, sol, sol_dict
   # sol_dict_iter, sol_iter, sol4_pars, vce
   
   # xovi_amat attributes changed via get_stats
   # spA, b, postfit_res, resid_wrmse

   coo = xovi_amat.weights.tocoo()
   if (np.all(coo.row == coo.col)):
      print("Weight matrix is diagonal")
      W_L = diags(xovi_amat.weights.diagonal()**0.5)
   else:
      print("Cholesky decomposition of the weight matrix")
      W_L = la.cholesky(xovi_amat.weights.todense(), lower=True)
      # W_L = sparse_cholesky(xovi_amat.weights.todense())

   last_iteration = not AccOpt.get("compute_vce")
   max_iter = 10
   for i in (i for i in range(max_iter)):
      if i == max_iter-1:
         last_iteration = True
      print(f"\nIteration {i}:")
      print(f"------------")
      
      print("Weights VCE obs/constr", xovi_amat.vce_obs, xovi_amat.vce_pen)
            
      ######################### MAKE NEW SOLVE ROUTINE (rename the one above as prepro_weights_constr())
      # Create spA_penal and b_penal from observations and pseudo observations (i.e., constraints)
      weight_obs = xovi_amat.vce_obs
      sqrt_weight_obs = np.sqrt(weight_obs)
      weight_constr = xovi_amat.vce_pen
      # Choleski decompose matrix and append to design matrix (weight_constr[0] applied except for constrain on avg)
      if len(xovi_amat.penalty_mat) > 0:
         penalty = sum(w * p for w, p in zip(weight_constr, xovi_amat.penalty_mat))
         if np.count_nonzero(penalty - np.diag(penalty.diagonal())):
            print("Cholesky decomposition of the constraint matrix")
            # spQ = sparse_cholesky(penalty)
            spQ = csr_matrix(la.cholesky(penalty.todense()))
            print("Cholesky decomposition done")
         else:
            print("Penalty matrix is diagonal")
            spQ = diags(penalty.diagonal()**0.5)
      else:
         penalty = []

      b_penal =  sum([ w * mask_obs.astype(float) * xovi_amat.b  for (w,mask_obs) in zip(sqrt_weight_obs,xovi_amat.obs_blocks)])
      b_penal = W_L.T * b_penal
      # add penalisation to residuals
      if previous_iter != None and previous_iter.sol_dict != None:
         # get previous solution reordered as sol4_pars_iter (and hence as Q) - contains the full solution but only for the
         # parameters also solved in this iteration (and hence consistent with Q, else it crashes).
         # should this rather be sol_dict?? Do we want to constrain the correction amplitude at each iter or the full correction?
         if i == 0:  # no need to update this at each iter
            prev_sol_ord = [previous_iter.sol_dict['sol'][key] if
                            key in previous_iter.sol_dict['sol'] else 0. for key in
                            xovi_amat.sol4_pars_iter]
                  
         Q = spQ.todense()
         b_penal = np.hstack([b_penal, -1. * np.ravel(np.dot(Q, prev_sol_ord))]) # WD: to check !!
      else:
         if len(xovi_amat.penalty_mat) > 0:
            b_penal = np.hstack([b_penal, -1. * np.zeros(penalty.shape[0])])

      spA_sol4_penal = sum([ w * diags(mask_obs.astype(float)) @ xovi_amat.spA_sol4 for (w,mask_obs) in zip(sqrt_weight_obs,xovi_amat.obs_blocks)])

      # apply weights
      # TODO clean-up: applying weights at this point makes it impossible to get covariance matrix
      if XovOpt.get("instrument") != "pawstel":
         spA_sol4_penal = W_L.T * spA_sol4_penal
      if len(xovi_amat.penalty_mat) > 0:
         spA_sol4_penal = sp.vstack([spA_sol4_penal, 1. * spQ])

      # # save penalised matrices
      # xovi_amat.spA_penal = spA_sol4_penal
      # xovi_amat.b_penal = b_penal

      if AccOpt.get("get_cov_only"):
         # add penalisation to partials matrix
         Q = spQ.todense()
         weights_penal = diags(np.concatenate([xovi_amat.weights.diagonal(), np.ones(Q.shape[0])]), 0)

         try:
            std_par_unconstrained = np.sqrt(spla.inv(
               xovi_amat.spA.transpose() * xovi_amat.weights * xovi_amat.spA).diagonal())
         except:
            print("** Factor is exactly singular, need some constraints")
            std_par_unconstrained = np.tile(0., len(xovi_amat.parNames))

         try:
            std_par = np.sqrt(spla.inv(
               spA_sol4_penal.transpose() * weights_penal * spA_sol4_penal).diagonal())
         except:
            print("** Matrix not positive definite, modify constraints.")
            print("** Passing unconstrained formal errors")
            std_par = std_par_unconstrained

         return xovi_amat.parNames, std_par, std_par_unconstrained

      else:
         # solve using lsqr
         print("Starting LSQR")
         start = time.time()
         xovi_amat.sol, xovi_amat.std, N, L = solve(spA_sol4_penal, b_penal, last_iteration,
                                                    tol=(1.e-10 / AccOpt.get("sigma_0")),
                                                    solving_method=AccOpt.get("solving_method"))
         end = time.time()
         print("lsqr finished after", int(end - start), "sec or ", round((end - start) / 60., 2), " min!")

      # Save to pkl
      orb_sol, glb_sol, sol_dict = analyze_sol(xovi_amat, xov_cmb.xovers, mode='iter')

      # check std of orbital parameters for systematics
      if XovOpt.get("debug"):
         plot_orbitstd(sol_dict)

      xovi_amat.sol_dict = clean_solution(sol_dict)

      if XovOpt.get("debug"):
         pd.set_option('display.max_rows', None)
         pd.set_option('display.max_columns', None)
         pd.set_option('display.width', None)
         pd.set_option('display.max_colwidth', -1)

      xovi_amat.resid_wrmse = get_stats(xovi_amat, spA_sol4_penal, b_penal)

      print(f"Solution for iteration {i}")
      print_sol(orb_sol, glb_sol, xov, xovi_amat)
      
      # store improvments from current iteration
      xovi_amat.sol_dict_iter = xovi_amat.sol_dict.copy()
      xovi_amat.sol_iter = list(xovi_amat.sol_dict['sol'].values())

      if last_iteration:
         break

      # VCE
      if AccOpt.get("compute_vce"):
         start = time.time()
         if AccOpt.get("solving_method") == "cholesky": # if cholesky decomposition was performed
            sigma2_obs, sigma2_constr = compute_vce_weights(xovi_amat,  L  , N, spA_sol4_penal, Ndiag=True)
         else: # N is directly passed (if scaling, N=None, and will be recomputed)
            sigma2_obs, sigma2_constr = compute_vce_weights(xovi_amat, None, N, spA_sol4_penal, Ndiag=True)
         end = time.time()
         print("compute_vce_weights finished after", int(end - start), "sec or ", round((end - start) / 60., 2), " min!")

         w_obs_impovement    = [np.abs(w - 1. / s)/w for (w,s) in zip(weight_obs,sigma2_obs)]
         w_constr_impovement = [np.abs(w - 1. / s)/w for (w,s) in zip(weight_constr,sigma2_constr)]

         keep_iterating_vce = any(w > 0.01 for w in w_obs_impovement) or any(w > 0.1 for w in w_constr_impovement)

         print("vce iter,weight (obs,constr):", i, xovi_amat.vce_obs, xovi_amat.vce_pen, keep_iterating_vce)
         print("w_obs updated by", [w * 100. for w in w_obs_impovement],
               '% and w_constr by',[w * 100. for w in w_constr_impovement], '%')
         if keep_iterating_vce:
            # update weights
            weight_obs    = [1. / sig for sig in sigma2_obs]
            weight_constr = [1. / sig for sig in sigma2_constr]
            xovi_amat.vce_obs = weight_obs
            xovi_amat.vce_pen = weight_constr
         else:
            print("Stop VCE, next iteratin will be the last one.")
            last_iteration = True
   
   # Cumulate with solution from previous iter (or from pre-processing)
   if previous_iter != None:  # (int(ext_iter) > 0) and (previous_iter != None):
      if previous_iter.sol_dict != None:
         # def dict2np(x):
         #     return np.array(list(x.values()))
         # print("test xTx update -soldictiter- pre=",np.sqrt(dict2np(xovi_amat.sol_dict_iter['sol']).T@dict2np(xovi_amat.sol_dict_iter['sol'])))
         # print("test xTx update -prevsol- pre=",np.sqrt(dict2np(previous_iter.sol_dict['sol']).T@dict2np(previous_iter.sol_dict['sol'])))
         # sum the values with same keys
         updated_sol = mergsum(xovi_amat.sol_dict_iter['sol'], previous_iter.sol_dict['sol'])
         updated_std = mergsum(xovi_amat.sol_dict_iter['std'],
                               previous_iter.sol_dict['std'].fromkeys(previous_iter.sol_dict['std'], 0.))
         # WD: is previous_iter.sol_dict['std'].fromkeys(previous_iter.sol_dict['std'], 0.) to enforce 0 values?
         # print("test xTx update -soldictiter- post=",np.sqrt(dict2np(xovi_amat.sol_dict_iter['sol']).T@dict2np(xovi_amat.sol_dict_iter['sol'])))
         # print("test xTx update -prevsol- post=",np.sqrt(dict2np(previous_iter.sol_dict['sol']).T@dict2np(previous_iter.sol_dict['sol'])))
         # print("test xTx update post=",np.sqrt(dict2np(updated_sol).T@dict2np(updated_sol)))

         # save total list of parameters (previous iters + current)
         xovi_amat.sol4_pars = list(updated_sol.keys())
         xovi_amat.sol_dict = {'sol': updated_sol, 'std': updated_std}
         # use dict to update amat.sol, keep std
         # xovi_amat.sol = (list(xovi_amat.sol_dict['sol'].values()), *xovi_amat.sol[1:-1],
         #                  list(xovi_amat.sol_dict['std'].values()))
         xovi_amat.sol = list(xovi_amat.sol_dict['sol'].values())
         xovi_amat.std = list(xovi_amat.sol_dict['std'].values())
         orb_sol, glb_sol, sol_dict = analyze_sol(xovi_amat, xov_cmb.xovers, mode='full')
         print("Cumulated solution")
         print_sol(orb_sol, glb_sol, xov, xovi_amat)
      else:
         print("previous_iter.sol_dict=", previous_iter.sol_dict)

# @profile
def solve(spA, b_penal, last_iteration, tol=1e-8, solving_method="cholesky"):
   
   estimate_var = False # not accurate enough

   start = time.time()
   L = None
   N = None
   scaling = False
   # scaling is helpful for method relying on approximated solvers
   if solving_method != "cholesky":
      scaling = True

   # Column scaling
   if scaling:
      # make each column of A have unit 2-norm
      col_norms = np.sqrt(spA.power(2).sum(axis=0)).A1

      # avoid divide-by-zero
      col_norms[col_norms==0] = 1.0
            
      spA = spA @ diags(1.0 / col_norms)
         
      print("Columns scaled with max:", np.max(col_norms))
 
   if solving_method == "lsqr":
      print("Solving using lsqr")
      result = spla.lsqr(spA, b_penal, damp=0, show=False, iter_lim=100000,
                         atol=tol, conlim=1e8, btol=tol, calc_var=True)
         
   if solving_method == "lsmr":
      print("Solving using lsmr")
      result = spla.lsmr(spA, b_penal, damp=0, show=False, maxiter=100000,
                         atol=tol, conlim=1e8, btol=tol)
         
   if solving_method in  ["lsqr","lsmr"]:
      end = time.time()
      print(f"lsqr/lsmr finished: {int(end - start)} sec")
         
      istop = result[1]
      if istop == 1:
         print("LSQR solution is an approximate solution to Ax = b.")
      elif (istop == 2):
         print("LSQR solution approximately solves the least-squares problem.")
         u, s, vt = spla.svds(spA, k=6)
         print("Estimated condition number:", max(s) / min(s))
         print("LSQR condition number:",result[6])
         if (max(s) / min(s) > 100):
            print("Unreliable variance")
      elif istop > 2:
         print("*** Accumxov.compute_solution: the system may be inconsistent.")
         print("The solution is an approximate solution to the corresponding least-squares problem.")
         exit(2)
      sol = result[0]

   # compute normal equation system (N,bn) if needed
   if solving_method in  ["cholesky","cg"] or last_iteration:
      N  = spA.T * spA
      if solving_method in  ["cholesky","cg"]:
         bn = spA.T * b_penal
         
   if solving_method == "cholesky":
      print("Solving using Cholesky factorization")
      start = time.time()
      N = N.todense()
      if N.shape[0] < 25000:
         L = la.cholesky(N, lower=True, overwrite_a=True)
         end = time.time()
         print(f"Cholesky factor: {int(end - start)} sec")
         N = None
      else:
         L = blockwise_cholesky(N, block_size=20000)
         end = time.time()
         print(f"Blockwise-Cholesky factor: {int(end - start)} sec")
         
      if last_iteration: # compute also variance
         start = time.time()
         # Compute L^{-1} efficiently
         # Each column of Linv satisfies: L @ x = e_i
         # So we can solve for all columns at once
         Linv = la.solve_triangular(L, np.eye(L.shape[0]), lower=True)
         end = time.time()
         print(f"solve_triangular L^-1: {int(end - start)} sec")

         y = Linv @ bn
         sol = Linv.T @ y
         var = np.sum(Linv**2, axis=0)
         
      else:
         # Forward solve L y = b
         y = la.solve_triangular(L, bn, lower=True)
         # Backward solve L.T sol = y
         sol = la.solve_triangular(L.T, y, lower=False)
         
   if solving_method == "cg":
      print("Solving using conjugate-gradient (cg)")
      start = time.time()
      sol, info = spla.cg(N, bn, tol=tol, maxiter=1000)
            
      end = time.time()
      print(f"Solve using cg: {int(end - start)} sec")
      if info != 0:
         print("cg finished with info=", info)

   # Compute variance if not available (full inverse or stochastic estimator)
   if solving_method != "cholesky" and last_iteration:
      start = time.time()
      if estimate_var:
         # estimate variance with N and cg may give <0 values
         # var = stochastic_diag_estimate_N(N, num_samples=20, tol=1e-8)
         var = estimate_diag_inv_AtA(spA, num_probes=5, tol=1e-8)
         end = time.time()
         print(f"Variance estimator using cg(N) s=5: {int(end - start)} sec")
         if np.min(var)<0:
            print(f"{len(var[var<0])}/{len(var)}")
      else:
         var = np.asarray(np.linalg.inv(N.todense()).diagonal())
         end = time.time()
         print(f"Variance from inv: {int(end - start)} sec")

   # scale back
   if scaling:
      spA = spA @ diags(col_norms) # (for later use?)
      sol = sol/col_norms
      N = None # cannot be use as it is, will have to be recomputed
      if last_iteration:
         var = var/(col_norms ** 2)

   if not last_iteration:
      var = None

   return sol, var, N, L

def create_observation_blocks(xovers):
   
   orbA0 = xovers['orbA'].str[0]
   orbB0 = xovers['orbB'].str[0]
      
   unique_chars = list(set(pd.concat([orbA0, orbB0])))
   if len(unique_chars) > 1:
      unique_chars.sort()
   
   print("Creating observation blocks based on orbA/B starting with", unique_chars)
   
   if True:
      blocks = [(orbA0 == c) & (orbB0 == c) for c in unique_chars]
      if len(unique_chars) > 1:
         blocks.append((orbA0 == unique_chars[0]) & (orbB0 == unique_chars[1]) |
                       (orbA0 == unique_chars[1]) & (orbB0 == unique_chars[0]))   
   elif False:
      lat_threshold = 88
      blocks = []
      i = 0
      for c in unique_chars:
         i+=1
         # if c == '2':
         #    lat_threshold = 85
         # else:
         #    lat_threshold = 75            
         print(f"Block #{i}: orbA=orbB={c}*, |lat|<{lat_threshold}°")
         blocks.append((orbA0 == c) & (orbB0 == c) & (np.abs(xovers['LAT'])<lat_threshold))
         i+=1
         print(f"Block #{i}: orbA=orbB={c}*, |lat|>={lat_threshold}°")
         blocks.append((orbA0 == c) & (orbB0 == c) & (np.abs(xovers['LAT'])>=lat_threshold))
      if len(unique_chars) > 1:
         lat_threshold = 75
         i+=1
         print(f"Block #{i}: (orbA,orbB)=({unique_chars})*, |lat|<{lat_threshold}°")
         blocks.append(((orbA0 == unique_chars[0]) & (orbB0 == unique_chars[1]) |
                       (orbA0 == unique_chars[1]) & (orbB0 == unique_chars[0])) &
                       (np.abs(xovers['LAT'])<lat_threshold))
         i+=1
         print(f"Block #{i}: (orbA,orbB)=({unique_chars})*, |lat|>={lat_threshold}°")
         blocks.append(((orbA0 == unique_chars[0]) & (orbB0 == unique_chars[1]) |
                       (orbA0 == unique_chars[1]) & (orbB0 == unique_chars[0])) &
                       (np.abs(xovers['LAT'])>=lat_threshold))
   else:
      blocks = []
      threshold=1000
      i = 0
      abs_lat = np.abs(xovers['LAT'])

      def process_block(filter_mask, label):
         nonlocal i
         lat_0, lat_1 = 0, 10
         while lat_0 < 90:
            lat_mask = (abs_lat >= lat_0) & (abs_lat < lat_1)
            b = filter_mask & lat_mask
            count = b.sum()
            if count > threshold or lat_1 == 90:
                i += 1
                print(f"Block #{i}: {label}, {lat_0}°<=|lat|<{lat_1}°")
                blocks.append(b)
                lat_0 = lat_1
            else:
                print(f"{count} obs for {label}, {lat_0}°<=|lat|<{lat_1}°: accumulating...")
            lat_1 += 10

      # Process same-orbit blocks
      for c in unique_chars:
         mask = (orbA0 == c) & (orbB0 == c)
         process_block(mask, f"orbA=orbB={c}*")

      # Process mixed-orbit block if more than one unique char
      if len(unique_chars) > 1:
         c1, c2 = unique_chars[:2]
         mask = ((orbA0 == c1) & (orbB0 == c2)) | ((orbA0 == c2) & (orbB0 == c1))
         process_block(mask, f"(orbA,orbB)=({c1},{c2})*")

   blocks = [b for b in blocks if b.any()] # remove block with all false

   print(f"Separating observations in {len(blocks)} block(s) of {[sum(b) for b in blocks]} observations")

   assert sum([sum(b) for b in blocks]) == len(blocks[0])

   return blocks

def make_blocks(xovers, unique_chars, threshold=100):
    blocks = []
    i = 0
    abs_lat = np.abs(xovers['LAT'])

    def process_block(filter_mask, label):
        nonlocal i
        lat_0, lat_1 = 0, 10
        while lat_0 < 90:
            lat_mask = (abs_lat >= lat_0) & (abs_lat < lat_1)
            b = filter_mask & lat_mask
            count = b.sum()
            if count > threshold or lat_1 == 90:
                i += 1
                print(f"Block #{i}: {label}, {lat_0}°<=|lat|<{lat_1}°")
                blocks.append(b)
                lat_0 = lat_1
            else:
                print(f"{count} obs for {label}, {lat_0}°<=|lat|<{lat_1}°: accumulating...")
            lat_1 += 10

    # Process same-orbit blocks
    orbA0 = xovers['orbA'].str[0]
    orbB0 = xovers['orbB'].str[0]
    for c in unique_chars:
        mask = (orbA0 == c) & (orbB0 == c)
        process_block(mask, f"orbA=orbB={c}*")

    # Process mixed-orbit block if more than one unique char
    if len(unique_chars) > 1:
        c1, c2 = unique_chars[:2]
        mask = ((orbA0 == c1) & (orbB0 == c2)) | ((orbA0 == c2) & (orbB0 == c1))
        process_block(mask, f"(orbA,orbB)=({c1},{c2})*")

    return blocks

def clean_solution(sol_dict):
   # remove corrections OF SINGLE ITER if "unreasonable" (larger than 100 meters in any direction, or 50 meters/day, or 20 arcsec)
   sol_dict_iter = sol_dict
   sol_dict_iter_clean = []
   std_dict_iter_clean = []
   regex = re.compile(".*_dR/d[A,C,R,Rl,Pt]0{0,1}$")
   tracks = list(set([x.split('_')[0] for x, v in sol_dict_iter['sol'].items() if regex.match(x)]))
   bad_count = 0
   for tr in tracks:
      regex = re.compile("^" + tr + "*")
      soltmp = dict([(x, v) for x, v in sol_dict_iter['sol'].items() if regex.match(x)])
      if 'std' in sol_dict_iter.keys():
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
      if 'std' in sol_dict_iter.keys():
         std_dict_iter_clean.append(stdtmp)

   # add back global parameters
   if len(XovOpt.get("sol4_glo")) > 0:
      sol_dict_iter_clean.append(
         dict([(x, v) for x, v in sol_dict_iter['sol'].items() if x in XovOpt.get("sol4_glo")]))
      if 'std' in sol_dict_iter.keys():
         std_dict_iter_clean.append(
            dict([(x, v) for x, v in sol_dict_iter['std'].items() if x in XovOpt.get("sol4_glo")]))

   sol_dict_iter_clean = {k: v for d in sol_dict_iter_clean for k, v in d.items()}
   if 'std' in sol_dict_iter.keys():
      std_dict_iter_clean = {k: v for d in std_dict_iter_clean for k, v in d.items()}
      sol_dict_iter_clean = dict(zip(['sol', 'std'], [sol_dict_iter_clean, std_dict_iter_clean]))
   else:
      sol_dict_iter_clean = {'sol': sol_dict_iter_clean}
   # print("cleaned solution")
   # print(len(sol_dict_iter_clean['sol']))
   # print(len(sol_dict_iter_clean['std']))
   # this probably goes to 0 when sol4_orb = []
   # WD: -3*badcount?
   print("New length of cleaned sol", len(sol_dict_iter_clean['sol']) - bad_count)
   
   return sol_dict_iter_clean

######## MAIN ##########
# @profile
def main(arg):
   ##############################################
   # launch program and clock
   # -----------------------------
   startT = time.time()

   datasets = arg[0]  # ['sim_mlatimes/0res_35amp']
   data_sim = arg[1]
   ext_iter = arg[2]
   opts     = arg[3]
   acc_opts = arg[4]

   # update options (needed when sending to slurm)
   XovOpt.clone(opts)
   XovOpt.check_consistency()
   AccOpt.clone(acc_opts)
   AccOpt.check_consistency()
   
   print("Crossovers loaded from directories:")
   print(datasets)
   print("data_sim (not used?)")
   print(data_sim)
   print(f"Iteration n°{ext_iter}")
   print("XovOpt")
   print("------")
   XovOpt.display()
   print("\nAccOpt")
   print("------")
   AccOpt.display()
   print("\n")

   # WD: Compared to before, different datasets should be processed in
   # separate AccumXov runs, unless combined
   # for ds in [datasets[0]]:
   ds = datasets[0] 
   data_pth = XovOpt.get("outdir") + ds
   vecopts = XovOpt.get("vecopts")
      
   start = time.time()
   
   if AccOpt.get("Abmat_infile") == "":
      print("Load xovers from datasets")
      xov_cmb = load_combine(datasets, vecopts)
      end = time.time()
      print("Xovers loaded in ", int(end - start), "sec or ", round((end - start) / 60., 2), " min!")

   # # count occurrences for each orbit ID
   # xov_cmb.nobs_x_track = xov_cmb.xovers[['orbA','orbB']].apply(pd.Series.value_counts).sum(axis=1).sort_values(ascending=False)

   if XovOpt.get("partials"):
      # load previous iter from disk (orbs, sols, etc) if available
      if ext_iter > 0:
         previous_iter = load_previous_iter_if_any(ds, ext_iter, xov_cmb)
      else:
         previous_iter = None
      if AccOpt.get("Abmat_infile") == "":

         if ext_iter > 0 and previous_iter.converged and 'dR/dh2' not in XovOpt.get("sol4_glo"):
            print("Adding h2 to sol4_glo as solution converged...")
            XovOpt.get("sol4_glo").extend(['dR/dh2'])

         # solve dataset
         par_list = ['orbA', 'orbB', 'xOvID']
         xovi_amat = prepare_Amat(xov_cmb, vecopts, par_list)

         # Xovers downsizing and subsampling
         if (AccOpt.get("downsize") or AccOpt.get("sampling")) and ext_iter == 0:
            # downsize dataset by removing worst weighted data (mostly at high latitudes)
            # also removes very bad xovers with dR > 1km
            max_xovers = 8.5e5  # 3.e6
            if AccOpt.get("downsize") and len(xovi_amat.xov.xovers) > max_xovers:
               # actually preparing weights and constraints for the solution (weights are needed for downsampling)
               prepro_weights_constr(xovi_amat, previous_iter=previous_iter)
               # downsize
               # xovi_amat.xov.xovers = downsize_xovers(xovi_amat.xov.xovers, max_xovers=max_xovers, max_dR = 1.e2)
               xovi_amat.xov.xovers = downsize_xovers(xovi_amat.xov.xovers, max_xovers=max_xovers, max_dR = 1.e4,lat_threshold = 88)
               # xovi_amat.xov.xovers = downsize_xovers(xovi_amat.xov.xovers, max_xovers=max_xovers, lat_threshold = 80)
               # xovi_amat.xov.xovers = xovi_amat.xov.xovers.loc[xovi_amat.xov.xovers['dR'].abs() < 1.e3]

               # reset weights and sol4pars variables for modified dataset
               xovi_amat.xov.combine([xovi_amat.xov])
               xovi_amat = prepare_Amat(xovi_amat.xov, vecopts, par_list)

            # subsample with replacement for bootstrap test
            if AccOpt.get("sampling") and ext_iter == 0:
               # get seed as experiment name
               rand_seed = int(datasets[0].split('/')[-3].split('_')[0][-1])
               xovi_amat.xov.xovers = subsample_xovers(xovi_amat.xov.xovers, size_samples=5.e5,
                                                       rand_seed=rand_seed)
               
               # reset weights and sol4pars variables for modified dataset
               xovi_amat.xov.combine([xovi_amat.xov])
               xovi_amat = prepare_Amat(xovi_amat.xov, vecopts, par_list)

         # actually preparing weights and constraints for the solution
         prepro_weights_constr(xovi_amat, previous_iter=previous_iter)
         
         Amat_fname = (data_pth + 'Abmat_' + ds.split('/')[0] + '_' + ds.split('/')[1] + '_')[:-1] + str(ext_iter + 1) + '_nosol.pkl'
         xovi_amat.save(Amat_fname)
      else:
         xovi_amat = Amat(vecopts)
         xovi_amat = xovi_amat.load(data_pth + AccOpt.get("Abmat_infile"))

      if not AccOpt.get("get_cov_only"):
         xovi_amat.spA = None

      xovi_amat.obs_blocks = create_observation_blocks(xovi_amat.xov.xovers)
      
      tracks_to_remove = get_bad_tracks(xovi_amat)

      xovi_amat.penalty_mat = compute_penalty_matrices(xovi_amat, tracks_to_remove)
      
      remove_tracks(xovi_amat, tracks_to_remove.axes[0].tolist())

      # Where all the weight from the penalty matrix went???
      if previous_iter != None and previous_iter.vce_obs != None and previous_iter.vce_pen != None:
         xovi_amat.vce_obs = previous_iter.vce_obs
         xovi_amat.vce_pen = previous_iter.vce_pen
      else:
         xovi_amat.vce_obs = AccOpt.get("weight_obs")
         xovi_amat.vce_pen = AccOpt.get("weight_constr")
         if len(xovi_amat.vce_obs) != len(xovi_amat.obs_blocks):
            print(f"{len(xovi_amat.obs_blocks)} penalty matrices for {xovi_amat.vce_obs}")
            xovi_amat.vce_obs = np.ones(len(xovi_amat.obs_blocks))
            print(f"Initial VCE weights set to {xovi_amat.vce_obs}")
         if len(xovi_amat.vce_pen) != len(xovi_amat.penalty_mat):
            print(f"{len(xovi_amat.penalty_mat)} penalty matrices for {xovi_amat.vce_pen}")
            xovi_amat.vce_pen = np.ones(len(xovi_amat.penalty_mat))
            print(f"Initial VCE weights set to {xovi_amat.vce_pen}")
         # xovi_amat.vce = [0.0002247404434024504, 5.0025679108113685, 0.0010878786212904351]

      start = time.time()
      compute_solution(xovi_amat, previous_iter, xovi_amat.xov)
      end = time.time()

      print("compute_solution finished after", int(end - start), "sec or ", round((end - start) / 60., 2), " min!")

   else:
      # TODO also compute weights and store them (no reason not to do so w/o partials)
      # create Amat
      xovi_amat = prepare_Amat(xov_cmb, vecopts)

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


   # set as converged if relative improvement of residuals RMSE lower than convergence criteria
   if ext_iter > 0 and XovOpt.get("partials"):
      relative_improvement = np.abs((xovi_amat.resid_wrmse - previous_iter.resid_wrmse) / xovi_amat.resid_wrmse)
      print("Relative improvement at ", (relative_improvement * 100.).round(2), "% at iteration", ext_iter)
      if relative_improvement <= AccOpt.get("convergence_criteria"):
         print("Solution converged at iter", ext_iter)
         xovi_amat.converged = True
      elif previous_iter.converged == True:
         print("Solution already converged...")
         xovi_amat.converged = True
      else:
         xovi_amat.converged = False

   # TODO not sure wether it can also be saved when just computing residuals
   # if partials:
   if AccOpt.get("Abmat_outfile") != "":
      Amat_fname =  data_pth + AccOpt.get("Abmat_outfile")
   elif len(ds.split('/')) > 2:
      Amat_fname = ('_').join((data_pth + 'Abmat_' + ds.split('/')[0] + '_' +
                               ds.split('/')[1]).split('_')[:-1]) + '_' + \
                                  str(ext_iter + 1) + '_' + ds.split('/')[2] + '.pkl'
   else:
      Amat_fname = (data_pth + 'Abmat_' + ds.split('/')[0] + '_' + ds.split('/')[1] + '_')[:-1] + str(ext_iter + 1) + '.pkl'
   print(Amat_fname)
   # What about spA_penal, b_penal, penalty_mat_avg
   xovi_amat.spA = None
   xovi_amat.b = None
   xovi_amat.weights = None
   xovi_amat.save(Amat_fname)

   print("AccumXov ended succesfully!")
   ##############################################
   # stop clock and print runtime
   # -----------------------------
   endT = time.time()
   print('----- Runtime Amat = ' + str(endT - startT) + ' sec -----' + str(
      (endT - startT) / 60.) + ' min -----')
   return


########################
if __name__ == '__main__':
   import sys

   ##############################################
   # launch program and clock
   # -----------------------------
   startT = time.time()

   # print(sys.argv)
   main(sys.argv[1:])

   ##############################################
   # stop clock and print runtime
   # -----------------------------
   endT = time.time()
   print('----- Runtime Amat = ' + str(endT - startT) + ' sec -----' + str(
      (endT - startT) / 60.) + ' min -----')
