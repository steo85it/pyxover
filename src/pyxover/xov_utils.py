# !/usr/bin/env python3
# ----------------------------------
# xov_utils.py
#
# Description: various methods to be applied to xovs
#
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 04-Feb-2020
import glob
import os

import numpy as np
import pandas as pd

# from eval_sol import rmse
from matplotlib import pyplot as plt
from scipy.sparse import identity, csr_matrix
from statsmodels.tools.eval_measures import rmse

# from AccumXov import remove_max_dist
# from accum_utils import analyze_dist_vs_dR
# from AccumXov import remove_max_dist
# from accum_utils import analyze_dist_vs_dR
# from examples.MLA.options import XovOpt.get("tmpdir"), XovOpt.get("vecopts"), XovOpt.get("local"), XovOpt.get("debug"), XovOpt.get("sol4_orbpar"), XovOpt.get("parOrb"), XovOpt.get("parGlo")
from accumxov.accum_opt import AccOpt
from config import XovOpt

from xovutil.project_coord import project_stereographic
import matplotlib.pyplot as plt
import statsmodels.api as sm

# @profile
from pyxover.xov_setup import xov


def get_tracks_rms(xovers_df, plot_xov_tseries=False):

   if XovOpt.get("debug"):
      print("Checking tracks rms @ iter ...")

   if 'huber' in xovers_df.columns:
      xovtmp = xovers_df[['LON', 'LAT', 'dtA', 'dR', 'orbA', 'orbB', 'huber']].copy()
      acceptif = xovtmp.huber > 0.01
   else:
      xovtmp = xovers_df[['LON', 'LAT', 'dtA', 'dR', 'orbA', 'orbB', 'weights']].copy()
      weights_mean = np.mean(xovtmp.weights)
      acceptif = xovtmp.weights > 0.1*weights_mean

   xovtmp = xovtmp.astype({'orbA': 'int32', 'orbB': 'int32'})
   total_occ_tracks = pd.DataFrame([xovtmp['orbA'].value_counts(), xovtmp['orbB'].value_counts()]).T.fillna(0).sum \
        (axis=1).sort_values(ascending=False)
   tracks = list(total_occ_tracks.index.values)[:]  # ['1403130156'] #

   # use range(n) with n order of fit cases in "parametrizations"
   rmslist = []
   rmsprelist = []
   biaslist = []
   driftlist = []
   trlist = []
   for tr in tracks:
      
      try:
         df = xovtmp.loc[((xovtmp.orbA == int(tr)) | (xovtmp.orbB == int(tr))) & acceptif].sort_values(by='dtA',
                                                                                                             ascending=True)
         tmp = pd.DataFrame(
            np.vstack(project_stereographic(df.LON.values, df.LAT.values, 0, 90, XovOpt.get("vecopts")['PLANETRADIUS'])).T,
            columns=['x0', 'y0'])
         tmp['dist'] = np.linalg.norm(tmp.diff().values, axis=1)
         df = pd.concat([df.reset_index(drop=True), tmp], axis=1).dropna()

         y = df.dR.values
         x = df.dtA.values

         # print(np.column_stack((np.ones(len(x)))))
         # print(np.array(np.ones(len(x))))
         # exit()

         parametrizations = [
            [np.ones(len(x))],
            # np.column_stack((x, np.ones(len(x)))),
            # np.column_stack((x**2, x, np.ones(len(x)))),
            # np.column_stack((x**3, x**2, x, np.ones(len(x))))
            ]
         X = parametrizations[0]
         # only if param with bias only
         X = X[0]

         result = sm.OLS(y, X).fit()

         # Fit model and print summary
         rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
         rlm_results = rlm_model.fit()

         if False and XovOpt.get("debug"):
            # print("rlm_results")
            # print(rlm_results.summary())
            print("rlm_results (x1,const):",np.array(rlm_results.params).round(2))

         rmspre = rmse(y, np.zeros(len(y)))
         # only if param with bias only
         X = X[..., np.newaxis]
         rmspost = rmse(y, X.dot(rlm_results.params))

         # if bias only
         Rbias = rlm_results.params.round(1)[0]
         Rdrift = 0.
         # if bias+drift
         # Rbias = rlm_results.params.round(1)[1]
         # Rdrift = rlm_results.params.round(1)[0]

         if XovOpt.get("local") and plot_xov_tseries and np.abs(Rbias) > 30.:
            fig = plt.figure(figsize=(12, 8))
            plt.style.use('seaborn-poster')
            ax = fig.add_subplot(111)
            ax.plot(x, y, 'o', label="dR #" + str(tr))

            ax.plot(x, np.ones(len(result.fittedvalues)) * rmspre, 'r--')
            ax.plot(x, -1. * np.ones(len(result.fittedvalues)) * rmspre, 'r--')
            ax.legend(loc="best")
            # ax.set_xlim(right=1500)
            # if bias+drift
            # title = "sol: R bias : " + str(rlm_results.params.round(1)[1]) + \
            #         "m, R drift : " + str(rlm_results.params.round(2)[0]) + " m/s -- RMSE : " + \
            #         str(rmspre.round(1)) + " m"  # / "+ str(rmspost.round(1)) + "m"
            # if bias only
            title = "sol: R bias : " + str(rlm_results.params.round(1)[0]) + " m -- RMSE : " + \
               str(rmspre.round(1)) + " m"  # / "+ str(rmspost.round(1)) + "m"
            ax.set_title(title)
            plt.savefig(XovOpt.get("tmpdir") + 'orb_res_xov_' + str(tr) + '.png')

         trlist.append(tr)
         biaslist.append(Rbias)
         driftlist.append(Rdrift)
         rmsprelist.append(rmspre.round(1))
         rmslist.append(rmspost.round(1))

         # print(len(trlist),len(biaslist),len(rmsprelist),len(driftlist),len(rmslist))

      except:
         trlist.append(tr)
         biaslist.append(1.e-6)
         driftlist.append(1.e-6)
         rmsprelist.append(1.e-6)
         rmslist.append(1.e-6)
         # print(tr,y)

      # exit()

   postfit = pd.DataFrame(np.vstack([trlist, rmsprelist, biaslist, driftlist, rmslist]).T,
                          columns=['track', 'pre', 'bias', 'drift', 'minus-Rbias']).astype(float).astype({'track': int})

   return postfit


def plot_tracks_histo(postfit_list, filename=XovOpt.get("tmpdir") + '/histo_tracks_eval.png'):
    # plot histo
    # print(postfit_list)

    xlim = 1.e2
    plt.clf()
    fig = plt.figure(figsize=(7, 3))
    plt.style.use('seaborn-paper')

    # plt.xlim(-1.*xlim, xlim)
    # the histogram of the data
    num_bins = 40  # 'auto'
    plt_labels = ['pre-fit','post-fit']
    for idx, postfit in enumerate(postfit_list):
        # print(postfit.pre.astype(float))
        n, bins, patches = plt.hist(postfit.pre.astype(float), bins=num_bins, alpha=0.7,
                                    label=plt_labels[idx],range=[0., xlim])  # , density=True) #, facecolor='blue',
    # alpha=0.7, range=[-1.*xlim, xlim])
    plt.xlabel('dR (m)')
    plt.ylabel('# tracks')
    plt.legend()
    plt.title('Histogram of track RMS at iters')
    # plt.semilogx()
    # # Tweak spacing to prevent clipping of ylabel
    # plt.subplots_adjust(left=0.15)
    fig.tight_layout()
    plt.savefig(filename)
    print("### Tracks histo saved as", filename)

    plt.clf()


def load_combine(xov_pth_,vecopts,dataset='sim'):
   # -------------------------------
   # Amat setup
   # -------------------------------
   pd.set_option('display.max_columns', 500)

   # Combine all xovers and setup Amat
   xov_ = xov(vecopts)

   # modify this selection to use sub-sample of xov only!!
   #------------------------------------------------------
   # print(xov_pth_)
   # allFiles = glob.glob(os.path.join(xov_pth, 'xov/xov_*.pkl'))
   allFiles = []
   for xov_pth in xov_pth_:
      allFiles = allFiles + glob.glob(os.path.join(XovOpt.get("outdir"), xov_pth, 'xov/xov_*.pkl'))

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
   if xov_list[0].pert_cloop_0 != None:
      test_pert = len(list(xov_list[0].pert_cloop_0.values())[0])
   else:
      test_pert = 0

   pertdict = [x.pert_cloop_0 for x in xov_list if hasattr(x, 'pert_cloop_0')]
   if test_pert>0 and len([v for x in pertdict for k,v in x.items() if v]) > 0 and XovOpt.get("sol4_orbpar") != [None]:
      pertdict = {k: v for x in pertdict for k, v in x.items() if v is not None}
      xov_cmb.pert_cloop_0 = pd.DataFrame(pertdict).T
      xov_cmb.pert_cloop_0.drop_duplicates(inplace=True)
   else:
      xov_cmb.pert_cloop_0 = pd.DataFrame()

   return xov_cmb


def clean_xov(xov, par_list=[]):
   from accumxov.accum_utils import analyze_dist_vs_dR
   # from accumxov.accum_opt import remove_max_dist

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

      if AccOpt.get("remove_max_dist"):
         print(len(xov.xovers[xov.xovers.dist_max < 0.4]),
               'xovers removed by dist from obs > 0.4km out of ', len(xov.xovers))
         xov.xovers = xov.xovers[xov.xovers.dist_max < 0.4]
         #xov.xovers = xov.xovers[xov.xovers.dist_min_mean < 1]

   return xov


def clean_partials(b, spA, glbpars, threshold = 1.e6):

   nglbpars = len(glbpars)

   if XovOpt.get("debug"):
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
                axlst[idx].plot(spA[:, -nglbpars + idx].todense(), label=glbpars[idx])
                # i.plot(spA[:, :-4].sum(axis=1).A1, label=sol4_glo[idx])
                axlst[idx].legend(loc='upper right')
        else:
            axlst.plot(spA[:, -nglbpars + 0].todense(), label=glbpars[0])
            # i.plot(spA[:, :-4].sum(axis=1).A1, label=sol4_glo[idx])
            axlst.legend(loc='upper right')
        # i.plot(b)
        plt.savefig(XovOpt.get("tmpdir") + 'b_and_A_pre.png')
   # exit()

   Nexcluded = 0
   # print(spla.norm(spA[:,-5:],axis=0))
   for i in range(nglbpars):
      # if an error arises, check the hard-coded list of solved for
      # global parameters
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
   if XovOpt.get("debug"):
      plt.clf()
      fig, axlst = plt.subplots(nglbpars)
      if nglbpars>1:
         for idx in range(nglbpars):
            axlst[idx].plot(spA[:, -nglbpars + idx].todense(), label=glbpars[idx])
            # i.plot(spA[:, :-4].sum(axis=1).A1, label=sol4_glo[idx])
            axlst[idx].legend(loc='upper right')
      else:
         axlst.plot(spA[:, -nglbpars + 0].todense(), label=glbpars[0])
         # i.plot(spA[:, :-4].sum(axis=1).A1, label=sol4_glo[idx])
         axlst.legend(loc='upper right')
      plt.savefig(XovOpt.get("tmpdir") + 'b_and_A_post.png')

   # print(spla.norm(spA[:,-5:],axis=0))
   # exit()

   return b, spA


def get_ds_attrib():
    # prepare pars keys, lists and dicts
    pars = ['d' + x + '/' + y for x in ['LON', 'LAT', 'R'] for y in
            list(XovOpt.get("parOrb").keys()) + list(XovOpt.get("parGlo").keys())]
    delta_pars = {**XovOpt.get("parOrb"), **XovOpt.get("parGlo")}
    etbcs = ['ET_BC_' + x + y for x in list(XovOpt.get("parOrb").keys()) + list(XovOpt.get("parGlo").keys()) for y in ['_p', '_m']]
    return delta_pars, etbcs, pars
