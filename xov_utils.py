#!/usr/bin/env python3
# ----------------------------------
# xov_utils.py
#
# Description: various methods to be applied to xovs
#
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 04-Feb-2020

import numpy as np
import pandas as pd

from statsmodels.tools.eval_measures import rmse

from prOpt import tmpdir, vecopts, debug, local
from project_coord import project_stereographic
import matplotlib.pyplot as plt
import statsmodels.api as sm

def get_tracks_rms(xovers_df, plot_xov_tseries=False):

    print("Checking tracks rms @ iter ...")

    xovtmp = xovers_df[['LON', 'LAT', 'dtA', 'dR', 'orbA', 'orbB', 'huber']].copy()
    xovtmp = xovtmp.astype({'orbA': 'int32', 'orbB': 'int32'})
    total_occ_tracks = pd.DataFrame([xovtmp['orbA'].value_counts(), xovtmp['orbB'].value_counts()]).T.fillna(0).sum \
        (axis=1).sort_values(ascending=False)
    tracks = list(total_occ_tracks.index.values)[:]  # ['1403130156'] #

    # use range(n) with n order of fit cases in "parametrizations"
    rmslist = []
    rmsprelist = []
    biaslist = []
    trlist = []
    for tr in tracks:

        # cols = dict(zip(xovtmp.columns,range(len(xovtmp.columns))))
        # print(cols)
        # print(cols['huber'])
        # tmp = xovtmp.values
        # tmp = tmp[tmp[:,cols['huber']]>0.01]
        # tmp = tmp[(tmp[:,cols['orbA']]==tr) | (tmp[:,cols['orbB']]==tr)]
        # print(list(cols.keys()))
        # df = pd.DataFrame(tmp,columns=list(cols.keys()))#.sort_values(by='dtA',ascending=True)
        # print(df)

        df = xovtmp.loc[((xovtmp.orbA == int(tr)) | (xovtmp.orbB == int(tr))) & (xovtmp.huber > 0.01)].sort_values(by='dtA',
                                                                                                         ascending=True)
        tmp = pd.DataFrame(
            np.vstack(project_stereographic(df.LON.values, df.LAT.values, 0, 90, vecopts['PLANETRADIUS'])).T,
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
        try:
            rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
            rlm_results = rlm_model.fit()
        except:
            print(tr,y)

        if False and debug:
            print(rlm_results.summary())
            print(rlm_results.params)

        rmspre = rmse(y, np.zeros(len(y)))
        # only if param with bias only
        X = X[..., np.newaxis]
        rmspost = rmse(y, X.dot(rlm_results.params))

        if local and plot_xov_tseries and rmspre > 100.:
            fig = plt.figure(figsize=(12, 8))
            plt.style.use('seaborn-poster')
            ax = fig.add_subplot(111)
            ax.plot(x, y, 'o', label="dR #"+str(tr))

            ax.plot(x, np.ones(len(result.fittedvalues)) * rmspre, 'r--')
            ax.plot(x, -1. * np.ones(len(result.fittedvalues)) * rmspre, 'r--')
            ax.legend(loc="best")
            ax.set_xlim(right=1500)
            title = "res R bias : " + str(rlm_results.params.round(1)[0]) + " m -- RMSE : " + str(
                rmspre.round(1)) + " m"  # / "+ str(rmspost.round(1)) + "m"
            ax.set_title(title)
            plt.savefig(tmpdir + 'orb_res_xov_' + str(tr) + '.png')

        trlist.append(tr)
        biaslist.append(rlm_results.params.round(1)[0])
        rmsprelist.append(rmspre.round(1))
        rmslist.append(rmspost.round(1))

    postfit = pd.DataFrame(np.vstack([trlist, rmsprelist, biaslist, rmslist]).T,
                           columns=['track', 'pre', 'bias', 'minus-Rbias']).astype(float).astype({'track': int})

    if local and plot_xov_tseries:
        plot_tracks_histo([postfit])

    return postfit


def plot_tracks_histo(postfit_list, filename=tmpdir + '/histo_tracks_eval.png'):
    # plot histo
    plt.figure(figsize=(8, 3))
    # plt.xlim(-1.*xlim, xlim)
    # the histogram of the data
    num_bins = 40  # 'auto'
    for idx, postfit in enumerate(postfit_list):
        n, bins, patches = plt.hist(postfit.pre.astype(np.float), bins=num_bins, alpha=0.7,
                                    label="iter "+str(idx))  # , density=True) #, facecolor='blue',
    # alpha=0.7, range=[-1.*xlim, xlim])
    plt.xlabel('dR (m)')
    plt.ylabel('# tracks')
    plt.legend()
    plt.title('Histogram of track RMS at iters')
    plt.semilogx()
    # # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig(filename)
    plt.clf()