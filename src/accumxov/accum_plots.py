#!/usr/bin/env python3
# ----------------------------------
# Plotting utilities
# ----------------------------------
# Author: Stefano Bertone
# Created: 04-Mar-2019
#
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from accumxov.accum_opt import AccOpt
from config import XovOpt
from xovutil.stat import rms

def plot_obs_weights(tmp):
   # plot histo
   plt.figure()  # figsize=(8,3))
   # plt.xlim(-1.*xlim, xlim)
   # the histogram of the data
   num_bins = 200  # 'auto'
   n, bins, patches = plt.hist(tmp, bins=num_bins,range=[1.e-4, 4.e-2])  # , cumulative=-1) #, density=True, facecolor='blue',
   # alpha=0.7, range=[-1.*xlim, xlim])
   plt.xlabel('obs weights')
   plt.ylabel('# tracks')
   plt.title('Resid+distance+offnadir+interp+weights: $\mu=' + str(np.mean(tmp)) + ', \sigma=' + str(
               np.std(tmp)) + '$')
   # # Tweak spacing to prevent clipping of ylabel
   plt.subplots_adjust(left=0.15)
   plt.savefig(XovOpt.get("tmpdir") + '/data_weights.png')
   plt.clf()

def plot_partials(spA_sol4, obs_weights):
   
   dw_dh2 = spA_sol4[:, -1].toarray()
   dw_dra = spA_sol4[:, -2].toarray()
   dw_dpm = spA_sol4[:, -4].toarray()
   dw_dl = spA_sol4[:, -3].toarray()
   dw_ddec = spA_sol4[:, -5].toarray()

   plt.clf()
   plt.figure()  # figsize=(8, 3))
   xlim = 1.
   # plt.xlim(-1. * xlim, xlim)
   # the histogram of the data
   num_bins = 200  # 'auto' #

   parnam = ['PM', 'L', 'DEC', 'RA', 'h2']

   for idx, par in enumerate([dw_dpm, dw_dl, dw_ddec, dw_dra, dw_dh2]):
      tmp = np.abs(par)
      n, bins, patches = plt.hist(tmp, bins=num_bins, density=False,
                                  alpha=0.8, label=parnam[idx], weights=obs_weights.diagonal())
   plt.ylim(bottom=AccOpt.get("sigma_0"))
   plt.semilogy()
   plt.legend()
   plt.ylabel('# of obs')
   plt.xlabel('meters/[par]')
   plt.savefig(XovOpt.get("tmpdir") + "partials_histo_weighted.png")

   plt.clf()
   for idx, par in enumerate([dw_dpm, dw_dl, dw_ddec, dw_dra, dw_dh2]):
      tmp = np.abs(par)
      n, bins, patches = plt.hist(tmp, bins=num_bins, density=False,
                                  alpha=0.8, label=parnam[idx])
   plt.semilogy()
   plt.legend()
   plt.ylabel('# of obs')
   plt.xlabel('meters/[par]')
   plt.savefig(XovOpt.get("tmpdir") + "partials_histo.png")

def plot_weight_distribution(tmp):
   
   plt.figure()  # figsize=(8, 3))
   num_bins = 100  # 'auto'  # 40  # v
   n, bins, patches = plt.hist(tmp.astype(float), bins=num_bins)
   plt.xlabel('dR (m)')
   plt.ylabel('# tracks')
   plt.savefig(XovOpt.get("tmpdir") + '/histo_tracks_weights.png')
   plt.clf()

def plot_huber_penal(huber_penal):
   num_bins = 100  # 'auto'
   plt.clf()
   n, bins, patches = plt.hist(np.where(huber_penal < 1., huber_penal, 1.).astype(float),
                               bins=num_bins, cumulative=True)
   plt.xlabel('huber_penal')
   plt.savefig(XovOpt.get("tmpdir") + '/histo_huber_h2.png')
   plt.clf()

def plot_res_h2_partials(tmp):
   # print("truc0",tmp['weights'].abs().min(),tmp['weights'].abs().max())
   tmp = tmp.loc[(tmp.dR.abs() < limit_h2) & (tmp['dR/dh2'].abs() > 0.3) & (
      
            tmp['weights'].abs() > 0.5 * AccOpt.get("sigma_0"))]

   w = np.abs(tmp[['dR']].abs().values)
   dw_dh2 = np.abs(tmp[['dR/dh2']].values)  # np.abs(spA_sol4[:,-1].toarray())
   # import statsmodels.api as sm
   # result = sm.OLS(dw_dh2, w).fit()
   print("lenw", len(tmp))

   fig, ax = plt.subplots(1)
   ax.scatter(x=w, y=dw_dh2)  # , color = rgb)
   # ax.set_xlim(0,2.5)
   # n, bins, patches = plt.plot(x=w,y=dw_dh2)
   # plt.semilogy()
   # plt.legend()
   # plt.ylabel('# of obs')
   # plt.xlabel('meters/[par]')
   plt.savefig(XovOpt.get("tmpdir") + "discr_vs_dwdh2.png")

   plt.clf()
   piv = pd.pivot_table(tmp.round({'LON': 0, 'LAT': 0}), values="dR/dh2", index=["LAT"], columns=["LON"],
                        fill_value=None, aggfunc=rms)
   ax = sns.heatmap(piv, xticklabels=10, yticklabels=10, cmap="YlGnBu")  # , square=False, annot=True)
   plt.tight_layout()
   ax.invert_yaxis()
   plt.savefig(XovOpt.get("tmpdir") + "geo_dwdh2.png")

def plot_orbitstd(sol_dict):
   testA = pd.DataFrame.from_dict(sol_dict).filter(like='dR/dA', axis=0).loc[:, 'std']
   testC = pd.DataFrame.from_dict(sol_dict).filter(like='dR/dC', axis=0).loc[:, 'std']
   testR = pd.DataFrame.from_dict(sol_dict).filter(like='dR/dR', axis=0).loc[:, 'std']
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
