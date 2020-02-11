#!/usr/bin/env python3
# ----------------------------------
# accum_utils.py
#
# Description: various methods to be applied to AccumXov
#
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 05-Feb-2020

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

from statsmodels.tools.eval_measures import rmse

from prOpt import tmpdir, vecopts
from project_coord import project_stereographic
import matplotlib.pyplot as plt
import statsmodels.api as sm

from xov_utils import get_tracks_rms

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
    tracks_map = dict(zip(unique_orb,range(len(unique_orb))))
    tmp['mapA'] = tmp['orbA'].map(tracks_map)
    tmp['mapB'] = tmp['orbB'].map(tracks_map)
    # generate sparse matrix of ones for each track, then sum to have 2 "ones-elements" for each xov
    csrA = csr_matrix((np.ones(len(tmp['xOvID'].values)), (tmp['xOvID'].values, tmp['mapA'].values)),
                   dtype=np.float32, shape=(len(tmp['xOvID'].values), len(unique_orb)))
    csrB = csr_matrix((np.ones(len(tmp['xOvID'].values)), (tmp['xOvID'].values, tmp['mapB'].values)),
                   dtype=np.float32, shape=(len(tmp['xOvID'].values), len(unique_orb)))
    A_tracks = csrA+csrB
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
    huber_threshold_track = 20
    tmp = tracks_rms_df.sort_values(by='track').pre.abs().values
    huber_weights_track = np.where(tmp > huber_threshold_track, (tmp / huber_threshold_track) ** 2, 1.)

    if False:
        # plot histo
        plt.figure(figsize=(8,3))
        # plt.xlim(-1.*xlim, xlim)
        # the histogram of the data
        num_bins = 40 # 'auto'
        n, bins, patches = plt.hist(huber_weights_track.astype(np.float), bins=num_bins) #, density=True, facecolor='blue',
        # alpha=0.7, range=[-1.*xlim, xlim])
        plt.xlabel('weight (1/m)')
        plt.ylabel('# tracks')
        # plt.title(r'Histogram of dR: $\mu=' + str(mean_dR) + ', \sigma=' + str(std_dR) + '$')
        # # Tweak spacing to prevent clipping of ylabel
        plt.subplots_adjust(left=0.15)
        plt.savefig(tmpdir+'/huber_weights_track.png')
        plt.clf()
        exit()

    cov_tracks = diags(huber_weights_track, 0)

    # project variance of individual tracks on xovers
    cov_xov_tracks = cov_tracks * A_tracks.transpose()
    cov_xov_tracks = A_tracks * cov_xov_tracks
    np.reciprocal(cov_xov_tracks.data, out=cov_xov_tracks.data)

    return cov_xov_tracks
