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
from scipy.sparse import csr_matrix, diags, issparse

from prOpt import tmpdir, full_covar, debug, local
import matplotlib.pyplot as plt

from xov_utils import get_tracks_rms
from util import multiply_sparse_get_diag

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
    tmp = tracks_rms_df.sort_values(by='track').bias.abs().values
    huber_weights_track = np.where(tmp > huber_threshold_track, (huber_threshold_track / tmp) ** 1, 1.)

    if plot_stuff and local: # and debug:
        # plot histo
        plt.figure() #figsize=(8, 3))
        # plt.xlim(-1.*xlim, xlim)
        # the histogram of the data
        num_bins = 'auto'
        n, bins, patches = plt.hist(huber_weights_track.astype(np.float),
                                    bins=num_bins) #, cumulative=True)  # , density=True, facecolor='blue',
        # alpha=0.7, range=[-1.*xlim, xlim])
        plt.xlabel('weight (1/m)')
        plt.ylabel('# tracks')
        # plt.title(r'Histogram of dR: $\mu=' + str(mean_dR) + ', \sigma=' + str(std_dR) + '$')
        # # Tweak spacing to prevent clipping of ylabel
        plt.subplots_adjust(left=0.15)
        plt.savefig(tmpdir + '/huber_weights_track.png')
        plt.clf()
        # exit()

    cov_tracks = diags(huber_weights_track.round(2).astype('float16'), 0)

    # project variance of individual tracks on xovers
    # print(A_tracks.getnnz() / np.prod(A_tracks.shape))

    cov_xov_tracks = cov_tracks * A_tracks.transpose()  # .round(2)
    # print(cov_xov_tracks)
    # print(cov_xov_tracks.getnnz(), np.prod(cov_xov_tracks.shape), cov_xov_tracks.getnnz() / np.prod(cov_xov_tracks.shape))
    if full_covar:
        cov_xov_tracks = A_tracks * cov_xov_tracks
    else:
        cov_xov_tracks = diags(multiply_sparse_get_diag(A_tracks, cov_xov_tracks))

    np.reciprocal(cov_xov_tracks.data, out=cov_xov_tracks.data)

    return cov_xov_tracks

def get_vce_factor(Ninv, Cinv, x, b = None, A = None, sapr=1., kind='obs'):
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
    s2apr = sapr**2

    if kind == 'obs':
        ri = (b - A * x)
        Ni = A.T * Cinv * A
    else:
        ri = x
        Ni = Cinv

    # numerator (basically the quantity to minimize)
    rTw = csr_matrix(ri.T)* Cinv
    # print(csr_matrix(ri.T),rTw * ri, Cinv,Cinv.max(),Cinv.min())
    rTwr = (rTw * ri)[0]
    # basically a modified dof for the subset
    redundancy = nelem - (1. / s2apr) * np.trace(Ni @ Ninv)
    print("kind, sqrt(rTwr),redundancy,chi2:",kind,np.sqrt(rTwr),redundancy,np.trace(Ni.todense()),rTwr/redundancy)

    # the new sigma^2 associated to the subset of data or constraint
    return rTwr / redundancy


def downsize_xovers(xov_df,max_xovers=1.e5):
    # remove very large dR (>1km)
    xov_df = xov_df.loc[xov_df['dR'].abs() < 1.e3]
    print(xov_df.columns)
    # print(tmp.weights)
    # exit()
    hilat_xov = xov_df.loc[xov_df.LAT > 60]
    print(hilat_xov[['dR', 'weights', 'huber']].abs().max())
    print(hilat_xov[['dR', 'weights', 'huber']].abs().min())
    print(hilat_xov[['dR', 'weights', 'huber']].abs().mean())
    print(hilat_xov[['dR', 'weights', 'huber']].abs().median())

    # select approx number of xovers to keep and derive proportion to keep at hi-lats
    to_keep = 1. - max_xovers / len(hilat_xov)
    to_keep_hilat = hilat_xov.loc[hilat_xov['weights'] > hilat_xov['weights'].quantile(to_keep)].xOvID.values
    # by default, keep 90% of xovers at low-lats
    lolat_xov = xov_df.loc[xov_df.LAT < 60]
    to_keep_lolat = lolat_xov.loc[lolat_xov['weights'] > lolat_xov['weights'].quantile(0.1)].xOvID.values

    # select very good xovers at LAT>60N OR decent xovers at low latitudes
    selected = xov_df.loc[(xov_df.xOvID.isin(to_keep_hilat)) | (xov_df.xOvID.isin(to_keep_lolat))]
    print(len(selected))
    print(selected[['dR', 'weights', 'huber', 'dist_min_mean']].abs().max())
    print(selected[['dR', 'weights', 'huber', 'dist_min_mean']].abs().min())
    print(selected[['dR', 'weights', 'huber', 'dist_min_mean']].abs().median())
    print(selected[['dR', 'weights', 'huber', 'dist_min_mean']].abs().mean())
    print("Downsized xovers to the 'best'",len(selected),"xovers out of",len(xov_df),". Done!")

    return selected