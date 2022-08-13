#!/usr/bin/env python3
# ----------------------------------
# xov_setup.py
#
# Description: 
# 
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 18-Feb-2019
import warnings

from pyaltsim import perlin2d
from scipy.interpolate import RectBivariateSpline

warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd
import time
from scipy import interpolate
import pickle
import re
import matplotlib.pyplot as plt
import subprocess
# from memory_profiler import profile

# from mapcount import mapcount
from xovutil.unproject_coord import unproject_stereographic
from pyxover.intersection import intersection
# from examples.MLA.options import XovOpt.get("debug"), XovOpt.get("partials"), XovOpt.get("OrbRep"), XovOpt.get("parGlo"), XovOpt.get("parOrb"), XovOpt.get("tmpdir"), XovOpt.get("auxdir"), XovOpt.get("local"), XovOpt.get("multi_xov"), XovOpt.get("new_algo")
from config import XovOpt

from xovutil.iterables import lflatten
from xovutil.stat import rms
from xovutil.units import sec2day


class xov:

    def __init__(self, vecopts):

        self.vecopts = vecopts
        self.xovers = pd.DataFrame(
            columns=['x0', 'y0', 'mla_idA', 'mla_idB', 'cmb_idA', 'cmb_idB', 'R_A', 'R_B', 'dR'])
        self.param = {'': 1.}
        self.proj_center = None
        self.pert_cloop = None
        self.pert_cloop_0 = None
        self.sol_prev_iter = None
        self.gtracks = None
        self.ladata_df = None
        self.msrm_sampl = None
        self.apply_texture = None
        self.parOrb_xy = None
        self.parGlo_xy = None
        self.par_xy = None
        self.t0_tracks = None

    #@profile
    def setup(self, gtracks):

        self.gtracks = gtracks

        if XovOpt.get("new_algo"):
            cols = ['ET_TX', 'TOF', 'orbID', 'seqid', 'ET_BC', 'offnadir', 'LON', 'LAT', 'R',
             'X_stgprj', 'Y_stgprj']
            # df = pd.concat([gtracks[0].ladata_df, gtracks[1].ladata_df],sort=True).reset_index(drop=True)
            self.ladata_df = pd.concat([gtracks[0].ladata_df.loc[:,cols], gtracks[1].ladata_df.loc[:,cols]],sort=False).reset_index(drop=True)
        else:
            df = pd.concat([gtracks[0].ladata_df, gtracks[1].ladata_df],sort=False).reset_index(drop=True)
            self.ladata_df = df.drop('chn', axis=1)

        # store involved tracks as dict
        self.tracks = dict(zip(self.ladata_df.orbID.unique(),list(range(2))))
        # map tracks to 0 and 1
        self.ladata_df['orbID'] = self.ladata_df['orbID'].map(self.tracks)

        if XovOpt.get("instrument") in ['BELA', 'CALA']: # TODO split the orbit to check with a smaller msrm_sampl w/o breaking memory
            self.msrm_sampl = 15
        else:
            self.msrm_sampl = 4

        # store the imposed perturbation (if closed loop simulation)
        self.pert_cloop = {list(self.tracks.keys())[0]:gtracks[0].pert_cloop, list(self.tracks.keys())[1]:gtracks[1].pert_cloop}
        self.pert_cloop_0 = {list(self.tracks.keys())[0]:gtracks[0].pert_cloop_0, list(self.tracks.keys())[1]:gtracks[1].pert_cloop_0}
        # store the solution from the previous iteration
        self.sol_prev_iter = {list(self.tracks.keys())[0]:gtracks[0].sol_prev_iter,list(self.tracks.keys())[1]:gtracks[1].sol_prev_iter}
        # print(df.orbID)
        # print(df.orbID.unique())
        # print(self.ladata_df.orbID)

        if XovOpt.get("debug") and False:
            # prepare surface texture "stamp" and assign the interpolated function as class attribute
            np.random.seed(62)
            shape_text = 1024
            res_text = 1
            depth_text = 8
            size_stamp = 0.25
            noise = perlin2d.generate_periodic_fractal_noise_2d(35, (shape_text, shape_text), (res_text, res_text),
                                                                depth_text)
            interp_spline = RectBivariateSpline(np.array(range(shape_text)) / shape_text * size_stamp,
                                                np.array(range(shape_text)) / shape_text * size_stamp,
                                                noise)
            self.apply_texture = interp_spline

        if XovOpt.get("new_algo"):
            # self.ladata_df = self.ladata_df[['ET_TX', 'TOF', 'orbID', 'seqid', 'ET_BC', 'offnadir', 'LON', 'LAT', 'R', 'X_stgprj', 'Y_stgprj']]
            # rough only as prepro
            nxov = self.get_xov_prelim()
        else:
            # Crossover computation (rough+fine+elev)
            nxov = self.get_xov()

        # Depending on prOpt, ignore pairs where more than 1 xov has been found
        if XovOpt.get("multi_xov"):
            multi_xov_check = nxov > 0
        else:
            multi_xov_check = nxov == 1
        # print(nxov,multi_xov_check)

        if multi_xov_check:

            if not XovOpt.get("new_algo"):
                # Compute and store distances between obs and xov coord
                self.set_xov_obs_dist()
                # Compute and store offnadir state for obs around xov
                self.set_xov_offnadir()

                if XovOpt.get("partials"):
                    self.set_partials()

                # Remap track names to df
                self.xovtmp['orbA'] = self.xovtmp['orbA'].map({v: k for k, v in self.tracks.items()})
                self.xovtmp['orbB'] = self.xovtmp['orbB'].map({v: k for k, v in self.tracks.items()})

                # Update general df
                self.xovers = self.xovers.append(self.xovtmp, sort=True)
                self.xovers.reset_index(drop=True, inplace=True)
                self.xovers['xOvID'] = self.xovers.index
            else:
                # Remap track names to df
                trackmap = {v: k for k, v in self.tracks.items()}
                self.xovtmp['orbA'] = trackmap[self.xovtmp['orbA']]
                self.xovtmp['orbB'] = trackmap[self.xovtmp['orbB']]

                # print(self.xovtmp)
                # Update general df
                # self.xovers.append(self.xovtmp)
        return multi_xov_check

    def set_xov_offnadir(self):
        obslist = self.xovtmp[['cmb_idA','cmb_idB']].values.astype(int).tolist()
        obslist = lflatten(obslist)
        # get offnadir angle for xovering obs
        offnadir = self.ladata_df.loc[obslist, 'offnadir'].values
        # reshape for multi-xover tracks combination
        offnadir = np.reshape(offnadir,(2,-1))

        tmp = pd.DataFrame(np.array(offnadir)).T
        tmp.columns = ['offnad_A', 'offnad_B']
        self.xovtmp = pd.concat([self.xovtmp, tmp], axis=1)

    # ####@profile
    def combine(self, xov_list):

        # Only select elements with number of xovers > 0
        xov_list = [x for x in xov_list if len(x.xovers) > 0]

        # concatenate df and reindex
        # print([x.xovers for x in xov_list])
        if len(xov_list) > 0:
            self.xovers = pd.concat([x.xovers for x in xov_list], sort=True)
            # check for duplicate rows
            print("len xovers (pre duplicate search):", len(self.xovers))
            if XovOpt.get("instrument") == "BELA": # doesn't really make sense... useful to have working tests
                self.xovers = self.xovers.drop(columns=['xOvID', 'xovid'],errors='ignore').round(6).drop_duplicates()
            else:
                self.xovers = self.xovers.drop(columns=['xOvID', 'xovid'],errors='ignore').drop_duplicates()
            # .reset_index().rename(
                # columns={"index": "xOvID"})
            print("new len xovers (post duplicates):", len(self.xovers))
            # reset index to have a sequential one
            self.xovers = self.xovers.reset_index(drop=True)
            self.xovers['xOvID'] = self.xovers.index
            # print(self.xovers)postpro_xov_elev

            # Retrieve all orbits involved in xovers
            orb_unique = self.xovers['orbA'].tolist()
            orb_unique.extend(self.xovers['orbB'].tolist())
            self.tracks = list(set(orb_unique))
            # print(self.tracks)
            # exit()
            # print(orb_unique)

            if XovOpt.get("partials") == 1:
                for xovi, xov in enumerate(xov_list):
                    if hasattr(xov_list[xovi],'parOrb_xy'):
                        self.parOrb_xy = xov_list[xovi].parOrb_xy
                        self.parGlo_xy = xov_list[xovi].parGlo_xy
                        break
                    else:
                        print("### xov element ", xovi," is missing partials!!")
            # print(self.parOrb_xy, self.parGlo_xy)

    def save(self, filnam):
        pklfile = open(filnam, "wb")
        # clean ladata, which is now useless
        if hasattr(self, 'ladata_df'):
            del (self.ladata_df)
        if hasattr(self, 'gtracks'):
            del (self.gtracks)
        pickle.dump(self, pklfile, protocol=-1)
        pklfile.close()

    # ##@profile
    # load groundtrack from file
    def load(self, filnam):

        try:
            #print(filnam)
            pklfile = open(filnam, 'rb')
            self = pickle.load(pklfile)
            pklfile.close()
            # print('Xov loaded from '+filnam)

        except:
            pass
            # print("Loading "+filnam+" failed")
        # print(self.ladata_df)
        # print(self.MGRx.tck)

        return self

    # remove outliers using the median method
    def remove_outliers(self, data_col,remove_bad = True):

        xoverstmp = self.xovers.copy()
        total_occ_tracks = pd.DataFrame([xoverstmp['orbA'].value_counts(),xoverstmp['orbB'].value_counts()]).T.fillna(0).sum(axis=1).sort_values(ascending=False)

        orig_len = len(xoverstmp)

        # print("orig_len", orig_len)
        # print(self.xovers.loc[:,data_col].median(axis=0))
        sorted = np.sort(abs(xoverstmp.loc[:, data_col].values - xoverstmp.loc[:, data_col].median(axis=0)))
        # print(len(sorted))
        std_median = sorted[round(0.68 * len(sorted))]
        # sorted = sorted[sorted<3*std_median]

        # print("Mean, std of data:", xoverstmp.loc[:, data_col].mean(axis=0), xoverstmp.loc[:, data_col].std(axis=0))
        # print("Median, std_med of data:", xoverstmp.loc[:, data_col].median(axis=0), std_median)
        # print("Len, rms", len(xoverstmp.loc[:, data_col]), rms(xoverstmp.loc[:, data_col].values))

        bad_xov = xoverstmp[
            abs(xoverstmp.loc[:, data_col] - xoverstmp.loc[:, data_col].median(axis=0)) >= 3 * std_median]

        # print(bad_xov['orbA'].value_counts().sort_values())
        # print(bad_xov['orbB'].value_counts().sort_values())
        worst_tracks = pd.DataFrame([bad_xov['orbA'].value_counts(),bad_xov['orbB'].value_counts()]).T.fillna(0).sum(axis=1).sort_values(ascending=False)
        worst_tracks = pd.concat([worst_tracks,total_occ_tracks],axis=1, sort=True).fillna(0)
        worst_tracks.columns = ['bad','total']
        worst_tracks['percent'] = (worst_tracks.bad/worst_tracks.total*100.).sort_values(ascending=False)

        worst_tracks = worst_tracks[worst_tracks.percent>=50].sort_values(by='percent',ascending=False)
        if len(worst_tracks)>0:
            print("worst tracks:")
            print(worst_tracks)

        if remove_bad:
            print(len(xoverstmp[abs(
                xoverstmp.loc[:, data_col] - xoverstmp.loc[:, data_col].median(axis=0)) >= 3 * std_median]),
                  "xovers removed out of", orig_len)

            xoverstmp = xoverstmp[
                abs(xoverstmp.loc[:, data_col] - xoverstmp.loc[:, data_col].median(axis=0)) < 3 * std_median]
            # print(self.xovers)

        # print("Len, rms (post)", len(xoverstmp.loc[:, data_col]), rms(xoverstmp.loc[:, data_col].values))
        self.xovers = xoverstmp.copy()

        return self.xovers.loc[:, data_col].mean(axis=0), self.xovers.loc[:, data_col].std(axis=0), worst_tracks[worst_tracks.percent>=50].index

    # Compute elevation R at crossover points by interpolation
    # (should be put in a function and looped over -
    # also, check the higher order interp)
    ###@profile
    def get_elev(self, arg, ii, jj, ind_A, ind_B, par='', x=0, y=0):

        ladata_df = self.ladata_df
        msrm_sampl = self.msrm_sampl

        # print(ii, jj, ind_A, ind_B)
        try:
            ind_A_int = np.atleast_1d(ind_A.astype(int))
            ind_B_int = np.atleast_1d(ind_B.astype(int))
        except:
            # print("flattening lists")
            ii, jj, ind_A, ind_B = np.array(lflatten(ii)), np.array(lflatten(jj)), np.array(lflatten(ind_A)), np.array(
                lflatten(ind_B))
            ind_A_int = np.atleast_1d(ind_A.astype(int))
            ind_B_int = np.atleast_1d(ind_B.astype(int))

        # Prepare
        param = self.param
        if XovOpt.get("partials"):
            param.update(XovOpt.get("parOrb"))
            param.update(XovOpt.get("parGlo"))
            # define column for update

        if (XovOpt.get("debug")):
            print("get_elev", arg, ii, jj, ind_A, ind_B, par)
            # print(ladata_df.loc[ladata_df['orbID'] == arg[0]].iloc[np.round(ind_A)])
            # print(ladata_df.loc[ladata_df['orbID'] == arg[1]].iloc[np.round(ind_B)])
        # print(par, param[par.partition('_')[0]] )

        # Apply elevation correction (if computing partial derivative)
        if (bool(re.search('_?A$', par)) or bool(re.search('_[p,m]$', par))):  # is not ''):

            if bool(re.search('_?A$', par)):
                ETBCparpm = 'ET_BC_'+ par[:-1]
            elif bool(re.search('_[p,m]$', par)):
                ETBCparpm = 'ET_BC_'+ par

            if XovOpt.get("debug"):
                print(par.partition('_')[0], ind_A_int)
                print(ladata_df.columns)
                print(ladata_df.filter(
                    regex='^dR/' + par.partition('_')[0] + '$').columns)

            regex = re.compile(r'^dR/' + par.partition('_')[0] + '$')
            dRdp = list(filter(regex.search, ladata_df.columns))[0]
            # select columns (fastest way)
            index_cols = [ladata_df.columns.get_loc(col) for col in ['orbID',ETBCparpm, 'R', 'genID',dRdp]]
            # df_1 = self.ladata_df.iloc[:,index_cols].values
            ldA_ = ladata_df.values[:, index_cols]
            # ldA_ = ladata_df[['orbID',ETBCparpm, 'R', 'genID',dRdp]].values
            ldA_ = ldA_[ldA_[:,0]==0][:,1:]
            xyintA = [ldA_[max(0, k - msrm_sampl):min(k + msrm_sampl, ldA_.shape[0])].T for k in ind_A_int]

            if XovOpt.get("debug"):
                print("xyintA", ladata_df.loc[ladata_df['orbID'] == arg[0]][np.hstack([ETBCparpm, 'R', 'genID', ladata_df.filter(
                    regex='^dR/' + par.partition('_')[0] + '$').columns.values])])
                print([ladata_df.loc[ladata_df['orbID'] == arg[0]][np.hstack([ETBCparpm, 'R', 'genID', ladata_df.filter(
                    regex='^dR/' + par.partition('_')[0] + '$').columns.values])].loc[max(0, k - msrm_sampl):min(k + msrm_sampl, ldA_.shape[0])] for k in ind_A_int])
            # exit()
            t_ldA = [xyintA[k][0] - ldA_[ind_A_int[k], 0] for k in range(0, len(ind_A_int))]

            # print(xyintA[0][1], xyintA[0][2])
            diff_step = np.linalg.norm(param[par.partition('_')[0]])

            for k in range(len(ind_A_int)):

                if XovOpt.get("debug"):
                    print("check elevpart", k, param[par.partition('_')[0]], diff_step, xyintA[k][1], xyintA[k][3] * diff_step)

                if (bool(re.search('_pA?$', par))):
                    xyintA[k][1] += xyintA[k][3] * diff_step
                    # print(par, k, xyintA[k][1])
                elif (bool(re.search('_mA?$', par))):
                    xyintA[k][1] -= xyintA[k][3] * diff_step
                    # print(par, k, xyintA[k][1])

        else:

            index_cols = [ladata_df.columns.get_loc(col) for col in ['orbID','ET_BC', 'R', 'genID']]
            ldA_ = ladata_df.values[:, index_cols]
            # ldA_ = ladata_df[['orbID','ET_BC', 'R', 'genID']].values
            ldA_ = ldA_[ldA_[:,0]==0][:,1:]
            xyintA = [ldA_[max(0, k - msrm_sampl):min(k + msrm_sampl, ldA_.shape[0])].T for k in ind_A_int]
            t_ldA = [xyintA[k][0] - ldA_[ind_A_int[k], 0] for k in range(0, len(ind_A_int))]

        fA_interp = [interpolate.interp1d(x=t_ldA[k], y=xyintA[k][1], kind='cubic') for k in range(0, len(ind_A_int))]
        tA_interp = [interpolate.interp1d(x=xyintA[k][2], y=t_ldA[k], kind='linear') for k in range(0, len(ind_A_int))]

        R_A = [fA_interp[k](tA_interp[k](ind_A.item(k))) for k in range(0, ind_A.size)]
        # exit()

        if XovOpt.get("debug") and False:
            ldA2_ = ladata_df.loc[ladata_df['orbID'] == arg[0]][['X_stgprj', 'Y_stgprj', 'R']].values
            xyintA = [ldA2_[max(0, k - msrm_sampl):min(k + msrm_sampl, ladata_df.shape[0])].T for k in ind_A_int]
            zfun_smooth_rbf = interpolate.Rbf(xyintA[0][0], xyintA[0][1], xyintA[0][2], function='cubic',
                                              smooth=0)  # default smooth=0 for interpolation
            z_dense_smooth_rbf = zfun_smooth_rbf(x,
                                                 y)  # not really a function, but a callable class instance
            print('R_A_Rbf (supp more accurate)', z_dense_smooth_rbf)
            print('R_A_1d', R_A)

        # old way TODO
        # ldA_ = ladata_df.loc[ladata_df['orbID'] == arg[0]][['genID','R']].values
        # xyintA = [ldA_[max(0, k - msrm_sampl):min(k + msrm_sampl, ladata_df.shape[0])].T for k in ind_A_int]
        # f_interp = [interpolate.interp1d(xyintA[k][0], xyintA[k][1], kind='linear') for k in range(0, len(ind_A_int))]
        # R_A = [f_interp[k](ind_A.item(k)) for k in range(0, ind_A.size)]

        # alternative method TODO
        # ldA_ = ladata_df.loc[ladata_df['orbID'] == arg[0]][['X_stgprj', 'Y_stgprj','R']].values
        # #print(x,y,ldA_[ind_A_int,:2])
        # tck = interpolate.bisplrep(ldA_[:,0], ldA_[:,1], ldA_[:,2], s=0)
        # R_A = interpolate.bisplev(x, y, tck)
        # print("zA",R_A)

        # Apply elevation correction (if partial)
        if bool(re.search('_?B$', par)) or bool(re.search('_[p,m]$', par)):  # is not ''):

            if bool(re.search('_?B$', par)):
                ETBCparpm = 'ET_BC_'+ par[:-1]
            elif bool(re.search('_[p,m]$', par)):
                ETBCparpm = 'ET_BC_'+ par

            # TODO regex not needed twice
            regex = re.compile(r'^dR/' + par.partition('_')[0] + '$')
            dRdp = list(filter(regex.search, ladata_df.columns))[0]

            index_cols = [ladata_df.columns.get_loc(col) for col in ['orbID',ETBCparpm, 'R', 'genID',dRdp]]
            ldB_ = ladata_df.values[:, index_cols]
            # ldB_ = ladata_df[['orbID',ETBCparpm, 'R', 'genID',dRdp]].values
            ldB_ = ldB_[ldB_[:,0]==1][:,1:]

            xyintB = [ldB_[max(0, k - len(ldA_) - msrm_sampl):min(k - len(ldA_) + msrm_sampl, ldB_.shape[0])].T for
                      k in ind_B_int]
            t_ldB = [xyintB[k][0] - ldB_[ind_B_int[k] - len(ldA_), 0] for k in range(0, len(ind_B_int))]

            diff_step = np.linalg.norm(param[par.partition('_')[0]])
            # print('xyintB',self.tracks)
            # print(len(xyintB))
            # print(len(xyintB[0]))

            for k in range(len(ind_A_int)):

                if XovOpt.get("debug"):
                    print("check elevpart0", xyintA)
                    print("check elevpart", param[par.partition('_')[0]], diff_step, xyintA[k][2] * diff_step)

                if (bool(re.search('_pB?$', par))):
                    xyintB[k][1] += xyintB[k][3] * diff_step
                elif (bool(re.search('_mB?$', par))):
                    xyintB[k][1] -= xyintB[k][3] * diff_step

        else:
            index_cols = [ladata_df.columns.get_loc(col) for col in ['orbID','ET_BC', 'R', 'genID']]
            ldB_ = ladata_df.values[:, index_cols]
            # ldB_ = ladata_df[['orbID','ET_BC', 'R', 'genID']].values
            ldB_ = ldB_[ldB_[:,0]==1][:,1:]

            xyintB = [ldB_[max(0, k - len(ldA_) - msrm_sampl):min(k - len(ldA_) + msrm_sampl, ldB_.shape[0])].T for
                      k in ind_B_int]
            t_ldB = [xyintB[k][0] - ldB_[ind_B_int[k] - len(ldA_), 0] for k in range(0, len(ind_B_int))]

        fB_interp = [interpolate.interp1d(t_ldB[k], xyintB[k][1], kind='cubic') for k in range(0, len(ind_B_int))]
        tB_interp = [interpolate.interp1d(xyintB[k][2], t_ldB[k], kind='linear') for k in range(0, len(ind_B_int))]
        R_B = [fB_interp[k](tB_interp[k](ind_B.item(k))) for k in range(0, ind_B.size)]

        if XovOpt.get("debug") and False:
            ldB2_ = ladata_df.loc[ladata_df['orbID'] == arg[1]][['X_stgprj', 'Y_stgprj', 'R']].values
            zfun_smooth_rbf = interpolate.Rbf(ldB2_[:, 0], ldB2_[:, 1], ldB2_[:, 2], function='cubic',
                                              smooth=0)  # default smooth=0 for interpolation
            z_dense_smooth_rbf = zfun_smooth_rbf(x,
                                                 y)  # not really a function, but a callable class instance
            print('R_B_Rbf (supp more accurate)', z_dense_smooth_rbf)
            print('R_B_1d', R_B)

            print('dR',arg,[a-b for a in R_A for b in R_B])

        # if np.abs([a-b for a in R_A for b in R_B])>50:
            exit()

        if XovOpt.get("debug") and len(ind_B_int) == 1 and XovOpt.get("local"): # and False:
            # self.plot_xov_elev(arg, fA_interp[0], fB_interp[0], ind_A[0], ind_A_int[0], ind_B[0], ind_B_int[0],
            #                    ladata_df, ldA_, ldB_,
            #                    tA_interp[0], tB_interp[0], t_ldA[0], t_ldB[0], param=par)
            self.plot_xov_elev2(R_A, R_B, fA_interp, fB_interp, ind_A, ind_A_int, ind_B, ldA_, tA_interp, tB_interp, t_ldA,
                            t_ldB, xyintA, xyintB)
        # exit()

        # xyintA = [ldB_[max(0, k - len(ldA_) - msrm_sampl):min(k - len(ldA_) + msrm_sampl, ladata_df.shape[0])].T for k in ind_B_int]
        # f_interp = [interpolate.interp1d(xyintA[k][0], xyintA[k][1], kind='linear') for k in range(0, len(ind_B_int))]
        #
        # #ind_B = ind_B + np.modf(jj)[0]
        # R_B = [f_interp[k](ind_B.item(k)) for k in range(0, ind_B.size)]
        # print('R_B', R_B)
        #
        # #alternative method TODO
        # ldB_ = ladata_df.loc[ladata_df['orbID'] == arg[1]][['X_stgprj', 'Y_stgprj','R']].values
        # #print(x,y,ldB_[ind_B_int-len(ldA_),:2])
        # tck = interpolate.bisplrep(ldB_[:,0], ldB_[:,1], ldB_[:,2],s=0,kx=1,ky=1)
        # R_B = interpolate.bisplev(x, y, tck)
        # print("zB", R_B)

        if XovOpt.get("debug") and False:
            print(xyintA[0][0], np.array(xyintA[0][0]))
            print(ldA_)
            plt.plot(xyintA[0][0], xyintA[0][1], 'o', xyintA[0][0], fA_interp[0](xyintA[0][0]), '-')
            plt.savefig(XovOpt.get("tmpdir") + 'test_cub.png')
            plt.clf()
            plt.close()
        # exit()

        if XovOpt.get("debug"):
            print("Get elevation diff")
            print(ind_A, ind_B, R_A, R_B)
            # exit()

        return ind_A, ind_B, R_A, R_B

    def plot_xov_elev2(self, R_A, R_B, fA_interp, fB_interp, ind_A, ind_A_int, ind_B, ldA_, tA_interp, tB_interp, t_ldA,
                       t_ldB, xyintA, xyintB):

        diffA = ldA_[ind_A_int + 1][0, 1] - ldA_[ind_A_int][0, 1]
        if diffA > 50:
            print(diffA)
            tmp = np.array([[t_ldA[k], xyintA[k][1]] for k in range(0, 1)])[0].T
            # print(tmp)
            fig = plt.figure(figsize=(12, 8))
            plt.style.use('seaborn-poster')
            ax = fig.add_subplot(111)
            ax.plot(tmp[:, 0], tmp[:, 1], 'k.', label="dR #" + str(0))
            ax.plot(tA_interp[0](ind_A.item(0)), R_A[0], 'ro')
            xtmp = np.linspace(np.min(t_ldA[0]), np.max(t_ldA[0]), 10000)
            ax.plot(xtmp, fA_interp[0](xtmp), 'b--')

            tmp = np.array([[t_ldB[k], xyintB[k][1]] for k in range(0, 1)])[0].T
            ax.plot(tmp[:, 0], tmp[:, 1], 'k.', label="dR #" + str(0))
            ax.plot(tB_interp[0](ind_B.item(0)), R_B[0], 'ro')
            xtmp = np.linspace(np.min(t_ldB[0]), np.max(t_ldB[0]), 10000)
            ax.plot(xtmp, fB_interp[0](xtmp), 'b--')

            # ax.legend(loc="best")
            ax.set_xlim((-1, 1))
            ax.set_xlabel('ET_bounce (s)')
            ax.set_ylabel('elevation (m)')
            # title = "res R bias : " + str(rlm_results.params.round(1)[0]) + " m -- RMSE : " + str(
            #     rmspre.round(1)) + " m"  # / "+ str(rmspost.round(1)) + "m"
            # ax.set_title(title)
            plt.savefig(XovOpt.get("tmpdir") + 'test_elev_prof_.png')

            if R_B[0] - R_A[0] > 100:
                exit()

    def plot_xov_elev(self, arg, fA_interp, fB_interp, ind_A, ind_A_int, ind_B, ind_B_int, ladata_df, ldA_, ldB_,
                      tA_interp, tB_interp, t_ldA, t_ldB,param=''):

        print("plot_xov_elev START")
        print(t_ldA[0], t_ldA[len(t_ldA) - 1])

        tf_ldA = np.arange(t_ldA[0], t_ldA[len(t_ldA) - 1], 0.01)
        ldA_interp = fA_interp(tf_ldA)
        _ = np.squeeze(ind_A_int)
        print(len(t_ldA),len(ldA_[:, 1]))
        print(t_ldA)
        print(ldA_[:, 1])
        # plt.plot(t_ldA, ldA_[:, 1], 'o')
        plt.plot(tf_ldA, ldA_interp, '-')
        plt.plot(tA_interp(ind_A), fA_interp(tA_interp(ind_A)), '*r')
        tf_ldB = np.arange(t_ldB[0], t_ldB[len(t_ldB) - 1], 0.01)
        ldB_interp = fB_interp(tf_ldB)
        _ = np.squeeze(ind_B_int - len(ldA_))
        # plt.plot(t_ldB, ldB_[:, 1], 'o')
        plt.plot(tf_ldB, ldB_interp, '-')
        plt.plot(tB_interp(ind_B), fB_interp(tB_interp(ind_B)), '*k')
        # xyproj = ladata_df.loc[ladata_df['orbID'] == arg[1]][['X_stgprj', 'Y_stgprj']].values
        # tck, u = interpolate.splprep([xyproj[:, 0],
        #                               xyproj[:, 1]]
        #                              , s=0.0)
        # x_i, y_i = interpolate.splev(np.linspace(0, 1, (len(t_ldB) - 1) * 10), tck)
        # lontmp, lattmp = unproject_stereographic(x_i, y_i, 0, 90, 2440)
        # ttmp = np.linspace(t_ldB[0], t_ldB[len(t_ldB) - 1], (len(t_ldB) - 1) * 10)
        # np.savetxt('tmp/gmt.in', list(zip(lontmp, lattmp)))
        # # r_dem = np.loadtxt('tmp/gmt_' + self.name + '.out')
        # r_dem = subprocess.check_output(
        #     ['grdtrack', 'gmt.in', '-G'+auxdir+'HDEM_64.GRD'], #MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_4ppd_HgM008frame.GRD'],
        #     universal_newlines=True, cwd='tmp')
        # r_dem = np.fromstring(r_dem, sep=' ').reshape(-1, 3)[:, 2]
        # # texture_noise = self.apply_texture(np.mod(lattmp, 0.25), np.mod(lontmp, 0.25), grid=False)
        # # update Rmerc with r_dem/text
        # radius = r_dem  # + texture_noise
        # # print(list(zip(t_ldB, radius)))
        # # plt.plot(ttmp, radius, 'x-')
        # # exit()
        print(t_ldB)
        # plt.xlim([t_ldB[_], t_ldB[_]])
        plt.xlim([-0.5,0.5])
        plt.ylim([fB_interp(tB_interp(ind_B))-100,fB_interp(tB_interp(ind_B))+100])
        plt.savefig(XovOpt.get("tmpdir") + 'test_elev_prof_'+self.tracks[0]+'_'+self.tracks[1]+'_'+param+'.png')
        plt.clf()
        plt.close()
        print(self.tracks)
        # exit()

    ###@profile
    def get_xover_fine(self, rough_indA, rough_indB, param):
        """
        Fine-tune xover index and coordinates from first rough guess,
        using a limited number of neighboring observations
        :param rough_indA: index of xov.obsA in sub-df ladata[orbID=orbA], from rough guess
        :param rough_indB: index of xov.obsB in sub-df ladata[orbID=orbB], from rough guess
        :param param: perturbed parameter for partial computation
        :return: fine_indA: index of xov.obsA in sub-df ladata[orbID=orbA], from refined guess
        :return: fine_indB: index of xov.obsB in sub-df ladata[orbID=orbB], from refined guess
        :return: ind_A: xov.obsA index in ladata(orbA+orbB)
        :return: ind_B: xov.obsB index in ladata(orbA+orbB)
        """

        # orb_lst = list(self.tracks.values())

        msrm_sampl = self.msrm_sampl
        # ladata_df = self.ladata_df.copy()

        X_stgA = 'X_stgprj'
        Y_stgA = 'Y_stgprj'
        X_stgB = 'X_stgprj'
        Y_stgB = 'Y_stgprj'

        # when computing partials, update variable names to retrieve column in ladata
        if param != '':
            #msrm_sampl = self.msrm_sampl  # to reduce processing time, just check close to unperturbed xover
            # if orbit related parameter
            if any(key in param for key in ['dA_', 'dC_', 'dR_', 'dRl_', 'dPt_']):
                if ('pA' in param or 'mA' in param):
                    X_stgA = X_stgA + '_' + param[:-1]
                    Y_stgA = Y_stgA + '_' + param[:-1]
                    X_stgB = X_stgB
                    Y_stgB = Y_stgB
                else:
                    X_stgA = X_stgA
                    Y_stgA = Y_stgA
                    X_stgB = X_stgB + '_' + param[:-1]
                    Y_stgB = Y_stgB + '_' + param[:-1]
            # if geophysical/dynamic/global parameter
            else:
                X_stgA = X_stgA + '_' + param
                Y_stgA = Y_stgA + '_' + param
                X_stgB = X_stgB + '_' + param
                Y_stgB = Y_stgB + '_' + param

        if (XovOpt.get("debug")):
            print("Check parts X_stg")
            print(X_stgA, \
                  Y_stgA, \
                  X_stgB, \
                  Y_stgB)

            # pd.set_option('display.max_columns', 500)
            # df = ladata_df.loc[ladata_df['orbID'] == orb_lst[0]]
            # print(df)
            # exit()

        # compute more accurate location
        # Retrieve ladata_df index of observations involved in the crossover
        # (can be used to extract orbit number with join btw ladata_df and xovers_df -
        # eg, (ladata_df.loc[ind0][['orbID']].values).reshape(1,-1) -
        # the orbit number can then be used to get the value at i and j by interpolation)
        # ind0 and ind1 now are the indeces of the points just before the
        # intersection in ladata_df, so that (ind0,ind0+1) and (ind1,ind1+1) are the
        # bracketing points' indeces
        if set([X_stgA, Y_stgA, X_stgB, Y_stgB]).issubset(self.ladata_df.columns): # if partials
            if X_stgA != X_stgB and Y_stgA != Y_stgB:
                # df_ = self.ladata_df[['orbID', X_stgA, Y_stgA, X_stgB, Y_stgB]].values
                index_cols = [self.ladata_df.columns.get_loc(col) for col in ['orbID', X_stgA, Y_stgA, X_stgB, Y_stgB]]
                # df_1 = self.ladata_df.iloc[:,index_cols].values
                df_ = self.ladata_df.values[:,index_cols]
                ldA_ = df_[df_[:, 0] == 0][:,1:]
                ldB_ = df_[df_[:, 0] == 1][:,3:]
            else: # if no partials
                # print(self.ladata_df.columns)
                # print([self.ladata_df.columns.get_loc(col) for col in ['orbID', X_stgA, Y_stgA]])
                # print(len(self.ladata_df))
                # df_ = self.ladata_df[['orbID', X_stgA, Y_stgA]].values
                index_cols = [self.ladata_df.columns.get_loc(col) for col in ['orbID', X_stgA, Y_stgA]]
                # df_1 = self.ladata_df.iloc[:,index_cols].values
                df_ = self.ladata_df.values[:,index_cols]
                # exit()

                # df_ = self.ladata_df[['orbID', X_stgA, Y_stgA]].values
                ldA_ = df_[df_[:, 0] == 0][:,1:]
                ldB_ = df_[df_[:, 0] == 1][:,1:]

            if XovOpt.get("debug"):
                print("Fine", param,self.proj_center)
                print(rough_indA, rough_indB)
                print(ldA_[rough_indA[0]],ldB_[rough_indB[0]- len(ldA_)])

            intersec_out = [
                intersection(ldA_[max(0, rough_indA[k] - msrm_sampl):min(rough_indA[k] + msrm_sampl, len(ldA_)), 0],
                             ldA_[max(0, rough_indA[k] - msrm_sampl):min(rough_indA[k] + msrm_sampl, len(ldA_)), 1],
                             ldB_[
                             max(0, rough_indB[k] - len(ldA_) - msrm_sampl):min(rough_indB[k] - len(ldA_) + msrm_sampl,
                                                                                len(ldB_)), 0],
                             ldB_[
                             max(0, rough_indB[k] - len(ldA_) - msrm_sampl):min(rough_indB[k] - len(ldA_) + msrm_sampl,
                                                                                len(ldB_)), 1])
                for k in range(len(rough_indA))]

            if XovOpt.get("debug"):
                print("intersection")
                print(len(rough_indA), len(intersec_out[0][0]),intersec_out[0][0] )
                print(intersec_out)

            if len(rough_indA) > 1:
                # print("len>1", len(rough_indA))
                # print(intersec_out)
                x, y, a, b, rough_tmpA, rough_tmpB = [], [], [], [], [], []
                for idx, r in enumerate(intersec_out):
                    rough_tmpA.extend(np.repeat(rough_indA[idx], len(r[0])))
                    rough_tmpB.extend(np.repeat(rough_indB[idx], len(r[0])))
                    x.extend(r[0])
                    y.extend(r[1])
                    a.extend(r[2])
                    b.extend(r[3])

                # print(x,y,a,b,rough_tmpA,rough_tmpB)
                idx_dup = [idx for idx, item in enumerate(x) if item in x[:idx]]
                # print(idx_dup)
                if len(idx_dup) > 0 and XovOpt.get("debug"):
                    print("*** Duplicate obs eliminated for ", self.tracks, "on computing ", param)

                rough_indA = np.delete(rough_tmpA, idx_dup)
                rough_indB = np.delete(rough_tmpB, idx_dup)
                tmp = [np.delete(l, idx_dup) for l in [x, y, a, b]]
                intersec_out = []
                for i in range(len(tmp[0])):
                    intersec_out.append([x[i] for x in tmp])

            elif len(rough_indA) == 1 and len(intersec_out[0][0]) > 1:
                intersec_out = np.transpose(intersec_out)

        else:
            intersec_out = []
            print("*** xov_setup.get_xover_fine: No ", X_stgA, Y_stgA, X_stgB, Y_stgB, " in df for ", self.tracks)

        intersec_out = np.reshape(intersec_out, (-1, 4))

        if intersec_out.size != 0 and intersec_out != []:

            intersec_x, intersec_y = intersec_out[:, 0], intersec_out[:, 1]
            fine_indA, fine_indB = intersec_out[:, 2], intersec_out[:, 3]

            # print(fine_indA, fine_indB)

            intersec_ind = intersec_out[:, 2:]
            rough_ind = np.array([rough_indA, rough_indB]).T

            # index in local ladata_df is given by the position in subsample
            # minus size of subsample plus the position of the rough guess (center of subsample)
            tmp = rough_ind - msrm_sampl
            ld_ind_A = np.where(tmp[:, 0] > 0, (tmp + intersec_ind)[:, 0], intersec_ind[:, 0])
            ld_ind_B = np.where(tmp[:, 1] - len(ldA_) > 0, (tmp + intersec_ind)[:, 1], len(ldA_) + intersec_ind[:, 1])

            # print(intersec_x)

            # plot and check intersections (rough, fine, ...)
            # if (debug):
            if XovOpt.get("debug") and len(intersec_x) > 0 and param == '':
                print("intersec_x, intersec_y",intersec_x, intersec_y)
                self.plot_xov_curves(ldA_, ldB_, intersec_x, intersec_y, rough_indA, rough_indB, param)
                # exit()

            isarray = all(intersec_x)
            if not isarray:
                ld_ind_A = [np.squeeze(x) for x in ld_ind_A[intersec_x > 0]][0]
                ld_ind_B = [np.squeeze(x) for x in ld_ind_B[intersec_x > 0]][0]
                # return lflatten(intersec_x[intersec_x > 0]), lflatten(intersec_y[intersec_x > 0]), lflatten(
                #     fine_indA[intersec_x > 0]), lflatten(fine_indB[
                #                                                  intersec_x > 0]), ld_ind_A, ld_ind_B
                return lflatten(intersec_x), lflatten(intersec_y), lflatten(
                    fine_indA), lflatten(fine_indB), ld_ind_A, ld_ind_B
            else:
                # return intersec_x[intersec_x > 0], intersec_y[intersec_x > 0], fine_indA[intersec_x > 0], fine_indB[
                #     intersec_x > 0], ld_ind_A[intersec_x > 0], ld_ind_B[intersec_x > 0]
                return intersec_x, intersec_y, fine_indA, fine_indB, ld_ind_A, ld_ind_B
        else:
            return [], [], [], [], [], []

    def plot_xov_curves(self, curve_A, curve_B, intersec_x, intersec_y, rough_A=0, rough_B=0, param = ''):

        if XovOpt.get("debug"):
            print('ladata_ind', curve_A, curve_B)
            print('XYproj', intersec_x, intersec_y)
            print('lonlat', unproject_stereographic(intersec_x, intersec_y, 0, 90, 2440))

        fig, ax = plt.subplots()
        ax.plot(curve_A[:, 0], curve_A[:, 1], 'x-r')
        ax.plot(curve_B[:, 0], curve_B[:, 1], 'x-')

        if all(rough_A != 0):
            xr = curve_A[rough_A, 0]
            yr = curve_A[rough_A, 1]
            ax.plot(xr, yr, '*g', label='rough xov A')
            xr = curve_B[rough_B - len(curve_A), 0]
            yr = curve_B[rough_B - len(curve_A), 1]
            ax.plot(xr, yr, '*r', label='rough xov B')

        ax.plot(intersec_x, intersec_y, '*k', label='fine xov')

        tmp = dict([v, k] for k, v in self.tracks.items())
        ax.set_title('Xover detected for tracks' + tmp[0] + '-' + tmp[1])
        ax.set_xlabel('x (distance from NP, km)')
        ax.set_ylabel('y (distance from NP, km)')
        ax.legend()

        delta = 1.
        if abs(np.amin(np.absolute(intersec_x))) > delta:
            xmin = np.amin(np.hstack(intersec_x)) - delta
            xmax = np.amax(np.hstack(intersec_x)) + delta
        else:
            xmax = 2
            xmin = -2
        plt.xlim(xmin, xmax)
        if abs(np.amin(np.absolute(intersec_y))) > delta:
            ymin = np.amin(np.array(intersec_y)) - delta
            ymax = np.amax(np.array(intersec_y)) + delta
        else:
            ymax = 2
            ymin = -2
        plt.ylim(ymin, ymax)

        plt.savefig(XovOpt.get("tmpdir")+'intersect_' + tmp[0] + '_' + tmp[1] + '_' + param + '.png')
        plt.clf()
        plt.close()
        # exit()

    # For each combination of 2 orbits, detect crossovers in 2 steps (rough, fine),
    # then get elevation for each point in the crossovers (by interpolating)
    def get_xOver_elev(self, arg):

        ladata_df = self.ladata_df
        msrm_sampl = self.msrm_sampl

        ind_A, ind_B, x, y = self.get_xover_rough(arg, ladata_df, msrm_sampl)

        # reassign index to full list (from down-sampled index)
        ind_A *= msrm_sampl
        ind_B *= msrm_sampl

        if len(x) > 0:
            # Retrieve ladata_df index of observations involved in the crossover
            # (can be used to extract orbit number with join btw ladata_df and xovers_df -
            # eg, (ladata_df.loc[ind0][['orbID']].values).reshape(1,-1) -
            # the orbit number can then be used to get the value at ind_A and ind_B by interpolation)
            # ind0 and ind1 now are the indeces of the points just before the
            # intersection in ladata_df, so that (ind0,ind0+1) and (ind1,ind1+1) are the
            # bracketing points' indeces
            rough_indA = ladata_df.loc[ladata_df['orbID'] == arg[0]].iloc[ind_A].index.values
            rough_indB = ladata_df.loc[ladata_df['orbID'] == arg[1]].iloc[ind_B].index.values

            # Check if any 2 xovers closer than msrm_sampl, if yes, remove one of them
            # (not doing this would result in a doubling of the xovers)
            f = np.insert((np.diff(rough_indA) > msrm_sampl) & (np.diff(rough_indB) > msrm_sampl), 0, 'True')
            # x = x[f]
            # y = y[f]
            rough_indA = rough_indA[f]
            rough_indB = rough_indB[f]

            # Recompute stereogr projection around average LON/LAT of track
            # (else low latitude xovers badly determined)
            self.xov_project(ladata_df, msrm_sampl, rough_indA)

            # try:
            x, y, subldA, subldB, ldA, ldB = self.get_xover_fine(rough_indA, rough_indB, '')
            if len(x) > 0:
                if XovOpt.get("debug"):
                    print("pre-elev (no partial)")
                    print(subldA, subldB, ldA, ldB)

                ldA, ldB, R_A, R_B = self.get_elev(arg, subldA, subldB, ldA, ldB, x=x, y=y)

                if XovOpt.get("debug"):
                    print(arg, x, y, ldA, ldB, R_A, R_B)
                    print(arg, x, y, R_A, R_B)
                # exit()
                return np.vstack((x, y, ldA, ldB, R_A, R_B)).T
            # except:
    ###@profile
    def xov_project(self, ladata_df, msrm_sampl, rough_indA):
        # compute central lon/lat of trackA (then use it for both, as close to intersection)
        df_ = ladata_df.loc[ladata_df['orbID'] == list(self.tracks.values())[0]][['LON', 'LAT']].values
        lon_mean_A = df_[max(0, rough_indA[0] - msrm_sampl):min(rough_indA[0] + msrm_sampl, len(df_)), 0].mean()
        lat_mean_A = df_[max(0, rough_indA[0] - msrm_sampl):min(rough_indA[0] + msrm_sampl, len(df_)), 1].mean()
        self.proj_center = {'lon': lon_mean_A, 'lat': lat_mean_A}
        # reproject both A and B intersecting tracks and replace ladata_df (also updates partials)
        # should be out of the if in case self.ladata is not updated
        tmp = [obj.project(self.proj_center['lon'], self.proj_center['lat'], inplace=False) for obj in
               self.gtracks]
        ladata_df = pd.concat([tmp[0], tmp[1]],sort=True).reset_index(drop=True)
        # clean up useless columns (speeds up pd.df indexing)
        self.ladata_df = self.ladata_df.drop(self.ladata_df.filter(regex='^dL(AT|ON)/d.*').columns,
                                             axis='columns')
        # updating self.ladata to keep the updated projection for partials processing
        self.ladata_df.update(ladata_df)
        self.ladata_df['orbID'] = self.ladata_df['orbID'].map(self.tracks)

    ##@profile
    def get_xover_rough(self, arg, ladata_df, msrm_sampl):
        # Decimate data and find rough intersection
        x, y, ind_A, ind_B = intersection(
            ladata_df.loc[ladata_df['orbID'] == arg[0]]['X_stgprj'].values[::msrm_sampl],
            ladata_df.loc[ladata_df['orbID'] == arg[0]]['Y_stgprj'].values[::msrm_sampl],
            ladata_df.loc[ladata_df['orbID'] == arg[1]]['X_stgprj'].values[::msrm_sampl],
            ladata_df.loc[ladata_df['orbID'] == arg[1]]['Y_stgprj'].values[::msrm_sampl])

        # plots
        if XovOpt.get("debug"):
            import geopandas as gpd
            print(f"rough intersection")
            print("## Check if correct crs is used for projections")
            if XovOpt.get("instrument") == 'LRO':
                crs_lonlat = "+proj=lonlat +units=m +a=1737.4e3 +b=1737.4e3 +no_defs"
                crs_stereo_km = '+proj=stere +lat_0=-90 +lon_0=0 +lat_ts=-90 +k=1 +x_0=0 +y_0=0 +units=km +a=1737.4e3 +b=1737.4e3 +no_defs'
            elif XovOpt.get("instrument") in ['MLA','BELA']:
                crs_lonlat = "+proj=lonlat +units=m +a=2440.e3 +b=2440.e3 +no_defs"
                crs_stereo_km = '+proj=stere +lat_0=90 +lon_0=0 +lat_ts=90 +k=1 +x_0=0 +y_0=0 +units=km +a=2440.e3 +b=2440.e3 +no_defs'

            print(x, y, ind_A, ind_B)
            df0 = ladata_df.loc[ladata_df['orbID'] == arg[0]]#.values[::msrm_sampl]
            gdf0= gpd.GeoDataFrame(
                df0, geometry=gpd.points_from_xy(df0.LON, df0.LAT),crs=crs_lonlat)
            print(gdf0[['X_stgprj','Y_stgprj']])
            print(gdf0.to_crs(crs_stereo_km))
            df1 = ladata_df.loc[ladata_df['orbID'] == arg[1]]#.values[::msrm_sampl]
            gdf1 = gpd.GeoDataFrame(
                df1, geometry=gpd.points_from_xy(df1.LON, df1.LAT),crs=crs_lonlat)
            print(gdf1[['X_stgprj','Y_stgprj']])
            print(gdf1.to_crs(crs_stereo_km))
            ax = plt.subplot()
            gdf0.to_crs(crs_stereo_km).plot(ax=ax) #, label=gdf0.orbID[0])  # , color='red')
            gdf1.to_crs(crs_stereo_km).plot(ax=ax) #, label=gdf1.orbID[0])  # , color='red')
            # plt.xlim(-40, 40)
            # plt.ylim(-40, 40)
            # plt.legend()
            plt.show()
            # exit()
        return ind_A, ind_B, x, y

    ###@profile
    def get_xov(self):
        """
        Read ladata_df and compute all xovers, then updates xovers dataframe
        :return: number of detected xovers
        """
        ladata_df = self.ladata_df  # adapt

        # Get intersections between all orbits (x of xover,y of xover,
        # i=index in first orbit,j=index in second orbit)

        # Copy index to column for book-keeping of crossovers
        ladata_df['genID'] = ladata_df.index

        # Call sequence only
        # Compute crossover position (2 steps procedure: first roughly locate,
        # downsampling data to 'msrm_sampl', then with full sampling around the
        # points located with the first pass). Use either the seq or parallel
        # version.

        # if(parallel):
        #  #print((mp.cpu_count() - 1))
        #  pool = mp.Pool(processes = mp.cpu_count() - 1)
        #  results = pool.map(self.get_xOver_elev, comb)  # parallel
        # else:
        results = self.get_xOver_elev(list(self.tracks.values()))  # seq

        if results is not None:
            # print(results)
            xovtmp = self.postpro_xov_elev(ladata_df, results)
            if len(results)==1:
                xovtmp = pd.DataFrame(xovtmp,index=[0]) # very slow and should be avoided if only 1 line

            # print(xovtmp)
            # exit()

            # Update xovtmp as attribute for partials
            self.xovtmp = xovtmp

            if XovOpt.get("debug"):
                print(str(len(xovtmp)) + " xovers found btw " + list(self.tracks.keys())[0] + " and " + list(self.tracks.keys())[1])

            return len(xovtmp)

        else:
            if XovOpt.get("debug"):
                if len(self.tracks):
                    tmp = dict([v, k] for k, v in self.tracks.items())
                    print("no xovers btw " + tmp[0] + " and " + tmp[1])

            return -1  # 0 xovers found

    ##@profile
    def get_xov_prelim(self):
        """
        Read ladata_df and compute all xovers, then updates xovers dataframe
        :return: number of detected xovers
        """
        ladata_df = self.ladata_df  # adapt

        # Get intersections between all orbits (x of xover,y of xover,
        # i=index in first orbit,j=index in second orbit)

        # Copy index to column for book-keeping of crossovers
        ladata_df['genID'] = ladata_df.index

        # Call sequence only
        # Compute crossover position (2 steps procedure: first roughly locate,
        # downsampling data to 'msrm_sampl', then with full sampling around the
        # points located with the first pass). Use either the seq or parallel
        # version.

        # if(parallel):
        #  #print((mp.cpu_count() - 1))
        #  pool = mp.Pool(processes = mp.cpu_count() - 1)
        #  results = pool.map(self.get_xOver_elev, comb)  # parallel
        # else:
        # results = self.get_xOver_elev(list(self.tracks.values()))  # seq

        arg = list(self.tracks.values())
        # ladata_df = self.ladata_df
        msrm_sampl = self.msrm_sampl

        ind_A, ind_B, x, y = self.get_xover_rough(arg, ladata_df, msrm_sampl)

        # reassign index to full list (from down-sampled index)
        ind_A *= msrm_sampl
        ind_B *= msrm_sampl

        if len(x) == 1:
            # Retrieve ladata_df index of observations involved in the crossover
            # (can be used to extract orbit number with join btw ladata_df and xovers_df -
            # eg, (ladata_df.loc[ind0][['orbID']].values).reshape(1,-1) -
            # the orbit number can then be used to get the value at ind_A and ind_B by interpolation)
            # ind0 and ind1 now are the indeces of the points just before the
            # intersection in ladata_df, so that (ind0,ind0+1) and (ind1,ind1+1) are the
            # bracketing points' indeces
            rough_indA = ladata_df.loc[ladata_df['orbID'] == arg[0]].iloc[ind_A].index.values
            rough_indB = ladata_df.loc[ladata_df['orbID'] == arg[1]].iloc[ind_B].index.values

            results = [[x[0] for x in [x,y,rough_indA,rough_indB,np.zeros(len(x)),np.zeros(len(x))]]]
        else:
            results = None

        if results is not None:
            xovtmp = self.postpro_xov_elev(ladata_df, results)

            # Update xovtmp as attribute for partials
            self.xovtmp = xovtmp
            # by default, just taking single xovers (will be an issue with other probes...)
            nxov = 1 # np.max([len([x]) for x in xovtmp])

            if XovOpt.get("debug"):
                print(str(len(xovtmp)) + " xovers found btw " + list(self.tracks.keys())[0] + " and " + list(self.tracks.keys())[1])

            return nxov

        else:
            if XovOpt.get("debug"):
                if len(self.tracks):
                    tmp = dict([v, k] for k, v in self.tracks.items())
                    print("no xovers btw " + tmp[0] + " and " + tmp[1])

            return -1  # 0 xovers found

    # #@profile
    def postpro_xov_elev(self, ladata_df, results):

        xovtmp = np.vstack([x for x in results if x is not None]).reshape(-1, np.shape(results)[1])

        # Store seqid of mla obs related to xover (only if single xov between orbits TODO: extend if interested)
        if len(xovtmp) == 1:

            xovtmp = dict(zip(['x0', 'y0', 'cmb_idA', 'cmb_idB', 'R_A', 'R_B'],[x[0] for x in xovtmp.T]))
            # print(xovtmp)
            # exit()

            # Get discrepancies (dR = R(obs1) - R(obs2)) at crossover time (where ideal dR=0)
            have_elev = (xovtmp['R_B'] == 0.) + (xovtmp['R_A'] == 0.)
            # print(xovtmp)
            if have_elev:
                xovtmp['dR'] = 0
            else:
                xovtmp['dR'] = xovtmp['R_B'] - xovtmp['R_A']

            # Store reference orbit IDs for each xov (it was actually just storing 0 for orbA and 1 for orbB, mapped later to orbit - and it was very slow)
            xovtmp['orbA'] = 0
            xovtmp['orbB'] = 1
            # this should always be 0 ...
            xovtmp['xOvID'] = 0
            # retrieve index of obs close to xover (for fine search)
            index_cols = [ladata_df.columns.get_loc(col) for col in ['genID', 'seqid']]
            ladata_np = ladata_df.values[:,index_cols]

            xovtmp['mla_idA'] = ladata_np[ladata_np[:,0]==np.round(xovtmp['cmb_idA'],0),1].astype('int')[0]
            xovtmp['mla_idB'] = ladata_np[ladata_np[:,0]==np.round(xovtmp['cmb_idB'],0),1].astype('int')[0]

            # xovtmp = pd.DataFrame(xovtmp, index=[0])
            # print(xovtmp)
            # exit()

        elif False: # update to use for multiple xovs (e.g., Bepi)

            xovtmp.columns = ['x0', 'y0', 'cmb_idA', 'cmb_idB', 'R_A', 'R_B']

            # Get discrepancies (dR = R(obs1) - R(obs2)) at crossover time (where ideal dR=0)
            xovtmp['dR'] = xovtmp.R_B - xovtmp.R_A

            # Store reference orbit IDs for each xov (it was actually just storing 0 for orbA and 1 for orbB, mapped later to orbit - and it was very slow)
            xovtmp['orbA'] = \
                ladata_df.loc[ladata_df['orbID'] == list(self.tracks.values())[0]].loc[
                    map(round, xovtmp.cmb_idA.values)][
                    ['orbID']].values
            xovtmp['orbB'] = \
                ladata_df.loc[ladata_df['orbID'] == list(self.tracks.values())[1]].loc[
                    map(round, xovtmp.cmb_idB.values)][
                    ['orbID']].values
            xovtmp['xOvID'] = xovtmp.index
            xovtmp['mla_idA'] = ladata_df.loc[ladata_df['genID'].isin(xovtmp.cmb_idA.values)]['seqid'].values
            xovtmp['mla_idB'] = ladata_df.loc[ladata_df['genID'].isin(xovtmp.cmb_idB.values)]['seqid'].values

        # returns dict, not df (might cause errors to correct)
        return xovtmp

    def set_xov_obs_dist(self):
        """
        Get distance of measurements around xover point

        :rtype: pandas df
        :param xovtmp: x-over dataframe
        :return: updated x-over dataframe with distance of xover from the 4 altimetry bounce points used to locate it
        """
        from xovutil.project_coord import project_stereographic

        xovtmp = self.xovtmp.copy()
        xovtmp.reset_index(drop=True, inplace=True)

        # get coordinates of neighbouring measurements in local projection
        msrmnt_crd = []

        #for trk in ['A', 'B']:
        obslist = xovtmp[['cmb_idA','cmb_idB']].values.astype(int).tolist()
        obslist = lflatten(obslist)
        # print(obslist)

        lonlat_bef = self.ladata_df[['LON', 'LAT']].loc[obslist].to_dict('records')
        lonlat_aft = self.ladata_df[['LON', 'LAT']].loc[[l + 1 for l in obslist]].to_dict('records')
        # print(lonlat_bef)
        # print(lonlat_aft)
        for obs in [lonlat_bef,lonlat_aft]:
            for x in range(len(obs)):
                # print(obs[x])
                msrmnt_crd.extend(project_stereographic(obs[x]['LON'],obs[x]['LAT'],
                                            self.proj_center['lon'],
                                            self.proj_center['lat'],
                                            self.vecopts['PLANETRADIUS']))

            # msrmnt_crd.extend(
            #     self.ladata_df.loc[obslist][['X_stgprj', 'Y_stgprj']].values)
            # msrmnt_crd.extend(
            #     self.ladata_df.loc[[l + 1 for l in obslist]][['X_stgprj', 'Y_stgprj']].values)

        msrmnt_crd = np.reshape(msrmnt_crd, (-1, 2))
        # print(msrmnt_crd)

        # get compatible array with xov coordinates
        xov_crd = xovtmp[['x0', 'y0']].values
        xov_crd = np.reshape(np.tile(xov_crd.ravel(), 4), (-1, 2))

        # print(xov_crd)
        # print(msrmnt_crd)
        # print(xov_crd)
        # print(np.linalg.norm(msrmnt_crd - xov_crd, axis=1))
        # print(np.reshape(np.linalg.norm(msrmnt_crd - xov_crd, axis=1),(-1,len(xovtmp))))
        # print(np.reshape(np.linalg.norm(msrmnt_crd - xov_crd, axis=1),(-1,len(xovtmp))).T)
        # exit()

        # compute distance and add to input df
        dist = np.reshape(np.linalg.norm(msrmnt_crd - xov_crd, axis=1), (-1, len(xovtmp))).T

        # print(dist)

        df_ = pd.DataFrame(dist,
                           columns=['dist_Am', 'dist_Ap', 'dist_Bm', 'dist_Bp'])
        # print(df_)
        # print(xovtmp)
        # print(pd.concat([xovtmp, df_], axis=1))
        # exit()

        if all(~df_[c].hasnans for c in df_) and (
                df_[['dist_Am', 'dist_Ap', 'dist_Bm', 'dist_Bp']].max() < 10000).all():
            self.xovtmp = pd.concat([xovtmp, df_], axis=1)
        else:
            print("*** error: NaN or unexpected value in set_xov_obs_dist")
            print(pd.concat([xovtmp, df_], axis=1))
            # exit(2)

    def get_xov_latlon(self, trackA):
        """
        Retrieve LAT/LON of xover from geolocalisation table and add to xovers table
        (useful to analyze/plot xovers on map). Updates input xov.xovers.

        :param trackA: gtrack containing ladata table
        """
        # TODO could also average trackB idB for more accuracy (or call this function
	    # for trackB too)
        idx = self.xovers.cmb_idA.round()
        tmp0 = trackA.ladata_df.iloc[idx][['LON', 'LAT']].reset_index(drop=True)
        idx[idx >= len(trackA.ladata_df)-1] = len(trackA.ladata_df)-2
        tmp1 = trackA.ladata_df.iloc[idx + 1][['LON', 'LAT']].reset_index(drop=True)
        tmp = pd.concat([tmp0, tmp1], axis=1)
        tmp = tmp.groupby(by=tmp.columns, axis=1).mean()

        self.xovers = pd.concat([self.xovers, tmp], axis=1)

    ###@profile
    def set_partials(self):
        # Compute crossover position for partials at + and -
        # and combine them for each observation and parameter

        startXovPart = time.time()

        ladata_df = self.ladata_df
        xovers_df = self.xovtmp

        # Prepare
        param = self.param
        param.update(XovOpt.get("parOrb"))
        param.update(XovOpt.get("parGlo"))

        # print(ladata_df.loc[ladata_df['orbID'] == self.tracks[0]].loc[map(round, xovers_df.cmb_idA.values)][['X_stgprj','Y_stgprj','X_stgprj_dA_p','Y_stgprj_dA_p','X_stgprj_dA_m','Y_stgprj_dA_m']])
        # print(ladata_df.loc[ladata_df['orbID'] == self.tracks[1]].loc[map(round, xovers_df.cmb_idB.values)][['X_stgprj','Y_stgprj','X_stgprj_dA_p','Y_stgprj_dA_p','X_stgprj_dA_m','Y_stgprj_dA_m']])

        # Update combination array for all orbits involved in actual xOvers
        # self.comb1 = np.hstack((ladata_df.loc[ladata_df['orbID'] == self.tracks[0]].loc[
        #                             map(round, xovers_df.cmb_idA.values)][['orbID']].values,
        #                         ladata_df.loc[ladata_df['orbID'] == self.tracks[1]].loc[
        #                             map(round, xovers_df.cmb_idB.values)][['orbID']].values))

        # Compute fine location of xOvers for perturbed observations and relative elevations
        par_suffix = [(a + b) for a in list(XovOpt.get("parOrb").keys()) for b in ['_pA', '_mA', '_pB', '_mB']]
        par_suffix.extend([(a + b) for a in list(XovOpt.get("parGlo").keys()) for b in ['_p', '_m']])

        # out_elev contains per param Pi and xover xi, array[xi.RA.DPi+A,xi.RB.DPi+A] for all xi,
        # array[xi.RA.DPi-A,xi.RB.DPi-A] for all xi,
        # array[xi.RA.DPi+B,xi.RB.DPi+B] for all xi,
        # array[xi.RA.DPi-B,xi.RB.DPi-B] for all xi,
        # ... other parameters
        out_elev = []

        results = [self.get_partials(l) for l in par_suffix]  # seq
        # out_elev.append([x for x in results if x is not None])
        out_elev.append(results)

        # Setup pandas containing all plus/minus dR_A/B and differentiate: dR/dp_A = ((R_B - R_A)_Aplus - (R_B - R_A)_Aminus)/2*diffstep
        parOrb_xy = [(a + b + c) for a in ['dR/'] for b in list(XovOpt.get("parOrb").keys()) for c in list(['_A', '_B'])]
        parGlo_xy = [(a + b) for a in ['dR/'] for b in list(XovOpt.get("parGlo").keys())]
        par_xy = parOrb_xy + parGlo_xy

        # parxy_df=pd.DataFrame(np.diff(np.diff(np.hstack(out_elev))[:,::2])[:,::2]/list(param.values())[1:],columns=par_xy)
        if XovOpt.get("debug"):
            print("par_xy")
            print(par_xy)
            print(out_elev)
            # exit()

        max_xov_part = np.max([len(i) for i in out_elev[0]])
        nxov_part_all_equal = len(set([len(i) for i in out_elev[0]])) == 1
        no_nan_part = [i!=None for i in out_elev]
        # print(nan_part)
        # print([i!=None for i in out_elev])
        # print(out_elev)

        if len(xovers_df.index) == max_xov_part and nxov_part_all_equal and no_nan_part:

            if len(xovers_df.index) > 1:
                # print("len(xovers_df.index) > 1 ")
                # print(out_elev)
                DelR = (np.diff(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[:, ::2])[:, ::2])[:len(xovers_df)]

                DelR_orb = np.array(DelR)[:, :2 * len(XovOpt.get("parOrb"))]
                DelR_orb /= (2. * np.repeat(list(param.values())[1:len(XovOpt.get("parOrb")) + 1], 2))
                DelR_glo = np.array(DelR)[:, 2 * len(XovOpt.get("parOrb")):]
                DelR_glo /= (2. * np.array([np.linalg.norm(x) for x in list(param.values())][len(XovOpt.get("parOrb")) + 1:]))

                # Concatenate xOvers and partials w.r.t. each parameter
                xovers_df = pd.concat(
                    [
                        xovers_df,
                        pd.DataFrame(DelR_orb, index=xovers_df.index, columns=parOrb_xy),
                        pd.DataFrame(DelR_glo, index=xovers_df.index, columns=parGlo_xy)
                    ], axis=1
                )
            else:

                if XovOpt.get("debug"):
                    print('check ders')
                    print(np.hstack(np.squeeze(out_elev, axis=1)),
                          [np.linalg.norm(x) for x in list(param.values())[1:]])
                    print(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[::2])
                    print((np.diff(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[::2])[::2])[:len(XovOpt.get("parOrb")) * 2],
                          (2. * np.repeat([np.linalg.norm(x) for x in list(param.values())[1:len(XovOpt.get("parOrb")) + 1]], 2)))
                    print((np.diff(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[::2])[::2])[len(XovOpt.get("parOrb")) * 2:],
                          (2. * np.array([np.linalg.norm(x) for x in list(param.values())[len(XovOpt.get("parOrb")) + 1:]])))
                    print((np.diff(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[::2])[::2])[:len(XovOpt.get("parOrb")) * 2] / (
                            2. * np.repeat([np.linalg.norm(x) for x in list(param.values())[1:len(XovOpt.get("parOrb")) + 1]], 2)))
                    print((np.diff(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[::2])[::2])[len(XovOpt.get("parOrb")) * 2:] / (
                            2. * np.array([np.linalg.norm(x) for x in list(param.values())[len(XovOpt.get("parOrb")) + 1:]])))

                diff_step = [np.linalg.norm(x) for x in list(param.values())[1:]]

                # print("len(xovers_df.index) = 1 ", len(xovers_df.index))
                # print(out_elev)
                if len(out_elev) == 1:
                    out_elev = out_elev[0]

                xovers_df = pd.concat(
                    [xovers_df,
                     pd.DataFrame(
                         (np.diff(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[::2])[::2])[:len(XovOpt.get("parOrb")) * 2] / \
                         (2. * np.repeat([np.linalg.norm(x) for x in list(param.values())[1:len(XovOpt.get("parOrb")) + 1]], 2)),
                         index=parOrb_xy, columns=xovers_df.index).T,
                     pd.DataFrame(
                         (np.diff(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[::2])[::2])[len(XovOpt.get("parOrb")) * 2:] / \
                         (2. * np.array([np.linalg.norm(x) for x in list(param.values())[len(XovOpt.get("parOrb")) + 1:]])),
                         index=parGlo_xy, columns=xovers_df.index).T
                     ], axis=1
                )

            xovers_df = pd.concat([xovers_df,pd.DataFrame(np.reshape(self.get_dt(ladata_df, xovers_df),(len(xovers_df),2)),
                                  columns=['dtA','dtB'])],axis=1)
            # if (OrbRep == 'lin' or OrbRep == 'quad'):
            #     xovers_df = self.upd_orbrep(xovers_df)

            self.parOrb_xy = xovers_df.filter(regex='^dR/[a-zA-Z0-9]+_.*$').columns.values  # update partials list
            self.parGlo_xy = parGlo_xy
            self.par_xy = xovers_df.filter(regex='^dR/[a-zA-Z0-9]+.*$').columns.values  # update partials list

            # print("parOrb_xy", self.parOrb_xy)
            # print("par_xy", self.par_xy)

            # Update xovtmp as attribute for partials
            self.xovtmp = xovers_df

            # print(xovers_df)
            # exit()

        else:
            print("Observations in ", self.tracks, " excluded for inconsistent number of partials")
            print("nxov = ", len(xovers_df.index))
            print("max_xov_part =", max_xov_part)
            print("nxov_part_all_equal=", nxov_part_all_equal)
            print("no nans in part = ", no_nan_part)
        # Update general df
        # self.xovers = self.xovers.append(xovers_df)

    def upd_orbrep(self, xovers_df):
        """
        Project orbit partials to linear or quadratic representation of the orbit
        :param dt:
        :param xovers_df:
        :return: updated xov
        """
        # project d/dACR on linear expansion parameters
        # if A = A0 + A1*dt --> ddR/dA0 = ddR/dA*dA/dA0, with ddR/dA numerically computed and dA/dA0=1, etc...
        dt = sec2day(xovers_df.loc[:,['dtA','dtB']].values) # dt in days
        if XovOpt.get("OrbRep") in ['cnt','lin', 'quad']:
            xovers_df[[strng.partition('_')[0] + '0_' + strng.partition('_')[2] for strng in
                   xovers_df.filter(regex='^dR/d[A,C,R]_.*$').columns.values]] = \
                xovers_df[xovers_df.filter(regex='^dR/d[A,C,R]_.*$').columns.values]  # ddR/dA0
        if XovOpt.get("OrbRep") in ['lin', 'quad']:
            xovers_df[[strng.partition('_')[0] + '1_' + strng.partition('_')[2] for strng in
                       xovers_df.filter(regex='^dR/d[A,C,R]_.*$').columns.values]] = \
                xovers_df.filter(regex='^dR/d[A,C,R]_.*$') * np.tile(dt, int(
                    0.5 * len(xovers_df.filter(regex='^dR/d[A,C,R]_.*$').columns)))  # ddR/dA1
        # project d/dACR on quadratic expansion parameters
        if (XovOpt.get("OrbRep") == 'quad'):
            xovers_df[[strng.partition('_')[0] + '2_' + strng.partition('_')[2] for strng in
                       xovers_df.filter(regex='^dR/d[A,C,R]_.*$').columns.values]] = \
                xovers_df.filter(regex='^dR/d[A,C,R]_.*$') * np.tile(0.5 * np.square(dt), int(
                    0.5 * len(xovers_df.filter(regex='^dR/d[A,C,R]_.*$').columns)))  # ddR/dA2
        # TODO the orbital period should be taken automatically from input, not manually set
        if XovOpt.get("OrbRep") == 'per':
            xovers_df[[strng.partition('_')[0] + 'C_' + strng.partition('_')[2] for strng in
                       xovers_df.filter(regex='^dR/d[A,C,R]_.*$').columns.values]] = \
                xovers_df.filter(regex='^dR/d[A,C,R]_.*$') * np.tile(np.cos(2.*np.pi*dt/(2./24.)), int(
                    0.5 * len(xovers_df.filter(regex='^dR/d[A,C,R]_.*$').columns)))  # ddR/dA2
            xovers_df[[strng.partition('_')[0] + 'S_' + strng.partition('_')[2] for strng in
                       xovers_df.filter(regex='^dR/d[A,C,R]_.*$').columns.values]] = \
                xovers_df.filter(regex='^dR/d[A,C,R]_.*$') * np.tile(np.sin(2.*np.pi*dt/(2./24.)), int(
                    0.5 * len(xovers_df.filter(regex='^dR/d[A,C,R]_.*$').columns)))  # ddR/dA2

        xovers_df = xovers_df.drop(columns=xovers_df.filter(regex='^dR/d[A,C,R]_.*$'))  # drop ddR/dA

        return xovers_df

    ####@profile
    def get_dt(self,ladata_df,xovers_df):
        # tracksid = list(self.tracks.values())

        dt = np.squeeze([ladata_df.loc[ladata_df['orbID'].values == 0].loc[
                             map(round, xovers_df.cmb_idA.values)][['dt']].values, \
                         ladata_df.loc[ladata_df['orbID'].values == 1].loc[
                             map(round, xovers_df.cmb_idB.values)][['dt']].values])

        # dt = np.squeeze([ladata_df.loc[ladata_df['orbID'].values == 0].loc[
        #                      map(round, xovers_df.cmb_idA.values)][['ET_TX']].values, \
        #                  ladata_df.loc[ladata_df['orbID'].values == 1].loc[
        #                      map(round, xovers_df.cmb_idB.values)][['ET_TX']].values])
        # t0 = np.squeeze([ladata_df.loc[ladata_df['orbID'].values == 0].iloc[
        #                       0][['ET_TX']].values, \
        #                   ladata_df.loc[ladata_df['orbID'].values == 1].iloc[
        #                       0][['ET_TX']].values])
        # if len(xovers_df)==1:
        #     dt = dt - t0
        # else:
        #     dt = dt - t0[:, np.newaxis]

        return dt

    ###@profile
    def get_partials(self, l):

        # comb1 = self.tracks
        # ladata_df = self.ladata_df
        xovers_df = self.xovtmp

        if XovOpt.get("debug"):
            print("xov fin")
            print(xovers_df[['cmb_idA']].values.astype(int).flatten(), l)

        out_finloc = np.vstack(self.get_xover_fine(xovers_df['cmb_idA'].values.astype(int).flatten(),
                                                   xovers_df['cmb_idB'].values.astype(int).flatten(), l))  # seq

        if len(xovers_df) != len(out_finloc[0]):
            if XovOpt.get("debug"):
                print("*** It will crash!! Len xov=", len(xovers_df),
                      " - len partials=", len(out_finloc[0]), "for part ", l)
            return np.empty([1, 2])
        else:
            elev_parder = self.get_elev(list(self.tracks.values()), out_finloc[2], out_finloc[3], out_finloc[4], out_finloc[5], l)

            if XovOpt.get("debug"):
                print("elev_parder")
                print(self.tracks, out_finloc[2], out_finloc[3], out_finloc[4], out_finloc[5], l)
                print(elev_parder)

            return np.reshape(elev_parder[-2:], (-1, 2))
