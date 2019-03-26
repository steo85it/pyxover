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

warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd
import time
from scipy import interpolate
import pickle
import re
import matplotlib.pyplot as plt

# from mapcount import mapcount
from unproject_coord import unproject_stereographic
from intersection import intersection
from prOpt import debug, partials, parallel, OrbRep, parGlo, parOrb
import astro_trans as astr
from util import lflatten

class xov:

    def __init__(self, vecopts):

        self.vecopts = vecopts
        self.xovers = pd.DataFrame(
            columns=['x0', 'y0', 'ladata_idA', 'ladata_idB', 'R_A', 'R_B', 'dR'])
        self.param = {'': 1.}

    def setup(self, df):

        self.ladata_df = df.drop('chn',axis=1)
        self.msrm_sampl = 100

        self.tracks = df.orbID.unique()

        nxov = self.get_xov()

        if nxov > 0:
            # Compute and store distances between obs and xov coord
            self.set_xov_obs_dist()

            if (partials):
                self.set_partials()

            # Update general df
            self.xovers = self.xovers.append(self.xovtmp)
            self.xovers.reset_index(drop=True,inplace=True)
            self.xovers['xOvID']=self.xovers.index

        #print(self.xovers)

    def combine(self, xov_list):

        # Only select elements with number of xovers > 0
        xov_list = [x for x in xov_list if len(x.xovers) > 0]

        # concatenate df and reindex
        print([x.xovers for x in xov_list])
        self.xovers = pd.concat([x.xovers for x in xov_list])  # , sort=True)
        self.xovers = self.xovers.reset_index(drop=True)
        self.xovers['xOvID'] = self.xovers.index
        # print(self.xovers)

        # Retrieve all orbits involved in xovers
        orb_unique = self.xovers['orbA'].tolist()
        orb_unique.extend(self.xovers['orbB'].tolist())
        self.tracks = list(set(orb_unique))
        # print(self.tracks)

        self.parOrb_xy = xov_list[0].parOrb_xy
        self.parGlo_xy = xov_list[0].parGlo_xy
        # print(self.parOrb_xy, self.parGlo_xy)

    def save(self, filnam):
        pklfile = open(filnam, "wb")
        # clean ladata, which is now useless
        if hasattr(self, 'ladata_df'):
            del (self.ladata_df)
        pickle.dump(self, pklfile)
        pklfile.close()

    # load groundtrack from file
    def load(self, filnam):

        try:
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

    # Compute elevation R at crossover points by interpolation
    # (should be put in a function and looped over -
    # also, check the higher order interp)
    def get_elev(self, arg, ii, jj, ind_A, ind_B, par=''):

        ladata_df = self.ladata_df
        msrm_sampl = self.msrm_sampl

        #print(ii, jj, ind_A, ind_B)
        try:
            ind_A_int = np.atleast_1d(ind_A.astype(int))
            ind_B_int = np.atleast_1d(ind_B.astype(int))
        except:
            #print("flattening lists")
            ii, jj, ind_A, ind_B = np.array(lflatten(ii)), np.array(lflatten(jj)), np.array(lflatten(ind_A)), np.array(lflatten(ind_B))
            ind_A_int = np.atleast_1d(ind_A.astype(int))
            ind_B_int = np.atleast_1d(ind_B.astype(int))

        # Prepare
        param = self.param
        if (partials):
            param.update(parOrb)
            param.update(parGlo)

        if (debug):
            print("get_elev", arg, ii, jj, ind_A, ind_B, par)
            #print(ladata_df.loc[ladata_df['orbID'] == arg[0]].iloc[np.round(ind_A)])
            #print(ladata_df.loc[ladata_df['orbID'] == arg[1]].iloc[np.round(ind_B)])
        # print(par, param[par.partition('_')[0]] )

        # Apply elevation correction (if computing partial derivative)
        if (bool(re.search('_?A$', par)) or bool(re.search('_[p,m]$', par))):  # is not ''):
            ldA_ = ladata_df.loc[ladata_df['orbID'] == arg[0]][np.hstack(['genID', 'R', ladata_df.filter(
                    regex='^dR/' + par.partition('_')[0] + '$').columns.values])].values
            xyint = [ldA_[max(0, k - msrm_sampl):min(k + msrm_sampl, ladata_df.shape[0])].T for k in ind_A_int]

            #print(xyint[0][1], xyint[0][2])
            diff_step = np.linalg.norm(param[par.partition('_')[0]])

            if (bool(re.search('_pA?$', par))):
                xyint[0][1] += xyint[0][2] * diff_step
            elif (bool(re.search('_mA?$', par))):
                xyint[0][1] -= xyint[0][2] * diff_step
            # print(xyint[0][1])
        else:

            ldA_ = ladata_df.loc[ladata_df['orbID'] == arg[0]][['genID', 'R']].values
            xyint = [ldA_[max(0, k - msrm_sampl):min(k + msrm_sampl, ladata_df.shape[0])].T for k in ind_A_int]


        f_interp = [interpolate.interp1d(xyint[k][0], xyint[k][1], kind='linear') for k in range(0, len(ind_A_int))]

        #ind_A = ind_A + np.modf(ii)[0]
        # print(ind_A)
        # print(ind_B)
        R_A = [f_interp[k](ind_A.item(k)) for k in range(0, ind_A.size)]

        # Apply elevation correction
        if (bool(re.search('_?B$', par)) or bool(re.search('_[p,m]$', par))):  # is not ''):
            ldB_ = ladata_df.loc[ladata_df['orbID'] == arg[1]][np.hstack(['genID', 'R', ladata_df.filter(
                    regex='^dR/' + par.partition('_')[0] + '$').columns.values])].values
            xyint = [ldB_[max(0, k - len(ldA_) - msrm_sampl):min(k - len(ldA_) + msrm_sampl, ladata_df.shape[0])].T for k in ind_B_int]

            # print(xyint[0][1], xyint[0][2])
            diff_step = np.linalg.norm(param[par.partition('_')[0]])
            if (bool(re.search('_pB?$', par))):
                xyint[0][1] += xyint[0][2] * diff_step
            elif (bool(re.search('_mB?$', par))):
                xyint[0][1] -= xyint[0][2] * diff_step
            # print(xyint[0][1])
        else:
            ldB_ = ladata_df.loc[ladata_df['orbID'] == arg[1]][['genID', 'R']].values
            xyint = [ldB_[max(0, k - len(ldA_) - msrm_sampl):min(k - len(ldA_) + msrm_sampl, ladata_df.shape[0])].T for k in ind_B_int]

        f_interp = [interpolate.interp1d(xyint[k][0], xyint[k][1], kind='linear') for k in range(0, len(ind_B_int))]

        #ind_B = ind_B + np.modf(jj)[0]
        R_B = [f_interp[k](ind_B.item(k)) for k in range(0, ind_B.size)]

        if (debug):
            plt.plot(xyint[0][0], xyint[0][1], 'o', xyint[0][0], f_interp[0](xyint[0][0]), '-')
            plt.savefig('test_cub.png')
            plt.clf()
            plt.close()

        return ind_A, ind_B, R_A, R_B

    def get_xOver_fine(self, rough_indA, rough_indB, param):
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

        orb_lst = self.tracks.copy()

        msrm_sampl = self.msrm_sampl
        ladata_df = self.ladata_df

        X_stgA = 'X_stgprj'
        Y_stgA = 'Y_stgprj'
        X_stgB = 'X_stgprj'
        Y_stgB = 'Y_stgprj'

        # when computing partials, update variable names to retrieve column in ladata
        if (param is not ''):
            msrm_sampl = 100  # to reduce processing time, just check close to unperturbed xover
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

        if (debug):
            print(X_stgA, \
                  Y_stgA, \
                  X_stgB, \
                  Y_stgB)

        # compute more accurate location
        # Retrieve ladata_df index of observations involved in the crossover
        # (can be used to extract orbit number with join btw ladata_df and xovers_df -
        # eg, (ladata_df.loc[ind0][['orbID']].values).reshape(1,-1) -
        # the orbit number can then be used to get the value at i and j by interpolation)
        # ind0 and ind1 now are the indeces of the points just before the
        # intersection in ladata_df, so that (ind0,ind0+1) and (ind1,ind1+1) are the
        # bracketing points' indeces

        ldA_ = ladata_df.loc[ladata_df['orbID'] == orb_lst[0]][[X_stgA, Y_stgA]].values
        ldB_ = ladata_df.loc[ladata_df['orbID'] == orb_lst[1]][[X_stgB, Y_stgB]].values
        #print("Fine")
        #print(rough_indA, rough_indB)

        intersec_out = [
            intersection(ldA_[max(0, rough_indA[k] - msrm_sampl):min(rough_indA[k] + msrm_sampl, len(ldA_)), 0],
                         ldA_[max(0, rough_indA[k] - msrm_sampl):min(rough_indA[k] + msrm_sampl, len(ldA_)), 1],
                         ldB_[max(0, rough_indB[k] - len(ldA_) - msrm_sampl):min(rough_indB[k] - len(ldA_) + msrm_sampl,
                                                                                 len(ldB_)), 0],
                         ldB_[max(0, rough_indB[k] - len(ldA_) - msrm_sampl):min(rough_indB[k] - len(ldA_) + msrm_sampl,
                                                                                 len(ldB_)), 1])
            for k in range(len(rough_indA))]

        #print(len(rough_indA), len(intersec_out[0][0]),intersec_out[0][0] )
        #print(intersec_out)

        if len(rough_indA)>1:
            #print("len>1", len(rough_indA))
            #print(intersec_out)
            x,y,a,b,rough_tmpA, rough_tmpB = [], [], [], [], [], []
            for idx,r in enumerate(intersec_out):
                rough_tmpA.extend(np.repeat(rough_indA[idx],len(r[0])))
                rough_tmpB.extend(np.repeat(rough_indB[idx],len(r[0])))
                x.extend(r[0])
                y.extend(r[1])
                a.extend(r[2])
                b.extend(r[3])

            #print(x,y,a,b,rough_tmpA,rough_tmpB)
            idx_dup = [idx for idx, item in enumerate(x) if item in x[:idx]]
            #print(idx_dup)
            if len(idx_dup)>0 and debug:
                print("*** Duplicate obs eliminated for ", self.tracks, "on computing ", param)

            rough_indA = np.delete(rough_tmpA,idx_dup)
            rough_indB = np.delete(rough_tmpB,idx_dup)
            tmp = [np.delete(l,idx_dup) for l in [x,y,a,b]]
            intersec_out = []
            for i in range(len(tmp[0])):
                intersec_out.append([x[i] for x in tmp])
        elif len(rough_indA)==1 and len(intersec_out[0][0])>1:
            intersec_out = np.transpose(intersec_out)

        intersec_out = np.reshape(intersec_out,(-1,4))

        if intersec_out.size != 0 and intersec_out != []:

            intersec_x, intersec_y = intersec_out[:,0],intersec_out[:,1]
            fine_indA, fine_indB = intersec_out[:,2],intersec_out[:,3]

            #print(fine_indA, fine_indB)

            intersec_ind = intersec_out[:,2:]
            rough_ind = np.array([rough_indA, rough_indB]).T

            # index in local ladata_df is given by the position in subsample
            # minus size of subsample plus the position of the rough guess (center of subsample)
            tmp = rough_ind - msrm_sampl
            ld_ind_A = np.where(tmp[:,0]>0,(tmp+intersec_ind)[:,0],intersec_ind[:,0])
            ld_ind_B = np.where(tmp[:,1]-len(ldA_)>0,(tmp+intersec_ind)[:,1],len(ldA_)+intersec_ind[:,1])

            #print(intersec_x)

            # plot and check intersections (rough, fine, ...)
            if (debug):
                self.plot_xov_curves(ldA_, ldB_, intersec_x, intersec_y, rough_indA, rough_indB)
                #exit()

            if not all(intersec_x):
                ld_ind_A = [np.squeeze(x) for x in ld_ind_A[intersec_x > 0]][0]
                ld_ind_B = [np.squeeze(x) for x in ld_ind_B[intersec_x > 0]][0]
                return lflatten(intersec_x[intersec_x > 0]), lflatten(intersec_y[intersec_x > 0]), lflatten(fine_indA[intersec_x > 0]), lflatten(fine_indB[
                    intersec_x > 0]), ld_ind_A, ld_ind_B
            else:
                return intersec_x[intersec_x>0], intersec_y[intersec_x>0], fine_indA[intersec_x>0], fine_indB[intersec_x>0], ld_ind_A[intersec_x>0], ld_ind_B[intersec_x>0]
        else:
            return [],[],[],[],[],[]

    def plot_xov_curves(self, curve_A, curve_B, intersec_x, intersec_y, rough_A=0, rough_B=0):

        if debug:
            print('ladata_ind', curve_A, curve_B)
            print('XYproj', intersec_x, intersec_y)
            print('lonlat', unproject_stereographic(intersec_x, intersec_y, 0, 90, 2440))

        fig, ax = plt.subplots()
        ax.plot(curve_A[:,0],curve_A[:,1],c='b')
        ax.plot(curve_B[:,0],curve_B[:,1],c='C9')

        if all(rough_A!=0):
            xr = curve_A[rough_A,0]
            yr = curve_A[rough_A,1]
            ax.plot(xr, yr, '*r',label='rough xov A')
            xr = curve_B[rough_B-len(curve_A),0]
            yr = curve_B[rough_B-len(curve_A),1]
            ax.plot(xr, yr, '*g',label='rough xov B')

        ax.plot(intersec_x, intersec_y, '*k',label='fine xov')

        ax.set_title('Xover detected for tracks' + self.tracks[0] + '-' + self.tracks[1])
        ax.set_xlabel('x (distance from NP, km)')
        ax.set_ylabel('y (distance from NP, km)')
        ax.legend()

        delta = 100.
        if abs(np.amin(np.absolute(intersec_x))) > 100.:
            xmin = np.amin(np.hstack(intersec_x)) - delta
            xmax = np.amax(np.hstack(intersec_x)) + delta
        else:
            xmax = 200
            xmin = -200
        plt.xlim(xmin, xmax)
        if abs(np.amin(np.absolute(intersec_y))) > 100.:
            ymin = np.amin(np.array(intersec_y)) - delta
            ymax = np.amax(np.array(intersec_y)) + delta
        else:
            ymax = 200
            ymin = -200
        plt.ylim(ymin, ymax)

        plt.savefig('tmp/img/intersect_' + self.tracks[0] + '_' + self.tracks[1] + '.png')
        plt.clf()
        plt.close()

    # For each combination of 2 orbits, detect crossovers in 2 steps (rough, fine),
    # then get elevation for each point in the crossovers (by interpolating)
    def get_xOver_elev(self, arg):

        ladata_df = self.ladata_df
        msrm_sampl = self.msrm_sampl

        # Decimate data
        x, y, ind_A, ind_B = intersection(ladata_df.loc[ladata_df['orbID'] == arg[0]][['X_stgprj']].values[::msrm_sampl],
                                  ladata_df.loc[ladata_df['orbID'] == arg[0]][['Y_stgprj']].values[::msrm_sampl],
                                  ladata_df.loc[ladata_df['orbID'] == arg[1]][['X_stgprj']].values[::msrm_sampl],
                                  ladata_df.loc[ladata_df['orbID'] == arg[1]][['Y_stgprj']].values[::msrm_sampl])

        if debug:
            print("rough intersection")
            print(x, y, ind_A, ind_B)

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
            x=x[f]
            y=y[f]
            rough_indA = rough_indA[f]
            rough_indB = rough_indB[f]

            # try:
            x, y, subldA, subldB, ldA, ldB = self.get_xOver_fine(rough_indA, rough_indB, '')
            if len(x) > 0:
                #print("pre-elev")
                #print(subldA, subldB, ldA, ldB)
                ldA, ldB, R_A, R_B = self.get_elev(arg, subldA, subldB, ldA, ldB)

                #print(arg, x, y, ldA, ldB, R_A, R_B)
                return np.vstack((x, y, ldA, ldB, R_A, R_B)).T
            # except:

    #        outf.write('Issue with: '+str(arg)+'\n')
    #  print('Issue with: '+str(arg)+'\n')
    #  return

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
        results = self.get_xOver_elev(self.tracks)  # seq

        if (results is not None):
            xovtmp = pd.DataFrame(np.vstack(x for x in results if x is not None).reshape(-1, np.shape(results)[1]))
            xovtmp.columns = ['x0', 'y0', 'ladata_idA', 'ladata_idB', 'R_A', 'R_B']

            # if(parallel):
            #  pool.close()
            #  pool.join()

            # Get discrepancies (dR = R(obs1) - R(obs2)) at crossover time (where ideal dR=0)
            xovtmp['dR'] = xovtmp.R_B - xovtmp.R_A

            # Store reference orbit IDs for each xov
            xovtmp['orbA'] = \
                ladata_df.loc[ladata_df['orbID'] == self.tracks[0]].loc[map(round, xovtmp.ladata_idA.values)][
                    ['orbID']].values
            xovtmp['orbB'] = \
                ladata_df.loc[ladata_df['orbID'] == self.tracks[1]].loc[map(round, xovtmp.ladata_idB.values)][
                    ['orbID']].values
            xovtmp['xOvID'] = xovtmp.index

            # Update xovtmp as attribute for partials
            self.xovtmp = xovtmp

            # print(self.xovers)
            if (debug):
                print(str(len(xovtmp)) + " xovers found btw " + self.tracks[0] + " and " + self.tracks[1])

            return len(xovtmp)

        else:
            if (debug):
                print("no xovers btw " + self.tracks[0] + " and " + self.tracks[1])

            return -1  # 0 xovers found

    def set_xov_obs_dist(self):
        """
        Get distance of measurements around xover point

        :rtype: pandas df
        :param xovtmp: x-over dataframe
        :return: updated x-over dataframe with distance of xover from the 4 altimetry bounce points used to locate it
        """

        xovtmp = self.xovtmp.copy()
        xovtmp.reset_index(drop=True, inplace=True)
        # get coordinates of neighbouring measurements
        msrmnt_crd = []

        for trk in ['A','B']:
            obslist = xovtmp[['ladata_id'+trk]].values.astype(int).tolist()
            obslist = lflatten(obslist)

            msrmnt_crd.extend(
                self.ladata_df.loc[obslist][['X_stgprj', 'Y_stgprj']].values)
            msrmnt_crd.extend(
                self.ladata_df.loc[[l+1 for l in obslist]][['X_stgprj', 'Y_stgprj']].values)

        msrmnt_crd = np.reshape(msrmnt_crd, (-1, 2))

        # get compatible array with xov coordinates
        xov_crd = xovtmp[['x0', 'y0']].values
        xov_crd = np.reshape(np.tile(xov_crd.ravel(),4),(-1,2))

        # print(msrmnt_crd)
        # print(xov_crd)
        # print(np.linalg.norm(msrmnt_crd - xov_crd, axis=1))
        # print(np.reshape(np.linalg.norm(msrmnt_crd - xov_crd, axis=1),(-1,len(xovtmp))))
        # print(np.reshape(np.linalg.norm(msrmnt_crd - xov_crd, axis=1),(-1,len(xovtmp))).T)
        # exit()

        # compute distance and add to input df
        dist = np.reshape(np.linalg.norm(msrmnt_crd - xov_crd, axis=1),(-1,len(xovtmp))).T
        df_ = pd.DataFrame(dist,
                           columns=['dist_Am', 'dist_Ap', 'dist_Bm', 'dist_Bp'])
        # print(df_)
        # print(xovtmp)
        # print(pd.concat([xovtmp, df_], axis=1))
        # exit()

        if all(~df_[c].hasnans for c in df_) and (df_[['dist_Am', 'dist_Ap', 'dist_Bm', 'dist_Bp']].max() < 10000).all():
            self.xovtmp = pd.concat([xovtmp, df_], axis=1)
        else:
            print("*** error: NaN or unexpected value in set_xov_obs_dist")
            print(pd.concat([xovtmp, df_], axis=1))
            #exit(2)

    def set_partials(self):
        # Compute crossover position for partials at + and -
        # and combine them for each observation and parameter

        startXovPart = time.time()

        ladata_df = self.ladata_df
        xovers_df = self.xovtmp

        # Prepare
        param = self.param
        param.update(parOrb)
        param.update(parGlo)

        # print(ladata_df.loc[ladata_df['orbID'] == self.tracks[0]].loc[map(round, xovers_df.ladata_idA.values)][['X_stgprj','Y_stgprj','X_stgprj_dA_p','Y_stgprj_dA_p','X_stgprj_dA_m','Y_stgprj_dA_m']])
        # print(ladata_df.loc[ladata_df['orbID'] == self.tracks[1]].loc[map(round, xovers_df.ladata_idB.values)][['X_stgprj','Y_stgprj','X_stgprj_dA_p','Y_stgprj_dA_p','X_stgprj_dA_m','Y_stgprj_dA_m']])

        # Update combination array for all orbits involved in actual xOvers
        # self.comb1 = np.hstack((ladata_df.loc[ladata_df['orbID'] == self.tracks[0]].loc[
        #                             map(round, xovers_df.ladata_idA.values)][['orbID']].values,
        #                         ladata_df.loc[ladata_df['orbID'] == self.tracks[1]].loc[
        #                             map(round, xovers_df.ladata_idB.values)][['orbID']].values))

        # Compute fine location of xOvers for perturbed observations and relative elevations
        par_suffix = [(a + b) for a in list(parOrb.keys()) for b in ['_pA', '_mA', '_pB', '_mB']]
        par_suffix.extend([(a + b) for a in list(parGlo.keys()) for b in ['_p', '_m']])

        # out_elev contains per param Pi and xover xi, array[xi.RA.DPi+A,xi.RB.DPi+A] for all xi,
        # array[xi.RA.DPi-A,xi.RB.DPi-A] for all xi,
        # array[xi.RA.DPi+B,xi.RB.DPi+B] for all xi,
        # array[xi.RA.DPi-B,xi.RB.DPi-B] for all xi,
        # ... other parameters
        out_elev = []

        results = [self.get_partials(l) for l in par_suffix]  # seq
        out_elev.append(results)

        # Setup pandas containing all plus/minus dR_A/B and differentiate: dR/dp_A = ((R_B - R_A)_Aplus - (R_B - R_A)_Aminus)/2*diffstep
        parOrb_xy = [(a + b + c) for a in ['dR/'] for b in list(parOrb.keys()) for c in list(['_A', '_B'])]
        parGlo_xy = [(a + b) for a in ['dR/'] for b in list(parGlo.keys())]
        par_xy = parOrb_xy + parGlo_xy

        # parxy_df=pd.DataFrame(np.diff(np.diff(np.hstack(out_elev))[:,::2])[:,::2]/list(param.values())[1:],columns=par_xy)
        if (debug):
            print(par_xy)
            # print(ladata_df)
            # exit()

        max_xov_part = np.max([len(i) for i in out_elev[0]])
        nxov_part_all_equal = len(set([len(i) for i in out_elev[0]]))==1
        if len(xovers_df.index) == max_xov_part and nxov_part_all_equal:

            if (len(xovers_df.index) > 1):
                #print("len(xovers_df.index) > 1 ")
                #print(out_elev)
                DelR = (np.diff(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[:, ::2])[:, ::2])[:len(xovers_df)]

                DelR_orb = np.array(DelR)[:,:2*len(parOrb)]
                DelR_orb /=(2. * np.tile(list(param.values())[1:len(parOrb) + 1], 2))
                DelR_glo = np.array(DelR)[:,2*len(parOrb):]
                DelR_glo/= (2.* np.array([np.linalg.norm(x) for x in list(param.values())][len(parGlo) + 1:]) )

                # Concatenate xOvers and partials w.r.t. each parameter
                xovers_df = pd.concat(
                    [
                        xovers_df,
                        pd.DataFrame(DelR_orb, index=xovers_df.index, columns=parOrb_xy),
                        pd.DataFrame(DelR_glo, index=xovers_df.index, columns=parGlo_xy)
                    ], axis=1
                )
            else:

                if (debug):
                    print('check ders')
                    print(np.hstack(np.squeeze(out_elev, axis=1)), [np.linalg.norm(x) for x in list(param.values())[1:]])
                    print(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[::2])
                    print((np.diff(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[::2])[::2])[:len(parOrb) * 2],
                          (2. * np.repeat([np.linalg.norm(x) for x in list(param.values())[1:len(parOrb) + 1]], 2)))
                    print((np.diff(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[::2])[::2])[len(parOrb) * 2:],
                          (2. * np.array([np.linalg.norm(x) for x in list(param.values())[len(parOrb) + 1:]])))
                    print((np.diff(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[::2])[::2])[:len(parOrb) * 2] / (
                            2. * np.repeat([np.linalg.norm(x) for x in list(param.values())[1:len(parOrb) + 1]], 2)))
                    print((np.diff(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[::2])[::2])[len(parOrb) * 2:] / (
                            2. * np.array([np.linalg.norm(x) for x in list(param.values())[len(parOrb) + 1:]])))

                diff_step = [np.linalg.norm(x) for x in list(param.values())[1:]]

                #print("len(xovers_df.index) = 1 ", len(xovers_df.index))
                #print(out_elev)

                xovers_df = pd.concat(
                    [xovers_df,
                     pd.DataFrame((np.diff(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[::2])[::2])[:len(parOrb) * 2] / \
                                  (2. * np.repeat([np.linalg.norm(x) for x in list(param.values())[1:len(parOrb) + 1]], 2)),
                                  index=parOrb_xy, columns=xovers_df.index).T,
                     pd.DataFrame((np.diff(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[::2])[::2])[len(parOrb) * 2:] / \
                                  (2. * np.array([np.linalg.norm(x) for x in list(param.values())[len(parOrb) + 1:]])),
                                  index=parGlo_xy, columns=xovers_df.index).T
                     ], axis=1
                )

            if (OrbRep == 'lin' or OrbRep == 'quad'):
                xovers_df = self.upd_orbrep(ladata_df, xovers_df)

            self.parOrb_xy = xovers_df.filter(regex='^dR/[a-zA-Z0-9]+_.*$').columns.values  # update partials list
            self.parGlo_xy = parGlo_xy
            self.par_xy = xovers_df.filter(regex='^dR/[a-zA-Z0-9]+.*$').columns.values  # update partials list

            # Update xovtmp as attribute for partials
            self.xovtmp = xovers_df

        else:
            print("Observations in ", self.tracks," excluded for inconsistent number of partials")
            print(xovers_df.index)
            print("max_xov_part =", max_xov_part)
            print("nxov_part_all_equal=", nxov_part_all_equal)
        # Update general df
        #self.xovers = self.xovers.append(xovers_df)

    def upd_orbrep(self, ladata_df, xovers_df):
        """
        Project orbit partials to linear or quadratic representation of the orbit
        :param ladata_df:
        :param xovers_df:
        :return: updated xov
        """
        # project d/dACR on linear expansion parameters
        # if A = A0 + A1*dt --> ddR/dA0 = ddR/dA*dA/dA0, with ddR/dA numerically computed and dA/dA0=1, etc...
        dt = np.squeeze([ladata_df.loc[ladata_df['orbID'] == self.tracks[0]].loc[
                             map(round, xovers_df.ladata_idA.values)][['ET_TX']].values, \
                         ladata_df.loc[ladata_df['orbID'] == self.tracks[1]].loc[
                             map(round, xovers_df.ladata_idB.values)][['ET_TX']].values])
        xovers_df[[strng.partition('_')[0] + '0_' + strng.partition('_')[2] for strng in
                   xovers_df.filter(regex='^dR/d[A,C,R]_.*$').columns.values]] = \
            xovers_df[xovers_df.filter(regex='^dR/d[A,C,R]_.*$').columns.values]  # ddR/dA0
        xovers_df[[strng.partition('_')[0] + '1_' + strng.partition('_')[2] for strng in
                   xovers_df.filter(regex='^dR/d[A,C,R]_.*$').columns.values]] = \
            xovers_df.filter(regex='^dR/d[A,C,R]_.*$') * np.tile(dt, int(
                0.5 * len(xovers_df.filter(regex='^dR/d[A,C,R]_.*$').columns)))  # ddR/dA1
        # project d/dACR on quadratic expansion parameters
        if (OrbRep == 'quad'):
            xovers_df[[strng.partition('_')[0] + '2_' + strng.partition('_')[2] for strng in
                       xovers_df.filter(regex='^dR/d[A,C,R]_.*$').columns.values]] = \
                xovers_df.filter(regex='^dR/d[A,C,R]_.*$') * np.tile(0.5 * np.square(dt), int(
                    0.5 * len(xovers_df.filter(regex='^dR/d[A,C,R]_.*$').columns)))  # ddR/dA2

        xovers_df = xovers_df.drop(columns=xovers_df.filter(regex='^dR/d[A,C,R]_.*$'))  # drop ddR/dA

        return xovers_df

    def get_partials(self, l):

        #comb1 = self.tracks
        # ladata_df = self.ladata_df
        xovers_df = self.xovtmp

        if debug:
            print("xov fin")
            print(xovers_df[['ladata_idA']].values.astype(int).flatten(),l)

        out_finloc = np.vstack(self.get_xOver_fine(xovers_df[['ladata_idA']].values.astype(int).flatten(),
                                                    xovers_df[['ladata_idB']].values.astype(int).flatten(), l))  # seq

        if len(xovers_df) != len(out_finloc[0]) and debug:
            print("*** It will crash!! Len xov=", len(xovers_df)," - len partials=", len(out_finloc[0]), "for part ", l)

        elev_parder = self.get_elev(self.tracks, out_finloc[2], out_finloc[3], out_finloc[4], out_finloc[5], l)

        if debug:
            print("elev_parder")
            print(self.tracks, out_finloc[2], out_finloc[3], out_finloc[4], out_finloc[5], l)
            print(elev_parder)


        return np.reshape(elev_parder[-2:], (-1, 2))
