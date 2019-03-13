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
from prOpt import debug, partials, parallel, OrbRep


class xov:

    def __init__(self, vecopts):

        self.vecopts = vecopts
        self.xovers = pd.DataFrame(
            columns=['x0', 'y0', 'i_orbA', 'j_orbB', 'ladata_idA', 'ladata_idB', 'R_A', 'R_B', 'dR'])

    def setup(self, df):

        self.ladata_df = df
        self.msrm_sampl = 100

        self.tracks = df.orbID.unique()

        # read ladata_df and compute all xovers
        nxov = self.get_xov()

        # compute partials if required and if
        # there are xovers in current pair
        if (partials and nxov > 0):
            self.set_partials()
            # self.xovpart_reorder()

        # print(self.xovers)

    def combine(self, xov_list):

        # Only select elements with number of xovers > 0
        # print("pre-selec "+str(len(xov_list)))
        xov_list = [x for x in xov_list if len(x.xovers) > 0]
        # print("post-selec "+str(len(xov_list)))

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
    def get_elev(self, arg, i, j, ii, jj, ind_A, ind_B, par=''):

        ladata_df = self.ladata_df
        msrm_sampl = self.msrm_sampl

        # Prepare
        if (partials):
            parOrb = {'dA': 100.}
            #           'dA':100., 'dC':100., 'dR':20., 'dRl':20e-6, 'dPt':20e-6, \
            parGlo = {'dh2': 1.}
            #           'dRA':[0.001, 0.000, 0.000], 'dDEC':[0.001, 0.000, 0.000], 'dPM':[0.0, 0.00001, 0.0], 'dL':0.01} #, \
            param = {'': 1.}
            param.update(parOrb)
            param.update(parGlo)
        # \
        # 'dh2':0.1}
        else:
            param = {'': 1.}

        if (debug):
            print("get_elev", arg, i, j, ii, jj, ind_A, ind_B, par)
            print(ladata_df.loc[ladata_df['orbID'] == arg[0]].iloc[np.round(ind_A)])
            print(ladata_df.loc[ladata_df['orbID'] == arg[1]].iloc[np.round(ind_B)])
        # print(par, param[par.partition('_')[0]] )

        # Apply elevation correction (if computing partial derivative)
        if (bool(re.search('_?A$', par)) or bool(re.search('_[p,m]$', par))):  # is not ''):
            xyint = np.array([ladata_df.loc[ladata_df['orbID'] == arg[0]].iloc[
                              max(0, k - msrm_sampl):min(k + msrm_sampl, ladata_df.shape[0])]
                              [np.hstack(['genID', 'R', ladata_df.filter(
                    regex='^dR/' + par.partition('_')[0] + '$').columns.values])].values.T for k in
                              (np.atleast_1d(ind_A))])
            # print(xyint[0][1], xyint[0][2])
            diff_step = np.linalg.norm(param[par.partition('_')[0]])

            if (bool(re.search('_pA?$', par))):
                xyint[0][1] += xyint[0][2] * diff_step
            elif (bool(re.search('_mA?$', par))):
                xyint[0][1] -= xyint[0][2] * diff_step
            # print(xyint[0][1])
        else:
            xyint = np.array([ladata_df.loc[ladata_df['orbID'] == arg[0]].iloc[
                              max(0, k - msrm_sampl):min(k + msrm_sampl, ladata_df.shape[0])]
                              [['genID', 'R']].values.T for k in (np.atleast_1d(ind_A))])

        f_interp = [interpolate.interp1d(xyint[k][0], xyint[k][1], kind='linear') for k in range(0, len(i))]

        ind_A = ind_A + np.modf(ii)[0]
        R_A = [f_interp[k](ind_A.item(k)) for k in range(0, ind_A.size)]

        # Apply elevation correction
        if (bool(re.search('_?B$', par)) or bool(re.search('_[p,m]$', par))):  # is not ''):
            xyint = np.array([ladata_df.loc[ladata_df['orbID'] == arg[1]].iloc[
                              max(0, k - msrm_sampl):min(k + msrm_sampl, ladata_df.shape[0])]
                              [np.hstack(['genID', 'R', ladata_df.filter(
                    regex='^dR/' + par.partition('_')[0] + '$').columns.values])].values.T for k in
                              (np.atleast_1d(ind_B))])
            # print(xyint[0][1], xyint[0][2])
            diff_step = np.linalg.norm(param[par.partition('_')[0]])
            if (bool(re.search('_pB?$', par))):
                xyint[0][1] += xyint[0][2] * diff_step
            elif (bool(re.search('_mB?$', par))):
                xyint[0][1] -= xyint[0][2] * diff_step
            # print(xyint[0][1])
        else:

            xyint = np.array([ladata_df.loc[ladata_df['orbID'] == arg[1]].iloc[
                              max(0, k - msrm_sampl):min(k + msrm_sampl, ladata_df.shape[0])]
                              [['genID', 'R']].values.T for k in (np.atleast_1d(ind_B))])

        f_interp = [interpolate.interp1d(xyint[k][0], xyint[k][1], kind='linear') for k in range(0, len(j))]

        ind_B = ind_B + np.modf(jj)[0]
        R_B = [f_interp[k](ind_B.item(k)) for k in range(0, ind_B.size)]

        if (debug):
            plt.plot(xyint[0][0], xyint[0][1], 'o', xyint[0][0], f_interp[0](xyint[0][0]), '-')
            plt.savefig('test_cub.png')
            plt.clf()
            plt.close()

            print('out_get_elev', arg, par, ind_A, ind_B, R_A, R_B)

        return ind_A, ind_B, R_A, R_B

    def get_xOver_fine(self, arg, i, j, param):

        msrm_sampl = self.msrm_sampl
        ladata_df = self.ladata_df

        # print('get_xOver_fine',arg, i, j, param,param[:-1])

        X_stgA = 'X_stgprj'
        Y_stgA = 'Y_stgprj'
        X_stgB = 'X_stgprj'
        Y_stgB = 'Y_stgprj'

        # when computing partials, update variable names to retrieve column in ladata
        if (param is not ''):
            msrm_sampl = 25  # to reduce processing time, just check close to unperturbed xover
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

        # Check if any 2 xovers closer than msrm_sampl, if yes, remove one of them
        # (not doing this would result in a doubling of the xovers)
        f = np.insert((np.diff(i) > msrm_sampl) & (np.diff(j) > msrm_sampl), 0, 'True')
        i = i[f]
        j = j[f]

        # compute more accurate location
        out = np.squeeze([intersection(ladata_df.loc[ladata_df['orbID'] == arg[0]].iloc[
                                       max(0, i[k] - msrm_sampl):min(i[k] + msrm_sampl, ladata_df.shape[0])][
                                           [X_stgA]].values,
                                       ladata_df.loc[ladata_df['orbID'] == arg[0]].iloc[
                                       max(0, i[k] - msrm_sampl):min(i[k] + msrm_sampl, ladata_df.shape[0])][
                                           [Y_stgA]].values,
                                       ladata_df.loc[ladata_df['orbID'] == arg[1]].iloc[
                                       max(0, j[k] - msrm_sampl):min(j[k] + msrm_sampl, ladata_df.shape[0])][
                                           [X_stgB]].values,
                                       ladata_df.loc[ladata_df['orbID'] == arg[1]].iloc[
                                       max(0, j[k] - msrm_sampl):min(j[k] + msrm_sampl, ladata_df.shape[0])][
                                           [Y_stgB]].values)
                          for k in range(0, len(i))]).T

        # If single "rough guess" hides several xovers in fine search,
        # the result has the wrong shape for processing
        # Temp. sol.: just keeping one of the xovers, TBU
        try:
            x, y, ii, jj = out
        except:
            if (len(out) > 0):
                x, y, ii, jj = out[0]
            else:
                #	  outf.write("xOver fine failed: "+str(arg)+str(param)+'\n')
                print("xOver fine failed: " + str(arg) + str(param) + '\n')

        # plot and check differences
        if (debug and 1 == 2):

            print('XYproj', x, y)
            print('lonlat', unproject_stereographic(x, y, 0, 90, 2440))

            plt.plot(ladata_df.loc[ladata_df['orbID'] == arg[0]][[X_stgA]].values,
                     ladata_df.loc[ladata_df['orbID'] == arg[0]][[Y_stgA]].values, c='b')
            plt.plot(ladata_df.loc[ladata_df['orbID'] == arg[1]][[X_stgB]].values,
                     ladata_df.loc[ladata_df['orbID'] == arg[1]][[Y_stgB]].values, c='C9')

            plt.plot(ladata_df.loc[ladata_df['orbID'] == arg[0]][[X_stgA]].values[
                     max(0, int(i[0]) - msrm_sampl2):min(int(i[0]) + msrm_sampl2, len(
                         ladata_df.loc[ladata_df['orbID'] == arg[0]][[X_stgA]].values))],
                     ladata_df.loc[ladata_df['orbID'] == arg[0]][[Y_stgA]].values[
                     max(0, int(i[0]) - msrm_sampl2):min(int(i[0]) + msrm_sampl2, len(
                         ladata_df.loc[ladata_df['orbID'] == arg[0]][[Y_stgA]].values))], c='r')
            plt.plot(ladata_df.loc[ladata_df['orbID'] == arg[1]][[X_stgB]].values[
                     max(0, int(j[0]) - msrm_sampl2):min(int(j[0]) + msrm_sampl2, len(
                         ladata_df.loc[ladata_df['orbID'] == arg[1]][[X_stgB]].values))],
                     ladata_df.loc[ladata_df['orbID'] == arg[1]][[Y_stgB]].values[
                     max(0, int(j[0]) - msrm_sampl2):min(int(j[0]) + msrm_sampl2, len(
                         ladata_df.loc[ladata_df['orbID'] == arg[1]][[Y_stgB]].values))], c='g')
            plt.plot(x, y, '*k')
            # plt.plot(x_raw,y_raw,'kv')

            # print(np.hstack(x),np.hstack(y))

            delta = 100.
            if (abs(np.amin(np.absolute(x))) > 100.):
                xmin = np.amin(np.hstack(x)) - delta
                xmax = np.amax(np.hstack(x)) + delta
            else:
                xmax = 200
                xmin = -200
            plt.xlim(xmin, xmax)

            if (abs(np.amin(np.absolute(y))) > 100.):
                ymin = np.amin(np.array(y)) - delta
                ymax = np.amax(np.array(y)) + delta
            else:
                ymax = 200
                ymin = -200
            plt.ylim(ymin, ymax)

            plt.savefig('img/intersect_' + arg[0] + '_' + arg[1] + '.png')
            plt.clf()
            plt.close()

        # Retrieve ladata_df index of observations involved in the crossover
        # (can be used to extract orbit number with join btw ladata_df and xovers_df -
        # eg, (ladata_df.loc[ind0][['orbID']].values).reshape(1,-1) -
        # the orbit number can then be used to get the value at i and j by interpolation)
        # ind0 and ind1 now are the indeces of the points just before the
        # intersection in ladata_df, so that (ind0,ind0+1) and (ind1,ind1+1) are the
        # bracketing points' indeces
        if (np.array(ii).size > 1):
            ind_A = []
            ind_B = []
            for l, k, m, n in zip(i, j, np.array(ii), np.array(jj)):
                ind_A.append(ladata_df.loc[ladata_df['orbID'] == arg[0]].iloc[
                             max(0, l - msrm_sampl):min(l + msrm_sampl, ladata_df.shape[0])].iloc[
                                 np.array(m).astype(int)][['genID']].values)
                ind_B.append(ladata_df.loc[ladata_df['orbID'] == arg[1]].iloc[
                             max(0, k - msrm_sampl):min(k + msrm_sampl, ladata_df.shape[0])].iloc[
                                 np.array(n).astype(int)][['genID']].values)

            ind_A = np.squeeze(np.vstack(ind_A).T)
            ind_B = np.squeeze(np.vstack(ind_B).T)
            ii = np.hstack(ii)
            jj = np.hstack(jj)
        else:

            # print(i,j,np.array(ii),np.array(jj))

            ind_A = np.squeeze([ladata_df.loc[ladata_df['orbID'] == arg[0]].iloc[
                                max(0, i.item() - msrm_sampl):min(i.item() + msrm_sampl, ladata_df.shape[0])].iloc[
                                    ii.astype(int)][['genID']].values])
            ind_B = np.squeeze([ladata_df.loc[ladata_df['orbID'] == arg[1]].iloc[
                                max(0, j.item() - msrm_sampl):min(j.item() + msrm_sampl, ladata_df.shape[0])].iloc[
                                    jj.astype(int)][['genID']].values])

            # print(ii, jj, ind_A, ind_B)

        return ii, jj, ind_A, ind_B

    # For each combination of 2 orbits, detect crossovers in 2 steps (rough, fine),
    # then get elevation for each point in the crossovers (by interpolating)
    def get_xOver_elev(self, arg):

        ladata_df = self.ladata_df
        msrm_sampl = self.msrm_sampl

        # Decimate data
        x, y, i, j = intersection(ladata_df.loc[ladata_df['orbID'] == arg[0]][['X_stgprj']].values[::msrm_sampl],
                                  ladata_df.loc[ladata_df['orbID'] == arg[0]][['Y_stgprj']].values[::msrm_sampl],
                                  ladata_df.loc[ladata_df['orbID'] == arg[1]][['X_stgprj']].values[::msrm_sampl],
                                  ladata_df.loc[ladata_df['orbID'] == arg[1]][['Y_stgprj']].values[::msrm_sampl])

        # reassign index to full list (from down-sampled index)
        i *= msrm_sampl
        j *= msrm_sampl

        if (len(x) > 0):
            # Retrieve ladata_df index of observations involved in the crossover
            # (can be used to extract orbit number with join btw ladata_df and xovers_df -
            # eg, (ladata_df.loc[ind0][['orbID']].values).reshape(1,-1) -
            # the orbit number can then be used to get the value at i and j by interpolation)
            # ind0 and ind1 now are the indeces of the points just before the
            # intersection in ladata_df, so that (ind0,ind0+1) and (ind1,ind1+1) are the
            # bracketing points' indeces

            ind_A = ladata_df.loc[ladata_df['orbID'] == arg[0]].iloc[i].index.values
            ind_B = ladata_df.loc[ladata_df['orbID'] == arg[1]].iloc[j].index.values

            c = ind_A
            d = ind_B

            # try:
            ii, jj, a, b = self.get_xOver_fine(arg, c, d, '')
            ind_A, ind_B, R_A, R_B = self.get_elev(arg, i, j, ii, jj, a, b)
            return np.vstack((x, y, i, j, ind_A, ind_B, R_A, R_B)).T
            # except:

    #        outf.write('Issue with: '+str(arg)+'\n')
    #  print('Issue with: '+str(arg)+'\n')
    #  return

    def get_xov(self):

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
            xovtmp = pd.DataFrame(np.vstack(x for x in results if x is not None).reshape(-1, 8))
            xovtmp.columns = ['x0', 'y0', 'i_orbA', 'j_orbB', 'ladata_idA', 'ladata_idB', 'R_A', 'R_B']

            # if(parallel):
            #  pool.close()
            #  pool.join()

            # Get discrepancies (dR = R(obs1) - R(obs2)) at crossover time (where ideal dR=0)
            xovtmp['dR'] = xovtmp.R_B - xovtmp.R_A

            # Update xovtmp as attribute for partials
            self.xovtmp = xovtmp

            if (not partials):
                # Update general df
                self.xovers = self.xovers.append(xovtmp)

            # print(self.xovers)
            if (debug):
                print(str(len(xovtmp)) + " xovers found btw " + self.tracks[0] + " and " + self.tracks[1])

            return len(xovtmp)

        else:
            if (debug):
                print("no xovers btw " + self.tracks[0] + " and " + self.tracks[1])

            return -1  # 0 xovers found

    def set_partials(self):
        # Compute crossover position for partials at + and -
        # and combine them for each observation and parameter

        startXovPart = time.time()

        ladata_df = self.ladata_df
        xovers_df = self.xovtmp

        # Prepare
        if (partials):
            # TODO put to prOpt as global
            parOrb = {'dA': 100.}
            #           'dA':100., 'dC':100., 'dR':20., 'dRl':20e-6, 'dPt':20e-6, \
            parGlo = {'dh2': 1.}
            #           'dRA':[0.001, 0.000, 0.000], 'dDEC':[0.001, 0.000, 0.000], 'dPM':[0.0, 0.00001, 0.0], 'dL':0.01} #, \
            param = {'': 1.}
            param.update(parOrb)
            param.update(parGlo)
        # \
        # 'dh2':0.1}
        else:
            param = {'': 1.}

        # print(ladata_df.loc[ladata_df['orbID'] == self.tracks[0]].loc[map(round, xovers_df.ladata_idA.values)][['X_stgprj','Y_stgprj','X_stgprj_dA_p','Y_stgprj_dA_p','X_stgprj_dA_m','Y_stgprj_dA_m']])
        # print(ladata_df.loc[ladata_df['orbID'] == self.tracks[1]].loc[map(round, xovers_df.ladata_idB.values)][['X_stgprj','Y_stgprj','X_stgprj_dA_p','Y_stgprj_dA_p','X_stgprj_dA_m','Y_stgprj_dA_m']])

        # Update combination array for all orbits involved in actual xOvers
        self.comb1 = np.hstack((ladata_df.loc[ladata_df['orbID'] == self.tracks[0]].loc[
                                    map(round, xovers_df.ladata_idA.values)][['orbID']].values,
                                ladata_df.loc[ladata_df['orbID'] == self.tracks[1]].loc[
                                    map(round, xovers_df.ladata_idB.values)][['orbID']].values))

        # Compute fine location of xOvers for perturbed observations and relative elevations
        par_suffix = [(a + b) for a in list(parOrb.keys()) for b in ['_pA', '_mA', '_pB', '_mB']]
        par_suffix.extend([(a + b) for a in list(parGlo.keys()) for b in ['_p', '_m']])

        # out_elev contains per param Pi and xover xi, array[xi.RA.DPi+A,xi.RB.DPi+A] for all xi,
        # array[xi.RA.DPi-A,xi.RB.DPi-A] for all xi,
        # array[xi.RA.DPi+B,xi.RB.DPi+B] for all xi,
        # array[xi.RA.DPi-B,xi.RB.DPi-B] for all xi,
        # ... other parameters
        out_elev = []

        if (parallel and 1 == 2):
            # print((mp.cpu_count() - 1))
            pool = mp.Pool(processes=mp.cpu_count() - 1)
            results = pool.map(self.get_partials, par_suffix)  # parallel
            out_elev.append(results)
        else:
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

        if (len(xovers_df.index) > 1):

            DelR = (np.diff(np.diff(np.hstack(np.squeeze(out_elev, axis=1)))[:, ::2])[:, ::2])[:len(xovers_df)]

            DelR_orb = np.array(DelR)[:,:2*len(parOrb)]
            DelR_orb /=(2. * np.tile(list(param.values())[1:len(parOrb) + 1], 2))
            DelR_glo = np.array(DelR)[:,2*len(parOrb):]
            DelR_glo/=(2. * np.array(list(param.values())[len(parGlo) + 1:]))

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

            # print('This step needs more than a single crossover (not sure why yet).')
            # exit()

        xovers_df['orbA'] = \
            ladata_df.loc[ladata_df['orbID'] == self.tracks[0]].loc[map(round, xovers_df.ladata_idA.values)][
                ['orbID']].values
        xovers_df['orbB'] = \
            ladata_df.loc[ladata_df['orbID'] == self.tracks[1]].loc[map(round, xovers_df.ladata_idB.values)][
                ['orbID']].values
        xovers_df['xOvID'] = xovers_df.index

        if (OrbRep == 'lin' or OrbRep == 'quad'):

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

        self.parOrb_xy = xovers_df.filter(regex='^dR/[a-zA-Z0-9]+_.*$').columns.values  # update partials list
        self.parGlo_xy = parGlo_xy
        self.par_xy = xovers_df.filter(regex='^dR/[a-zA-Z0-9]+.*$').columns.values  # update partials list

        # Update xovtmp as attribute for partials
        self.xovtmp = xovers_df

        # Update general df
        self.xovers = self.xovers.append(xovers_df)

    def get_partials(self, l):

        comb1 = self.comb1
        # ladata_df = self.ladata_df
        xovers_df = self.xovtmp

        out_finloc = np.vstack([self.get_xOver_fine(comb1[k], xovers_df[['ladata_idA']].values[k].astype(int),
                                                    xovers_df[['ladata_idB']].values[k].astype(int), l) for k in
                                range(0, len(comb1))])  # seq
        elev_parder = [self.get_elev(comb1[k], xovers_df[['i_orbA']].values[k], xovers_df[['j_orbB']].values[k],
                                     out_finloc[k, 0], out_finloc[k, 1], out_finloc[k, 2], out_finloc[k, 3], l) for k in
                       range(0, len(comb1))]

        return np.reshape([np.squeeze(k[-2:], axis=1) for k in elev_parder], (-1, 2))
