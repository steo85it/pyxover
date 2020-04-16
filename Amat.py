#!/usr/bin/env python3
# ----------------------------------
# xov_setup.py
#
# Description: 
# 
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 18-Feb-2019

import pickle
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# from mapcount import mapcount
from prOpt import debug


class Amat:

    def __init__(self, vecopts):

        self.vecopts = vecopts
        self.parNames = None
        self.pert_cloop = None
        self.pert_cloop_glo = None
        self.pert_cloop_0 = None
        self.sol = None
        self.sol_dict = None
        self.sol_iter = None
        self.sol_dict_iter = None
        self.weights = None
        self.penalty_mat = None
        self.spA = None
        self.b = None
        self.spA_penal = None
        self.converged = False
        self.resid_wrmse = None

    def setup(self, xov):

        self.xov = xov
        # print(self.xov.tracks)
        self.xovpart_reorder()
        # print(self.A)
        self.pert_cloop_0 = xov.pert_cloop_0
        self.pert_cloop = xov.pert_cloop
        self.pert_cloop_glo = self.pert_cloop.filter(['dL','dRA', 'dDEC', 'dPM', 'dh2']).iloc[0]
        self.pert_cloop.drop(columns=['dL','dRA', 'dDEC', 'dPM', 'dh2'],errors='ignore',inplace=True)

        if len(self.pert_cloop.columns) > 0 or not self.pert_cloop.empty:
            print("Max perturb cloop", self.pert_cloop.abs().max())
            print("Mean perturb cloop", self.pert_cloop.mean())

            self.pert_cloop.drop_duplicates(inplace=True)
            self.pert_cloop.sort_index(inplace=True)
            print("self.pert_cloop",self.pert_cloop.dropna())

    def save(self, filnam):
        pklfile = open(filnam, "wb")
        # clean ladata, which is now useless
        if hasattr(self, 'ladata_df'):
            del (self.ladata_df)
        # clean ladata, which is now useless
        if hasattr(self, 'A'):
            del (self.A)
            # print('A removed')
        pickle.dump(self, pklfile)
        pklfile.close()

        if debug:  # check if correctly saved
            vecopts = {}
            tmp = Amat(vecopts)
            tmp = tmp.load(filnam)
            print(tmp.spA)
            print(tmp.sol)

    # load groundtrack from file
    def load(self, filnam):

        pklfile = open(filnam, 'rb')
        self = pickle.load(pklfile)
        pklfile.close()

        print('Amat loaded from ' + filnam)
        # print(self.ladata_df)
        # print(self.MGRx.tck)

        return self

    # reorder and fill to sparse A and prepare for lsqr solution
    def xovpart_reorder(self):

        xovers_df = self.xov.xovers.reset_index(drop=True)
        # TODO check if this makes sense, seems redundant or second row taking wrong input from self....
        parOrb_xy = list(set([part.split('_')[0] for part in sorted(self.xov.parOrb_xy)]))
        parOrb_xy = list(set([part for part in sorted(self.xov.parOrb_xy)]))
        # print(parOrb_xy)
        parGlo_xy = sorted(self.xov.parGlo_xy)
        xovers_df.fillna(0,inplace=True)

        # Retrieve all orbits involved in xovers
        orb_unique = self.xov.tracks

        # select cols
        OrbParFull = [x + '_' + y.split('_')[0] for x in orb_unique for y in parOrb_xy]
        Amat_col = list(set(OrbParFull)) + parGlo_xy
#        print(np.array(Amat_col))

        dict_ = dict(zip(Amat_col, range(len(Amat_col))))
#        print(dict_)
        self.parNames = dict_

        # exit()
        # Retrieve and re-organize partials w.r.t. observations, parameters and orbits

        # Set-up columns to extract
        # Extract from df to np arrays for orbit A and B, then stack togheter all partials for
        # parameters/observations for each orbit (dR/dp_1,...,dR/dp_n,orb,xovID)

        regex = [re.compile(r'^dR/.*_A$'), re.compile(r'^dR/.*_B$'), re.compile(r'^dR/.*[^_^A][^_^B]$')]
#        regex = [re.compile(r'^dR/.*[^_^A][^_^B]$')]
        orbit = ['orbA', 'orbB', '']
        csr = []
        for rex, orb in zip(regex, orbit):
            print(orb, rex)
            if (orb != ''):
                par_xy_loc = list(filter(rex.search, parOrb_xy))
                partder = xovers_df[par_xy_loc].values
                orb_loc = xovers_df[orb].values
                #
                col = np.array(
                    list(map(dict_.get, [str(x) + '_' + str(y).split('_')[0] for x in orb_loc for y in par_xy_loc])))
            else:
                par_xy_loc = list(filter(rex.search, parGlo_xy))
                partder = xovers_df[par_xy_loc].values
                #
                col = np.tile(list(map(dict_.get, [str(y) for y in par_xy_loc])), len(xovers_df.xOvID.values))

            # row = np.repeat(xovers_df.xOvID.values, len(par_xy_loc))
            row = np.repeat(xovers_df.index.values, len(par_xy_loc))
            val = partder.flatten()
            # negate value of orbit partial for orbB (dR = R_A - R_B) ... but works worse... never mind
            # if rex == re.compile(r'^dR/.*_B$'):
            #     print('val',val)
            #     val *= 1.
            #     print(val)

            if debug:
                print("analyze df")
                par_xy_loc = list(set([str(y).split('_')[0] for y in par_xy_loc]))
                print(par_xy_loc)
                #  print(xovers_df[par_xy_loc])

                if (orb != ''):
                    print(orb_loc)
                    print(np.array([str(x) + '_' + str(y) for x in orb_loc for y in par_xy_loc]))
                else:
                    print([str(y) for y in par_xy_loc])
                print(np.column_stack((row, col, val)))
            # exit()

            csr.append(csr_matrix((val, (row, col)), dtype=np.float32, shape=(len(orb_loc), len(Amat_col))))
            print("done")

        csr = sum(csr)
	
        def sparse_memory_usage(mat):
            try:
                return mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes
            except AttributeError:
                return -1

        print("Memory of csr:",sparse_memory_usage(csr))

        def sparse_memory_usage(mat):
            try:
                return mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes
            except AttributeError:
                return -1

        print(sparse_memory_usage(csr))

        if debug:
            print(csr)
            print(list([np.array(map({v: k for k, v in dict_.items()}.get, csr.indices)), csr.data]))
            print(sys.getsizeof(csr))

        # Save A and b matrix/array for least square (Ax = b)
        self.spA = csr
        self.b = xovers_df.dR.values

        if (debug):
            print(csr)
            print(xovers_df.dR)

    # backup
    def xovpart_reorder2(self):

        xovers_df = self.xov.xovers
        parOrb_xy = sorted(self.xov.parOrb_xy)
        parGlo_xy = sorted(self.xov.parGlo_xy)
        # self.par_xy

        # Retrieve all orbits involved in xovers
        orb_unique = self.xov.tracks

        # Retrieve and re-organize partials w.r.t. observations, parameters and orbits

        # Set-up columns to extract
        # Extract from df to np arrays for orbit A and B, then stack togheter all partials for
        # parameters/observations for each orbit (dR/dp_1,...,dR/dp_n,orb,xovID)
        partder = xovers_df.filter(regex='^dR.*_A$').columns.values
        part_npA = np.array(
            [xovers_df.loc[xovers_df['orbA'] == orb_unique[k]][np.append(partder, ['orbA', 'xOvID']).tolist()].values
             for k in range(len(orb_unique))])

        partder = xovers_df.filter(regex='^dR.*_B$').columns.values
        part_npB = [
            xovers_df.loc[xovers_df['orbB'] == orb_unique[k]][np.append(partder, ['orbB', 'xOvID']).tolist()].values for
            k in range(len(orb_unique))]

        # same for global parameters (orbit ID is irrelevant here)
        partder_glb = xovers_df.filter(regex='^dR.*[^_^A][^_^B]$').columns.values
        part_glb = [
            xovers_df.loc[xovers_df['orbB'] == orb_unique[k]][np.append(partder_glb, ['orbB', 'xOvID']).tolist()].values
            for k in range(len(orb_unique))]

        # exit()

        # Set-up first design matrix A, associating each coefficient to the right column (orbID_dR/dp_i)
        # and row (observation number as in xovers_df)

        Amat_df = pd.DataFrame(np.nan, columns=range(10000), index=range(1000000), dtype='float32')
        # Amat_df.fillna(0, inplace=True)

        Amat_df.info(memory_usage='deep')

        exit()

        for dtype in ['float', 'int', 'object']:
            selected_dtype = Amat_df.select_dtypes(include=[dtype])
            mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
            mean_usage_mb = mean_usage_b / 1024 ** 2
            print("Average memory usage for {} columns: {:03.2f} MB".format(dtype, mean_usage_mb))

        # fill df_ with derivatives in xover_df by column name
        # for i in range(0,len(np_)):
        for i in range(0, len(np.vstack(part_npA))):
            for j in range(0, int(len(partder))):
                Amat_df.ix[np.vstack(part_npA)[i, len(partder) + 1], [
                    str(np.vstack(part_npA)[i, len(partder)]) + '_' + parOrb_xy[2 * j]]] = np.vstack(part_npA)[i, j]
                Amat_df.ix[np.vstack(part_npB)[i, len(partder) + 1], [
                    str(np.vstack(part_npB)[i, len(partder)]) + '_' + parOrb_xy[2 * j + 1]]] = np.vstack(part_npB)[i, j]
        for i in range(0, len(np.vstack(part_glb))):
            for j in range(0, int(len(partder_glb))):
                Amat_df.ix[np.vstack(part_glb)[i, len(partder_glb) + 1], [parGlo_xy[j]]] = np.vstack(part_glb)[i, j]

        # self.A = Amat_df

        # Save A and b matrix/array for least square (Ax = b)
        self.spA = Amat_df.to_sparse(fill_value=0)
        self.b = xovers_df.dR

        print(Amat_df)

        if (debug):
            print(Amat_df)
            print(xovers_df.dR)

    def corr_mat(self):

        A = self.spA
        N = len(self.b)
        C = ((A.T * A - (sum(A).T * sum(A) / N)) / (N - 1)).todense()
        V = np.sqrt(np.mat(np.diag(C)).T * np.mat(np.diag(C)))

        # par_names = [x.split('/')[-1] for x in self.parNames]
        par_names = [x for x in self.parNames]

        return pd.DataFrame(np.divide(C, V + 1e-119),index=par_names,columns=par_names)
