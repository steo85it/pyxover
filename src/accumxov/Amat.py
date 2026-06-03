#!/usr/bin/env python3
# ----------------------------------
# Amat.py
#
# Description: define class Amat for first design/partial derivatives matrix (least square) and
# required attributes and functions
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 18-Feb-2019

import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

# from mapcount import mapcount
from config import XovOpt
from pyxover.xov_setup import xov


class Amat:

   def __init__(self, vecopts):

      self.vecopts = vecopts
      self.parNames = None
      self.pert_cloop = None
      self.pert_cloop_glo = None
      self.pert_cloop_0 = None
      self.sol = None
      self.sol_dict = None  # 2 x npar
      self.sol_iter = None  # 2 x npar
      self.sol_dict_iter = None # npar
      self.weights = None # nobsxnobs
      self.vce_obs = None
      self.vce_pen = None
      self.penalty_mat = None  # nobsxnobs
      self.spA = None # nobsxnpar
      self.spA_sol4 = None # nobsxnpar
      self.b = None
      self.converged = False
      self.resid_wrmse = None
      self.xov = None
      self.to_constrain    = None # can be removed
      self.sol4_pars       = None # npar
      self.sol4_pars_iter  = None # npar
      self.penalty_mat     = None
      self.penalty_mat_avg = None

   def setup(self, xov):

      self.xov = xov
      self.xovpart_reorder()

      if XovOpt.get("cloop_sim"):
         self.setup_cloop(xov)

   def setup_cloop(self, xov):
      self.pert_cloop_0 = xov.pert_cloop_0
      self.pert_cloop = xov.pert_cloop
      self.pert_cloop_glo = self.pert_cloop.filter(['dLIB','dL', 'dRA', 'dDEC', 'dPM', 'dh2']).iloc[0]
      self.pert_cloop.drop(columns=['dLIB','dL', 'dRA', 'dDEC', 'dPM', 'dh2'], errors='ignore', inplace=True)
      if len(self.pert_cloop.columns) > 0 or not self.pert_cloop.empty:
         print("Max perturb cloop", self.pert_cloop.abs().max())
         print("Mean perturb cloop", self.pert_cloop.mean())

         self.pert_cloop.drop_duplicates(inplace=True)
         self.pert_cloop.sort_index(inplace=True)
         print("self.pert_cloop\n", self.pert_cloop.dropna())

   def save(self, filnam):
      base_dir = os.path.dirname(filnam)
      base_name = os.path.basename(filnam)
      if "Abmat" in base_name:
         base_name = base_name.replace("Abmat", "xov")
      else:
         base_name = "xov_" + base_name
      xov_base = os.path.join(base_dir, base_name)
      if getattr(self, "xov", None) is not None:
         self.xov.save(xov_base)
      self.save_matrices(filnam)
      self.save_metadata(filnam, xov_base)

   def save_matrices(self, filnam):
      base = filnam
      matrices_path = base + "_mat.npz"
      matrices_nosol_path = base + "_mat_nosol.npz"

      sparse_mats = {}
      array_mats = {}
      sparse_mats_nosol = {}
      array_mats_nosol = {}
      sparse_list_mats = {}
      array_list_mats = {}
      sparse_list_mats_nosol = {}
      array_list_mats_nosol = {}

      def _store_sparse(container, name, mat):
         mat = mat.tocsr()
         prefix = f"{name}__sparse"
         container[f"{prefix}_data"] = mat.data
         container[f"{prefix}_indices"] = mat.indices
         container[f"{prefix}_indptr"] = mat.indptr
         container[f"{prefix}_shape"] = np.array(mat.shape)

      def _store_sparse_list(container, name, mats):
         container[f"{name}__sparse_list_count"] = np.array(len(mats), dtype=np.int64)
         for i, mat in enumerate(mats):
            mat = mat.tocsr()
            prefix = f"{name}__sparse_list_{i}"
            container[f"{prefix}_data"] = mat.data
            container[f"{prefix}_indices"] = mat.indices
            container[f"{prefix}_indptr"] = mat.indptr
            container[f"{prefix}_shape"] = np.array(mat.shape)

      def _store_array_list(container, name, arrs):
         container[f"{name}__array_list_count"] = np.array(len(arrs), dtype=np.int64)
         for i, arr in enumerate(arrs):
            container[f"{name}__array_list_{i}"] = np.asarray(arr)

      def _as_ndarray_list(val):
         if not isinstance(val, (list, tuple)) or len(val) == 0:
            return None

         arrs = []
         for v in val:
            if issparse(v) or isinstance(v, (str, bytes, dict, set)):
               return None
            try:
               arr = np.asarray(v)
            except Exception:
               return None
            if arr.dtype == object and not isinstance(v, np.ndarray):
               return None
            arrs.append(arr)

         return arrs

      for key, val in self.__dict__.items():
         if key == "xov":
            continue

         is_nosol = key in ["spA", "b", "weights"]

         if issparse(val):
            if is_nosol:
               sparse_mats_nosol[key] = val
            else:
               sparse_mats[key] = val
         elif isinstance(val, np.ndarray):
            if is_nosol:
               array_mats_nosol[key] = val
            else:
               array_mats[key] = val
         elif isinstance(val, (list, tuple)) and len(val) > 0 and all(issparse(v) for v in val):
            if is_nosol:
               sparse_list_mats_nosol[key] = list(val)
            else:
               sparse_list_mats[key] = list(val)
         else:
            arr_list = _as_ndarray_list(val)
            if arr_list is not None:
               if is_nosol:
                  array_list_mats_nosol[key] = arr_list
               else:
                  array_list_mats[key] = arr_list

      if sparse_mats_nosol or array_mats_nosol or sparse_list_mats_nosol or array_list_mats_nosol:
         container = {}
         for name, mat in sparse_mats_nosol.items():
            _store_sparse(container, name, mat)
         for name, arr in array_mats_nosol.items():
            container[f"{name}__array"] = arr
         for name, mats in sparse_list_mats_nosol.items():
            _store_sparse_list(container, name, mats)
         for name, arrs in array_list_mats_nosol.items():
            _store_array_list(container, name, arrs)
         np.savez(matrices_nosol_path, **container)

      if sparse_mats or array_mats or sparse_list_mats or array_list_mats:
         container = {}
         for name, mat in sparse_mats.items():
            _store_sparse(container, name, mat)
         for name, arr in array_mats.items():
            container[f"{name}__array"] = arr
         for name, mats in sparse_list_mats.items():
            _store_sparse_list(container, name, mats)
         for name, arrs in array_list_mats.items():
            _store_array_list(container, name, arrs)
         np.savez(matrices_path, **container)

   def save_metadata(self, filnam, xov_base):
      def _is_jsonable(obj):
         try:
            json.dumps(obj)
            return True
         except TypeError:
            return False

      def _as_ndarray_list(val):
         if not isinstance(val, (list, tuple)) or len(val) == 0:
            return None

         arrs = []
         for v in val:
            if issparse(v) or isinstance(v, (str, bytes, dict, set)):
               return None
            try:
               arr = np.asarray(v)
            except Exception:
               return None
            if arr.dtype == object and not isinstance(v, np.ndarray):
               return None
            arrs.append(arr)

         return arrs

      base = filnam
      meta_path = base + ".json"
      matrices_path = base + "_mat.npz"
      matrices_nosol_path = base + "_mat_nosol.npz"

      meta = {"format": "amat_split_v3", "version": 3, "amat": {}, "files": {}}

      for key, val in self.__dict__.items():
         if key == "xov":
            continue
         if issparse(val):
            meta["amat"][key] = {"kind": "sparse_csr", "stored": "matrices"}
         elif isinstance(val, np.ndarray):
            meta["amat"][key] = {"kind": "ndarray", "stored": "matrices"}
         elif isinstance(val, (list, tuple)) and len(val) > 0 and all(issparse(v) for v in val):
            meta["amat"][key] = {"kind": "sparse_csr_list", "stored": "matrices"}
         else:
            arr_list = _as_ndarray_list(val)
            if arr_list is not None:
               meta["amat"][key] = {"kind": "ndarray_list", "stored": "matrices"}
            elif _is_jsonable(val):
               meta["amat"][key] = {"kind": "json", "value": val}
            else:
               meta["amat"][key] = {"kind": "repr", "value": repr(val)}
            continue

      if os.path.exists(matrices_path):
         meta["files"]["matrices"] = os.path.basename(matrices_path)
      if os.path.exists(matrices_nosol_path):
         meta["files"]["matrices_nosol"] = os.path.basename(matrices_nosol_path)
      if os.path.exists(xov_base + ".json"):
         meta["files"]["xov_metadata"] = os.path.basename(xov_base + ".json")

      with open(meta_path, "w", encoding="utf-8") as f:
         json.dump(meta, f, indent=2, ensure_ascii=False)

   # load Abmat from file
   def load(self, filnam, read_matrices=True, read_matrices_nosol=True):
      meta_path = filnam + ".json"

      if os.path.exists(meta_path):
         with open(meta_path, "r") as f:
            meta = json.load(f)

         files = meta.get("files", {}) if isinstance(meta.get("files"), dict) else {}
         matrices_path = files.get("matrices")
         matrices_nosol_path = files.get("matrices_nosol")

         if matrices_path and not os.path.isabs(matrices_path):
            matrices_path = os.path.join(os.path.dirname(meta_path), matrices_path)
         if matrices_nosol_path and not os.path.isabs(matrices_nosol_path):
            matrices_nosol_path = os.path.join(os.path.dirname(meta_path), matrices_nosol_path)

         mats, arrays = ({}, {})
         mats_nosol, arrays_nosol = ({}, {})
         if read_matrices:
            mats, arrays = self.load_matrices(matrices_path) if matrices_path and os.path.exists(matrices_path) else ({}, {})
         if read_matrices_nosol:
            mats_nosol, arrays_nosol = self.load_matrices(matrices_nosol_path) if matrices_nosol_path and os.path.exists(matrices_nosol_path) else ({}, {})

         for key, entry in meta.get("amat", {}).items():
            kind = entry.get("kind")
            if kind == "sparse_csr" and key in mats:
               setattr(self, key, mats[key])
            elif kind == "ndarray" and key in arrays:
               setattr(self, key, arrays[key])
            elif kind == "sparse_csr_list" and key in mats:
               setattr(self, key, mats[key])
            elif kind == "ndarray_list" and key in arrays:
               setattr(self, key, arrays[key])
            elif kind == "sparse_csr" and key in mats_nosol:
               setattr(self, key, mats_nosol[key])
            elif kind == "ndarray" and key in arrays_nosol:
               setattr(self, key, arrays_nosol[key])
            elif kind == "sparse_csr_list" and key in mats_nosol:
               setattr(self, key, mats_nosol[key])
            elif kind == "ndarray_list" and key in arrays_nosol:
               setattr(self, key, arrays_nosol[key])
            elif kind == "json":
               setattr(self, key, entry.get("value"))
            elif kind == "repr":
               setattr(self, key, entry.get("value"))
            else:
               setattr(self, key, entry.get("value", None))
         # load xov via its own json+parquet format (path from metadata)
         xov_meta = files.get("xov_metadata")
         if xov_meta:
            if not os.path.isabs(xov_meta):
               xov_meta = os.path.join(os.path.dirname(meta_path), xov_meta)
            self.xov = xov(vecopts={}).load(xov_meta)

         print('Amat loaded from ' + filnam)
         return self

      raise FileNotFoundError(f"Missing metadata file: {meta_path}")

   def load_matrices(self, matrices_path):
      npz = np.load(matrices_path)
      mats = {}

      for key in npz.files:
         if key.endswith("__sparse_list_count"):
            base = key[:-len("__sparse_list_count")]
            count = int(npz[key])
            mats[base] = []
            for i in range(count):
               prefix = f"{base}__sparse_list_{i}"
               data = npz[f"{prefix}_data"]
               indices = npz[f"{prefix}_indices"]
               indptr = npz[f"{prefix}_indptr"]
               shape = tuple(npz[f"{prefix}_shape"])
               mats[base].append(csr_matrix((data, indices, indptr), shape=shape))

      for key in npz.files:
         if key.endswith("__sparse_data") and "__sparse_list_" not in key:
            base = key[:-len("__sparse_data")]
            data = npz[f"{base}__sparse_data"]
            indices = npz[f"{base}__sparse_indices"]
            indptr = npz[f"{base}__sparse_indptr"]
            shape = tuple(npz[f"{base}__sparse_shape"])
            mats[base] = csr_matrix((data, indices, indptr), shape=shape)

      arrays = {}
      for key in npz.files:
         if key.endswith("__array_list_count"):
            base = key[:-len("__array_list_count")]
            count = int(npz[key])
            arrays[base] = [npz[f"{base}__array_list_{i}"] for i in range(count)]

      for key in npz.files:
         if key.endswith("__array") and "__array_list_" not in key:
            base = key[:-len("__array")]
            arrays[base] = npz[key]

      return mats, arrays

   @staticmethod
   def migrate_legacy(pkl_path, out_path=None):
      """
      Migrate a legacy pickle file to the split-format storage.
      If out_path is None, uses pkl_path as the target base name.
      """
      if out_path is None:
         out_path = pkl_path

      with open(pkl_path, "rb") as pklfile:
         obj = pickle.load(pklfile)

      if not isinstance(obj, Amat):
         raise TypeError(f"Expected Amat in {pkl_path}, got {type(obj)}")

      obj.save(out_path)
      return out_path

   # reorder and fill to sparse A and prepare for lsqr solution
   def xovpart_reorder(self):

      xovers_df = self.xov.xovers.reset_index(drop=True)
      # TODO check if this makes sense, seems redundant or second row taking wrong input from self....
      # parOrb_xy = list(set([part.split('_')[0] for part in sorted(self.xov.parOrb_xy)]))
      parOrb_xy = list(set([part for part in sorted(self.xov.parOrb_xy)]))
      parGlo_xy = sorted(self.xov.parGlo_xy)
      xovers_df.fillna(0,inplace=True)

      # Retrieve all orbits involved in xovers
      orb_unique = [str(x) for x in self.xov.tracks]

      # select cols
      OrbParFull = [x + '_' + y.split('_')[0] for x in orb_unique for y in parOrb_xy]
      Amat_col = list(set(OrbParFull)) + parGlo_xy

      dict_ = dict(zip(Amat_col, range(len(Amat_col))))
      self.parNames = dict_

      # Retrieve and re-organize partials w.r.t. observations, parameters and orbits

      # Set-up columns to extract
      # Extract from df to np arrays for orbit A and B, then stack togheter all partials for
      # parameters/observations for each orbit (dR/dp_1,...,dR/dp_n,orb,xovID)

      regex = [re.compile(r'^dR/.*_A$'), re.compile(r'^dR/.*_B$'), re.compile(r'^dR/.*[^_^A][^_^B]$')]
      # regex = [re.compile(r'^dR/.*[^_^A][^_^B]$')]
      orbit = ['orbA', 'orbB', '']
      csr = []
      for rex, orb in zip(regex, orbit):
         if (orb != ''):
            par_xy_loc = list(filter(rex.search, parOrb_xy))
            partder = xovers_df[par_xy_loc].values
            orb_loc = xovers_df[orb].values
            
            col = np.array(
               list(map(dict_.get, [str(x) + '_' + str(y).split('_')[0] for x in orb_loc for y in par_xy_loc])))
         else:
            par_xy_loc = list(filter(rex.search, parGlo_xy))
            partder = xovers_df[par_xy_loc].values
            
            col = np.tile(list(map(dict_.get, [str(y) for y in par_xy_loc])), len(xovers_df.xOvID.values))

         # row = np.repeat(xovers_df.xOvID.values, len(par_xy_loc))
         row = np.repeat(xovers_df.index.values, len(par_xy_loc))
         val = partder.flatten()
         # negate value of orbit partial for orbB (dR = R_A - R_B) ... but works worse... never mind
         # if rex == re.compile(r'^dR/.*_B$'):
         #     print('val',val)
         #     val *= 1.
         #     print(val)

         if XovOpt.get("debug"):
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

         csr.append(csr_matrix((val, (row, col)), dtype=np.float32, shape=(len(orb_loc), len(Amat_col))))
         # print("done")

      csr = sum(csr)
	
      def sparse_memory_usage(mat):
         try:
            return mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes
         except AttributeError:
            return -1

      # if XovOpt.get("debug"):
      print("Memory of csr:",sparse_memory_usage(csr))

      def sparse_memory_usage(mat):
         try:
            return mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes
         except AttributeError:
            return -1

      if XovOpt.get("debug"):
         print(csr)
         print(list([np.array(map({v: k for k, v in dict_.items()}.get, csr.indices)), csr.data]))
         print(sys.getsizeof(csr))

      # Save A and b matrix/array for least square (Ax = b)
      self.spA = csr
      self.b = xovers_df.dR.values

      if (XovOpt.get("debug")):
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

      if XovOpt.get("debug"):
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
