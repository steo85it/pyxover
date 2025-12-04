import logging
import os
import unittest
import numpy as np
import json
import pandas as pd
import scipy.sparse as sp
import sys

# Ensure Python can find the `src` package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from accumxov.accum_opt import AccOpt
from config import XovOpt

from accumxov import AccumXov
from accumxov.Amat import Amat
from pygeoloc import PyGeoloc
from pyxover import PyXover
from xovutil.units import deg2as

# PyTest requires parallel = False
class MlaXoverTest(unittest.TestCase):

    def setUp(self) -> None:

        # update paths and check options
        XovOpt.set("basedir", 'MLA/data/')
        XovOpt.set("instrument", 'MLA')
        XovOpt.set("spice_meta", 'mymeta')
        XovOpt.set("local", False)
        XovOpt.set("parallel", False)
        XovOpt.set("expopt", 'BS0')

        XovOpt.set("new_gtrack", 2)
        vecopts = XovOpt.get('vecopts')
        vecopts['SCFRAME'] = 'MSGR_SPACECRAFT'
        vecopts['ALTIM_BORESIGHT'] = [0.0022105, 0.0029215, 0.9999932892]
        XovOpt.set('vecopts', vecopts)
        # WD: spice interpolation should be checked
        # Do we get the same result using the interpolation or not?
        XovOpt.set("SpInterp", 0)
        XovOpt.set("spauxdir", 'GSFC_spk/')
        
        XovOpt.set("msrm_sampl", 4)
        XovOpt.set("n_interp",4)
        XovOpt.set("parGlo", {'dRA': [0.2, 0.000, 0.000], 'dDEC': [0.36, 0.000, 0.000], 'dPM': [0, 0.013, 0.000],
              'dL': 1.e-3 * deg2as(1.) * np.linalg.norm([0.00993822, -0.00104581, -0.00010280, -0.00002364, -0.00000532]),
              'dh2': 0.1})
        XovOpt.set("par_constr", {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2})
        XovOpt.set("mean_constr", {})
        
        AccOpt.set("solving_method","lsqr")
        AccOpt.set("minobs_per_track", 1)
        AccOpt.set("weight_obs", [0.001])
        AccOpt.set("weight_constr", [5.0])
   
        XovOpt.check_consistency()
        AccOpt.check_consistency()
        # if downloading kernels is needed, refer to examples/MLA/data/aux dir

    def assertSparseMatrixEqual(self, A, B, tol=1e-8):
        A, B = A.tocsr(), B.tocsr()
        self.assertEqual(A.shape, B.shape, f"Shape mismatch: {A.shape} != {B.shape}")
        self.assertTrue(np.array_equal(A.indices, B.indices), "Sparsity pattern mismatch")
        self.assertTrue(np.array_equal(A.indptr, B.indptr), "Sparsity index pointer mismatch")
        self.assertTrue(np.allclose(A.data, B.data, atol=tol),
                        f"Matrix values differ (max diff = {np.max(np.abs(A.data - B.data))})")

    def save_sparse_container(self, path, **matrices):
       """Save multiple sparse matrices (CSR) in one .npz."""
       container = {}
       for name, M in matrices.items():
          container[f"{name}_data"] = M.data
          container[f"{name}_indices"] = M.indices
          container[f"{name}_indptr"] = M.indptr
          container[f"{name}_shape"] = np.array(M.shape)
       np.savez(path, **container)

    def load_sparse_container(self, path):
        npz = np.load(path)
        mats = {}
    
        # group by base name
        base_names = set(k.split("_")[0] for k in npz.files)
    
        for base in base_names:
            data    = npz[f"{base}_data"]
            indices = npz[f"{base}_indices"]
            indptr  = npz[f"{base}_indptr"]
            shape   = tuple(npz[f"{base}_shape"])
            mats[base] = sp.csr_matrix((data, indices, indptr), shape=shape)
    
        return mats

    def save_output(self, out, out_nosol, path):
       b_sparse = sp.csr_matrix(out_nosol.b.reshape(-1, 1))
       self.save_sparse_container(
          path+"matrices.npz",
          b=b_sparse,
          spA=out_nosol.spA,
          weights=out_nosol.weights
         )
       
       metadata = {}
       for key, value in out.__dict__.items():
          try:
             json.dumps(value)     # check serializability
             metadata[key] = value
          except TypeError:
             metadata[key] = repr(value)  # safe fallback
             
       with open(path + "Abmat_metadata.json", "w") as f:
         json.dump(metadata, f)

       out_nosol.xov.xovers.to_parquet(path+"xovers.parquet", engine='pyarrow')
       metadata = {}
       for key, value in out_nosol.xov.__dict__.items():
          try:
             json.dumps(value)     # check serializability
             metadata[key] = value
          except TypeError:
             metadata[key] = repr(value)  # safe fallback
             
       with open(path + "xovers_metadata.json", "w") as f:
         json.dump(metadata, f)
       
    def test_proc_pipeline(self):
       
        os.chdir('tests/')
        
        id = 'BS0'
        iter = 0
        in_folder = f'{id}/'
        out_folder = f'{id}_{iter}/'
        gtrack_dirs = out_folder + 'gtrack_'
        ref_folder = f'{XovOpt.get("instrument")}/ref/'
        
        # run full pipeline on a few MLA test data
        PyGeoloc.main(['1201', 'SIM_12/' + in_folder, gtrack_dirs + '12', '', iter, XovOpt.to_dict()])
        PyGeoloc.main(['1301', 'SIM_13/' + in_folder, gtrack_dirs + '13', '', iter, XovOpt.to_dict()])
        PyXover.main(['12', gtrack_dirs, out_folder, ('1201','1301'), iter, XovOpt.to_dict()])
        AccumXov.main([[out_folder], '', 0, XovOpt.to_dict(), AccOpt.to_dict()])

        out = Amat(vecopts=XovOpt.get("vecopts"))
        out_nosol = out.load(XovOpt.get("outdir") + out_folder + "Abmat_BS0_0_1_nosol.pkl")
        out = out.load(XovOpt.get("outdir") + out_folder + "Abmat_BS0_0_1.pkl")

        # generate new template (when needed)
        # self.save_output(out, out_nosol, ref_folder)
        
        # load template test results
        with open(f'{ref_folder}Abmat_metadata.json') as f:
            metadata = json.load(f)

        mats = self.load_sparse_container(f'{ref_folder}matrices.npz')
        
        xovers = pd.read_parquet(f'{ref_folder}/xovers.parquet', engine='pyarrow')

        # perform test
        # round up to avoid issues with package updates
        errors = []
        # columns_to_test = ["LAT", "LON", "R_A", "R_B"]
        # columns_to_test = ['LAT', 'LON', 'R_A', 'R_B', 'cmb_idA', 'cmb_idB', 'dR',
        #                    'dR/dA_A','dR/dA_B', 'dR/dC_A', 'dR/dC_B', 'dR/dDEC',
        #                    'dR/dL', 'dR/dPM', 'dR/dRA', 'dR/dR_A', 'dR/dR_B',
        #                    'dR/dh2', 'dist_Am', 'dist_Ap', 'dist_Bm','dist_Bp',
        #                    'dtA', 'dtB', 'mla_idA', 'mla_idB', 'offnad_A', 'offnad_B',
        #                    'orbA', 'orbB', 'x0', 'y0', 'xOvID', 'dist_max', 'dist_min_mean',
        #                    'huber', 'huber_trks', 'interp_weight', 'weights']
        columns_to_test = ['LAT', 'LON', 'R_A', 'R_B', 'cmb_idA', 'cmb_idB', 'dR',
                           'dR/dA_A','dR/dA_B', 'dR/dC_A', 'dR/dC_B', 'dR/dDEC',
                           'dR/dL', 'dR/dPM', 'dR/dRA', 'dR/dR_A', 'dR/dR_B',
                           'dR/dh2', 'dist_Am', 'dist_Ap', 'dist_Bm','dist_Bp',
                           'dtA', 'dtB', 'mla_idA', 'mla_idB', 'offnad_A', 'offnad_B',
                           'x0', 'y0', 'xOvID', 'dist_max', 'dist_min_mean']
        # columns_to_test = ['LAT', 'LON', 'R_A', 'R_B', 'cmb_idA', 'cmb_idB', 'dR',
        #                    'mla_idA', 'mla_idB', 'xOvID']:
        for field in columns_to_test:
            out_vals = [round(x, 4) if isinstance(x, (int, float, np.floating)) else x for x in getattr(out.xov.xovers, field)]
            val_vals = [round(x, 4) if isinstance(x, (int, float, np.floating)) else x for x in getattr(xovers, field)]

            try:
               self.assertEqual(out_vals, val_vals, msg=f"Mismatch in {field}")
            except AssertionError as e:
               errors.append(str(e))
           
        # check xovers residuals
        # round up to avoid issues with package updates
        b_sparse = sp.csr_matrix(out_nosol.b.reshape(-1, 1))
        res_out = [round(x, 4) for x in b_sparse]
        res_val = [round(x, 4) for x in mats["b"]]
        
        try:
           self.assertEqual(res_out, res_val, msg=f"Mismatch in b")
        except AssertionError as e:
           errors.append(str(e))
          
        try:
           self.assertSparseMatrixEqual(out_nosol.spA_sol4, mats["spA"][:,-5:-1], tol=1e-8)
        except AssertionError as e:
           errors.append(str(e))

        # check parameter solutions
        # round up to avoid issues with package updates
        res_out = {key : round(out.sol_dict['sol'][key], 4) for key in out.sol_dict['sol']}
        res_val = {key : round(metadata['sol_dict']['sol'][key], 4) for key in metadata['sol_dict']['sol']}

        # perform test
        try:
           self.assertEqual(res_out, res_val)
        except AssertionError as e:
           errors.append(str(e))
           
        # Fail at the end if any mismatches occurred
        if errors:
           self.fail("\n".join(errors))

    def TearDown(self):
        # os.chdir("../")
        logging.info("TestMlaXover done!")

if __name__ == '__main__':
    unittest.main()