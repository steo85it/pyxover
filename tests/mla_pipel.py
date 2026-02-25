import logging
import os
import unittest
import numpy as np
import json
import pandas as pd
from pandas.testing import assert_frame_equal
import scipy.sparse as sp
import sys
import datetime as dt
import glob
import shutil

# Ensure Python can find the `src` package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from accumxov.accum_opt import AccOpt
from config import XovOpt

from accumxov import AccumXov
from accumxov.Amat import Amat
from pygeoloc import PyGeoloc
from pyaltsim import PyAltSim
from pyxover import PyXover
from xovutil.units import deg2as

# PyTest requires parallel = False
class MlaXoverTest(unittest.TestCase):

    def setUp(self) -> None:

        # update paths and check options
        XovOpt.set("basedir", '/home/wdesprat/nobackup/pyxover/tests/MLA/data/')
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

    def test_sim_pipeline(self):

        # mirror the example workflow: run_pyAltSim=True, grid=False
        XovOpt.set("partials", False)
        XovOpt.set("parallel", False)
        XovOpt.set("new_illumNG", True)
        XovOpt.set("apply_topo", False)
        XovOpt.set("small_scale_topo", False)
        XovOpt.set("range_noise", False)
        XovOpt.set("sampling_rate", 1)
        XovOpt.set("resopt", 3)
        XovOpt.set("amplopt", 20)
        XovOpt.check_consistency()

        # small window to keep the test light
        d_first = dt.datetime(2012, 1, 1, 10, 15, 0)
        d_last = dt.datetime(2012, 1, 1, 11, 15, 0)
        out_folder = 'SIM_12/BS1/'
        outdir = XovOpt.get("rawdir") + out_folder

        # clean previous outputs
        if os.path.exists(outdir):
            shutil.rmtree(outdir)

        PyAltSim.main([XovOpt.get("amplopt"), XovOpt.get("resopt"), out_folder, d_first, d_last, XovOpt.to_dict()])

        out_file = os.path.join(outdir, "MLASIMRDR1201011020.TAB")
        self.assertTrue(os.path.exists(out_file), f"Expected output not found: {out_file}")

        ref_file = os.path.join(
            "/home/wdesprat/nobackup/pyxover/tests/MLA/ref/raw/MLASIMRDR1201011020.TAB"
        )
        self.assertTrue(os.path.exists(ref_file), f"Reference file not found: {ref_file}")

        df_out = pd.read_csv(out_file, skipinitialspace=True)
        df_ref = pd.read_csv(ref_file, skipinitialspace=True)

        self.assertGreater(len(df_out), 0, "Simulated file has no rows")
        self.assertGreater(len(df_ref), 0, "Reference file has no rows")
        self.assertEqual(df_out.columns.tolist(), df_ref.columns.tolist())

        # allow minor numerical noise
        df_out = df_out.round(6)
        df_ref = df_ref.round(6)
        assert_frame_equal(df_out, df_ref, check_dtype=False)
       
    def test_proc_pipeline(self):

        id = 'BS0'
        iter = 0
        in_folder = f'{id}/'
        out_folder = f'{id}_{iter}/'
        gtrack_dirs = out_folder + 'gtrack_'
        ref_folder = f'/home/wdesprat/nobackup/pyxover/tests/{XovOpt.get("instrument")}/ref/'
        
        # run full pipeline on a few MLA test data
        PyGeoloc.main(['1201', 'SIM_12/' + in_folder, gtrack_dirs + '12', '', iter, XovOpt.to_dict()])
        PyGeoloc.main(['1301', 'SIM_13/' + in_folder, gtrack_dirs + '13', '', iter, XovOpt.to_dict()])
        PyXover.main(['12', gtrack_dirs, out_folder, ('1201','1301'), iter, XovOpt.to_dict()])
        AccumXov.main([[out_folder], '', 0, XovOpt.to_dict(), AccOpt.to_dict()])

        out = Amat(vecopts=XovOpt.get("vecopts"))
        out = out.load(XovOpt.get("outdir") + out_folder + "Abmat_BS0_0_1")

        # load template test results
        ref = Amat(vecopts=XovOpt.get("vecopts"))
        ref = ref.load(f'{ref_folder}Abmat_BS0_0_1')
        
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
            val_vals = [round(x, 4) if isinstance(x, (int, float, np.floating)) else x for x in getattr(ref.xov.xovers, field)]

            try:
               self.assertEqual(out_vals, val_vals, msg=f"Mismatch in {field}")
            except AssertionError as e:
               errors.append(str(e))
           
        # check xovers residuals
        # round up to avoid issues with package updates
        res_out = [round(x, 4) for x in np.asarray(out.b, dtype=object).ravel() if x is not None]
        mat_b = ref.b
        if sp.issparse(mat_b):
           mat_b = mat_b.toarray()
        res_val = [round(x, 4) for x in np.asarray(mat_b, dtype=object).ravel() if x is not None]
        try:
           self.assertEqual(res_out, res_val, msg=f"Mismatch in b")
        except AssertionError as e:
           errors.append(str(e))
          
        try:
           self.assertSparseMatrixEqual(out.spA_sol4, ref.spA_sol4, tol=1e-8)
        except AssertionError as e:
           errors.append(str(e))

        # check parameter solutions
        # round up to avoid issues with package updates
        res_out = {key : round(out.sol_dict['sol'][key], 4) for key in out.sol_dict['sol']}
        res_val = {key : round(ref.sol_dict['sol'][key], 4) for key in ref.sol_dict['sol']}

        # perform test
        try:
           self.assertEqual(res_out, res_val)
        except AssertionError as e:
           errors.append(str(e))
           
        # Fail at the end if any mismatches occurred
        if errors:
           self.fail("\n".join(errors))

    def TearDown(self):
        logging.info("TestMlaXover done!")

if __name__ == '__main__':
    unittest.main()
