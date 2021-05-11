import logging
import os
import shutil
import unittest

import filecmp
import glob

from accumxov import AccumXov
from accumxov.Amat import Amat
from accumxov.accum_opt import AccOpt
from pyaltsim import PyAltSim
from pygeoloc import PyGeoloc
from pyxover import PyXover
from config import XovOpt

# PyTest requires parallel = False
class TestBelaXover(unittest.TestCase):

    def setUp(self) -> None:
        # update paths and check options
        XovOpt.set("basedir", 'BELA/data/')
        XovOpt.set("instrument", 'BELA')

        XovOpt.set("sol4_orb", [])
        XovOpt.set("sol4_orbpar", [None])
        XovOpt.set("sol4_glo", ['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dL', 'dR/dh2'])

        vecopts = {'SCID': '-121',  # '-236',
                   'SCNAME': 'MPO',  # 'MESSENGER',
                   'SCFRAME': -121000,  # -236000,
                   # 'INSTID': (-236500, -236501),
                   # 'INSTNAME': ('MSGR_MLA', 'MSGR_MLA_RECEIVER'),
                   'PLANETID': '199',
                   'PLANETNAME': 'MERCURY',
                   'PLANETRADIUS': 2440.,
                   'PLANETFRAME': 'IAU_MERCURY',
                   'OUTPUTTYPE': 1,
                   'ALTIM_BORESIGHT': '',
                   'INERTIALFRAME': 'J2000',
                   'INERTIALCENTER': 'SSB',
                   'PM_ORIGIN': 'J2013.0',
                   'PARTDER': ''}
        XovOpt.set("vecopts", vecopts)

        XovOpt.set("expopt", 'BE0')
        XovOpt.set("resopt", 3)
        XovOpt.set("amplopt", 20)
        XovOpt.set("SpInterp", 0)
        XovOpt.set("spauxdir", 'MPO_spk/')
        XovOpt.set("parallel", True)
        XovOpt.set("unittest", True)

        XovOpt.check_consistency()

        if os.path.exists(XovOpt.get("outdir")):
            shutil.rmtree(XovOpt.get("outdir"))

        AccOpt.check_consistency()

    # # add simulation test for BELA data
    def test_sim_pipeline(self):

        months_to_process = ['2604'] #, '2612']

        XovOpt.set("sim_altdata", 1); XovOpt.set("partials", 0); XovOpt.set("parallel", 0); XovOpt.set("SpInterp", 2)
        XovOpt.set("apply_topo", False); XovOpt.set("range_noise", False); XovOpt.set("new_illumNG", True)

        for monyea in months_to_process:
            indir_in = f'SIM_{monyea[:2]}/{XovOpt.get("expopt")}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
            # outdir_in = f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_{monyea[:2]}'
            # geolocation step
            PyAltSim.main([XovOpt.get("amplopt"), XovOpt.get("resopt"), indir_in, f'{monyea}'])

        template_data = [x.split("/")[-1] for x in glob.glob(f"{XovOpt.get('rawdir')}SIM_26/BE0/template_tab_data/*.TAB")]
        cmp_results = filecmp.cmpfiles(f"{XovOpt.get('outdir')}SIM_26/BE0/3res_20amp/",
                               f"{XovOpt.get('rawdir')}SIM_26/BE0/template_tab_data/",
                                       common=template_data,
                                       shallow=False)
        cmp_results = dict(zip(["match","mismatch","errors"],cmp_results))
        # print(cmp_results["match"], template_data)
        assert cmp_results["match"] == template_data

    def test_proc_pipeline(self):

        # run full pipeline on a few MLA test data
        months_to_process = ['2604', '2612']
        for monyea in months_to_process:
            indir_in = f'SIM_{monyea[:2]}/{XovOpt.get("expopt")}/input_tab_data_proc/'
            outdir_in = f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_{monyea[:2]}'
            # geolocation step
            PyGeoloc.main([f'{monyea}', indir_in, outdir_in, 'MLASCIRDR', 0])
        # crossovers location step
        PyXover.main(['0', f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_',
                      f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/', 'MLASIMRDR', 0])
        # # lsqr solution step
        out = AccumXov.main([[f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'], 'sim', 0])
        #
        # generate new template (when needed)
        # out.save('bela_proc_test_out.pkl')

        # load template test results
        val = Amat(vecopts=XovOpt.get("vecopts"))
        try:
            val = val.load('bela_proc_test_out.pkl')
        except:
            val = val.load('BELA/bela_proc_test_out.pkl')

        # check xovers residuals
        # round up to avoid issues with package updates
        res_out = set([round(x, 4) for x in out.b])
        res_val = set([round(x, 4) for x in val.b])

        # perform test
        self.assertEqual(res_out, res_val)

        # check parameter solutions
        # round up to avoid issues with package updates
        out = {key: round(out.sol_dict['sol'][key], 4) for key in out.sol_dict['sol']}
        val = {key: round(val.sol_dict['sol'][key], 4) for key in val.sol_dict['sol']}

        # perform test
        self.assertEqual(out, val)

    def TearDown(self):
        logging.info("TestBelaXov done!")

if __name__ == '__main__':
    unittest.main()