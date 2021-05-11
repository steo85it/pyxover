import logging
import os
import unittest

from accumxov.accum_opt import AccOpt
from config import XovOpt

from accumxov import AccumXov
from accumxov.Amat import Amat
from pygeoloc import PyGeoloc
from pyxover import PyXover
# from examples.MLA.options import XovOpt.get("vecopts")

# PyTest requires parallel = False
class MlaXoverTest(unittest.TestCase):

    def setUp(self) -> None:

        # update paths and check options
        XovOpt.set("basedir", 'MLA/data/')
        XovOpt.set("instrument", 'MLA')
        XovOpt.check_consistency()
        AccOpt.check_consistency()

        # os.chdir("MLA/")

    def test_proc_pipeline(self):

        # run full pipeline on a few MLA test data
        PyGeoloc.main(['1201', 'SIM_12/BS0/0res_1amp/', 'sim/BS0_0/0res_1amp/gtrack_12', 'MLASCIRDR', 0])
        PyGeoloc.main(['1301', 'SIM_13/BS0/0res_1amp/', 'sim/BS0_0/0res_1amp/gtrack_13', 'MLASCIRDR', 0])
        PyXover.main(['12', 'sim/BS0_0/0res_1amp/gtrack_', 'sim/BS0_0/0res_1amp/', 'MLASIMRDR', 0])

        # generate new template (when needed)
        # out.save('mla_pipel_test_out.pkl')

        out = AccumXov.main([['sim/BS0_0/0res_1amp/'], 'sim', 0])

        # load template test results
        val = Amat(vecopts=XovOpt.get("vecopts"))
        try:
            print(f'{XovOpt.get("instrument")}/mla_pipel_test_out.pkl')
            val = val.load(f'{XovOpt.get("instrument")}/mla_pipel_test_out.pkl')
        except:
            val = val.load(f'mla_pipel_test_out.pkl')

        # check xovers residuals
        # round up to avoid issues with package updates
        res_out = [round(x, 4) for x in out.b]
        res_val = [round(x, 4) for x in val.b]

        # perform test
        self.assertEqual(res_out, res_val)

        # check parameter solutions
        # round up to avoid issues with package updates
        out = {key : round(out.sol_dict['sol'][key], 4) for key in out.sol_dict['sol']}
        val = {key : round(val.sol_dict['sol'][key], 4) for key in val.sol_dict['sol']}

        # perform test
        self.assertEqual(out, val)

    def TearDown(self):
        # os.chdir("../")
        logging.info("TestMlaXover done!")

if __name__ == '__main__':
    unittest.main()
