import unittest

from accumxov import AccumXov
from accumxov.Amat import Amat
from pygeoloc import PyGeoloc
from pyxover import PyXover
from examples.MLA.options import vecopts

# PyTest requires parallel = False
class MlaXoverTest(unittest.TestCase):
    def test_proc_pipeline(self):
        # run full pipeline on a few MLA test data
        PyGeoloc.main(['1201', 'SIM_12/BS0/0res_1amp/', 'sim/BS0_0/0res_1amp/gtrack_12', 'MLASIMRDR', 0])
        PyGeoloc.main(['1301', 'SIM_13/BS0/0res_1amp/', 'sim/BS0_0/0res_1amp/gtrack_13', 'MLASIMRDR', 0])
        PyXover.main(['12', 'sim/BS0_0/0res_1amp/gtrack_', 'sim/BS0_0/0res_1amp/', 'MLASIMRDR', 0])
        out = AccumXov.main([['sim/BS0_0/0res_1amp/'], 'sim', 0])

        # generate new template (when needed)
        # out.save('mla_pipel_test_out.pkl')

        # load template test results
        val = Amat(vecopts=vecopts)
        try:
            val = val.load('mla_pipel_test_out.pkl')
        except:
            val = val.load('tests/MLA/mla_pipel_test_out.pkl')

        # round up to avoid issues with package updates
        out = {key : round(out.sol_dict['sol'][key], 4) for key in out.sol_dict['sol']}
        val = {key : round(val.sol_dict['sol'][key], 4) for key in val.sol_dict['sol']}

        # perform test
        self.assertEqual(out, val)


if __name__ == '__main__':
    unittest.main()
