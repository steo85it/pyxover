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

# update paths and check options
XovOpt.set("basedir", 'data/')
XovOpt.set("instrument", 'MLA')
XovOpt.check_consistency()
AccOpt.check_consistency()

# os.chdir("MLA/")

# run full pipeline on a few MLA test data
PyGeoloc.main(['1201', 'SIM_12/BS0/0res_1amp/', 'sim/BS0_0/0res_1amp/gtrack_12', 'MLASCIRDR', 0])
PyGeoloc.main(['1301', 'SIM_13/BS0/0res_1amp/', 'sim/BS0_0/0res_1amp/gtrack_13', 'MLASCIRDR', 0])
PyXover.main(['12', 'sim/BS0_0/0res_1amp/gtrack_', 'sim/BS0_0/0res_1amp/', 'MLASIMRDR', 0])

# generate new template (when needed)
# out.save('mla_pipel_test_out.pkl')

out = AccumXov.main([['sim/BS0_0/0res_1amp/'], 'sim', 0])
