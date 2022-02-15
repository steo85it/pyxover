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
XovOpt.set("local", False)
XovOpt.set("parallel", False)
XovOpt.set("expopt", 'AA2')

XovOpt.set("new_gtrack", 2)
vecopts = XovOpt.get('vecopts')
vecopts['SCFRAME'] = 'MSGR_SPACECRAFT'
XovOpt.set('vecopts', vecopts)
XovOpt.set("SpInterp", 1)
XovOpt.check_consistency()
AccOpt.check_consistency()

# os.chdir("MLA/")

# run full pipeline on a few MLA test data
PyGeoloc.main(['1206', 'SIM_12/KX5/0res_1amp/', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack_12', 'MLASCIRDR', 0, XovOpt.to_dict()])
#PyGeoloc.main(['1301', 'SIM_13/KX5/0res_1amp/', 'sim/BS0_0/0res_1amp/gtrack_13', 'MLASCIRDR', 0])
#PyXover.main(['7', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack_', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/', 'MLASIMRDR', 0, XovOpt.to_dict()])
exit()

# generate new template (when needed)
# out.save('mla_pipel_test_out.pkl')

XovOpt.set("sol4_orb", [])  # '1503250029'] #'1107021838','1210192326','1403281002','1503191143']  #                                                                                                                                         
XovOpt.set("sol4_orbpar", ['dA','dC','dR']) #,'dRl','dPt'] #,'dA1','dC1','dR1','dA2','dC2','dR2']  #] #

out = AccumXov.main([[f'sim/{XovOpt.get("expopt")}_0/0res_1amp/'], 'sim', 0])
