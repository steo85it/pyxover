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
import spiceypy as spice

# PyTest requires parallel = False

# update paths and check options
XovOpt.display()
XovOpt.set("debug", False)
XovOpt.set("basedir", 'data/')
XovOpt.set("instrument", 'MLA')
XovOpt.set("local", True)
XovOpt.set("parallel", False)
XovOpt.set("partials", False)
XovOpt.set("expopt", 'PAM')

XovOpt.set("new_gtrack", 2)
vecopts = XovOpt.get('vecopts')
vecopts['ALTIM_BORESIGHT'] = [2.2104999983228e-3, 2.9214999977833e-3, 9.9999328924124e-1]
vecopts['SCFRAME'] = 'MSGR_SPACECRAFT'
XovOpt.set('vecopts', vecopts)
XovOpt.set("SpInterp", 0)
XovOpt.set("compute_input_xov", True)

XovOpt.check_consistency()

spice.furnsh(f'{XovOpt.get("auxdir")}mymeta')

# run full pipeline on a few MLA test data
PyGeoloc.main(['1', '2001/', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack', 'MLASCIRDR', 0, XovOpt.to_dict()])
# PyGeoloc.main(['1', 'test/', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack', 'MLASCIRDR', 0, XovOpt.to_dict()])
PyXover.main(['0', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/', 'mlascirdr', 0, XovOpt.to_dict()])

# generate new template (when needed)
# out.save('mla_pipel_test_out.pkl')

# XovOpt.set("sol4_orb", [])  # '1503250029'] #'1107021838','1210192326','1403281002','1503191143']  #
# XovOpt.set("sol4_orbpar", ['dA','dC','dR']) #,'dRl','dPt'] #,'dA1','dC1','dR1','dA2','dC2','dR2']  #] #
#
# out = AccumXov.main([[f'sim/{XovOpt.get("expopt")}_0/0res_1amp/'], 'sim', 0, XovOpt.to_dict()])
