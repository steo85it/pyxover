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
XovOpt.set("local", True)
XovOpt.set("parallel", False)
XovOpt.set("expopt", 'BS0')

XovOpt.set("new_gtrack", 2)
vecopts = XovOpt.get('vecopts')
vecopts['SCFRAME'] = 'MSGR_SPACECRAFT'
XovOpt.set('vecopts', vecopts)
XovOpt.set("SpInterp", 0)
XovOpt.check_consistency()
AccOpt.check_consistency()

if XovOpt.get("SpInterp")==0:
    if not os.path.exists("data/aux/kernels"):
        os.makedirs("data/aux/kernels")
    os.chdir("data/aux/kernels")
    import wget
    furnsh_input = ["https://naif.jpl.nasa.gov/pub/naif/pds/data/mess-e_v_h-spice-6-v1.0/messsp_1000/data/spk/msgr_110716_120430_recon_gsfc_1.bsp",
                    "https://naif.jpl.nasa.gov/pub/naif/pds/data/mess-e_v_h-spice-6-v1.0/messsp_1000/data/spk/msgr_120501_130430_recon_gsfc_1.bsp",
                    "https://naif.jpl.nasa.gov/pub/naif/pds/data/mess-e_v_h-spice-6-v1.0/messsp_1000/data/spk/de405.bsp",
                    "https://naif.jpl.nasa.gov/pub/naif/pds/data/mess-e_v_h-spice-6-v1.0/messsp_1000/data/ck/msgr_1201_v01.bc",
                    "https://naif.jpl.nasa.gov/pub/naif/pds/data/mess-e_v_h-spice-6-v1.0/messsp_1000/data/ck/msgr_1301_v01.bc",
                    "https://naif.jpl.nasa.gov/pub/naif/pds/data/mess-e_v_h-spice-6-v1.0/messsp_1000/data/pck/pck00010_msgr_v23.tpc",
                    "https://naif.jpl.nasa.gov/pub/naif/pds/data/mess-e_v_h-spice-6-v1.0/messsp_1000/data/lsk/naif0011.tls",
                    "https://naif.jpl.nasa.gov/pub/naif/pds/data/mess-e_v_h-spice-6-v1.0/messsp_1000/data/fk/msgr_v231.tf",
                    "https://naif.jpl.nasa.gov/pub/naif/pds/data/mess-e_v_h-spice-6-v1.0/messsp_1000/data/sclk/messenger_2548.tsc"]
    for f in furnsh_input:
        if not os.path.exists(f.split('/')[-1]):
            wget.download(f)
    os.chdir('../../../')
    # exit()

# os.chdir("MLA/")

# run full pipeline on a few MLA test data
PyGeoloc.main(['1201', 'SIM_12/BS0/0res_1amp/', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack_12', 'MLASCIRDR', 0, XovOpt.to_dict()])
PyGeoloc.main(['1301', 'SIM_13/BS0/0res_1amp/', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack_13', 'MLASCIRDR', 0, XovOpt.to_dict()])
PyXover.main(['12', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack_', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/', 'MLASIMRDR', 0, XovOpt.to_dict()])

# generate new template (when needed)
# out.save('mla_pipel_test_out.pkl')

XovOpt.set("sol4_orb", [])  # '1503250029'] #'1107021838','1210192326','1403281002','1503191143']  #                                                                                                                                         
XovOpt.set("sol4_orbpar", ['dA','dC','dR']) #,'dRl','dPt'] #,'dA1','dC1','dR1','dA2','dC2','dR2']  #] #

out = AccumXov.main([[f'sim/{XovOpt.get("expopt")}_0/0res_1amp/'], 'sim', 0, XovOpt.to_dict()])
