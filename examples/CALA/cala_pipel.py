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

# update paths and check options
from pyaltsim import PyAltSim

XovOpt.set("body", 'CALLISTO')
XovOpt.set("basedir", 'data/')
XovOpt.set("instrument", 'CALA')
XovOpt.set("debug", False)
# XovOpt.set("multi_xov", True)
XovOpt.set("selected_hemisphere", 'S')

# Subset of parameters to solve for
XovOpt.set("sol4_orb", [])
XovOpt.set("sol4_orbpar", [None])
XovOpt.set("sol4_glo", ['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dL', 'dR/dh2'])

vecopts = {'SCID': -555,  # '-236',
           'SCNAME': '-555',  # 'MESSENGER',
           'SCFRAME': -555000,  # -236000,
           'PLANETID': 504,
           'PLANETNAME': 'CALLISTO',
           'PLANETRADIUS': 2410.,
           'PLANETFRAME': 'IAU_CALLISTO',
           'OUTPUTTYPE': 1,
           'ALTIM_BORESIGHT': '',
           'INERTIALFRAME': 'J2000',
           'INERTIALCENTER': 'CALLISTO',
           'PM_ORIGIN': 'J2013.0',
           'PARTDER': ''}
XovOpt.set("vecopts", vecopts)

# Parameter constrains
XovOpt.set("par_constr",
           {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 3.e-1, 'dR/dA': 1.e2,
            'dR/dC': 1.e2, 'dR/dR': 2.e1})  # , 'dR/dRl':5.e1, 'dR/dPt':5.e1} #
XovOpt.set("mean_constr", {'dR/dA': 1.e0, 'dR/dC': 1.e0, 'dR/dR': 1.e0})

XovOpt.set("expopt", 'CA0')
XovOpt.set("resopt", 3)   # roughness topo small scale
XovOpt.set("amplopt", 20) #
XovOpt.set("spauxdir", 'CAL_spk/')

# interpolation/spice direct call (0: use spice, 1: yes, use interpolation, 2: yes, create interpolation)
XovOpt.set("SpInterp", 0) # TODO for some reason, SPICE interpolation gave weird results... beware+correct

XovOpt.check_consistency()
AccOpt.check_consistency()

if XovOpt.get("SpInterp") == 0:
    if not os.path.exists("data/aux/kernels"):
        os.makedirs("data/aux/kernels")
    os.chdir("data/aux/kernels")
    import wget

    furnsh_input = [
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup365.bsp",
        "http://spiftp.esac.esa.int/data/SPICE/BEPICOLOMBO/kernels/spk/de432s.bsp",
        "https://naif.jpl.nasa.gov/pub/naif/pds/data/mess-e_v_h-spice-6-v1.0/messsp_1000/data/pck/pck00010_msgr_v23.tpc",
        "http://spiftp.esac.esa.int/data/SPICE/BEPICOLOMBO/kernels/lsk/naif0012.tls"]
    for f in furnsh_input:
        if not os.path.exists(f.split('/')[-1]):
            wget.download(f)
    os.chdir('../../../')

# months_to_process = ['3105']
months_to_process = []

XovOpt.set("sim_altdata", True)
XovOpt.set("partials", False)
XovOpt.set("parallel", False)
XovOpt.set("apply_topo", False)
XovOpt.set("range_noise", False)
XovOpt.set("new_illumNG", True) # recompute a priori (set to false to nor recompute)
# XovOpt.set("unittest", True) # this restricts simulated data to the first day of the month (see d_last in PyAltSim.main)

# generate a few BELA test data
# Altimetry simulation
for monyea in months_to_process:
    indir_in = f'SIM_{monyea[:2]}/{XovOpt.get("expopt")}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
    # PyAltSim.main([XovOpt.get("amplopt"), XovOpt.get("resopt"), indir_in, f'{monyea}', XovOpt.to_dict()])

XovOpt.set("sim_altdata", False)
XovOpt.set("partials", True)
XovOpt.set("parallel", False)
XovOpt.set("SpInterp", 0)

# months_to_process = []
XovOpt.set("weekly_sets", True)
months_to_process = ['3105']
# run full pipeline on a few BELA test data
for monyea in months_to_process:
    indir_in = f'SIM_{monyea[:2]}/{XovOpt.get("expopt")}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
    outdir_in = f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_{monyea[:2]}'
    # geolocation step
    # 4th argument ('BELASCIRDR') unused?
    # PyGeoloc.main([f'{monyea}', indir_in, outdir_in, 'BELASCIRDR', 0, XovOpt.to_dict()])
# # crossovers location step
XovOpt.set("parallel", False)  # not sure why, but parallel gets crazy
# XovOpt.set("debug", True) # To test, to plot several useful plots, just for pyXover (set to False afterward)
PyXover.main(['0', f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_',
              f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/', 'MLASIMRDR', 0,
              XovOpt.to_dict()])

# Join North and South hemisphere xover
# # lsqr solution step
# WD out = AccumXov.main(
# WD     [[f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'], 'sim', 0,
# WD      XovOpt.to_dict()])