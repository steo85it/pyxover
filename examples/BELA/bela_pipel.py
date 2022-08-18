import logging
import os
import unittest
import numpy as np

import submitit
from accumxov.accum_opt import AccOpt
from config import XovOpt

from accumxov import AccumXov
from accumxov.Amat import Amat
from pygeoloc import PyGeoloc
from pyxover import PyXover
# from examples.MLA.options import XovOpt.get("vecopts")

# update paths and check options
from pyaltsim import PyAltSim

XovOpt.set("body", 'MERCURY')
XovOpt.set("basedir", 'data/')
XovOpt.set("instrument", 'BELA')
XovOpt.set("debug", False)


XovOpt.set("sol4_orb", [])
XovOpt.set("sol4_orbpar", [None])
XovOpt.set("sol4_glo", ['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dL', 'dR/dh2'])

vecopts = {'SCID': '-121',  # '-236',
           'SCNAME': 'MPO',  # 'MESSENGER',
           'SCFRAME': -121000,  # -236000,
           'INSTID': (-121102, -121101),
           'INSTNAME': ('MPO_BELA_TRANSMITTER', 'MPO_BELA_RECEIVER'),
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

XovOpt.set("par_constr",
           {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 3.e-1, 'dR/dA': 1.e2,
            'dR/dC': 1.e2, 'dR/dR': 2.e1})  # , 'dR/dRl':5.e1, 'dR/dPt':5.e1} #
XovOpt.set("mean_constr", {'dR/dA': 1.e0, 'dR/dC': 1.e0, 'dR/dR': 1.e0})

XovOpt.set("expopt", 'BE0')
XovOpt.set("resopt", 3)
XovOpt.set("amplopt", 20)
XovOpt.set("SpInterp", 0)

XovOpt.check_consistency()
AccOpt.check_consistency()

if XovOpt.get("SpInterp") == 0:
    if not os.path.exists("data/aux/kernels"):
        os.makedirs("data/aux/kernels")
    os.chdir("data/aux/kernels")
    import wget

    furnsh_input = [
        "http://spiftp.esac.esa.int/data/SPICE/BEPICOLOMBO/kernels/spk/bc_mpo_mlt_50037_20260314_20280529_v04.bsp",
        "http://spiftp.esac.esa.int/data/SPICE/BEPICOLOMBO/kernels/spk/de432s.bsp",
        "https://naif.jpl.nasa.gov/pub/naif/pds/data/mess-e_v_h-spice-6-v1.0/messsp_1000/data/pck/pck00010_msgr_v23.tpc",
        "http://spiftp.esac.esa.int/data/SPICE/BEPICOLOMBO/kernels/lsk/naif0012.tls",
        "http://spiftp.esac.esa.int/data/SPICE/BEPICOLOMBO/kernels/fk/bc_mpo_v31.tf",
        "http://spiftp.esac.esa.int/data/SPICE/BEPICOLOMBO/kernels/sclk/bc_mpo_step_20220420.tsc",
        "http://spiftp.esac.esa.int/data/SPICE/BEPICOLOMBO/kernels/fk/bc_sci_v10.tf"]
    for f in furnsh_input:
        if not os.path.exists(f.split('/')[-1]):
            wget.download(f)
    os.chdir('../../../')

# PYALTSIM
# --------
months_to_process = ['2604', '2612']

XovOpt.set("sim_altdata", True)
XovOpt.set("partials", False)
XovOpt.set("parallel", False)
XovOpt.set("apply_topo", False)
XovOpt.set("range_noise", False)
XovOpt.set("new_illumNG", True)
XovOpt.set("unittest", True) # this restricts simulated data to the first day of the month (see d_last in PyAltSim.main)

# generate a few BELA test data
for monyea in months_to_process:
    indir_in = f'SIM_{monyea[:2]}/{XovOpt.get("expopt")}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
    # PyAltSim.main([XovOpt.get("amplopt"), XovOpt.get("resopt"), indir_in, f'{monyea}', XovOpt.to_dict()])

# PYGEOLOC
# --------
# executor is the submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder="log_slurm", cluster='local')
# executor = submitit.LocalExecutor(folder="log_slurm")
# set timeout in min, and partition for running the job
executor.update_parameters( #slurm_account='j1010',
                           slurm_name="pygeoloc",
                           slurm_cpus_per_task=1,
                           slurm_nodes=1,
                           slurm_time=60*99, # minutes
                           slurm_mem='90G',
                           slurm_array_parallelism=1)

XovOpt.set("sim_altdata", False)
XovOpt.set("partials", False)
XovOpt.set("parallel", False)
XovOpt.set("SpInterp", 0)

# run full pipeline on a few BELA test data
pygeoloc_in = []
for monyea in months_to_process:
    indir_in = f'SIM_{monyea[:2]}/{XovOpt.get("expopt")}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
    outdir_in = f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_' \
                f'{XovOpt.get("amplopt")}amp/gtrack_{monyea[:2]}'
    # geolocation step
    pygeoloc_in.append([f'{monyea}', indir_in, outdir_in, 'BELASCIRDR', 0, XovOpt.to_dict()])

if False:
    #job = executor.submit(PyXover.main, pygeoloc_in[0]) # single job
    jobs = executor.map_array(PyGeoloc.main, pygeoloc_in)
    for job in jobs:
        print(job.result())

# PYXOVER
# -------
executor.update_parameters( #slurm_account='j1010',
                           slurm_name="pyxover",
                           slurm_mem='2G',
                           slurm_array_parallelism=2)

XovOpt.set("monthly_sets", True)
XovOpt.set("compute_input_xov", True)

pyxover_in = [[comb, f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_',
               f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/',
               'MLASIMRDR', 0, XovOpt.to_dict()]
               for comb in [69, 77, 209]] #np.arange(1)]
if True:
    #job = executor.submit(PyXover.main, pyxover_in[5]) # single job
    jobs = executor.map_array(PyXover.main, pyxover_in)
    for job in jobs:
        print(job.result())

exit()
# ACCUMXOV
# --------
out = AccumXov.main(
    [[f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'], 'sim', 0,
     XovOpt.to_dict()])