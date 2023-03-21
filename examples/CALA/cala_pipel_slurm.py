import logging
import os
import unittest
import submitit
import datetime as dt

from accumxov.accum_opt import AccOpt
from config import XovOpt

from accumxov import AccumXov
from accumxov.Amat import Amat
from pygeoloc import PyGeoloc
from pyxover import PyXover
# from examples.MLA.options import XovOpt.get("vecopts")

# update paths and check options
from pyaltsim import PyAltSim

grid = True
run_pyAltSim  = False # 3-4 hours per month
run_pyGeoLoc  = False # quite fase per gtrack
run_pyXover   = True
run_accuXover = False

XovOpt.set("body", 'CALLISTO')
XovOpt.set("basedir", 'data/')
XovOpt.set("instrument", 'CALA')

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

XovOpt.set("expopt", 'CA6')
XovOpt.set("resopt", 3)
XovOpt.set("amplopt", 20)
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

XovOpt.set("sim_altdata", True)
XovOpt.set("partials", False)
XovOpt.set("parallel", False)
XovOpt.set("apply_topo", False)
XovOpt.set("range_noise", False)
XovOpt.set("new_illumNG", True)
XovOpt.set("unittest", True) # this restricts simulated data to the first day of the month (see d_last in PyAltSim.main)
XovOpt.set("debug", True)

if grid:
    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder="log_slurm")
    # set timeout in min, and partition for running the job
    executor.update_parameters( #slurm_account='j1010',
                               slurm_name="pyaltsim",
                               slurm_partition="aiub",
                               slurm_cpus_per_task=2,
                               slurm_nodes=1,
                               slurm_time=60*24, # minutes
                               slurm_mem='8G',
                               slurm_array_parallelism=50)

# pyaltsim_in = [[f'{yy:02d}{mm:02d}', f'SIM_{yy:02d}/KX5/0res_1amp/', f'sim/{XovOpt.get("expopt")}_{iternum}/0res_1amp/gtrack_{yy:02d}', 'MLASCIRDR', iternum, opt_dict] for mm in range(13)[1:] for yy in [8,11,12,13,14,15]]

# generate a few BELA test data
# Altimetry simulation
d_sess = [dt.datetime(2031, 5, 1, 0, 0, 1),
    dt.datetime(2031, 5, 17, 18, 41, 15),
    dt.datetime(2031, 6,  3, 13, 23, 18),
    dt.datetime(2031, 6, 20,  8,  6, 13),
    dt.datetime(2031, 7,  7,  2, 50,  1),
    dt.datetime(2031, 7, 23, 21, 34, 35),
    dt.datetime(2031, 8,  9, 16, 19, 40)]

maneuver_time = [988632000, 990081675, 991531398, 992981173, 994431001, 995880875, 997330780]
mjd2000 = dt.datetime(2000,1,1,12,0,0)

d_sess = []
for sec in maneuver_time:
    d_sess.append(mjd2000+dt.timedelta(seconds=sec))

# WD should really organize the folders in weekly folders
if run_pyAltSim:
    pyaltsim_in = []
    for i in range(0, len(d_sess)-1):
    # for i in range(0, 1):
        d_first = d_sess[i] + dt.timedelta(seconds=1) # avoid out of bound
        d_last = d_sess[i+1] - dt.timedelta(seconds=1) # avoid overlaps
        monyea = d_first.strftime('%y%m%d')
        # indir_in = f'SIM_{monyea[:2]}/{XovOpt.get("expopt")}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
        indir_in = f'SIM_{monyea}/{XovOpt.get("expopt")}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
        pyaltsim_in.append([XovOpt.get("amplopt"), XovOpt.get("resopt"), indir_in, f'{monyea}', XovOpt.to_dict(),d_first, d_last])
    if grid:
        # job = executor.submit(PyAltSim.main, [XovOpt.get("amplopt"), XovOpt.get("resopt"), indir_in, f'{monyea}', XovOpt.to_dict()]) # single job
        # print(job.result())
        jobs = executor.map_array(PyAltSim.main, pyaltsim_in)
        for job in jobs:
            print(job.result())
    else:
        for arg in pyaltsim_in:
            # PyAltSim.main([XovOpt.get("amplopt"), XovOpt.get("resopt"), indir_in, f'{monyea}', XovOpt.to_dict()])
            PyAltSim.main(arg)    

XovOpt.set("sim_altdata", False)
XovOpt.set("partials", True)
XovOpt.set("parallel", False)
XovOpt.set("SpInterp", 0)

# run full pipeline on a few BELA test data

executor.update_parameters( #slurm_account='j1010',
                           slurm_name="pygeoloc",
                           slurm_mem='10G',
                           slurm_array_parallelism=50)

XovOpt.set("selected_hemisphere",'S')
# WD should really organize the folders in weekly folders (should be done already for simulation)
# geolocation step
if run_pyGeoLoc:
    for i in range(0, len(d_sess)-1):
        d_last = d_sess[i+1] - dt.timedelta(seconds=1) # avoid overlaps
        monyea = d_first.strftime('%y%m%d')
        indir_in = f'SIM_{monyea}/{XovOpt.get("expopt")}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
        indir_in = f'SIM_{monyea}/CA5/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
        # outdir_in = f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_{monyea[:2]}'
        outdir_in = f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_{monyea}'
        if grid:
            import re
            pattern = 'CALASIMRDR'
            pygeoloc_in = []
            for file in os.listdir(f'{XovOpt.get("rawdir")}{indir_in}'):
                result = re.search(f'{pattern}(.*).TAB', file)
                epo_in = result.group(1)
                print(epo_in)
                # 4th argument ('BELASCIRDR') unused?
                pygeoloc_in.append([epo_in, indir_in, outdir_in, pattern, 0, XovOpt.to_dict()])
                print(pygeoloc_in)
            jobs = executor.map_array(PyGeoloc.main, pygeoloc_in)
            for job in jobs:
                print(job.result())
        else:
            # 4th argument ('BELASCIRDR') unused?
            PyGeoloc.main([f'{monyea}', indir_in, outdir_in, pattern, 0, XovOpt.to_dict()])

executor.update_parameters( #slurm_account='j1010',
                           slurm_name="pyxover",
                           slurm_mem='30G',
                           slurm_cpus_per_task=4,
                           slurm_array_parallelism=12)

XovOpt.set("weekly_sets", True)
XovOpt.set("monthly_sets",True)
# # crossovers location step
if run_pyXover:
    XovOpt.set("parallel", True)  # not sure why, but parallel gets crazy
    if grid: # WD: change MLASIMRDR?
        job = executor.submit(PyXover.main, ['0', f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_',
                   f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/', 'MLASIMRDR', 0,
                   XovOpt.to_dict()]) # single job
        print(job.result())
    else:
        PyXover.main(['0', f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_',
           f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/', 'MLASIMRDR', 0,
           XovOpt.to_dict()])


# # lsqr solution step
if run_accuXover:
    out = AccumXov.main(
         [[f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'], 'sim', 0,
          XovOpt.to_dict()])