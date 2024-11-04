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
import datetime as dt
import math
import itertools as itert

# update paths and check options
from pyaltsim import PyAltSim

grid = True
run_pyAltSim  = False
run_pyGeoLoc  = True
run_pyXover   = True
run_accuXover = False

camp = "/storage/research/aiub_gravdet/WD_BELA/"
OrbDir = f"{camp}ORB/"

# AA0: Simulation no noise, DEM, small scale, 10 Hz

simid = 'AA0'
estid_N = 'AA3'
estid_S = 'AA2'
XovOpt.set("selected_hemisphere",'N')
XovOpt.set("range_noise_mean_std",[0.,2.])

if XovOpt.get("selected_hemisphere") == 'N':
   estid = estid_N
else:
   estid = estid_S

XovOpt.set("body", 'MERCURY')
XovOpt.set("spice_meta", 'mymeta')
# XovOpt.set("basedir", 'data/')
XovOpt.set("basedir", f'{camp}pyXover/')
XovOpt.set("instrument", 'BELA')
XovOpt.set("debug", False)
XovOpt.set("compute_input_xov", True) # Test
# XovOpt.set("compute_input_xov", False)
XovOpt.set("msrm_sampl", 20)


XovOpt.set("sol4_orb", [])
XovOpt.set("sol4_orbpar", [None])
XovOpt.set("sol4_glo", ['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dL', 'dR/dh2'])

vecopts = {'SCID': '-121',
           'SCNAME': 'MPO',
           'SCFRAME': -121000,
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

# XovOpt.set("pert_cloop_orb", {'dA':10, 'dC':2, 'dR':0.02}) # Thor+2020
XovOpt.set("par_constr",
           {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 3.e-1, 'dR/dA': 1.e2,
            'dR/dC': 1.e2, 'dR/dR': 2.e1})  # , 'dR/dRl':5.e1, 'dR/dPt':5.e1} #
XovOpt.set("mean_constr", {'dR/dA': 1.e0, 'dR/dC': 1.e0, 'dR/dR': 1.e0})

XovOpt.set("expopt", simid)
XovOpt.set("resopt", 3)
XovOpt.set("amplopt", 20)
XovOpt.set("SpInterp", 0)

XovOpt.check_consistency()
AccOpt.check_consistency()

if XovOpt.get("SpInterp") == 0 and False:
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


# d_start = dt.datetime(2025,12,1,0,0,0)
d_start = dt.datetime(2026,3,16,0,0,0)
# d_start = dt.datetime(2026,11,30,0,0,0)
# d_end   = dt.datetime(2026,12,1,0,0,0)
d_end   = dt.datetime(2027,3,16,0,0,0)
nWeeks = math.floor((d_end - d_start).days/7)
nWeeks = 2

XovOpt.set("parallel", False)
XovOpt.set("apply_topo", True)
XovOpt.set("small_scale_topo", True)
XovOpt.set("range_noise", False)
XovOpt.set("new_illumNG", True)
XovOpt.set("unittest", False) # this restricts simulated data to the first day of the month (see d_last in PyAltSim.main)

if grid:
   # executor is the submission interface (logs are dumped in the folder)
   # executor = submitit.AutoExecutor(folder="log_slurm", cluster='local')
   executor = submitit.AutoExecutor(folder="log_slurm")
   # set timeout in min, and partition for running the job
   executor.update_parameters(slurm_partition="aiub",#epyc2, aiub
                           slurm_cpus_per_task=1,
                           slurm_nodes=1,
                           slurm_time=60*99, # minutes
                           slurm_mem='5G',
                           slurm_array_parallelism=100)

# PYALTSIM
# --------
if run_pyAltSim:
    XovOpt.set("sim_altdata", True)
    XovOpt.set("partials", False)
    XovOpt.set("expopt", simid)

    # generate a few BELA test data
    pyaltsim_in = []
    print(f'Simulatation of BELA data for {nWeeks} weeks, from {d_start} to {d_end}')
    for w in range(0,nWeeks):
       monyea = (d_start + dt.timedelta(weeks=w)).strftime('%y%m%d')    
       # indir_in = f'SIM_{monyea[:2]}/{XovOpt.get("expopt")}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
       indir_in = f'SIM_{monyea}/{XovOpt.get("expopt")}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
       pyaltsim_in.append([XovOpt.get("amplopt"), XovOpt.get("resopt"), indir_in,
                           d_start + dt.timedelta(weeks=w),
                           d_start + dt.timedelta(weeks=w+1),
                           XovOpt.to_dict()])
    if grid:
       executor.update_parameters(slurm_name="pyaltsim",
                                   slurm_array_parallelism=60,
                                   slurm_time=60*3, # minutes
                                   slurm_mem='10G') # 4GB for 10Hz
       if len(pyaltsim_in) == 1:
          job = executor.submit(PyAltSim.main, pyaltsim_in[0]) # single job
          print(job.result())
       else:
          jobs = executor.map_array(PyAltSim.main, pyaltsim_in)
          for job in jobs:
             print(job.result())
    else:
       for arg in pyaltsim_in:
          PyAltSim.main(arg)
          
if run_pyGeoLoc or run_pyXover or run_accuXover:
    XovOpt.set("sim_altdata", False)
    XovOpt.set("partials", True)
    XovOpt.set("parallel", False)
    XovOpt.set("SpInterp", 0)
    XovOpt.set("expopt", estid)

# PYGEOLOC (geolocation step)
# --------
if run_pyGeoLoc:
    if grid:
        executor.update_parameters(slurm_name="pygeoloc",
                                   slurm_mem='1G',
                                   slurm_cpus_per_task=1,
                                   slurm_time=20) # minutes
    for w in range(0,nWeeks):
        import glob
        monyea = (d_start + dt.timedelta(weeks=w)).strftime('%y%m%d')
        indir_in = f'SIM_{monyea}/{simid}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
        outdir_in = f'{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_{monyea}'
        pygeoloc_in = []
        epo_in = ""
        allFiles = glob.glob(os.path.join(f'{XovOpt.get("rawdir")}{indir_in}', f'{XovOpt.get("instrument")}*RDR*.*'))
        # Retrieve the dates from the names of all the files in the directory
        d_files = [dt.datetime.strptime(fil.split('.')[0][-10:], '%y%m%d%H%M')  for fil in allFiles[:]]
        d_files = list(set(d_files))
        d_files.sort()
        pattern = 'BELASCIRDR'
        pygeoloc_in = []
        for j in range(0,len(d_files)-1): # WD: You should think of // processing within one run
           pygeoloc_in.append([epo_in, indir_in, outdir_in, [d_files[j], d_files[j+1]], 0, XovOpt.to_dict()]) # correct way
           # pygeoloc_in.append([f'{monyea}', indir_in, outdir_in, pattern, 0, XovOpt.to_dict()]) # not working, adapt from above
        if grid:
           if len(pygeoloc_in) == 1:
              job = executor.submit(PyGeoloc.main, pygeoloc_in[0]) # single job
              print(job.result())
           else:
              jobs = executor.map_array(PyGeoloc.main, pygeoloc_in)
              for job in jobs:
                 print(job.results())
        else:
           PyGeoloc.main(pygeoloc_in[0])
        # else:
        #     # 4th argument ('BELASCIRDR') unused?
        #     PyGeoloc.main([f'{monyea}', indir_in, outdir_in, d_files, 0, XovOpt.to_dict()])

# PYXOVER
# -------
if run_pyXover:
    XovOpt.set("parallel", False)  # not sure why, but parallel gets crazy
    XovOpt.set("weekly_sets", True)
    XovOpt.set("monthly_sets", False)
    indir_in =  f'{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_'
    outdir_in = f'{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
    
    misy = [(d_start + dt.timedelta(weeks=w)).strftime('%y%m%d') for w in range(0,nWeeks)]
    misycmb = [x for x in itert.combinations_with_replacement(misy, 2)]
    print("Choose grid element among:", dict(map(reversed, enumerate(misycmb))))
    
    if grid:
       executor.update_parameters(slurm_name="pyxover",
                                  slurm_mem='10G',
                                  slurm_cpus_per_task=1,
                                  slurm_time=60*5, # minutes
                                  slurm_array_parallelism=100)
       
       pyxover_in = []
       for par in range(0,len(misycmb)):
          # create symlink to rough xovs from other tests
          input_xov_path = XovOpt.get("outdir") + outdir_in + 'xov/xov_' + str(misycmb[par][0]) + '_' + str(misycmb[par][1]) + '.pkl'
          if os.path.exists(input_xov_path):
             print("input xov file already exists in", input_xov_path)
          else:
             pyxover_in.append([f'{par}',indir_in, outdir_in, misycmb[par], 0,XovOpt.to_dict()])
       print(f'{len(pyxover_in)} combinations to process')
       if(len(pyxover_in) == 1):
          job = executor.submit(PyXover.main, pyxover_in[0]) # single job
          print(job.result())
       else:
          jobs = executor.map_array(PyXover.main, pyxover_in)
          for job in jobs:
             print(job.results())
    else:
       par = 0
       PyXover.main([f'{par}',indir_in, outdir_in, misycmb[par], 0, XovOpt.to_dict()])

    # Might be nicer to form a array as below
    # pyxover_in = [[comb, f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_',
    #            f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/',
    #            'MLASIMRDR', 0, XovOpt.to_dict()]
    #            # for comb in np.arange(33,78)]
    #             for comb in [33, 41, 77]]  # np.arange(1)]

# ACCUMXOV
# --------
if run_accuXover:
   out = AccumXov.main(
       [[f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'], 'sim', 0,
      XovOpt.to_dict()])
