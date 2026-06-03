import os
import numpy as np
import sys

import submitit
from accumxov.accum_opt import AccOpt
from config import XovOpt

from accumxov import AccumXov
from pygeoloc import PyGeoloc
from pyxover import PyXover
import datetime as dt
import math
import itertools as itert

from pyaltsim import PyAltSim
from xovutil.units import deg2as

grid = True
run_pyAltSim  = False
run_pyGeoLoc  = False
run_pyXover   = False
run_accuXover = True

basedir = "/home/wdesprat/nobackup/pyxover/examples/BELA/data/"
log_folder = f"{basedir}log/"

# removed below
# AA0: Simulation 2m noise, DEM, small scale, 10 Hz
# AA4: gtracks and xov North from AA0 w/o perturbations
# AA5: gtracks and xov South from AA0 w/o perturbations
# AA6: gtracks and xov North from AA0 w/o perturbations ?
# AA7: gtracks and xov South from AA0 w/o perturbations ?
# AA8: gtracks and xov North from AA0 w/ perturbations
# AA9: gtracks and xov South from AA0 w/ perturbations
# h2 = 0.95
# AB0: Simulation 0.2m noise, no DEM, small scale, 10 Hz
# AB0: gtracks and xov North from AB0 w/o perturbations
# AB1: gtracks and xov South from AB0 w/o perturbations
# AB2: Simulation 0.2m noise, DEM, small scale, 10 Hz
# AB2: gtracks and xov North from AB2 w/o perturbations
# AB3/AC3: gtracks and xov South from AB2 w/o perturbations
# AB4: gtracks and xov North from AB2: {'dA':10, 'dC':5, 'dR':0.02}
# AB5: gtracks and xov South from AB2: {'dA':10, 'dC':5, 'dR':0.02}
# AB6: gtracks and xov North from AB2: {'dA':10, 'dC':5, 'dR':0.02, 'dRl':5, 'dPt':5}
# AB7: gtracks and xov South from AB2: {'dA':10, 'dC':5, 'dR':0.02, 'dRl':5, 'dPt':5}
# AB8: gtracks and xov North from AB2: {'dA':10, 'dC':5, 'dR':0.02, 'dRl':2, 'dPt':2}
# AB9: gtracks and xov South from AB2: {'dA':10, 'dC':5, 'dR':0.02, 'dRl':2, 'dPt':2}
# AC0: gtracks and xov North from AB2: {'dA':10, 'dC':5, 'dR':0.02, 'dRl':20, 'dPt':20} (q01-l80)
# AC1/AC2: gtracks and xov South from AB2: {'dA':10, 'dC':5, 'dR':0.02, 'dRl':20, 'dPt':20}
# AC4: gtracks and xov North from AB2: {'dRl':20, 'dPt':20}
# AC5: gtracks and xov South from AB2: {'dRl':20, 'dPt':20}
# AD0: Simulation noise model, DEM, small scale, 10 Hz probably with {'dRl': 20, 'dPt': 20} ...
# AD0: gtracks and xov South from AD0: {'dA':10, 'dC':5, 'dR':0.02} incorrect close loop sim?
# AD1: gtracks and xov North from AD0: {'dA':10, 'dC':5, 'dR':0.02} incorrect close loop sim?
# AD2: gtracks and xov North from AD0: {'dA':10, 'dC':5, 'dR':0.02} (dR/dR needs to be constrained ...)
#      A:'dR/dh2': 1.e-5
#      B:'dR/dh2': 1.e-5, 'dR/dR': 1e-1
#      C:'dR/dh2': 1.e-5, 'dR/dR': 1
#      D:'dR/dh2': 1.e-5, 'dR/dR': 0.5
#      E:'dR/dh2': 1.e-5, 'dR/dR': 0.2
#      F:'dR/dh2': 1.e-5, 'dR/dA': 20,'dR/dR': 0.2
#      G:'dR/dh2': 1.e-5, 'dR/dA': 10,'dR/dR': 0.2
#      H:'dR/dh2': 1.e-5, 'dR/dA': 1,'dR/dR': 0.2
#      I:'dR/dh2': 3.e-1, 'dR/dA': 10,'dR/dR': 0.2
#      J:'dR/dh2': 3.e-1, 'dR/dA': 10,'dR/dA': 5,'dR/dR': 0.2
#      K:'dR/dh2': 1, 'dR/dA': 10,'dR/dA': 5,'dR/dR': 0.2
#      L:'dR/dh2': 1, 'dR/dA': 10,'dR/dA': 5,'dR/dR': 0.2 (extended)
# AD3: gtracks and xov South from AD0: {'dA':10, 'dC':5, 'dR':0.02}
# AD4: Simulation noise model, DEM, small scale, 10 Hz test
# AD4: gtracks and xov North from AD4: {'dA':10, 'dC':5, 'dR':0.02}
#     A: extended
#     B: nominal
#     C: nominal VCE weight = [1 1 1]
#     D: extended VCE weight = [1 1 1]
# AD5: gtracks and xov South from AD4: {'dA':10, 'dC':5, 'dR':0.02}
# AD6: Simulation noise model, DEM, small scale, 10 Hz , PM_ORIGIN: J2000
# AD6: gtracks and xov North from AD6: {'dA':10, 'dC':5, 'dR':0.02} most likely pertubation not applied
# AD7: gtracks and xov South from AD6: {'dA':10, 'dC':5, 'dR':0.02} most likely pertubation not applied

# DA0: Simulation from DLR, Northern hemisphere
# DA1: Simulation from DLR, Southern hemisphere
# DA0: gtracks and xov North from DA0
# DA1: gtracks and xov South from DA1
# redo for testing
# DA2: gtracks and xov North from DA0
# DA3: gtracks and xov South from DA1
# PM_ORIGIN: J2000
# DA4: gtracks and xov North from DA0
# DA5: gtracks and xov South from DA1
# DA6: gtracks and xov North from DA0, 1050km cutoff
# DB0: Simulation from DLRv3, Northern hemisphere
# DB0: gtracks and xov North from DB0

# AD8: Simulation no noise, DEM, 10 Hz, PM_ORIGIN: J2000
# AD8: gtracks and xov North from AD9 h2=0
# AD8: Simulation no noise, DEM, 10 Hz, PM_ORIGIN: J2000

# AE0: gtracks and xov North from AD6: {'dA':10, 'dC':5, 'dR':0.02}
# C: VCE
# D: no VCE
# E: no VCE nominal
# F: VCE nominal
# G: no VCE nominal, 85° threshold
# H: VCE nominal, 85° threshold
# AE1: gtracks and xov South from AD6: {'dA':10, 'dC':5, 'dR':0.02}
# correct outliers and altitude issue
# AE2: gtracks and xov North from AD6: {'dA':10, 'dC':5, 'dR':0.02}
# A: VCE ext
# C: VCE nom
# D: VCE nom, 80% obs
# E: VCE nom, 80% at hilat>89°
# F: VCE nom, 2 lat blocks 88deg
# G: VCE nom, 2 lat blocks 88deg, 80% obs
# H: VCE nom, 2 lat blocks 88deg, 80% at hilat>89°
# I: VCE nom, 80% at hilat>88°
# J: VCE nom, 50% at hilat>88°
# K: VCE nom, 10% at hilat>88°
# L: VCE nom, 1% at hilat>88°
# M: VCE nom, 20% at hilat>88°
# AE3: gtracks and xov South from AD6: {'dA':10, 'dC':5, 'dR':0.02}
# AE4: gtracks and xov North from AD6: {'dA':10, 'dC':5, 'dR':0.02} dLIB1-11
# A: VCE ext, 2 lat blocks 88deg, max_xovers = 8.5e5 at hilat>88°
# B: VCE ext
# AE5: gtracks and xov South from AD6: {'dA':10, 'dC':5, 'dR':0.02} dLIB1-11
# AE6: gtracks and xov North from AD6: {'dA':10, 'dC':5, 'dR':0.02, 'dRl':3, 'dPt':3} dLIB1-11
# AE7: gtracks and xov South from AD6: {'dA':10, 'dC':5, 'dR':0.02, 'dRl':3, 'dPt':3} dLIB1-11
# AE8: gtracks and xov North from AD6: {'dA':10, 'dC':5, 'dR':0.02, 'dRl':3, 'dPt':3} dLIB1-11 wrong h2
# AE9: gtracks and xov South from AD6: {'dA':10, 'dC':5, 'dR':0.02, 'dRl':3, 'dPt':3} dLIB1-11 wrong h2
# AF0: gtracks and xov North from AD6: {'dA':10, 'dC':5, 'dR':0.02, 'dRl':3, 'dPt':3} dLIB1-5 wrong h2
# AF1: gtracks and xov South from AD6: {'dA':10, 'dC':5, 'dR':0.02, 'dRl':3, 'dPt':3} dLIB1-5 wrong h2


simid = 'AD6'
estid_N = 'AF0'
estid_S = 'AF1'
iter = 0
XovOpt.set("selected_hemisphere",'N')

max_job = 1500
max_parallel = 200   
max_parallel = 12*10  

if XovOpt.get("selected_hemisphere") == 'N':
   estid = estid_N
else:
   estid = estid_S
   

d_start = dt.datetime(2027,4,1,0,0,0)
# d_end   = dt.datetime(2028,4,1,0,0,0) # nominal mission
d_end   = dt.datetime(2029,4,1,0,0,0) # extended mission
nWeeks = math.floor((d_end - d_start).days/7)
nMonths = math.ceil((d_end - d_start).days/30)
   
# General options
XovOpt.set("body", 'MERCURY')
XovOpt.set("spice_meta", 'mymeta_MPO')
XovOpt.set("basedir", basedir)
XovOpt.set("instrument", 'BELA')
XovOpt.set("max_range_altitude", 1050)
XovOpt.set("SpInterp", 0)

vecopts = {'SCID': '-121',
           'SCNAME': 'MPO',
           'SCFRAME': 'MPO_SPACECRAFT', # '-121000'
           'INSTID': (-121102, -121101),
           'INSTNAME': ('MPO_BELA_TRANSMITTER', 'MPO_BELA_RECEIVER'),
           'PLANETID': '199',
           'PLANETNAME': 'MERCURY',
           'PLANETRADIUS': 2440., # 2439400 accordidng to DEM
           'PLANETFRAME': 'IAU_MERCURY',
           'OUTPUTTYPE': 1,
           'ALTIM_BORESIGHT': '',
           'INERTIALFRAME': 'J2000',
           'INERTIALCENTER': 'SSB',
           'PM_ORIGIN': 'J2000', #'J2013.0',
           'PARTDER': ''}
XovOpt.set("vecopts", vecopts)
# XovOpt.get("vecopts")['ALTIM_BORESIGHT'] = [0., 0., 1.]  # was like this in pyGeoloc/PyXover for BELA

# PYALTSIM
# --------
# PySimAlt options
if run_pyAltSim:
   XovOpt.set("sampling_rate",10) # [Hz]
   XovOpt.set("partials", False)
   XovOpt.set("expopt", simid)
   XovOpt.set("resopt", 3)
   XovOpt.set("amplopt", 20)
   XovOpt.set("apply_topo", True)
   XovOpt.set("small_scale_topo", True)
   XovOpt.set("range_noise", False)
   XovOpt.set("range_noise_mean_std",[0.,0.2])
   XovOpt.set("new_illumNG", True)
   XovOpt.check_consistency()

   # generate a few BELA test data
   pyaltsim_in = []
   print(f'Simulatation of BELA data for {nWeeks} weeks, from {d_start} to {d_end}')
   for w in range(0,nWeeks):
      monyea = (d_start + dt.timedelta(weeks=w)).strftime('%y%m')    
      indir_in = f'SIM_{monyea}/{XovOpt.get("expopt")}/'
      pyaltsim_in.append([XovOpt.get("amplopt"), XovOpt.get("resopt"), indir_in,
                          d_start + dt.timedelta(weeks=w),
                          d_start + dt.timedelta(weeks=w+1),
                          XovOpt.to_dict()])
   # pyaltsim_in = [pyaltsim_in[0]]
   if grid:
      executor = submitit.AutoExecutor(folder=f'{log_folder}{simid}/pyaltsim')
      executor.update_parameters(slurm_cpus_per_task=1,
                                 slurm_nodes=1,
                                 slurm_name="pyaltsim",
                                 slurm_array_parallelism=max_parallel,
                                 slurm_time=60*3, # minutes
                                 slurm_mem='10G') # 4GB for 10Hz
      if len(pyaltsim_in) == 1:
         job = executor.submit(PyAltSim.main, pyaltsim_in[0]) # single job
         (job.result())
      else:
         jobs = executor.map_array(PyAltSim.main, pyaltsim_in)
         for job in jobs:
            job.result()
   else:
      for arg in pyaltsim_in:
         PyAltSim.main(arg)
          
if not (run_pyGeoLoc or run_pyXover or run_accuXover):
   sys.exit()

XovOpt.set("partials", True)
XovOpt.set("expopt", estid)

# pyGeoloc options
if run_pyGeoLoc:
   # XovOpt.set("pert_cloop_orb", {'dA':10, 'dC':2, 'dR':0.02}) # Thor+2020
   # XovOpt.set("pert_cloop_orb", {'dA':10, 'dC':5, 'dR':0.02}) # Imperi+2018 (max values)
   # XovOpt.set("pert_cloop_orb", {'dA':10, 'dC':5, 'dR':0.02, 'dRl':5, 'dPt':5}) # Imperi+2018 (max values)
   XovOpt.set("pert_cloop_orb", {'dA':10, 'dC':5, 'dR':0.02, 'dRl':3, 'dPt':3}) # Imperi+2018 (max values)
   # XovOpt.set("pert_cloop_orb", {'dA':10, 'dC':5, 'dR':0.02, 'dRl':20, 'dPt':20}) # Imperi+2018 (max values)
   
if run_pyGeoLoc or run_pyXover:
   perturbations = {
      'dRA':  [0.2, 0.000, 0.000],
      'dDEC': [0.36, 0.000, 0.000],
      'dPM':  [0, 0.013, 0.000],
      'dL':   1.e-3 * deg2as(1.) * np.linalg.norm([
         0.00993822, -0.00104581, -0.00010280, -0.00002364, -0.00000532
         ]),
      'dh2':  0.1
      }
   # Add dLIB1–dLIB11
   perturbations.update({f'dLIB{i}': 1.e-3 * deg2as(1.) for i in range(1, 12)})
   XovOpt.set("parGlo", perturbations)
   
# pyXover options
if run_pyXover:
   XovOpt.set("compute_input_xov", True)
   # XovOpt.set("import_proj", True)
   XovOpt.set("msrm_sampl", 20)
   XovOpt.set("n_interp",6)
   XovOpt.set("weekly_sets", False)
   XovOpt.set("monthly_sets", True)

# AccumXov options
if run_accuXover:
   XovOpt.set("sol4_orb", [])
   XovOpt.set("sol4_orbpar", ['dA','dC','dR'])
   # XovOpt.set("sol4_orbpar", [None])
   params = ['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dL', 'dR/dh2']
   # params = ['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dh2']
   # params += [f'dR/dLIB{i}' for i in range(1, 12)]
   XovOpt.set("sol4_glo", params)

   # 'dR/dh2': 3.e-1
   # XovOpt.set("par_constr",
   #            {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 3.e-1, 'dR/dA': 1.e2,
   #             'dR/dC': 1.e2, 'dR/dR': 2.e1})  # , 'dR/dRl':5.e1, 'dR/dPt':5.e1} #
   XovOpt.set("par_constr",
            {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 1, 'dR/dA': 10,
               'dR/dC': 5, 'dR/dR': 0.2})  # , 'dR/dRl':5.e1, 'dR/dPt':5.e1} #
   # XovOpt.set("mean_constr", {'dR/dA': 1.e0, 'dR/dC': 1.e0, 'dR/dR': 1.e0})
   XovOpt.set("par_constr",
            {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 1,
               '2.*dR/dA': 10, '2.*dR/dC': 5, '2.*dR/dR': 0.2})
   XovOpt.set("mean_constr", {})

   AccOpt.set("downsize", False) 

   AccOpt.set("Abmat_outfile", f"Abmat_{estid_N}_{iter}_{iter+1}_B.pkl")
   # AccOpt.set("compute_vce",False)
   AccOpt.set("compute_vce",True)
   # AccOpt.set("Abmat_infile", f"Abmat_{estid_N}_{iter}_{iter+1}_nosol.pkl")
   AccOpt.set("weight_obs",[1])
   # AccOpt.set("weight_obs",[1, 1])
   AccOpt.set("weight_constr",[1])
   AccOpt.set("solving_method","cholesky")
   AccOpt.set("apply_xov_cov_tracks",False)

XovOpt.check_consistency()

# PYGEOLOC (geolocation step)
# --------
if run_pyGeoLoc:
   if grid:
      executor = submitit.AutoExecutor(folder=f'{log_folder}{estid}/pygeoloc')
      executor.update_parameters(slurm_nodes=1,
                                 slurm_array_parallelism=max_parallel,
                                 slurm_name="pygeoloc",
                                 slurm_mem='1G',
                                 slurm_cpus_per_task=1,
                                 slurm_time=20) # minutes
      if iter>0:
         executor.update_parameters(slurm_mem='20G')
   pygeoloc_in = []
   for m in range(0,nMonths):
   # for m in range(0,10):
   # for m in range(10,nMonths):
   # for m in [7,22]:
      import glob
      monyea = (d_start + dt.timedelta(days=m*30)).strftime('%y%m')
      indir_in = f'SIM_{monyea}/{simid}/'
      outdir_in = f'{XovOpt.get("expopt")}_{iter}/gtrack_{monyea}'
      epo_in = ""
      allFiles = glob.glob(os.path.join(f'{XovOpt.get("rawdir")}{indir_in}', f'{XovOpt.get("instrument")}*RDR*.*'))
      # Retrieve the dates from the names of all the files in the directory
      d_files = [dt.datetime.strptime(fil.split('.')[0][-10:], '%y%m%d%H%M')  for fil in allFiles[:]]
      d_files = list(set(d_files))
      d_files.sort()
      # d_files = d_files[152:154]
      nt = 9
      pygeoloc_in.extend([[epo_in, indir_in, outdir_in, d_files[i*nt:min((i+1)*nt,len(d_files))+1], iter, XovOpt.to_dict()] for i in range(0,math.ceil(len(d_files)/nt))])
      pattern = 'BELASCIRDR'
   # pygeoloc_in = [pygeoloc_in[0]]
   if grid:
      if len(pygeoloc_in) == 1:
         job = executor.submit(PyGeoloc.main, pygeoloc_in[0]) # single job
         print(job.result())
      else:
         # Launch max_job jobs in //
         i_j = 0
         while i_j < len(pygeoloc_in):
            ip_j = i_j + max_job
            if ip_j>len(pygeoloc_in):
               ip_j = len(pygeoloc_in)
            jobs = executor.map_array(PyGeoloc.main, pygeoloc_in[i_j:ip_j+1])
            i_j = ip_j + 1
            for job in jobs:
               job.result()
   else:
      PyGeoloc.main(pygeoloc_in[0])

# PYXOVER
# -------
if run_pyXover:
   indir_in =  f'{XovOpt.get("expopt")}_{iter}/gtrack_'
   outdir_in = f'{XovOpt.get("expopt")}_{iter}/'
   xov_dir = XovOpt.get("outdir") + outdir_in + 'xov/'
    
   misy = [(d_start + dt.timedelta(days=30*w)).strftime('%y%m') for w in range(0,nMonths)]
   misycmb = [x for x in itert.combinations_with_replacement(misy, 2)]
   print("Choose grid element among:", dict(map(reversed, enumerate(misycmb))))
    
   if grid:
      executor = submitit.AutoExecutor(folder=f'{log_folder}{estid}/pyxover')
      executor.update_parameters(slurm_name="pyxover",
                                 slurm_nodes=1,
                                 slurm_mem='25G', # 11G
                                 slurm_cpus_per_task=2,
                                 slurm_time=60*5, # minutes
                                 slurm_array_parallelism=max_parallel)
       
   pyxover_in = []
   for par in range(0,len(misycmb)):
   # for par in range(7, len(misycmb)):
   # for par in range(14, 48):
   # for par in [6]:
      # create symlink to rough xovs from other tests
      input_xov_path = xov_dir + 'xov_' + str(misycmb[par][0]) + '_' + str(misycmb[par][1]) + '.pkl'
      rough_xov_path = xov_dir + 'tmp/xovin_' +  str(misycmb[par][0]) + '_' + str(misycmb[par][1]) + '.pkl.gz'
      if os.path.exists(input_xov_path):
         print("input xov file already exists in", input_xov_path)
      else:
         if XovOpt.get('compute_input_xov') and os.path.exists(rough_xov_path):
            print("rough xov file already exists in", rough_xov_path)
         elif not XovOpt.get('compute_input_xov') and not os.path.exists(rough_xov_path):
            print("input rough xov file does not exists in", rough_xov_path)
         else:
            gtrack_dirs = [os.path.join(XovOpt.get("outdir"), indir_in + par1 ) for par1 in misycmb[par]]
            pyxover_in.append([f'{par}',gtrack_dirs, outdir_in, misycmb[par], iter,XovOpt.to_dict()])
            # pyxover_in.append([f'{par}',indir_in, outdir_in, misycmb[par], iter,XovOpt.to_dict()])
   # pyxover_in = pyxover_in[:1]
   print(f'{len(pyxover_in)} combinations to process')
   if grid:
      if(len(pyxover_in) == 1):
         job = executor.submit(PyXover.main, pyxover_in[0]) # single job
         print(job.result())
      else:
         jobs = executor.map_array(PyXover.main, pyxover_in)
         for job in jobs:
            job.result()
   else:
      PyXover.main(pyxover_in[0])

    # Might be nicer to form a array as below
    # pyxover_in = [[comb, f'sim/{XovOpt.get("expopt")}_{iter}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_',
    #            f'sim/{XovOpt.get("expopt")}_{iter}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/',
    #            'MLASIMRDR', 0, XovOpt.to_dict()]
    #            # for comb in np.arange(33,78)]
    #             for comb in [33, 41, 77]]  # np.arange(1)]

# ACCUMXOV
# --------
if run_accuXover:
   datasets = [f'{estid_N}_{iter}/',
              f'{estid_S}_{iter}/']
   # datasets = [f'{estid_N}_{iter}/']
   # XovOpt.set("sol4_glo", [None])
   # XovOpt.set("sol4_orbpar", [None])
   if grid:
      executor = submitit.AutoExecutor(folder=f'{log_folder}{estid}/accumxov')
      executor.update_parameters(slurm_nodes=1,
                                 slurm_name="accumXov",
                                 slurm_mem='100G',
                                 slurm_cpus_per_task=1,
                                 slurm_time=60*12, # minutes
                                 slurm_array_parallelism=max_parallel)
      job = executor.submit(AccumXov.main, [datasets, '', iter, XovOpt.to_dict(), AccOpt.to_dict()]) # single job
      print(job.result())
   else:
      out = AccumXov.main([datasets, '', iter, XovOpt.to_dict(), AccOpt.to_dict()])
    
   # out = AccumXov.main(
   #    [[f'sim/{XovOpt.get("expopt")}_{iter}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'], 'sim', 0,
   #   XovOpt.to_dict()])