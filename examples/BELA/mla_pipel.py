import os
import numpy as np
import sys

import submitit
from accumxov.accum_opt import AccOpt
from config import XovOpt
import glob

from accumxov import AccumXov
from accumxov.Amat import Amat
from pygeoloc import PyGeoloc
from pyxover import PyXover
import datetime as dt
import math
import itertools as itert
import re
import time

# update paths and check options
from pyaltsim import PyAltSim
from xovutil.units import deg2as

grid = True
run_pyAltSim  = False # rather fast: up slightly more than 2GB for some days
run_pyGeoLoc  = True
run_pyXover   = True
run_accuXover = True

basedir = "/home/wdesprat/nobackup/pyxover/examples/MLA/data/"
log_folder = f"{basedir}log/"

# os.chdir('examples/BELA')

# CA0: Simulation 0.2m noise, no DEM, small scale, 10 Hz
# CA0: gtracks and xov North from CA0 w/o perturbations
# CA1: Simulation 0.2m noise, DEM, small scale, 10 Hz. A few arcs might be missing because of bookkeeping when more than max_jobs
# CA1: gtracks and xov North from CA1 w/o perturbations
# CA2: gtracks and xov North from CA1: 'dA':50., 'dC':50., 'dR':20., 'dRl':0.5, 'dPt':0.5, 'dA1':40., 'dC1':40., 'dR1':10.
# CA3: gtracks and xov North from CA1: 'dA':50., 'dC':50., 'dR':20., 'dRl':0.5, 'dPt':0.5
# CA4: gtracks and xov North from CA1: 'dA':50., 'dC':50., 'dR':20., 'dRl':20, 'dPt':20
# CA5: gtracks and xov North from CA1: 'dR':20., 'dRl':20, 'dPt':20
# CB0: Simulation 0.25m noise, DEM, small scale, 8 Hz. (what perturbation did you add ?? most likely {'dA': 20.0, 'dC': 20.0, 'dR': 5.0})
# CB0: gtracks and xov North from CB0: 'dA':20., 'dC':20., 'dR':5.
# CB1: gtracks and xov North from CB0: 'dRl':20, 'dPt':20
# CB2: gtracks and xov North from CB0: 'dA':20., 'dC':20., 'dR':5., 'dRl':20, 'dPt':20
#     A: 'dR/dh2': 3e-1 (harcoded for BELA tracks)
#     B: 'dR/dh2': 1 (harcoded for BELA tracks)
#     C: 'dR/dh2': 1, 'dR/dA': 20, 'dR/dC': 20, 'dR/dR': 5 (then h2 true error is worse ...)
#     D: 'dR/dh2': 1e-5, 'dR/dA': 20, 'dR/dC': 20, 'dR/dR': 5
# CB3: Simulation 0.25m noise, DEM, small scale, 8 Hz. test
# CB3: gtracks and xov North from CB3: {'dA':20., 'dC':20., 'dR':5.,'dRl':20, 'dPt':20}
#     A:
#     B: VCE weight = [1 1 1]
# CB4: gtracks and xov North from CB3: {'dA':20., 'dC':20., 'dR':5.}
#     A:
#     B: VCE
#     C: VCE weight = [1 1 1]
# CB5: Simulation 0.25m noise, DEM, small scale, 8 Hz, PM_ORIGIN: J2000
# CB5: gtracks and xov North from CB5: {'dA':20., 'dC':20., 'dR':5.} (most likely no perturbation applied)
#     C: VCE weight = [1 1 1]
#     D: VCE weight = [1 1 1] w/ avg
# CB6: gtracks and xov North from CB5: {'dA':20., 'dC':20., 'dR':5.}
#     A: VCE
#     B: VCE
#     C: VCE
#     D: VCE: estimate constr on globals
#     E: VCE: don't estimate constr on globals
#     F: no VCE
# CB7: gtracks and xov North from CB5: no perturbations
# CB8: Simulation no noise, no DEM, no small scale, 8 Hz, PM_ORIGIN: J2000
# CB9: Simulation 0.25m noise, DEM, small scale, 8 Hz, PM_ORIGIN: J2000, correct tides
# correct outliers and altitude issue
# CB9_old: gtracks and xov North from CB5: {'dA':20., 'dC':20., 'dR':5.}
#     A : 28074095
#     C : 24512432
# CB9: gtracks and xov North from CB9: {'dA':20., 'dC':20., 'dR':5.}

# CC0: real data from pds
# CC0: gtracks and xov North from CC0
# CC1: gtracks and xov North from MLA
# A: VCE
# B: VCE, no h2
# C: VCE, no h2, "apply_xov_cov_tracks":True
# CC2: gtracks and xov North from MLA (up to iter=4)
# A: copied from CC1_C
# CC3: gtracks and xov North from CB9 dLIB5
# CC4: gtracks and xov North from CB9 dLIB1-11
# CC5: gtracks and xov North from MLA
# A: VCE, no h2, "apply_xov_cov_tracks":False
# B: VCE, no h2, "apply_xov_cov_tracks":True
# C: VCE, no h2, "apply_xov_cov_tracks":True, maxR=1000m
# D: VCE, no h2, "apply_xov_cov_tracks":True, maxR=1000m, constraints on globals
# F: VCE, no h2, "apply_xov_cov_tracks":True, maxR=1000m, (ACR) (100,100,20)
# E: VCE, no h2, "apply_xov_cov_tracks":True, maxR=1000m, (ACR) (100,100,20), 2 lat blocks 80deg
# G: VCE, no h2, "apply_xov_cov_tracks":True, maxR=1000m, (ACR) (100,100,20), mean_constr=1m
# H: VCE, no h2, "apply_xov_cov_tracks":True, maxR=1000m, (ACR) (100,100,20), mean_constr=1m max_xovers@lat>80=8.5e5
# CC6: gtracks and xov North from MLA
# A: VCE, no h2, "apply_xov_cov_tracks":True copied from CC5_B
# B: VCE, no h2 at all, "apply_xov_cov_tracks":True copied from CC5_B "compute_vce",False) ... to check, aldo redo accumxov iter10 w/o lat split
# CC7: gtracks and xov North from MLA
# A: VCE, no h2, "apply_xov_cov_tracks":True copied from CC5_C
# CC8: gtracks and xov North from MLA
# A: VCE, no h2 at all, copied from CC5_E
# CC9: gtracks and xov North from MLA PAM
# A: VCE, no h2 at all, "apply_xov_cov_tracks":True, maxR=1000m, (ACR) (100,100,20), mean_constr=1m
# CD0: gtracks and xov North from MLA
# A: VCE, no h2 at all, copied from CC5_H
# CD1: gtracks and xov North from MLA h2=0
# A: VCE, no h2 at all, downsize=False
# CD2: up to iter3?

# CD3: gtracks and xov North from CB9: {'dA':20., 'dC':20., 'dR':5.,'dRl':3, 'dPt':3}
# CD4: gtracks and xov North from CB5: {'dA':20., 'dC':20., 'dR':5.,'dRl':3, 'dPt':3} wrong dh2, scale factor only until 11
# CD5: gtracks and xov North from CB5: {'dA':20., 'dC':20., 'dR':5.,'dRl':3, 'dPt':3} wrong dh2, scale factor only until 5

# CD6: gtracks and xov North from MLA
# CD7: gtracks and xov North from MLA kinetx AG
# CD8: gtracks and xov North from MLA spaux
# CD9: gtracks and xov North from MLA kinetx IAU

# CE0: gtracks and xov North from CB5: {'dA':20., 'dC':20., 'dR':5.,'dRl':20, 'dPt':20} wrong dh2, scale factor only until 5

simid = 'CB5'
estid = 'CE0'
iter = 0
XovOpt.set("selected_hemisphere",'N')

max_job = 1500
max_parallel = 200   
max_parallel = 12*10

# SPK timespans (ET)
de_start = [dt.datetime(2011, 3,18, 6,56, 6,185),
           dt.datetime(2011, 7,16, 7,31, 6,183),
           dt.datetime(2012, 4,30,19,16, 6,185),
           dt.datetime(2013, 5, 1,00, 1, 7,185),
           dt.datetime(2013,12, 6,21,41, 7,183),
           dt.datetime(2013,12,21,21,46, 7,183),
           dt.datetime(2014, 1,26,22, 1, 7,184),
           dt.datetime(2014, 3,11,20,36, 7,185),
           dt.datetime(2014, 5, 1,00, 1, 7,185),
           dt.datetime(2014,12, 3,20, 6, 7,183),
           dt.datetime(2015, 2,14,21,56, 7,185),
           dt.datetime(2015, 4,17,18,31, 7,185)]

de_end = [dt.datetime(2011, 7,14,19,51, 6,183),
         dt.datetime(2012, 5, 1,18,51, 6,185),
         dt.datetime(2013, 5, 1,20,21, 7,185),
         dt.datetime(2013,12, 5,21,26, 7,183),
         dt.datetime(2013,12,20,19,51, 7,183),
         dt.datetime(2014, 1,25,21,46, 7,184),
         dt.datetime(2014, 3,10,22, 1, 7,185),
         dt.datetime(2014, 5, 1,00, 1, 7,185),
         dt.datetime(2014,12, 2,18,31, 7,183),
         dt.datetime(2015, 2,13,12,36, 7,185),
         dt.datetime(2015, 4,17,17,16, 7,185),
         dt.datetime(2015, 4,29,22,26, 7,185)]

nMonths = math.ceil((de_end[-1] - de_start[0]).days/30)
nYears = math.ceil((de_end[-1] - de_start[0]).days/365)


# Read date from lbl files (information file for altimetry raw files)
d_start = []
d_end  = []
# folder = "/storage/homefs/desprats/pyxover/examples/BELA/data/raw/2015"
# files = [y for x in os.walk(folder) for y in glob.glob(os.path.join(x[0], '*.lbl'))]
folder = f"{basedir}data/raw/"
# files = [z for y in os.walk(folder) for x in os.walk(y[0]) for z in glob.glob(os.path.join(x[0], '*.lbl'))]
files = [z for y in os.walk(folder) for z in glob.glob(os.path.join(y[0], '*.lbl'))]
# files = ["/storage/homefs/desprats/pyxover/examples/BELA/data/raw/2011/apr/mlascirdr1104060314.lbl"]
files.sort()
for file_path in files:
   with open(file_path, "r") as f:
      for line in f:
         if re.search("START_TIME", line) or re.search("STOP_TIME", line):
            date = dt.datetime.strptime(line.split('= ')[-1][:-1], '%Y-%m-%dT%H:%M:%S')
            # date = dt.datetime.fromisoformat(line.split('= ')[-1][:-1])
            if re.search("START_TIME", line):
               d_start.append(date)
            else:
               d_end.append(date)
               break


# General options
XovOpt.set("body", 'MERCURY')
XovOpt.set("spice_meta", 'mymeta_MLA')
# XovOpt.set("spice_meta", 'mymeta_MLA_KX_AG')
XovOpt.set("basedir", basedir)
XovOpt.set("instrument", 'MLA')
XovOpt.set("parallel", False)
XovOpt.set("max_range_altitude", 1050)
# XovOpt.set("SpInterp", 2)


vecopts = {'SCID': '-236',
           'SCNAME': 'MESSENGER',
           'SCFRAME': '-236000',
           'INSTID': (-236500, -236501), # not used
           'INSTNAME': ('MSGR_MLA', 'MSGR_MLA_RECEIVER'), # not used
           'PLANETID': '199',
           'PLANETNAME': 'MERCURY',
           'PLANETRADIUS': 2440.,
           'PLANETFRAME': 'IAU_MERCURY',
           'OUTPUTTYPE': 1,
           'ALTIM_BORESIGHT': [0.0022105, 0.0029215, 0.9999932892],
           'INERTIALFRAME': 'J2000',
           'INERTIALCENTER': 'SSB',
           'PM_ORIGIN': 'J2000',
           'PARTDER': ''}

vecopts['SCFRAME'] = 'MSGR_SPACECRAFT'
XovOpt.set("vecopts", vecopts)

# PYALTSIM
# --------
# In 2012 we don't have 01, 02,03,05,06,07 ...
if run_pyAltSim:
   # PySimAlt options
   XovOpt.set("sampling_rate",8) # [Hz]
   XovOpt.set("resopt", 3)
   XovOpt.set("amplopt", 20)
   XovOpt.set("apply_topo", True)
   XovOpt.set("small_scale_topo", True)
   XovOpt.set("range_noise", True)
   XovOpt.set("new_illumNG", True)
   XovOpt.set("range_noise_mean_std",[0.,0.25])

   XovOpt.set("partials", False)
   XovOpt.set("expopt", simid)
   XovOpt.check_consistency()

   pyaltsim_in = []
   print(f'Simulatation of MLA data for {nMonths} months, from {d_start[0]} to {d_end[-1]}')
   for y in range(0,len(d_start)):
      inephem = True
      for j in range(0,len(de_start)-1):
         if ((d_start[y]>de_end[j] and d_start[y]<de_start[j+1]) or
             (d_end[y]>de_end[j] and d_end[y]<de_start[j+1])):
            inephem=False
      if (d_start[y]>de_end[-1] or d_end[y]>de_end[-1]):
         print(d_start[y], de_end[-1], d_end[y], de_end[-1])
         inephem=False
      # if (d_start[y]<dt.datetime(2014, 5, 20,0,0, 0)):
      #    inephem=False
      # if (d_start[y]<dt.datetime(2013, 11, 2,0,0, 0) or d_end[y]>dt.datetime(2013, 11,3,0,0, 0)):
      #    inephem=False
      if inephem:
         monyea = d_start[y].strftime('%y')
         indir_in = f'SIM_{monyea}/{XovOpt.get("expopt")}/'
         pyaltsim_in.append([XovOpt.get("amplopt"), XovOpt.get("resopt"), indir_in,
                             d_start[y], d_end[y], XovOpt.to_dict()])
   print(len(pyaltsim_in))
   if grid:
      executor = submitit.AutoExecutor(folder=f'{log_folder}{simid}/pyaltsim')
      executor.update_parameters(slurm_cpus_per_task=1,
                                 slurm_nodes=1,
                                 slurm_name="pyaltsim",
                                 slurm_array_parallelism=max_parallel,
                                 slurm_time=60*3, # minutes
                                 slurm_mem='3G') # 4GB for 10Hz
      if len(pyaltsim_in) == 1:
         job = executor.submit(PyAltSim.main, pyaltsim_in[0]) # single job
      else:
         # Launch max 400 jobs in //
         i_j = 0
         while i_j < len(pyaltsim_in):
            ip_j = i_j + max_job
            if ip_j>len(pyaltsim_in):
               ip_j = len(pyaltsim_in)
            print(f"Running {ip_j-i_j+1} jobs in // from {pyaltsim_in[i_j][3]} to {pyaltsim_in[ip_j-1][4]}")
            jobs = executor.map_array(PyAltSim.main, pyaltsim_in[i_j:ip_j])
            i_j = ip_j
            print("End")
            for job in jobs:
               job.result() # wait for job to finish
   else:
      PyAltSim.main( pyaltsim_in[0])
      # for arg in pyaltsim_in:
      #    PyAltSim.main(arg)

if not (run_pyGeoLoc or run_pyXover or run_accuXover):
   sys.exit()
   
XovOpt.set("partials", True)
XovOpt.set("expopt", estid)

# pyGeoloc options
if run_pyGeoLoc:
   # Add a check wether the perturbation is consistent with previous iteration !
   # XovOpt.set("pert_cloop_orb", {'dA':50., 'dC':50., 'dR':20., 'dRl':0.5, 'dPt':0.5, 'dA1':40., 'dC1':40., 'dR1':10.})
   # XovOpt.set("pert_cloop_orb", {'dA':50., 'dC':50., 'dR':20., 'dRl':0.5, 'dPt':0.5})
   # XovOpt.set("pert_cloop_orb", {'dA':20., 'dC':20., 'dR':5.})
   # XovOpt.set("pert_cloop_orb", {'dRl':20, 'dPt':20})
   XovOpt.set("pert_cloop_orb", {'dA':20., 'dC':20., 'dR':5.,'dRl':20, 'dPt':20})
   # XovOpt.set("pert_cloop_orb", {'dA':20., 'dC':20., 'dR':5.,'dRl':3, 'dPt':3})
   # Perturbations have been set to an RMSE of 50 m (+40 m/day) in AC
   # and 20 m (+10 m/day) in R, 0.5 arcsec for the pointing (Bertone+2021)
   # "pert_cloop_orb": {},  # 'dA':50., 'dC':50., 'dR':20.,'dRl':0.5, 'dPt':0.5} #} #, 'dA1':20., 'dC1':20., 'dR1':5.

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
   # perturbations = {k: perturbations[k] for k in params if k in perturbations}
   XovOpt.set("parGlo", perturbations)

# pyXover options
if run_pyXover:
   XovOpt.set("compute_input_xov", True)
   # XovOpt.set("import_proj", True)
   XovOpt.set("msrm_sampl", 20)
   XovOpt.set("n_interp",6)
   XovOpt.set("monthly_sets", True)

# AccumXov options
if run_accuXover:
   XovOpt.set("sol4_orb", [])
   XovOpt.set("sol4_orbpar", [None])
   XovOpt.set("sol4_orbpar", ['dA','dC','dR'])
   params = ['dR/dRA', 'dR/dDEC', 'dR/dPM','dR/dL']
   # params = ['dR/dRA', 'dR/dDEC', 'dR/dPM','dR/dL', 'dR/dh2']
   # params = ['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dh2']
   # params += [f'dR/dLIB{i}' for i in range(1, 12)]
   XovOpt.set("sol4_glo", params)

   # XovOpt.set("par_constr",
   #            {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 3.e-1, 'dR/dA': 1.e2,
   #             'dR/dC': 1.e2, 'dR/dR': 2.e1})  # , 'dR/dRl':5.e1, 'dR/dPt':5.e1} #
   # XovOpt.set("par_constr",
   #            {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 1, 'dR/dA': 1.e2,
   #             'dR/dC': 1.e2, 'dR/dR': 2.e1})  # , 'dR/dRl':5.e1, 'dR/dPt':5.e1} #
   # XovOpt.set("par_constr",
   #            {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 1, 'dR/dA': 20,
   #             'dR/dC': 20, 'dR/dR': 5})  # , 'dR/dRl':5.e1, 'dR/dPt':5.e1} #
   XovOpt.set("par_constr",
              {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 1, '1.*dR/dA': 20,
               '1.*dR/dC': 20, '1.*dR/dR': 5})
   # XovOpt.set("par_constr",
   #            {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 1, '1.*dR/dA': 100,
   #             '1.*dR/dC': 100, '1.*dR/dR': 20})

   # XovOpt.set("mean_constr", {'dR/dA': 1.e0, 'dR/dC': 1.e0, 'dR/dR': 1.e0})
   XovOpt.set("mean_constr", {})


   # AccOpt.set("weight_obs",[0.0025])
   # AccOpt.set("weight_constr",[1.5])
   # AccOpt.set("weight_obs",[1])
   # AccOpt.set("weight_constr",[10])
   AccOpt.set("compute_vce",True)
   AccOpt.set("solving_method","cholesky")
   # XovOpt.set("sol4_orbpar", [None])
   # XovOpt.set("sol4_glo", [None])
   AccOpt.set("apply_xov_cov_tracks",True)
   AccOpt.set("downsize", True)
   # XovOpt.set("sol4_glo", ['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dL'])
   # AccOpt.set("convergence_criteria",0)

XovOpt.check_consistency()
# d_start0 = d_start[0]
d_start0 = dt.datetime(2011,3,1)

for iter in range(0,1):

   if iter > 0:
      XovOpt.set("import_abmat", f"Abmat_{estid}_{iter-1}_{iter}_A.pkl")

   AccOpt.set("Abmat_outfile", f"Abmat_{estid}_{iter}_{iter+1}_A.pkl")
   # AccOpt.set("Abmat_infile",f"Abmat_{estid}_{iter}_{iter+1}_nosol.pkl")

# PYGEOLOC (geolocation step)
# --------
   if run_pyGeoLoc:
      if grid:
         executor = submitit.AutoExecutor(folder=f'{log_folder}{estid}/pygeoloc')
         executor.update_parameters(slurm_nodes=1,
                                    slurm_array_parallelism=max_parallel,
                                    slurm_name="pygeoloc",
                                    slurm_mem='3G',
                                    slurm_cpus_per_task=1,
                                    slurm_time=40) # minutes
         if iter>0:
            # executor.update_parameters(slurm_mem='20G')
            executor.update_parameters(slurm_mem='5G')
      pygeoloc_in = []
      for y in range(0,nYears):
      # for y in range(4,nYears):
      # for y in [4]:
         import glob
         monyea = (d_start0 + dt.timedelta(days=y*365)).strftime('%y')
         print(f"Geolocation year {monyea}")
         indir_in = f'SIM_{monyea}/{simid}/'
         indir_in = f'MLA_{monyea}/'
         outdir_in = f'{XovOpt.get("expopt")}_{iter}/gtrack_{monyea}'
         epo_in = ""
         allFiles = glob.glob(os.path.join(f'{XovOpt.get("rawdir")}{indir_in}', f'{XovOpt.get("instrument")}*RDR*.*'))
         # allFiles = glob.glob(os.path.join(f'{XovOpt.get("rawdir")}{indir_in}', f'{XovOpt.get("instrument").lower()}*rdr*.*'))
         # Retrieve the dates from the names of all the files in the directory
         d_files = [dt.datetime.strptime(fil.split('.')[0][-10:], '%y%m%d%H%M')  for fil in allFiles[:]]
         d_files = list(set(d_files))
         d_files.sort()
         # d_files = d_files[60]
         # d_files = [d_files[21]]
         nt = 9
         pygeoloc_in.extend([[epo_in, indir_in, outdir_in, d_files[i*nt:min((i+1)*nt,len(d_files))+1], iter, XovOpt.to_dict()] for i in range(0,math.ceil(len(d_files)/nt))])
         # Add last file ending
         yyyy = d_files[-1].year + 1
         pygeoloc_in.extend([[epo_in, indir_in, outdir_in, [d_files[-1], dt.datetime(yyyy, 1, 1,0,0, 0)], iter, XovOpt.to_dict()] ])

         #pygeoloc_in.append([epo_in, indir_in, outdir_in, [d_files[j], d_files[j+1]], iter, XovOpt.to_dict()]) # correct way
         # for j in range(0,len(d_files)-1): # WD: You should think of // processing within one run
         #   if d_files[j] < dt.datetime(11,11,4,9,0) or  d_files[j] > dt.datetime(11,11,4,10,0):
         #      continue
         #   pygeoloc_in.append([epo_in, indir_in, outdir_in, [d_files[j], d_files[j+1]], iter, XovOpt.to_dict()]) # correct way
            # pygeoloc_in.append([f'{monyea}', indir_in, outdir_in, pattern, 0, XovOpt.to_dict()]) # not working, adapt from above
      # pygeoloc_in = [pygeoloc_in[0]]
      if grid:
         if len(pygeoloc_in) == 1:
            print(f"Running single job from {pygeoloc_in[0][3][0]} to {pygeoloc_in[0][3][-1]}")
            job = executor.submit(PyGeoloc.main, pygeoloc_in[0]) # single job
         else:
            # Launch max 400 jobs in //
            i_j = 0
            while i_j < len(pygeoloc_in):
               ip_j = i_j + max_job
               if ip_j>len(pygeoloc_in):
                  ip_j = len(pygeoloc_in)
               print(f"Running {ip_j-i_j+1} jobs in // from {pygeoloc_in[i_j][3][0]} to {pygeoloc_in[ip_j-1][3][-1]}")
               jobs = executor.map_array(PyGeoloc.main, pygeoloc_in[i_j:ip_j])
               i_j = ip_j
               time.sleep(10)
               print("End")
               for job in jobs:
                  job.result()
      else:
         PyGeoloc.main(pygeoloc_in[0])

# PYXOVER
# -------
   if run_pyXover:
      indir_in =  f'{XovOpt.get("expopt")}_{iter}/gtrack_'
      outdir_in = f'{XovOpt.get("expopt")}_{iter}/'

      misy = [str(m) for m in range(1103,1113)] + \
         ['1201','1202','1204','1207','1208','1212'] + \
         [f"{yy}{m:02}" for yy in range(13,15) for m in range(1,13)] +  \
         [str(m) for m in range(1501,1505)]

      # misy =  [str(m) for m in range(1103,1113)]
      # misy += ['1201','1202','1204','1207','1208','1212']

      # misy = [(d_start[0] + dt.timedelta(days=30*w)).strftime('%y%m') for w in range(0,nMonths)]
      misycmb = [x for x in itert.combinations_with_replacement(misy, 2)]
      print("Choose grid element among:", dict(map(reversed, enumerate(misycmb))))

      if grid:
         executor = submitit.AutoExecutor(folder=f'{log_folder}{estid}/pyxover')
         executor.update_parameters(slurm_name="pyxover",
                                    slurm_nodes=1,
                                    slurm_mem='5G', # 11G
                                    slurm_cpus_per_task=1,
                                    slurm_time=60*5, # minutes
                                    slurm_array_parallelism=max_parallel)
         if iter>0:
            executor.update_parameters(slurm_mem='11G')

      pyxover_in = []
      for par in range(0,len(misycmb)):
      # for par in range(31, len(misycmb)):
      # for par in range(14, 48):
      # for par in [0,13,25,36,46,55,63,70,76,81,85,88,90]:
      # for par in [434]:
         # create symlink to rough xovs from other tests
         input_xov_path = XovOpt.get("outdir") + outdir_in + 'xov/xov_' + str(misycmb[par][0]) + '_' + str(misycmb[par][1]) + '.pkl'
         if os.path.exists(input_xov_path):
            print("input xov file already exists in", input_xov_path)
         else:
            gtrack_dirs = [os.path.join(XovOpt.get("outdir"), indir_in + par1[:2] ) for par1 in misycmb[par]]
            pyxover_in.append([f'{par}',gtrack_dirs, outdir_in, misycmb[par], iter, XovOpt.to_dict()])
            # pyxover_in.append([f'{par}',indir_in, outdir_in, misycmb[par], 0,XovOpt.to_dict()])
      print(f'{len(pyxover_in)} combinations to process')
      if grid:
         if(len(pyxover_in) == 1):
            job = executor.submit(PyXover.main, pyxover_in[0]) # single job
            print(job.result())
         else:
            # Launch max_jobs in //
            i_j = 0
            while i_j < len(pyxover_in):
               ip_j = i_j + max_job
               if ip_j>len(pyxover_in):
                  ip_j = len(pyxover_in)
               jobs = executor.map_array(PyXover.main, pyxover_in[i_j:ip_j+1])
               i_j = ip_j + 1
               for job in jobs:
                  job.result()
      else:
         PyXover.main(pyxover_in[0])

       # Might be nicer to form a array as below
       # pyxover_in = [[comb, f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_',
       #            f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/',
       #            'MLASIMRDR', 0, XovOpt.to_dict()]
       #            # for comb in np.arange(33,78)]
       #             for comb in [33, 41, 77]]  # np.arange(1)]

# ACCUMXOV
# --------
   if run_accuXover:
      datasets = [f'{estid}_{iter}/']
      if grid:
         executor = submitit.AutoExecutor(folder=f'{log_folder}{estid}/accumxov')
         executor.update_parameters(slurm_nodes=1,
                                    slurm_name="accumXov",
                                    slurm_mem='20G',
                                    slurm_cpus_per_task=2,
                                    slurm_time=60*5) # minutes
         job = executor.submit(AccumXov.main, [datasets, '', iter, XovOpt.to_dict(), AccOpt.to_dict()]) # single job
         print(job.result())
      else:
         out = AccumXov.main([datasets, '', iter, XovOpt.to_dict(), AccOpt.to_dict()])
