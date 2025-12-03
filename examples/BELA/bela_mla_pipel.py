import logging
import os
import unittest
import numpy as np

import submitit
from accumxov.accum_opt import AccOpt
from config import XovOpt

from accumxov import AccumXov
from accumxov.Amat import Amat
from pyxover import PyXover
import datetime as dt
import math
import itertools as itert
from xovutil.units import deg2as

grid = True
run_pyXover   = False
run_accuXover = True

camp = "/storage/research/aiub_gravdet/WD_BELA/"
OrbDir = f"{camp}ORB/"
log_folder = f"{camp}pyXover/log/"


# BA0: estid_N1 = 'SA', estid_N2 = 'AA5' (removed)
# BA1: estid_N1 = 'CA0', estid_N2 = 'AB0'
# BA2/3: estid_N1 = 'CA1', estid_N2 = 'AB2' (q70=0.1)
# v2: demi tracks
# v3: demi tracks, est dACR
# v4: demi tracks, redo MLA/BEL, est dACR
# BA4: estid_N1 = 'CA5', estid_N2 = 'AC4' (q70=0.1)
# BA5: estid_N1 = 'CB1', estid_N2 = 'AD1'
# BA6: estid_N1 = 'CB1', estid_N2 = 'AD2' no accum
# BA7: estid_N1 = 'CB2', estid_N2 = 'AD2'
#     A: 'dR/dh2': 1 (harcoded for BELA tracks) (less biased MLA tracks with dR_BELA/dR constrained)
#     B: 'dR/dA': 20, 'dR/dC': 20, 'dR/dR': 5})
#     C: 'dR/dh2': 1 (harcoded for BELA tracks) (extended)
# BA8: estid_N1 = 'CB0', estid_N2 = 'AD2'
#     B: 'dR/dh2': 1 (harcoded for BELA tracks)
# BA9: estid_N1 = 'CB3', estid_N2 = 'AD4'
#    A: extended BELA
#    B: nominal BELA
#    C: extended VCE weight = [1 1 1] (redo)
#    D: nominal VCE weight = [1 1 1]
# BB0: estid_N1 = 'CB4', estid_N2 = 'AD4'
#    A: nominal BELA 0.7^2, .0.7, 1
#    B: nominal BELA
#    C: nominal BELA 0.5^2, .0.5, 1
#    D: nominal BELA 0.25^2, .0.25, 1
#    E: nominal BELA 0.01^2, .0.01, 1
#    F: nominal VCE weight = [1 1 1]
#    G: extended VCE weight = [1 1 1] (redo)
# BB1: estid_N1 = 'CB5', estid_N2 = 'AD6'
# BB2: estid_N1 = 'CB6', estid_N2 = 'AE0'
# C: VCE
# D: no VCE
# E: no VCE nominal
# F: VCE nominal
# G: VCE nominal, 85° threshold
# H: VCE nominal, 85° threshold, 75° for MLA/BELA
# I0: VCE nominal, 80° threshold, 88° for BELA/BELA
# J0: VCE nominal, 75° threshold, 85° for BELA/BELA
# K (J): VCE nominal, 10° bands, threshold = 100
# L (I): VCE nominal, 10° bands, threshold = 300
# M: VCE nominal, 10° bands, threshold = 1000 try to fix
# N: VCE nominal, 10° bands, threshold = 1000
# A: VCE
# BB3: estid_N1 = 'CB9', estid_N2 = 'AE2', estid_S  = 'AE3'
# A: no VCE, EM, blockwise-cholesky
# B: VCE, PM, cholesky
# C: no VCE, EM, cholesky
# D: no VCE, EM, cg, var_est
# E: VCE, EM, cholesky
# F (E): VCE, EM, no combined MLA/BELA
# BB4: estid_N1 = 'CC4', estid_N2 = 'AE4', estid_S  = 'AE5' dLIB1-11
# CC4_BB4_A: VCE, EM, no combined MLA/BELA
# A: VCE, EM, no combined MLA/BELA threshold 88deg
# B: VCE, EM
# BB5: estid_N1 = 'CD3', estid_N2 = 'AE6', estid_S  = 'AE7'
# BB6: estid_N1 = 'CD4', estid_N2 = 'AE8', estid_S  = 'AE9'
# BB7: estid_N1 = 'CD5', estid_N2 = 'AF0', estid_S  = 'AF1'

estid_N1 = 'CD5'
estid_N2 = 'AF0'
estid_S  = 'AF1'
estid = 'BB7'
iter = 0
XovOpt.set("selected_hemisphere",'N')
max_job = 250
partition = "epyc2"# epyc2, icpu-aiub
partition = "icpu-aiub"# epyc2, 
max_job = 1500
if partition == "icpu-aiub": # 120GB not possible
   max_job = 200

XovOpt.set("body", 'MERCURY')
XovOpt.set("basedir", f'{camp}pyXover/')
XovOpt.set("instrument", 'BELA')
XovOpt.set("debug", False)
XovOpt.set("compute_input_xov", True)
# XovOpt.set("import_proj", True)
XovOpt.set("msrm_sampl", 20)
XovOpt.set("n_interp",6)

XovOpt.set("sol4_orb", [])
XovOpt.set("sol4_orbpar", [None])
XovOpt.set("sol4_orbpar", ['dA','dC','dR'])

params = ['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dL', 'dR/dh2']
# params = ['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dh2']
# params += [f'dR/dLIB{i}' for i in range(1, 12)]
XovOpt.set("sol4_glo", params)

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
# XovOpt.set("par_constr",
#            {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 3.e-1, 'dR/dA': 1.e2,
#             'dR/dC': 1.e2, 'dR/dR': 2.e1})  # , 'dR/dRl':5.e1, 'dR/dPt':5.e1} #
XovOpt.set("par_constr",
           {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 1, 'dR/dA': 1.e2,
            'dR/dC': 1.e2, 'dR/dR': 2.e1})  # , 'dR/dRl':5.e1, 'dR/dPt':5.e1} #
XovOpt.set("par_constr",
           {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 1, '1.*dR/dA': 20,
            '1.*dR/dC': 20, '1.*dR/dR': 5, '2.*dR/dA': 10, '2.*dR/dC': 5, '2.*dR/dR': 0.2})  # , 'dR/dRl':5.e1, 'dR/dPt':5.e1} #
# XovOpt.set("par_constr",
#            {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 1, 'dR/dA': 20,
#            'dR/dC': 20, 'dR/dR': 5})  # , 'dR/dRl':5.e1, 'dR/dPt':5.e1} #

# XovOpt.set("mean_constr", {'dR/dA': 1.e0, 'dR/dC': 1.e0, 'dR/dR': 1.e0})
XovOpt.set("mean_constr", {})

XovOpt.set("resopt", 3)
XovOpt.set("amplopt", 20)
XovOpt.set("SpInterp", 0)

# BELA timespan
d_start = dt.datetime(2027,4,1,0,0,0)
d_end   = dt.datetime(2028,4,1,0,0,0) # nominal mission
d_end   = dt.datetime(2029,4,1,0,0,0) # extended mission
nMonths = math.ceil((d_end - d_start).days/30)

XovOpt.set("parallel", False)
XovOpt.set("apply_topo", True)
XovOpt.set("small_scale_topo", True)
XovOpt.set("range_noise", False)
XovOpt.set("new_illumNG", True)
XovOpt.set("unittest", False) # this restricts simulated data to the first day of the month (see d_last in PyAltSim.main)

XovOpt.set("partials", True)
XovOpt.set("expopt", estid)
XovOpt.set("weekly_sets", False)
XovOpt.set("monthly_sets", True)
XovOpt.check_consistency()

AccOpt.set("weight_obs",[1, 1, 1])
# AccOpt.set("weight_obs",[1, 1])
# AccOpt.set("weight_obs",[1, 1, 1, 1, 1, 1])
AccOpt.set("weight_constr",[1, 1])
AccOpt.set("Abmat_outfile", f"Abmat_{estid}_{iter}_{iter+1}_A.pkl")
# AccOpt.set("Abmat_infile",f"Abmat_{estid}_{iter}_{iter+1}_nosol.pkl")
AccOpt.set("compute_vce",True)
# AccOpt.set("downsize", True)
# AccOpt.set("solving_method","cg")
AccOpt.set("solving_method","cholesky")
AccOpt.set("apply_xov_cov_tracks",False)
AccOpt.check_consistency()   


# PYXOVER
# -------
if run_pyXover:
   misy2 = [(d_start + dt.timedelta(days=30*m)).strftime('%y%m') for m in range(0,nMonths)]

   misy1 = [str(m) for m in range(1103,1113)] + \
      ['1201','1202','1204','1207','1208','1212'] + \
      [f"{yy}{m:02}" for yy in range(13,15) for m in range(1,13)] +  \
      [str(m) for m in range(1501,1504)]   
      
   misycmb = list(itert.product(misy1, misy2))
   # misycmb = [('1304', '2604')]
   # misycmb = [x for x in itert.combinations_with_replacement(misy1, 2)]
   indir_in =  f'{XovOpt.get("expopt")}_0/gtrack_'
   outdir_in = f'{XovOpt.get("expopt")}_0/'
   xov_dir = XovOpt.get("outdir") + outdir_in + 'xov/'
   
   print("Choose grid element among:", dict(map(reversed, enumerate(misycmb))))

   if grid:
      executor = submitit.AutoExecutor(folder=f'{log_folder}{estid}/pyxover')
      executor.update_parameters(slurm_partition=partition,
                                 slurm_name="pyxover",
                                 slurm_nodes=1,
                                 slurm_mem='10G',
                                 slurm_cpus_per_task=2,
                                 slurm_time=60*5, # minutes
                                 slurm_array_parallelism=150)
      if partition == "icpu-aiub":
         executor.update_parameters(slurm_qos="job_icpu-aiub")
      else:
         executor.update_parameters(slurm_array_parallelism=400)
    
   pyxover_in = []
   for par in range(0,len(misycmb)):
   # for par in range(1500, len(misycmb)):
   # for par in [0]:
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
            gtrack_dirs = [os.path.join(XovOpt.get("outdir"), f"{estid_N1}_0/gtrack_" + str(misycmb[par][0][:2])),
                           os.path.join(XovOpt.get("outdir"), f"{estid_N2}_0/gtrack_" + str(misycmb[par][1]))]
            # gtrack_dirs = [os.path.join(XovOpt.get("outdir"), indir_in + str(misycmb[par][0][:2])),
            #                os.path.join(XovOpt.get("outdir"), indir_in + str(misycmb[par][1]))]
            # gtrack_dirs = [os.path.join(XovOpt.get("outdir"), indir_in + str(misycmb[par][0][:2])),
            #                os.path.join(XovOpt.get("outdir"), indir_in + str(misycmb[par][1][:2]))]
            # pyxover_in.append([f'{par}',indir_in, outdir_in, misycmb[par], 0,XovOpt.to_dict()])
            pyxover_in.append([f'{par}',gtrack_dirs, outdir_in, misycmb[par], 0,XovOpt.to_dict()])
         
   print(f'{len(pyxover_in)} combinations to process')
      
   if grid:
      if(len(pyxover_in) == 1):
         job = executor.submit(PyXover.main, pyxover_in[0]) # single job
         print(job.result())
      else:
         # Launch max_jobs in //
         i_j = 0
         while i_j <= len(pyxover_in):
            ip_j = i_j + max_job
            if ip_j>len(pyxover_in):
               ip_j = len(pyxover_in) + 1
            # jobs = executor.map_array(PyXover.main, pyxover_in[i_j:ip_j+1])
            jobs = executor.map_array(PyXover.main, pyxover_in[i_j:ip_j])
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
   datasets = [f'{estid}_{iter}/',f'{estid_N1}_{iter}/',f'{estid_N2}_{iter}/',f'{estid_S}_{iter}/']
   # datasets = [f'{estid_N1}_{iter}/',f'{estid_N2}_{iter}/',f'{estid_S}_{iter}/']
   # datasets = [f'{estid}_{iter}/',f'{estid_N1}_{iter}/',f'{estid_N2}_{iter}/']
   if grid:
      executor = submitit.AutoExecutor(folder=f'{log_folder}{estid}/accumxov')
      executor.update_parameters(slurm_partition=partition,#epyc2, icpu-aiub
                                 slurm_name="accumXov",
                                 slurm_cpus_per_task=2,
                                 slurm_nodes=1,
                                 slurm_time=60*12, # minutes
                                 slurm_mem='500G', # 70G
                                 slurm_array_parallelism=100)
      if partition == "icpu-aiub":
         executor.update_parameters(slurm_qos="job_icpu-aiub")
   
      job = executor.submit(AccumXov.main, [datasets, '', iter, XovOpt.to_dict(), AccOpt.to_dict()]) # single job
      job.result()
   else:
      out = AccumXov.main([datasets, '', iter, XovOpt.to_dict(), AccOpt.to_dict()])
