import logging
import os
import unittest
import submitit
import datetime as dt
from  wd_utils import build_sessiontable_man
import math

from accumxov.accum_opt import AccOpt
from config import XovOpt

from accumxov import AccumXov
from accumxov.Amat import Amat
from pygeoloc import PyGeoloc
from pyxover import PyXover
import itertools as itert
# from examples.MLA.options import XovOpt.get("vecopts")

# update paths and check options
from pyaltsim import PyAltSim

grid = True
run_pyAltSim  = False # 3-4 hours per month, 30min-1h15 per week (up to 2.8GB)
run_pyGeoLoc  = False # quite fase per gtrack
run_pyXover   = False # 10min or 20min
run_accuXover = True

camp = "/storage/research/aiub_gravdet/WD_XOV"
OrbDir = f"{camp}/ORB/"
SIMID     = "Ah7"  # Simulation ID
ORBID      = "014"  # Input CR3BP orbit
MANFIL = f"{OrbDir}CAL_{ORBID}_{SIMID}_CR3BP.ORB"

# CA7 : simulation from Ag5
# CA7 : gtracks and cov North from Ag5
# CA8 : gtracks and cov South from Ag5
# CB0 : gtracks and cov North from Ag5
# CB1 : gtracks and cov South from Ag5
# CB2 : simulation from Ah5
# CB2 : gtracks and cov South from Ah5 with Je8
# CB3 : gtracks and cov North from CB2 with Ah5 (being redone with Je8)
# CB4 : simulation from Ah5 sample 1s
# CB4 : gtracks and cov North from CB4 with Ah5
# CB5 : simulation from Ah5 sample 0.05s
# CB5 : gtracks and cov North from CB5 with Ah5
# CB6 : gtracks and xov North from CB2 with Jf3
# CB7 : gtracks and xov South from CB2 with Jf3
# CB8 : gtracks and xov North from CB2 with Jf4
# CB9 : gtracks and xov South from CB2 with Jf4
# CC0 : gtracks and xov North from CB2 with Jf5 (test pyxover)
# CC1 : gtracks and xov South from CB2 with Jf5
# CC2 : simulation from Ah7 (sampling 1s)
# CC2 : gtracks and xov North from CC2 with Ah7

simid = 'CB2'
estid = 'CC0'
XovOpt.set("selected_hemisphere",'N')
XovOpt.set("spice_meta",'mymeta_Jf5')

XovOpt.set("body", 'CALLISTO')
XovOpt.set("basedir", f'{camp}/pyXover/')
XovOpt.set("instrument", 'CALA')

# Subset of parameters to solve for
# For "sol4_orb" and "sol4_orbpar, laisser une liste vide signifie "tous", sinon mettre "None" pour pas estimer
XovOpt.set("sol4_orb", []) # pour quelles orbites tu veux estimer les parametres "sol4_orbpar"
XovOpt.set("sol4_orbpar", []) # quels parametres d'orbite, quelle direction ou pointing tu veux estimer
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

XovOpt.set("expopt", simid)
XovOpt.set("resopt", 3)
XovOpt.set("amplopt", 20)
XovOpt.set("spauxdir", 'CAL_spk/')


# interpolation/spice direct call (0: use spice, 1: yes, use interpolation, 2: yes, create interpolation)
XovOpt.set("SpInterp", 0) # TODO for some reason, SPICE interpolation gave weird results... beware+correct


XovOpt.check_consistency()
AccOpt.check_consistency()

if XovOpt.get("SpInterp") == 0:
    if not os.path.exists('data/aux/kernels'):
        os.makedirs('data/aux/kernels')
    os.chdir('data/aux/kernels')
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

XovOpt.set("parallel", True)
XovOpt.set("new_illumNG", True)
XovOpt.set("unittest", True) # this restricts simulated data to the first day of the month (see d_last in PyAltSim.main)
XovOpt.set("debug", False)

if grid:
    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder="log_slurm")
    # set timeout in min, and partition for running the job
    executor.update_parameters( #slurm_account='j1010',
                               slurm_partition="aiub",
                               slurm_cpus_per_task=1,
                               slurm_nodes=1,
                               slurm_array_parallelism=50)

# pyaltsim_in = [[f'{yy:02d}{mm:02d}', f'SIM_{yy:02d}/KX5/0res_1amp/', f'sim/{XovOpt.get("expopt")}_{iternum}/0res_1amp/gtrack_{yy:02d}', 'MLASCIRDR', iternum, opt_dict] for mm in range(13)[1:] for yy in [8,11,12,13,14,15]]

# Altimetry simulation
# WD should really organize the folders in weekly folders
if run_pyAltSim:
    XovOpt.set("sim_altdata", True)
    XovOpt.set("partials", False)
    XovOpt.set("apply_topo", False)
    XovOpt.set("range_noise", False)
    XovOpt.set("expopt", simid)
    d_sess = build_sessiontable_man(MANFIL,500,500) #darc > 1 Cday

    # Divide by weeks
    nWeeks = math.floor((d_sess[-1] - d_sess[0]).days/7)
    d_weeks = []
    for w in range(0,nWeeks+1):
        d_weeks.append(d_sess[0] + dt.timedelta(weeks=w))

    d_sess.extend(d_weeks[1:])
    d_sess.sort()
    # print(d_sess)
    pyaltsim_in = []
    j = 0
    for i in range(0, len(d_sess)-1):
        d_first = d_sess[i] + dt.timedelta(seconds=1) # avoid out of bound
        d_last = d_sess[i+1] - dt.timedelta(seconds=1) # avoid overlaps
        if ((d_sess[i] >= d_weeks[j+1])): # Change directory if new week
            j+=1
        monyea = d_weeks[j].strftime('%y%m%d')
        # indir_in = f'SIM_{monyea[:2]}/{XovOpt.get("expopt")}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
        indir_in = f'SIM_{monyea}/{XovOpt.get("expopt")}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
        pyaltsim_in.append([XovOpt.get("amplopt"), XovOpt.get("resopt"), indir_in, d_first, d_last, XovOpt.to_dict()])
    if grid:
        executor.update_parameters(slurm_name="pyaltsim",
                                   slurm_cpus_per_task=2,
                                   slurm_time=60*3, # minutes
                                   slurm_mem='10G') # 4GB for 10Hz
        # job = executor.submit(PyAltSim.main, [XovOpt.get("amplopt"), XovOpt.get("resopt"), indir_in, f'{monyea}', XovOpt.to_dict()]) # single job
        # print(job.result())
        jobs = executor.map_array(PyAltSim.main, pyaltsim_in)
        for job in jobs:
            print(job.results())
    else:
        for arg in pyaltsim_in:
            # PyAltSim.main([XovOpt.get("amplopt"), XovOpt.get("resopt"), indir_in, f'{monyea}', XovOpt.to_dict()])
            PyAltSim.main(arg)

if run_pyGeoLoc or run_pyXover or run_accuXover:
    XovOpt.set("sim_altdata", False)
    XovOpt.set("partials", True)
    XovOpt.set("parallel", False)
    XovOpt.set("SpInterp", 0)
    XovOpt.set("expopt", estid)
    d_sess = build_sessiontable_man(MANFIL,24,26)

    # Divide by weeks
    nWeeks = math.floor((d_sess[-1] - d_sess[0]).days/7)
    d_weeks = []
    for w in range(0,nWeeks+1):
        d_weeks.append(d_sess[0] + dt.timedelta(weeks=w))

    d_sess.extend(d_weeks[1:])
    d_sess.sort()

    d_weeks = d_weeks[:-3]
    print(d_weeks)


# WD should really organize the folders in weekly folders (should be done already for simulation)
# geolocation step
if run_pyGeoLoc:
    if grid:
        executor.update_parameters(slurm_name="pygeoloc",
                                   slurm_mem='1G',
                                   slurm_cpus_per_task=1,
                                   slurm_time=10) # minutes
    for i in range(0, len(d_weeks)-1):
        import glob
        d_first = d_weeks[i] + dt.timedelta(seconds=1) # avoid out of bound
        d_last = d_weeks[i+1] - dt.timedelta(seconds=1) # avoid overlaps
        # print(f"d_last {d_last}")
        monyea = d_first.strftime('%y%m%d')
        indir_in = f'SIM_{monyea}/{simid}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
        outdir_in = f'{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_{monyea}'
        pygeoloc_in = []
        epo_in = ""
        allFiles = glob.glob(os.path.join(f'{XovOpt.get("rawdir")}{indir_in}', f'{XovOpt.get("instrument")}*RDR*.*'))
        # Retrieve the dates from the names of all the files in the directory
        d_files = [dt.datetime.strptime(fil.split('.')[0][-10:], '%y%m%d%H%M')  for fil in allFiles[:]]
        # Add separation from arc-wise orbit fit
        for date in d_sess:
            if (date>=d_first and date<=d_last):
                d_files.append(date)
        d_files = list(set(d_files))
        d_files.sort()
        # print(d_files)
        if grid:
            import re
            import glob
            pattern = 'CALASIMRDR'
            pygeoloc_in = []
            for j in range(0,len(d_files)-1):
               d_first = d_files[j] + dt.timedelta(seconds=1) # avoid out of bound
               d_last = d_files[j+1] - dt.timedelta(seconds=1) # avoid overlaps
               # pygeoloc_in.append([epo_in, indir_in, outdir_in, d_files[j:j+2], 0, XovOpt.to_dict()])
               pygeoloc_in.append([epo_in, indir_in, outdir_in, [d_first, d_last], 0, XovOpt.to_dict()])
            if(len(pygeoloc_in) == 1):
               job = executor.submit(PyGeoloc.main, pygeoloc_in[0]) # single job
               print(job.result())
            else:
               jobs = executor.map_array(PyGeoloc.main, pygeoloc_in)
               for job in jobs:
                  print(job.results())
        else:
            # 4th argument ('BELASCIRDR') unused?
            PyGeoloc.main([f'{monyea}', indir_in, outdir_in, d_files, 0, XovOpt.to_dict()])


# # crossovers location step
if run_pyXover:
    XovOpt.set("parallel", False)  # not sure why, but parallel gets crazy
    XovOpt.set("weekly_sets", True)
    XovOpt.set("monthly_sets",True)
    # misy = ['310501','310508','310515','310522','310529','310605','310612']
    # misy = ['310501','310508','310515']
    misy = [date.strftime('%y%m%d') for date in d_weeks]
    misycmb = [x for x in itert.combinations_with_replacement(misy, 2)]
    print("Choose grid element among:",dict(map(reversed, enumerate(misycmb))))
    if grid: # WD: change MLASIMRDR?
        executor.update_parameters(slurm_name="pyxover",
                                   slurm_mem='12G',
                                   slurm_cpus_per_task=1,
                                   slurm_time=60*2, # minutes
                                   slurm_array_parallelism=30)
        pyxover_in = []
        for par in range(0,1):
        # for par in range(0,len(misycmb)):
            indir_in =  f'{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_'
            outdir_in = f'{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
            input_xov_path = XovOpt.get("outdir") + outdir_in + 'xov/xov_' + str(misycmb[par][0]) + '_' + str(misycmb[par][1]) + '.pkl'
            if os.path.exists(input_xov_path):
                print("input xov file already exists in", input_xov_path)
            else:
                pyxover_in.append([f'{par}',indir_in, outdir_in, misycmb[par], 0,XovOpt.to_dict()])
        if(len(pyxover_in) == 1):
            job = executor.submit(PyXover.main, pyxover_in[0]) # single job
            print(job.result())
        else:
            jobs = executor.map_array(PyXover.main, pyxover_in)
            for job in jobs:
                print(job.results())
    else:
        PyXover.main(['0', f'{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_',
                      f'{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/', 'MLASIMRDR', 0,
                      XovOpt.to_dict()])


# # lsqr solution step
if run_accuXover:
    XovOpt.set("expopt", estid)
    # XovOpt.set("sol4_glo", [None])
    out = AccumXov.main(
        [[f'{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'], '', 0,
         XovOpt.to_dict()])
