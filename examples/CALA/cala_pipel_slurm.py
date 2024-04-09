import logging
import os
import unittest
import submitit
import datetime as dt
from wd_utils import build_sessiontable_man
import math

from accumxov.accum_opt import AccOpt
from config import XovOpt

from accumxov import AccumXov
from pygeoloc import PyGeoloc
from pyxover import PyXover
import itertools as itert

# update paths and check options
from pyaltsim import PyAltSim

grid = True
run_pyAltSim  = False # 3-4 hours per month, 30min-1h15 per week (up to 2.8GB)
run_pyGeoLoc  = True # quite fast per gtrack
run_pyXover   = True # 10min or 20min
run_accuXover = False

camp = "/storage/research/aiub_gravdet/WD_XOV"
OrbDir = f"{camp}/ORB/"
SIMIDBSW   = "Am0"  # Simulation ID
ESTIDBSW   = "Ph3i1"
# ESTIDBSW   = "Pg9i2A"
ORBID      = "034"  # Input CR3BP orbit
MANFIL = f"{OrbDir}CAL_{ORBID}_{SIMIDBSW}_CR3BP.ORB"

#region old tests
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
# CC0 : gtracks and xov North from CB2 with Jf5
# CC1 : gtracks and xov South from CB2 with Jf5
# CC2 : simulation from Ah7 (sampling 1s)
# CC2 : gtracks and xov North from CC2 with Ah7
# CC3 : simulation from Ah4 (sampling 10Hz)
# CC3 : gtracks and xov North from CC3 with Jf7
# CC4 : gtracks and xov South from CC3 with Jf7
# CC5 : simulation from Ah4 (sampling 10Hz)
# CC5 : gtracks and xov North from CC5 with Jf7
# CC6 : gtracks and xov South from CC5 with Jf7
# CC7 : gtracks and xov North from CB2 with Pb7i2R
# CC8 : gtracks and xov South from CB2 with Pb7i2R
# CC9 : gtracks and xov North from CC5 with Pc0i1
# CD0 : gtracks and xov South from CC5 with Pc0i1
# CD1 : simulation from Ah9 (sampling 10Hz)
# CD1 : gtracks and xov North from CD1 with Pc5i1
# CD2 : gtracks and xov South from CD1 with Pc5i1
# CD3 : gtracks and xov North from CD1 with Pc8i1
# CD4 : gtracks and xov South from CD1 with Pc8i1
# CD5 : gtracks and xov North from CC5 with Pc9i1
# CD6 : gtracks and xov South from CC5 with Pc9i1
# CD7 : gtracks and xov North from CB2 with Pc6i1
# CD8 : gtracks and xov South from CB2 with Pc6i1
# CD9 : gtracks and xov North from CD1 with Pd9i2
# CE0 : gtracks and xov South from CD1 with Pd9i2
#endregion

#region error h2
# CE1 : simulation from Am0 (sampling 10Hz)
# CE1 : gtracks and xov North from CE1 with Pe0i1
# CE2 : simulation from Ah9 (sampling 10Hz)
# CE3 : simulation from Am0 (sampling 10Hz) small scale topo
# CE3 : gtracks and xov North from CE3 with Pe1i1
# CE4 : simulation from Am0 (sampling 10Hz) large scale topo
# CE4 : gtracks and xov North from CE4 with Pe1i1
# CE5 : simulation from Am0 (sampling 10Hz) small/large scale topo
# CE5 : gtracks and xov North from CE5 with Pe1i1
# CE6 : gtracks and xov South from CE1 with Pe1i1
# CE7 : gtracks and xov South from CE3 with Pe1i1
# CE8 : gtracks and xov South from CE4 with Pe1i1
# CE9 : gtracks and xov South from CE5 with Pe1i1
# CF0 : gtracks and xov North from CE3 with Am0
# CF1 : gtracks and xov South from CE3 with Am0
# CF2 : gtracks and xov North from CE4 with Am0
# CF3 : gtracks and xov South from CE4 with Am0
# CF4 : gtracks and xov North from CE1 with Am0
# CF5 : gtracks and xov South from CE1 with Am0
# CF6 : simulation from Am0 (sampling 10Hz) w/ noise 0.5m (was h2 corrected ?) (noise was not applied...)
# CF6 : gtracks and xov North from CF6 with Pe1i1
# CF7 : gtracks and xov South from CF6 with Pe1i1
#endregion

#region corrected h2
# CF8 : simulation from Am0 (sampling 30Hz) small scale topo
# CF8: gtracks and xov North from CF8 with Pe1i1
# CF9: gtracks and xov South from CF8 with Pe1i1
# CG0 : simulation from Am0 (sampling 10Hz) small/large scale topo
# CG0: gtracks and xov North from CG0 with Pe1i1
# CG1: gtracks and xov South from CG0 with Pe1i1
# CG2 : simulation from Am0 (sampling 10Hz) no topo
# CG2: gtracks and xov North from CG2 with Pe1i1
# CG3: gtracks and xov South from CG2 with Pe1i1
# CG4 : simulation from Al9 (sampling 10Hz) no topo
# CG4: gtracks and xov North from CG4 with Pf0i1
# CG5: gtracks and xov South from CG4 with Pf0i1
# CG6 : simulation from Al9 (sampling 10Hz) small/large scale topo
# CG6: gtracks and xov North from CG6 with Pf0i1
# CG7: gtracks and xov South from CG6 with Pf0i1
# CG8: gtracks and xov North from CG4 with Pf0i1 (dL = 0.1) instead of 0.01
# CG9: gtracks and xov North from CG4 with Pf0i1 (dL = 0.001) instead of 0.01
# CH0: simulation from Am0 (sampling 10Hz) large scale topo
# CH0: gtracks and xov North from CH0 with Pe1i1
# CH1: gtracks and xov South from CH0 with Pe1i1
# CH2: simulation from Am0 (sampling 10Hz) noise 10m
# CH2: gtracks and xov North from CH2 with Pe1i1
# CH3: gtracks and xov South from CH2 with Pe1i1
# CH4: simulation from Am0 (sampling 10Hz) noise 15m
# CH4: gtracks and xov North from CH4 with Pe1i1
# CH5: gtracks and xov South from CH4 with Pe1i1
# CH6: gtracks and xov North from CG2 with Pf4i1
# CH7: gtracks and xov South from CG2 with Pf4i1
# CH8: simulation from Am0 (sampling 30Hz) small scale topo + large scale
# CH8: gtracks and xov North from CH8 with Pe1i1
# CH9: gtracks and xov South from CH8 with Pe1i1
# CI0: gtracks and xov North from CG0 with Pf4i1 done
# CI1: gtracks and xov South from CG0 with Pf4i1 done
# CI2: gtracks and xov North from CG0 with Pe8i2B done
# CI3: gtracks and xov South from CG0 (CG1) with Pe8i2B done
# CI4: gtracks and xov North from CH0 with Pf2i2I done
# CI5: gtracks and xov South from CH0 (CH1) with Pf2i2I done
# CI6: gtracks and xov North from CG2 with Pe9i2O done
# CI7: gtracks and xov South from CG2 (CG3) with Pe9i2O done
# CI8: gtracks and xov North from CH2 with Pf3i2B done (hope geoloc is okay)
# CI9: gtracks and xov South from CH2 (CH3) with Pf3i2B done
# CJ0: gtracks and xov North from CG4 with Pf0i2F done
# CJ1: gtracks and xov South from CG4 (CG5) with Pf0i2F done
# CJ2: gtracks and xov North from CG6 with Pf1i2B done
# CJ3: gtracks and xov South from CG6 (CG7) with Pf1i2B done
# CJ4: gtracks and xov North from CH8 with Pf5i2B done
# CJ5: gtracks and xov South from CH8 (CH9) with Pf5i2B
# CJ6: simulation from Am1 (sampling 10Hz) small scale topo + large scale
# CJ6: gtracks and xov North from CJ6 with Pf7i1
# CJ7: gtracks and xov South from CJ6 with Pf7i1
# CJ8: gtracks and xov North from CJ6 with Pf8i1
# CJ9: gtracks and xov South from CJ6 with Pf8i1
# CK0: gtracks and xov North from CG0 with Pf9i1
# CK1: gtracks and xov South from CG0 with Pf9i1
# CK2: gtracks and xov North from CG0 with Pg0i1
# CK3: gtracks and xov South from CG0 with Pg0i1
# CK4: gtracks and xov North from CG0 with Pg1i1
# CK5: gtracks and xov South from CG0 with Pg1i1
# CK6: gtracks and xov North from CG6 with Pg2i1
# CK7: gtracks and xov South from CG6 with Pg2i1
# CK8: gtracks and xov North from CG6 with Pg3i1
# CK9: gtracks and xov South from CG6 with Pg3i1
#endregion

# corrected rotation
# CL0 : simulation from Am0 (sampling 10Hz) small/large scale topo
# CL0 : gtracks and xov North from CL0 with Pg5i1
# CL1 : gtracks and xov South from CL0 with Pg5i1
# CL2 : simulation from Al9 (sampling 10Hz) small/large scale topo
# CL2 : gtracks and xov North from CL2 with Pg6i1
# CL3 : gtracks and xov South from CL2 with Pg6i1
# CL4 : simulation from Am1 (sampling 10Hz) small/large scale topo
# CL4 : gtracks and xov North from CL4 with Pg7i1
# CL5 : gtracks and xov South from CL4 with Pg7i1
# CL6 : simulation from Am0 (sampling 10Hz) no topo (to remove)
# CL6 : gtracks and xov South from CL6 with Pg5i1 (to remove)
# CL7 : gtracks and xov North from CL6 with Pg5i1 (to remove)
# CL8 : simulation from Am0 (sampling 10Hz) large scale topo
# CL8 : gtracks and xov South from CL8 with Pg5i1
# CL9 : gtracks and xov North from CL8 with Pg5i1
# CM0: simulation from Am0 (sampling 10Hz) noise 10m
# CM0 : gtracks and xov North from CM0 with Pg5i1
# CM1 : gtracks and xov South from CM0 with Pg5i1
# CM2 : simulation from Am0 (sampling 10Hz) no topo
# CM3 : gtracks and xov North from CL6 with Pg5i1 to remove
# CM2 : gtracks and xov North from CM2 with Pg5i1
# CM4 : gtracks and xov South from CM2 with Pg5i1
# CM5 : simulation from Am0 (sampling 10Hz) small/large scale topo 30 Hz
# CM5 : gtracks and xov North from CM5 with Pg5i1
# CM6 : gtracks and xov South from CM5 with Pg5i1
# CM7 : simulation from Am0 (sampling 10Hz) noise 15m
# CM7 : gtracks and xov North from CM7 with Pg5i1
# CM8 : gtracks and xov South from CM7 with Pg5i1
# CM9 : simulation from Am0 (sampling 10Hz) noise 13m
# CM9 : gtracks and xov North from CM7 with Pg5i1
# CN0 : gtracks and xov South from CM7 with Pg5i1
# CN1 : simulation from Am0 (sampling 10Hz) noise 12m
# CN1 : gtracks and xov North from CM7 with Pg5i1
# CN2 : gtracks and xov South from CM7 with Pg5i1
# CN3 : gtracks and xov North from CM2 with Pg8i2A
# CN4 : gtracks and xov South from CM2 with Pg8i2A
# CN5 : gtracks and xov North from CL8 with Pg9i2A
# CN6 : gtracks and xov South from CL8 with Pg9i2A

# CM2 : simulation from Am0 (sampling 10Hz) no topo
# CN7 : gtracks and xov North from CM2 with Pg5i1
# CN8 : gtracks and xov South from CM2 with Pg5i1
# CL0 : simulation from Am0 (sampling 10Hz) small/large scale topo
# CN9 : gtracks and xov North from CL0 with Pg5i1
# CO0 : gtracks and xov South from CL0 with Pg5i1


XovOpt.set("local",True)
        
simid = 'CL0'
estid_N = 'CN9'
estid_S = 'CO0'
XovOpt.set("selected_hemisphere",'N')
# XovOpt.set("import_proj",True)
# XovOpt.set("compute_input_xov",False) # to use already computed rough xov (in xov/tmp/*pkl.gz)
# XovOpt.set("new_xov",False) # to replace previous xov (rough or final?)
XovOpt.set("debug",False)
XovOpt.set("spice_meta",f'mymeta')
if run_pyAltSim:
   XovOpt.set("spice_spk",[f'{camp}/ORB/CAL{SIMIDBSW}31121.SPK'])
elif run_pyGeoLoc:
   XovOpt.set("spice_spk",[f'{camp}/ORB/dA{ESTIDBSW}311210.SPK'])

if XovOpt.get("selected_hemisphere") == 'N':
   estid = estid_N
else:
   estid = estid_S

XovOpt.set("body", 'CALLISTO')
XovOpt.set("basedir", f'{camp}/pyXover/')
# XovOpt.set("basedir", "examples/CALA/data/")
XovOpt.set("instrument", 'CALA')

# Subset of parameters to solve for
# For "sol4_orb" and "sol4_orbpar, laisser une liste vide signifie "tous", sinon mettre "None" pour pas estimer
XovOpt.set("sol4_orb", []) # pour quelles orbites tu veux estimer les parametres "sol4_orbpar"
XovOpt.set("sol4_orbpar", []) # quels parametres d'orbite, quelle direction ou pointing tu veux estimer
# XovOpt.set("sol4_glo", ['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dh2'])
# XovOpt.set("sol4_glo", ['dR/dh2']) #'dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dL', 'dR/dh2'])
XovOpt.set("sol4_glo", ['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dL', 'dR/dh2'])

XovOpt.set("parGlo", {'dRA': [0.2, 0.000, 0.000], 'dDEC': [0.36, 0.000, 0.000], 'dPM': [0, 0.013, 0.000],
              'dL': 0.01, 'dh2': 0.1})

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
           'PM_ORIGIN': 'J2000',
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
XovOpt.set("apply_topo",False) # use large scale topography (DEM)
XovOpt.set("small_scale_topo", False) # apply small scale topography (simulated)
XovOpt.set("spauxdir", 'CAL_spk/')


# interpolation/spice direct call (0: use spice, 1: yes, use interpolation, 2: yes, create interpolation)
XovOpt.set("SpInterp", 0)  # TODO for some reason, SPICE interpolation gave weird results... beware+correct


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

# XovOpt.set("parallel", True)
XovOpt.set("parallel", False)
XovOpt.set("new_illumNG", True)
# this restricts simulated data to the first day of the month (see d_last in PyAltSim.main)
XovOpt.set("unittest", False)
# XovOpt.set("debug", False)

if grid:
    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder="log_slurm")
    # set timeout in min, and partition for running the job
    executor.update_parameters(
        slurm_partition="aiub",#epyc2, aiub
        slurm_cpus_per_task=1,
        slurm_nodes=1,
        slurm_array_parallelism=100)

# pyaltsim_in = [[f'{yy:02d}{mm:02d}', f'SIM_{yy:02d}/KX5/0res_1amp/', f'sim/{XovOpt.get("expopt")}_{iternum}/0res_1amp/gtrack_{yy:02d}', 'MLASCIRDR', iternum, opt_dict] for mm in range(13)[1:] for yy in [8,11,12,13,14,15]]

# Altimetry simulation
# WD should really organize the folders in weekly folders
if run_pyAltSim:
    XovOpt.set("sim_altdata", True)
    XovOpt.set("partials", False)
    XovOpt.set("range_noise", True)
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
        pyaltsim_in.append(
            [XovOpt.get("amplopt"), XovOpt.get("resopt"), indir_in,  d_first, d_last, XovOpt.to_dict()])
    # pyaltsim_in = pyaltsim_in[1:]
    if grid:
        executor.update_parameters(slurm_name="pyaltsim",
                                   slurm_cpus_per_task=2,
                                   slurm_time=60*3, # minutes
                                   slurm_mem='10G') # 4GB for 10Hz
        # job = executor.submit(PyAltSim.main, [XovOpt.get("amplopt"), XovOpt.get("resopt"), indir_in, f'{monyea}', XovOpt.to_dict()]) # single job
        # print(job.result())
        # print(pyaltsim_in)
        # pyaltsim_in = pyaltsim_in[0]
        # print(pyaltsim_in)
        if len(pyaltsim_in) == 1:
              job = executor.submit(PyAltSim.main, pyaltsim_in[0]) # single job
              print(job.result())
        else:
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

    if True:
        d_sess = build_sessiontable_man(MANFIL, 24, 26)
        d_sess = d_sess[:81]
        d_verylast = d_sess[80]
    else:
        d_sess = [dt.datetime(2031, 5, 1, 0, 0, 1),
                  dt.datetime(2031, 5, 17, 18, 41, 15),
                  dt.datetime(2031, 6, 3, 13, 23, 18),
                  dt.datetime(2031, 6, 20, 8, 6, 13),
                  dt.datetime(2031, 7, 7, 2, 50, 1),
                  dt.datetime(2031, 7, 23, 21, 34, 35),
                  dt.datetime(2031, 8, 9, 16, 19, 40)]

        maneuver_time = [988632000, 990081675, 991531398, 992981173, 994431001, 995880875, 997330780]
        mjd2000 = dt.datetime(2000, 1, 1, 12, 0, 0)

        d_sess = []
        for sec in maneuver_time:
            d_sess.append(mjd2000 + dt.timedelta(seconds=sec))

    # Divide by weeks
    nWeeks = math.floor((d_sess[-1] - d_sess[0]).days / 7)
    nWeeks = 12
    d_weeks = []
    for w in range(0, nWeeks + 1):
        d_weeks.append(d_sess[0] + dt.timedelta(weeks=w))

    d_sess.extend(d_weeks[1:-1])
    d_sess.sort()

    # d_weeks = d_weeks[:-1]
    # d_weeks = d_weeks[-3:]
    # d_weeks = d_weeks[2:]
    # d_weeks = d_weeks[11:13]
    # d_weeks = d_weeks[6:8]# test
    print(d_weeks)

# WD should really organize the folders in weekly folders (should be done already for simulation)
# geolocation step
if run_pyGeoLoc:
    if grid:
        executor.update_parameters(slurm_name="pygeoloc",
                                   slurm_mem='1G',
                                   slurm_cpus_per_task=1,
                                   slurm_time=20) # minutes
    for i in range(0, len(d_weeks) - 1):
        import glob
        d_first = d_weeks[i] + dt.timedelta(seconds=1) # avoid out of bound
        d_last = d_weeks[i + 1] - dt.timedelta(seconds=1)  # avoid overlaps
        # print(f"d_last {d_last}")
        monyea = d_first.strftime('%y%m%d')
        indir_in = f'SIM_{monyea}/{simid}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
        outdir_in = f'{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_{monyea}'
        pygeoloc_in = []
        epo_in = ""
        allFiles = glob.glob(os.path.join(f'{XovOpt.get("rawdir")}{indir_in}', f'{XovOpt.get("instrument")}*RDR*.*'))
        # allFiles = allFiles[19:20] # WD test
        # Retrieve the dates from the names of all the files in the directory
        d_files = [dt.datetime.strptime(fil.split('.')[0][-10:], '%y%m%d%H%M')  for fil in allFiles[:]]
        # Take only the files in the time span
        d_files = [date for date in d_files if date <= d_verylast]
        # Add separation from arc-wise orbit fit
        for date in d_sess:
            if (date>=d_first and date<=d_last):
                d_files.append(date)
        d_files = list(set(d_files))
        d_files.sort()
        # d_files = d_files[30:32] #test
        # print(d_files)
        pattern = 'CALASIMRDR'
        pygeoloc_in = []
        for j in range(0,len(d_files)-1):
           d_first = d_files[j] + dt.timedelta(seconds=1) # avoid out of bound
           d_last = d_files[j+1] - dt.timedelta(seconds=1) # avoid overlaps
           if d_last>d_first + dt.timedelta(minutes=1):
              # pygeoloc_in.append([epo_in, indir_in, outdir_in, d_files[j:j+2], 0, XovOpt.to_dict()])
              pygeoloc_in.append([epo_in, indir_in, outdir_in, [d_first, d_last], 0, XovOpt.to_dict()])
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


# # crossovers location step
if run_pyXover:
    XovOpt.set("parallel", False)  # not sure why, but parallel gets crazy
    XovOpt.set("weekly_sets", True)
    XovOpt.set("monthly_sets", False)
    indir_in =  f'{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_'
    outdir_in = f'{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'

    misy = [date.strftime('%y%m%d') for date in d_weeks[:-1]]
    misycmb = [x for x in itert.combinations_with_replacement(misy, 2)]
    # print(misycmb)
    # misycmb = [('310501','310703'),('310515','310626')]
    print("Choose grid element among:", dict(map(reversed, enumerate(misycmb))))
    if grid:  # WD: change MLASIMRDR? SB: YES, name should be an option
        executor.update_parameters(slurm_name="pyxover",
                                   slurm_mem='15G', #12
                                   slurm_cpus_per_task=1,
                                   slurm_time=60*5, # minutes
                                   slurm_array_parallelism=50)
        pyxover_in = []
        # for par in range(0,1):
        for par in range(0,len(misycmb)):
            # create symlink to rough xovs from other tests
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
       par = 0
       PyXover.main([f'{par}',indir_in, outdir_in, misycmb[par], 0, XovOpt.to_dict()])
        # PyXover.main(['0', f'{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_',
        #               f'{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/', 'MLASIMRDR', 0,
        #               XovOpt.to_dict()])


# # lsqr solution step
if run_accuXover:
    XovOpt.set("expopt", estid_N)
    # datasets = [f'{estid_N}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/',
    #            f'{estid_S}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/']
    datasets = [f'{estid_N}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/']
    # XovOpt.set("sol4_glo", [None])
    out = AccumXov.main([datasets, '', 0, XovOpt.to_dict()])
