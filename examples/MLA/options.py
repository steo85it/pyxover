#!/usr/bin/env python3
# ----------------------------------
import multiprocessing as mp

import numpy as np
from src.xovutil.units import deg2as

##############################################
# #@profile
# local or PGDA
local = 1
# debug mode
debug = 0
# parallel processing?
parallel = 0
# compute partials?
partials = 1

# processing names and experiments
datasimopt = 'sim'  # 'data' #
expopt = 'BS0' # 'BS2' # 'AGTP' # 'AGTP' # 'tp4' # 'KX1r4' #
resopt = [0]
amplopt = [1]

# std perturbations for finite differences (dRl, dPt, dRA, dDEC, dL in arcsec; dPM in arcsec/Julian year)
parOrb = {'dA': 20.,'dC': 20.,'dR': 5.} #,'dRl':0.2, 'dPt':0.2} #
parGlo = {'dRA':[0.2, 0.000, 0.000], 'dDEC':[0.36, 0.000, 0.000],'dPM':[0, 0.013, 0.000],'dL':1.e-3*deg2as(1.)*np.linalg.norm([0.00993822,-0.00104581,-0.00010280,-0.00002364,-0.00000532]), 'dh2': 0.1}

# parameter constraints for solution
par_constr = {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2,'dR/dL':1.e2, 'dR/dPM': 1.e2, 'dR/dh2':3.e-1, 'dR/dA':1.e2, 'dR/dC':1.e2,'dR/dR':2.e1} #, 'dR/dRl':5.e1, 'dR/dPt':5.e1} #
              # 'dR/dA1':1.e-1, 'dR/dC1':1.e-1,'dR/dR1':1.e-1, 'dR/dA2':1.e-2, 'dR/dC2':1.e-2,'dR/dR2':1.e-2} #, 'dR/dA2':1.e-4, 'dR/dC2':1.e-4,'dR/dR2':1.e-2} # 'dR/dA':100., 'dR/dC':100.,'dR/dR':100.} #, 'dR/dh2': 1} #
mean_constr = {'dR/dA':1.e0, 'dR/dC':1.e0,'dR/dR':1.e0} #, 'dR/dRl':1.e-1, 'dR/dPt':1.e-1}

# define if it's a closed loop simulation run
cloop_sim = False
# perturbations for closed loop sims (dRl, dPt, dRA, dDEC, dL in arcsec; dPM in arcsec/Julian year)
pert_cloop_orb = {} #'dA':50., 'dC':50., 'dR':20.,'dRl':0.5, 'dPt':0.5} #} #, 'dA1':20., 'dC1':20., 'dR1':5.
# in deg and deg/day as reminder pert_cloop_glo = {'dRA':[0.0015deg, 0.000, 0.000], 'dDEC':[0.0015deg, 0.000, 0.000],'dPM':[0, 2.e-6deg/day, 0.000],'dL':~3*1.5as, 'dh2':-1.} # compatible with current uncertitudes
pert_cloop_glo = {} #'dRA':[3.*5., 0.000, 0.000], 'dDEC':[3.*5., 0.000, 0.000],'dPM':[0, 3.*3., 0.000],'dL':3.*deg2as(1.5*0.03)*np.linalg.norm([0.00993822,-0.00104581,-0.00010280,-0.00002364,-0.00000532]), 'dh2':-1.} #
pert_cloop = {'orb': pert_cloop_orb, 'glo': pert_cloop_glo}
# perturb individual tracks
pert_tracks = [] #'1107021838','1210192326','1403281002','1503191143'] #

# select subset of parameters to solve for
sol4_orb = [None] #'1503250029'] #'1107021838','1210192326','1403281002','1503191143']  #
sol4_orbpar = [None] #['dA','dC','dR'] #,'dRl','dPt'] #,'dA1','dC1','dR1','dA2','dC2','dR2']  #] #
sol4_glo = ['dR/dRA', 'dR/dDEC', 'dR/dPM','dR/dL'] #,'dR/dh2'] #,  None]

# orbital representation
OrbRep = 'cnt' # 'lin' # 'quad' #

# interpolation/spice direct call (0: use spice, 1: yes, use interpolation, 2: yes, create interpolation)
SpInterp = 1
# create new gtrack (0:no, 1:yes, if not already present, 2: yes, create and replace)
new_gtrack = 2
# create new xov (0:no, 1:yes, if not already present, 2: yes, create and replace)
new_xov = 2

# Other options
# monthly or yearly sets for PyXover
monthly_sets = False
# analyze multi-xov pairs
multi_xov = False
# compute full covariance (could give memory issues)
full_covar = False # True #
# roughness map
roughn_map = False
# new algo
new_algo = True # False #
# load input xov
compute_input_xov = True

# PyAltSim options
# simulation mode
sim_altdata = 0
# recompute a priori
new_illumNG = 0
# use topo
apply_topo = 0
# range noise
range_noise = 0

# vecopts
# Setup some useful options
vecopts = {'SCID': '-236',
           'SCNAME': 'MESSENGER',
           'SCFRAME': -236000,
           'INSTID': (-236500, -236501),
           'INSTNAME': ('MSGR_MLA', 'MSGR_MLA_RECEIVER'),
           'PLANETID': '199',
           'PLANETNAME': 'MERCURY',
           'PLANETRADIUS': 2440.,
           'PLANETFRAME': 'IAU_MERCURY',
           'OUTPUTTYPE': 1,
           'ALTIM_BORESIGHT': '',
           'INERTIALFRAME': 'J2000',
           'INERTIALCENTER': 'SSB',
           'PM_ORIGIN':'J2013.0',
           'PARTDER': ''}


# useful directories for input/output
if (local == 0): # if pgda
    basedir = '/att/nobackup/sberton2/MLA/'
else: # if local!!!
    basedir = '/home/sberton2/Works/NASA/Mercury_tides/pyxover_release/examples/MLA/data/'

rawdir = f'{basedir}raw/'
outdir = f'{basedir}out/'
auxdir = f'{basedir}aux/'
tmpdir = f'{basedir}tmp/'
spauxdir = 'KX_spk/' #'AG_AC_spk/' #'KX_spk/' #'OD380_spk/' #'AG_spk/'

# pyxover options
# set number of processors to use
n_proc = mp.cpu_count() - 3
import_proj = False
import_abmat = (False,outdir+"sim/BS2_0/0res_1amp/Abmat*.pkl")