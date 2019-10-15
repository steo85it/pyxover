#!/usr/bin/env python3
# ----------------------------------
import numpy as np
##############################################
# @profile
# local or PGDA
local = 0
# debug mode
debug = 0
# parallel processing?
parallel = 1

# compute partials?
partials = 1
# std perturbations for finite differences
parOrb = {'dA': 20.,'dC': 20.,'dR': 5., 'dRl':2e-6, 'dPt':2e-6} #
parGlo = {'dRA':[0.0001, 0.000, 0.000], 'dDEC':[0.0001, 0.000, 0.000],'dPM':[0, 1.e-8, 0.000], 'dL':1.e-2*np.linalg.norm([0.00993822,-0.00104581,-0.00010280,-0.00002364,-0.00000532])} #,'dh2': 0.1} #
# parameter constraints for solution
par_constr = {'dR/dRA': 100., 'dR/dDEC': 100.,'dR/dL':100, 'dR/dPM': 100., 'dR/dA':50, 'dR/dC':100.,'dR/dR':0.001, 'dR/dRl':1.e-13, 'dR/dPt':1.e-13} #, 'dR/dh2': 1} # 'dR/dA0':50., 'dR/dC0':50.,'dR/dR0':20, 'dR/dA1':1.e-2, 'dR/dC1':1.e-2,'dR/dR1':1.e-2} #
mean_constr = {'dR/dA':0.1, 'dR/dC':10,'dR/dR':0.1, 'dR/dRl':1.e-13, 'dR/dPt':1.e-13}
# ... and closed loop sims
pert_cloop_orb = {'dA':50., 'dC':50., 'dR':20.,'dRl':3e-5, 'dPt':3e-5} #, 'dA1':20.e-3, 'dC1':20.e-3, 'dR1':5.e-3} # {'dA':100., 'dC':100., 'dR':20.} #

pert_cloop_glo = {'dRA':[0.001, 0.000, 0.000], 'dDEC':[0.0013, 0.000, 0.000],'dPM':[0, 1.6e-5, 0.000], 'dL':0.03*np.linalg.norm([0.00993822,-0.00104581,-0.00010280,-0.00002364,-0.00000532]), 'dh2':0.} #
pert_cloop = {'orb': pert_cloop_orb, 'glo': pert_cloop_glo}

pert_tracks = [] #'1107021838','1210192326','1403281002','1503191143'] #
# select subset of parameters
sol4_orb = [] #'1107021838','1210192326','1403281002','1503191143']  #
sol4_orbpar = [None] #'dA','dC','dR','dRl','dPt'] # 'dA1','dC1','dR1']  #
sol4_glo = ['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dL'] #, 'dR/dh2'] #,  None]

# orbital representation
OrbRep = 'cnt' #'lin' #
# interpolation/spice direct call (0:no, 1:yes, use, 2: yes, create)
SpInterp = 1
# create new gtrack (0:no, 1:yes, check if present, 2: yes, create)
new_gtrack = 2
# create new xov (0:no, 1:yes, check if present, 2: yes, create)
new_xov = 2

# PyAltSim stuff
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


# out and aux
if (local == 0):
    outdir = '/att/nobackup/sberton2/MLA/out/'
    auxdir = '/att/nobackup/sberton2/MLA/aux/'
    tmpdir = '/att/nobackup/sberton2/MLA/tmp/'
else:
#    outdir = '/home/sberton2/Works/NASA/Mercury_tides/out/'
    outdir = '/home/sberton2/Works/NASA/Mercury_tides/out/'
    auxdir = '/home/sberton2/Works/NASA/Mercury_tides/aux/'
    tmpdir = '/home/sberton2/Works/NASA/Mercury_tides/PyXover/tmp/'
