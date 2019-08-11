#!/usr/bin/env python3
# ----------------------------------

##############################################
# @profile
# local or PGDA
local = 1
# debug mode
debug = 0
# parallel processing?
parallel = 1

# compute partials?
partials = 1
# std perturbations for finite differences
parOrb = {'dA': 10.,'dC': 50.,'dR': 1.} #'dA': 10.,'dC': 50.,'dR': 1.} # 'dA': 10.,'dC': 50.,'dR': 1.} # 'dC':1., 'dRl':20e-6, 'dPt':20e-6} #, 'dA':100., 'dC':100., 'dR':20., } #, 'dRl':20e-6, 'dPt':20e-6}
parGlo = {'dRA':[0.01, 0.000, 0.000], 'dDEC':[0.01, 0.000, 0.000],'dPM':[0, 0.001, 0.000], 'dh2': 0.5} # 'dL':0.1,'dRA':[0.01, 0.000, 0.000], 'dDEC':[0.001, 0.000, 0.000], 'dPM':[0.0, 0.00001, 0.0], 'dh2': 1.}
# parameter constraints for solution
par_constr = {'dR/dRA':0.1, 'dR/dDEC':0.1,'dR/dPM':1, 'dR/dh2': 0.000000001,'dR/dA': 1,'dR/dC': 0.1,'dR/dR': 1} # ,'dR/dL':1,'dR/dRA':10, 'dR/dDEC':10, 'dR/dPM':10, 'dR/dh2': 10.} #'dR/dA': 50,'dR/dC': 50,'dR/dR': 100, 'dR/dh2': 10} #'dR/dA': 100,'dR/dC': 50,'dR/dR': 100} #, 'dR/dC': 50, 'dR/dL': 0.001, 'dR/dh2': 0.001}
# ... and closed loop sims
pert_cloop_orb = {} #{'dA':50., 'dC':50., 'dR':10.} # {'dA':50., 'dC':50., 'dR':10.} #'dA':100., 'dC':100., 'dR':20.} # {'dA':100., 'dC':100., 'dR':20.} # 'dRl':20e-6, 'dPt':60e-6} #
pert_cloop_glo = {} #{'dRA':[0.02, 0.000, 0.000], 'dDEC':[0.05, 0.000, 0.000], 'dPM':[0, 0.002, 0.000], 'dh2': 0.8} # 'dL':0.4} #, 'dh2': 1.8,'dRA':[0.02, 0.000, 0.000], 'dDEC':[0.05, 0.000, 0.000], 'dPM':[0.0, 0.0005, 0.0]} #'dC':100., 'dR':20., 'dRl':20e-6, 'dPt':20e-6}
pert_cloop = {'orb': pert_cloop_orb, 'glo': pert_cloop_glo}

pert_tracks = [] # ['1104281831','1206290713','1407081256','1503212147'] #'1301312356','1301101544','1301240758','1301281555','1301031543'] # ['1301010743', '1301011544', '1301012343']  # ['1301042351','1301012343']

# orbital representation
OrbRep = 'cnt' #'lin'
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
           'PARTDER': ''}


# out and aux
if (local == 0):
    outdir = '/att/nobackup/sberton2/MLA/out/'
    auxdir = '/att/nobackup/sberton2/MLA/aux/'
else:
#    outdir = '/home/sberton2/Works/NASA/Mercury_tides/out/'
    outdir = '/home/sberton2/Works/NASA/Mercury_tides/out/'
    auxdir = '/home/sberton2/Works/NASA/Mercury_tides/aux/'
