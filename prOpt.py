#!/usr/bin/env python3
# ----------------------------------

##############################################
# @profile
# local or PGDA
local = 1
# debug mode
debug = 0
# parallel processing?
parallel = 0

# compute partials?
partials = 1
# std perturbations for finite differences
parOrb = {'dA': 100., 'dC':100., 'dR':20., } # 'dRl':20e-6, 'dPt':20e-6} #, 'dA':100., 'dC':100., 'dR':20., } #, 'dRl':20e-6, 'dPt':20e-6}
parGlo = {'dL':0.01,'dRA':[0.0001, 0.000, 0.000], 'dDEC':[0.001, 0.000, 0.000], 'dPM':[0.0, 0.00001, 0.0], 'dh2': 1.}
# ... and closed loop sims
pert_cloop_orb = {}  # 'dRl':20e-6, 'dPt':60e-6} #'dA':100., 'dC':100., 'dR':20.} #, 'dC':100., 'dR':2.} #, 'dRl':20e-6, 'dPt':20e-6}
pert_cloop_glo = {}  # 'dL':0.01,'dh2': .2} #'dRA':[0.0005, 0.000, 0.000]}#, 'dC':100., 'dR':20., 'dRl':20e-6, 'dPt':20e-6}
pert_cloop = {'orb': pert_cloop_orb, 'glo': pert_cloop_glo}

pert_tracks = []  # ['1301011544', '1301042351']  # ['1301042351','1301012343']

# orbital representation
OrbRep = 'cnt' #'lin'
# interpolation/spice direct call (0:no, 1:yes, use, 2: yes, create)
SpInterp = 2
# interpolation/spice direct call (0:no, 1:yes, use, 2: yes, create)
new_gtrack = 2
# interpolation/spice direct call (0:no, 1:yes, use, 2: yes, create)
new_xov = 2

# PyAltSim stuff
# simulation mode
sim = 0
# recompute a priori
new_illumNG = 0
# use topo
apply_topo = 1
# range noise
range_noise = 1

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
