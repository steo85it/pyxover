#!/usr/bin/env python3
# ----------------------------------

##############################################
# @profile
# local or PGDA
local = 0
# debug mode
debug = 0
# simulation mode
sim = 1
# parallel processing?
parallel = 0
# compute partials?
partials = 0
# orbital representation
OrbRep = 'cnt' #'lin'
# interpolation/spice direct call (0:no, 1:yes, use, 2: yes, create)
SpInterp = 2
# interpolation/spice direct call (0:no, 1:yes, use, 2: yes, create)
new_gtrack = 1
# interpolation/spice direct call (0:no, 1:yes, use, 2: yes, create)
new_xov = 1

# out and aux
if (local == 0):
    outdir = '/att/nobackup/sberton2/MLA/out/'
    auxdir = '/att/nobackup/sberton2/MLA/aux/'
else:
#    outdir = '/home/sberton2/Works/NASA/Mercury_tides/out/'
    outdir = '/home/sberton2/Works/NASA/Mercury_tides/out1/'
    auxdir = '/home/sberton2/Works/NASA/Mercury_tides/aux/'
