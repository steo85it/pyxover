#!/usr/bin/env python3
# ----------------------------------
# orient_setup.py
#
# Description: "manual" update rotational parameters
# 
# Remark: translated from setupROT.m, M. Barker
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 07-Feb-2019

import numpy as np

from config import XovOpt
import spiceypy as spice
from xovutil.units import as2deg

AG = False # True
ZAP = False

def orient_setup(offsetRA, offsetDEC, offsetPM, offsetL, offsetLIB):
   
    nc, POLE_RA0 = spice.bodvrd(XovOpt.get('body'), 'POLE_RA', 3)
    nc, POLE_DEC0 = spice.bodvrd(XovOpt.get('body'), 'POLE_DEC', 3)
    nc, PM0 = spice.bodvrd(XovOpt.get('body'), 'PM', 3)

    # Extrapolated from different a priori PM_rate and librations
    if XovOpt.get("vecopts")['PM_ORIGIN'] == 'J2013.0':
       if AG:
          PM0 = np.array([318.4455, 6.1385054, 0.])
          #PM0 = np.array([318.2245, 6.1385054, 0.])
       elif ZAP:
          # from zero
          PM0 = np.array([318.2245, 0., 0.])
       else:
          J2013 = 4748.5
          # WD: check for PM0[2] term
          PM0[0] = np.mod(PM0[0]+J2013*PM0[1] + PM0[2] * np.square(J2013) / 2, 360)          

    # WD: Find a generic way to retrieve these values
    if XovOpt.get('body') == 'MERCURY':
       n_nutpre = 11 #5
       nbody = '1'
    elif XovOpt.get('body') == 'CALLISTO':
       n_nutpre = 16
       nbody = '5'
    else:
       print(f"*** orient_setup: {XovOpt.get('body')} not recognized.")
       exit()

    nc, NUT_PREC_RA0     = spice.bodvrd(XovOpt.get('body'), 'NUT_PREC_RA', n_nutpre)
    nc, NUT_PREC_DEC0    = spice.bodvrd(XovOpt.get('body'), 'NUT_PREC_DEC', n_nutpre)
    nc, NUT_PREC_PM0     = spice.bodvrd(XovOpt.get('body'), 'NUT_PREC_PM', n_nutpre)
    nc, NUT_PREC_ANGLES0 = spice.bodvrd(nbody, 'NUT_PREC_ANGLES', 2*n_nutpre)
    
    if XovOpt.get('body') == 'CALLISTO':
       # Synthetic librations
       NUT_PREC_ANGLES0[-2] = 96.968560
       NUT_PREC_ANGLES0[-1] = 3729658.8916926
       
    rotpar = {'ORIENT0': '',
              'NUT_PREC_RA0'     : NUT_PREC_RA0,
              'NUT_PREC_DEC0'    : NUT_PREC_DEC0,
              'NUT_PREC_PM0'     : NUT_PREC_PM0,
              'NUT_PREC_ANGLES0' : np.vstack([[NUT_PREC_ANGLES0[i], NUT_PREC_ANGLES0[i+1]] for i in range(0,len(NUT_PREC_ANGLES0),2)])
              }


    rotpar['ORIENT0'] = np.vstack([POLE_RA0, POLE_DEC0, PM0])

    # Convert offsets to degrees or degrees/day and apply them
    POLE_RA = as2deg(offsetRA) + POLE_RA0
    POLE_DEC = as2deg(offsetDEC) + POLE_DEC0
    PM = as2deg(offsetPM)/365.25 + PM0

    upd_rotpar = {'ORIENT': '',
                  'NUT_PREC_RA'     : rotpar['NUT_PREC_RA0'],
                  'NUT_PREC_DEC'    : rotpar['NUT_PREC_DEC0'],
                  'NUT_PREC_PM'     : rotpar['NUT_PREC_PM0'],
                  'NUT_PREC_ANGLES' : rotpar['NUT_PREC_ANGLES0']
                  }

    upd_rotpar['NUT_PREC_PM'] +=  as2deg(offsetL)
    upd_rotpar['NUT_PREC_PM'][:len(offsetLIB)] +=  [as2deg(x) for x in offsetLIB]

    if AG:
        upd_rotpar['NUT_PREC_PM'] += as2deg(1.5)
    elif ZAP:
        upd_rotpar['NUT_PREC_PM'] = rotpar['NUT_PREC_PM0']


    if XovOpt.get("debug"):
        print("librations", rotpar['NUT_PREC_PM0'], offsetL * rotpar['NUT_PREC_PM0'])
        print(as2deg(offsetRA), as2deg(offsetDEC), as2deg(offsetPM)/365.25, as2deg(offsetL))
        # exit()

    upd_rotpar['ORIENT'] = np.vstack([POLE_RA, POLE_DEC, PM])

    return rotpar, upd_rotpar
