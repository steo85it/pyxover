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

# from examples.MLA.options import XovOpt.get("vecopts"), XovOpt.get("debug")
from config import XovOpt

from src.xovutil.units import as2deg

AG = False # True
ZAP = False

def orient_setup(offsetRA, offsetDEC, offsetPM, offsetL):

    if AG:
        POLE_RA0 = np.array([281.0082, -0.0328, 0.])
        POLE_DEC0 = np.array([61.4164, -0.0049, 0.])
        if XovOpt.get("vecopts")['PM_ORIGIN'] == 'J2000':
            PM0 = np.array([329.75, 6.1385054, 0.])
        elif XovOpt.get("vecopts")['PM_ORIGIN'] == 'J2013.0':
            PM0 = np.array([318.4455, 6.1385054, 0.]) # @J2013.0 (extrapolated with a priori PM_rate and librations)
            #PM0 = np.array([318.2245, 6.1385054, 0.])
    elif ZAP:
        # from zero
        POLE_RA0 = np.array([0., -0.0328, 0.])
        POLE_DEC0 = np.array([0., -0.0049, 0.])
        if XovOpt.get("vecopts")['PM_ORIGIN'] == 'J2000':
            PM0 = np.array([329.5469, 0., 0.])
        elif XovOpt.get("vecopts")['PM_ORIGIN'] == 'J2013.0':
            PM0 = np.array([318.2245, 0., 0.])  # @J2013.0 (extrapolated with a priori PM_rate and librations)
    else:
        # IAU
        POLE_RA0 = np.array([281.0103, -0.0328, 0.])
        POLE_DEC0 = np.array([61.4155, -0.0049, 0.])
        if XovOpt.get("vecopts")['PM_ORIGIN'] == 'J2000':
            PM0 = np.array([329.5988, 6.1385108, 0.])
        elif XovOpt.get("vecopts")['PM_ORIGIN'] == 'J2013.0':
            PM0 = np.array([318.3201, 6.1385108, 0.])  # @J2013.0 (extrapolated with a priori PM_rate and librations)
        #old weird mix
        #POLE_RA0 = np.array([281.0097, -0.0328, 0.])
        #POLE_DEC0 = np.array([61.4143, -0.0049, 0.])
        #if vecopts['PM_ORIGIN'] == 'J2000':
        #    PM0 = np.array([329.5469, 6.1385025, 0.])
        #elif vecopts['PM_ORIGIN'] == 'J2013.0':
        #    PM0 = np.array([318.2245, 6.1385025, 0.])  # @J2013.0 (extrapolated with a priori PM_rate and librations)

    rotpar = {'ORIENT0': '',
              'NUT_PREC_PM0': np.transpose([0.01067257,
                                            -0.00112309,
                                            -0.00011040,
                                            -0.00002539,
                                            -0.00000571]),
              'NUT_PREC_ANGLES0': np.vstack([[174.791086, 4.092335],
                                             [349.582171, 8.184670],
                                             [164.373257, 12.277005],
                                             [339.164343, 16.369340],
                                             [153.955429, 20.461675]])
              }

    rotpar['ORIENT0'] = np.vstack([POLE_RA0, POLE_DEC0, PM0])

    # Convert offsets to degrees or degrees/day and apply them
    POLE_RA = as2deg(offsetRA) + POLE_RA0
    POLE_DEC = as2deg(offsetDEC) + POLE_DEC0
    PM = as2deg(offsetPM)/365.25 + PM0

    upd_rotpar = {'ORIENT': '',
                  'NUT_PREC_PM': rotpar['NUT_PREC_PM0'] + as2deg(offsetL),
                  'NUT_PREC_ANGLES': rotpar['NUT_PREC_ANGLES0']
                  }
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
