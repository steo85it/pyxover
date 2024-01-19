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

import spiceypy as spice

from xovutil.units import as2deg

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
    elif XovOpt.get('body') == 'MERCURY':
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
    elif XovOpt.get('body') == 'CALLISTO':
        # IAU
        nc, POLE_RA0 = spice.bodvrd(XovOpt.get('body'), 'POLE_RA', 3)
        nc, POLE_DEC0 = spice.bodvrd(XovOpt.get('body'), 'POLE_DEC', 3)
        nc, PM0 = spice.bodvrd(XovOpt.get('body'), 'PM', 3)
    else:
       print(f"*** orient_setup: {XovOpt.get('body')} not recognized.")
       exit()

    if XovOpt.get('body') == 'MERCURY':
       rotpar = {'ORIENT0': '',
                 'NUT_PREC_RA0'     : np.transpose([0., 0., 0., 0., 0.]),
                 'NUT_PREC_DEC0'    : np.transpose([0., 0., 0., 0., 0.]),
                 'NUT_PREC_PM0'     : np.transpose([0.01067257,
                                                    -0.00112309,
                                                    -0.00011040,
                                                    -0.00002539,
                                                    -0.00000571]),
                 'NUT_PREC_ANGLES0' : np.vstack([[174.791086,0.14947253587500003E+06], #degrees/century
                                                [349.582171, 0.29894507175000006E+06],
                                                [164.373257, 0.44841760762500006E+06],
                                                [339.164343, 0.59789014350000012E+06],
                                                [153.955429, 0.74736267937499995E+06]])
                 # degrees/day
                 # 'NUT_PREC_ANGLES0' : np.vstack([[174.791086, 4.092335],
                 #                                [349.582171, 8.184670],
                 #                                [164.373257, 12.277005],
                 #                                [339.164343, 16.369340],
                 #                                [153.955429, 20.461675]])
                 }
    elif XovOpt.get('body') == 'CALLISTO':
       nc, NUT_PREC_RA0     = spice.bodvrd(XovOpt.get('body'), 'NUT_PREC_RA', 16)
       nc, NUT_PREC_DEC0    = spice.bodvrd(XovOpt.get('body'), 'NUT_PREC_DEC', 16)
       nc, NUT_PREC_PM0     = spice.bodvrd(XovOpt.get('body'), 'NUT_PREC_PM', 16)
       nc, NUT_PREC_ANGLES0 = spice.bodvrd('5', 'NUT_PREC_ANGLES', 2*16)
       # Synthetic librations
       NUT_PREC_ANGLES0[-2] = 96.968560
       NUT_PREC_ANGLES0[-1] = 3729658.8916926
       rotpar = {'ORIENT0': '',
                 'NUT_PREC_RA0'     : NUT_PREC_RA0,
                 'NUT_PREC_DEC0'    : NUT_PREC_DEC0,
                 'NUT_PREC_PM0'     : NUT_PREC_PM0,
                 'NUT_PREC_ANGLES0' : np.vstack([[NUT_PREC_ANGLES0[i], NUT_PREC_ANGLES0[i+1]] for i in range(0,len(NUT_PREC_ANGLES0),2)])
       }
    else:
       print(f"*** orient_setup: {XovOpt.get('body')} not recognized.")
       exit()

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
    if XovOpt.get('body') == 'MERCURY':
       upd_rotpar['NUT_PREC_PM'] +=  as2deg(offsetL)
    elif XovOpt.get('body') == 'CALLISTO':
       upd_rotpar['NUT_PREC_PM'][-1] +=  as2deg(offsetL)
       
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
