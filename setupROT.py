#!/usr/bin/env python3
# ----------------------------------
# setupROT.py
#
# Description: "manual" update rotational parameters
# 
# Remark: translated from setupROT.m, M. Barker
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 07-Feb-2019

import numpy as np


def setupROT(offsetRA, offsetDEC, offsetPM, offsetL):

    # IAU
    POLE_RA0 = np.array([281.0097, -0.0328, 0.])
    POLE_DEC0 = np.array([61.4143, -0.0049, 0.])
    PM0 = np.array([329.5469, 6.1385025, 0.])
    # AG
    #POLE_RA0 = np.array([281.0082, -0.0328, 0.])
    #POLE_DEC0 = np.array([61.4164, -0.0049, 0.])
    #PM0 = np.array([329.75, 6.1385054, 0.])

    rotpar = {'ORIENT0': '',
              'NUT_PREC_PM0': np.transpose([0.00993822, \
                                            -0.00104581, \
                                            -0.00010280, \
                                            -0.00002364, \
                                            -0.00000532]),
              'NUT_PREC_ANGLES0': np.vstack([[174.791086, 4.092335], \
                                             [349.582171, 8.184670], \
                                             [164.373257, 12.277005], \
                                             [339.164343, 16.369340], \
                                             [153.955429, 20.461675]])
              }

    rotpar['ORIENT0'] = np.vstack([POLE_RA0, POLE_DEC0, PM0])

    POLE_RA = offsetRA + POLE_RA0
    POLE_DEC = offsetDEC + POLE_DEC0
    PM = offsetPM + PM0

    upd_rotpar = {'ORIENT': '',
                  'NUT_PREC_PM': (1 + offsetL) * rotpar['NUT_PREC_PM0'],
                  'NUT_PREC_ANGLES': rotpar['NUT_PREC_ANGLES0']
                  }

    upd_rotpar['ORIENT'] = np.vstack([POLE_RA, POLE_DEC, PM])

    return rotpar, upd_rotpar
