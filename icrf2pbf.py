#!/usr/bin/env python3
# ----------------------------------
# icrf2pbf.py
#
# Description: Return a 3x3 matrix that transforms positions in inertial 
#    coordinates to positions in body-equator-and-prime-meridian 
#    coordinates (body-fixed).
# 
# Remark: translated from icrf2pbf.m, M. Barker
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 07-Feb-2019

from math import pi

import numpy as np


def icrf2pbf(ET, rotpar):

    p = rotpar['ORIENT']
    w = rotpar['NUT_PREC_PM']
    a = rotpar['NUT_PREC_ANGLES']

    RA0 = p[0, 0]
    DEC0 = p[1, 0]
    W0 = p[2, 0]
    RA1 = p[0, 1]
    DEC1 = p[1, 1]
    W1 = p[2, 1]
    RA2 = p[0, 2]
    DEC2 = p[1, 2]
    W2 = p[2, 2]

    ## time
    T = ET / (86400. * 365. * 100)  # sec per Julian century
    d = ET / 86400.

    ## Pole position
    RA = RA0 + RA1 * T + RA2 * np.square(T) / 2
    DEC = DEC0 + DEC1 * T + DEC2 * np.square(T) / 2

    ## libration amplitude
    rpd = pi / 180

    # amplibtmp = w(1)*sin(rpd*(174.791086+4.092335*d)) ...
    #     +w(2)*sin(rpd*(349.582171+8.184670*d)) ...
    #     +w(3)*sin(rpd*(164.373257+12.277005*d)) ...
    #     +w(4)*sin(rpd*(339.164343+16.369340*d)) ...
    #     +w(5)*sin(rpd*(153.955429+20.461675*d));
    amplibtmp = np.dot(w, np.transpose([np.sin(rpd * (a[:, 0] + a[:, 1] * t)) for t in d]))

    ## Longitude of the prime meridian
    W = W0 + W1 * d + W2 * np.square(d) / 2 + amplibtmp

    ## Rotation matrix
    # the R1,R2,R3 functions are defined as rotating vectors (rather than
    # rotating coordinate axes).  Hence, we need a minus sign on each angle.
    # see http://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/pck.html%Using%20the%20PCK%20System:%20Overview
    # see http://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/rotation.html
    # see http://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/tipbod_c.html
    ##
    alpha = -W * rpd
    M1 = np.reshape(np.hstack([np.column_stack((np.cos(alpha), -np.sin(alpha), np.zeros(W.shape))),
                               np.column_stack((np.sin(alpha), np.cos(alpha), np.zeros(W.shape))),
                               np.column_stack((np.zeros(W.shape), np.zeros(W.shape), np.ones(W.shape)))]), (-1, 3, 3))
    alpha = -(pi / 2 - DEC * rpd)
    M2 = np.reshape(np.hstack([np.column_stack((np.ones(DEC.shape), np.zeros(DEC.shape), np.zeros(DEC.shape))),
                               np.column_stack((np.zeros(DEC.shape), np.cos(alpha), -np.sin(alpha))),
                               np.column_stack((np.zeros(DEC.shape), np.sin(alpha), np.cos(alpha)))]), (-1, 3, 3))
    alpha = -(pi / 2 + RA * rpd)
    M3 = np.reshape(np.hstack([np.column_stack((np.cos(alpha), -np.sin(alpha), np.zeros(W.shape))),
                               np.column_stack((np.sin(alpha), np.cos(alpha), np.zeros(W.shape))),
                               np.column_stack((np.zeros(W.shape), np.zeros(W.shape), np.ones(W.shape)))]), (-1, 3, 3))
    #tsipm = 0  # mtimesx(M1,mtimesx(M2,M3));

    # Combine rotations
    tmp = np.einsum('ijk,ikl->ijl', M2, M3)

    return np.einsum('ijk,ikl->ijl', M1, tmp)
