#!/usr/bin/env python3
# ----------------------------------
# tidal_deform.py
#
# Description: Compute vertical tidal deformation
# due to Sun tides on Mercury (to be generalized)
# ----------------------------------
# Author: Stefano Bertone
# Created: 10-Feb-2019
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Legendre poly expansion
# Returns urtot, total radial displacement in meters
#
# Inputs:
# xyz_bf = body fixed coordinates of bouncing point (cartesian)
# ET = ephemeris time (at bouncing point)
#
# Variables:
# LO = long. on Moon's surface
# TH = lat. on Moon's surface

from math import pi

import numpy as np
import spiceypy as spice
from scipy.special import lpmv

import astro_trans as astr
# mylib
from prOpt import SpInterp

##############################################

h2 = 0.8  # 0.77 - 0.93 #Viscoelastic Tides of Mercury and the Determination
l2 = 0.17  # 0.17-0.2    #of its Inner Core Size, G. Steinbrugge, 2018
# https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2018JE005569

GMsun = 1.32712440018e20  # Sun's GM value (m^3/s^2)
Gm = 0.022032e15  # Mercury's GM value (m^3/s^2)


def sind(x):
    return np.sin(np.deg2rad(x))


def cosd(x):
    return np.cos(np.deg2rad(x))


def cosz(TH, LO, latSUN, lonSUN):
    dpr = 180 / pi
    return sind(TH) * sind(latSUN * dpr) + cosd(TH) * cosd(latSUN * dpr) * cosd(lonSUN * dpr - LO)


# @profile
def tidal_deform(vecopts, xyz_bf, ET, SpObj):
    plarad = vecopts['PLANETRADIUS'] * 1.e3
    gSurf = Gm / np.square(plarad)  # surface g of body

    # print(gSurf)

    dpr = 180 / pi

    [LO, TH, R] = astr.cart2sph(xyz_bf)

    LO0 = LO
    TH0 = TH
    CO = 90. - TH
    CO0 = CO
    nmax = 3

    obs = vecopts['PLANETNAME']
    frame = vecopts['PLANETFRAME']

    # get Sun position and distance from body
    if (SpInterp > 0):
        sunpos = np.transpose(SpObj['SUNx'].eval(ET))
        merpos = np.transpose(SpObj['MERx'].eval(ET))
        sunpos -= merpos
    else:
        sunpos, tmp = spice.spkpos('SUN', ET, frame, 'NONE', obs)

    sunpos = 1.e3 * np.array(sunpos)
    dSUN = np.linalg.norm(sunpos, axis=1)
    [lonSUN, latSUN, rSUN] = astr.cart2sph(sunpos)

    coszSUN = cosz(TH, LO, latSUN, lonSUN)

    Psun = [lpmv(0, j, coszSUN) for j in range(2, nmax)]
    terms = [(plarad / dSUN) ** j * Psun[j - 2] for j in range(2, nmax)]

    Vsun = (GMsun / (dSUN)) * np.sum(terms, 0)
    Vtot0 = Vsun

    # explicit equation for degree 2 term
    # Vsun = (GMsun/(dSUN)) * np.square(plarad/dSUN) * 0.5*(3*np.square(coszSUN)-1);
    # apply to get vertical displacement of surface due to tides
    urtot = h2 * (Vsun) / gSurf

    # lon displacement
    dLO = 1. / 100
    LO_ = LO0 + dLO

    coszSUN = cosz(TH, LO_, latSUN, lonSUN)

    Psun = [lpmv(0, j, coszSUN) for j in range(2, nmax)]
    terms = [(plarad / dSUN) ** j * Psun[j - 2] for j in range(2, nmax)]

    Vsun = (GMsun / (dSUN)) * np.sum(terms, 0)
    Vtot = Vsun

    # compute derivative of V w.r.t. a lon displacement
    dV = (Vtot - Vtot0) / (dLO / dpr)
    # apply to get longitude displacement
    lotot = l2 * dV / (gSurf * sind(CO))

    # lat displacement - do in terms of CO = colatitude
    dCO = 1. / 100
    CO_ = CO0 + dCO
    CO_[CO_ > 180] = -1 * CO_[CO_ > 180]

    coszSUN = cosd(CO_) * sind(latSUN * dpr) + sind(CO_) * cosd(latSUN * dpr) * cosd(lonSUN * dpr - LO)

    Psun = [lpmv(0, j, coszSUN) for j in range(2, nmax)]
    terms = [(plarad / dSUN) ** j * Psun[j - 2] for j in range(2, nmax)]

    Vsun = (GMsun / (dSUN)) * np.sum(terms, 0)
    Vtot = Vsun

    # compute derivative of V w.r.t. a lat displacement
    dV = (Vtot - Vtot0) / (dCO / dpr)
    # apply to get latitude displacement at surface
    thtot = l2 * dV / gSurf

    # print(urtot,lotot,thtot)
    # exit()

    return urtot, lotot, thtot


def tidepart_h2(vecopts, xyz_bf, ET, SpObj):
    # print(  'vecopts check',  vecopts['PARTDER'])

    return np.array(tidal_deform(vecopts, xyz_bf, ET, SpObj)[0]) / h2, 0., 0.
