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
# LO = long. on planet's surface
# TH = lat. on planet's surface

from math import pi

import numpy as np
import pandas as pd
import spiceypy as spice
from scipy.special import lpmv

import astro_trans as astr
# mylib
from prOpt import SpInterp, tmpdir


##############################################

def set_const(h2_sol):
    from prOpt import pert_cloop

    h2 = 0.7 # 1.e-8  # 0.77 - 0.93 #Viscoelastic Tides of Mercury and the Determination
    l2 = 0.17  # 0.17-0.2    #of its Inner Core Size, G. Steinbrugge, 2018
    # https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2018JE005569
    tau = 0. #84480. # time lag in seconds, corresponding to 4 deg, G. Steinbrugge, 2018

    GMsun = 1.32712440018e20  # Sun's GM value (m^3/s^2)
    Gm = 0.022032e15  # Mercury's GM value (m^3/s^2)

    # print("pert_cloop['glo']['dh2']", pert_cloop['glo']['dh2'])
    # print('h2sol', h2_sol)
    # print('h2tot_pre',h2)

    # # check if h2 is perturbed
    if 'dh2' in pert_cloop['glo'].keys():
        h2 += pert_cloop['glo']['dh2']
    h2 += h2_sol
    # print('h2tot',h2)

    return h2, l2, tau, GMsun, Gm


def sind(x):
    return np.sin(np.deg2rad(x))


def cosd(x):
    return np.cos(np.deg2rad(x))

# compute cos of Sun zenith angle
def cosz(TH, LO, latSUN, lonSUN):

    return np.sin(TH) * np.sin(latSUN)+ np.cos(TH) * np.cos(latSUN) * np.cos(lonSUN - LO)


# @profile
def tidal_deform(vecopts, xyz_bf, ET, SpObj, delta_par):

    if isinstance(delta_par, dict) and 'dh2' in delta_par.keys():
        h2, l2, tau, GMsun, Gm = set_const(h2_sol=delta_par['dh2'])
    else:
        h2, l2, tau, GMsun, Gm = set_const(0)

    plarad = vecopts['PLANETRADIUS'] * 1.e3
    gSurf = Gm / np.square(plarad)  # surface g of body (ok)

    # dpr = 180 / pi

    [R, TH, LO] = astr.cart2sph(xyz_bf)

    LO0 = LO
    TH0 = TH
    CO = 90. - TH
    CO0 = CO
    nmax = 3

    obs = vecopts['PLANETNAME']
    frame = vecopts['PLANETFRAME']

    # get Sun position and distance from body
    if (SpInterp > 500):
        sunpos = np.transpose(SpObj['SUNx'].eval(ET-tau))
        merpos = np.transpose(SpObj['MERx'].eval(ET-tau))
        sunpos -= merpos
    else:
        sunpos, tmp = spice.spkpos('SUN', ET-tau, frame, 'NONE', obs)

    sunpos = 1.e3 * np.array(sunpos)
    dSUN = np.linalg.norm(sunpos, axis=1)
    [rSUN, latSUN, lonSUN] = astr.cart2sph(sunpos)

    coszSUN = cosz(TH, LO, latSUN, lonSUN)
    # print("xyz_bf",np.rad2deg(TH), np.rad2deg(LO),R)
    # print("lat, lon, r Sun",np.rad2deg(latSUN),np.rad2deg(lonSUN),rSUN)
    # print(coszSUN)

    # see, e.g., Van Hoolst, T., and Jacobs, C. ( 2003), Mercury's tides and interior structure,
    # J. Geophys. Res., 108, 5121, doi:10.1029/2003JE002126, E11.
    Psun = [lpmv(0, j, coszSUN) for j in range(2, nmax)]

    # TODO should use plarad (fix) or radius from measurement? (difference is up to 2km)
    terms = [(plarad / dSUN) ** j * Psun[j - 2] for j in range(2, nmax)]

    Vsun = (GMsun / (dSUN)) * np.sum(terms, 0)
    Vtot0 = Vsun

    # explicit equation for degree 2 term (not used here)
    # Vsun = (GMsun/(dSUN)) * np.square(plarad/dSUN) * 0.5*(3*np.square(coszSUN)-1);
    # apply to get vertical displacement of surface due to tides
    urtot = h2 * Vsun / gSurf
    # print(Psun)
    # print(Vsun)
    # print(gSurf)
    # print(h2)
    # exit()
    ##################################################
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3)

    axes[0].plot(np.rad2deg(LO),np.rad2deg(TH))
    axes[0].set(xlabel='lon (deg)', ylabel='lat (deg)')
    axes[0].set_xlim([-180.,180.])
    axes[0].set_ylim([-90.,90.])
    axes[0].plot(np.rad2deg(lonSUN),np.rad2deg(latSUN),linewidth=20)
    axes[1].plot(ET,coszSUN)
    axes[1].set(xlabel='ET (secJ2000)', ylabel='cosZ (Sun zenith)')
    axes[2].plot(ET,urtot)
    axes[2].set(xlabel='ET (secJ2000)', ylabel='ur_tid (m)')
    # axes[2].plot(ET,(GMsun / (dSUN))*np.power(plarad/dSUN,2))
    plt.savefig(tmpdir+'test_tid.png')

    import matplotlib.pyplot as plt
    from matplotlib import cm
    import imageio

    plt.clf()
    fig, axes = plt.subplots(1)
    x = np.deg2rad(np.arange(-180, 180, 0.1))
    y = np.deg2rad(np.arange(-90, 90, 0.1))
    lon, lat = np.meshgrid(x, y)
    tmp = cosz(lat, lon, 0., 0.)
    Psun = [lpmv(0, j, tmp) for j in range(2, nmax)]
    terms = [(plarad / dSUN[0]) ** j * Psun[j - 2] for j in range(2, nmax)]
    Vsun = (GMsun / (dSUN[0])) * np.sum(terms, 0)
    Vtot0 = Vsun
    urtot = h2 * Vsun / gSurf
    h = plt.contourf(np.rad2deg(lon), np.rad2deg(lat), urtot, cmap=cm.coolwarm)
    # h = axes.imshow(urtot, interpolation='nearest', cmap=cm.coolwarm)
    cbar = fig.colorbar(h)
    plt.savefig(tmpdir+'test_tid2.png')

    #loop on all Sun positions (176 Earth days)
    def plot_tides(d):
        step = 16
        sunpos, tmp = spice.spkpos('SUN', ET+step*d*86400., frame, 'NONE', obs)
        # print(ET,sunpos)
        sunpos = 1.e3 * np.array(sunpos)
        dSUN = np.linalg.norm(sunpos, axis=1)
        [rSUN, latSUN, lonSUN] = astr.cart2sph(sunpos)

        tmp = cosz(lat, lon, latSUN[0], lonSUN[0])
        Psun = [lpmv(0, j, tmp) for j in range(2, nmax)]
        terms = [(plarad / dSUN[0]) ** j * Psun[j - 2] for j in range(2, nmax)]
        Vsun = (GMsun / (dSUN[0])) * np.sum(terms, 0)
        Vtot0 = Vsun
        urtot = h2 * Vsun / gSurf
        # print(np.shape(urtot))
        # exit()

        plt.clf()
        fig, axes = plt.subplots(1)
        h = axes.contourf(np.rad2deg(lon), np.rad2deg(lat), urtot, cmap=cm.coolwarm, vmin=-2, vmax=2)
        axes.plot(np.rad2deg(lonSUN), np.rad2deg(latSUN), linewidth=20)
        axes.set(xlabel='LON (deg)', ylabel='LAT (deg)',title='Vertical tides over Mercury solar year (day '+str(d*step)+')')
        # h = axes.imshow(urtot, interpolation='nearest', cmap=cm.coolwarm)
        cbar = fig.colorbar(h,label='ur (meters)')
        # plt.savefig(tmpdir + 'test_tid3.png')
        # Used to return the plot as an image rray
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image, urtot

    kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
    tmp = np.array([plot_tides(d) for d in range(11)])
    urtot = np.stack(tmp[:,1])
    # print(urtot)
    urtot_max = np.max(urtot,axis=0)
    urtot_min = np.min(urtot,axis=0)
    # print(urtot_max-urtot_min)
    plt.clf()
    fig, axes = plt.subplots(1)
    h = axes.contourf(np.rad2deg(lon), np.rad2deg(lat), urtot_max-urtot_min, cmap=cm.coolwarm)
    axes.set(xlabel='LON (deg)', ylabel='LAT (deg)',
             title='Amplitude range of vertical tides over Mercury solar year')
    # h = axes.imshow(urtot, interpolation='nearest', cmap=cm.coolwarm)
    cbar = fig.colorbar(h, label='ur (meters)')
    plt.savefig(tmpdir + 'test_tid3.png')

    imageio.mimsave(tmpdir+'powers.gif', tmp[:,0], fps=1)


    exit()
    ###########################################

    # lon displacement
    dLO = 1. / 100.
    LO_ = LO0 + dLO

    # computation with perturbed LO_
    coszSUN = cosz(TH, LO_, latSUN, lonSUN)

    Psun = [lpmv(0, j, coszSUN) for j in range(2, nmax)]
    terms = [(plarad / dSUN) ** j * Psun[j - 2] for j in range(2, nmax)]

    Vsun = (GMsun / (dSUN)) * np.sum(terms, 0)
    Vtot = Vsun

    # compute derivative of V w.r.t. a lon displacement
    dV = (Vtot - Vtot0) / (np.deg2rad(dLO))
    # apply to get longitude displacement
    lotot = l2 * dV / (gSurf * sind(CO))

    # lat displacement - do in terms of CO = colatitude
    dCO = 1. / 100
    CO_ = CO0 + dCO
    CO_[CO_ > 180] = -1 * CO_[CO_ > 180]
    CO_ = np.deg2rad(CO_)

    # compute zenith angle from Sun with perturbed colatitude
    coszSUN = np.cos(CO_) * np.sin(latSUN) + np.sin(CO_) * np.cos(latSUN) * np.cos(lonSUN - LO)

    Psun = [lpmv(0, j, coszSUN) for j in range(2, nmax)]
    terms = [(plarad / dSUN) ** j * Psun[j - 2] for j in range(2, nmax)]

    Vsun = (GMsun / (dSUN)) * np.sum(terms, 0)
    Vtot = Vsun

    # compute derivative of V w.r.t. a lat displacement
    dV = (Vtot - Vtot0) / (np.deg2rad(dCO))
    # apply to get latitude displacement at surface
    thtot = l2 * dV / gSurf

    print(urtot,lotot,thtot)
    # exit()

    return urtot, lotot, thtot


def tidepart_h2(vecopts, xyz_bf, ET, SpObj, delta_par=0):
    # print(  'vecopts check',  vecopts['PARTDER'])
    # print("partial call", delta_par)

    dh2 = delta_par['dh2']
    # if isinstance(delta_par, pd.DataFrame) and 'dR/dh2' in delta_par.par.values:
    #     dh2 = delta_par.set_index('par').apply(pd.to_numeric, errors='ignore',
    #                                           downcast='float'
    #                                           ).to_dict('index')['dR/dh2']['sol']
    #
    # h2, l2, tau, GMsun, Gm = set_const(h2_sol=dh2)

    if isinstance(delta_par, dict) and 'dh2' in delta_par.keys():
        h2, l2, tau, GMsun, Gm = set_const(h2_sol=delta_par['dh2'])
    else:
        h2, l2, tau, GMsun, Gm = set_const(0)
    # print("dh2", dh2)

    return np.array(tidal_deform(vecopts, xyz_bf, ET, SpObj, delta_par={'dh2':dh2})[0]) / h2, 0., 0.
