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
import logging
from math import pi

import numpy as np
import pandas as pd
import spiceypy as spice
from scipy.special import lpmv

from xovutil import astro_trans as astr
from config import XovOpt

from wspice import spice_spkpos


##############################################

def set_const(h2_sol, central_body):
       
   if XovOpt.get('body') == 'MERCURY':
      h2 = 0.95 # 0.77 - 0.93 #Viscoelastic Tides of Mercury and the Determination
      # h2 = 2 # 0.77 - 0.93 #Viscoelastic Tides of Mercury and the Determination
      l2 = 0. # 0.17  # 0.17-0.2    #of its Inner Core Size, G. Steinbrugge, 2018
      # https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2018JE005569
      tau = 0. #84480. # time lag in seconds, corresponding to 4 deg, G. Steinbrugge, 2018

      Gm = 0.022032e15  # Mercury's GM value (m^3/s^2)
   elif XovOpt.get('body') == 'MOON':
      h2 = 0.04 # https://doi.org/10.1007/s00190-020-01455-8
      l2 = 0.
      tau = 0. # time lag in seconds,

      Gm = 4.90486959e12  # Moon's GM value (m^3/s^2)
   elif XovOpt.get('body') == 'CALLISTO':
      h2 = 1.2 # Genova et al 2022
      l2 = 0.
      tau = 0.
      Gm = 0.7179292e13  # Callisto's GM value (m^3/s^2)
   else:
      print(f"*** tidal_deform: {XovOpt.get('body')} not recognized.")
      exit()

   if central_body == 'SUN':
      GMsun = 1.32712440018e20  # Sun's GM value (m^3/s^2)
   elif central_body == 'JUPITER':
      GMsun = 0.1266865341960128e18  # Jupiter's GM value (m^3/s^2)
   elif central_body == 'EARTH':
      GMsun = 3.9860044188e14  # Earth's GM value (m^3/s^2)

   # # check if h2 is perturbed
   if 'dh2' in XovOpt.get("pert_cloop")['glo'].keys():
      h2 += XovOpt.get("pert_cloop")['glo']['dh2']
   h2 += h2_sol

   return h2, l2, tau, GMsun, Gm

# @profile
def tidal_deform(vecopts, R, TH, LO, ET, SpObj, central_body, delta_par):
   
   if isinstance(delta_par, dict) and 'dh2' in delta_par.keys():
        h2, l2, tau, GMsun, Gm = set_const(h2_sol=delta_par['dh2'], central_body=central_body)
   else:
        h2, l2, tau, GMsun, Gm = set_const(0, central_body=central_body)

   nmax = 3
   
   # get Sun position
   pertpos = get_sun_pos(SpObj, ET, tau, central_body, vecopts)

   # get tidal potential and derivatives
   Vsun, dVdLO, dVdCO = tidal_potential(vecopts, R, TH, LO, ET, pertpos, GMsun, nmax)
   
   plarad = vecopts['PLANETRADIUS'] * 1.e3
   gSurf = Gm / np.square(plarad)  # surface g of body (ok)
   
   CO = np.pi/2 - TH
   
   # apply to get vertical displacement of surface due to tides
   urtot = h2 * Vsun / gSurf

   # apply to get longitude displacement
   lotot = l2 * dVdLO / (gSurf * np.sin(CO))

   # apply to get latitude displacement at surface
   thtot = l2 * dVdCO / gSurf
   
   if XovOpt.get("debug") and XovOpt.get("local"):
      dSUN = np.linalg.norm(pertpos, axis=1)
      plot_whatever1(nmax, dSUN, GMsun, gSurf, h2)
      plot_whatever2(ET, nmax, GMsun, gSurf, h2, vecopts, central_body)
      exit()

   return urtot, lotot, thtot

def tidal_potential(vecopts, R, TH, LO, ET, pertpos, GMsun, nmax):

   plarad = vecopts['PLANETRADIUS'] * 1.e3

   LO0 = LO
   CO = np.pi/2 - TH
   CO0 = CO

   dSUN = np.linalg.norm(pertpos, axis=1)
   [rSUN, latSUN, lonSUN] = astr.cart2sph(pertpos)

   coszSUN = cosz(TH, LO, latSUN, lonSUN)

   # see, e.g., Van Hoolst, T., and Jacobs, C. ( 2003), Mercury's tides and interior structure,
   # J. Geophys. Res., 108, 5121, doi:10.1029/2003JE002126, E11.
   Psun = [lpmv(0, j, coszSUN) for j in range(2, nmax)]

   # TODO should use plarad (fix) or radius from measurement? (difference is up to 2km)
   terms = [(plarad / dSUN) ** j * Psun[j - 2] for j in range(2, nmax)]

   Vsun = (GMsun / (dSUN)) * np.sum(terms, axis=0)
   # explicit equation for degree 2 term (not used here)
   # Vsun = (GMsun/(dSUN)) * np.square(plarad/dSUN) * 0.5*(3*np.square(coszSUN)-1);

   if XovOpt.get("debug") and XovOpt.get("local"):
      plot_whatever(LO, TH, lonSUN, latSUN, ET, coszSUN)

   # lon displacement
   dLO = 1. / 100.
   LO_ = LO0 + dLO # radians

   # computation with perturbed LO_
   coszSUN = cosz(TH, LO_, latSUN, lonSUN)

   Psun = [lpmv(0, j, coszSUN) for j in range(2, nmax)]
   terms = [(plarad / dSUN) ** j * Psun[j - 2] for j in range(2, nmax)]

   Vtot0 = Vsun
   Vtot = (GMsun / (dSUN)) * np.sum(terms, 0)

   # compute derivative of V w.r.t. a lon displacement
   dVdLO = (Vtot - Vtot0) / dLO

   # lat displacement - do in terms of CO = colatitude
   dCO = 1. / 100.
   CO_ = CO0 + dCO
   CO_[CO_ > np.pi] = -1 * CO_[CO_ > np.pi]

   # compute zenith angle from Sun with perturbed colatitude
   coszSUN = np.cos(CO_) * np.sin(latSUN) + np.sin(CO_) * np.cos(latSUN) * np.cos(lonSUN - LO)

   Psun = [lpmv(0, j, coszSUN) for j in range(2, nmax)]
   terms = [(plarad / dSUN) ** j * Psun[j - 2] for j in range(2, nmax)]

   Vtot = (GMsun / (dSUN)) * np.sum(terms, 0)

   # compute derivative of V w.r.t. a lat displacement
   dVdCO = (Vtot - Vtot0) / dCO

   return Vsun, dVdLO, dVdCO

def tidepart_h2(vecopts, R, TH, LO, ET, SpObj, central_body, delta_par=0):

    dh2 = delta_par['dh2']
    # if isinstance(delta_par, pd.DataFrame) and 'dR/dh2' in delta_par.par.values:
    #     dh2 = delta_par.set_index('par').apply(pd.to_numeric, errors='ignore',
    #                                           downcast='float'
    #                                           ).to_dict('index')['dR/dh2']['sol']
    #
    # h2, l2, tau, GMsun, Gm = set_const(h2_sol=dh2)

    if isinstance(delta_par, dict) and 'dh2' in delta_par.keys():
        h2, l2, tau, GMsun, Gm = set_const(h2_sol=delta_par['dh2'], central_body=central_body)
    else:
        h2, l2, tau, GMsun, Gm = set_const(0, central_body=central_body)

    nmax = 3

    # get Sun position
    pertpos = get_sun_pos(SpObj, ET, tau, central_body, vecopts)

    # get tidal potential and derivatives
    Vsun, dVdLO, dVdCO = tidal_potential(vecopts, R, TH, LO, ET, pertpos, GMsun, nmax)
    
    plarad = vecopts['PLANETRADIUS'] * 1.e3
    gSurf = Gm / np.square(plarad)

    return np.array(Vsun/gSurf), 0., 0.

def get_sun_pos(SpObj, ET, tau, central_body, vecopts):
   
   # get Sun position
   if (XovOpt.get("SpInterp") > 0):
      if XovOpt.get('body') == 'MERCURY':
         pertpos = np.transpose(SpObj['SUNx'].eval(ET-tau))
         merpos = np.transpose(SpObj['MERx'].eval(ET-tau))
         pertpos -= merpos
      else:
         logging.error(f"** tides with spice_interp not implemented for {XovOpt.get('body')}.")
         exit()
      return  1.e3 * np.array(pertpos)
   else:
      return spice_spkpos(central_body, ET-tau, vecopts['PLANETFRAME'], vecopts['PLANETNAME'], "get_sun_pos")

# compute cos of Sun zenith angle
def cosz(TH, LO, latSUN, lonSUN):

    return np.sin(TH) * np.sin(latSUN)+ np.cos(TH) * np.cos(latSUN) * np.cos(lonSUN - LO)

def plot_whatever(LO, TH, lonSUN, latSUN, ET, coszSUN):

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
   plt.savefig(XovOpt.get("tmpdir") + 'test_tid.png')

def plot_whatever1(nmax, dSUN, GMsun, gSurf, h2):

   import matplotlib.pyplot as plt
   from matplotlib import cm
   
   plarad = vecopts['PLANETRADIUS'] * 1.e3

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
   plt.savefig(XovOpt.get("tmpdir") + 'test_tid2.png')

def plot_whatever2(ET,nmax, GMsun, gSurf, h2, vecopts, central_body):

   import matplotlib.pyplot as plt
   from matplotlib import cm
   import imageio
   
   obs = vecopts['PLANETNAME']
   frame = vecopts['PLANETFRAME']
   plarad = vecopts['PLANETRADIUS'] * 1.e3
   
   x = np.deg2rad(np.arange(-180, 180, 0.1))
   y = np.deg2rad(np.arange(-90, 90, 0.1))
   lon, lat = np.meshgrid(x, y)

   #loop on all Sun positions (176 Earth days)
   def plot_tides(d):
      step = 16
      pertpos = spice_spkpos(central_body, ET+step*d*86400., frame, obs, "plot_tides")
      dSUN = np.linalg.norm(pertpos, axis=1)
      [rSUN, latSUN, lonSUN] = astr.cart2sph(pertpos)

      tmp = cosz(lat, lon, latSUN[0], lonSUN[0])
      Psun = [lpmv(0, j, tmp) for j in range(2, nmax)]
      terms = [(plarad / dSUN[0]) ** j * Psun[j - 2] for j in range(2, nmax)]
      Vsun = (GMsun / (dSUN[0])) * np.sum(terms, 0)
      Vtot0 = Vsun
      urtot = h2 * Vsun / gSurf

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
   urtot_max = np.max(urtot,axis=0)
   urtot_min = np.min(urtot,axis=0)
   plt.clf()
   fig, axes = plt.subplots(1)
   h = axes.contourf(np.rad2deg(lon), np.rad2deg(lat), urtot_max-urtot_min, cmap=cm.coolwarm)
   axes.set(xlabel='LON (deg)', ylabel='LAT (deg)',
            title='Amplitude range of vertical tides over Mercury solar year')
   # h = axes.imshow(urtot, interpolation='nearest', cmap=cm.coolwarm)
   cbar = fig.colorbar(h, label='ur (meters)')
   plt.savefig(XovOpt.get("tmpdir") + 'test_tid3.png')

   imageio.mimsave(XovOpt.get("tmpdir") + 'powers.gif', tmp[:, 0], fps=1)