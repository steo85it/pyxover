#!/usr/bin/env python3
# ----------------------------------
# Description: spice routines wrapper
# returning nan if insufficient data
# ----------------------------------
# Author: William Desprats
# Created: 26-May-2025

import numpy as np
import spiceypy as spice
import warnings

def get_total_spk_coverage(targ):
    """
    Return a SPICE window containing *merged coverage* for target 'targ'
    across all loaded SPK kernels.
    """
    if isinstance(targ, str):
       idcode = targ
    else:
       idcode = spice.bodn2c(targ)

    # empty SPICE window
    cover_total = spice.cell_double(0)

    # Loop over all loaded SPK files
    count = spice.ktotal("SPK")
    for i in range(count):
        file, filetype, source, handle = spice.kdata(i, "SPK")

        # Each file's coverage window
        try:
            cover = spice.spkcov(file, idcode)
        except Exception:
            continue   # Ignore files not containing this ID

        # Union with the cumulative window
        cover_total = spice.wnunid(cover_total, cover)

    return cover_total
 
def et_in_total_spk(targ, et):
    et = np.atleast_1d(et)
    cover_total = get_total_spk_coverage(targ)

    inside = np.zeros(len(et), dtype=bool)
    for i in range(spice.wncard(cover_total)):
        left, right = spice.wnfetd(cover_total, i)
        inside |= (et >= left) & (et <= right)

    return inside
 
def spice_spkezr(targ, et, ref, obs, fname=""):
   """
   Safe wrapper around spice.spkezr that returns NaN for failed state lookups.
   Returns:
       scpos (N,3) position in meters
       scvel (N,3) velocity in meters/second
   """
   
   # Ensure ET is numpy array for consistent handling
   et = np.atleast_1d(et)

   try:
      scpv, _ = spice.spkezr(targ, et, ref, 'NONE', obs)
   except:
      import datetime as dt
      start = dt.datetime(2000, 1, 1, 12, 0, 0) + dt.timedelta(seconds=et[0])
      end = dt.datetime(2000, 1, 1, 12, 0, 0) + dt.timedelta(seconds=et[-1])
      warnings.warn(
         f"Vectorized spkezr failed for {targ} w.r.t {obs} in {fname}, "
         f"between et {start} and {end}."
         "Falling back to scalar mode."
         )
      
      mask = et_in_total_spk(targ, et)
      
      # Prepare full array with NaNs for missing times
      scpv = np.full((len(et), 6), np.nan)
      
      # If no valid ET values -> return all NaNs
      if mask.sum() != 0:
         # Valid times via vectorized spkezr
         scpv_valid, _ = spice.spkezr(targ, et[mask], ref, 'NONE', obs)
         scpv[mask, :] = scpv_valid

   scpv = np.atleast_2d(np.squeeze(scpv))
   # Convert km, km/s → m, m/s
   scpos = scpv[:, 0:3] * 1e3
   scvel = scpv[:, 3:6] * 1e3

   return scpos, scvel

def spice_spkpos(targ, et, ref, obs, fname=""):
   """
   Safe wrapper around spice.spkpos that returns NaN for failed state lookups.
   Returns:
       scpos (N,3) position in meters
   """
   
   # Ensure ET is numpy array for consistent handling
   et = np.atleast_1d(et)

   try:
      scpv, _ = spice.spkpos(targ, et, ref, 'NONE', obs)
   except:
      warnings.warn(
         f"Vectorized spkpos failed for {targ} w.r.t {obs} in {fname}. "
         "Falling back to scalar mode."
         )
      
      mask = et_in_total_spk(targ, et)
      
      # Prepare full array with NaNs for missing times
      scpv = np.full((len(et), 3), np.nan)
      
      # If no valid ET values -> return all NaNs
      if mask.sum() != 0:

         # Valid times via vectorized spkezr
         scpv_valid, _ = spice.spkpos(targ, et[mask], ref, 'NONE', obs)
      
         scpv[mask, :] = scpv_valid

   scpv = np.atleast_2d(np.squeeze(scpv))
   # Convert km → m
   scpos = scpv[:, 0:3] * 1e3

   return scpos

