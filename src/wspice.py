#!/usr/bin/env python3
# ----------------------------------
# Description: spice routines wrapper
# returning nan if insufficient data
# ----------------------------------
# Author: William Desprats
# Created: 26-May-2025

import numpy as np
import spiceypy as spice

def spice_spkezr(targ, et, ref, obs, fname):
   # stable version os spkezr, setting nan where value is not found
   
   try:
      scpv, lt = spice.spkezr(targ, et, ref, 'NONE', obs)
   except:
      # Unvectorized
      print(f"Not all state vectors of {targ} w.r.t. {obs} were retieved in {fname}")
      scpv = []
      for et_loc in et:
         try:
            scpv_loc, lt = spice.spkezr(targ, et_loc, ref, 'NONE', obs)
         except:
            print("State vector not retrieved at ET=",et_loc)
            scpv_loc = [np.nan for i in range(0,6)]
         scpv.append(scpv_loc)
   scpv = np.atleast_2d(np.squeeze(scpv))

   scpos = 1.e3 * scpv[:, :3]
   scvel = 1.e3 * scpv[:, 3:]

   return scpos, scvel


def spice_spkpos(targ, et, ref, obs, fname):
   # stable version os spkpos, setting nan where value is not found
   
   try:
      scpos, lt = spice.spkpos(targ, et, ref, 'NONE', obs)
   except:
      # Unvectorized
      print(f"Not all position vectors of {targ} w.r.t. {obs} were retieved in {fname}")
      scpos = []
      for et_loc in et:
         try:
            scpos_loc, lt = spice.spkpos(targ, et_loc, ref, 'NONE', obs)
         except:
            print("Position vector not retrieved at ET=",et_loc)
            scpos_loc = [np.nan for i in range(0,3)]
         scpos.append(scpos_loc)
   scpos = np.atleast_2d(np.squeeze(scpos))
   # scpos = np.array(scpos)

   scpos *= 1.e3

   return scpos
