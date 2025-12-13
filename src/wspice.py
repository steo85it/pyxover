"""SPICE convenience wrappers that gracefully handle missing kernels."""

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


def get_total_spk_coverage(targ: str | int) -> spice.support_types.Cell_Double:
    """Aggregate coverage windows for all loaded SPK kernels.

    Parameters
    ----------
    targ : str | int
        Target name or NAIF ID to query.

    Returns
    -------
    spice.support_types.Cell_Double
        SPICE window describing the merged coverage intervals for the target.
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


def et_in_total_spk(targ: str | int, et: np.ndarray | float) -> np.ndarray:
    """Check whether ephemeris times fall within any loaded SPK coverage.

    Parameters
    ----------
    targ : str | int
        Target name or NAIF ID to query.
    et : numpy.ndarray | float
        Ephemeris time values in seconds past J2000.

    Returns
    -------
    numpy.ndarray
        Boolean array indicating coverage for each provided ``et`` sample.
    """

    et = np.atleast_1d(et)
    cover_total = get_total_spk_coverage(targ)

    inside = np.zeros(len(et), dtype=bool)
    for i in range(spice.wncard(cover_total)):
        left, right = spice.wnfetd(cover_total, i)
        inside |= (et >= left) & (et <= right)

    return inside


def spice_spkezr(
   targ: str,
   et: np.ndarray | float,
   ref: str,
   obs: str,
   fname: str = "",
) -> tuple[np.ndarray, np.ndarray]:
   """Wrapper around :func:`spiceypy.spkezr` that fills missing epochs with NaNs.

   Parameters
   ----------
   targ : str
       Target identifier passed to SPICE.
   et : numpy.ndarray | float
       Ephemeris times in seconds past J2000.
   ref : str
       Reference frame name.
   obs : str
       Observer identifier passed to SPICE.
   fname : str, optional
       Context string used in warning messages, by default ``""``.

   Returns
   -------
   tuple[numpy.ndarray, numpy.ndarray]
       Tuple of spacecraft position and velocity arrays in meters and meters per second.
   """

   # Ensure ET is numpy array for consistent handling
   et = np.atleast_1d(et)

   try:
      scpv, _ = spice.spkezr(targ, et, ref, 'NONE', obs)
   except Exception:
      warnings.warn(
         f"Vectorized spkezr failed for {targ} w.r.t {obs} in {fname}. "
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


def spice_spkpos(
   targ: str,
   et: np.ndarray | float,
   ref: str,
   obs: str,
   fname: str = "",
) -> np.ndarray:
   """Wrapper around :func:`spiceypy.spkpos` that fills missing epochs with NaNs.

   Parameters
   ----------
   targ : str
       Target identifier passed to SPICE.
   et : numpy.ndarray | float
       Ephemeris times in seconds past J2000.
   ref : str
       Reference frame name.
   obs : str
       Observer identifier passed to SPICE.
   fname : str, optional
       Context string used in warning messages, by default ``""``.

   Returns
   -------
   numpy.ndarray
       Spacecraft positions in meters with shape ``(N, 3)``.
   """

   # Ensure ET is numpy array for consistent handling
   et = np.atleast_1d(et)

   try:
      scpv, _ = spice.spkpos(targ, et, ref, 'NONE', obs)
   except Exception:
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
