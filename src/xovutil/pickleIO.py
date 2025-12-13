"""Lightweight helpers for serializing and loading pickled Python objects."""

# Description: interpolated object class for spice
#              orbits and attitude
#
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 08-Feb-2019
import glob
import pickle
from typing import Any


def save(objIn: Any, filnam: str) -> None:
    """Persist a Python object to disk using pickle."""

    pklfile = open(filnam, "wb")
    pickle.dump(objIn, pklfile, protocol=-1)
    pklfile.close()


# load groundtrack from file
def load(filnam: str) -> Any:
    """Load a pickled object, falling back to nearby timestamps if needed."""

    try:
        pklfile = open(filnam, 'rb')
    except Exception:
        # TODO this just applies to pickled spice orbits... should not be a general fallback
        # accounts for different minutes (time-scale?) btw real and simulated data
        HH = int(filnam[-8:-6])
        MM = int(filnam[-6:-4])
        if HH == 23 and MM > 52 and [glob.glob(filnam[:-8] + f'{(int(filnam[-8:-6])):02}' + "??.pkl")]==[[]]:
            tmp = [glob.glob(filnam[:-10] + f'{(int(filnam[-10:-8]) + i):02}' + "00" + "??.pkl") for i in [1]]
        else:
            tmp = [glob.glob(filnam[:-8] + f'{(int(filnam[-8:-6]) + i):02}' + "??.pkl")
               for i in [-1, 0, 1]]
        tmp = [x for x in tmp if x != []][0][0]
        pklfile = open(tmp, 'rb')

    objOut = pickle.load(pklfile)
    pklfile.close()

    return objOut
