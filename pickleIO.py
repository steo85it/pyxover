#!/usr/bin/env python3
# ----------------------------------
# pickleIO.py
#
# Description: interpolated object class for spice 
#              orbits and attitude
# 
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 08-Feb-2019

import pickle


def save(objIn, filnam):
    pklfile = open(filnam, "wb")
    pickle.dump(objIn, pklfile)
    pklfile.close()


# load groundtrack from file
def load(filnam):
    pklfile = open(filnam, 'rb')
    objOut = pickle.load(pklfile)
    pklfile.close()

    return objOut
