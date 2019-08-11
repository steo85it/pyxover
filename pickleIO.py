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
import glob
import pickle


def save(objIn, filnam):
    pklfile = open(filnam, "wb")
    pickle.dump(objIn, pklfile)
    pklfile.close()


# load groundtrack from file
def load(filnam):

    try:
        pklfile = open(filnam, 'rb')
    except:
        # accounts for different minutes (time-scale?) btw real and simulated data
        pklfile = open(glob.glob(filnam[:-6]+"??.pkl")[0], 'rb')

    objOut = pickle.load(pklfile)
    pklfile.close()

    return objOut
