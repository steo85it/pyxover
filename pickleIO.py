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
        tmp = [glob.glob(filnam[:-8] + f'{(int(filnam[-8:-6]) + i):02}' + "??.pkl") for i in [-1, 0, 1]]
        print("alternate file for same orbit: ",tmp)
        tmp = [x for x in tmp if x != []][0][0]
        pklfile = open(tmp, 'rb')

    objOut = pickle.load(pklfile)
    pklfile.close()

    return objOut
