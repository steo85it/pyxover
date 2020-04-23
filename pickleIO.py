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
    pickle.dump(objIn, pklfile, protocol=-1)
    pklfile.close()


# load groundtrack from file
def load(filnam):
    try:
        pklfile = open(filnam, 'rb')
    except:
        # accounts for different minutes (time-scale?) btw real and simulated data
        HH = int(filnam[-8:-6])
        MM = int(filnam[-6:-4])
#        print(HH,MM)
        if HH == 23 and MM > 52 and [glob.glob(filnam[:-8] + f'{(int(filnam[-8:-6])):02}' + "??.pkl")]==[[]]:
#            print("case 0")
            tmp = [glob.glob(filnam[:-10] + f'{(int(filnam[-10:-8]) + i):02}' + "00" + "??.pkl") for i in [1]]
        #elif HH == 0 and MM < 8:
        #    tmp = [glob.glob(filnam[:-10] + f'{(int(filnam[-10:-8]) + i):02}' + "23" + "??.pkl") for i in [-1, 0]]
        else:
#            print("case 1")
            tmp = [glob.glob(filnam[:-8] + f'{(int(filnam[-8:-6]) + i):02}' + "??.pkl")
               for i in [-1, 0, 1]]
#        print(filnam[:-8] + f'{(int(filnam[-8:-6])):02}')
#        print([glob.glob(filnam[:-8] + f'{(int(filnam[-8:-6])):02}' + "??.pkl")])
#        print("alternate file for same orbit: ",tmp)
        tmp = [x for x in tmp if x != []][0][0]
        pklfile = open(tmp, 'rb')

    objOut = pickle.load(pklfile)
    pklfile.close()

    return objOut
