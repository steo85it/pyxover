#!/usr/bin/env python3
# ----------------------------------
# interp_obj.py
#
# Description: interpolated object class for spice 
#              orbits and attitude
# 
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 08-Feb-2019

import pickle

import numpy as np
import numpy.polynomial.chebyshev as cheby
from scipy import interpolate


class interp_obj:

    def __init__(self, name):
        self.name = name
        # self.tck = None
        # self.u = None

    def interpSpl(self, x, t_in=0, s_in=0):
        if (t_in != 0).any:
            tck, u = interpolate.splprep(x, u=t_in, k=5, s=s_in)
            self.timeb = [t_in.min, t_in.max]
        else:
            tck, u = interpolate.splprep(x, k=5, s=s_in)

        self.tck = tck
        self.u = u

    def evalSpl(self, t_in):

        return interpolate.splev(t_in, self.tck)

    def interpCby(self, x_in, t_in, deg=15):
        # print("t_in",t_in)
        # print("x_in",x_in)
        x_in = np.transpose(x_in)
        cheb_coef = cheby.chebfit(x=t_in, y=x_in, deg=deg)
        # print(cheby.chebfit(x=t_in,y=x_in,deg=deg,full=True))
        # print("cheb_coef",cheb_coef)
        self.timeb = [t_in.min, t_in.max]
        self.cheb = cheb_coef
        self.chebdeg = deg

    def evalCby(self, t_in):

        return cheby.chebval(t_in, self.cheb)

    def savetopkl(self, filename):
        with open(filename, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump((self.tck, self.u), f, pickle.HIGHEST_PROTOCOL)
