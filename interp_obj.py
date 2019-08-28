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
#from scipy.spatial.transform import Slerp
#from scipy.spatial.transform import Rotation as R
from rotation import Slerp
from rotation import Rotation as R

class interp_obj:

    def __init__(self, name):
        self.name = name
        self.timeb = None
        self.cheb = None
        self.chebdeg = None
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

    def interpCby(self, x_in, t_in, deg=20):
        # print("t_in",t_in)
        # print("x_in",x_in)

        self.timeb = [t_in.min(), t_in.max()]
        self.chebdeg = deg

        x_in = np.transpose(x_in)
        t_fit = t_in - self.timeb[0]

        cheb_coef = cheby.chebfit(x=t_fit, y=x_in, deg=deg, full=False)
        # print(cheby.chebfit(x=t_in,y=x_in,deg=deg,full=True))
        # print("cheb_coef",cheb_coef)
        # cheb_coef = cheb_coef[0]
        self.cheb = cheb_coef

    def evalCby(self, t_in):

        t_eval = t_in - self.timeb[0]

        return cheby.chebval(t_eval, self.cheb)

    def interpCmat(self, cmat, t_in):

        key_rots = R.from_dcm(np.stack(cmat, axis=0))

        self.slerp = Slerp(t_in, key_rots)

    def evalCmat(self, t_eval):

        return self.slerp(t_eval).as_dcm()

    def savetopkl(self, filename):
        with open(filename, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump((self.tck, self.u), f, pickle.HIGHEST_PROTOCOL)
