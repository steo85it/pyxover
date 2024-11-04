#!/usr/bin/env python3
# ----------------------------------
# interp_obj.py
#
# Description: Read probe and planet orbit and interpolate polynomial
# 
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 08-Feb-2019
#
##############################################
import glob

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice

from src.pygeoloc.ground_track import gtrack
from src.xovutil.interp_obj import interp_obj
# from examples.MLA.options import XovOpt.get("outdir"), XovOpt.get("vecopts"), XovOpt.get("auxdir")
from config import XovOpt

pltcurve = 0
pltdiff = 1


def simple_test():
    np.random.seed(0)

    x = np.linspace(-1, 1, 200)
    y = np.cos(x) + 0.3 * np.random.rand(200)

    deg = 100
    p = np.polynomial.Chebyshev.fit(x, y, deg)

    t = np.linspace(-1, 1, 2*deg)
    plt.plot(x, y, 'r.')
    plt.plot(t, p(t), 'k-', lw=3)
    plt.savefig('tmp/simpleInterp.png')  # save the figure to file
    plt.close()  # close the figure

def testInterp(ladata_df, vecopts):
    MGRx = interp_obj('MGRx')

    # Define call times for the SPICE
    t_spc = np.array([x for x in np.arange(ladata_df['ET_TX'].values.min(), ladata_df['ET_TX'].values.max(), 1)])
    # t1_spc = np.array(
    #     [x for x in np.arange(ladata_df['ET_TX'].values.min() + 1., ladata_df['ET_TX'].values.max() + 1., 1)])
    t_spc = t_spc[:]
    # t1_spc = t1_spc[:]

    # print("Start spkezr MGR")
    # trajectory
    xv_spc = np.array([spice.spkpos(vecopts['SCNAME'],
                                    t,
                                    vecopts['INERTIALFRAME'],
                                    'NONE',
                                    vecopts['INERTIALCENTER']) for t in t_spc])[:, 0]

    # xv_spc = np.reshape(np.concatenate(xv_spc),(-1,6))
    xv_spc = np.vstack(xv_spc) * 1e3
    # print(xv_spc, xv_spc.shape)

    MGRx.interpSpl(np.transpose(xv_spc), t_spc)
    # print(xv_spc.shape, t_spc.shape)
    # xv_spc=np.linspace(-1,1,len(t_spc))
    MGRx.interpCby(np.transpose(xv_spc), t_spc, 15)

    # test
    # xv1_spc = np.array([spice.spkpos(vecopts['SCNAME'],
    #                                  t,
    #                                  vecopts['INERTIALFRAME'],
    #                                  'NONE',
    #                                  vecopts['INERTIALCENTER']) for t in t1_spc])[:, 0]
    # xv1_spc = np.vstack(xv1_spc) * 1e3

    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis

    if (pltcurve):
        ax.plot(t_spc, np.linalg.norm(np.transpose(MGRx.evalSpl(t_spc)), axis=1) - np.linalg.norm(xv_spc, axis=1), label="Spline", color='C1')
        ax.plot(t_spc, np.linalg.norm(np.transpose(MGRx.evalCby(t_spc)), axis=1) - np.linalg.norm(xv_spc, axis=1), label="Cheby", color='C2')
        ax.scatter(t_spc, np.linalg.norm(xv_spc, axis=1) - np.linalg.norm(xv_spc, axis=1), label="Real data", s=1)
        ax.legend()

    elif (pltdiff):
        ax.plot(t_spc, np.linalg.norm(np.transpose(MGRx.evalSpl(t_spc)) - xv_spc, axis=1), label="Spline",
                color='C1')
        ax.plot(t_spc, np.linalg.norm(np.transpose(MGRx.evalCby(t_spc)) - xv_spc, axis=1), label="Cheby", color='C2')
        ax.legend()

    fig.savefig('tmp/testInterp.png')  # save the figure to file
    plt.close(fig)  # close the figure

    return np.max(np.linalg.norm(np.transpose(MGRx.evalCby(t_spc)) - xv_spc, axis=1))

if __name__ == '__main__':

    # track_id = '1301052351'
    files = glob.glob('/home/sberton2/Works/NASA/Mercury_tides/out/sim/1301_per2_0/0res_1amp/gtrack_13/gtrack_*.pkl')
    trackA = gtrack(XovOpt.get("vecopts"))
    # load kernels
    spice.furnsh(XovOpt.get("auxdir") + 'mymeta')

    for f in files:
        trackA = trackA.load(f)
        # simple_test()
        max = testInterp(trackA.ladata_df, XovOpt.get("vecopts"))
        print(f.split('/')[-1],max)

    exit()
