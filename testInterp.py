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

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice

from interp_obj import interp_obj

pltcurve = 0
pltdiff = 1


def testInterp(ladata_df, vecopts):
    MGRx = interp_obj('MGRx')

    # Define call times for the SPICE
    t_spc = np.array([x for x in np.arange(ladata_df['ET_TX'].values.min(), ladata_df['ET_TX'].values.max(), 1)])
    t1_spc = np.array(
        [x for x in np.arange(ladata_df['ET_TX'].values.min() + 1., ladata_df['ET_TX'].values.max() + 1., 1)])

    print("Start spkezr MGR")
    # trajectory
    xv_spc = np.array([spice.spkpos(vecopts['SCNAME'],
                                    t,
                                    vecopts['INERTIALFRAME'],
                                    'NONE',
                                    vecopts['INERTIALCENTER']) for t in t_spc])[:, 0]

    # xv_spc = np.reshape(np.concatenate(xv_spc),(-1,6))
    xv_spc = np.vstack(xv_spc) * 1e3
    print(xv_spc, xv_spc.shape)

    MGRx.interpSpl(np.transpose(xv_spc), t_spc)
    print(xv_spc.shape, t_spc.shape)
    # xv_spc=np.linspace(-1,1,len(t_spc))
    MGRx.interpCby(np.transpose(xv_spc), t_spc, 20)

    # test
    xv1_spc = np.array([spice.spkpos(vecopts['SCNAME'],
                                     t,
                                     vecopts['INERTIALFRAME'],
                                     'NONE',
                                     vecopts['INERTIALCENTER']) for t in t1_spc])[:, 0]
    xv1_spc = np.vstack(xv1_spc) * 1e3

    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis

    if (pltcurve):
        ax.scatter(t_spc, np.linalg.norm(xv_spc, axis=1), label="Real data", s=1)
        ax.plot(t1_spc, np.linalg.norm(np.transpose(MGRx.evalSpl(t1_spc)), axis=1), label="Spline", color='C1')
        ax.plot(t1_spc, np.linalg.norm(np.transpose(MGRx.evalCby(t1_spc)), axis=1), label="Cheby", color='C2')
    elif (pltdiff):
        ax.plot(t1_spc, np.linalg.norm(np.transpose(MGRx.evalSpl(t1_spc)) - xv1_spc, axis=1), label="Spline",
                color='C1')
        ax.plot(t1_spc, np.linalg.norm(np.transpose(MGRx.evalCby(t1_spc)) - xv1_spc, axis=1), label="Cheby", color='C2')
        ax.legend()

    fig.savefig('testInterp.png')  # save the figure to file
    plt.close(fig)  # close the figure

    exit()
