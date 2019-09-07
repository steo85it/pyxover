#!/usr/bin/env python3
# ----------------------------------
# Compare solutions
# ----------------------------------
# Author: Stefano Bertone
# Created: 30-Aug-2019
#

import glob

import numpy as np

from pickleIO import load
from prOpt import outdir

def compare_sols():
    listA = glob.glob(outdir+"sim/KX1r_2/0res_1amp/gtrack_13/gtrack_*pkl")
    listB = glob.glob(outdir+"sim/KX1r_3/0res_1amp/gtrack_13/gtrack_*pkl")

    diffs = []
    for lA, lB in zip(listA, listB):
#        print("Processing ", lA, lB)

        trackA = load(lA)
#        print(trackA.sol_prev_iter)
        if not trackA.sol_prev_iter is None and len(trackA.sol_prev_iter['orb'].values) > 0:
#            print(trackA.sol_prev_iter['orb'].columns[:4].values) #if sol setup by altimetry only
            solA = trackA.sol_prev_iter['orb'].values[0].astype(float)[:4]
            trackB = load(lB)
            if not trackB.sol_prev_iter is None and len(trackB.sol_prev_iter['orb'].values) > 0:
                solB = trackB.sol_prev_iter['orb'].values[0].astype(float)[:4]

#                print(solA)
#                print(solB)
#                print(solB - solA)
                
                diffs.append(solB - solA)

    return np.array(diffs)

if __name__ == '__main__':

    diffs = compare_sols()
    print("ACR diff")
    print(np.array(diffs))
    print("RMS:", np.sqrt(np.mean(np.square(diffs),axis=0)))
    print("mean, std:", np.mean(diffs,axis=0), np.std(diffs,axis=0))
