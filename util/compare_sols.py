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
    listA = glob.glob(outdir+"sim/d00_1/0res_1amp/gtrack_13/gtrack_1301*pkl")
    listB = glob.glob(outdir+"sim/d00_2/0res_1amp/gtrack_13/gtrack_1301*pkl")

    diffs = []
    for lA, lB in zip(listA, listB):
        print("Processing ", lA, lB)

        trackA = load(lA)
        print(trackA.sol_prev_iter['orb'].columns[:4].values)
        solA = trackA.sol_prev_iter['orb'].values[0].astype(float)[:4]
        trackB = load(lB)
        solB = trackB.sol_prev_iter['orb'].values[0].astype(float)[:4]

        print(solA)
        print(solB)
        print(solB - solA)
        diffs.append(solB - solA)

    return np.array(diffs)

if __name__ == '__main__':

    diffs = compare_sols()
    print(diffs)