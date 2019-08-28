import itertools as itert
import sys
import time

import numpy as np
import pandas as pd

from pickleIO import save
from prOpt import tmpdir
from xov_setup import xov


def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def intersect1d_searchsorted(A,B,assume_unique=False):
    if assume_unique==0:
        B_ar = np.unique(B)
    else:
        B_ar = B
    idx = np.searchsorted(B_ar,A)
    idx[idx==len(B_ar)] = 0
    return A[B_ar[idx] == A]

def run(proc):
    vecopts = {}
    xov_ = xov(vecopts)
    xov_ = xov_.load(tmpdir+"dKX_clean.pkl")
    print("Loaded...")

    # dR absolute value taken
    xov_.xovers['dR_orig'] = xov_.xovers.dR
    xov_.xovers['dR'] = xov_.xovers.dR.abs()

    start = time.time()

    # get list of all orbA-orbB giving xov at low lats
    lowlat_xov = xov_.xovers.loc[xov_.xovers.LAT < 45]
    xov_occ = list(zip(lowlat_xov.orbA.values,lowlat_xov.orbB.values))

    xov_occ_str = [a+'-'+b for a, b in xov_occ]

    # build up random combinations of N orbA-orbB
    orbs = pd.DataFrame([xov_.xovers['orbA'].value_counts(), xov_.xovers['orbB'].value_counts()]).T.fillna(0).sum(axis=1)
    print(orbs.index)
    orbs = orbs.index.values

    nxov_old = 0
    np.random.seed(proc)

    for i in range(100000):
        orb_sel = np.random.choice(orbs, 500)

        s = [(a,b) for a,b in list(
            itert.product(orb_sel, orb_sel))
               if a != b]

        sampl_str = [a+'-'+b for a, b in s]
        # print(np.array(sampl_str))
        # intersect = intersect1d_searchsorted(sampl_str,xov_occ_str,assume_unique=True)

        nxov = len(intersection(sampl_str, xov_occ_str))
        if nxov > nxov_old:
            s_max = s
            nxov_old = nxov
            print("New max = ", nxov_old, " @ sample ",i)
            save([np.array(s_max),nxov_old],tmpdir+'bestROItracks.pkl')

    # print(load(tmpdir+'bestROItracks.pkl'))

    print(nxov_old)
    print(np.array(s_max))

    end = time.time()

    print("Got it in ", str(end-start), " seconds!")
    exit()

if __name__ == '__main__':

    proc = int(sys.argv[1])
    run(proc)