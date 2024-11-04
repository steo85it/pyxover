#!/usr/bin/env python3
# ----------------------------------
# Select uniform orbit subset for simulation purposes and apply to dataset
# ----------------------------------
# Author: Stefano Bertone
# Created: 24-Jul-2019
#
# if main PyXover in other folder, first launch "export PYTHONPATH="$PWD:$PYTHONPATH" from PyXover dir

import glob
import itertools as itert
import shutil
import sys
import time

import numpy as np
import pandas as pd

from xovutil.pickleIO import save, load
# from examples.MLA.options import XovOpt.get("tmpdir"), XovOpt.get("local")
from pyxover.xov_setup import xov
from config import XovOpt



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

def compare_subsets():

    files = glob.glob(XovOpt.get("tmpdir") + 'bestROItracks100*.pkl')
    tstnam = [f.split('/')[-1].split('_')[-1].split('.')[0] for f in files]

    tracks = []
    for idx,f in enumerate(files):
        subs = load(f)
        print(tstnam[idx],subs[1])
        tracks.append(set(subs[0].ravel()))

    intersect_percent = np.array([(tstnam[a],tstnam[b],round(len(tracks[a]&tracks[b])/len(tracks[0])*100.,1))
                                  for a in range(len(tracks)) for b in np.arange(a+1,len(tracks))])

    return intersect_percent

# note: revert with **for f in /explore/nobackup/people/sberton2/MLA/data/SIM_??/tp6/*res_*amp/*.BAK; do mv "$f" "${f%.BAK}.TAB";done**
#       check with **for f in /explore/nobackup/people/sberton2/MLA/data/SIM_??/tp6/*res_*amp/*.BAK; do echo $f;done**
# to generate soft links from main dataset before subsetting it, e.g.,
# for f in {08..15}; do cp -rs /explore/nobackup/people/sberton2/MLA/data/SIM_$f/KX1 /explore/nobackup/people/sberton2/MLA/data/SIM_$f/KX1r4 ;done

def apply_selection(tracklist,exp='tp2',kind='3res_30amp'):

    subs = load(tracklist)
    print(subs)
    tracks = set(subs[0].ravel())
    print(tracks)
    exit()

    if XovOpt.get("local"):
        obsfil = glob.glob("/home/sberton2/Works/NASA/Mercury_tides/data/SIM_??/" + exp + "/" + kind + "/*.TAB")
    else:
        obsfil = glob.glob("/explore/nobackup/people/sberton2/MLA/data/SIM_??/" + exp + "/" + kind + "/*.TAB")

    # need tr[:-3] because of min:sec differences between sim and real data
    selected = [s for s in obsfil for tr in tracks if tr[:-3] in s]

    print('tracks selected: ', len(selected))

    remove_these = list(set(obsfil) ^ set(selected))
    print("removing:", len(remove_these), "out of", len(obsfil))
    for rmf in remove_these:
        if XovOpt.get("local"):
            # print(rmf)
            shutil.move(rmf, rmf[:-3] + 'BAK')
            # os.remove(rmf)
            # pass
        else:
            shutil.move(rmf, rmf[:-3] + 'BAK')
            # pass
    print("Done")

def run(seed,sub_len=100):
    vecopts = {}
    xov_ = xov(vecopts)
    xov_ = xov_.load(XovOpt.get("tmpdir") + "dKX_clean.pkl")
    print("Loaded...")

    # dR absolute value taken
    xov_.xovers['dR_orig'] = xov_.xovers.dR
    xov_.xovers['dR'] = xov_.xovers.dR.abs()

    start = time.time()

    # get list of all orbA-orbB giving xov at low lats
    lowlat_xov = xov_.xovers.loc[xov_.xovers.LAT < 50]
    xov_occ = list(zip(lowlat_xov.orbA.values,lowlat_xov.orbB.values))

    xov_occ_str = [a+'-'+b for a, b in xov_occ]

    # build up random combinations of N orbA-orbB
    orbs = pd.DataFrame([xov_.xovers['orbA'].value_counts(), xov_.xovers['orbB'].value_counts()]).T.fillna(0).sum(axis=1)
    print(orbs.index)
    orbs = orbs.index.values

    nxov_old = 0
    np.random.seed(seed)

    for i in range(1000000):
        orb_sel = np.random.choice(orbs, sub_len)

        s = [(a,b) for a,b in list(
            itert.product(orb_sel, orb_sel))
               if a != b]

        sampl_str = [a+'-'+b for a, b in s]
        # print(np.array(sampl_str))
        # intersect = intersect1d_searchsorted(sampl_str,xov_occ_str,assume_unique=True)

        nxov = len(intersection(sampl_str, xov_occ_str))
        if nxov >= nxov_old: # good number based on full database # nxov_old:
            s_max = s
            nxov_old = nxov
            print("New max = ", nxov_old, " @ sample ",i)
            save([np.array(s_max),nxov_old], XovOpt.get("tmpdir") + 'bestROItracks' + str(sub_len) + '_' + str(nxov_old) + '-' + str(i) + '.pkl')

    # print(load(tmpdir+'bestROItracks.pkl'))
    # print(nxov_old)
    # print(np.array(s_max))

    end = time.time()

    print("Got it in ", str(end-start), " seconds!")
    # exit()

def select_from_stats(sol ='KX1', subexp ='0res_1amp', outfil='bestROItracks.pkl'):

    from accumxov.Amat import Amat
    # from examples.MLA.options import XovOpt.get("outdir"), XovOpt.get("vecopts")
    # import numpy as np
    from config import XovOpt

    subfolder = ''
    # sol = 'KX1_0'  # r4_1'
    # subexp = '0res_1amp'

    tmp = Amat(XovOpt.get("vecopts"))
    tmp = tmp.load(XovOpt.get("outdir") + 'sim/' + subfolder + sol + '/' + subexp + '/Abmat_sim_' + sol.split('_')[0] + '_' + str(
        int(sol.split('_')[-1]) + 1) + '_' + subexp + '.pkl')

    xov_df = tmp.xov.xovers
    # remove very large dR (>1km)
    xov_df = xov_df.loc[xov_df['dR'].abs() < 1.e3]
    print(xov_df.columns)
    # print(tmp.weights)
    # exit()
    hilat_xov = xov_df.loc[xov_df.LAT > 60]
    print(hilat_xov[['dR', 'weights', 'huber']].abs().max())
    print(hilat_xov[['dR', 'weights', 'huber']].abs().min())
    print(hilat_xov[['dR', 'weights', 'huber']].abs().mean())
    print(hilat_xov[['dR', 'weights', 'huber']].abs().median())

    to_keep = 1. - 8.e5 / len(hilat_xov)
    to_keep_hilat = hilat_xov.loc[hilat_xov['weights'] > hilat_xov['weights'].quantile(to_keep)].xOvID.values

    lolat_xov = xov_df.loc[xov_df.LAT < 60]
    to_keep_lolat = lolat_xov.loc[lolat_xov['weights'] > lolat_xov['weights'].quantile(0.1)].xOvID.values

    # select very good xovers at LAT>60N OR decent xovers at low latitudes
    selected = xov_df.loc[(xov_df.xOvID.isin(to_keep_hilat)) | (xov_df.xOvID.isin(to_keep_lolat))]
    print(len(selected))
    print(selected[['dR', 'weights', 'huber', 'dist_min_mean']].abs().max())
    print(selected[['dR', 'weights', 'huber', 'dist_min_mean']].abs().min())
    print(selected[['dR', 'weights', 'huber', 'dist_min_mean']].abs().median())
    print(selected[['dR', 'weights', 'huber', 'dist_min_mean']].abs().mean())

    # set of orbits giving
    orbs = list(set(np.hstack([selected.orbA.values, selected.orbB.values])))
    print(orbs)
    print(len(orbs), "orbits giving the 'best'", len(selected), "xovers out of", len(xov_df))

    save([np.array(orbs)], XovOpt.get("tmpdir") + outfil)
    return orbs

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('# Use as python3 max_roi_tracks.py rand_seed')

    tracks_fil = 'bestROItracks.pkl'

    proc = int(sys.argv[1])
    print(proc)
    #run(seed=proc,sub_len=500)

    #intersect_percent = compare_subsets()
    #print(intersect_percent)

    tracks = select_from_stats(sol ='KX1_0', subexp ='0res_1amp', outfil=tracks_fil)

    apply_selection(tracklist=XovOpt.get("tmpdir") + tracks_fil,  #'bestROItracks500_563-464424.pkl',
                    exp ='KX1r5', kind ='0res_1amp')
