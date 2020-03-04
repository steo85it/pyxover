import glob
import itertools as itert
import shutil
import sys
import time

import numpy as np
import pandas as pd

from pickleIO import save, load
from prOpt import tmpdir, local
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

def compare_subsets():

    files = glob.glob(tmpdir+'bestROItracks100*.pkl')
    tstnam = [f.split('/')[-1].split('_')[-1].split('.')[0] for f in files]

    tracks = []
    for idx,f in enumerate(files):
        subs = load(f)
        print(tstnam[idx],subs[1])
        tracks.append(set(subs[0].ravel()))

    intersect_percent = np.array([(tstnam[a],tstnam[b],round(len(tracks[a]&tracks[b])/len(tracks[0])*100.,1))
                                  for a in range(len(tracks)) for b in np.arange(a+1,len(tracks))])

    return intersect_percent

def apply_selection(tracklist,exp='tp2',kind='3res_30amp'):

    subs = load(tracklist)
    tracks = set(subs[0].ravel())
    # print(tracks)

    if local:
        obsfil = glob.glob("/home/sberton2/Works/NASA/Mercury_tides/data/SIM_??/" + exp + "/" + kind + "/*.TAB")
    else:
        obsfil = glob.glob("/att/nobackup/sberton2/MLA/data/SIM_??/" + exp + "/" + kind + "/*.TAB")

    # need tr[:-3] because of min:sec differences between sim and real data
    selected = [s for s in obsfil for tr in tracks if tr[:-3] in s]

    print('obs selected: ', len(selected))

    remove_these = list(set(obsfil) ^ set(selected))
    print("removing:", len(remove_these), "out of", len(obsfil))
    for rmf in remove_these:
        if local:
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
    xov_ = xov_.load(tmpdir+"dKX_clean.pkl")
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
            save([np.array(s_max),nxov_old],tmpdir+'bestROItracks'+str(sub_len)+'_'+str(nxov_old)+'-'+str(i)+'.pkl')

    # print(load(tmpdir+'bestROItracks.pkl'))
    # print(nxov_old)
    # print(np.array(s_max))

    end = time.time()

    print("Got it in ", str(end-start), " seconds!")
    # exit()

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('# Use as python3 max_roi_tracks.py rand_seed')

    proc = int(sys.argv[1])
    run(seed=proc,sub_len=500)

    intersect_percent = compare_subsets()
    print(intersect_percent)

    apply_selection(tracklist=tmpdir + 'bestROItracks500_563-464424.pkl',
                    exp ='tp2', kind ='3res_30amp')