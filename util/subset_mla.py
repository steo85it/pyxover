#!/usr/bin/env python3
# ----------------------------------
# Select uniform orbit subset for simulation purposes (spk chby and simdata TAB)
# ----------------------------------
# Author: Stefano Bertone
# Created: 24-Jul-2019
#

import os
import shutil

import numpy as np
from glob import glob
import pickle

def read_all_files(path):
   file_names = glob(path)

   return file_names

if __name__ == '__main__':

    local = 0
    exp = "tp6"
    use_existing_sel = True
    ntracks = 500

    if local:
       spk_path = '/home/sberton2/Works/NASA/Mercury_tides/aux/spaux_*.pkl'
       rem_path = '/home/sberton2/Works/NASA/Mercury_tides/aux/subset_list.pkl'
    else:
       spk_path = '/att/nobackup/sberton2/MLA/aux/spaux_*.pkl'
       rem_path = '/att/nobackup/sberton2/MLA/aux/subset_list.pkl'

    all_spk = read_all_files()

    # make selection and remove other spk
    if not use_existing_sel:
        selected = np.sort(np.random.choice(all_spk, ntracks, replace=False))
    else:
        selected = all_spk
        with open (rem_path, 'rb') as fp:
            selected = pickle.load(fp)

    print(len(selected),len(all_spk))
    remove_these = list(set(all_spk)^set(selected))

    with open(rem_path, 'wb') as fp:
       pickle.dump(remove_these, fp)

    for rmf in remove_these:
        # Probably no need to remove these, sufficient to rename
        if local:
            os.remove(rmf)
        else:
            shutil.move(rmf,'_'+rmf[:-3]+'.bak')

    # select same orbits on simulated data
    orbs = [f.split('_')[-1].split('.')[0] for f in selected]

    if local:
       obsfil = glob("/home/sberton2/Works/NASA/Mercury_tides/data/SIM_??/"+exp+"/0res_1amp/*.TAB")
    else:
       obsfil = glob("/att/nobackup/sberton2/MLA/data/SIM_??/"+exp+"/*res_*amp/*.TAB")

    selected = [s for s in obsfil for orb in orbs if orb in s]
    print(selected)
    print(len(all_spk))
    print(len(selected))
    remove_these = list(set(obsfil)^set(selected))
    for rmf in remove_these:
        if local:
            os.remove(rmf)
        else:
            shutil.move(rmf,'_'+rmf[:-3]+'.BAK')

    exit()
