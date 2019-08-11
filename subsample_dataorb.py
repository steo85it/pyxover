#!/usr/bin/env python3
# ----------------------------------
# Select uniform orbit subset for simulation purposes (spk chby and simdata TAB)
# ----------------------------------
# Author: Stefano Bertone
# Created: 24-Jul-2019
#

import os

import numpy as np
from glob import glob

def read_all_files():
   file_names = glob('/home/sberton2/Works/NASA/Mercury_tides/aux/spaux_*.pkl')

   return file_names

if __name__ == '__main__':

    exp = "tp2"
    sub_exp = "*"
    use_existing_sel = True
    apply_to_data = True

    all_spk = read_all_files()

    # make selection and remove other spk
    if not use_existing_sel:
        selected = np.sort(np.random.choice(all_spk, 100, replace=False))
        print(len(selected),len(all_spk))
        remove_these = list(set(all_spk)^set(selected))

        for rmf in remove_these:
            os.remove(rmf)
    else:
        selected = all_spk

    # select same orbits on simulated data
    orbs = [f.split('_')[-1].split('.')[0] for f in selected]

    if apply_to_data:
        obsfil = glob("/home/sberton2/Works/NASA/Mercury_tides/data/MLA_??/*.TAB")
    else:
        obsfil = glob("/home/sberton2/Works/NASA/Mercury_tides/data/SIM_??/"+exp+"/"+sub_exp+"/*.TAB")

    selected = [s for s in obsfil for orb in orbs if orb[:-2] in s]
    print("selected: ", selected)
    print("# old files: ", len(all_spk))
    print("# selected: ", len(selected))

    remove_these = list(set(obsfil)^set(selected))

    for rmf in remove_these:
        # print("rm ",rmf)
        os.remove(rmf)

    exit()