#!/usr/bin/env python3
# ----------------------------------
# Select uniform orbit subset for simulation purposes (spk chby and simdata TAB)
# ----------------------------------
# Author: Stefano Bertone
# Created: 24-Jul-2019
#
# if main PyXover in other folder, first launch "export PYTHONPATH="$PWD:$PYTHONPATH" from PyXover dir
# note: revert with **for f in data/SIM_??/tp6/*res_*amp/*.BAK; do mv "$f" "${i%..BAK}.TAB";done**
#       check with **for f in data/SIM_??/tp6/*res_*amp/*.BAK; do echo $f;done**

import os
import shutil

import numpy as np
import glob
import pickle

from prOpt import auxdir


def read_all_files(path):
   file_names = glob.glob(path)

   return file_names

if __name__ == '__main__':

    local = 0
    exp = "tp8"
    use_existing_sel =True
    ntracks = 500

    if local:
       spk_path = auxdir+'spaux_*.pkl'
       # rem_path = '/home/sberton2/Works/NASA/Mercury_tides/aux/subset_list.pkl'
       rem_path = auxdir+'bestROItracks100.pkl'

    else:
       spk_path = auxdir+'spaux_*.pkl'
       rem_path = '/att/nobackup/sberton2/MLA/aux/subset_list.pkl'
       #rem_path = '/att/nobackup/sberton2/MLA/tmp/bestRoI_tracks.pkl'

    all_spk = read_all_files(spk_path)

    # make selection and remove other spk
    if not use_existing_sel:
        selected = np.sort(np.random.choice(all_spk, ntracks, replace=False))
        with open(rem_path, 'wb') as fp:
             pickle.dump(selected, fp)
    else:
        selected = all_spk
        with open (rem_path, 'rb') as fp:
            sel = pickle.load(fp)
            sel = set(np.array(sel[0]).ravel())

            selected = []
            for f in sel:
                _ = glob.glob(auxdir + 'spaux_' + f[:-2] + '*.pkl')
                print(_)
                if len(_)>0:
                    selected.append(_[0])

    print('selected spk: ',len(selected),' out of ',len(all_spk))
    remove_these = list(set(all_spk)^set(selected))
#    print(remove_these)

    for rmf in remove_these:
        # Probably no need to remove these, sufficient to rename
        if local:
            # os.remove(rmf)
            shutil.move(rmf,rmf[:-3]+'bak')
        else:
#            shutil.move(rmf,rmf[:-3]+'bak')
            pass

    # use if just want to select same orbits as existing spaux
    selected = glob.glob(auxdir + 'spaux_' + '*.pkl')
    # select same orbits on simulated data
    orbs = [f.split('_')[-1].split('.')[0] for f in selected]

    if local:
       obsfil = glob.glob("/home/sberton2/Works/NASA/Mercury_tides/data/SIM_??/"+exp+"/3res_20amp/*.TAB")
    else:
       obsfil = glob.glob("/att/nobackup/sberton2/MLA/data/SIM_??/"+exp+"/3res_20amp/*.TAB")
       # use if want to rename gtracks instead of data
       #obsfil = glob("/att/nobackup/sberton2/MLA/out/sim/"+exp+"/*res_*amp/gtrack_??/*.pkl")
    
    print(obsfil)
    print(orbs)
    selected = [s for s in obsfil for orb in orbs if orb[:-2] in s]

    print('total spk:',len(all_spk))
    print('obs selected: ',len(selected))

    remove_these = list(set(obsfil)^set(selected))
    for rmf in remove_these:
        if local:
            shutil.move(rmf, rmf[:-3] + 'BAK')
            # os.remove(rmf)
        else:
            shutil.move(rmf,rmf[:-3]+'BAK')
#            pass
    exit()
