#!/usr/bin/env python3
# ----------------------------------
# PyXover
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-Oct-2018
#
import re
import warnings

import pandas as pd

from accumxov.Amat import Amat
from accumxov import AccumXov as xovacc, accum_utils

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import glob

# from itertools import izip, count
import multiprocessing as mp
# from geopy.distance import vincenty

import spiceypy as spice
import numpy as np

# from collections import defaultdict
# import mpl_toolkits.basemap as basemap

import time
from tqdm import tqdm

# mylib
# from mapcount import mapcount
from pygeoloc.ground_track import gtrack
# from examples.MLA.options import XovOpt.get("parallel"), XovOpt.get("SpInterp"), XovOpt.get("new_gtrack"), XovOpt.get("outdir"), XovOpt.get("auxdir"), XovOpt.get("local"), XovOpt.get("vecopts"), XovOpt.get("debug"), XovOpt.get("OrbRep"), XovOpt.get("rawdir")
from config import XovOpt


# from xovutil import lflatten
########################################
# # test space
#
# tst = [np.array([ 89.72151033, 103.94256763]), np.array([139.94256763])]
# tst = lflatten(tst)
# print(tst)
# exit()
# vecopts = {'SCID':'-236'}
#
# tmp_ser = Amat(vecopts)
# for f in glob.glob('/home/sberton2/Works/NASA/Mercury_tides/out/xov_130101*_13.pkl'):
#     tmp_ser = tmp_ser.load(f)
#
#     print(tmp_ser.xovers)
#
# # tmp_par = Amat(vecopts)
# # tmp_par = tmp_par.load('out/small_par/Amat_small_dataset.pkl')
#
# # print(tmp_par.spA.to_dense().equals(tmp_ser.spA.to_dense()))
#
# # print(tmp_par.spA)
# # print(tmp_ser.spA)
#
# exit()

########################################

def launch_gtrack(args):
    track, infil, outdir_in = args
    track_id = 'gtrack_' + track.name
    # track = gtrack(vecopts)

    if XovOpt.get("new_gtrack"):
        gtrack_out = XovOpt.get("outdir") + outdir_in + '/' + track_id + '.pkl'
        if os.path.isfile(gtrack_out) == False or XovOpt.get("new_gtrack") == 2:

            if not os.path.exists(XovOpt.get("outdir") + outdir_in):
                os.makedirs(XovOpt.get("outdir") + outdir_in, exist_ok=True)

            track.setup(infil)
            #
            if XovOpt.get("debug"):
                print("track#:", track.name)
                print("max diff R",abs(track.ladata_df.loc[:,'R']-(track.ladata_df.loc[:,'altitude']-XovOpt.get("vecopts")['PLANETRADIUS']) * 1.e3).max())
                # print("R (check if radius included + units)",track.ladata_df.loc[:,'R'].max(),track.ladata_df.loc[:,'altitude'].max())
                print("max diff LON", XovOpt.get("vecopts")['PLANETRADIUS'] * 1.e3 * np.sin(np.deg2rad(abs(track.ladata_df.loc[:, 'LON'] - track.ladata_df.loc[:, 'geoc_long']).max())))
                print("max diff LAT", XovOpt.get("vecopts")['PLANETRADIUS'] * 1.e3 * np.sin(np.deg2rad(abs(track.ladata_df.loc[:, 'LAT'] - track.ladata_df.loc[:, 'geoc_lat']).max())))
                print("max elev sim", abs(track.ladata_df.loc[:,'altitude']).max())
            #exit()
            # pd.set_option('display.max_columns', None)

            # print(track.ladata_df)
            # exit()
            if len(track.ladata_df) > 0:
                track.save(gtrack_out)
                if not XovOpt.get("local") or XovOpt.get("debug"):
                    print('Orbit ' + track_id.split('_')[1] + ' processed and written to ' + gtrack_out + '!')
            else:
                print(f"Orbit {track_id.split('_')[1]} contains no valid data. No gtrack created.")    
        # except:
        #    print('failed to process ' + track_id)
        else:
            # track = track.load('out/'+track_id+'.pkl')
            if not XovOpt.get("local") or XovOpt.get("debug"):
                print('Gtrack file ' + gtrack_out + ' already existed!')

def main(args):

    # print(args)

    # read input args
    print('Number of arguments:', len(args), 'arguments.')
    print('Argument List:', str(args))

    epo_in = args[0]
    indir_in = args[1]
    outdir_in = args[2]
    # args[3] ??
    iter_in = args[4]
#    if len(args) > 4: # passing a fct to slurm doesn't pass these updated Opt
    opts = args[5]

    # update options (needed when sending to slurm)
    XovOpt.clone(opts)

    # locate data
    data_pth = f'{XovOpt.get("rawdir")}'
    dataset = indir_in
    data_pth += dataset

    if XovOpt.get("SpInterp") in [0,2]:
        # load kernels
        if not XovOpt.get("local"):
            spice.furnsh([f'{XovOpt.get("auxdir")}furnsh.MESSENGER.def',
                         f'{XovOpt.get("auxdir")}mymeta_pgda'])
        else:
            spice.furnsh(f'{XovOpt.get("auxdir")}mymeta')
        # or, add custom kernels
        # load additional kernels
        # spice.furnsh(['XXX.bsp'])

    # set ncores
    ncores = mp.cpu_count() - 1  # 8

    if XovOpt.get("parallel"):
        print('Process launched on ' + str(ncores) + ' CPUs')

    ##############################################
    # updated w.r.t. SPICE from Mike's scicdr2mat.m
    if XovOpt.get("instrument") == 'BELA':
        XovOpt.get("vecopts")['ALTIM_BORESIGHT'] = [0.,0.,1.]
    elif XovOpt.get("instrument") == 'LOLA':
        XovOpt.get("vecopts")['ALTIM_BORESIGHT'] = np.loadtxt(
            glob.glob(f'{XovOpt.get("auxdir")}{epo_in}/slewcheck_0/' + '_boresights_LOLA_ch12345_*_laser2_fov_bs0.inc')[0])
    else:
        XovOpt.get("vecopts")['ALTIM_BORESIGHT'] = [0.0022105, 0.0029215, 0.9999932892]  # out[2]

    ###########################

    # -------------------------------
    # File reading and ground-tracks computation
    # -------------------------------

    startInit = time.time()

    # read all MLA datafiles (*.TAB in data_pth) corresponding to the given years
    # for orbitA and orbitB.
    allFiles = glob.glob(os.path.join(data_pth, f'{XovOpt.get("instrument")}*RDR*' + epo_in + '*.*'))

    if len(allFiles) == 0:
        print(str.lower(f'{XovOpt.get("instrument")}*RDR*' + epo_in + '*.*'))
        allFiles = glob.glob(os.path.join(data_pth, str.lower(f'{XovOpt.get("instrument")}*RDR*' + epo_in + '*.*')))
        if len(allFiles) == 0:
            print("# No files found in", os.path.join(data_pth, f'{XovOpt.get("instrument")}*RDR*' + epo_in + '*.*') )
        
    endInit = time.time()
    print(
        '----- Runtime Init= ' + str(endInit - startInit) + ' sec -----' + str(
            (endInit - startInit) / 60.) + ' min -----')

    startPrepro = time.time()

    # Prepare list of tracks to geolocalise
    tracknames = ['gtrack_' + fil.split('.')[0][-10:] for fil in allFiles[:]]

    if XovOpt.get("new_gtrack"):

        # Import solution at previous iteration
        if int(iter_in) > 0:
            tmp = Amat(XovOpt.get("vecopts"))
            tmp = tmp.load(('_').join(((XovOpt.get("outdir") + ('/').join(outdir_in.split('/')[:-2]))).split('_')[:-1]) +
                           '_' + str(iter_in - 1) + '/' +
                           outdir_in.split('/')[-2] + '/Abmat_' + ('_').join(outdir_in.split('/')[:-1]) + '.pkl')
            import_prev_sol = hasattr(tmp,'sol4_pars')
            if import_prev_sol:
                orb_sol, glo_sol, sol_dict = accum_utils.analyze_sol(tmp, tmp.xov)
        # epo_in=[]
        tracks = []
        for track_id, infil in tqdm(zip(tracknames, allFiles), total=len(allFiles)):
            track = track_id

            track = gtrack(XovOpt.to_dict())
            # try:
            # Read and fill
            track.prepro(infil)
            # except:
            #    print('Issue in preprocessing for '+track_id)
            # epo_in.extend(track.ladata_df.ET_TX.values)

            if int(iter_in) > 0 and import_prev_sol:
                try:
                    track.pert_cloop_0 = tmp.pert_cloop_0.loc[str(track.name)].to_dict()
                except:
                    if XovOpt.get("debug"):
                        print("No pert_cloop_0 for ", track.name)
                    pass

                regex = re.compile(track.name+"_dR/d.*")
                soltmp = [('sol_'+x.split('_')[1],v) for x, v in tmp.sol_dict['sol'].items() if regex.match(x)]

                if len(soltmp)>0:
                    stdtmp = [('std_' + x.split('_')[1], v) for x, v in tmp.sol_dict['std'].items() if regex.match(x)]
                    soltmp = pd.DataFrame(np.vstack([('orb', str(track.name)), soltmp, stdtmp])).set_index(0).T

                    if XovOpt.get("debug"):
                        print("orbsol prev iter")
                        print(orb_sol.reset_index().orb.values)
                        print(orb_sol.columns)
                        print(str(track.name))
                        print(orb_sol.loc[orb_sol.reset_index().orb.values==str(track.name)])
		       
                    track.sol_prev_iter = {'orb':soltmp,
                                       'glo':glo_sol}
                else:
                    track.sol_prev_iter = {'orb':orb_sol,
                                       'glo':glo_sol}
            # if first iter, check if track has been pre-processed by fit2dem and import corrections
            elif int(iter_in) == 0:
                try:
                    gtrack_fit2dem = XovOpt.get("outdir") + outdir_in + '/' + track_id + '.pkl'
                    fit2dem_res = gtrack(XovOpt.to_dict)
                    fit2dem_res = fit2dem_res.load(gtrack_fit2dem).sol_prev_iter
                    # if debug:
                    print("Solution of fit2dem for file",track_id+".pkl imported: \n", fit2dem_res['orb'])
                    track.sol_prev_iter = fit2dem_res
                except:
                    True

            tracks.append(track)

        # epo_in = np.array(epo_in)
        # print(epo_in)
        # print(epo_in.shape)
        # print(np.sort(epo_in)[0])
        # print(np.sort(epo_in)[-1])
        # np.savetxt("tmp/epo_mla_1301.in", epo_in, fmt="%10.5f")

        if XovOpt.get("SpInterp") == 3:
            print(
                'Orbit and attitude data loaded for years 20' + str(misycmb[par][0]) + ' and 20' + str(misycmb[par][1]))
            endPrepro = time.time()
            print('----- Runtime Init= ' + str(endPrepro - startPrepro) + ' sec -----' + str(
                (endPrepro - startPrepro) / 60.) + ' min -----')
            exit()

    endPrepro = time.time()
    print('----- Runtime Prepro= ' + str(endPrepro - startPrepro) + ' sec -----' + str(
        (endPrepro - startPrepro) / 60.) + ' min -----')

    startGeoloc = time.time()

    args = ((tr, fil, outdir_in) for (tr, fil) in zip(tracks,allFiles))

    # loop over all gtracks
    if XovOpt.get("parallel"):
        # print((mp.cpu_count() - 1))
        if XovOpt.get("local"):
            # forks everything, if much memory needed, use the remote option with get_context
            from tqdm.contrib.concurrent import process_map  # or thread_map
            _ = process_map(launch_gtrack, args, max_workers=ncores, total=len(allFiles))
        else:
            pool = mp.Pool(processes=ncores)  # mp.cpu_count())
            _ = pool.map(launch_gtrack, args)  # parallel
            pool.close()
            pool.join()
    else:
#        from tqdm import tqdm
        for arg in tqdm(args, total=len(allFiles)):
            launch_gtrack(arg)  # seq

    endGeoloc = time.time()
    print('----- Runtime Geoloc= ' + str(endGeoloc - startGeoloc) + ' sec -----' + str(
        (endGeoloc - startGeoloc) / 60.) + ' min -----')


##############################################
# locate data
if __name__ == '__main__':
    import sys

    ##############################################
    # launch program and clock
    # -----------------------------
    start = time.time()

    args = sys.argv[1:]
    print(args)
    main(args)

    # stop clock and print runtime
    # -----------------------------
    end = time.time()
    print('----- Runtime = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
