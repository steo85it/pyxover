#!/usr/bin/env python3
# ----------------------------------
# Get LOLA PyAltsim output (geolocalised) and save it to illumNG output format (evt compare with original file)
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-Oct-2018
#

import glob
import os
import time

import pandas as pd
import numpy as np

from ground_track import gtrack
import astro_trans as astr
from PyAltSim import prepro_ilmNG
import matplotlib.pyplot as plt
import sys
from prOpt import local, inpdir

if __name__ == '__main__':

    ##############################################
    # launch program and clock
    # -----------------------------
    start = time.time()

    if len(sys.argv) > 1:
        args = sys.argv[1]
    else:
        print("Specify dir or test")
        exit(2)

    # read originals and save nrows

    if local:
      # files = glob.glob('/home/sberton2/Works/NASA/LOLA/aux/'+args+'/slewcheck_1000/boresight_*.?')
      files = glob.glob('/home/sberton2/Works/NASA/LOLA/aux/'+args+'/slewcheck_0/boresight_position_*.?')
    else:
      #files = glob.glob('/att/nobackup/sberton2/LOLA/aux/'+args+'/slewcheck_0/boresight_*.?')
      files = glob.glob(inpdir+args+'/boresight_position_*.?')

    li = {}
    for f in files:
        print("Processing", f)
        tmp = np.loadtxt(f)
        li[f.split('.')[-1]] = tmp
    #print(li)
    df_xin = pd.DataFrame(li['x'])
    df_yin = pd.DataFrame(li['y'])
    df_zin = pd.DataFrame(li['z'])
    
    print(len(df_xin))
    len_table = len(df_xin)



    print("Processing "+args)

    if local:
      dirs = glob.glob('/home/sberton2/Works/NASA/LOLA/data/SIM_'+args[:2]+'/lola/'+args+'/0res_*amp/*') # lro_iaumoon_spice/0res_*')
      files = glob.glob('/home/sberton2/Works/NASA/LOLA/data/SIM_'+args[:2]+'/lola/'+args+'/0res_*amp/MLA*.TAB') # lro_iaumoon_spice/0res_*/MLA*.TAB')
    else:
      dirs = glob.glob('/att/nobackup/sberton2/LOLA/data/SIM_'+args[:2]+'/lola/'+args+'/0res_*amp/*') # lro_iaumoon_spice/0res_*')
      files = glob.glob('/att/nobackup/sberton2/LOLA/data/SIM_'+args[:2]+'/lola/'+args+'/0res_*amp/MLA*.TAB') # lro_iaumoon_spice/0res_*/MLA*.TAB')


    ncols = len(dirs)
    print(ncols)

    #print(dirs,files)
    print(len(files))

    tracks = []
    bores = []
    for f in files:
        bores.append(int(f.split('/')[-2].split('_')[1][:-3]))
        vecopts = {}
        track = gtrack(vecopts)
        track.read_fill(f)
        tracks.append(track)

    # print(track.ladata_df.columns)

    dfx = []
    dfy = []
    dfz = []
    dflat = []
    dflon = []
    dfelv = []

    for idx,track in enumerate(tracks):
        df = track.ladata_df[['ET_TX','geoc_long','geoc_lat','altitude','seqid']]
        df[['seqid']] = df[['seqid']]+1
        df.loc[:,'altitude'] = 1737.4e3 + df.loc[:,'altitude'].values
        dfxyz = pd.DataFrame(np.transpose(astr.sph2cart(df['altitude'].values, df['geoc_lat'].values, df['geoc_long'].values)),columns=['geoc_x','geoc_y','geoc_z'])
        df = pd.concat([df,dfxyz],axis=1)

        df_full = pd.DataFrame(list(range(len_table)),columns=['seqid'])
        df = df_full.merge(df,on='seqid',how='outer').drop('seqid',axis=1)
        dfx.append(df['geoc_x'])
        dfy.append(df['geoc_y'])
        dfz.append(df['geoc_z'])

        dflat.append(df['geoc_lat'])
        dflon.append(df['geoc_long'])
        dfelv.append(df['altitude'])

    dfx = pd.concat(dfx, axis=1)
    dfy = pd.concat(dfy, axis=1)
    dfz = pd.concat(dfz, axis=1)
    dflat = pd.concat(dflat, axis=1)
    dflon = pd.concat(dflon, axis=1)
    dfelv = pd.concat(dfelv, axis=1)
    dfx.columns=dfy.columns=dfz.columns=dflat.columns=dflon.columns=dfelv.columns=bores

    missing_cols = list(set(range(ncols)) - set(bores))
    if len(missing_cols)>0:
        df_miss = pd.DataFrame(columns=missing_cols)
        # print(df_miss)
        dfx = pd.concat([dfx,df_miss],axis=1)
        dfy = pd.concat([dfy,df_miss],axis=1)
        dfz = pd.concat([dfz,df_miss],axis=1)
        dflat = pd.concat([dflat,df_miss],axis=1)
        dflon = pd.concat([dflon,df_miss],axis=1)
        dfelv = pd.concat([dfelv,df_miss],axis=1)

    # reorder
    if ncols>1:
        dfx = dfx[list(range(ncols))]
        dfy = dfy[list(range(ncols))]
        dfz = dfz[list(range(ncols))]
        dflat = dflat[list(range(ncols))]
        dflon = dflon[list(range(ncols))]
        dfelv = dfelv[list(range(ncols))]

    if local:
       outdir_ = '/home/sberton2/Works/NASA/LOLA/out/'+args
    else:
       outdir_ = '/att/nobackup/sberton2/LOLA/out/'+args

    if not os.path.exists(outdir_):
        os.makedirs(outdir_, exist_ok=True)

    dfx.to_csv(outdir_+'/out.x', header=None, index=None, sep='\t',na_rep='Nan')
    dfy.to_csv(outdir_+'/out.y', header=None, index=None, sep='\t',na_rep='Nan')
    dfz.to_csv(outdir_+'/out.z', header=None, index=None, sep='\t',na_rep='Nan')
    dflat.to_csv(outdir_+'/out.lat', header=None, index=None, sep='\t',na_rep='Nan')
    dflon.to_csv(outdir_+'/out.lon', header=None, index=None, sep='\t',na_rep='Nan')
    dfelv.to_csv(outdir_+'/out.elv', header=None, index=None, sep='\t',na_rep='Nan')

    #print(files)

    # df_xin.columns=df_yin.columns=df_zin.columns=bores

    print((dfy))
    print((df_yin*1.e3))
    print((dfy - df_yin*1.e3).dropna())

    fig, ax = plt.subplots()
    (dfx - df_xin*1.e3).mean(axis=1).plot(ax=ax, label='x')
    (dfy - df_yin*1.e3).mean(axis=1).plot(ax=ax, label='y')
    (dfz - df_zin*1.e3).mean(axis=1).plot(ax=ax, label='z')
    ax.legend()
    ax.set_xlabel('')
    ax.set_ylabel('diff_meters')
    plt.savefig(outdir_+'/illumNG_vs_altsim_'+args+'.png')

    # stop clock and print runtime
    # -----------------------------
    end = time.time()
    print('----- Runtime = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
