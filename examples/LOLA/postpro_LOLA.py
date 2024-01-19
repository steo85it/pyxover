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
from tqdm import tqdm

import pandas as pd
import numpy as np

from pygeoloc.ground_track import gtrack
import xovutil.astro_trans as astr
from pyaltsim.PyAltSim import prepro_ilmNG
import matplotlib.pyplot as plt
import sys
from config import XovOpt
#from prOpt import local, inpdir
from setup_lola import setup_lola

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

    setup_lola()
    XovOpt.set("debug", True)

    # read originals and save nrows

#    if XovOpt.get("local"):
      # files = glob.glob('/home/sberton2/Works/NASA/LOLA/aux/'+args+'/slewcheck_1000/boresight_*.?')
#      files = glob.glob('/home/sberton2/Works/NASA/LOLA/aux/'+args+'/slewcheck_0/boresight_position_*.?')
#    else:
      #files = glob.glob('/att/nobackup/sberton2/LOLA/aux/'+args+'/slewcheck_0/boresight_*.?')
    files = glob.glob(f'{XovOpt.get("inpdir")}{args}/boresight_position_*.?')

    li = {}
    for f in tqdm(files): #, desc=files):
        #print("Processing", f)
        tmp = np.loadtxt(f)
        li[f.split('.')[-1]] = tmp
    #print(li)
    df_xin = pd.DataFrame(li['x'])
    df_yin = pd.DataFrame(li['y'])
    df_zin = pd.DataFrame(li['z'])

    #print(df_xin.values.shape)
    #print(np.transpose(astr.cart2sph(np.vstack([df_xin.values.ravel(), df_yin.values.ravel(), df_zin.values.ravel()]).T)).shape)
    dfrll = pd.DataFrame(np.transpose(astr.cart2sph(np.vstack([df_xin.values.ravel(), df_yin.values.ravel(), df_zin.values.ravel()]).T)),columns=['geoc_r','geoc_lat','geoc_lon'])
    #print(dfrll.dropna())
    #dfrll.geoc_r.plot()
    #plt.show()
    #exit()
   
    #print(df_xin.shape)
    len_table = len(df_xin)

    print("Processing "+args)

    #    if local:
    #      dirs = glob.glob('/home/sberton2/Works/NASA/LOLA/data/SIM_'+args[:2]+'/lola/'+args+'/0res_*amp/*') # lro_iaumoon_spice/0res_*')
    #      files = glob.glob('/home/sberton2/Works/NASA/LOLA/data/SIM_'+args[:2]+'/lola/'+args+'/0res_*amp/MLA*.TAB') # lro_iaumoon_spice/0res_*/MLA*.TAB')
    #    else:
    print(XovOpt.get("rawdir")+'SIM_'+args[:2]+'/'+args+'/0res_*amp/*')
    dirs = glob.glob(XovOpt.get("rawdir")+'SIM_'+args[:2]+'/'+args+'/0res_*amp/*') # lro_iaumoon_spice/0res_*')
    files = glob.glob(XovOpt.get("rawdir")+'SIM_'+args[:2]+'/'+args+'/0res_*amp/MLA*.TAB') # lro_iaumoon_spice/0res_*/MLA*.TAB')

    ncols = len(dirs)
    #print(ncols)

    #print(dirs,files)
    print(len(files))

    tracks = []
    bores = []
    for f in files:
        bores.append(int(f.split('/')[-2].split('_')[1][:-3]))
        vecopts = {}
        track = gtrack(vecopts)
        #print(f)
        track.read_fill(f)
        tracks.append(track)

        #print(track.ladata_df)
        #exit()

    dftdb = []
    dfx = []
    dfy = []
    dfz = []
    dflat = []
    dflon = []
    dfelv = []

    for idx,track in enumerate(tracks):
        df = track.ladata_df[['ET_TX','geoc_long','geoc_lat','altitude','seqid']]
        df.loc[:, 'seqid'] = df.loc[:, 'seqid'].values + 1
        df.loc[:, 'altitude'] = 1737.4e3 + df.loc[:, 'altitude'].values
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


    dftdb = df.loc[:, 'ET_TX']
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
        dfx = pd.concat([dfx,df_miss], axis=1)
        dfy = pd.concat([dfy,df_miss], axis=1)
        dfz = pd.concat([dfz,df_miss], axis=1)
        dflat = pd.concat([dflat,df_miss], axis=1)
        dflon = pd.concat([dflon,df_miss], axis=1)
        dfelv = pd.concat([dfelv,df_miss], axis=1)

    # reorder
    if ncols>1:
        dfx = dfx[list(range(ncols))]
        dfy = dfy[list(range(ncols))]
        dfz = dfz[list(range(ncols))]
        dflat = dflat[list(range(ncols))]
        dflon = dflon[list(range(ncols))]
        dfelv = dfelv[list(range(ncols))]

#    if local:
#       outdir_ = '/home/sberton2/Works/NASA/LOLA/out/'+args
#    else:
    outdir_ = XovOpt.get("outdir")+args

    if not os.path.exists(outdir_):
        os.makedirs(outdir_, exist_ok=True)

    # copy file with epochs
    if not os.path.islink(outdir_+f'/{args}_out.tt'):
        os.symlink(f'{XovOpt.get("inpdir")}{args}/boresight_time_slewcheck.xyzd', outdir_+f'/{args}_out.tt')
    dftdb.to_csv(outdir_+f'/{args}_out.tdb', header=None, index=None, sep='\t', na_rep='Nan')

    
    # save new xyz and lonlatelv
    dfx.to_csv(outdir_+f'/{args}_out.x', header=None, index=None, sep='\t', na_rep='Nan')
    dfy.to_csv(outdir_+f'/{args}_out.y', header=None, index=None, sep='\t', na_rep='Nan')
    dfz.to_csv(outdir_+f'/{args}_out.z', header=None, index=None, sep='\t', na_rep='Nan')
    dflat.to_csv(outdir_+f'/{args}_out.lat', header=None, index=None, sep='\t', na_rep='Nan')
    dflon.to_csv(outdir_+f'/{args}_out.lon', header=None, index=None, sep='\t', na_rep='Nan')
    dfelv.to_csv(outdir_+f'/{args}_out.elv', header=None, index=None, sep='\t', na_rep='Nan')

    #print(files)

    # df_xin.columns=df_yin.columns=df_zin.columns=bores

    #print((dfx - df_xin*1.e3).dropna())
    #print((dfy - df_yin*1.e3).dropna())
    #print((dfz - df_zin*1.e3).dropna())

    (dfx - df_xin*1.e3).to_csv(outdir_+f'/{args}_out.dx', header=None, index=None, sep='\t', na_rep='Nan')
    (dfy - df_yin*1.e3).to_csv(outdir_+f'/{args}_out.dy', header=None, index=None, sep='\t', na_rep='Nan')
    (dfz - df_zin*1.e3).to_csv(outdir_+f'/{args}_out.dz', header=None, index=None, sep='\t', na_rep='Nan')
    
    fig, ax = plt.subplots()
    (dfx - df_xin*1.e3).mean(axis=1).plot(ax=ax, label='x')
    (dfy - df_yin*1.e3).mean(axis=1).plot(ax=ax, label='y')
    (dfz - df_zin*1.e3).mean(axis=1).plot(ax=ax, label='z')
    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('diff_meters')
    plt.savefig(outdir_+'/illumNG_vs_altsim_'+args+'.png')

    df_rin = dfrll.geoc_r.values.reshape(df_xin.shape)
    df_latin = np.rad2deg(dfrll.geoc_lat.values.reshape(df_xin.shape))
    df_lonin = np.rad2deg(dfrll.geoc_lon.values.reshape(df_xin.shape))

    #print((dfelv - pd.DataFrame(df_rin*1.e3)).dropna())
    #print((dflat - pd.DataFrame(df_latin)).dropna())
    #print((dflon - pd.DataFrame(df_lonin)).dropna())

    (dflat - pd.DataFrame(df_latin)).to_csv(outdir_+f'/{args}_out.dlat', header=None, index=None, sep='\t',na_rep='Nan')
    (dflon - pd.DataFrame(df_lonin)).to_csv(outdir_+f'/{args}_out.dlon', header=None, index=None, sep='\t',na_rep='Nan')
    (dfelv - pd.DataFrame(df_rin*1.e3)).to_csv(outdir_+f'/{args}_out.dr', header=None, index=None, sep='\t',na_rep='Nan')
    
    plt.clf()
    fig, ax = plt.subplots()
    (dfelv - pd.DataFrame(df_rin*1.e3)).mean(axis=1).plot(ax=ax, label='r')
    ((dflat - pd.DataFrame(df_latin)).mean(axis=1)*1e4).plot(ax=ax, label='lat')
    ((dflon - pd.DataFrame(df_lonin)).mean(axis=1)*1e4).plot(ax=ax, label='lon')
    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('diff_geod (meters, deg*1e4)')
    plt.savefig(outdir_+'/illumNG_vs_altsim_geod_'+args+'.png')

    print(f"- Data series and plots saved to {outdir_}.")
    
    # stop clock and print runtime
    # -----------------------------
    end = time.time()
    print('----- Runtime = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
