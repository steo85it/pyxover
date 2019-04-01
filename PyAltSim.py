#!/usr/bin/env python3
# ----------------------------------
# PyXover
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-Oct-2018
#
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import shutil
import sys
import glob
import time
import pickleIO
import datetime

import numpy as np
import pandas as pd
from scipy.constants import c as clight
import multiprocessing as mp
import subprocess

import spiceypy as spice

# mylib
from prOpt import debug, parallel, new_gtrack, outdir, auxdir, local, new_illumNG
import astro_trans as astr
from ground_track import gtrack

########################################
# start clock
start = time.time()

##############################################
class sim_gtrack(gtrack):
    def __init__(self, vecopts, orbID):
        gtrack.__init__(self,vecopts)
        self.orbID = orbID
        self.name = str(orbID)

    def setup(self, df):
        df_ = df.copy()

        # get time of flight in ns from probe one-way range in km
        df_['TOF']=df_['rng']*2.*1.e3/clight
        # preparing df for geoloc
        df_['seqid'] = df_.index
        df_ = df_.rename(columns={"epo_tx": "ET_TX"})
        df_ = df_.reset_index(drop=True)
        # copy to self
        self.ladata_df = df_[['ET_TX', 'TOF','orbID', 'seqid']]

        # retrieve spice data for geoloc
        if not hasattr(self, 'SpObj'):
            # create interp for track
            self.interpolate()
        else:
            self.SpObj = pickleIO.load(auxdir + 'spaux_' + self.name + '.pkl')

        # actual processing
        self.lt_iter(itmax=100,df=df_)
        self.setup_rdr()

    def lt_iter(self,itmax,df):

        olddr = 1000
        for it in range(itmax):

            self.geoloc()

            if debug:
        	print("it = "+str(it))
        	print(max(abs(olddr - self.dr_simit)))
        	print(len(self.dr_simit) - np.sum([abs(olddr - self.dr_simit) < 1.e-4]))

            df['TOF'] += 2.*self.dr_simit/clight
            if (max(abs(olddr - self.dr_simit)) < 1.e-4):
                #print("Convergence reached",df.columns)
                break
            if (it == itmax - 1):
                print('### altsim: Max number of iterations reached!')
                break

            olddr = self.dr_simit
            # update TOF
            self.ladata_df = df.copy()

    def setup_rdr(self):
        df_ = self.ladata_df.copy()
        mlardr_cols = ['geoc_long', 'geoc_lat', 'altitude', 'EphemerisTime', 'MET', 'frm',
                       'chn', 'Pulswd', 'thrsh', 'gain', '1way_range', 'Emiss', 'TXmJ',
                       'UTC', 'TOF_ns_ET', 'Sat_long', 'Sat_lat', 'Sat_alt', 'Offnad', 'Phase',
                       'Sol_inc', 'SCRNGE', 'seqid']
        self.rdr_df = pd.DataFrame(columns=mlardr_cols)

        df_['TOF_ns_ET']= np.round(df_['TOF'].values*1.e9,1)
        df_['chn']= 0

        df_['UTC']= pd.to_datetime(df_['ET_TX'], unit='s',
                   origin=pd.Timestamp('2000-01-01T12:00:00'))

        df_ = df_.rename(columns={'ET_TX':'EphemerisTime',
                                  'LON':'geoc_long','LAT':'geoc_lat','R':'altitude',
                                  })
        df_ = df_.reset_index(drop=True)
        self.rdr_df = self.rdr_df.append(df_[['EphemerisTime','geoc_long','geoc_lat','altitude',
                                         'UTC', 'TOF_ns_ET','chn','seqid']], sort=True)[mlardr_cols]

##############################################

def prepro_ilmNG(df_):
    #df_ = dfin.copy()
    df_ = pd.concat(li, axis=1)
    df_=df_.apply(pd.to_numeric, errors='coerce')
    #print(df_.rng.min())
    df_ = df_[df_.rng < 1200]
    df_=df_.rename(columns={"xyzd": "epo_tx"})
    #print(df_.dtypes)

    df_['diff'] = df_.epo_tx.diff().fillna(0)
    #print(df_[df_['diff'] > 1].index.values)
    arcbnd = [0]
    arcbnd.extend(df_[df_['diff'] > 1].index.values)
    arcbnd.extend([df_.index.max() + 1])
    #print(arcbnd)
    df_['orbID'] = 0
    for i,j in zip(arcbnd,arcbnd[1:]):
        orbid = (datetime.datetime(2000, 1, 1, 12, 0) + datetime.timedelta(seconds=df_.loc[i, 'epo_tx'])).strftime("%y%m%d%H%M")
        df_.loc[df_.index.isin(np.arange(i, j)), 'orbID'] = orbid

    return df_

##############################################
# locate data

if local == 0:
    #data_pth = '/att/nobackup/sberton2/MLA/MLA_RDR/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
    #dataset = ''  # 'small_test/' #'test1/' #'1301/' #
    #data_pth += dataset
    # load kernels
    spice.furnsh('/att/nobackup/emazaric/MESSENGER/data/furnsh/furnsh.MESSENGER.def')  # 'aux/mymeta')
else:
    #data_pth = '/home/sberton2/Works/NASA/Mercury_tides/data/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
    #dataset = "test/"  # ''  # 'small_test/' #'1301/' #
    #data_pth += dataset
    # load kernels
    spice.furnsh(auxdir + 'mymeta')  # 'aux/mymeta')

if parallel:
    # set ncores
    ncores = mp.cpu_count() - 1  # 8
    print('Process launched on ' + str(ncores) + ' CPUs')

# Setup some useful options
vecopts = {'SCID': '-236',
           'SCNAME': 'MESSENGER',
           'SCFRAME': -236000,
           'INSTID': (-236500, -236501),
           'INSTNAME': ('MSGR_MLA', 'MSGR_MLA_RECEIVER'),
           'PLANETID': '199',
           'PLANETNAME': 'MERCURY',
           'PLANETRADIUS': 2440.,
           'PLANETFRAME': 'IAU_MERCURY',
           'OUTPUTTYPE': 1,
           'ALTIM_BORESIGHT': '',
           'INERTIALFRAME': 'J2000',
           'INERTIALCENTER': 'SSB',
           'PARTDER': ''}

#out = spice.getfov(vecopts['INSTID'][0], 1)
# updated w.r.t. SPICE from Mike's scicdr2mat.m
vecopts['ALTIM_BORESIGHT'] = [0.0022105, 0.0029215, 0.9999932892]  # out[2]
###########################

# generate list of epochs
epo0 = 410270400 # get as input parameter
epo_tx = np.array([epo0+i for i in range(86400*7)])
#epo_tx = np.array([epo0+i/step for i in range(86400*step)])

# read illumNG output and generate df
#    os.makedirs('', exist_ok=True)
#    os.makedirs(os.path.dirname(filename), exist_ok=True)

if local:
   if new_illumNG:
        np.savetxt("tmp/epo.in", epo_tx, fmt="%4d")
        print("Do you have all of illumNG predictions?")
   path = '../aux/illumNG/sph/' #grd/' # use your path
        #illumNGf = glob.glob(path + "bore*")
else:
   if new_illumNG:
        np.savetxt("illumNG/epo.in", epo_tx, fmt="%4d")
        print("illumNG call")
        illumNG_call = subprocess.call(
            ['sbatch', 'doslurmEM', 'MLA_raytraces.cfg'],
            universal_newlines=True, cwd="illumNG/")
        for f in glob.glob("illumNG/bore*"):
            shutil.move(f, auxdir+'/illumNG/grd/'+str(epo0)+"_"+f.split('/')[1])
    
   path = auxdir+'illumNG/grd/' #sph/' # use your path

illumNGf = glob.glob(path+str(epo0)+"_"+"bore*")

li = []
for f in illumNGf:
    print("Processing", f)
    df = pd.read_csv(f, index_col=None, header=0, names=[f.split('.')[-1]])
    li.append(df)

#else:
# launch illumNG directly

df = prepro_ilmNG(df)

tracks = []
for i in list(df.groupby('orbID').groups.keys()):
    if debug:
        print("Processing",i)
    track = sim_gtrack(vecopts, i)
    #track.pertPar['dR'] = 100
    track.setup(df[df['orbID']==i])
    #tracks.append(track)
    track.rdr_df.to_csv(outdir+'MLASIMRDR'+track.name+'.TAB', index=False, sep=',',na_rep='NaN')
    print('Simulated observations written to',outdir+'MLASIMRDR'+track.name+'.TAB')

# stop clock and print runtime
# -----------------------------
end = time.time()
print('----- Runtime = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
