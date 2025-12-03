#!/usr/bin/env python3
import os
import time
import glob
import datetime as dt
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pdr
from scipy.constants import c as clight


data_path = "/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/"

id_ = ["DA0","AD8"]
id_ = ["DA2"]
min_time0 = 0
max_time0 = 1e15
c = 'b'
b = 0

fig, axs = plt.subplots(3,1)
for id in id_:

   path= os.path.join(data_path,'SIM_2704',id, '*.TAB')

   allFiles1 = glob.glob(path)
   allFiles1.sort()
   allFiles1 = allFiles1[1:2]
   allFiles1 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA1/BELASIMRDR2704050855.TAB",
                "/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA1/BELASIMRDR2704300632.TAB"]
   
   allFiles1 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA0/BELASIMRDR2704010213.TAB",
                "/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA0/BELASIMRDR2704300745.TAB"]
   # allFiles1 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA0/BELASIMRDR2704010213.TAB"]
   # allFiles1 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA0/BELASIMRDR2704010435.TAB"]
   # allFiles1 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA0/BELASIMRDR2704010213.TAB",
   #              "/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA0/BELASIMRDR2704010435.TAB"]
   # # allFiles1 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA0/BELASIMRDR2704300745.TAB"]
   # allFiles1 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA0/BELASIMRDR2704010057.TAB",
   #              "/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA0/BELASIMRDR2704010213.TAB",
   #              "/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA0/BELASIMRDR2704010435.TAB"]
   # allFiles1 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA2/BELASIMRDR2704010057.TAB",
   #              "/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA2/BELASIMRDR2704010213.TAB"]
   allFiles1 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA0/BELASIMRDR2704041515.TAB",
                "/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA0/BELASIMRDR2704080417.TAB"]
   # allFiles1 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/SIM_2704/DA0/BELASIMRDR2704041737.TAB"]
   
   
   allFiles2 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/out/DA3_0/gtrack_2704/gtrack_ladata_2704010213.parquet",
                "/storage/research/aiub_gravdet/WD_BELA/pyXover/out/DA3_0/gtrack_2704/gtrack_ladata_2704300745.parquet"]
   
   allFiles3 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/prm_2027_2028/bel_l3d_sc_o00000.xml",
                "/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/prm_2027_2028/bel_l3d_sc_o00297.xml"]
   # allFiles3 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/prm_2027_2028/bel_l3d_sc_o00000.xml"]
   # allFiles3 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/prm_2027_2028/bel_l3d_sc_o00000.xml",
   #              "/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/prm_2027_2028/bel_l3d_sc_o00001.xml"]
   # # allFiles3 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/prm_2027_2028/bel_l3d_sc_o00297.xml"]
   allFiles3 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/prm_2027_2028/bel_l3d_sc_o00036.xml",
                "/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/prm_2027_2028/bel_l3d_sc_o00072.xml"]
   # allFiles3 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/prm_2027_2028/bel_l3d_sc_o00037.xml"]
   
   allFiles2 = ["/storage/research/aiub_gravdet/WD_BELA/pyXover/out/CB8_0/gtrack_12/gtrack_ladata_1202010852.parquet",
                "/storage/research/aiub_gravdet/WD_BELA/pyXover/out/CB8_0/gtrack_13/gtrack_ladata_1301280753.parquet"]
   # allFiles2 = []
   allFiles1 = []
   allFiles3 = []
   min_time = 0
   nobs = 0
   for infil in allFiles1:
      print(infil)
      df = pd.read_csv(infil, sep=',', header=0)
      #df = df[df['geoc_lat']<0]
      df.loc[df['geoc_long']<0,'geoc_long'] += 360
      if min_time == 0:
         min_time = df['EphemerisTime'][0]
      dft = pd.to_datetime(df['EphemerisTime'], unit='s',origin=pd.Timestamp('2000-01-01T12:00:00'))
      print(dft[0])
      print(dft.tail(1).values[0])
      # plt.plot(df['EphemerisTime']+b*df['TOF_ns_ET']*1e-9, df['TOF_ns_ET']*1e-6,color=c)
      # plt.plot(df['TOF_ns_ET']*1e-6,df['geoc_lat'],color=c)
      # plt.plot(df['TOF_ns_ET']*1e-9*3e8/2,df['geoc_lat'],color=c)
      # axs[0].plot(dft,df['altitude']*1e3 - df['TOF_ns_ET']*1e-9*clight/2,'+')
      # axs[0].plot(dft,df['geoc_lat'],'+')
      # axs[1].plot(dft,df['geoc_long'],'+')
      
      axs[0].plot(df['geoc_long'],df['geoc_lat'],'+')
      axs[1].plot(df['geoc_long'],df['altitude']*1e3,'+')
      axs[2].plot(df['geoc_lat'],df['altitude']*1e3 ,'+')
      
      # plt.plot(df['geoc_long']+b,df['altitude']*1e3 - df['TOF_ns_ET']*1e-9*3e8/2,'+')
      # plt.plot(df['geoc_long']+b,df['geoc_lat'],'+')
      # plt.plot(df['geoc_long']+b,df['R'],'+')
      # plt.plot(df['EphemerisTime'],df['geoc_long']+b,color=c)
      max_time = df['EphemerisTime'].tail(1).values[0]
      nobs+=len(df['EphemerisTime'])

   for infil in allFiles3:
      print(infil)
      data = pdr.read(infil)
      df = data['BELA_L3D_ADR']
      df = df[df['spot_lat']>0]
      # print(df['et'][0])
      dft = pd.to_datetime(df['et'], unit='s',origin=pd.Timestamp('2000-01-01T12:00:00'))
      print(dft[0])
      print(dft.tail(1).values[0])
      
      # df['altitude'] = df['sc_radius']-2440
      # df['TOF_ns_ET'] = 2*(df['range']*1e3)/3e8*1e9 # km to ns
      # axs[0].plot(dft,df['altitude']*1e3 - df['TOF_ns_ET']*1e-9*3e8/2,'+')
      # axs[0].plot(dft,(df['spot_radius']-2440)*1e3,'+')
      # axs[0].plot(dft,df['spot_lat'],'+')
      # axs[1].plot(dft,df['spot_lon'],'+')
      
      # axs[1].plot(df['spot_lon'],df['altitude']*1e3 - df['TOF_ns_ET']*1e-9*3e8/2,'+')
      # axs[2].plot(df['spot_lat'],df['altitude']*1e3 - df['TOF_ns_ET']*1e-9*3e8/2,'+')
   
      axs[0].plot(df['spot_lon'],df['spot_lat'],'+')
      axs[1].plot(df['spot_lon'],(df['spot_radius']-2440)*1e3,'+')
      axs[2].plot(df['spot_lat'],(df['spot_radius']-2440)*1e3,'+')
   print(nobs)
   c = 'r'
   b=0
   # min_time0 = max(min_time0,min_time)
   # max_time0 = min(max_time0,max_time)
   
for infil in allFiles2:
   print(infil)
   track = pd.read_parquet(infil, engine='pyarrow')
   track.loc[track['LON']<0,'LON'] += 360
   # df.loc[df['geoc_long']<0,'geoc_long'] += 360
   # plt.plot(track['LON'],track['LAT'],'+')
   # plt.plot(track['LON'],track['TOF']*3e8/2,'+')
   axs[0].plot(track['LON'],track['LAT'],'+')
   axs[1].plot(track['LON'],track['R'],'+')
   axs[2].plot(track['LAT'],track['R'],'+')

# -167.6842
# 89.97682
lon_xov = -167.6842 + 360
# lon_xov = -171.1508 + 360
lat_xov = 89.97682
min_lon=  185
max_lon=  210
min_lat=  89.9
max_lat=  90.05
elev_min = -2100
elev_max = -500
# axs[0].set_xlim(min_lon,max_lon)
# axs[0].set_ylim(min_lat,max_lat)
# axs[1].set_xlim(min_lon,max_lon)
# axs[1].set_ylim(elev_min,elev_max)
# axs[2].set_xlim(min_lat,max_lat)
# axs[2].set_ylim(elev_min,elev_max)
# axs[0].plot(np.array([1, 1])*lon_xov,np.array([min_lat, max_lat]))
# axs[0].plot(np.array([min_lon,max_lon]),lat_xov*np.array([1, 1]),'g')
axs[0].set_ylabel('LAT [°]')
axs[0].set_xlabel('LON [°]')
axs[1].set_ylabel('Elevation [m]')
axs[1].set_xlabel('LON [°]')
axs[2].set_ylabel('Elevation [m]')
axs[2].set_xlabel('LAT [°]')
# axs[1].plot(np.array([1, 1])*lon_xov,np.array([elev_min, elev_max]))
# axs[2].plot(np.array([min_lon,max_lon]),lat_xov*np.array([1, 1]),'g')
# axs[2].plot(lat_xov*np.array([1, 1]),np.array([elev_min, elev_max]))
# plt.xlim(210.42,210.44)
# plt.ylim(79.2,79.6)
# plt.ylim(-1950,-1700)
# plt.plot(np.array([1, 1])*210.43,np.array([-2000, 79]))
max_time0 = 8.6041e8
min_time0 = 8.6040e8
max_time0 = 8.60404e8+650
min_time0 = 8.60404e8+400
# plt.xlim(min_time0,max_time0)
# plt.xlabel('TOF [ms]')
# plt.ylim(3.08,3.12)
# plt.xlabel('altitude [m]')
# plt.xlabel('LON [°]')
# plt.ylabel('LAT [°]')
# plt.ylim(0,90)
fig.tight_layout(pad=0.2)
fig_name = "test"
plt.show()
plt.savefig(f"examples/BELA/{fig_name}.png")