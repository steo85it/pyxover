import pdr
import os
import glob
import datetime as dt
import pandas as pd
from scipy.constants import c as clight
import numpy as np

radius = 2440.

pyxover_path = '/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/'
input_folder = 'bel_l3d_sc_o00xxx'
id = "DB0"

# mlardr_cols = ['geoc_long', 'geoc_lat', 'altitude', 'EphemerisTime',
#                      'chn', 'UTC', 'TOF_ns_ET', 'seqid']
mlardr_cols = ['geoc_long', 'geoc_lat', 'altitude', 'EphemerisTime',
                     'chn', 'TOF_ns_ET', 'seqid']

# df_all = pd.DataFrame(columns = ['geoc_long', 'geoc_lat', 'altitude', 'EphemerisTime','chn', 'TOF_ns_ET'])

def writeToFile(df_tosave, mlardr_cols, pyxover_path, id):
   
   df_tosave['chn'] = 0
   df_tosave['seqid'] = range(1,len(df_tosave.index)+1)   
   df_tosave = df_tosave[mlardr_cols]
   orbid = (dt.datetime(2000, 1, 1, 12, 0) +
            dt.timedelta(seconds=df_tosave['EphemerisTime'].iloc[0])).strftime("%y%m%d%H%M")
   
   outdir = f"{pyxover_path}SIM_{orbid[:4]}/{id}/"
   if not os.path.exists(outdir):
      os.makedirs(outdir)
            
   filename = f"{outdir}BELASIMRDR{orbid}.TAB"

   if os.path.isfile(filename):
      print(f'{filename} already exists.')
      df_tosave.to_csv(filename, index=False, sep=',', na_rep='NaN')
      # exit(1)
   else:
      print(filename)
      df_tosave.to_csv(filename, index=False, sep=',', na_rep='NaN')



allFiles1 = glob.glob(os.path.join(pyxover_path, input_folder, 'bel_l3d_sc_o*.xml'))
# allFiles1 = allFiles1[:1]
allFiles1.sort()
allFiles2 = []
pd.options.mode.copy_on_write = False
first =  True
for filename in allFiles1+allFiles2:
# for filename in allFiles1[:5]:
   # filename = f'bel_l3d_sc_o{i:05d}.xml'
   # print(filename)
   # data = pdr.read(f"{pyxover_path}/{input_folder}/{filename}")
   data = pdr.read(filename)
   df = data['BELA_L3D_ADR']
   # df = pdr.read(filename)['BELA_L3D_ADR']
   spot_radius = df.Rp1234m12180 - (df.spot_radius_1+df.spot_radius_2+df.spot_radius_3+df.spot_radius_4 - 12180)
   df = df[np.abs(df.spot_radius_1-spot_radius)<5]
   df = df.rename(columns={'range_1': 'range'})
   # df = df.drop(columns=['spot_radius_1','spot_radius_2','spot_radius_3', 'spot_radius_4',
   #     'fwhm_1', 'fwhm_2', 'fwhm_3','fwhm_4',
   #     'amp_1', 'amp_2', 'amp_3', 'amp_4',
   #     'range_2','range_3', 'range_4','Rp1234m12180',
   #     'sc_lat','sc_lon','sun_dist', 'sun_inc', 'off_nadir', 'tidal_potential', 'orbit'])
   df = df[['et','spot_lat','spot_lon','sc_radius','range']]
   df = df.rename(columns={'et': 'EphemerisTime','spot_lat': 'geoc_lat','spot_lon': 'geoc_long'})
   df['altitude'] = df['sc_radius']-radius
   df['TOF_ns_ET'] = 2*(df['range']*1e3)/clight*1e9 # km to ns
   df = df.drop(columns=['range','sc_radius'])
   
     
   writeToFile(df, mlardr_cols, pyxover_path, id)