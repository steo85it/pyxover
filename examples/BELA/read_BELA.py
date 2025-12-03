import pdr
import os
import glob
import datetime as dt
import pandas as pd
from scipy.constants import c as clight

radius = 2440.

pyxover_path = '/storage/homefs/desprats/pyxover/'
pyxover_path = '/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/'
input_folder  = 'prm_2027_2028'
input_folder2 = 'exm_2028_2029'
id_N = "DA2"
id_S = "DA3"

# mlardr_cols = ['geoc_long', 'geoc_lat', 'altitude', 'EphemerisTime',
#                      'chn', 'UTC', 'TOF_ns_ET', 'seqid']
mlardr_cols = ['geoc_long', 'geoc_lat', 'altitude', 'EphemerisTime',
                     'chn', 'TOF_ns_ET', 'seqid']

# df_all = pd.DataFrame(columns = ['geoc_long', 'geoc_lat', 'altitude', 'EphemerisTime','chn', 'TOF_ns_ET'])

def writeToFile(df_tosave, mlardr_cols, pyxover_path, id_N, id_S):
   
   df_tosave['chn'] = 0
   df_tosave['seqid'] = range(1,len(df_tosave.index)+1)   
   df_tosave = df_tosave[mlardr_cols]
   orbid = (dt.datetime(2000, 1, 1, 12, 0) +
            dt.timedelta(seconds=df_tosave['EphemerisTime'].iloc[0])).strftime("%y%m%d%H%M")
         
   if df_tosave['geoc_lat'].head(1).values[0] >= 0:
      outdir = f"{pyxover_path}SIM_{orbid[:4]}/{id_N}/"
   else:
      outdir = f"{pyxover_path}SIM_{orbid[:4]}/{id_S}/"
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
allFiles1.sort()
allFiles2 = glob.glob(os.path.join(pyxover_path, input_folder2, 'bel_l3d_sc_o*.xml'))
allFiles2.sort()
first =  True
for filename in allFiles1+allFiles2:
# for filename in allFiles1[:5]:
   # filename = f'bel_l3d_sc_o{i:05d}.xml'
   # print(filename)
   # data = pdr.read(f"{pyxover_path}/{input_folder}/{filename}")
   data = pdr.read(filename)
   df = data['BELA_L3D_ADR']
   df.rename(columns={'et': 'EphemerisTime','spot_lat': 'geoc_lat','spot_lon': 'geoc_long'}, inplace=True)
   df['altitude'] = df['sc_radius']-radius
   df['TOF_ns_ET'] = 2*(df['range']*1e3)/clight*1e9 # km to ns
   df.drop(columns=['spot_radius','range','sc_lat','sc_lon','sc_radius','sun_dist', 'sun_inc', 'off_nadir', 'tidal_potential', 'orbit'], inplace=True)
   
   if first:
      df_all = df
      first = False
   else:
      df_all = pd.concat([df_all, df], axis=0,ignore_index=True)
      
   while True:
      # first change of sign
      index = df_all[df_all['geoc_lat'] * df_all['geoc_lat'].head(1).values[0] < 0]

      if len(index) == 0:
         break
      index = index.index[0]
      df_tosave = df_all.loc[:index-1]
      df_all = df_all.loc[index:]
      
      writeToFile(df_tosave, mlardr_cols, pyxover_path, id_N, id_S)
      
writeToFile(df_all, mlardr_cols, pyxover_path, id_N, id_S)