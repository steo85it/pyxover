
#!/usr/bin/env python3
# # from xovutils.astrotrans import rsw_2_xyz # en vectoriel
import os
from  wd_utils import xov_pkl2csv, getgtrackt0
import time
import pyarrow.parquet as pq
import pandas as pd
import pyarrow as pa
import pickle

id1 = "AA1"
id2 = "CN8"
data_path = "/storage/homefs/desprats/pyxover/examples/CALA/data/"
data_path = "/storage/research/aiub_gravdet/WD_XOV/"
data_path = "/storage/research/aiub_gravdet/WD_BELA/"
in_folder1 = f"{data_path}pyXover/out/{id1}_0/3res_20amp/"
in_folder2 = f"{data_path}pyXover/out/{id2}_0/3res_20amp/"
out_folder = f"{data_path}OBS/"
# list_arcs = [ file.split('_')[1:] for file in os.listdir(f'{out_folder}xov/*pkl')]
list_file = os.listdir(f'{in_folder1}xov/')
list_arcs = [file.split(".")[-2].split('_')[1:] for file in list_file if file.split(".")[-1]=='pkl']
pyout_folder = f"{data_path}pyXover/out/"
track_folder = f"{in_folder1}gtrack_{list_arcs[0][0]}/"
list_tracks = os.listdir(track_folder)

ipk = 1

dir = f"{data_path}pyXover/raw/SIM_2603/AA0/3res_20amp/"
file = "BELASIMRDR2603160004"

df = pd.read_csv(dir+file+".TAB", sep=',', header=0)
df.to_parquet(dir+file+".parquet", engine='pyarrow')


time1 = time.perf_counter()
filename = f"{track_folder}{list_tracks[ipk]}"
print(f"Test file: {filename}")
with open(filename, 'rb') as f:
   content = f.read()
time2 = time.perf_counter()
print(f'Loading gtrack pkl:{time2-time1:.1f}sec')

import gc
gc.disable()
time1 = time.perf_counter()
track = pickle.loads(content)
time2 = time.perf_counter()
gc.enable()
print(f'Unpickling gtrack pkl:{time2-time1:.1f}sec')
   
#self.XovOpt = XovOpt
vecopts = track.vecopts
dr_simit = track.dr_simit
# Laser Altimeter Data (dataframe) ?
ladata_df = track.ladata_df
# self.df_input = None
name = track.name
# Mercury (central body)
MERv = track.MERv  # Velocity
MERx = track.MERx  # Position
# Messenger (probe)
MGRa = track.MGRa  # acceleration
MGRv = track.MGRv  # velocity
MGRx = track.MGRx  # position
# Sun
SUNx = track.SUNx  # position
param = track.param
# Set-up empty offset arrays at init (sim only)
pertPar = track.pertPar
# imposed perts for closed loop sim
pert_cloop = track.pert_cloop
pert_cloop_0 = track.pert_cloop_0
# parameter solution from previous iterations (cumulated)
sol_prev_iter = track.sol_prev_iter
# initial epoch of track (useful for cheby interp
# and linear corrections to track)
t0_orb = track.t0_orb
# store interpolated DEM
dem = track.dem
# spice data (if interp used)
SpObj = track.SpObj

print(ladata_df)

time1 = time.time()
table = pa.Table.from_pandas(ladata_df)
parquet_file = f"{track_folder}{list_tracks[ipk]}.parquet"
pq.write_table(table, parquet_file)
time2 = time.time()
print(f'pkl to parquet + writing parquet:{time2-time1:.1f}sec')

time1 = time.time()
table2 = pq.read_table(parquet_file)
time2 = time.time()
print(f'Loading gtrack parquet:{time2-time1:.1f}sec')

time1 = time.time()
pklFileName = f"{track_folder}{list_tracks[ipk]}2.pkl"
with open(pklFileName, 'wb') as f:
    pickle.dump(ladata_df, f, protocol=-1)
    time2 = time.time()
    print(f'Writing gtrack pkl:{time2-time1:.1f}sec')

time1 = time.time()
with open(pklFileName, "rb") as f:
   track = pickle.load(f)
   time2 = time.time()
   print(f'Loading gtrack pkl:{time2-time1:.1f}sec')

arc1 = 310501
arc2 = 310710
ids = []
ids.append(id1)
ids = [id1,id2]