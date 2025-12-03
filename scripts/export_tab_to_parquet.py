
import pdr
import os
import glob
import datetime as dt
import pandas as pd
from scipy.constants import c as clight

radius = 2440.

pyxover_path = '/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/'
out_path = '/storage/research/aiub_gravdet/WD_BELA/pyXover/raw/AD6/'
id = "AD6"

allFiles = glob.glob(os.path.join(pyxover_path, '*', id, '*.TAB'))
# allFiles = [allFiles[0]]

for filename in allFiles:
   df = pd.read_csv(filename, sep=',', header=0)
   df = df[['geoc_long','geoc_lat','altitude','EphemerisTime']]
   filnam = filename.split('/')[-1][:-4]
   df.to_parquet(out_path + filnam + ".parquet", engine='pyarrow')