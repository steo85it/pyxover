import pandas as pd
import sys
import numpy as np
from functools import reduce
from tqdm import tqdm
import glob
import os

slew_doy = sys.argv[1]

center_cols = {'spot_1':0, 'spot_2':73, 'spot_3':146, 'spot_4':219, 'spot_5':292}

outdir = f"/explore/nobackup/people/sberton2/LOLA/PyXover/examples/LOLA/data/out/{slew_doy}/"
csvout = f"{outdir}corr_{slew_doy}.center"
if os.path.exists(csvout):
    print(f"- {csvout} exists. Exit.")
    exit()
else:
    print(f"Processing {slew_doy}...")
    
#doys = glob.glob(f"/explore/nobackup/people/sberton2/LOLA/PyXover/examples/LOLA/data/out/?????????")
#print(doys.split('/')[-1])

try:
    ET = pd.read_csv(f"{outdir}out.time", sep="\s+", header=None)
except:
    print(f"- No ET epochs file for {slew_doy}. Exit.")
    exit()
    
spotlist = []
for var in tqdm(['lon', 'lat', 'elv']):
    var_cent = pd.read_csv(f"{outdir}out.{var}", sep="\s+", header=None)
    try:
        var_cent = var_cent.iloc[:, list(center_cols.values())]
    except:
        print(var_cent)
        exit()
        
    var_cent.columns = [f"{c}_{var}" for c in center_cols.keys()]
    var_cent = var_cent.replace('Nan', np.nan)
    var_cent['ET'] = ET.values
    var_cent = var_cent.set_index('ET')
    spotlist.append(var_cent)

df = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), spotlist)
df = df.dropna(axis=0, how='all').apply(pd.to_numeric)
print(df)
df.to_csv(f"{outdir}corr_{slew_doy}.center", na_rep='NaN')
print(f"- Selection saved to {outdir}corr_{slew_doy}.center .")
