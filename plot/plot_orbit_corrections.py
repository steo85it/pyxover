
#!/usr/bin/env python3
import glob
import os
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

id = "CC5"
iter = 0
suffix = "_nosol"
suffix = "_D"
data_path = "/storage/research/aiub_gravdet/WD_BELA/"

startInit = time.time()


pyout_folder = f"{data_path}pyXover/out/{id}_{iter}"
with open(f"{pyout_folder}/Abmat_{id}_{iter}_{iter+1}{suffix}.pkl", "rb") as file:
   Abmat = pickle.load(file)
   sol_dict = Abmat.sol_dict['sol']
   for lbl in ['A','C','R']:
      filtered = {k.split('_')[0]: v for k, v in sol_dict.items() if k.endswith(f"_dR/d{lbl}")}
      # track_id   = list(filtered.keys()).astype('int')
      track_id   = [int(k) for k in filtered.keys()]
      track_corr = list(filtered.values())
      # plt.plot(track_id,track_corr,'o',label=lbl)
      plt.plot(range(0,len(track_corr)),track_corr,'o',label=lbl, markersize=5)

plt.legend()


# plt.xlim(-100, 100)
plt.xlabel("MLA track index")
plt.ylabel("Estimated orbit correction [m]")
plt.show()
fig_name = f"orbcor_{id}_{iter}{suffix}"
plt.savefig(f"{fig_name}.png")

endInit = time.time()
print(f"Finished after {str(endInit-startInit)}s")