
#!/usr/bin/env python3
import glob
import os
import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

id = "CC9"
iter_max = 4
suffix = "_nosol"
suffix = "_A"
data_path = "/storage/research/aiub_gravdet/WD_BELA/"
plot_tracks = False
plot_weigth = False

startInit = time.time()

nbins = 200

for iter in range(0,iter_max+1):
   pyout_folder = f"{data_path}pyXover/out/{id}_{iter}"
   with open(f"{pyout_folder}/Abmat_{id}_{iter}_{iter+1}{suffix}.pkl", "rb") as file:
      Abmat = pickle.load(file)
      xovers = Abmat.xov.xovers   

   xovers = xovers[['orbA','orbB','dR','weights']]
   if plot_tracks:
      long_df = xovers.melt(
         id_vars=['dR'], 
         value_vars=['orbA', 'orbB'], 
         value_name='orbID'
         )

      # Group by orbID and compute the median of 'dR'
      long_df['abs_dR'] = long_df['dR'].abs()
      long_df = long_df[long_df['abs_dR']<100]
      orb_dict = long_df.groupby('orbID')['abs_dR'].median().to_dict()
      
      plt.hist(orb_dict.values(), nbins, alpha=0.5, label=f"i={iter}")
   elif plot_weigth:
      # plt.hist(xovers['weights'], nbins, alpha=0.5, label=f"i={iter}")
      plt.plot(abs(xovers['dR']),xovers['weights'],'+')
      plt.xscale('log')
      plt.yscale('log')
   else:
      # dR = xovers.loc[np.abs(xovers['dR'])<100, 'dR']
      dR = xovers.loc[np.abs(xovers['dR'])<200, 'dR']
      # dR = xovers['dR']
      plt.hist(dR, nbins, alpha=0.5, label=fr"#{iter}: {int(len(dR)/1000)}kX $\mu$={np.mean(dR):.1f}m $\sigma$={int(np.std(dR))}m")
      # plt.hist(dR, nbins, alpha=0.5, label=fr"#{iter}: {int(len(dR)/1000)}kX $\mu$={np.mean(dR):.1f}m $\sigma$={int(np.std(dR))}m $med$={np.median(dR):.1f}m")

plt.legend()

if plot_tracks:
   # plt.xlim(-100, 100)
   plt.xlabel("Radial bias [m]")
   plt.ylabel("Number of tracks")
   fig_name = f"tracks_{id}{suffix}"
elif plot_weigth:
   plt.xlabel(r"$|\nu|$ [m]")
   plt.ylabel("Weight")
   plt.grid()
   fig_name = f"weight_{id}{suffix}"
else:
   # plt.xlim(-100, 100)
   plt.xlabel(r"$\nu$ [m]")
   plt.ylabel("Number of crossovers")
   fig_name = f"xovers_{id}{suffix}"
   
plt.show()
plt.savefig(f"{fig_name}.png")

endInit = time.time()
print(f"Finished after {str(endInit-startInit)}s")