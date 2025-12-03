import pickle
import time
import gc
import pandas as pd
import numpy as np
import itertools as itert
import matplotlib.pyplot as plt


import sys
import os.path
import glob
import csv


std_sol = False # formal error or solution

data_path = "/storage/research/aiub_gravdet/WD_BELA/"
in_folder = f"{data_path}pyXover/out/{id}_0/"

sol = []
std = []
glob_nam = ['RA', 'DEC','PM','L', 'h2']
glob_nam = ['RA', 'DEC','PM','L']
# glob_nam = ['RA', 'DEC','PM', 'h2']
# glob_nam += [f'LIB{i}' for i in range(1, 12)]
plot_bertone2021 = False

file_names = [f"{data_path}pyXover/out/CC5_0/Abmat_CC5_0_1_A.pkl",
              f"{data_path}pyXover/out/CC5_0/Abmat_CC5_0_1_B.pkl",
              f"{data_path}pyXover/out/CC5_0/Abmat_CC5_0_1_C.pkl",
              f"{data_path}pyXover/out/CC5_0/Abmat_CC5_0_1_D.pkl",
              f"{data_path}pyXover/out/CC5_0/Abmat_CC5_0_1_E.pkl",
              f"{data_path}pyXover/out/CC5_0/Abmat_CC5_0_1_F.pkl",
              f"{data_path}pyXover/out/CC5_0/Abmat_CC5_0_1_G.pkl",
              f"{data_path}pyXover/out/CC5_0/Abmat_CC5_0_1_H.pkl",]
leg = [f[-5:-4] for f in file_names]
file_names = [f"{data_path}pyXover/out/CC5_3/Abmat_CC5_3_4_A.pkl",
              f"{data_path}pyXover/out/CC6_3/Abmat_CC6_3_4_A.pkl",
              f"{data_path}pyXover/out/CC6_9/Abmat_CC6_9_10_B.pkl",
              f"{data_path}pyXover/out/CC7_3/Abmat_CC7_3_4_A.pkl",
              f"{data_path}pyXover/out/CC8_9/Abmat_CC8_9_10_A.pkl",
              f"{data_path}pyXover/out/CC9_23/Abmat_CC9_23_24_A.pkl",
              f"{data_path}pyXover/out/CD0_6/Abmat_CD0_6_7_A.pkl"]
leg = ["5_3A", "6_3A", "6_9B", "7_3A", "8_9A", "9_23A", "0_6A"]
file_names = [f"{data_path}pyXover/out/CC6_9/Abmat_CC6_9_10_B.pkl",
              f"{data_path}pyXover/out/CC8_9/Abmat_CC8_9_10_A.pkl",
              f"{data_path}pyXover/out/CC9_23/Abmat_CC9_23_24_A.pkl",
              f"{data_path}pyXover/out/CD0_7/Abmat_CD0_7_8_A.pkl"]
leg = ["6_9", "8_9", "9_23", "0_7"]
file_names = [f"{data_path}pyXover/out/CC6_7/Abmat_CC6_7_8_B.pkl",
              f"{data_path}pyXover/out/CC8_7/Abmat_CC8_7_8_A.pkl",
              f"{data_path}pyXover/out/CC9_7/Abmat_CC9_7_8_A.pkl",
              f"{data_path}pyXover/out/CD0_7/Abmat_CD0_7_8_A.pkl"]
leg = ["6_7", "8_7", "9_7", "0_7"]


id = "CC5"
fig_name = f"sol_{id}"

glob_lbl = {'RA' : r"$\alpha_0$ [as]",
            'DEC': r"$\delta_0$ [as]",
            'PM' : r"$\omega$ [as.yr$^{-1}$]",
            'L'  : r"$L$ [as]"}

# glob_lbl['h2'] = r"$h_2$ [-]"
# for i in range(1, 12):
#     glob_lbl[f'LIB{i}'] = fr"$\lambda_{{{i}}}$ [as]"
color = ['#000000','#4477AA', '#228833',
            '#66CCEE', '#EE6677', '#CCBB44', "#7844CC", "#44CC6D"]

units = {'RA':'[as]', 'DEC':'[as]','PM':'[as/y]','L':'[as]', 'h2':'[-]'}

sol = {nam: [] for nam in glob_nam}
std = {nam: [] for nam in glob_nam}

for (iter,filename) in enumerate(file_names):
   with open(filename, "rb") as file:
      Abmat = pickle.load(file)
      for nam in glob_nam:
         dnam = f"dR/d{nam}"
         if dnam in Abmat.sol_dict['sol'].keys():
            sol[nam].append(Abmat.sol_dict['sol'][dnam])
            std[nam].append(Abmat.sol_dict['std'][dnam])
         else:
            sol[nam].append(np.nan)
            std[nam].append(np.nan)

# fig, ax = plt.subplots(layout='constrained',figsize=(8,4.8))
fig, axs = plt.subplots(2, 2)

axs = axs.ravel()  # Flatten the 2D array of axes

for i, name in enumerate(glob_nam):
   if np.isnan(sol[name][-1]):
      continue
   axs[i].errorbar(leg, sol[name], yerr=[3*s for s in std[name]])
   axs[i].set_ylabel(f"{name} {units[name]}")
   axs[i].tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.savefig(f"{fig_name}.png")