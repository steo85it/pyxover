
#!/usr/bin/env python3
import glob
import os
import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

id = "CD6"
suffixes = ["B","A","A","A"]
suffixes = ["A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A"]
suffixes = ["A" for i in range(0,25)]
# suffixes = ["A","A","B","B","B","B","B","B","B","B"]
# suffixes = ["A","A","A","A","A"]
# suffixes = ["A","A"]
data_path = "/storage/research/aiub_gravdet/WD_BELA/"
rel = False
old_sol = True

startInit = time.time()

color = ['#000000','#4477AA', '#228833', '#66CCEE', '#EE6677',
         '#CCBB44', "#7844CC", "#44CC6D", "#CC4444"]
# glob_nam = ['RA', 'DEC','PM','L', 'h2']
units = {'RA':'[as]', 'DEC':'[as]','PM':'[as/y]','L':'[as]', 'h2':'[-]'}
glob_nam = ['RA', 'DEC','PM','L']
sol = {nam: [] for nam in glob_nam}
std = {nam: [] for nam in glob_nam}

for (iter,suffix) in enumerate(suffixes):
   pyout_folder = f"{data_path}pyXover/out/{id}_{iter}"
   filename=f"{pyout_folder}/Abmat_{id}_{iter}_{iter+1}_{suffix}.pkl"
   if not os.path.exists(filename):
      continue
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


# fig, axs = plt.subplots(2, 3)
fig, axs = plt.subplots(2, 2)

axs = axs.ravel()  # Flatten the 2D array of axes

apr = {'RA':281.0103,
       'DEC':61.4155,
       'PM':6.1385108,
       'L':38.5,
       'h2':0.95}

apr2 = {'RA':281.0082,
       'DEC':61.4164,
       'PM':6.1385054,
       'L':38.5,
       'h2':0.95}

othersols = {'Bertone(2021)': {
                'RA':281.0093,
                'DEC':61.4153,
                'PM':6.138510,
                'L':39.03,
                'h2':1.55},
             'Stark(2015)': {
                'RA':281.00980,
                'DEC':61.4156,
                'PM':6.13851804,
                'L':38.9,
                'h2':np.nan},
             'Margot(2012)': {
                'RA':281.0103,
                'DEC':61.4155,
                'PM':6.1385025,
                'L':38.5,
                'h2':np.nan},
             'Mazarico(2014)':{
                'RA':281.00480,
                'DEC':61.41436,
                'PM':6.138511,
                'L':np.nan,
                'h2':np.nan},
             'Genova(2019)': {
                'RA':281.0082,
                'DEC':61.4164,
                'PM':6.1385054,
                'L':40.0,
                'h2':np.nan},
             'Verma(2016)':{
                'RA':281.00975,
                'DEC':61.41828,
                'PM':np.nan,
                'L':np.nan,
                'h2':np.nan},
             'Konopliv(2020)':{
                'RA':281.0138,
                'DEC':61.4161,
                'PM':6.138514,
                'L':np.nan,
                'h2':np.nan}}

otherstds = {'Bertone(2021)': {
                'RA':6.3e-4,
                'DEC':4.8e-4,
                'PM':2.8e-6,
                'L':1.1,
                'h2':0.65},
             'Stark(2015)': {
                'RA':8.8e-4,
                'DEC':1.6e-3,
                'PM':9.4e-7,
                'L':1.3,
                'h2':np.nan},
             'Margot(2012)': {
                'RA':1.4e-3,
                'DEC':1.4e-3,
                'PM':0,
                'L':1.6,
                'h2':np.nan},
             'Mazarico(2014)':{
                'RA':0.0054,
                'DEC':0.0021,
                'PM':1.15e-6,
                'L':np.nan,
                'h2':np.nan},
             'Genova(2019)': {
                'RA':29.e-4,
                'DEC':3.e-4,
                'PM':0.0000013,
                'L':8.7,
                'h2':np.nan},
             'Verma(2016)':{
                'RA':0.0048,
                'DEC':0.0028,
                'PM':np.nan,
                'L':np.nan,
                'h2':np.nan},
             'Konopliv(2020)':{
                'RA':2.5e-3,
                'DEC':1.7e-3,
                'PM':6e-6,
                'L':np.nan,
                'h2':np.nan}}


for sid in othersols.keys():
   for name in glob_nam:
      othersols[sid][name]  -= apr[name]
      if name == 'RA' or name == 'DEC': # deg2as
         othersols[sid][name] *= 3600
         otherstds[sid][name] *= 3600
      elif name == 'PM': # deg/day to as/y
         othersols[sid][name] *= 3600*365.25
         otherstds[sid][name] *= 3600*365.25

# for name in glob_nam:
#    sol[name] = np.asarray(sol[name]) - np.asarray(apr[name]) + np.asarray(apr2[name])
#    #sol[name]+=apr[name]-apr2[name]

print("Solution")
for name in glob_nam:
   if name == 'RA' or name == 'DEC':
      print(f"{name}:{apr[name] + sol[name][-1]/3600}+-{3*std[name][-1]/3600}")
   elif name == 'PM':
      print(f"{name}:{apr[name] + sol[name][-1]/3600/365.25}+-{3*std[name][-1]/3600/365.25}")
   else:
      print(f"{name}:{apr[name] + sol[name][-1]}+-{3*std[name][-1]}")
print(r"$\sigma$ ratio at convergence")
for i, name in enumerate(glob_nam):
   if np.isnan(sol[name][-1]):
      continue
   print(f"{name}: {np.abs(sol[name][-2]-sol[name][-1])/std[name][-1]/3}")
   if rel:
      improv = [np.abs(sol[name][j]-sol[name][j+1]) for j in range(0,len(sol[name])-1)]
      axs[i].plot(improv/std[name][-1]/3,label=name)
      axs[i].plot([0,len(sol[name])],np.ones(2))
      axs[i].set_ylabel(fr"{name} [3$\sigma$]")
   else:
      axs[i].plot(sol[name],color=color[1],label="sol")
      axs[i].plot(np.ones(2)*(len(sol[name])-1),
                  np.array([-1, 1])*3*std[name][-1]+sol[name][-1],color=color[1])
      # axs[i].plot([0,len(sol[name])],np.ones(2)*bertone2021[name],label='B')
      axs[i].set_ylabel(f"{name} {units[name]}")
      if old_sol:
         for j,sid in enumerate(othersols):
            axs[i].plot([0,len(sol[name])-1],np.ones(2)*othersols[sid][name],'--',color=color[2+j],label=sid)
            axs[i].plot(np.ones(2)*j/len(othersols)*(len(sol[name])-1),
                        np.array([-1, 1])*otherstds[sid][name]+othersols[sid][name],color=color[2+j])
   # axs[i].set_xlim(0,len(suffixes)-1)

if not rel and old_sol:
   handles, labels = axs[0].get_legend_handles_labels()
   fig.legend(handles, labels, loc='upper center', ncols = 4,
              bbox_to_anchor=(0.5, 1.1))   # move slightly above the subplots)
   
plt.tight_layout()
fig_name = f"sol_{id}_convergence"
plt.savefig(f"{fig_name}.png", bbox_inches='tight')

endInit = time.time()
print(f"Finished after {str(endInit-startInit)}s")