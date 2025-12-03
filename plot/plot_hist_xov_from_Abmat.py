
#!/usr/bin/env python3
import glob
import os
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

id = "AE2"
iter = 0
suffix = "_nosol"
suffix = "_C"
data_path = "/storage/research/aiub_gravdet/WD_BELA/"
pyout_folder = f"{data_path}pyXover/out/{id}_{iter}"
out_folder = f"{data_path}OBS/"

startInit = time.time()

if True:
   with open(f"{pyout_folder}/Abmat_{id}_{iter}_{iter+1}{suffix}.pkl", "rb") as file:
      import scipy as sp
      Abmat = pickle.load(file)
      xovers = Abmat.xov.xovers
      # sp.sparse.save_npz(f"{pyout_folder}/Abmat_{id}_{iter}_{iter+1}_pen.npz", Abmat.penalty_mat)
      # sp.sparse.save_npz(f"{pyout_folder}/Abmat_{id}_{iter}_{iter+1}_penavg.npz", Abmat.penalty_mat_avg)
else:
   filename = f"{pyout_folder}/xov/xov_1202_1301.pkl"
   # filename = f"{pyout_folder}/xov/xov_2704_2704.pkl"
   # filename = f"{pyout_folder}/old/xov/xov_2704_2704.pkl"
   # for filename in glob.glob(f"{pyout_folder}/xov/xov_*.pkl"):
   with open(filename, "rb") as file:
         xov = pickle.load(file)
         xovers = xov.xovers
         # plt.plot(xov.xovers.dR,xov.xovers.LAT,'b+')
         # xov = Abmat.xov

lat  = xovers['LAT']
print(len(lat))
print(sum(lat>0))

nbins = 100
c1 = '1'
c2 = '2'
pairs = [
    (c1, c1, "MLA/MLA"),
    (c2, c2, "BELA/BELA"),
    (c1, c2, "MLA/BELA"),
]
pairs = [(c2, c2, "BELA/BELA")]
pairs = [(c1, c1, "MLA/MLA")]

# xovers['dR'] = xovers['dR']*Abmat.weights.diagonal()


# 
# outliers = xovers.loc[np.abs(xovers['dR'])>70]
# outliers = xovers.loc[np.abs(xovers['dR'])>2, ['dR','orbA','orbB']]
# print(Counter(outliers[['orbA','orbB']].values.flatten()))

orbA0 = xovers['orbA'].str[0]
orbB0 = xovers['orbB'].str[0]
    
for a, b, label in pairs:
    mask = (orbA0 == a) & (orbB0 == b)
    # mask = (xovers['orbA'] != '1312281727') & (xovers['orbB'] != '1312281727')
    lat_vals = np.abs(xovers.loc[mask, 'LAT'])
    # dR = np.abs(xovers.loc[mask, 'dR'])
    dR = xovers.loc[mask, 'dR']
    dR = dR[np.abs(dR)<100]
    # plt.plot(lat_vals,dR,'+', label=label)
    plt.hist(lat_vals, nbins, alpha=0.5, density=True, label=label, range=[80, 90])
    # plt.hist(dR, nbins, alpha=0.5, density=True, label=label)
    # plt.hist2d(lat_vals, dR, label=label)

plt.legend()

# id = "DA6"
# pyout_folder = f"{data_path}pyXover/out/{id}_{iter}"
# with open(f"{pyout_folder}/xov/xov_2704_2704.pkl", "rb") as file:
#       xov2 = pickle.load(file)
#       xovers2 = xov2.xovers
# xovers2 = xovers2[xovers2['orbA']<=max(xovers['orbA'])]
# xovers2 = xovers2[xovers2['orbB']<=max(xovers['orbB'])]
# print(max(abs((xov.xovers.dR))))
# xov.xovers = xov.xovers[abs(xov.xovers.dR)<5]
# dR = xov.xovers.dR
# xov.xovers = xov.xovers[abs(xov.xovers.dR)>1000]
# dR = dR[abs(dR)<1000]
# print(np.mean(dR))
# int(np.std(dR))
#print(xov.xovers.columns.tolist())
# max_alt = [max(xv.xov.xovers]

# plt.plot(xov.xovers.tA,xov.xovers.dR,'o',markersize=2)
# plt.plot(xov.xovers.tB,xov.xovers.dR,'o',markersize=2)
# plt.plot(np.amax(abs(xov.xovers[['dist_Ap','dist_Am','dist_Bp','dist_Bm']]), axis=1),xov.xovers.dR,'o',markersize=2)
# plt.plot(np.log10(np.abs(xov.xovers.dR)),xov.xovers.LON,'+')
# plt.plot(xov.xovers.dR,xov.xovers.LON,'+')
# plt.plot(xov.xovers.dR,xov.xovers.LAT,'+')
# plt.plot(xov.xovers.LON,xov.xovers.LAT,'+')
# plt.plot(xovers2.R_A,xovers2.LAT,'+')
# plt.ylim(85,90)
# plt.hist(dR, alpha = 0.5)

# plt.xlim(80, 91)
# plt.xlim(-100, 100)
# plt.ylim(1, 1e6)
# plt.xlabel("dR [m]")
# plt.xlabel("$\nu$ [m]")
plt.xlabel("$\phi [°]")
# plt.yscale('log')
# plt.ylabel("LON [°]")
# plt.ylabel("count")
plt.ylabel("Number of crossovers")
fig_name = "hist_lat"
plt.savefig(f"examples/BELA/{fig_name}.png")

endInit = time.time()
print(f"Finished after {str(endInit-startInit)}s")