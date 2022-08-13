import glob

import numpy as np
import submitit

import matplotlib.pyplot as plt
from config import XovOpt

from pygeoloc import PyGeoloc
from pyxover import PyXover
import spiceypy as spice
from pyxover.xov_setup import xov
import pandas as pd
from matplotlib import pyplot as plt

from pyxover.xov_setup import xov

XovOpt.display()
XovOpt.set("debug", False)
XovOpt.set("basedir", 'data/')
XovOpt.set("instrument", 'MLA')
XovOpt.set("local", True)
XovOpt.set("parallel", False)
XovOpt.set("partials", False)
XovOpt.set("expopt", 'SMM')
XovOpt.set("monthly_sets", True)

XovOpt.set("new_gtrack", 2)
vecopts = XovOpt.get('vecopts')
vecopts['ALTIM_BORESIGHT'] = [2.2104999983228e-3, 2.9214999977833e-3, 9.9999328924124e-1]
vecopts['SCFRAME'] = 'MSGR_SPACECRAFT'
XovOpt.set('vecopts', vecopts)
XovOpt.set("SpInterp", 0)
XovOpt.set("compute_input_xov", True)

XovOpt.check_consistency()

# executor is the submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder="log_slurm") #, cluster='local')
# executor = submitit.LocalExecutor(folder="log_slurm")
# set timeout in min, and partition for running the job
executor.update_parameters( #slurm_account='j1010',
                           slurm_name="pygeoloc",
                           slurm_cpus_per_task=1,
                           slurm_nodes=1,
                           slurm_time=60*99, # minutes
                           slurm_mem='90G',
                           slurm_array_parallelism=1)

# run full pipeline on a few MLA test data
PyGeoloc.main(['1', '2001/', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack', 'MLASCIRDR', 0, XovOpt.to_dict()])
#PyGeoloc.main(['1', 'pgda11/', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack', 'MLASCIRDR', 0, XovOpt.to_dict()])
#PyXover.main([5, f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/', 'MLASCIRDR', 0, XovOpt.to_dict()])
#exit()
pygeoloc_in = [['1', 'pgda/', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack', 'MLASCIRDR', 0, XovOpt.to_dict()]]

if False:
    spice.furnsh(f'{XovOpt.get("auxdir")}mymeta')
    job = executor.submit(PyGeoloc.main, pygeoloc_in[0])
    print(job.result())

executor.update_parameters( #slurm_account='j1010',
                           slurm_name="pyxover",
                           slurm_mem='8G',
                           slurm_array_parallelism=60)

pyxover_in = [[comb, f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/', 'MLASCIRDR', 0, XovOpt.to_dict()]
               for comb in np.arange(700)]
pyxover_in2 = [[comb, f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/', 'MLASCIRDR', 0, XovOpt.to_dict()]
               for comb in np.arange(700,1378,1)]

if False:
    #job = executor.submit(PyXover.main, pyxover_in[5]) # single job
    jobs = executor.map_array(PyXover.main, pyxover_in)
    for job in jobs:
        print(job.result())
    jobs = executor.map_array(PyXover.main, pyxover_in2)
    for job in jobs:
        print(job.result())
#exit()

xov_ = xov(XovOpt.get('vecopts'))
xov_list = [xov_.load(x) for x in glob.glob(f"data/out/sim/{XovOpt.get('expopt')}_0/0res_1amp/xov/xov_1*_1*.pkl")]
xov_cmb = xov(XovOpt.get('vecopts'))
xov_cmb.combine(xov_list)
#print(xov_cmb.xovers)
print(xov_cmb.xovers.loc[:,['orbA','orbB','dR']])
# xov_ = xov_.load("data/out/sim/PAM_0/0res_1amp/xov/xov_08_08.pkl")
# print(xov_.xovers)
xov_cmb.xovers['dR'] = xov_cmb.xovers['dR'].apply(pd.to_numeric, errors='coerce')

#print(xov_.xovers.columns)
#print(xov_.xovers.loc[:,['orbA','orbB','dR']])
#exit()


print(f"- Removed {len(xov_cmb.xovers.loc[xov_cmb.xovers.dR.abs() >= 2000])}/{len(xov_cmb.xovers)} xovers >2000 m.")
xov_cmb.xovers = xov_cmb.xovers.loc[xov_cmb.xovers.dR.abs() < 2000]
xov_cmb.xovers = xov_cmb.xovers.loc[xov_cmb.xovers.LAT <= 70]
print(f"- Got nobs:{len(xov_cmb.xovers)}, mean:{xov_cmb.xovers.dR.mean()} m, median:{xov_cmb.xovers.dR.median()} m, and std:{xov_cmb.xovers.dR.std()} m.")
#print(xov_cmb.xovers.loc[:,['orbA','orbB']])
xov_cmb.xovers.hist('dR', bins=100)
plt.title(f"dR@{XovOpt.get('expopt')} (nxov:{len(xov_cmb.xovers)}, mean/std:{xov_cmb.xovers.dR.mean().round(2)}/{xov_cmb.xovers.dR.std().round(2)} m)")
#plt.xlim(left=-200, right=200)
pltfil = f"data/out/sim/{XovOpt.get('expopt')}_0/0res_1amp/xov/histo_w_lowlat.png"
plt.savefig(pltfil)
print(f"- Xovers residuals plot saved to {pltfil}.")

exit()

#print(f"- Removed {len(xov_cmb.xovers.loc[xov_cmb.xovers.dR.abs() >= 100])}/{len(xov_cmb.xovers)} xovers >100 m.")
#xov_cmb.xovers = xov_cmb.xovers.loc[xov_cmb.xovers.dR.abs() < 100]
#print(f"- Got mean:{xov_cmb.xovers.dR.mean()} m, median:{xov_cmb.xovers.dR.median()} m, and std:{xov_cmb.xovers.dR.std()} m.")
#print(xov_cmb.xovers.loc[:,['orbA','orbB']])
#xov_cmb.xovers.hist('dR')
#plt.xlim(left=-200, right=200)
#pltfil = f"data/out/sim/{XovOpt.get('expopt')}_0/0res_1amp/xov/histo_w.png"
#plt.savefig(pltfil)
#print(f"- Xovers residuals plot saved to {pltfil}.")
#plt.show()

# generate new template (when needed)
# out.save('mla_pipel_test_out.pkl')

# XovOpt.set("sol4_orb", [])  # '1503250029'] #'1107021838','1210192326','1403281002','1503191143']  #
# XovOpt.set("sol4_orbpar", ['dA','dC','dR']) #,'dRl','dPt'] #,'dA1','dC1','dR1','dA2','dC2','dR2']  #] #
#
# out = AccumXov.main([[f'sim/{XovOpt.get("expopt")}_0/0res_1amp/'], 'sim', 0, XovOpt.to_dict()])
