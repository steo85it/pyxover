import glob

import numpy as np
import submitit

import matplotlib.pyplot as plt
from config import XovOpt

from pygeoloc import PyGeoloc
from pyxover import PyXover
import spiceypy as spice

from src.pyxover.xov_setup import xov

XovOpt.display()
XovOpt.set("debug", False)
XovOpt.set("basedir", 'data/')
XovOpt.set("instrument", 'MLA')
XovOpt.set("local", True)
XovOpt.set("parallel", False)
XovOpt.set("partials", False)
XovOpt.set("expopt", 'PAM')
XovOpt.set("monthly_sets", True)

XovOpt.set("new_gtrack", 2)
vecopts = XovOpt.get('vecopts')
vecopts['ALTIM_BORESIGHT'] = [2.2104999983228e-3, 2.9214999977833e-3, 9.9999328924124e-1]
vecopts['SCFRAME'] = 'MSGR_SPACECRAFT'
XovOpt.set('vecopts', vecopts)
XovOpt.set("SpInterp", 0)
XovOpt.set("compute_input_xov", True)

XovOpt.check_consistency()

spice.furnsh(f'{XovOpt.get("auxdir")}mymeta')

# executor is the submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder="log_slurm", cluster='local')
# executor = submitit.LocalExecutor(folder="log_slurm")
# set timeout in min, and partition for running the job
executor.update_parameters( #slurm_account='j1010',
                           slurm_name="pyxover",
                           slurm_cpus_per_task=1,
                           slurm_nodes=1,
                           slurm_time=60*99, # minutes
                           slurm_mem='8G',
                           slurm_array_parallelism=5)

# run full pipeline on a few MLA test data
# PyGeoloc.main(['1', '2101/', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack', 'MLASCIRDR', 0, XovOpt.to_dict()])
# PyGeoloc.main(['1', 'pgda/', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack', 'MLASCIRDR', 0, XovOpt.to_dict()])
pyxover_in = [[comb, f'sim/{XovOpt.get("expopt")}_0/0res_1amp/gtrack', f'sim/{XovOpt.get("expopt")}_0/0res_1amp/', 'MLASIMRDR', 0, XovOpt.to_dict()]
              for comb in np.arange(162,167)]

if False:
    # job = executor.submit(PyXover.main, pyxover_in[0]) # single job
    jobs = executor.map_array(PyXover.main, pyxover_in)
    for job in jobs:
        print(job.result())
# exit()

xov_ = xov(XovOpt.get('vecopts'))
xov_list = [xov_.load(x) for x in glob.glob("data/out/sim/PAM_0/0res_1amp/xov/xov_1*_1*.pkl")]
xov_cmb = xov(XovOpt.get('vecopts'))
xov_cmb.combine(xov_list)
print(xov_cmb.xovers)
print(xov_cmb.xovers.loc[:,['orbA','orbB','dR']])
# xov_ = xov_.load("data/out/sim/PAM_0/0res_1amp/xov/xov_08_08.pkl")
# print(xov_.xovers)
# print(xov_.xovers.loc[:,['orbA','orbB','dR']])
# exit()
print(xov_cmb.xovers.loc[:,['orbA','orbB']])
xov_cmb.xovers.hist('dR')
plt.show()

# generate new template (when needed)
# out.save('mla_pipel_test_out.pkl')

# XovOpt.set("sol4_orb", [])  # '1503250029'] #'1107021838','1210192326','1403281002','1503191143']  #
# XovOpt.set("sol4_orbpar", ['dA','dC','dR']) #,'dRl','dPt'] #,'dA1','dC1','dR1','dA2','dC2','dR2']  #] #
#
# out = AccumXov.main([[f'sim/{XovOpt.get("expopt")}_0/0res_1amp/'], 'sim', 0, XovOpt.to_dict()])
