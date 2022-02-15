import logging
import os
import unittest
import submitit

from accumxov.accum_opt import AccOpt
from config import XovOpt

from accumxov import AccumXov
from accumxov.Amat import Amat
from pygeoloc import PyGeoloc
from pyxover import PyXover
# from examples.MLA.options import XovOpt.get("vecopts")

# PyTest requires parallel = False

conf = XovOpt()
# update paths and check options
conf.set("basedir", 'data/')
conf.set("instrument", 'MLA')
conf.set("local", False)
conf.set("parallel", False)
conf.set("expopt", 'AA2')

conf.set("new_gtrack", 1)
vecopts = conf.get('vecopts')
vecopts['SCFRAME'] = 'MSGR_SPACECRAFT'
conf.set('vecopts', vecopts)
conf.set("SpInterp", 0)

conf.check_consistency()
AccOpt.check_consistency()

# os.chdir("MLA/")
opt_dict = conf.to_dict()

iternum = 0
pygeoloc_in = [[f'{yy:02d}{mm:02d}', f'SIM_{yy:02d}/KX5/0res_1amp/', f'sim/{XovOpt.get("expopt")}_{iternum}/0res_1amp/gtrack_{yy:02d}', 'MLASCIRDR', iternum, opt_dict] for mm in range(13)[1:] for yy in [8,11,12,13,14,15]]
pyxover_in = [[comb, f'sim/{XovOpt.get("expopt")}_{iternum}/0res_1amp/gtrack_', f'sim/{XovOpt.get("expopt")}_{iternum}/0res_1amp/', 'MLASIMRDR', iternum, opt_dict] for comb in range(21)]

# executor is the submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder="log_slurm")
# set timeout in min, and partition for running the job
executor.update_parameters( #slurm_account='j1010',
                           slurm_name="pygeoloc",
                           slurm_cpus_per_task=1,
                           slurm_nodes=1,
                           slurm_time=60*99, # minutes
                           slurm_mem='8G',
                           slurm_array_parallelism=50)

if True:
    #job = executor.submit(PyGeoloc.main, pygeoloc_in[0]) # single job
    jobs = executor.map_array(PyGeoloc.main, pygeoloc_in)
    for job in jobs:
        print(job.result())

# run PyXover
if True:
    conf.set("parallel", True)
    executor.update_parameters(slurm_name="pyxover",
                               slurm_cpus_per_task=10,
                               slurm_mem='90G',
                               slurm_array_parallelism=5)

    #PyXover.main(pyxover_in[12])
    jobs = executor.map_array(PyXover.main, pyxover_in)
    for job in jobs:
        print(job.result())

if False:
    out = AccumXov.main([[f'sim/{XovOpt.get("expopt")}_{iternum}/0res_1amp/'], 'sim', iternum])
