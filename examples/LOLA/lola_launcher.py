# import csv
# import numpy as np
import pandas as pd
import subprocess as s

from config import XovOpt
from examples.LOLA.prepro_LOLA import prepro_LOLA
from pygeoloc import PyGeoloc


def launch_slurm(filnamout,phase):

    iostat = 0
    iostat = s.call(['/home/sberton2/launchLISTslurm',filnamout,'PyAltSim_'+str(phase),'1', '99:00:00', '20Gb', '100'])
    if iostat != 0:
        print("*** PyAltSimLOLA (phase "+str(phase)+") failed")
        exit(iostat)


if __name__ == '__main__':

    XovOpt.set("instrument", 'LOLA')
    XovOpt.set("partials", 0)
    XovOpt.set("new_sim", 2)
    XovOpt.set("apply_topo", 1)

    vecopts = {'SCID': '-85',
               'SCNAME': 'LRO',
               'SCFRAME': -85000,
               'INSTID': (0, 0),
               'INSTNAME': ('', ''),
               'PLANETID': '10',
               'PLANETNAME': 'MOON',
               'PLANETRADIUS': 1737.4,
               'PLANETFRAME': 'MOON_ME',
               'OUTPUTTYPE': 1,
               'ALTIM_BORESIGHT': '',
               'INERTIALFRAME': 'J2000',
               'INERTIALCENTER': 'SSB',
               'PARTDER': ''}
    XovOpt.set("vecopts",vecopts)

    XovOpt.set("outdir",'/att/nobackup/sberton2/LOLA/out/')
    XovOpt.set("auxdir",'/att/nobackup/sberton2/LOLA/aux/')
    XovOpt.set("inpdir",'/att/nobackup/dmao1/LOLA/slew_check/')


    tmpdir = './' # '/home/sberton2/tmp/'
    filnamin = 'testLOLA.in'
    filnamout = 'loadPyAltSim'

    f = open(tmpdir+filnamin)
    df = pd.read_csv(f,sep='\s+',names=['trk','d/n','target'])

    prefix_prepro = "python3 prepro_LOLA.py "
    prefix_proc = "python3 launch_test.py 0 "
    prefix_postpro = "python3 postpro_LOLA.py "

    # prepro
    print("Pre-processing started...")
    prepro_cols = ['trk','d/n','target']
    strs = df[prepro_cols].values
    f = open(tmpdir+filnamout,'w')
    f.truncate(0) # erase file
    # write file
    for row in range(strs.shape[0]):
        f.write((' ').join(['python3 prepro_LOLA.py', str(strs[row,0]),str(strs[row,1]),str(strs[row,2]), '\n']))
    f.close()
    if XovOpt.get("local"):
        prepro_LOLA(str(strs[row, 0]), str(strs[row, 1]), str(strs[row, 2]))
    else:
        launch_slurm(filnamout,phase=0)

    # processing
    print("Processing started...")
    proc_cols = ['trk']
    strs = df[prepro_cols].values
    f = open(tmpdir+filnamout,'w')
    f.truncate(0)
    # write file
    for row in range(strs.shape[0]):
        f.write((' ').join(['python3 launch_test.py 0', str(strs[row,0]), '1', '\n']))
    f.close()
    print((' ').join(['python3 launch_test.py 0', str(strs[row,0]), '1', '\n']))
    if XovOpt.get("local"):
        PyGeoloc.main(['1201', 'SIM_12/BS0/0res_1amp/', 'sim/BS0_0/0res_1amp/gtrack_12', 'MLASCIRDR', 0])
        # prepro_LOLA(str(strs[row, 0]), str(strs[row, 1]), str(strs[row, 2]))
    else:
        launch_slurm(filnamout,phase=1)

    # postpro
    print("Post-processing started...")
    postpro_cols = ['trk']
    strs = df[prepro_cols].values
    f = open(tmpdir+filnamout,'w')
    f.truncate(0)
    # write file
    for row in range(strs.shape[0]):
        f.write((' ').join(['python3 postpro_LOLA.py', str(strs[row,0]), '\n']))
    f.close()
    launch_slurm(filnamout,phase=2)
