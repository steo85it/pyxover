import os
import pandas as pd
import subprocess as s

from config import XovOpt
from prepro_LOLA import prepro_LOLA
from pygeoloc import PyGeoloc
from setup_lola import setup_lola

# add to PYTHONPATH before running (should work with setup.py, but it doesn't...)
# export PYTHONPATH=/att/nobackup/sberton2/LOLA/PyXover/src:/att/nobackup/sberton2/LOLA/PyXover:$PYTHONPATH

def launch_slurm(filnamout,phase):

    print(filnamout)
    iostat = 0
    iostat = s.call(['/home/sberton2/launchLISTslurm',filnamout,'PyAltSim_'+str(phase),'1', '99:00:00', '20Gb', '100'])
    if iostat != 0:
        print("*** PyAltSimLOLA (phase "+str(phase)+") failed")
        exit(iostat)


if __name__ == '__main__':

    setup_lola()

    tmpdir = XovOpt.get("tmpdir") #'./' # '/home/sberton2/tmp/'
    filnamin = f'testLOLA.in' # copy LOLA_template.in to tmpdir (see config) and  add your slews
    filnamout = f'loadPyAltSim' # prepro generates this in the same folder

    f = open(tmpdir+filnamin)
    df = pd.read_csv(f,sep='\s+',names=['trk','d/n','target'])

    prefix_prepro = "python3 prepro_LOLA.py "
    prefix_proc = "python3 launch_test.py 0 "
    prefix_postpro = "python3 postpro_LOLA.py "

    # prepro
    print("Pre-processing started...")
    prepro_cols = ['trk','d/n','target']
    strs = df[prepro_cols].values
    f = open(filnamout,'w')
    f.truncate(0) # erase file
    # write file
    for row in range(strs.shape[0]):
        f.write((' ').join(['python3 prepro_LOLA.py', str(strs[row,0]),str(strs[row,1]),str(strs[row,2]), '\n']))
    f.close()
    if XovOpt.get("local"):
        prepro_LOLA(str(strs[row, 0]), str(strs[row, 1]), str(strs[row, 2]))
    else:
        #launch_slurm(filnamout,phase=0)
        print("ok prepro")

    # processing
    print("Processing started...")
    proc_cols = ['trk']
    strs = df[prepro_cols].values
    f = open(filnamout,'w')
    f.truncate(0)
    # write file
    for row in range(strs.shape[0]):
        f.write((' ').join(['python3 lola_interface.py 0', str(strs[row,0]), '1', '\n']))
    f.close()
    # TODO use the more advanced launch_slurm with local option
    launch_slurm(filnamout,phase=1)

    # postpro
    print("Post-processing started...")
    postpro_cols = ['trk']
    strs = df[prepro_cols].values
    f = open(filnamout,'w')
    f.truncate(0)
    # write file
    for row in range(strs.shape[0]):
        f.write((' ').join(['python3 postpro_LOLA.py', str(strs[row,0]), '\n']))
    f.close()
    launch_slurm(filnamout,phase=2)
