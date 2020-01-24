#!/bin/sh
cd $NOBACKUP/MLA/tmp
sbatch -W /home/sberton2/projects/Mercury_tides/PyXover_sim/lib/ladata2grd.sh
#wait
#cd /home/sberton2/projects/Mercury_tides/PyXover_sim/lib
#/home/sberton2/launchLISTslurm load altresplot 1 05:30:00 1
#python3 plot_netcdf.py
