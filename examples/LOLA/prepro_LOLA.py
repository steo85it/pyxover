#!/usr/bin/env python3
# ----------------------------------
# Preparing LOLA data for processing in PyAltsim (create one dir for each boresight)
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-Oct-2018
#

import os
import shutil

import numpy as np
import sys
import time

from prOpt import local, instr, inpdir

def prepro_LOLA(args, borefil, grdfil):

    print(inpdir+args)
    if local:
       basedir = 'data/' #'/home/sberton2/Works/NASA/LOLA/aux/'+args+'/'
       indir = inpdir+args+'/'
       print(basedir+'_boresights_LOLA_ch12345_'+borefil+'_laser2_fov.inc')
       bores = np.loadtxt(basedir+'_boresights_LOLA_ch12345_'+borefil+'_laser2_fov.inc')
       bores = np.vsplit(bores, np.shape(bores)[0])
    else:
       indir = inpdir+args+'/'
       basedir = '/att/nobackup/sberton2/LOLA/aux/'+args+'/'
       bores = np.loadtxt('/att/nobackup/sberton2/LOLA/aux/_boresights_LOLA_ch12345_'+borefil+'_laser2_fov.inc')
       bores = np.vsplit(bores, np.shape(bores)[0])

    rngs = np.loadtxt(indir+'boresight_range_slewcheck.rng')
    rngs = np.hsplit(rngs, np.shape(rngs)[1])

    posx = np.loadtxt(indir+'boresight_position_slewcheck.x')
    posx = np.hsplit(posx, np.shape(posx)[1])
    posy = np.loadtxt(indir+'boresight_position_slewcheck.y')
    posy = np.hsplit(posy, np.shape(posy)[1])
    posz = np.loadtxt(indir+'boresight_position_slewcheck.z')
    posz = np.hsplit(posz, np.shape(posz)[1])

    if local:
        outdir0_ = f'{basedir}out/{args}/slewcheck_'
    else:
        outdir0_ = basedir+'slewcheck_'

    for i,x in enumerate(rngs):
        outdir_ = outdir0_+str(i)
        if not os.path.exists(outdir_):
            os.makedirs(outdir_, exist_ok=True)
        np.savetxt(outdir_+'/boresight_range_slewcheck_'+args+'_64ppd_bs'+str(i)+'.rng',x)
        np.savetxt(outdir_ + '/boresight_position_slewcheck_'+args+'_64ppd_bs' + str(i) + '.x', posx[i])
        np.savetxt(outdir_ + '/boresight_position_slewcheck_'+args+'_64ppd_bs' + str(i) + '.y', posy[i])
        np.savetxt(outdir_ + '/boresight_position_slewcheck_'+args+'_64ppd_bs' + str(i) + '.z', posz[i])

    for i, x in enumerate(bores):
        outdir_ = outdir0_+str(i)
        np.savetxt(outdir_+'/_boresights_LOLA_ch12345_'+borefil+'_laser2_fov_bs'+str(i)+'.inc',x)
        # shutil.copy(indir+'boresight_time_slewcheck.xyzd',
        #             outdir_)
        os.symlink(indir+'boresight_time_slewcheck.xyzd',
                   outdir_+'/boresight_time_slewcheck.xyzd')

    # copy selected grid to experiment folder
    if local:
        os.symlink(indir+'../'+grdfil+'_SLDEM2015_512PPD.GRD',
                   basedir+"out/args/SLDEM2015_512PPD.GRD")
    else:
        # shutil.copy(indir+'../'+grdfil+'_SLDEM2015_512PPD.GRD',
        #                 basedir+"SLDEM2015_512PPD.GRD")
        os.symlink(indir+'../'+grdfil+'_SLDEM2015_512PPD.GRD',
                   basedir+"SLDEM2015_512PPD.GRD")

    # stop clock and print runtime
    # -----------------------------
    end = time.time()
    print('----- Runtime = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')

if __name__ == '__main__':

    ##############################################
    # launch program and clock
    # -----------------------------
    start = time.time()

    if len(sys.argv) > 3:
        args = sys.argv[1]
        borefil = sys.argv[2]
        grdfil = sys.argv[3]
    else:
        print("Specify dir or test, e.g., python3 prepro_LOLA.py 191530337 night SIL4")

    print("Processing "+args)

    prepro_LOLA(args, borefil, grdfil)