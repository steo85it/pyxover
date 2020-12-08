import glob
import os
import time

import pandas as pd

from pyxover.fine_xov import compute_fine_xov
from pyxover.prepro_xov import prepro_mla_xov
from pyxover.project_gtracks import project_mla

from src.accumxov.Amat import Amat
from examples.MLA.options import outdir, vecopts, compute_input_xov, new_xov, import_proj, import_abmat

## MAIN ##
# @profile
def xov_prc_iters_run(outdir_in, xov_iter, cmb, input_xov):
    start = time.time()

    # Exit process if file already exists and no option to recreate
    outpath = outdir + outdir_in + 'xov/xov_' + str(cmb[0]) + '_' + str(cmb[1]) + '.pkl'
    if (new_xov != 2) and (os.path.isfile(outpath)):
        print("Fine xov", outpath," already exists. Stop!")
        return

    # create useful dirs
    if not os.path.exists(outdir + outdir_in + 'xov/'):
        os.mkdir(outdir + outdir_in + 'xov/')
    if not os.path.exists(outdir + outdir_in + 'xov/tmp/'):
        os.mkdir(outdir + outdir_in + 'xov/tmp/')
    if not os.path.exists(outdir + outdir_in + 'xov/tmp/proj'):
        os.mkdir(outdir + outdir_in + 'xov/tmp/proj')

    if xov_iter==0:
        msrm_smpl = 4 # same as used for rough_xovs in PyXover (should be automatic)
    else:
        msrm_smpl = 4  # should be even...

    if msrm_smpl % 2 != 0:
        print("*** ERROR: msrm_smpl not an even number:", msrm_smpl)

    # compute projected mla_data around old xovs
    if not import_proj:

        useful_columns = ['LON', 'LAT', 'xOvID', 'orbA', 'orbB', 'mla_idA', 'mla_idB']

        # depending on available input xov, get xovers location from AbMat or from xov_rough
        if xov_iter > 0 or import_abmat[0]:  # len(input_xov)==0:
            # read old abmat file
            if xov_iter > 0:
                outdir_old = outdir_in.replace('_' + str(xov_iter) + '/', '_' + str(xov_iter - 1) + '/')
                abmat = outdir + outdir_old + 'Abmat*.pkl'
            else: # read a user defined abmat file
                abmat = import_abmat[1]

            # print(outdir_old, outdir_in)
            tmp_Amat = Amat(vecopts)
            # print(outdir + outdir_old + 'Abmat*.pkl')
            tmp = tmp_Amat.load(glob.glob(abmat)[0])
            old_xovs = tmp.xov.xovers[useful_columns]
        else:
            input_xov_path = outdir + outdir_in + 'xov/tmp/xovin_' + str(cmb[0]) + '_' + str(cmb[1]) + '.pkl.gz'
            if not compute_input_xov:
                input_xov = pd.read_pickle(input_xov_path)
                print("Input xovs read from", input_xov_path, ". Done!")
            else:
                # save to file (just in case...)
                input_xov.to_pickle(input_xov_path)
            old_xovs = input_xov[useful_columns]
            old_xovs = old_xovs.drop('xOvID', axis=1).rename_axis('xOvID').reset_index()

        # preprocessing of old xov and mla_data
        mla_proj_df, part_proj_dict = prepro_mla_xov(old_xovs, msrm_smpl, outdir_in, cmb)

        # projection of mla_data around old xovs
        mla_proj_df = project_mla(mla_proj_df, part_proj_dict, outdir_in, cmb)

    # or just retrieve them from file
    else:
        proj_pkl_path = outdir + outdir_in + 'xov/tmp/proj/xov_' + str(cmb[0]) + '_' + str(
            cmb[1]) + '_project.pkl.gz'
        mla_proj_df = pd.read_pickle(proj_pkl_path)
        print("mla_proj_df loaded from", proj_pkl_path, ". Done!!")

    # compute new xovs
    xov_tmp = compute_fine_xov(mla_proj_df, msrm_smpl, outdir_in, cmb)

    # Save to file
    if not os.path.exists(outdir + outdir_in + 'xov/'):
        os.mkdir(outdir + outdir_in + 'xov/')
    xov_tmp.save(outdir + outdir_in + 'xov/xov_' + str(cmb[0]) + '_' + str(
        cmb[1]) + '.pkl')  # one can split the df by trackA and save multiple pkl, one for each trackA if preferred

    end = time.time()

    print('Xov for ' + str(cmb) + ' processed and written to ' + outdir + outdir_in + 'xov/xov_' + str(
        cmb[0]) + '_' + str(
        cmb[1]) + '.pkl @' + time.strftime("%H:%M:%S", time.gmtime()))

    print("Fine xov determination finished after", int(end - start), "sec or ", round((end - start) / 60., 2), " min!")

    return xov_tmp

## MAIN ##
if __name__ == '__main__':
    start = time.time()

    cmb = [12,18]

    xov_tmp = xov_prc_iters_run()

    end = time.time()

    print("Process finished after", int(end - start), "sec or ", round((end - start)/60.,2), " min!")
