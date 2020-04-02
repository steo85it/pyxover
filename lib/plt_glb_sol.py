#!/usr/bin/env python3
# ----------------------------------
# Plot solutions for geodesic parameters of Mercury
# Compare btw different setups in sim environment
# ----------------------------------
# Author: Stefano Bertone
# Created: 5-Dec-2019
#
import glob

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
# if using a Jupyter notebook, include:
import pandas as pd
import scipy.linalg as la

# from prOpt import tmpdir
# from util import as2deg

# Glb => Glb
#             sol       std       err
# dRA  -10.856200  0.009906  0.056200
# dDEC -13.838201  0.023064  0.038201
# dPM  -62.832527  0.029964  0.167473
# dL    -3.088999  0.020943  0.148944
# Glb+orb+pnt => Glb
#             sol       std       err
# dRA      -11.773871  0.015301  0.973871
# dDEC     -13.608818  0.033968  0.191182
# dPM      -62.680869  0.044108  0.319131
# dL        -2.720072  0.033820  0.517872
# Glb+orb+pnt => Glb+orb
#            sol           std         err
# dRA  -11.933522  0.166746   1.13
# dDEC -13.988870  0.253585   0.18
# dPM  -63.949461  0.095495    0.94
# dL    -3.484789  0.076176    0.25

# Glb+orb+pnt => Glb+orb+pnt
#             sol       std       err
# dRA   10.437802  0.205824  0.362198
# dDEC  14.208006  0.319789  0.408006
# dPM   61.023063  0.104240  1.976937
# dL     3.179636  0.081248  0.058307
# dh2   -0.084235  0.085218  0.084235
from scipy.sparse import csr_matrix, diags

from Amat import Amat
from prOpt import tmpdir, outdir, vecopts
from util import deg2as


def other_groups():

    return

if __name__ == '__main__':

    exp = 'tp4'
    iter = 1

    pert_glb_init = {'dRA':[3.*5., 0.000, 0.000], 'dDEC':[3.*5., 0.000, 0.000],'dPM':[0, 3.*3., 0.000],
                                      'dL':3.*deg2as(1.5*0.03)*np.linalg.norm([0.00993822,-0.00104581,-0.00010280,-0.00002364,-0.00000532]),
                                      'dh2':-1.}
    pert_glb_init = pd.Series(pert_glb_init).apply(lambda x: np.linalg.norm(x))

    sol_list = [outdir+'sim/archived/tp4_pertglb_fitglb/tp4_8/3res_20amp/Abmat_sim_tp4_9_3res_20amp.pkl',
                outdir+'sim/archived/tp4_pertall_fitglborb/tp4_8/3res_20amp/Abmat_sim_tp4_9_3res_20amp.pkl',
                outdir+'sim/archived/tp4_pertall_fitall/tp4_8/3res_20amp/Abmat_sim_tp4_9_3res_20amp.pkl']
        #np.sort(glob.glob(outdir+'sim/'+exp+'_'+str(iter)+'/3res_20amp/Abmat_sim_'+exp+'_'+str(iter+1)+'_3res_20amp.pkl'))
    list_exp = []

    for sol in sol_list:
        print("Processing",sol)
        prev = Amat(vecopts)
        solmat = prev.load(sol)
        glbpar = ['dR/d'+x for x in ['RA','DEC','PM','L','h2']]
        solglb = {x:y for x,y in solmat.sol_dict['sol'].items() if x in glbpar}
        solglb = dict(zip([x.split('/')[1] for x in list(solglb.keys())],solglb.values())) #
        solglb = pd.Series(solglb)
        # exit()
        # formal error
        # TODO test impact of projecting out orbit pars on uncertainties
        stdglb = {x:y for x,y in solmat.sol_dict['std'].items() if x in glbpar}
        stdglb = dict(zip([x.split('/')[1] for x in list(stdglb.keys())],stdglb.values())) #
        stdglb = pd.Series(stdglb)
        # residual perturbation
        residerr_glb = solmat.pert_cloop_glo.apply(lambda x: np.linalg.norm(x))
        print("solglb:\n",pert_glb_init)
        print("stdglb:\n",stdglb)
        print("residerr_glb:\n",residerr_glb)
        # print(stdglb/pert_glb_init)
        # print(residerr_glb/pert_glb_init)
        # exit()
        df = pd.DataFrame([stdglb/pert_glb_init,residerr_glb/pert_glb_init], index=['frm','real']).abs()*100.
        list_exp.append(df.T)


        # N = (solmat.spA_penal.transpose()*solmat.weights*solmat.spA_penal).todense() #ATPA
        ATP = solmat.spA.T * solmat.weights
        N = (ATP * solmat.spA).todense()
        # print(np.round(N[-5:,-5:],3))

        # print(solmat.sol4_pars)

        # project arc par on global pars
        ATA = csr_matrix(N[:-5,:-5])
        print(ATA.shape)
        ATB = csr_matrix(N[:-5,-5:])
        BTA = csr_matrix(N[-5:,:-5])
        BTB = csr_matrix(N[-5:,-5:])

        tmp = np.linalg.pinv(ATA.todense())*ATB
        N_proj = BTB - BTA*tmp

        for N_tmp in [N,N_proj]:
            m_0 = solmat.resid_wrmse
            posterr = np.linalg.pinv(N_tmp)
            posterr = np.sqrt(posterr.diagonal())
            m_X = dict(zip(solmat.sol4_pars[-5:],np.ravel(m_0 * posterr[0])[-5:]))
            print(m_X)

            # M = la.cholesky(N_tmp)
            #
            # m_0 = solmat.resid_wrmse
            # ell = diags(np.abs(solmat.b))
            # print(M.shape,ell.shape)
            # posterr = np.linalg.pinv((M.T * ell * M).todense())
            # posterr = np.sqrt(posterr.diagonal())
            # m_X = dict(zip(solmat.sol4_pars,np.ravel(m_0 * posterr[0])))
            # print(m_X)


        # exit()

    # exit()

    dict_grv=['11','73','77'] #,'73','77']
    # grvimp = pd.DataFrame()
    # grverr = pd.DataFrame()
    #
    fig, axes = plt.subplots(4)
    my_colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c', 'tab:brown', 'tab:pink']
    #
    # par = ['RA','DEC','PM','L']
    err = ['frm','eff']

    print(dict(zip(dict_grv,list_exp)))
    num_exp = len(list_exp)
    # plot deviations from mean
    fig, axes = plt.subplots(1,num_exp,figsize=(10,4),sharey=True)
    for idx,ax in enumerate(axes):
        df = list_exp[idx]
        df.index = [x[1:] for x in df.index]
        df.plot.bar(ax=ax,legend=False,logy=True)
        ax.title.set_text(dict_grv[idx])
    # plt.title("Deviations from mean")
    ax.set_ylim(0.1,200.)
    plt.xlabel("Parameter")
    plt.ylabel("% of perturbation")
    # ax.legend()
    plt.legend(err)
    # ax.set_ylim(top=1.5)
    plt.savefig(tmpdir+'simul_errors.png')