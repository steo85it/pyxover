#!/usr/bin/env python3
# ----------------------------------
# Plot solutions for geodesic parameters of Mercury
# Compare with other sols
# ----------------------------------
# Author: Stefano Bertone
# Created: 5-Dec-2019
#

import numpy as np
import matplotlib.pyplot as plt
# if using a Jupyter notebook, include:
import pandas as pd
from scipy.sparse import diags
from scipy.linalg import pinv, pinvh

from Amat import Amat
from prOpt import tmpdir, outdir, sol4_glo
from util import as2deg, mergsum


def get_apost_parerr(amat):
    ATP = amat.spA.T * amat.weights
    m_0 = amat.resid_wrmse
    PA = amat.weights * amat.spA
    ell = diags(np.abs(amat.b))
    posterr = pinvh((ATP * ell * PA).todense(),check_finite=False)
    posterr = np.sqrt(posterr.diagonal())
    m_X = dict(zip(amat.sol4_pars, np.sqrt(np.ravel(m_0 * posterr)) ))
    return m_X


if __name__ == '__main__':

    ind = np.array(['RA', 'DEC', 'PM', 'L','h2'])
    consist_check = True
    add_other_sols = False #True

    # a priori values
    IAU = {'sol': {'RA': 281.0103, 'DEC': 61.4155, 'PM': 6.1385108, 'L': 38.5},
           'std': {'RA': 0.0014, 'DEC': 0.0014, 'PM': 0.0000013, 'L': 1.6}}
    AG = {'sol': {'RA': 281.0082, 'DEC': 61.4164, 'PM': 6.1385054, 'L': 40.},
          'std': {'RA': 9.e-4, 'DEC': 3.e-4, 'PM': 0.0000013, 'L': 8.7}}
    # systematic errors
    SB = {'sol': {'RA': 0., 'DEC': 0., 'PM': 0., 'L': 0.},
          'std': {'RA': 5.8e-4, 'DEC': 4.6e-4, 'PM': 0.0000027, 'L': 0.85, 'h2':0.3}}
    MIX = {'sol': {'RA': 281.00975, 'DEC': 61.4143, 'PM': 6.1385025, 'L': 38.5},
           'std': {'RA': 0.0048, 'DEC': 0.0028, 'PM': 0., 'L': 1.6}}

    # other sols
    cols = np.array(
        ['Stark (2015)', 'Margot (2012)', 'Mazarico (2014)', 'Genova (2019)', 'Verma (2016)'])
    RA = np.array([281.00980, 281.0103, 281.00480, 281.0082, 281.00975])
    err_RA = np.array([8.8e-4, 1.4e-3, 0.0054, 9.e-4, 0.0048])  # 3* as2deg(0.014652)*63])
    DEC = np.array([61.4156, 61.4155, 61.41436, 61.4164, 61.41828])
    err_DEC = np.array([1.6e-3, 1.4e-3, 0.0021, 3.e-4, 0.0028])  # 3* as2deg(0.036279)*5.6, 0])
    PM = np.array([6.13851804, 6.1385025, 6.138511, 6.1385054, 0])
    err_PM = np.array([9.4e-7, 0, 1.15e-6, 0.0000013, 0.0000013])  # 3* (as2deg(0.041573)/365.25)*7.2])
    L = np.array([38.9, 38.5, 0, 40.0, 0])
    err_L = np.array([1.3, 1.6, 0, 8.7, 0])  # 3* 0.028632*15.3])

    unit_transf = [as2deg(1),as2deg(1),as2deg(1)/365.25,1.,1.]

    # exit()

    subfolder = ''
    fignam = 'sols_comp.png'
    vecopts = {}
    # sols = [('BS2_0','IAU'),('BS2_1','IAU'),('BS2_2','IAU'),('BS2_3','IAU'),('BS2_4','IAU'),('BS2_5','IAU'),('BS2_6','IAU'),('BS2_7','IAU')]
    # sols = [('KX3_3','AG'),('KX2_8','IAU'),('KX3_7','AG'),('BS0_6','IAU'),('BS1_7','IAU'),('BS2_7','IAU'),('BS3_7','IAU'),('BS4_7','IAU'),('BS5_7','IAU'),('BS6_7','IAU')] #,('BS7_7','IAU')]
    sols = [('KX1_7','MIX'),('AGTb_7','AG')] #,('KX2b_0','IAU')] #,('KX2_2','IAU'),('KX2_3','IAU'),('KX2_4','IAU'),('KX2_5','IAU'),('KX2_6','IAU'),('KX2_7','IAU'),
            # ('KX3_0','AG'),('KX3_1','AG'),('KX3_2','AG'),('KX3_3','AG'),
            # ('AGTP_1','AG'),('AGTP_2','AG'),('AGTP_3','AG'),('AGTb_0','AG'),('AGTb_1','AG'),('AGTb_2','AG'),('AGTb_3','AG')]

    if add_other_sols:
        my_colors = ['r', 'g', 'b', 'k', 'y', 'm'] #, 'c'] #,
            # 'c', 'tab:brown',
            #          'tab:pink', 'tab:purple']
    else:
        my_colors = ['r', 'g', 'tab:brown','b','c'] #r', 'r', 'r', 'b', 'b', 'b', 'b', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'tab:brown',
                 # 'tab:brown', 'tab:brown', 'tab:brown']

    subexp = '0res_1amp'
    solout = {}
    errout = {}

    for idx,(solnam, ap) in enumerate(sols):
        amat = Amat(vecopts)
        try:
            amat = amat.load(outdir + 'sim/' + subfolder + solnam + '/' + subexp + '/Abmat_sim_' + solnam.split('_')[0] + '_' + str(int(solnam.split('_')[-1]) + 1) + '_' + subexp + '.pkl')
        except:
            amat = amat.load(outdir + 'Abmat/Abmat_sim_' + solnam.split('_')[0] + '_' + str(int(solnam.split('_')[-1]) + 1) + '_' + subexp + '.pkl')


        print("rmse of solution at iter):", amat.resid_wrmse)
        print("vce weights (data,constr,avg_constr):", amat.vce)
        for var in ['A','C','R']:
            sol_orb = {i:amat.sol_dict['sol'][i] for i in amat.sol_dict['sol'].keys() if '_dR/d'+var in i}
            print("Mean values",var,":",np.mean(list(sol_orb.values())),"meters")

        sol_glb = {i:amat.sol_dict['sol']['dR/d' + i] for i in ind if 'dR/d' + i in amat.sol_dict['sol'].keys()}
        if solnam[:3] in ['BS0','BS1','BS2','BS3','BS4']:
            err_glb = {i:np.sqrt(amat.sol_dict['std']['dR/d' + i]) for i in ind if 'dR/d' + i in amat.sol_dict['std'].keys()}
        else:
            err_glb = {i:amat.sol_dict['std']['dR/d' + i] for i in ind if 'dR/d' + i in amat.sol_dict['std'].keys()}

        # print(err_glb)

        # gives more or less the same
        # if idx == len(sols)-1:
        #     std_X = get_apost_parerr(amat=amat)
        #     print("A post err:")
        #     print({x.split('/')[-1][1:]:y for x,y in std_X.items() if x in sol4_glo})

        # convert units to (deg, deg, deg/day, 1, 1)
        sol_glb = {a:unit_transf[i]*b for i,(a,b) in enumerate(sol_glb.items())}
        if solnam[:3] in ['KX1','AGT']:
            err_glb = {a:unit_transf[i]*np.sqrt(b)*3 for i,(a,b) in enumerate(err_glb.items())}
        else:
            err_glb = {a:unit_transf[i]*b*3 for i,(a,b) in enumerate(err_glb.items())}
        # add systematic errors from tests
        if add_other_sols:
            err_glb = mergsum(err_glb,SB['std'])

        # sum to apriori
        tot_glb = mergsum(eval(ap)['sol'],sol_glb)

        solout[solnam+'_'+ap] = tot_glb
        errout[solnam+'_'+ap] = err_glb

    print(solout)
    if add_other_sols:
        for idx,ref in enumerate(cols):
            solout[ref] = {'RA':RA[idx],'DEC':DEC[idx],'PM':PM[idx],'L':L[idx],'h2':0.}
            errout[ref] = {'RA':err_RA[idx],'DEC':err_DEC[idx],'PM':err_PM[idx],'L':err_L[idx],'h2':0.}
    else:
        # add reference sols
        for ref in ['IAU', 'AG', 'MIX']:
            solout[ref] = eval(ref)['sol']
            errout[ref] = eval(ref)['std']

    # print([y['sol'] for x in out for y in x.values()])
    # exit()
    # sols = {a:b for a,b in out}
    sol_df = pd.DataFrame.from_dict(solout)
    err_df = pd.DataFrame.from_dict(errout)

    print(sol_df)

    # exit()

    if consist_check:

        sol_df = sol_df.T
        err_df = err_df.T
        sol_df = sol_df.rename({'IAU': '0IAU', 'AG': '0AG', 'MIX': '0MIX'}, axis='index').sort_index()
        err_df = err_df.rename({'IAU': '0IAU', 'AG': '0AG', 'MIX': '0MIX'}, axis='index').sort_index()

        if add_other_sols:
            sol_df = sol_df.rename({'BS5_7_IAU': 'Bertone (2020)'}, axis='index').sort_index()
            err_df = err_df.rename({'BS5_7_IAU': 'Bertone (2020)'}, axis='index').sort_index()

        # print(sol_df.RA)
        # exit()
        print(sol_df)
        print(err_df)

        #my_colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c', 'tab:brown','tab:pink','tab:orange', 'tab:purple','tab:gray']  # ,
        # 'c', 'tab:brown',
        #          'tab:pink', 'tab:purple']

        fig, axes = plt.subplots(2,2)
        plt.style.use('seaborn-paper')

        print(np.ravel(axes))
        for ax,(par,unit) in zip(np.ravel(axes),[('RADEC','deg'),('PM','deg/day'),('L','as'),('h2','u')]):

            for idx in range(len(sol_df)):
                # if idx not in [4]:

                    if sol_df.index[idx] in ['0IAU','0AG','0MIX']:
                        fmtkey = 'v'
                    else:
                        fmtkey = '.'

                    if (par == 'RADEC') & (sol_df.RA[idx] != 0.) & (sol_df.DEC[idx] != 0.):
                        ax.errorbar(sol_df.RA[idx], sol_df.DEC[idx],
                                xerr=err_df.RA[idx],
                                yerr=err_df.DEC[idx],
                                fmt=fmtkey, color=my_colors[idx], label=sol_df.index[idx])
                        ax.set_xlabel('RA (deg)')
                        ax.xaxis.tick_top()
                        ax.set_ylabel('DEC (deg)')
                        # uncomment for general legend!!!
                        # ax.legend(bbox_to_anchor=(-0.5, -0.), loc='upper center', ncol=1)

                    elif (sol_df[par][idx] != 0.) & (sol_df[par][idx] != 0.):

                        if par in ['PM','h2']:
                            ax.yaxis.tick_right()

                        ax.ticklabel_format(axis="y", useMathText=True)

                        ax.errorbar(idx, sol_df[par][idx],
                                yerr=err_df[par][idx],
                                fmt=fmtkey, color=my_colors[idx], label=sol_df.index[idx])

                        ax.set_ylabel(par+' ('+unit+')')

            if par=='h2':
                ax.errorbar(0, #len(sol_df)-1,
                            1.02,
                            yerr=0.7,
                            fmt=fmtkey, color='tab:orange', label='SG_model')
                ax.legend(loc='lower center', ncol=1)

        # plt.legend(bbox_to_anchor=(0, 1.3), loc='upper left', ncol=3)
        # ax.ticklabel_format(axis="y",useMathText=True)
        fig.tight_layout()

        ax.set_title('')

        plt.savefig(tmpdir+fignam)
        plt.clf()

        # fig, ax = plt.subplots()
        # plt.style.use('seaborn-paper')
        #
        # for idx, color in enumerate(my_colors):
        #     # if idx not in [4]:
        #
        #         if idx in [0,1]:
        #             fmtkey = 'v'
        #         else:
        #             fmtkey = 'o'
        #
        #         ax.errorbar(idx, PM[idx],
        #                         yerr=err_PM[idx],
        #                         fmt=fmtkey, color=color, label=cols[idx])
        # ax.legend()
        # ax.set_xlabel('')
        # ax.set_ylabel('PM (deg/day)')
        # ax.set_title('')
        #
        # plt.savefig(tmpdir+'PM_GSFC.png')
        # plt.clf()
        #
        # fig, ax = plt.subplots()
        # plt.style.use('seaborn-paper')
        #
        # for idx, color in enumerate(my_colors):
        #     # if idx not in [4]:
        #
        #         if idx in [0, 1]:
        #             fmtkey = 'v'
        #         else:
        #             fmtkey = 'o'
        #
        #         ax.errorbar(idx, L[idx],
        #                     yerr=err_L[idx],
        #                     fmt=fmtkey, color=color, label=cols[idx])
        # ax.legend()
        # # ax.ticklabel_format(axis="y", useMathText=True)
        # ax.set_xlabel('')
        # ax.set_ylabel('L (as)')
        # ax.set_title('')
        #
        # plt.savefig(tmpdir + 'L_GSFC.png')
        # plt.clf()
        #
        # fig, ax = plt.subplots()
        # plt.style.use('seaborn-paper')
        #
        # for idx, color in enumerate(my_colors):
        #     # if idx not in [4]:
        #
        #         if idx in [0, 1]:
        #             fmtkey = 'v'
        #             ax.errorbar(idx, h2[idx],
        #                     yerr=err_h2[idx],
        #                     fmt=fmtkey, color=color, label=["inter_model_SG","-"][idx])
        #         else:
        #             fmtkey = 'o'
        #
        #             ax.errorbar(idx, h2[idx],
        #                     yerr=err_h2[idx],
        #                     fmt=fmtkey, color=color, label=cols[idx])
        # ax.legend()
        # # ax.ticklabel_format(axis="y", useMathText=True)
        # ax.set_xlabel('')
        # ax.set_ylabel('h2')
        # ax.set_title('')
        #
        # plt.savefig(tmpdir + 'h2_GSFC.png')
        # plt.clf()

    else:
        # comparison with other groups/techniques
        #########################################
        # SB2019
        # dRA   -3.597977  0.246835
        # dDEC  -0.646556  0.394172
        # dPM   17.169734  0.136825
        # dL    -1.263475  0.109559

        # cols = np.array(['Stark (2015)', 'Margot (2012)', 'Mazarico (2014)', 'Barker (2016)', 'Genova (2019)', 'Verma (2016)', 'Bertone (2019)'])
        # RA = np.array([281.00980,281.0103, 281.00480, 281.0097+1.5e-3, 281.0082, 281.00975, 281.0097 + as2deg(0.865331)])
        # err_RA = np.array([8.8e-4, 1.4e-3, 0.0054, 0, 9.e-4, 0.0048, 3* as2deg(0.014652)*63])
        # DEC = np.array([61.4156,61.4155, 61.41436, 61.4143 + 0, 61.4164, 61.41828, 61.4143 + as2deg(- 0.791120)])
        # err_DEC = np.array([1.6e-3, 1.4e-3, 0.0021, 0, 3.e-4, 0.0028, 3* as2deg(0.036279)*5.6, 0])
        # PM = np.array([6.13851804, 6.1385025, 6.138511, 6.1385025-0.2e-5, 6.1385054, 0, 6.1385025 + as2deg(2.735854)/365.25])
        # err_PM = np.array([9.4e-7, 0, 1.15e-6, 0, 0.0000013, 0, 3* (as2deg(0.041573)/365.25)*7.2])
        # L = np.array([38.9, 38.5, 0, 38.5-0.05, 40.0, 0, 38.5 - 1.058871])
        # err_L = np.array([1.3, 1.6, 0, 0, 8.7, 0, 3* 0.028632*15.3])

        cols = np.array(['Stark (2015)', 'Margot (2012)', 'Mazarico (2014)', 'Genova (2019)', 'Verma (2016)', 'Bertone (2019)'])
        RA = np.array([281.00980,281.0103, 281.00480, 281.0082, 281.00975, 281.0097 + as2deg(0.865331)])
        err_RA = np.array([8.8e-4, 1.4e-3, 0.0054, 9.e-4, 0.0048, 1.e-3]) # 3* as2deg(0.014652)*63])
        DEC = np.array([61.4156,61.4155, 61.41436, 61.4164, 61.41828, 61.4143 + as2deg(- 0.791120)])
        err_DEC = np.array([1.6e-3, 1.4e-3, 0.0021, 3.e-4, 0.0028, 3.6e-4]) # 3* as2deg(0.036279)*5.6, 0])
        PM = np.array([6.13851804, 6.1385025, 6.138511, 6.1385054, 0, 6.1385025 + as2deg(2.735854)/365.25])
        err_PM = np.array([9.4e-7, 0, 1.15e-6, 0, 0.0000013, 7.e-6]) # 3* (as2deg(0.041573)/365.25)*7.2])
        L = np.array([38.9, 38.5, 0, 40.0, 0, 38.5 - 1.058871])
        err_L = np.array([1.3, 1.6, 0, 8.7, 0, 1.]) #3* 0.028632*15.3])

        df = pd.DataFrame(np.vstack([RA,DEC,PM,L]),columns=cols,index=ind)
        print(df.T)

        # x = np.linspace(0, 5.5, 10)
        # y = 10 * np.exp(-x)
        # xerr = np.random.random_sample(10)
        # yerr = np.random.random_sample(10)

        my_colors = ['r', 'g', 'b', 'k', 'y', 'm'] #, 'c'] #,
            # 'c', 'tab:brown',
            #          'tab:pink', 'tab:purple']

        fig, ax = plt.subplots()
        plt.style.use(['seaborn-poster'])

        for idx, color in enumerate(my_colors):
            if idx not in [8]:
                fmtkey = 'o'
                ax.errorbar(RA[idx], DEC[idx],
                            xerr=err_RA[idx],
                            yerr=err_DEC[idx],
                            fmt=fmtkey, color=color, label='') #=cols[idx]) #
        ax.legend()
        # Cassini state
        x = np.linspace(281.004, 281.012, 100)
        plt.plot(x, (61.4164-61.412)/(281.0082-281.011) * (x-281.0082) + 61.4164, '--k', label='Cassini state')
        # ax.ticklabel_format(axis="y",style='sci',useMathText=True)
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('DEC (deg)')
        ax.set_title('')
        # ax.set_xlim(0,1)
        # plt.legend()
        # ax.tick_params(labelsize=10)
        # plt.figure(dpi=400)
        plt.savefig(tmpdir+'RADEC.eps')
        plt.clf()

        # fig, ax = plt.subplots()
        # plt.style.use('seaborn-poster')
        # plt.figure(dpi=1200)
        # ax.ticklabel_format(axis="y", useMathText=True)
        #
        # for idx, color in enumerate(my_colors):
        #     if idx not in [4]:
        #
        #         fmtkey = 'o'
        #         ax.errorbar(idx, PM[idx],
        #                         yerr=err_PM[idx],
        #                         fmt=fmtkey, color=color, label='') #, label=cols[idx])
        # ax.legend()
        # ax.set_xlabel('')
        # ax.set_ylabel('PM (deg/day)')
        # ax.set_title('')
        # # ax.tick_params(labelsize=24)
        #
        # # ax.ticklabel_format(axis="y",style='sci') #,useMathText=True)
        # plt.savefig(tmpdir+'PM.png')
        # plt.clf()

        # fig, ax = plt.subplots()
        # plt.style.use('seaborn-poster')
        # plt.figure(dpi=1200)
        #
        # for idx, color in enumerate(my_colors):
        #    if idx not in [2,4]:
        #        fmtkey = 'o'
        #
        #        ax.errorbar(idx, L[idx],
        #                         yerr=err_L[idx],
        #                         fmt=fmtkey, color=color, label='') #, label=cols[idx])
        # ax.legend()
        # # ax.ticklabel_format(axis="y", useMathText=True)
        # ax.set_xlabel('')
        # ax.set_ylabel('L (as)')
        # ax.set_title('')
        # # ax.tick_params(labelsize=24)
        #
        # plt.savefig(tmpdir+'L.png')
        # plt.clf()