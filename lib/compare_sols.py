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
    m_X = dict(zip(amat.sol4_pars, np.ravel(m_0 * posterr)))
    return m_X


if __name__ == '__main__':

    ind = np.array(['RA', 'DEC', 'PM', 'L','h2'])
    consist_check = True

    # a priori values
    IAU = {'sol': {'RA': 281.00975, 'DEC': 61.4143, 'PM': 6.1385025, 'L': 38.5},
           'std': {'RA': 0.0048, 'DEC': 0.0028, 'PM': 0.0000013, 'L': 1.6}}
    AG = {'sol': {'RA': 281.0082, 'DEC': 61.4164, 'PM': 6.1385054, 'L': 40.},
          'std': {'RA': 9.e-4, 'DEC': 3.e-4, 'PM': 0., 'L': 8.7/10.}}

    unit_transf = [as2deg(1),as2deg(1),as2deg(1)/365.25,1.,1.]

    # exit()

    subfolder = ''
    fignam = 'sols_BS_iter_ap.png'
    vecopts = {}
    # sols = [('BS0_6','IAU'),('BS1_7','IAU'),('BS2_6','IAU'),('BS3_7','IAU')]
    sols = [('KX2_0','IAU'),('KX2_1','IAU'),('KX2_2','IAU'),('KX2_3','IAU'),('KX2_4','IAU'),('KX2_5','IAU'),('KX2_6','IAU'),('KX2_7','IAU'),
            ('KX3_0','AG'),('KX3_1','AG'),('KX3_2','AG'),('KX3_3','AG'),
            ('AGTP_1','AG'),('AGTP_2','AG'),('AGTP_3','AG'),('AGTb_0','AG'),('AGTb_1','AG'),('AGTb_2','AG'),('AGTb_3','AG')]

    subexp = '0res_1amp'
    solout = {}
    errout = {}

    for idx,(solnam, ap) in enumerate(sols):
        amat = Amat(vecopts)
        amat = amat.load(outdir + 'sim/' + subfolder + solnam + '/' + subexp + '/Abmat_sim_' + solnam.split('_')[0] + '_' + str(int(solnam.split('_')[-1]) + 1) + '_' + subexp + '.pkl')

        print("rmse of solution at iter):", amat.resid_wrmse)
        print("vce weights (data,constr,avg_constr):", amat.vce)
        for var in ['A','C','R']:
            sol_orb = {i:amat.sol_dict['sol'][i] for i in amat.sol_dict['sol'].keys() if '_dR/d'+var in i}
            print("Mean values",var,":",np.mean(list(sol_orb.values())),"meters")

        sol_glb = {i:amat.sol_dict['sol']['dR/d' + i] for i in ind if 'dR/d' + i in amat.sol_dict['sol'].keys()}
        err_glb = {i:amat.sol_dict['std']['dR/d' + i] for i in ind if 'dR/d' + i in amat.sol_dict['std'].keys()}

        # if idx == len(sols)-1:
        #     m_X = get_apost_parerr(amat=amat)
        #     print({x.split('/')[-1][1:]:y for x,y in m_X.items() if x in sol4_glo})

        # convert units to (deg, deg, deg/day, 1, 1)
        sol_glb = {a:unit_transf[i]*b for i,(a,b) in enumerate(sol_glb.items())}
        if solnam[:3]=='KX2':
            err_glb = {a:unit_transf[i]*np.sqrt(b)*3 for i,(a,b) in enumerate(err_glb.items())}
        else:
            err_glb = {a:unit_transf[i]*b*3 for i,(a,b) in enumerate(err_glb.items())}
            
        # sum to apriori
        tot_glb = mergsum(eval(ap)['sol'],sol_glb)

        solout[solnam+'_'+ap] = tot_glb
        errout[solnam+'_'+ap] = err_glb

    print(solout)
    # add reference sols
    for ref in ['IAU','AG']:
        solout[ref] = eval(ref)['sol']
        errout[ref] = eval(ref)['std']

    # print([y['sol'] for x in out for y in x.values()])
    # exit()
    # sols = {a:b for a,b in out}
    sol_df = pd.DataFrame.from_dict(solout)
    err_df = pd.DataFrame.from_dict(errout)

    print(sol_df)

    # exit()

    # internal consistency check
    ############################
    if False:
        cols = np.array([#'Stark (2015)', 'Margot (2012)', 'Mazarico (2014)', 'Barker (2016)', 'Genova (2019)', 'Verma (2016)',
            'ap0', 'ap1','orb0/ap0', 'orb0/ap0/rgh', 'orb0/ap1',
                         'orb1/ap1','orb1/ap0','orb0/ap1/par1','orb0/ap0/FM'])  # orb: 0=KX, 2=AG, 1=AG2/SG, ap: 0=IAU, 1=AG, par:-: orb+glb, 1:orb+pnt+glb



        # first digit is orb, second gives apriori values
        # old
        # test00 = {'sol':{'RA':-0.621,'DEC':- 0.173,'PM':0.152,'L':- 1.152, 'h2':1.983},
        #       'std':{'RA':0.,'DEC':0.,'PM':0.,'L':0., 'h2':0.}}
        # new
        # test00 = {'sol':{'RA':0.36,'DEC':- 0.098,'PM':0.610,'L':- 0.866, 'h2':1.3},
        #       'std':{'RA':0.21,'DEC':0.06,'PM':0.19,'L':0.14, 'h2':0.54}}
        # test00 = {'sol':{'RA':0.008,'DEC':- 0.137,'PM':0.35,'L':- 1.02, 'h2':1.42},
        #       'std':{'RA':0.8,'DEC':0.98,'PM':0.26,'L':0.23, 'h2':1.42}}
        test00 = {'sol':{'RA':2.73,'DEC':1.35,'PM':3.2,'L':0.56, 'h2':3.17},
              'std':{'RA':0.52,'DEC':0.16,'PM':0.34,'L':0.27, 'h2':1.22}}
        test002 = {'sol':{'RA':-1.6,'DEC':-0.04,'PM':2.9,'L':- 0.72, 'h2':2.92},
              'std':{'RA':0.8,'DEC':0.89,'PM':0.24,'L':0.22, 'h2':1.47}}
        # test01 = {'sol':{'RA':4.849,'DEC':-4.007,'PM':1.502,'L':- 0.641, 'h2':3.227},
        #       'std':{'RA':0.454,'DEC':0.097,'PM':0.197,'L':0.205,'h2':0.823}}
        test01 = {'sol':{'RA':4.94,'DEC':-3.37,'PM':1.35,'L':- 0.29, 'h2':1.09},
              'std':{'RA':0.78,'DEC':0.95,'PM':0.26,'L':0.24,'h2':1.4}}
        test11 = {'sol':{'RA':6.53,'DEC':-4.73,'PM':4.50,'L': 0.67, 'h2':1.13},
              'std':{'RA':1.05,'DEC':1.28,'PM':0.38,'L':0.34,'h2':1.95}}
        test011 = {'sol':{'RA':4.74,'DEC':-3.64,'PM':1.13,'L': -0.31, 'h2':2.78},
              'std':{'RA':1.3,'DEC':1.57,'PM':0.61,'L':0.31,'h2':2.16}}
        test10 = {'sol':{'RA':1.32,'DEC':0.35,'PM':1.09,'L': -0.59, 'h2':0.68},
              'std':{'RA':1.19,'DEC':1.48,'PM':0.37,'L':0.35,'h2':2.0}}
        test00f = {'sol':{'RA':-0.94,'DEC':-0.25,'PM':0.99,'L': -0.32, 'h2':1.34},
              'std':{'RA':0.10,'DEC':0.13,'PM':0.036,'L':0.03,'h2':0.24}}
        # test00f = {'sol':{'RA':-0.67,'DEC':-0.29,'PM':1.73,'L': -0.47, 'h2':2.21},
        #       'std':{'RA':0.17,'DEC':0.21,'PM':0.05,'L':0.05,'h2':0.36}}
        # test21 = {'sol':{'RA':281.00975,'DEC':61.4143,'PM':6.1385025,'L':38.5},
        #       'std':{'RA':0.,'DEC':0.,'PM':0.,'L':0.}}
        # test10 = {'sol':{'RA':281.00975,'DEC':61.4143,'PM':6.1385025,'L':38.5},
        #       'std':{'RA':0.,'DEC':0.,'PM':0.,'L':0.}}

############### PREPARE ##################################################

        RA = np.array([#281.00980,281.0103, 281.00480, 281.0097+1.5e-3, 281.0082,
            IAU['sol']['RA'], AG['sol']['RA'], # IAU, AG
            IAU['sol']['RA'] + as2deg(test00['sol']['RA']),
            IAU['sol']['RA'] + as2deg(test002['sol']['RA']),
            AG['sol']['RA'] + as2deg(test01['sol']['RA']),
            AG['sol']['RA'] + as2deg(test11['sol']['RA']),
            IAU['sol']['RA'] + as2deg(test10['sol']['RA']),
            AG['sol']['RA'] + as2deg(test011['sol']['RA']),
            IAU['sol']['RA'] + as2deg(test00f['sol']['RA'])])
        err_RA = np.array([#8.8e-4, 1.4e-3, 0.0054, 0, 9.e-4, 0.0048,
            1* (IAU['std']['RA']), 1* (AG['std']['RA']),
            1* as2deg(test00['std']['RA']),
            1 * as2deg(test002['std']['RA']),
            1* as2deg(test01['std']['RA']),
            1* as2deg(test11['std']['RA']),
            1 * as2deg(test10['std']['RA']),
            1* as2deg(test011['std']['RA']),
            1* as2deg(test00f['std']['RA'])])
        DEC = np.array([#61.4156,61.4155, 61.41436, 61.4143 + 0, 61.4164,
            IAU['sol']['DEC'], AG['sol']['DEC'],
            IAU['sol']['DEC'] + as2deg(test00['sol']['DEC']),
            IAU['sol']['DEC'] + as2deg(test002['sol']['DEC']),
            AG['sol']['DEC'] + as2deg(test01['sol']['DEC']),
            AG['sol']['DEC'] + as2deg(test11['sol']['DEC']),
            IAU['sol']['DEC'] + as2deg(test10['sol']['DEC']),
            AG['sol']['DEC'] + as2deg(test011['sol']['DEC']),
            IAU['sol']['DEC'] + as2deg(test00f['sol']['DEC'])])
        err_DEC = np.array([#1.6e-3, 1.4e-3, 0.0021, 0, 3.e-4, 0.0028,
            1* (IAU['std']['DEC']), 1* (AG['std']['DEC']),
            1* as2deg(test00['std']['DEC']),
            1 * as2deg(test002['std']['DEC']),
            1* as2deg(test01['std']['DEC']),
            1* as2deg(test11['std']['DEC']),
            1 * as2deg(test10['std']['DEC']),
            1* as2deg(test011['std']['DEC']),
            1* as2deg(test00f['std']['DEC'])])
        PM = np.array([#6.13851804, 6.1385025, 6.138511, 6.1385025-0.2e-5, 6.1385054, 0,
            IAU['sol']['PM'], AG['sol']['PM'],
            IAU['sol']['PM'] + as2deg(test00['sol']['PM'])/365.25,
            IAU['sol']['PM'] + as2deg(test002['sol']['PM']) / 365.25,
            AG['sol']['PM'] + as2deg(test01['sol']['PM'])/365.25,
            AG['sol']['PM'] + as2deg(test11['sol']['PM']) / 365.25,
            IAU['sol']['PM'] + as2deg(test10['sol']['PM']) / 365.25,
            AG['sol']['PM'] + as2deg(test011['sol']['PM']) / 365.25,
            IAU['sol']['PM'] + as2deg(test00f['sol']['PM']) / 365.25])
        err_PM = np.array([#9.4e-7, 0, 1.15e-6, 0, 0.0000013, 0,
            (IAU['std']['PM']), (AG['std']['PM']),
            1* as2deg(test00['std']['PM']) / 365.25,
            1 * as2deg(test002['std']['PM']) / 365.25,
            1* as2deg(test01['std']['PM']) / 365.25,
            1* as2deg(test11['std']['PM']) / 365.25,
            1 * as2deg(test10['std']['PM']) / 365.25,
            1* as2deg(test011['std']['PM']) / 365.25,
            1* as2deg(test00f['std']['PM']) / 365.25])
        L = np.array([#38.9, 38.5, 0, 38.5-0.05, 40.0, 0,
            IAU['sol']['L'], AG['sol']['L'],
            IAU['sol']['L'] + test00['sol']['L'],
            IAU['sol']['L'] + test002['sol']['L'],
            AG['sol']['L'] + test01['sol']['L'],
            AG['sol']['L'] + test11['sol']['L'],
            IAU['sol']['L'] + test10['sol']['L'],
            AG['sol']['L'] + test011['sol']['L'],
            IAU['sol']['L'] + test00f['sol']['L']])
        err_L = np.array([#1.3, 1.6, 0, 0, 8.7, 0,
            1* IAU['std']['L'],1* AG['std']['L'],
            1* test00['std']['L'],
            1 * test002['std']['L'],
            1* test01['std']['L'],
            1* test11['std']['L'],
            1 * test10['std']['L'],
            1* test011['std']['L'],
            1* test00f['std']['L']])
        h2 = np.array([#38.9, 38.5, 0, 38.5-0.05, 40.0, 0,
            1.02,0.,
            0. + test00['sol']['h2'],
            0. + test002['sol']['h2'],
            0. + test01['sol']['h2'],
            0. + test11['sol']['h2'],
            0. + test10['sol']['h2'],
            0. + test011['sol']['h2'],
            0. + test00f['sol']['h2']])
        err_h2 = np.array([#1.3, 1.6, 0, 0, 8.7, 0,
            0.7,0.,
            1* test00['std']['h2'],
            1 * test002['std']['h2'],
            1* test01['std']['h2'],
            1* test11['std']['h2'],
            1 * test10['std']['h2'],
            1* test011['std']['h2'],
            1* test00f['std']['h2']])

        df = pd.DataFrame(np.vstack([RA, DEC, PM, L, h2]), columns=cols, index=ind)
        print(df.T)

        # exit()

        # print(df.T.iloc[[2, 3, 5]].max(axis=0))
        # print(df.T.iloc[[2, 3, 5]].min(axis=0))
        # print(df.T.iloc[[2, 3, 5]].max(axis=0) - df.T.iloc[[2, 3, 5]].min(axis=0))
        # exit()

        # x = np.linspace(0, 5.5, 10)
        # y = 10 * np.exp(-x)
        # xerr = np.random.random_sample(10)
        # yerr = np.random.random_sample(10)

    elif consist_check:

        sol_df = sol_df.T
        err_df = err_df.T
        sol_df = sol_df.rename({'IAU': '0IAU', 'AG': '0AG'}, axis='index').sort_index()
        err_df = err_df.rename({'IAU': '0IAU', 'AG': '0AG'}, axis='index').sort_index()

        # print(sol_df.RA)
        # exit()
        print(sol_df)
        print(err_df)

        my_colors = ['r', 'g', 'r','r','r','b','b','b','b','g','g', 'g','g', 'g','g', 'g','g','tab:brown','tab:brown','tab:brown','tab:brown']
        #my_colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c', 'tab:brown','tab:pink']  # ,
        # 'c', 'tab:brown',
        #          'tab:pink', 'tab:purple']

        fig, axes = plt.subplots(2,2)
        plt.style.use('seaborn-paper')

        print(np.ravel(axes))
        for ax,(par,unit) in zip(np.ravel(axes),[('RADEC','deg'),('PM','deg/day'),('L','as'),('h2','')]):

            for idx in range(len(sol_df)):
                #if idx not in [4]:

                    if sol_df.index[idx] in ['IAU','AG']:
                        fmtkey = 'v'
                    else:
                        fmtkey = '.'

                    if par == 'RADEC':
                        ax.errorbar(sol_df.RA[idx], sol_df.DEC[idx],
                                xerr=err_df.RA[idx],
                                yerr=err_df.DEC[idx],
                                fmt=fmtkey, color=my_colors[idx], label=sol_df.index[idx])
                        ax.set_xlabel('RA (deg)')
                        ax.xaxis.tick_top()
                        ax.set_ylabel('DEC (deg)')
                    else:
                        ax.ticklabel_format(axis="y", useMathText=True)

                        ax.errorbar(idx, sol_df[par][idx],
                                yerr=err_df[par][idx],
                                fmt=fmtkey, color=my_colors[idx], label=sol_df.index[idx])

                        ax.set_ylabel(par+' ('+unit+')')

        # ax.legend()
        # ax.ticklabel_format(axis="y",useMathText=True)

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
