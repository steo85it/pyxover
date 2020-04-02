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

from prOpt import tmpdir
from util import as2deg

if __name__ == '__main__':

    ind = np.array(['RA', 'DEC', 'PM', 'L','h2'])
    consist_check = True

    # internal consistency check
    ############################
    if consist_check:
        cols = np.array([#'Stark (2015)', 'Margot (2012)', 'Mazarico (2014)', 'Barker (2016)', 'Genova (2019)', 'Verma (2016)',
            'ap0', 'ap1','orb0/ap0', 'orb0/ap1',
                         'orb1/ap1'])  # orb: 0=KX, 2=AG, 1=AG2/SG, ap: 0=IAU, 1=AG

        IAU = {'sol':{'RA':281.00975,'DEC':61.4143,'PM':6.1385025,'L':38.5},
              'std':{'RA':0.,'DEC':0.,'PM':0.,'L':0.}}
        AG = {'sol':{'RA':281.0082,'DEC':61.4164,'PM':6.1385054,'L':40.},
              'std':{'RA':0.,'DEC':0.,'PM':0.,'L':0.}}
        # first digit is orb, second gives apriori values
        # old
        # test00 = {'sol':{'RA':-0.621,'DEC':- 0.173,'PM':0.152,'L':- 1.152, 'h2':1.983},
        #       'std':{'RA':0.,'DEC':0.,'PM':0.,'L':0., 'h2':0.}}
        # new
        test00 = {'sol':{'RA':0.36,'DEC':- 0.098,'PM':0.610,'L':- 0.866, 'h2':1.3},
              'std':{'RA':0.21,'DEC':0.06,'PM':0.19,'L':0.14, 'h2':0.54}}
        # test01 = {'sol':{'RA':4.849,'DEC':-4.007,'PM':1.502,'L':- 0.641, 'h2':3.227},
        #       'std':{'RA':0.454,'DEC':0.097,'PM':0.197,'L':0.205,'h2':0.823}}
        test01 = {'sol':{'RA':4.94,'DEC':-3.37,'PM':1.35,'L':- 0.29, 'h2':1.09},
              'std':{'RA':0.454,'DEC':0.097,'PM':0.197,'L':0.205,'h2':0.823}}
        test11 = {'sol':{'RA':6.48,'DEC':-3.88,'PM':3.62,'L': 0.39, 'h2':0.09},
              'std':{'RA':0.58,'DEC':0.15,'PM':0.3,'L':0.305,'h2':0.}}
        # test21 = {'sol':{'RA':281.00975,'DEC':61.4143,'PM':6.1385025,'L':38.5},
        #       'std':{'RA':0.,'DEC':0.,'PM':0.,'L':0.}}
        # test10 = {'sol':{'RA':281.00975,'DEC':61.4143,'PM':6.1385025,'L':38.5},
        #       'std':{'RA':0.,'DEC':0.,'PM':0.,'L':0.}}

        RA = np.array([#281.00980,281.0103, 281.00480, 281.0097+1.5e-3, 281.0082,
            IAU['sol']['RA'], AG['sol']['RA'], # IAU, AG
            IAU['sol']['RA'] + as2deg(test00['sol']['RA']),
            AG['sol']['RA'] + as2deg(test01['sol']['RA']),
            AG['sol']['RA'] + as2deg(test11['sol']['RA'])])
        err_RA = np.array([#8.8e-4, 1.4e-3, 0.0054, 0, 9.e-4, 0.0048,
            0., 0.,
            1* as2deg(test00['std']['RA']),
            1* as2deg(test01['std']['RA']),
            1* as2deg(test11['std']['RA'])])
        DEC = np.array([#61.4156,61.4155, 61.41436, 61.4143 + 0, 61.4164,
            IAU['sol']['DEC'], AG['sol']['DEC'],
            IAU['sol']['DEC'] + as2deg(test00['sol']['DEC']),
            AG['sol']['DEC'] + as2deg(test01['sol']['DEC']),
            AG['sol']['DEC'] + as2deg(test11['sol']['DEC'])])
        err_DEC = np.array([#1.6e-3, 1.4e-3, 0.0021, 0, 3.e-4, 0.0028,
            0., 0.,
            1* as2deg(test00['std']['DEC']),
            1* as2deg(test01['std']['DEC']),
            1* as2deg(test11['std']['DEC'])])
        PM = np.array([#6.13851804, 6.1385025, 6.138511, 6.1385025-0.2e-5, 6.1385054, 0,
            IAU['sol']['PM'], AG['sol']['PM'],
            IAU['sol']['PM'] + as2deg(test00['sol']['PM'])/365.25,
            AG['sol']['PM'] + as2deg(test01['sol']['PM'])/365.25,
            AG['sol']['PM'] + as2deg(test11['sol']['PM']) / 365.25])
        err_PM = np.array([#9.4e-7, 0, 1.15e-6, 0, 0.0000013, 0,
            0.,0.,
            1* as2deg(test00['std']['PM']) / 365.25,
            1* as2deg(test01['std']['PM']) / 365.25,
            1* as2deg(test11['std']['PM']) / 365.25])
        L = np.array([#38.9, 38.5, 0, 38.5-0.05, 40.0, 0,
            IAU['sol']['L'], AG['sol']['L'],
            IAU['sol']['L'] + test00['sol']['L'],
            AG['sol']['L'] + test01['sol']['L'],
            AG['sol']['L'] + test11['sol']['L']])
        err_L = np.array([#1.3, 1.6, 0, 0, 8.7, 0,
            0.,0.,
            1* test00['std']['L'],
            1* test01['std']['L'],
            1* test11['std']['L']])
        h2 = np.array([#38.9, 38.5, 0, 38.5-0.05, 40.0, 0,
            0.,0.,
            0. + test00['sol']['h2'],
            0. + test01['sol']['h2'],
            0. + test11['sol']['h2']])
        err_h2 = np.array([#1.3, 1.6, 0, 0, 8.7, 0,
            0.,0.,
            1* test00['std']['h2'],
            1* test01['std']['h2'],
            1* test11['std']['h2']])

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

        my_colors = ['r', 'g', 'b', 'k', 'y'] #, 'm']  # ,
        # 'c', 'tab:brown',
        #          'tab:pink', 'tab:purple']

        fig, ax = plt.subplots()
        plt.style.use('seaborn-paper')


        for idx, color in enumerate(my_colors):
            # if idx not in [4]:

                if idx in [0,1]:
                    fmtkey = 'v'
                else:
                    fmtkey = 'o'

                ax.errorbar(RA[idx], DEC[idx],
                            xerr=err_RA[idx],
                            yerr=err_DEC[idx],
                            fmt=fmtkey, color=color, label=cols[idx])
        ax.legend()
        # ax.ticklabel_format(axis="y",useMathText=True)
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('DEC (deg)')
        ax.set_title('')

        plt.savefig(tmpdir+'RADEC_GSFC.png')
        plt.clf()

        fig, ax = plt.subplots()
        plt.style.use('seaborn-paper')
        ax.ticklabel_format(axis="y", useMathText=True)

        for idx, color in enumerate(my_colors):
            # if idx not in [4]:

                if idx in [0,1]:
                    fmtkey = 'v'
                else:
                    fmtkey = 'o'

                ax.errorbar(idx, PM[idx],
                                yerr=err_PM[idx],
                                fmt=fmtkey, color=color, label=cols[idx])
        ax.legend()
        ax.set_xlabel('')
        ax.set_ylabel('PM (deg/day)')
        ax.set_title('')

        plt.savefig(tmpdir+'PM_GSFC.png')
        plt.clf()

        fig, ax = plt.subplots()
        plt.style.use('seaborn-paper')

        for idx, color in enumerate(my_colors):
            # if idx not in [4]:

                if idx in [0, 1]:
                    fmtkey = 'v'
                else:
                    fmtkey = 'o'

                ax.errorbar(idx, L[idx],
                            yerr=err_L[idx],
                            fmt=fmtkey, color=color, label=cols[idx])
        ax.legend()
        # ax.ticklabel_format(axis="y", useMathText=True)
        ax.set_xlabel('')
        ax.set_ylabel('L (as)')
        ax.set_title('')

        plt.savefig(tmpdir + 'L_GSFC.png')
        plt.clf()

        fig, ax = plt.subplots()
        plt.style.use('seaborn-paper')

        for idx, color in enumerate(my_colors):
            # if idx not in [4]:

                if idx in [0, 1]:
                    fmtkey = 'v'
                else:
                    fmtkey = 'o'

                ax.errorbar(idx, h2[idx],
                            yerr=err_h2[idx],
                            fmt=fmtkey, color=color, label=cols[idx])
        ax.legend()
        # ax.ticklabel_format(axis="y", useMathText=True)
        ax.set_xlabel('')
        ax.set_ylabel('h2')
        ax.set_title('')

        plt.savefig(tmpdir + 'h2_GSFC.png')
        plt.clf()

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