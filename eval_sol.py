#!/usr/bin/env python3
# ----------------------------------
# Analyze, screen and plot AccumXov results
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-Oct-2018
#

import AccumXov as xovacc
import numpy as np
import itertools as itert
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from AccumXov import plt_geo_dR
from ground_track import gtrack
from xov_setup import xov
from Amat import Amat
from prOpt import outdir, tmpdir, local

def xovnum_plot():

    vecopts = {}

    if True:
        #xov_cmb = xov.load_combine('/home/sberton2/Works/NASA/Mercury_tides/out/sim_mlatimes/0res_1amp/',vecopts)
        #xov_sim = xovacc.load_combine('/home/sberton2/Works/NASA/Mercury_tides/out/sim/mlatimes/1301_sph_0/0res_1amp/', vecopts)
        # xov_real = xovacc.load_combine('/home/sberton2/Works/NASA/Mercury_tides/out/sim/mlatimes/1301/0res_1amp/', vecopts)
        # xov_real = xovacc.load_combine('/home/sberton2/Works/NASA/Mercury_tides/out/sim/mlatimes/full/0res_1amp/', vecopts)
        #xov_real = xovacc.load_combine('/home/sberton2/Works/NASA/Mercury_tides/out/real/xov/', vecopts,'real')

        mean_dR, std_dR = xov_sim.remove_outliers('dR')
        mean_dR, std_dR = xov_real.remove_outliers('dR')

        mon = [i for i in range(1, 13)]
        yea = [i for i in range(11, 16)]
        monyea = [str("{:02}".format(y)) + str("{:02}".format(x)) for x in mon for y in yea]

        monyea2 = itert.combinations_with_replacement(monyea, 2)

        xov_sim.xovers['orbA'] = xov_sim.xovers.orbA.str[:8]
        xov_sim.xovers['orbB'] = xov_sim.xovers.orbB.str[:8]
        xov_real.xovers['orbA'] = xov_real.xovers.orbA.str[:8]
        xov_real.xovers['orbB'] = xov_real.xovers.orbB.str[:8]

        merged_Frame = pd.merge(xov_sim.xovers, xov_real.xovers, on=['orbA','orbB'], how='inner')
        print(merged_Frame.columns)
        print(merged_Frame.loc[:,['x0_x','x0_y','y0_x','y0_y','R_A_x','R_A_y','R_B_x','R_B_y','dR/dA_A','dR/dA0_A']]) #'dR/dA0_A','dR/dA_A','dR/dL_x','dR/dL_y']])
        print(merged_Frame.loc[:,['dR_x','dR_y']].max())
        exit()

        print('xov_real')
        print(xov_real.xovers[xov_real.xovers['orbA'].str.contains("130102")]
                             [xov_real.xovers['orbB'].str.contains("130103")]
              .loc[:,['x0','y0','orbA','orbB','dR']].iloc[:10])
        print('xov_sim')
        print(xov_sim.xovers[xov_sim.xovers['orbA'].str.contains("130102")]
                            [xov_sim.xovers['orbB'].str.contains("130103")]
              .loc[:,['x0','y0','orbA','orbB','dR']].iloc[:10])

        exit()
        print(xov_real.xovers.columns)

        for epo in [orb[:8] for orb in xov_sim.xovers.orbA.unique()]:

            dx0 = (xov_real.xovers[xov_real.xovers['orbA'].str.contains(epo)].reset_index().x0 - xov_sim.xovers[
                xov_sim.xovers['orbA'].str.contains(epo)].reset_index().x0).abs()
            dy0 = (xov_real.xovers[xov_real.xovers['orbA'].str.contains(epo)].reset_index().y0 - xov_sim.xovers[
                xov_sim.xovers['orbA'].str.contains(epo)].reset_index().y0).abs()
            a = xov_real.xovers[xov_real.xovers['orbA'].str.contains(epo)].reset_index().filter(regex='^dist_.*$')[:]
            b = xov_sim.xovers[xov_sim.xovers['orbA'].str.contains(epo)].reset_index().filter(regex='^dist_.*$')[:]
            # print(a)
            # print(b)
            print(epo,'distMax.mean',(a-b).max().values.max())
            print(epo,'xy0.mean', dx0.mean(), dy0.mean())
            print(epo,'xy0.max', dx0.max(), dy0.max())
            print(epo,'xy0.min', dx0.min(), dy0.min())

            # print(epo, 'xy0', (dx0,dy0).norm())

        exit()

        df_ = pd.DataFrame(xov_cmb.xovers.loc[:, 'orbA'].astype(str).str[0:4], columns=['orbA'])
        df_['orbB'] = xov_cmb.xovers.loc[:, 'orbB'].astype(str).str[0:4]
        df_ = df_.groupby(['orbA', 'orbB']).size().reset_index().rename(columns={0: 'count'})
        print(df_)

        xov_ = xov(vecopts)
        xov_.xovers = df_.copy()
        print(xov_.xovers)
        xov_.save('/home/sberton2/Works/NASA/Mercury_tides/PyXover/tmp/df_pltxovmonth.pkl')
    else:
        xov_ = xov(vecopts)
        xov_ = xov_.load('/home/sberton2/Works/NASA/Mercury_tides/PyXover/tmp/df_pltxovmonth.pkl')
        df_ = xov_.xovers.copy()


    print(df_.max())
    plt_xovrms(df_)

    #
    # _ = [xov_cmb.xovers.loc[(xov_cmb.xovers.orbA.str.contains('^'+a+'.*$')) & (xov_cmb.xovers.orbB.str.contains('^'+b+'.*$')),'xOvID'].count() for a,b in monyea2]
    # print(len(monyea2),len(monyea))


def plt_xovrms(df_):
    # create pivot table, days will be columns, hours will be rows
    piv = pd.pivot_table(df_, values="count", index=["orbA"], columns=["orbB"], fill_value=0)
    # plot pivot table as heatmap using seaborn
    fig, ax1 = plt.subplots(nrows=1)
    ax1 = sns.heatmap(piv, square=False, annot=True)
    # , robust=True,
    #               cbar_kws={'label': 'RMS (m)'}, xticklabels=piv.columns.values.round(2), fmt='.4g')
    ax1.set(xlabel='Topog scale (1st octave, km)',
            ylabel='Topog ampl rms (1st octave, m)')
    plt.savefig(tmpdir+'xover_month.png')
    plt.close()

def analyze_sol(sol):
    from matplotlib.cm import get_cmap

    vecopts = {}
    tmp = Amat(vecopts)
    # tmp = tmp.load('/home/sberton2/Works/NASA/Mercury_tides/out/Abmat_mladata_1.pkl')
    tmp = tmp.load(outdir+'sim/'+sol+'/0res_1amp/Abmat_sim_'+sol[:-1]+str(int(sol[-1])+1)+'_0res_1amp.pkl')

    if tmp.xov.xovers.filter(regex='^dist_.*$').empty==False:
        tmp.xov.xovers['dist_max'] = tmp.xov.xovers.filter(regex='^dist_.*$').max(axis=1)
        print(len(tmp.xov.xovers[tmp.xov.xovers.dist_max > 1]),
              'xovers removed by dist from obs > 1km')
        tmp.xov.xovers = tmp.xov.xovers[tmp.xov.xovers.dist_max < 1]

    # print(tmp.xov.xovers.sort_values(by='dR').max())
    # print(tmp.xov.xovers.sort_values(by='dR').min())
    #
    # print(len(tmp.xov.xovers))
    # exit()

    if True:
        mlacount = tmp.xov.xovers.round(0).groupby(['LON','LAT']).size().rename('count').reset_index()
        print(mlacount.sort_values(['LON']))

        empty_geomap_df = pd.DataFrame(0, index=np.arange(0, 91),
                            columns = np.arange(-180, 181))
        print(empty_geomap_df)

        fig, ax1 = plt.subplots(nrows=1)
        # ax0.errorbar(range(len(dR_avg)),dR_avg, yerr=dR_std, fmt='-o')
        # ax0.set(xlabel='Exp', ylabel='dR_avg (m)')
        # ax1 = sns.heatmap(mlacount, square=False, annot=True, robust=True)
        # cmap = sns.palplot(sns.light_palette("green"), as_cmap=True) #sns.diverging_palette(220, 10, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        piv = pd.pivot_table(mlacount, values="count", index=["LAT"], columns=["LON"], fill_value=0)
        # plot pivot table as heatmap using seaborn
        piv = (piv+empty_geomap_df).fillna(0)
        print(piv)
        sns.heatmap(piv, xticklabels=10, yticklabels=10)
        plt.tight_layout()
        ax1.invert_yaxis()
        #         ylabel='Topog ampl rms (1st octave, m)')
        fig.savefig(tmpdir+'mla_count_'+sol+'.png')
        plt.clf()
        plt.close()

        orb_sol, glb_sol = xovacc.analyze_sol(tmp, tmp.xov)
        xovacc.print_sol(orb_sol, glb_sol, tmp.xov, tmp)

        # trackA = gtrack(vecopts)
        # trackA = trackA.load('/home/sberton2/Works/NASA/Mercury_tides/out/sim/1301_'+sol+'_0/0res_1amp/gtrack_13/gtrack_' + '1301142347' + '.pkl')
        # print(trackA.pert_cloop)
        # exit()

        cols = orb_sol.filter(regex='sol.*$', axis=1).columns
        print(list(cols))
        orb_sol[cols] = orb_sol.filter(regex='sol.*$', axis=1).apply(pd.to_numeric, errors='ignore')

        fig, ax = plt.subplots(nrows=1)
        name = "Accent"
        cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
        colors = cmap.colors  # type: list
        for idx, col in enumerate(cols):
            ax.set_prop_cycle(color=colors)
            orb_sol.reset_index().plot(kind="scatter", x="index", y=col, color=colors[idx], label=col, ax=ax)

        ax.set_xticks(orb_sol.index.values)
        ax.locator_params(nbins=10, axis='x')
        ax.set_ylabel('sol (m)')
        # ax.set_ylim(-300,300)
        plt.savefig(tmpdir+'orbcorr_tseries_'+sol+'.png')
        plt.close()

        print(tmp.xov.xovers.dR)
        mean_dR, std_dR = tmp.xov.remove_outliers('dR')
        xovacc.plt_histo_dR(sol, mean_dR, std_dR,
                            tmp.xov.xovers)  # [tmp.xov.xovers.orbA.str.contains('14', regex=False)])

        xovacc.plt_geo_dR(empty_geomap_df, sol, tmp.xov)

    if False:

        tmp.xov.xovers['dR/dL'] = tmp.xov.xovers.loc[:,['dR/dL']].abs()
        tmp.xov.xovers['dR/dh2'] = tmp.xov.xovers.loc[:,['dR/dh2']].abs()

        mladRdL = tmp.xov.xovers.round({'LON': 0, 'LAT': 0, 'dR': 3}).groupby(['LON', 'LAT']).median().reset_index()
        print(mladRdL)

        fig, ax1 = plt.subplots(nrows=1)
        # ax0.errorbar(range(len(dR_avg)),dR_avg, yerr=dR_std, fmt='-o')
        # ax0.set(xlabel='Exp', ylabel='dR_avg (m)')
        # ax1 = sns.heatmap(mlacount, square=False, annot=True, robust=True)
        # cmap = sns.palplot(sns.light_palette("green"), as_cmap=True) #sns.diverging_palette(220, 10, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        piv = pd.pivot_table(mladRdL, values="dR/dL", index=["LAT"], columns=["LON"], fill_value=0)
        # plot pivot table as heatmap using seaborn
        piv = (piv+empty_geomap_df).fillna(0)
        print(piv)
        # exit()
        sns.heatmap(piv, xticklabels=10, yticklabels=10, vmax=5)
        plt.tight_layout()
        ax1.invert_yaxis()
        #         ylabel='Topog ampl rms (1st octave, m)')
        fig.savefig(tmpdir+'mla_dL_'+sol+'.png')
        plt.clf()
        plt.close()

        fig, ax1 = plt.subplots(nrows=1)
        # ax0.errorbar(range(len(dR_avg)),dR_avg, yerr=dR_std, fmt='-o')
        # ax0.set(xlabel='Exp', ylabel='dR_avg (m)')
        # ax1 = sns.heatmap(mlacount, square=False, annot=True, robust=True)
        # cmap = sns.palplot(sns.light_palette("green"), as_cmap=True) #sns.diverging_palette(220, 10, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        piv = pd.pivot_table(mladRdL, values="dR/dh2", index=["LAT"], columns=["LON"], fill_value=0)
        # plot pivot table as heatmap using seaborn
        piv = (piv+empty_geomap_df).fillna(0)
        print(piv)
        # exit()
        sns.heatmap(piv, xticklabels=10, yticklabels=10, vmax=0.8)
        plt.tight_layout()
        ax1.invert_yaxis()
        #         ylabel='Topog ampl rms (1st octave, m)')
        fig.savefig(tmpdir+'mla_dh2_'+sol+'.png')
        plt.clf()
        plt.close()

if __name__ == '__main__':

    analyze_sol(sol='sph_0')
    # xovnum_plot()
