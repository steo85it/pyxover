#!/usr/bin/env python3
# ----------------------------------
# Analyze, screen and plot AccumXov results
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-May-2019
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

def print_corrmat(amat,filename):
    # Compute the covariance matrix
    # print(np.linalg.pinv((ref.spA.transpose() * ref.spA).todense()))
    corr = amat.corr_mat()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5,annot=True, fmt='.1f', cbar_kws={"shrink": .5})
    f.savefig(filename)
    plt.close()

def analyze_sol(sol, ref_sol = '', subexp = ''):
    from matplotlib.cm import get_cmap

    vecopts = {}
    tmp = Amat(vecopts)
    tmp = tmp.load(outdir+'sim/'+sol+'/'+subexp+'/Abmat_sim_'+sol.split('_')[0]+'_'+str(int(sol.split('_')[-1])+1)+'_'+subexp+'.pkl')

    if ref_sol != '':
        ref = Amat(vecopts)
        ref = ref.load('/home/sberton2/Works/NASA/Mercury_tides/out/sim/'+ref_sol+'/'+subexp+'/Abmat_sim_'+ref_sol.split('_')[0]+'_'+str(int(ref_sol.split('_')[-1])+1)+'_'+subexp+'.pkl')

        # print(ref.corr_mat())

    if tmp.xov.xovers.filter(regex='^dist_.*$').empty==False:
        tmp.xov.xovers['dist_max'] = tmp.xov.xovers.filter(regex='^dist_.*$').max(axis=1)
        print(len(tmp.xov.xovers[tmp.xov.xovers.dist_max > 1]),
              'xovers removed by dist from obs > 1km')
        tmp.xov.xovers = tmp.xov.xovers[tmp.xov.xovers.dist_max < 1]

    if True:
        mlacount = tmp.xov.xovers.round(0).groupby(['LON','LAT']).size().rename('count').reset_index()
        print(mlacount.sort_values(['LON']))

        empty_geomap_df = pd.DataFrame(0, index=np.arange(0, 91),
                            columns = np.arange(-180, 181))

        fig, ax1 = plt.subplots(nrows=1)
        # ax0.errorbar(range(len(dR_avg)),dR_avg, yerr=dR_std, fmt='-o')
        # ax0.set(xlabel='Exp', ylabel='dR_avg (m)')
        # ax1 = sns.heatmap(mlacount, square=False, annot=True, robust=True)
        # cmap = sns.palplot(sns.light_palette("green"), as_cmap=True) #sns.diverging_palette(220, 10, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        piv = pd.pivot_table(mlacount, values="count", index=["LAT"], columns=["LON"], fill_value=0)
        # plot pivot table as heatmap using seaborn
        piv = (piv+empty_geomap_df).fillna(0)

        sns.heatmap(piv, xticklabels=10, yticklabels=10)
        plt.tight_layout()
        ax1.invert_yaxis()
        #         ylabel='Topog ampl rms (1st octave, m)')
        fig.savefig(tmpdir+'mla_count_'+sol+'_'+subexp+'.png')
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

        name = "Accent"
        cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
        colors = cmap.colors  # type: list

        if pd.Series(['dA', 'dC','dR']).isin(tmp.pert_cloop.columns).any():
            fig, ax = plt.subplots(nrows=1)

            for idx, col in enumerate(cols):
                ax.set_prop_cycle(color=colors)
                orb_sol.reset_index().plot(kind="scatter", x="index", y=col, color=colors[idx], label=col, ax=ax)

            ax.set_xticks(orb_sol.index.values)
            ax.locator_params(nbins=10, axis='x')
            ax.set_ylabel('sol (m)')
            # ax.set_ylim(-300,300)
            plt.savefig(tmpdir+'orbcorr_tseries_'+sol+'.png')
            plt.close()

            # print residuals (original cloop perturbation - latest cumulated solution)
            orbpar_sol = list(set([x.split("_")[0].split("/")[1] for x in tmp.xov.parOrb_xy]))
            tmp.pert_cloop = tmp.pert_cloop[orbpar_sol].dropna()

            if ref_sol != '':
                ref.pert_cloop = ref.pert_cloop[orbpar_sol].dropna()

            # to do this, we would need the full initial perturbed value of parameters (we don't have it...)
            # tmp.pert_cloop.columns = ["sol_dR/" + x for x in tmp.pert_cloop.columns]
            # postfit_res = orb_sol.set_index('orb').apply(pd.to_numeric, errors='ignore',
            #                                              downcast='float') + tmp.pert_cloop
            # postfit_res = postfit_res[["sol_dR/" + x for x in orbpar_sol]].fillna(0)
            print("par recovery avg, std:",tmp.pert_cloop.columns.values)
            print("iter", tmp.pert_cloop.mean(axis=0).values, tmp.pert_cloop.std(axis=0).values)
            print("pre-fit",ref.pert_cloop.mean(axis=0).values, ref.pert_cloop.std(axis=0).values)

            fig, ax = plt.subplots(nrows=1)
            name = "Accent"
            cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
            colors = cmap.colors  # type: list
            # print('postfit_res')
            # print(postfit_res)
            # postfit_res.reset_index().plot(x="orb", color=colors, ax=ax)
            if ref_sol != '':
                ref.pert_cloop.reset_index().plot(x="index", color=colors, style=':', ax=ax)
                # ref.pert_cloop.apply(lambda x: x.abs()).reset_index().plot(x="index", color=colors, style=':', ax=ax)
            tmp.pert_cloop.reset_index().plot(x="index", color=colors, style='-', ax=ax)
            # tmp.pert_cloop.apply(lambda x: x.abs()).reset_index().plot(x="index", color=colors, style='-', ax=ax)
            # ax.set_xticks(orb_sol.index.values)

            ax.set_xlabel('orbit #')
            ax.set_ylabel('sol (m)')
            ax.set_ylim(-200,200)
            plt.savefig(tmpdir+'residuals_tseries_'+sol+'_'+subexp+'.png')
            plt.close()

        print(tmp.xov.xovers.dR)
        mean_dR, std_dR = tmp.xov.remove_outliers('dR')
        xovacc.plt_histo_dR(sol+subexp, mean_dR, std_dR,
                            tmp.xov.xovers)  # [tmp.xov.xovers.orbA.str.contains('14', regex=False)])

        xovacc.plt_geo_dR(empty_geomap_df, sol+subexp, tmp.xov)


    if False:

        # Check dR/dC / dR/dA
        fig, ax = plt.subplots(nrows=1)
        tmp.xov.xovers['dR/dC_dR/dA_A'] = tmp.xov.xovers['dR/dC_A'] / tmp.xov.xovers['dR/dA_A']
        tmp.xov.xovers['dR/dC_dR/dA_B'] = tmp.xov.xovers['dR/dC_B'] / tmp.xov.xovers['dR/dA_B']
        mladRdL = tmp.xov.xovers.round({'LON': 0, 'LAT': 0, 'dR': 3}).groupby(['LAT']).median()
        print(mladRdL[['dR/dC_dR/dA_A','dR/dC_dR/dA_B']])
        mladRdL[['dR/dC_dR/dA_A','dR/dC_dR/dA_B']].reset_index().plot(x="LAT", color=colors, style='-', ax=ax)
        # ax.set_xticks(orb_sol.index.values)

        ax.set_xlabel('LAT')
        ax.set_ylabel('dR/dC / dR/dA')
        ax.set_ylim(-0.1,0.1)
        plt.savefig('/home/sberton2/Works/NASA/Mercury_tides/PyXover/tmp/dRdC_dA_'+sol+'.png')
        plt.close()

        # plot dR/dL and dR/dh2
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

    analyze_sol(sol='d00_0', ref_sol='d00_0', subexp = '0res_1amp')
