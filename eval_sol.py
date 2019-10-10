#!/usr/bin/env python3
# ----------------------------------
# Analyze, screen and plot AccumXov results
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-May-2019
#
import glob
import re

import AccumXov as xovacc
import numpy as np
import itertools as itert
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from AccumXov import plt_geo_dR
from ground_track import gtrack
from xov_setup import xov
from Amat import Amat
from prOpt import outdir, tmpdir, local, pert_cloop_glo, OrbRep, pert_cloop

remove_max_dist = False
remove_3sigma_median = False

def xovnum_plot():

    vecopts = {}

    if True:
        #xov_cmb = xov.load_combine('/home/sberton2/Works/NASA/Mercury_tides/out/sim_mlatimes/0res_1amp/',vecopts)
        xov_sim = xovacc.load_combine('/home/sberton2/Works/NASA/Mercury_tides/out/sim/mlatimes/1301_sph_0/0res_1amp/', vecopts)
        xov_real = xovacc.load_combine('/home/sberton2/Works/NASA/Mercury_tides/out/sim/mlatimes/1301/0res_1amp/', vecopts)
        # xov_real = xovacc.load_combine('/home/sberton2/Works/NASA/Mercury_tides/out/sim/mlatimes/full/0res_1amp/', vecopts)
        #xov_real = xovacc.load_combine('/home/sberton2/Works/NASA/Mercury_tides/out/real/xov/', vecopts,'real')

        mean_dR, std_dR, worst_tracks = xov_sim.remove_outliers('dR',remove_bad=remove_3sigma_median)
        mean_dR, std_dR, worst_tracks = xov_real.remove_outliers('dR',remove_bad=remove_3sigma_median)

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
            a = xov_real.xovers[xov_real.xovers['orbA'].str.contains(epo)].reset_index().filter(regex='^dist_[A,B].*$')[:]
            b = xov_sim.xovers[xov_sim.xovers['orbA'].str.contains(epo)].reset_index().filter(regex='^dist_[A,B].*$')[:]
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
    print(len(corr))
    # mask to select only parameters with corrs > 0.9
    m = (corr.mask(np.eye(len(corr), dtype=bool)).abs() > 0.95).any()
    # exit()
    corr = corr.loc[m,m]
    print(len(corr))
    # print(corr.columns)
    # print(np.sort(corr.columns))
    # print(corr.index)
    corr.sort_index(axis=1, inplace=True)
    corr.sort_index(axis=0, inplace=True)
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5,annot=False, fmt='.1f', cbar_kws={"shrink": .5})
    f.savefig(filename)
    plt.close()

    # nice visualization to test from https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
    # plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)  # Setup a 1x15 grid
    # ax = plt.subplot(plot_grid[:, :-1])  # Use the leftmost 14 columns of the grid for the main plot
    #
    # ax.scatter(
    #     x=x.map(x_to_num),  # Use mapping for x
    #     y=y.map(y_to_num),  # Use mapping for y
    #     s=size * size_scale,  # Vector of square sizes, proportional to size parameter
    #     c=color.apply(value_to_color),  # Vector of square colors, mapped to color palette
    #     marker='s'  # Use square as scatterplot marker
    # )
    # # ...
    #
    # # Add color legend on the right side of the plot
    # ax = plt.subplot(plot_grid[:, -1])  # Use the rightmost column of the plot
    #
    # col_x = [0] * len(palette)  # Fixed x coordinate for the bars
    # bar_y = np.linspace(color_min, color_max, n_colors)  # y coordinates for each of the n_colors bars
    #
    # bar_height = bar_y[1] - bar_y[0]
    # ax.barh(
    #     y=bar_y,
    #     width=[5] * len(palette),  # Make bars 5 units wide
    #     left=col_x,  # Make bars start at 0
    #     height=bar_height,
    #     color=palette,
    #     linewidth=0
    # )
    # ax.set_xlim(1, 2)  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
    # ax.grid(False)  # Hide grid
    # ax.set_facecolor('white')  # Make background white
    # ax.set_xticks([])  # Remove horizontal ticks
    # ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))  # Show vertical ticks for min, middle and max
    # ax.yaxis.tick_right()  # Show vertical ticks on the right

def draw_map(m, scale=0.2):
    from itertools import chain

    # draw a shaded-relief image
    # m.shadedrelief(scale=scale)

    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13),labels=[False,True,True,False])
    lons = m.drawmeridians(np.linspace(-180, 180, 13),labels=[False,True,True,False])

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)

    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')

def rmse(y, y_pred=0):
    return np.sqrt(np.mean(np.square(y - y_pred)))

def analyze_sol(sol, ref_sol = '', subexp = ''):
    from matplotlib.cm import get_cmap

    vecopts = {}
    tmp = Amat(vecopts)
    tmp = tmp.load(outdir+'sim/'+sol+'/'+subexp+'/Abmat_sim_'+sol.split('_')[0]+'_'+str(int(sol.split('_')[-1])+1)+'_'+subexp+'.pkl')

    if ref_sol != '':
        ref = Amat(vecopts)
        ref = ref.load(outdir+'sim/'+ref_sol+'/'+subexp+'/Abmat_sim_'+ref_sol.split('_')[0]+'_'+str(int(ref_sol.split('_')[-1])+1)+'_'+subexp+'.pkl')
        # if correlation matrix wanted (long, only prints >0.95)
        # print(ref.corr_mat())

    if tmp.xov.xovers.filter(regex='^dist_.*$').empty==False:

        add_xov_separation(tmp)
        xovacc.analyze_dist_vs_dR(tmp.xov)

        if remove_max_dist:
            print(len(tmp.xov.xovers[tmp.xov.xovers.dist_max < 0.4]),
                  'xovers removed by dist from obs > 1km')
            tmp.xov.xovers = tmp.xov.xovers[tmp.xov.xovers.dist_max < 0.4]
            tmp.xov.xovers = tmp.xov.xovers[tmp.xov.xovers.dist_min_mean < 1]

    # Remove huge outliers
    mean_dR, std_dR, worst_tracks = tmp.xov.remove_outliers('dR',remove_bad=remove_3sigma_median)
    tmp.xov.xovers['dR_abs'] = tmp.xov.xovers.dR.abs()
    print("Largest dR ( # above 200m", len(tmp.xov.xovers[tmp.xov.xovers.dR_abs > 200])," or ",
          (len(tmp.xov.xovers[tmp.xov.xovers.dR_abs > 200])/len(tmp.xov.xovers)*100.),'%)')
    print(tmp.xov.xovers[['orbA','orbB','dist_max','dist_min_mean','dR_abs']].nlargest(10,'dR_abs'))
    print(tmp.xov.xovers[['orbA','orbB','dist_max','dist_min_mean','dR_abs']].nsmallest(10,'dR_abs'))

    # Recheck distance after cleaning
    xovacc.analyze_dist_vs_dR(tmp.xov)
    _ = tmp.xov.xovers.dR.values ** 2
    print("Total RMS:", np.sqrt(np.mean(_[~np.isnan(_)], axis=0)), len(tmp.xov.xovers.dR.values))

    # Check convergence over iterations
    if True:
        sol_iters = sol.split('_')[:-1][0]
        prev_sols = np.sort(glob.glob(outdir+'sim/'+sol_iters+'_*/'+subexp+'/Abmat_sim_'+sol_iters+'_*_'+subexp+'.pkl'))

        iters_rms = []
        iters_orbcorr = []
        iters_orbcorr_avg = []
        iters_orbcorr_lin = []
        iters_orbcorr_avg_lin = []
        iters_orbres = []
        iters_glocorr = []
        for idx,isol in enumerate(prev_sols):
            prev = Amat(vecopts)
            prev = prev.load(isol)
            add_xov_separation(prev)

            # prev.xov.xovers = prev.xov.xovers[prev.xov.xovers.dist_max < 0.4]
            # prev.xov.xovers = prev.xov.xovers[prev.xov.xovers.dist_min_mean < 1]
            # mean_dR, std_dR, worst_tracks = prev.xov.remove_outliers('dR', remove_bad=remove_3sigma_median)
            _ = prev.xov.xovers.dR.values ** 2
            tst = isol.split('/')[-3].split('_')[1]
            tst_id = isol.split('/')[-3].split('_')[0]+tst.zfill(2)
            iters_rms.append([tst_id, np.sqrt(np.mean(_[~np.isnan(_)], axis=0)), len(_)])

            filter_string_orb = ["/dA","/dC","/dR","/dRl","/dPt"]
            sol_rms = []
            sol_avg = []
            for filt in filter_string_orb:
                filtered_dict = {k:v for (k,v) in prev.sol_dict['sol'].items() if re.search(filt+'['',0]$',k)} #filt in k if k not in ['dR/dRA']}
                filtered_dict = list(filtered_dict.values())
                sol_rms.append(np.sqrt(np.mean(np.array(filtered_dict) ** 2)))
                sol_avg.append(np.mean(np.array(filtered_dict)))

            iters_orbcorr.append(np.hstack([tst_id,sol_rms]))
            iters_orbcorr_avg.append(np.hstack([tst_id,sol_avg]))

            if OrbRep == 'lin':
                filter_string_orb_lin = ["/dA1","/dC1","/dR1"]
                sol_rms = []
                sol_avg = []
                for filt in filter_string_orb_lin:
                    filtered_dict = {k:v for (k,v) in prev.sol_dict['sol'].items() if re.search(filt+'$',k)} #filt in k if k not in ['dR/dRA']}
                    filtered_dict = list(filtered_dict.values())
                    sol_rms.append(np.sqrt(np.mean(np.array(filtered_dict) ** 2)))
                    sol_avg.append(np.mean(np.array(filtered_dict)))

                iters_orbcorr_lin.append(np.hstack([tst_id,sol_rms]))
                iters_orbcorr_avg_lin.append(np.hstack([tst_id,sol_avg]))

            if simulated_data:
                df_ = pd.DataFrame.from_dict(prev.sol_dict).reset_index()
                df_.columns = ['key', 'sol','std']
                df_ = df_.loc[df_.key.isin(filter_string_orb)]
                if len(df_) > 0:
                    df_[['orb', 'par']] = df_['key'].str.split('_', expand=True)
                    df_.drop('key', axis=1, inplace=True)
                        # df_[['orb','par']] = df_[['par','orb']].where(df_['par'] == None, df_[['orb','par']].values)
                    df_ = df_.replace(to_replace='None', value=np.nan).dropna()
                    df_sol = pd.pivot_table(df_, values=['sol','std'], index=['orb'], columns=['par'], aggfunc=np.sum).sol

                    if OrbRep == 'lin':
                        # print("isol",isol)
                        # print(prev.pert_cloop.columns)
                        # print(df_sol.columns.values)
                        regex = re.compile(".*d[A,C,R]0$")
                        df_sol.columns = [x[:-1] if x in list(filter(regex.match, df_sol.columns)) else x for x in df_sol.columns.values ]
                        # if idx == 0:
                        #     df_sol = df_sol.drop(df_sol.filter(regex=".*[A,C,R]1").columns,axis=1)
                        # print(df_sol.columns)
                        # print(prev.pert_cloop)

                    df_sol.columns = prev.pert_cloop.columns
                    _ = prev.pert_cloop
                    _.columns = ['/'+k for k in _.columns]
                    iters_orbres.append((_ ** 2).mean(axis=0)**0.5)
                    # print(iters_orbres)

            filter_string_glo = ["/dRA","/dDEC","/dPM","/dL","/dh2"]
            sol_rms = []
            for filt in filter_string_glo:
                filtered_dict = {k:v for (k,v) in prev.sol_dict['sol'].items() if filt in k}
                filtered_dict = list(filtered_dict.values())
                sol_rms.append(np.sqrt(np.mean(np.array(filtered_dict) ** 2)))

            iters_glocorr.append(np.hstack([tst_id,sol_rms]))

        # exit()
        if simulated_data:
            fig, [ax1,ax2,ax3,ax4,ax5] = plt.subplots(nrows=5,sharex=True)
        else:
            fig, [ax1,ax2,ax4] = plt.subplots(nrows=3,sharex=True)

        print("Total RMS for iters: ")
        iters_rms = np.array(iters_rms)
        iters_rms = iters_rms[np.argsort(iters_rms[:,0])]
        print(iters_rms)

        ax1.plot(iters_rms[:,1].astype('float'),'-r')
        ax1.set_ylabel('rms (m)')
        ax1a = ax1.twinx()
        ax1a.plot(iters_rms[:,2].astype('float'),'.k')
        ax1a.set_ylabel('num of obs (post-screening)')

        print("Total RMS/avg for solutions (orbpar): ")
        printout_list = [iters_orbcorr,iters_orbcorr_avg]
        for idx,printout in enumerate(printout_list):
            printout = pd.DataFrame(printout,columns=np.hstack(['tst_id',filter_string_orb]))
            printout = printout.sort_values(by='tst_id').reset_index(drop=True).drop(columns='tst_id').astype('float') #.round(2)
            printout[["/dRl","/dPt"]] *= 1e4
            print(printout)
            printout_list[idx] = printout

            if idx is 0:
                printout.plot(ax=ax2)
                ax2.set_ylabel('rms (tot orb sol)')

        if OrbRep == 'lin':
            print("Total RMS/avg for solutions (orbpar, lin): ")
            printout_list = [iters_orbcorr_lin, iters_orbcorr_avg_lin]
            for idx, printout in enumerate(printout_list):
                printout = pd.DataFrame(printout, columns=np.hstack(['tst_id', filter_string_orb_lin]))
                printout = printout.sort_values(by='tst_id').reset_index(drop=True).drop(columns='tst_id').astype(
                    'float')  # .round(2)
                # printout[["/dRl", "/dPt"]] *= 1e4
                print(printout)
                printout_list[idx] = printout

            if simulated_data:
                print("Total RMS for orbpar residuals (lin): ")
                filter_string_orb_lin = [x[:-1]+'1' for x in filter_string_orb_lin]
                iters_orbres_lin = pd.DataFrame(iters_orbres, columns=np.hstack(['tst_id', filter_string_orb_lin]))
                iters_orbres_lin = iters_orbres_lin.sort_values(by='tst_id').reset_index(drop=True).drop(
                    columns='tst_id').astype('float')  # .round(2)
                print(iters_orbres_lin)

        if simulated_data:
            print("Total RMS for orbpar residuals: ")
            iters_orbres = pd.DataFrame(iters_orbres,columns=np.hstack(['tst_id',filter_string_orb]))
            iters_orbres = iters_orbres.sort_values(by='tst_id').reset_index(drop=True).drop(columns='tst_id').astype('float') #.round(2)
            iters_orbres[["/dRl","/dPt"]] *= 1e5
            print(iters_orbres)

        if simulated_data and len(iters_orbres)>0:
            iters_orbres.plot(ax=ax3)
            ax3.get_legend().remove()
            ax3.set_ylabel('rms (orb res)')

        print("Total RMS for solutions (glopar): ")
        iters_glocorr = pd.DataFrame(iters_glocorr,columns=np.hstack(['tst_id',filter_string_glo]))
        iters_glocorr = iters_glocorr.sort_values(by='tst_id').reset_index(drop=True).drop(columns='tst_id').astype('float') #.round(2)
        print(iters_glocorr)

        if simulated_data:
            print("Residual % to be recovered (glopar): ")
            # pert_cloop_glo = {'dRA': np.linalg.norm([0.02, 0.000, 0.000]),
            #                                     'dDEC':np.linalg.norm([-0.01, 0.000, 0.000]),
            #                                     'dPM':np.linalg.norm([0, 2.e-5, 0.000]),
            #                                     'dL':-0.5*np.linalg.norm([0.00993822,-0.00104581,-0.00010280,-0.00002364,-0.00000532]),
            #                     'dh2': 0.}
            pert_cloop_glo = pert_cloop['glo']
            pert_cloop_glo = [np.linalg.norm(x) for x in list(pert_cloop_glo.values())]
            iters_glores = iters_glocorr.sub(pert_cloop_glo, axis='columns').abs()
            iters_glores = pd.DataFrame(iters_glores,columns=np.hstack(['tst_id',filter_string_glo]))
            iters_glores = iters_glores.sort_values(by='tst_id').reset_index(drop=True).drop(columns='tst_id').astype('float') #.round(2)
            iters_glores = iters_glores.divide(pert_cloop_glo,axis='columns')
            iters_glores *= 100
            print(iters_glores)
        # exit()

        iters_glocorr.plot(ax=ax4)
        ax4.set_ylabel('sol (glo sol)')

        if simulated_data:
            iters_glores.plot(logy=True,ax=ax5)
            ax5.get_legend().remove()
            ax5.set_ylabel('% NOT recovered')

            print(pert_cloop_glo)
        # exit()
        # ax2.set_ylabel('rms (m)')
        # ax1a = ax1.twinx()
        # ax1a.plot(iters_rms[:, 2].astype('float'), '.k')
        # ax1a.set_ylabel('num of obs (post-screening)')
        #
        plt.savefig(tmpdir + 'rms_iters_' + sol + '.png')
        plt.close()

        # exit()
    #print_corrmat(tmp,tmpdir+"corrmat.png")

    if False and local:
        from mpl_toolkits.basemap import Basemap
        mlacount = tmp.xov.xovers.round(0).groupby(['LON','LAT']).size().rename('count').reset_index()
        print(mlacount.sort_values(['LON']))

        fig = plt.figure(figsize=(8, 6), edgecolor='w')
        # m = Basemap(projection='moll', resolution=None,
        #             lat_0=0, lon_0=0)
        m = Basemap(projection='npstere',boundinglat=10,lon_0=0,resolution='l')
        x, y = m(mlacount.LON.values, mlacount.LAT.values)
        map = m.scatter(x, y,c=np.log(mlacount['count'].values), cmap='afmhot') # , marker=',', s=3**piv.count(),
        plt.colorbar(map)
        draw_map(m)
        fig.savefig(tmpdir+'mla_count_nps_'+sol+'_'+subexp+'.png')
        plt.clf()
        plt.close()
        print("npstere printed")

        #
        # fig, ax1 = plt.subplots(nrows=1)
        # # ax0.errorbar(range(len(dR_avg)),dR_avg, yerr=dR_std, fmt='-o')
        # # ax0.set(xlabel='Exp', ylabel='dR_avg (m)')
        # # ax1 = sns.heatmap(mlacount, square=False, annot=True, robust=True)
        # # cmap = sns.palplot(sns.light_palette("green"), as_cmap=True) #sns.diverging_palette(220, 10, as_cmap=True)
        # # Draw the heatmap with the mask and correct aspect ratio
        # # plot pivot table as heatmap using seaborn
        # piv = pd.pivot_table(mlacount, values="count", index=["LAT"], columns=["LON"], fill_value=0)
        # piv = (piv+empty_geomap_df).fillna(0)
        #
        # sns.heatmap(piv, xticklabels=10, yticklabels=10)
        # plt.tight_layout()
        # ax1.invert_yaxis()
        # #         ylabel='Topog ampl rms (1st octave, m)')
        # fig.savefig(tmpdir+'mla_count_'+sol+'_'+subexp+'.png')
        # plt.clf()
        # plt.close()

    if True:
        orb_sol, glb_sol, sol_dict = xovacc.analyze_sol(tmp, tmp.xov)
        xovacc.print_sol(orb_sol, glb_sol, tmp.xov, tmp)

        # trackA = gtrack(vecopts)
        # trackA = trackA.load('/home/sberton2/Works/NASA/Mercury_tides/out/sim/1301_'+sol+'_0/0res_1amp/gtrack_13/gtrack_' + '1301142347' + '.pkl')
        # print(trackA.pert_cloop)
        # exit()

        cols = orb_sol.filter(regex='sol.*$', axis=1).columns
        # print(list(cols))
        orb_sol[cols] = orb_sol.filter(regex='sol.*$', axis=1).apply(pd.to_numeric, errors='ignore')
        orb_std = orb_sol.copy().filter(regex='std.*$', axis=1).apply(pd.to_numeric, errors='ignore')

        name = "Accent"
        cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
        colors = cmap.colors  # type: list

        fig, ax = plt.subplots(nrows=1)


        for idx, col in enumerate(cols):
            ax.set_prop_cycle(color=colors)
            orb_sol.reset_index().plot(kind="scatter", x="index", y=col, color=colors[idx], label=col, ax=ax)
            # plt.errorbar(orb_std.index, orb_std['gas'], yerr=orb_std['std'])

        ax.set_xticks(orb_sol.index.values)
        ax.locator_params(nbins=10, axis='x')
        ax.set_ylabel('sol (m)')
        # ax.set_ylim(-300,300)
        plt.savefig(tmpdir + 'orbcorr_tseries_' + sol + '.png')
        plt.close()

        # num_bins = 'auto'
        # for idx, col in enumerate(cols):
        #     ax.set_prop_cycle(color=colors)
        #     n, bins, patches = plt.hist(orb_sol[col], bins=num_bins, density=True, facecolor=colors[idx], label=col, alpha=0.7)
        #
        # plt.legend()
        # plt.xlabel('delta (m)')
        # plt.ylabel('Probability')
        # plt.title(r'Histogram of par corr') #: $\mu=' + str(mean_dR) + ', \sigma=' + str(std_dR) + '$')
        # plt.savefig(tmpdir + '/histo_corr_' + sol + "_" + str(idx) + '.png')
        # plt.clf()

    if False:
        if pd.Series(['dA', 'dC','dR$']).isin(tmp.pert_cloop.columns).any() and False:

            # print residuals (original cloop perturbation - latest cumulated solution)
            orbpar_sol = list(set([x.split("_")[0].split("/")[1] for x in tmp.xov.parOrb_xy]))
            tmp_orb_sol = orb_sol.iloc[:,:4].copy()
            tmp_orb_sol.columns = ['orb'] + orbpar_sol
            tmp_orb_sol = tmp_orb_sol.set_index('orb')

            # initial pert + corrections from previous iteration
            tmp.pert_cloop = tmp.pert_cloop[orbpar_sol].dropna()

            if ref_sol != '' and len(ref.pert_cloop.columns)>0:
                print(ref.pert_cloop)
                ref.pert_cloop = ref.pert_cloop[orbpar_sol].dropna()

            # to do this, we would need the full initial perturbed value of parameters (we don't have it...)
            # tmp.pert_cloop.columns = ["sol_dR/" + x for x in tmp.pert_cloop.columns]
            # postfit_res = orb_sol.set_index('orb').apply(pd.to_numeric, errors='ignore',
            #                                              downcast='float') + tmp.pert_cloop
            # postfit_res = postfit_res[["sol_dR/" + x for x in orbpar_sol]].fillna(0)
            tmp.pert_cloop.sort_index(axis=1, inplace=True)
            print("par recovery avg, std:",tmp.pert_cloop.columns.values)
            tmp_orb_sol.sort_index(axis=1, inplace=True)
            print("sol curr iter", tmp_orb_sol.mean(axis=0).values, tmp_orb_sol.std(axis=0).values)
            print("initial pert + corrections prev.iter.", tmp.pert_cloop.mean(axis=0).values, tmp.pert_cloop.std(axis=0).values)
            print("rmse",rmse(tmp.pert_cloop,0).values)
            if ref_sol != '':
                ref.pert_cloop.sort_index(axis=1, inplace=True)
                print("initial pert",ref.pert_cloop.mean(axis=0).values, ref.pert_cloop.std(axis=0).values)

            fig, ax = plt.subplots(nrows=1)
            name = "Accent"
            cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
            colors = cmap.colors  # type: list
            # print('postfit_res')
            # print(postfit_res)
            # postfit_res.reset_index().plot(x="orb", color=colors, ax=ax)
            if ref_sol != '' and len(ref.pert_cloop.columns)>0:
                ref.pert_cloop.reset_index().plot(x="index", color=colors, style=':', ax=ax)
                # ref.pert_cloop.apply(lambda x: x.abs()).reset_index().plot(x="index", color=colors, style=':', ax=ax)
            tmp.pert_cloop.reset_index().plot(x="index", color=colors, style='-', ax=ax)
            # tmp.pert_cloop.apply(lambda x: x.abs()).reset_index().plot(x="index", color=colors, style='-', ax=ax)
            # ax.set_xticks(orb_sol.index.values)

            ax.set_xlabel('orbit #')
            ax.set_ylabel('sol (m)')
            # ax.set_ylim(-500,500)
            plt.savefig(tmpdir+'residuals_tseries_'+sol+'_'+subexp+'.png')
            plt.close()

            num_bins = 'auto'
            plt.clf()
            fig, ax = plt.subplots(nrows=1)
            for idx, col in enumerate(cols):
                ax.set_prop_cycle(color=colors)
                if ref_sol != '' and len(ref.pert_cloop.columns) > 0:
                    n, bins, patches = plt.hist(np.abs(ref.pert_cloop[col.split('/')[-1]].values.astype(np.float)), bins=num_bins, density=False,
                                                facecolor=colors[idx], label=col.split('/')[-1],
                                            alpha=0.3)
                # print(np.abs(tmp.pert_cloop[col.split('/')[-1]].values.astype(np.float)))
                n, bins, patches = plt.hist(np.abs(tmp.pert_cloop[col.split('/')[-1]].values.astype(np.float)), bins=num_bins, density=False,
                                            facecolor=colors[idx], label=col.split('/')[-1],
                                            alpha=0.7)

            plt.legend()
            plt.xlabel('delta (m)')
            plt.ylabel('Probability')
            plt.title(r'Histogram of par corr')  #: $\mu=' + str(mean_dR) + ', \sigma=' + str(std_dR) + '$')
            plt.savefig(tmpdir + '/histo_orbiter_' + sol + "_" + str(idx) + '.png')
            plt.clf()

        if ref_sol != '':
            xovacc.plt_histo_dR(sol+subexp, mean_dR, std_dR,
                            tmp.xov.xovers,xov_ref=ref.xov.xovers)
        else:
            xovacc.plt_histo_dR(sol+subexp, mean_dR, std_dR,
                            tmp.xov.xovers)  # [tmp.xov.xovers.orbA.str.contains('14', regex=False)])

        xovacc.plt_geo_dR(empty_geomap_df, sol+subexp, tmp.xov)
        # exit()

    if False:
        tmp_plot = tmp.xov.xovers.copy()
        fig, ax1 = plt.subplots(nrows=1)
        tmp_plot[['xOvID', 'dR/dL']].plot(x='xOvID',y=['dR/dL'], ax=ax1)
        # ax1.set_ylim(-30,30)
        fig.savefig(tmpdir+'mla_dR_dL_'+sol+'.png')
        plt.clf()
        plt.close()

        pd.set_option('display.max_columns', 500)
        print(tmp_plot.loc[np.abs(tmp_plot['dR/dL'].values)>0.])
        exit()

        # print(tmp_plot.dtypes)
        tmp_plot = tmp_plot[["dR/dL","LAT","LON"]].round(0)

        _ = tmp_plot[["dR/dL","LAT","LON"]].round(0).groupby(["LAT","LON"])["dR/dL"].apply(lambda x: rmse(x)).fillna(0).reset_index()
        piv = pd.pivot_table(_, values="dR/dL", index=["LAT"], columns=["LON"], fill_value=0)
        piv = (piv+empty_geomap_df).fillna(0)

        sns.heatmap(piv, xticklabels=10, yticklabels=10)
        plt.tight_layout()
        ax1.invert_yaxis()
        #         ylabel='Topog ampl rms (1st octave, m)')
        fig.savefig(tmpdir+'mla_dR_dL_piv_'+sol+'_'+subexp+'.png')
        plt.clf()
        plt.close()
        # exit()

        fig = plt.figure(figsize=(8, 6), edgecolor='w')
        # m = Basemap(projection='moll', resolution=None,
        #             lat_0=0, lon_0=0)
        m = Basemap(projection='npstere',boundinglat=10,lon_0=0,resolution='l')
        x, y = m(_.LON.values, _.LAT.values)
        map = m.scatter(x, y,c=_['dR/dL'].values, s=_['dR/dL'].values, cmap='Reds') # afmhot') # , marker=',', s=3**piv.count(),
        plt.colorbar(map)
        draw_map(m)
        fig.savefig(tmpdir+'mla_dR_dL_npstere_'+sol+'_'+subexp+'.png')
        plt.clf()
        plt.close()

    if False:
        # Check dR/dC / dR/dA
        # plot dR/dL and dR/dh2
        tmp_plot = tmp.xov.xovers.copy()
        tmp_plot['dR/dC'] = tmp.xov.xovers.loc[:,['dR/dC_A','dR/dC_B']].mean(axis=1)
        tmp_plot['dR/dA'] = tmp.xov.xovers.loc[:,['dR/dA_A','dR/dA_B']].mean(axis=1)
        tmp_plot['dR/dR'] = tmp.xov.xovers.loc[:,['dR/dR_A','dR/dR_B']].mean(axis=1)

        print(tmp_plot.columns)
        print(tmp_plot[['xOvID','dR/dC','dR/dA','dR/dR']])
        # exit()
        fig, ax1 = plt.subplots(nrows=1)
        tmp_plot[['xOvID', 'dR/dC', 'dR/dA','dR/dR']].plot(x='xOvID',y=['dR/dC','dR/dA','dR/dR'], ax=ax1)
        ax1.set_ylim(-3,3)
        fig.savefig(tmpdir+'mla_dR_dCAR_'+sol+'.png')
        plt.clf()
        plt.close()

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
        # ax.set_ylim(-0.1,0.1)
        plt.savefig('/home/sberton2/Works/NASA/Mercury_tides/PyXover/tmp/dRdC_dA_'+sol+'.png')
        plt.close()

    if False:

        # plot dR/dL and dR/dh2
        tmp.xov.xovers['dR/dL'] = tmp.xov.xovers.loc[:,['dR/dL']].abs()

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

        if False:
            tmp.xov.xovers['dR/dh2'] = tmp.xov.xovers.loc[:,['dR/dh2']].abs()

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

def add_xov_separation(tmp):
    tmp.xov.xovers['dist_max'] = tmp.xov.xovers.filter(regex='^dist_[A,B].*$').max(axis=1)
    tmp.xov.xovers['dist_min'] = tmp.xov.xovers.filter(regex='^dist_[A,B].*$').min(axis=1)
    tmp.xov.xovers['dist_minA'] = tmp.xov.xovers.filter(regex='^dist_A.*$').min(axis=1)
    tmp.xov.xovers['dist_minB'] = tmp.xov.xovers.filter(regex='^dist_B.*$').min(axis=1)
    tmp.xov.xovers['dist_min_mean'] = tmp.xov.xovers.filter(regex='^dist_min.*$').mean(axis=1)


if __name__ == '__main__':

    simulated_data = True
    # analyze_sol(sol='KX1r_0', ref_sol='KX1r_0', subexp = '0res_1amp')
    analyze_sol(sol='tp8_0', ref_sol='tp8_0', subexp = '3res_20amp')
