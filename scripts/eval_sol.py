#!/usr/bin/env python3
# ----------------------------------
# Analyze, screen and plot AccumXov results
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-May-2019
#
import glob
import itertools as itert
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import diags

from src.accumxov import AccumXov as xovacc, accum_utils
from src.xovutil import plot_utils
from src.pyxover import xov_utils
from src.accumxov.Amat import Amat
from src.accumxov.accum_utils import get_xov_cov_tracks
from src.xovutil.xovres2weights import get_roughness_at_coord, get_interpolation_weight
from config import XovOpt
# from AccumXov import plt_geo_dR
# from ground_track import gtrack
from src.pyxover.xov_setup import xov
from src.pyxover.xov_utils import get_tracks_rms, plot_tracks_histo

remove_max_dist = True
remove_3sigma_median = True

plot_all_track_iters = False

subfolder = '' #'archived/KX1r4_KX/' #'archived/tp9_0test_tides/'

def xovnum_plot():

    vecopts = {}

    if True:
        #xov_cmb = xov.load_combine('/home/sberton2/Works/NASA/Mercury_tides/out/sim_mlatimes/0res_1amp/',vecopts)
        xov_sim = xov_utils.load_combine('/home/sberton2/Works/NASA/Mercury_tides/out/sim/mlatimes/1301_sph_0/0res_1amp/', vecopts)
        xov_real = xov_utils.load_combine('/home/sberton2/Works/NASA/Mercury_tides/out/sim/mlatimes/1301/0res_1amp/', vecopts)
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
    plt.savefig(XovOpt.get("tmpdir") + 'xover_month.png')
    plt.close()

def print_corrmat(amat,filename):
    # Compute the covariance matrix
    # print(np.linalg.pinv((ref.spA.transpose() * ref.spA).todense()))
    corr = amat.corr_mat()
    print("corr")
    print(corr)
    # mask to select only parameters with corrs > 0.9
    m = (corr.mask(np.eye(len(corr), dtype=bool)).abs() > 0.95).any()
    # exit()
    if len(m)>0:
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
    f = plt.figure(figsize=(200, 200))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5,annot=False, fmt='.1f', cbar_kws={"shrink": .5})
    plt.yticks(rotation=0)
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

#####################################################
def analyze_sol(sols, ref_sol = '', subexp = ''):
    from matplotlib.cm import get_cmap

    # run options
    plot_resmap = True
    plot_reshisto= True
    plot_weights_components = False # True #
    compare_spk = False
    plot_sol_errors = False #True
    plot_orbcorr = True

    vecopts = {}
    # introduce reference solution in plot
    if ref_sol != '':
        ref = Amat(vecopts)
        ref = ref.load(XovOpt.get("outdir") + 'sim/' + subfolder + ref_sol + '/' + subexp + '/Abmat_sim_' + ref_sol.split('_')[0] + '_' + str(
            int(ref_sol.split('_')[-1]) + 1) + '_' + subexp + '.pkl')

    dfs = []
    for sol in sols:
        tmp = Amat(vecopts)

        try:
            tmp = tmp.load(
                XovOpt.get("outdir") + 'sim/' + subfolder + sol + '/' + subexp + '/Abmat_sim_' + sol.split('_')[0] + '_' + str(
                    int(sol.split('_')[-1]) + 1) + '_' + subexp + '.pkl')
        except:
            tmp = tmp.load(
                XovOpt.get("outdir") + 'Abmat/Abmat_sim_' + sol.split('_')[0] + '_' + str(
                    int(sol.split('_')[-1]) + 1) + '_' + subexp + '.pkl')

        # select subset, e.g., for tests
        tmp.xov.xovers = tmp.xov.xovers.loc[:,:]
        print("Processing",sol,", containing",len(tmp.xov.xovers),"xovers...")
        print(tmp.vce)
        # print(tmp.sol_dict['sol'])
        # exit()
        dfs.append(tmp)


    if plot_resmap:
        for idx,amat in enumerate(dfs):
            plot_utils.plt_geo_dR(sols[idx] + '_' + subexp, amat.xov, truncation=20)

    if plot_reshisto:
        for idx,amat in enumerate(dfs):
            # if ref_sol != '':
            #     xovacc.plt_histo_dR(sol+subexp, mean_dR, std_dR,
            #                     amat.xov.xovers,xov_ref=ref.xov.xovers)
            # else:
            plot_utils.plt_histo_dR(sols[idx] + '_' + subexp,
                                    amat.xov.xovers, xlim=100)  # [tmp.xov.xovers.orbA.str.contains('14', regex=False)])
            if idx == len(dfs)-1 and ref_sol != '':
                plot_utils.plt_histo_dR(idx=sols[idx] + '_' + subexp,
                                        xov_df=amat.xov.xovers, xov_ref=ref.xov.xovers,
                                        xlim=100)
            # plot histo of offnadir
            plot_utils.plt_histo_offnadir(sols[idx] + '_' + subexp,
                                          amat.xov.xovers, xlim=10)


    if plot_weights_components:
        check_weights = pd.DataFrame()
        check_weights['interp_weights'] = get_interpolation_weight(tmp.xov).weight

        if ['dtA','huber'] in list(tmp.xov.xovers.columns):
            tmp_trkw = tmp.xov.xovers.copy()[
                ['xOvID', 'LON', 'LAT', 'dtA', 'dR', 'orbA', 'orbB', 'huber']]  # .astype('float16')
        else:
            tmp_trkw = tmp.xov.xovers.copy()[
                ['xOvID', 'LON', 'LAT', 'dR', 'orbA', 'orbB']]
            tmp_trkw['huber'] = 1.
            tmp_trkw['dtA'] = tmp_trkw['LAT'].values
            tmp_trkw=tmp_trkw.loc[(tmp_trkw.orbA.str.startswith('140706'))] # or (tmp_trkw.orbB.str.startswith('140706'))]
            print(tmp_trkw)
        # exit()

        print("pre xovcov types", tmp_trkw.dtypes)
        check_weights['tracks_weights'] = get_xov_cov_tracks(df=tmp_trkw, plot_stuff=False).diagonal()

        check_weights['huber'] = tmp_trkw.huber #tmp.xov.xovers.copy().huber
        check_weights['weights'] = check_weights['tracks_weights']*check_weights['interp_weights']*check_weights['huber']
        # tmp.xov.xovers['weights'] = check_weights['weights'].values/check_weights['weights'].max()

        plt.figure()  # figsize=(8, 3))
        num_bins = 'auto'  # 40  # v

        # check_weights.hist(bins=num_bins) #, cumulative=True)
        # plt.xlim([0, 1.0])
        # plt.xlabel('roughness@baseline700 (m/m)')


        sns.distplot(check_weights['huber'], kde=True, label='huber', rug=False, bins=500) #, hist_kws={'range':(0.05,1)})
        sns.distplot(check_weights['tracks_weights'], kde=True, label='tracks_weights', rug=False) #, bins=100, hist_kws={'range':(0.1,10)})
        sns.distplot(check_weights['interp_weights'], kde=True, label='interp_weights', rug=False) #, hist_kws={'range':(0.1,40)})
        sns.distplot(check_weights['weights'], kde=True, label='weights', rug=False, bins=500) #, hist_kws={'range':(0.05,10)})
        plt.xlim([0.05, 40.0])
        plt.semilogx()
        plt.semilogy()

        plt.legend()
        plt.savefig(XovOpt.get("tmpdir") + '/weights.png')
        plt.clf()

        print(check_weights)

        print("max:",check_weights.max())
        print("min:",check_weights.min())
        print("mean:",check_weights.mean())
        print("median:",check_weights.median())

        # exit()

    if compare_spk:
        sol, tmp = compare_MLA_spk(dfs, sol, sols, tmp)


    if plot_sol_errors:
        for idx,amat in enumerate(dfs):

            # compute formal errors
            sol = amat.sol_dict['sol']
            formal_err = amat.sol_dict['std']
            # print(formal_err)

            # compute a posteriori
            ATP = amat.spA.T * amat.weights
            # degf = len(amat.xov.xovers['dR'].values) - len(sol4_glo)
            m_0 = amat.resid_wrmse
            # iters_rms.append([tst_id, np.sqrt(lTPl/degf)[0][0], m_0, degf])
            PA = amat.weights * amat.spA
            ell = diags(np.abs(amat.b))

            # if idx == len(dfs[:])-1:
            posterr = np.linalg.pinv((ATP * ell * PA).todense())
            posterr = np.sqrt(posterr.diagonal())
            m_X = dict(zip(amat.sol4_pars,np.ravel(m_0 * posterr[0])))

            apost_err = m_X
            print(apost_err)

            df = pd.DataFrame([sol,formal_err,apost_err]).T
            df.columns = ['sol','formal_LS','apost_uw']

            df = df.sort_index()
            print(df)
            print("### plot_sol_errors: m_X produced for sol", sols[idx])

    if plot_orbcorr:
        for idx,amat in enumerate(dfs):

            orb_sol, glb_sol, sol_dict = accum_utils.analyze_sol(amat, amat.xov)
            cols = orb_sol.filter(regex='sol.*$', axis=1).columns
            orb_sol[cols] = orb_sol.filter(regex='sol.*$', axis=1).apply(pd.to_numeric, errors='ignore')
            # orb_std = orb_sol.copy().filter(regex='std.*$', axis=1).apply(pd.to_numeric, errors='ignore')

            name = "Accent"
            cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
            colors = cmap.colors  # type: list

            fig, ax = plt.subplots(nrows=1)


            for idx_col, col in enumerate(cols):
                ax.set_prop_cycle(color=colors)
                orb_sol_plot = orb_sol.copy()
                orb_sol_plot[orb_sol_plot.filter(regex='sol.*d[P,R][l,t]$', axis=1).columns] *= 1.e6
                orb_sol_plot.reset_index().plot(kind="scatter", x="index", y=col, color=colors[idx_col], label=col, ax=ax)

            ax.set_xticks(orb_sol.index.values)
            ax.locator_params(nbins=10, axis='x')
            ax.set_ylabel('sol (m)')
            ax.set_xlabel('track #')

            plt.savefig(XovOpt.get("tmpdir") + 'orbcorr_tseries_' + sols[idx] + '.png')
            plt.close()
            print("### plot_orbcorr: orbit solutions traced for sol", sols[idx])

    exit()

    # print([x.xov.xovers.dR for x in dfs])
    # exit()


    # exit()
    #
    #
    # exit()
    # exit()

    fig = plt.figure("resid_iters")

    for idx,tmp in enumerate(dfs):
        # evaluate residuals
        xovers = tmp.xov.xovers.copy()
        print('sol',idx,'contains',len(xovers),'xovers with mean',xovers['dR'].mean(), ', median',xovers['dR'].median(),'and std',xovers['dR'].std())
        plt.hist(xovers['dR'].values, label='dR_'+str(idx), density= True,  bins=10000, alpha= 1./(idx+1),range=[-10000,10000])
        plt.xlim([-500., 500.0])
        plt.semilogy()

        # plt.hist((xovers[['dR','weights']].product(axis='columns',skipna=True)).values, label=str('weighted'), bins=50, range=[-100, 100])

        # plt.title('Mean')
        # plt.xlabel("value")
        # plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(XovOpt.get("tmpdir") + "histo_resid.png")

    # exit()

    # get roughness at 700 meters from http://www.planetary.brown.edu/html_pages/mercury_roughness-maps.html
    # ref baseline in meters (depending on input map)
    # ref_baseline = 700.
    # lonlat = interp_weights.loc[:,['LON','LAT']].values
    # roughness_df = get_roughness_at_coord(lon=lonlat[:,0],lat=lonlat[:,1],
    #                                       roughness_map=auxdir+'MLA_Roughness_composite.tif')


    # Remove huge outliers
    # mean_dR, std_dR, worst_tracks = tmp.xov.remove_outliers('dR',remove_bad=remove_3sigma_median)
    # tmp.xov.xovers['dR_abs'] = tmp.xov.xovers.dR.abs()
    # print("Largest dR ( # above 200m", len(tmp.xov.xovers[tmp.xov.xovers.dR_abs > 200])," or ",
    #       (len(tmp.xov.xovers[tmp.xov.xovers.dR_abs > 200])/len(tmp.xov.xovers)*100.),'%)')
    # print(tmp.xov.xovers[['orbA','orbB','dist_max','dist_min_mean','dR_abs']].nlargest(10,'dR_abs'))
    # print(tmp.xov.xovers[['orbA','orbB','dist_max','dist_min_mean','dR_abs']].nsmallest(10,'dR_abs'))
    #
    # # Recheck distance after cleaning
    # xovacc.analyze_dist_vs_dR(tmp.xov)
    # _ = tmp.xov.xovers.dR.values ** 2
    # print("Total RMS:", np.sqrt(np.mean(_[~np.isnan(_)], axis=0)), len(tmp.xov.xovers.dR.values))

    if XovOpt.get("local") and False:
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
        fig.savefig(XovOpt.get("tmpdir") + 'mla_count_nps_' + sol + '_' + subexp + '.png')
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
        orb_sol, glb_sol, sol_dict = accum_utils.analyze_sol(tmp, tmp.xov)
        accum_utils.print_sol(orb_sol, glb_sol, tmp.xov, tmp)

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
            orb_sol_plot = orb_sol.copy()
            # print(orb_sol)
            orb_sol_plot[orb_sol_plot.filter(regex='sol.*d[P,R][l,t]$', axis=1).columns] *= 1.e6
            # exit()
            orb_sol_plot.reset_index().plot(kind="scatter", x="index", y=col, color=colors[idx], label=col, ax=ax)
            # plt.errorbar(orb_std.index, orb_std['gas'], yerr=orb_std['std'])

        ax.set_xticks(orb_sol.index.values)
        ax.locator_params(nbins=10, axis='x')
        ax.set_ylabel('sol (m)')
        # ax.set_ylim(-5,5)
        plt.savefig(XovOpt.get("tmpdir") + 'orbcorr_tseries_' + sol + '.png')
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
            print("par recovery avg, std:", tmp.pert_cloop.columns.values)
            tmp_orb_sol.sort_index(axis=1, inplace=True)
            print("sol curr iter", tmp_orb_sol.mean(axis=0).values, tmp_orb_sol.std(axis=0).values)
            print("initial pert + corrections prev.iter.", tmp.pert_cloop.mean(axis=0).values, tmp.pert_cloop.std(axis=0).values)
            print("rmse", rmse(tmp.pert_cloop, 0).values)
            if ref_sol != '':
                ref.pert_cloop.sort_index(axis=1, inplace=True)
                print("initial pert", ref.pert_cloop.mean(axis=0).values, ref.pert_cloop.std(axis=0).values)

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
            plt.savefig(XovOpt.get("tmpdir") + 'residuals_tseries_' + sol + '_' + subexp + '.png')
            plt.close()

            num_bins = 'auto'
            plt.clf()
            fig, ax = plt.subplots(nrows=1)
            for idx, col in enumerate(cols):
                ax.set_prop_cycle(color=colors)
                if ref_sol != '' and len(ref.pert_cloop.columns) > 0:
                    n, bins, patches = plt.hist(np.abs(ref.pert_cloop[col.split('/')[-1]].values.astype(float)), bins=num_bins, density=False,
                                                facecolor=colors[idx], label=col.split('/')[-1],
                                                alpha=0.3)
                # print(np.abs(tmp.pert_cloop[col.split('/')[-1]].values.astype(float)))
                n, bins, patches = plt.hist(np.abs(tmp.pert_cloop[col.split('/')[-1]].values.astype(float)), bins=num_bins, density=False,
                                            facecolor=colors[idx], label=col.split('/')[-1],
                                            alpha=0.7)

            plt.legend()
            plt.xlabel('delta (m)')
            plt.ylabel('Probability')
            plt.title(r'Histogram of par corr')  #: $\mu=' + str(mean_dR) + ', \sigma=' + str(std_dR) + '$')
            plt.savefig(XovOpt.get("tmpdir") + '/histo_orbiter_' + sol + "_" + str(idx) + '.png')
            plt.clf()



    if False:
        tmp_plot = tmp.xov.xovers.copy()
        fig, ax1 = plt.subplots(nrows=1)
        tmp_plot[['xOvID', 'dR/dL']].plot(x='xOvID',y=['dR/dL'], ax=ax1)
        # ax1.set_ylim(-30,30)
        fig.savefig(XovOpt.get("tmpdir") + 'mla_dR_dL_' + sol + '.png')
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
        fig.savefig(XovOpt.get("tmpdir") + 'mla_dR_dL_piv_' + sol + '_' + subexp + '.png')
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
        fig.savefig(XovOpt.get("tmpdir") + 'mla_dR_dL_npstere_' + sol + '_' + subexp + '.png')
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
        fig.savefig(XovOpt.get("tmpdir") + 'mla_dR_dCAR_' + sol + '.png')
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
        for par in ['dL','dRA','dDEC']:
            tmp.xov.xovers['dR/'+par] = tmp.xov.xovers.loc[:,['dR/'+par]].abs()

            mladRdL = tmp.xov.xovers.round({'LON': 0, 'LAT': 0, 'dR': 3}).groupby(['LON', 'LAT']).median().reset_index()
            print(mladRdL)

            fig, ax1 = plt.subplots(nrows=1)
            # ax0.errorbar(range(len(dR_avg)),dR_avg, yerr=dR_std, fmt='-o')
            # ax0.set(xlabel='Exp', ylabel='dR_avg (m)')
            # ax1 = sns.heatmap(mlacount, square=False, annot=True, robust=True)
            # cmap = sns.palplot(sns.light_palette("green"), as_cmap=True) #sns.diverging_palette(220, 10, as_cmap=True)
            # Draw the heatmap with the mask and correct aspect ratio
            piv = pd.pivot_table(mladRdL, values='dR/'+par, index=["LAT"], columns=["LON"], fill_value=0)
            # plot pivot table as heatmap using seaborn
            # piv = (piv+empty_geomap_df).fillna(0)
            # print(piv)
            # exit()
            sns.heatmap(piv, xticklabels=10, yticklabels=10, vmax=10000)
            plt.tight_layout()
            ax1.invert_yaxis()
            #         ylabel='Topog ampl rms (1st octave, m)')
            fig.savefig(XovOpt.get("tmpdir") + 'mla_' + par + '_' + sol + '.png')
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
            fig.savefig(XovOpt.get("tmpdir") + 'mla_dh2_' + sol + '.png')
            plt.clf()
            plt.close()


def compare_MLA_spk(dfs, sol, sols, tmp):
    # uniform datasets
    dfA = dfs[0].xov.xovers[['dR', 'orbA', 'orbB']]
    dfA['dR'] = dfA.dR.abs()
    dfA.columns = ['dRA', 'orbA', 'orbB']
    dfB = dfs[1].xov.xovers[['dR', 'orbA', 'orbB']]
    dfB['dR'] = dfB.dR.abs()
    dfB.columns = ['dRB', 'orbA', 'orbB']
    dfC = dfs[2].xov.xovers[['dR', 'orbA', 'orbB']]
    dfC['dR'] = dfC.dR.abs()
    dfC.columns = ['dRC', 'orbA', 'orbB']
    tmp = pd.merge(dfA, dfB, how='left', left_on=['orbA', 'orbB'], right_on=['orbA', 'orbB'])
    df_merged = pd.merge(tmp, dfC, how='left', left_on=['orbA', 'orbB'], right_on=['orbA', 'orbB'])
    print(df_merged)
    # define subsets by year
    subs = [''] + [str(x) for x in np.arange(11, 16)]
    fig = plt.figure("resid_iters")
    for sub in subs[:1]:
        print("sub=", sub)
        plt.clf()
        if sub == '':
            df = df_merged.filter(regex='dR?')
        else:
            df = df_merged.loc[
                (df_merged['orbA'].str.startswith(sub)) & (df_merged['orbB'].str.startswith(sub))].filter(regex='dR?')
        df.columns = sols
        df.plot.hist(alpha=0.5, density=False, bins=200, range=[0, 1000])
        plt.title('Histogram of xovers discrepancies')
        plt.xlim([0., 1000.0])
        plt.xlabel('meters')
        plt.ylabel('# of xovs (log)')
        plt.semilogy()

        plt.savefig(XovOpt.get("tmpdir") + "histo_resid_" + sub + ".png")
    df_merged = df_merged.astype({'orbA': 'int32', 'orbB': 'int32'})
    # select only if acceptable xovs (< 1km)
    acceptif = (df_merged.dRA < 2.e3) & (df_merged.dRB < 2.e3) & (df_merged.dRC < 2.e3)
    total_occ_tracks = pd.DataFrame([df_merged['orbA'].value_counts(), df_merged['orbB'].value_counts()]).T.fillna(
        0).sum \
        (axis=1).sort_values(ascending=False)
    tracks = list(total_occ_tracks.index.values)[:]
    dRs = ['dRA', 'dRB', 'dRC']
    tot = len(df_merged)
    print("Out of ", tot, "xovers:")
    for idx, dR in enumerate(dRs):
        print(sols[idx], "has ", round(df_merged.loc[df_merged[dR] < 100., dR].count() / tot * 100., 2),
              "% xovers with dR<100 m,",
              round(df_merged.loc[df_merged[dR] < 50., dR].count() / tot * 100., 2), "% xovers with dR <50 m and",
              round(df_merged.loc[df_merged[dR] < 10., dR].count() / tot * 100., 2), "% xovers with dR <10 m.")
    df_median = []
    df_mean = []
    df_merged = df_merged.loc[acceptif]
    for tr in tracks[:]:
        df = df_merged.loc[((df_merged.orbA == int(tr)) | (df_merged.orbB == int(tr)))]
        df_mean.append(df.filter(regex='dR?').mean(axis=0).append(pd.Series([tr], index=['track'])))
        df_median.append(df.filter(regex='dR?').median(axis=0).append(pd.Series([tr], index=['track'])))
    df_median = pd.concat(df_median, axis=1).T.set_index('track').abs()
    df_median.index = df_median.index.map(str)
    df_median.columns = sols
    df_median.dropna(inplace=True)
    df_median.plot.hist(alpha=0.5, density=False, bins=500)
    plt.title('Histogram of tracks median dR')
    plt.xlim([0., 300.0])
    plt.xlabel('meters')
    plt.ylabel('# of tracks')
    # plt.semilogy()
    plt.savefig(XovOpt.get("tmpdir") + "histo_tracks1.png")
    print(df_median.sort_index())
    print(df_median.mean(axis=0))
    print(df_median.std(axis=0))
    for sol in sols:
        print("##############")
        print("best 10 by", sol)
        print(df_median.sort_values(by=sol).iloc[:10, :])
        print("##############")
        print("worst 10 by", sol)
        print(df_median.sort_values(by=sol).iloc[-10:, :])

        # print(df.filter(regex='dR?').std(axis=0))
    return sol, tmp


# Check convergence over iterations
# @profile
def check_iters(sol, subexp=''):

    plt_tracks = False

    ind = np.array(['RA', 'DEC', 'PM', 'L','h2'])

    np.set_printoptions(precision=3)

    sol_iters = sol.split('_')[:-1][0]
    prev_sols = np.sort(glob.glob(XovOpt.get("outdir") + 'sim/' + subfolder + sol_iters + '_*/' + subexp + '/Abmat_sim_' + sol_iters + '_*_' + subexp + '.pkl'))
    # prev_sols = np.sort(glob.glob(outdir+'Abmat/KX1r4_AG2/'+subexp+'/Abmat_sim_'+sol_iters+'_*_'+subexp+'.pkl'))

    if len(prev_sols) == 0:
        print("*** eval_sol: No solutions selected")
        exit(1)

    sols_iters = []
    rmse_iters = []
    iters_track_rms = []
    for idx,isol in enumerate(prev_sols[:]):
        # print(prev_sols)
        amat = Amat(XovOpt.get("vecopts"))
        amat = amat.load(isol)
        # solution of orbit pars
        sol_glb = {i:amat.sol_dict['sol']['dR/d' + i] for i in ind if 'dR/d' + i in amat.sol_dict['sol'].keys()}
        err_glb = {i:1.*np.sqrt(amat.sol_dict['std']['dR/d' + i]) for i in ind if 'dR/d' + i in amat.sol_dict['std'].keys()}

        sols_iters.append(sol_glb)
        rmse_iters.append(amat.resid_wrmse)

        # track by track evaluation
        if idx in [0,len(prev_sols)-1] and plt_tracks:
            iters_track_rms.append(get_tracks_rms(amat.xov.xovers.copy()))

    #errors
    errs = list(err_glb.values())
    errs.extend([1.])

    df = pd.DataFrame.from_dict(sols_iters)
    df['rmse'] = pd.DataFrame(rmse_iters)
    print(df)
    df_diff = df.diff(axis=0).abs()
    print(df_diff)
    # put full correction as first row
    df_diff.loc[0] = df.loc[0].abs()
    # get relative to formal errors
    df_diff = df_diff/errs #/df.iloc[-1]

    # get pre-fit RMS
    # xover residuals
    amat = amat.load(prev_sols[0])
    w = amat.xov.xovers['dR'].values
    nobs = len(w)
    npar = len(amat.sol_dict['sol'].values())

    lTP = w.reshape(1, -1) @ amat.weights
    lTPl = lTP @ w.reshape(-1, 1)
    pre_fit_m0 = np.sqrt(lTPl / (nobs - npar))
    print("pre-fit m0 =", pre_fit_m0)
    df_diff.loc[0,'rmse'] = np.abs(pre_fit_m0-df['rmse'].iloc[0])

    # for rmse, get percentage
    df_diff['rmse'] = df_diff['rmse']/df['rmse'].iloc[-1]*100

    print(df_diff)
    print("err_dict=",err_glb)
    print("err_dict=",list(err_glb.values()))

    fig = plt.figure(figsize=(5, 5))
    plt.style.use('seaborn-paper')

    axes = df_diff[:].plot(subplots=True, layout=(2, 3), sharex=True, logy=False, legend=False) #, sharey=True)
    plt.subplots_adjust(wspace=0.4, hspace=0.4) #left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    for row,c in enumerate(axes): # row

        for col, ax in enumerate(c):  # column
            idx = 3*row+col
            if idx < len(ind):
                lbl = ind[idx]
            else:
                lbl = 'RMSE'
            ax.set_title(lbl)
            # ax.yaxis.set_label_position("right")

            if idx < len(ind):
                ax.axhline(y=1., color='r',linestyle='dashed')
                # ax.axhline(y=list(err_glb.values())[idx], color='r',linestyle='dashed')
            else:
                ax.axhline(y=1, color='r',linestyle='dashed')
                ax.axhline(y=5, color='b',linestyle='dashed')

    ax.set_xlabel('# iters', fontsize=12)
    ax.xaxis.set_label_coords(-1, -0.3)
    # fig.text(0, 0, '# iters', ha='center')


    fig.tight_layout()

    filnam = XovOpt.get("tmpdir") + "evol_iters.png"
    plt.savefig(filnam)
    print("### Iters plot saved as", filnam)

    # get evolution of tracks
    if plt_tracks:
        plot_tracks_histo(iters_track_rms) #,filename=tmpdir + '/histo_tracks_eval' + isol +'.png')

        # print(err_glb)
    exit()


def check_iters_old(sol, subexp=''):

    np.set_printoptions(precision=3)

    sol_iters = sol.split('_')[:-1][0]
    prev_sols = np.sort(glob.glob(
        XovOpt.get("outdir") + 'sim/' + subfolder + sol_iters + '_*/' + subexp + '/Abmat_sim_' + sol_iters + '_*_' + subexp + '.pkl'))

    iters_rms = []
    iters_orbcorr = []
    iters_orbcorr_avg = []
    iters_orbcorr_it = []
    iters_orbcorr_avg_it = []
    iters_orbcorr_lin = []
    iters_orbcorr_avg_lin = []
    iters_orbres = []
    iters_orbres_mean = []
    iters_glocorr = []
    m_X_iters = []
    iters_track_rms = []

    if len(prev_sols) == 0:
        print("*** eval_sol: No solutions selected")
        exit(1)

    for idx,isol in enumerate(prev_sols[:]):
        # print(prev_sols)
        prev = Amat(XovOpt.get("vecopts"))
        prev = prev.load(isol)

        # if (plot_all_track_iters) or (idx == len(prev_sols)-1) or (idx == 0):
        #     iters_track_rms.append(get_tracks_rms(prev.xov.xovers.copy()))
        #
        # add_xov_separation(prev)
        # prev.xov.xovers = prev.xov.xovers[prev.xov.xovers.dist_max < 0.4]
        # prev.xov.xovers = prev.xov.xovers[prev.xov.xovers.dist_min_mean < 1]
        # mean_dR, std_dR, worst_tracks = prev.xov.remove_outliers('dR', remove_bad=remove_3sigma_median)
        # weigh observations

        # prev.xov.xovers['dR'] *= (prev.weights/prev.weights.max())
        # _ = prev.xov.xovers.dR.values ** 2

        # print(isol)

        tst = isol.split('/')[-3].split('_')[1]
        tst_id = isol.split('/')[-3].split('_')[0]+tst.zfill(2)
        # print("tst_id",tst_id)
        # iters_rms.append([tst_id, np.sqrt(np.mean(_[~np.isnan(_)], axis=0)), len(_)])

        ################ print histo ###############
        if (idx == 0) or (idx == len(prev_sols) - 1) and False:
            if idx == 0 :
                fig = plt.figure("resid_iters")
            else:
                plt.figure("resid_iters")
            plt.hist(prev.xov.xovers['dR'].values, label=str(idx),bins=50, range=[-100, 100])
            # plt.title('Mean')
            # plt.xlabel("value")
            # plt.ylabel("Frequency")
            plt.legend()
            if idx == len(prev_sols) - 1 :
                plt.savefig(XovOpt.get("tmpdir") + "histo_resid_iters.png")
        ############################################

        # print(prev.xov.xovers[['dR', 'huber', 'weights']])
        #print("@iter", tst_id)
        lTP = prev.xov.xovers['dR'].values.reshape(1,-1)@prev.weights
        lTPl = lTP @ prev.xov.xovers['dR'].values.reshape(-1,1)

        if idx == len(prev_sols)-1:
            # print_corrmat(prev, tmpdir + 'corrmat_' + sol + '.pdf')
            pass

        # filter_string_glo = ["/" + x.split('/')[-1] for x in sol4_glo]  # ["/dRA","/dDEC","/dPM","/dL","/dh2"]
        xsol = []
        xstd = []
        for filt in prev.sol4_pars:
            filtered_dict = {k: v for (k, v) in prev.sol_dict['sol'].items() if filt in k}
            xsol.append(list(filtered_dict.values())[0])
            filtered_dict = {k: v for (k, v) in prev.sol_dict['std'].items() if filt in k}
            xstd.append(list(filtered_dict.values())[0])

        if prev.sol4_pars != []:
            # select columns of design matrix corresponding to chosen parameters to solve for
            # print([xovi_amat.parNames[p] for p in sol4_pars])
            prev.spA = prev.spA[:, [prev.parNames[p] for p in prev.parNames.keys()]]
            # set b=0 for rows not involving chosen set of parameters
            nnz_per_row = prev.spA.getnnz(axis=1)
            prev.b[np.where(nnz_per_row == 0)[0]] = 0

        # xT = np.array(xsol).reshape(1,-1)
        ATP = prev.spA.T * prev.weights
        # ATPb = ATP * prev.b

        # when iterations have inconsistent npars, some are imported from previous sols for
        # consistency reasons. Here we remove these values from the computation of vTPv
        # else matrix shapes don't match
        # if len(ATPb) != len(xT):
        #   missing = [x for x in prev.sol4_pars if x not in prev.parNames]
        #   missing = [prev.sol4_pars.index(x) for x in missing]
        #   xT = np.delete(xT,missing)
        # #print(prev.sol4_pars)
        # ##################
        # vTPv = lTPl # - xT@ATPb # TODO CORRECT THIS!!!!
        degf = len(prev.xov.xovers['dR'].values) - len(XovOpt.get("sol4_glo"))
        #print("vTPv = ", vTPv, vTPv/degf)
        #print("degf = ", degf)
        # m_0 = np.sqrt(vTPv/degf)[0][0]

        m_0 = prev.resid_wrmse
        # print("m0 = ",m_0)
        iters_rms.append([tst_id, np.sqrt(lTPl/degf)[0][0], m_0, degf])

        # ATPA = ATP * prev.spA
        PA = prev.weights * prev.spA
        # N = ATPA
        # Ninv = np.linalg.pinv(N)
        ell = diags(np.abs(prev.b))

        if idx == len(prev_sols[:])-1:
            posterr = np.linalg.pinv((ATP * ell * PA).todense())
            posterr = np.sqrt(posterr.diagonal())
            m_X = dict(zip(prev.sol4_pars,np.ravel(m_0 * posterr[0])))
            #print("a post error on params", m_X)
            # sigma_X = dict(zip(prev.sol4_pars,xstd))
            # print("a priori error on params", sigma_X)
            # print("ratios", {k: m_X[k]/sigma_X[k] for k in m_X.keys() &  sigma_X})

            m_X_iters.append(m_X)

        #prev.xov.xovers['dR'].values.T * prev.xov.xovers['weights'].values * prev.xov.xovers['dR'].values)
        # exit()

        ####################################

        filter_string_orb = XovOpt.get("sol4_orbpar") #["/dA","/dC","/dR","/dRl","/dPt"]
        if XovOpt.get("OrbRep") == 'lin':
            filter_string_orb = list(set([x+y if x in ['dA', 'dC', 'dR'] else x for x in filter_string_orb for y in ['0','1'] ]))
        if filter_string_orb != [None]:
            sol_rms = []
            sol_avg = []
            sol_rms_iter = []
            sol_avg_iter = []
            for idx_filt, filt in enumerate(filter_string_orb):
                filtered_dict = {k:v for (k,v) in prev.sol_dict['sol'].items() if re.search(filt+'$',k)} #filt in k if k not in ['dR/dRA']}
                filtered_dict = list(filtered_dict.values())

                # sol_rms.append(np.sqrt(np.mean(np.array(filtered_dict) ** 2)))
                sol_rms.append(np.median(filtered_dict))
                sol_avg.append(np.mean(np.array(filtered_dict)))
                filtered_dict_iter = {k:v for (k,v) in prev.sol_dict_iter['sol'].items() if re.search(filt+'$',k)} #filt in k if k not in ['dR/dRA']}
                filtered_dict_iter = list(filtered_dict_iter.values())

                ################ print histo ###############
                if (idx == 0) or (idx == len(prev_sols) - 1):
                    if idx == 0 and idx_filt==0:
                        fig1, ax = plt.subplots(len(filter_string_orb),num="orbcorr")
                    else:
                        plt.subplots(len(filter_string_orb),num="orbcorr")
                    ax[idx_filt].hist(filtered_dict_iter,label=str(idx),bins=50,range=[-20,20])
                    # plt.title('Mean')
                    # plt.xlabel("value")
                    # plt.ylabel("Frequency")
                    plt.legend()
                    if idx == len(prev_sols) - 1 and idx_filt==len(filter_string_orb)-1:
                        plt.savefig(XovOpt.get("tmpdir") + "histo_orbcorr_iters.png")
                ############################################

                sol_rms_iter.append(np.median(filtered_dict_iter))
                sol_avg_iter.append(np.mean(np.array(filtered_dict_iter)))

            iters_orbcorr.append(np.hstack([tst_id,sol_rms]))
            iters_orbcorr_avg.append(np.hstack([tst_id,sol_avg]))
            iters_orbcorr_it.append(np.hstack([tst_id,sol_rms_iter]))
            iters_orbcorr_avg_it.append(np.hstack([tst_id,sol_avg_iter]))

            if XovOpt.get("OrbRep") == 'lin':
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
                df_.columns = ['key', 'sol', 'std']
                df_[['orb', 'par']] = df_['key'].str.split('_', expand=True)
                df_.drop('key', axis=1, inplace=True)
                df_ = df_.loc[df_.par.isin(['dR/' + x for x in filter_string_orb])]

                if len(df_) > 0:
                    # df_[['orb','par']] = df_[['par','orb']].where(df_['par'] == None, df_[['orb','par']].values)
                    df_ = df_.replace(to_replace='None', value=np.nan).dropna()
                    df_sol = pd.pivot_table(df_, values=['sol', 'std'], index=['orb'], columns=['par'],
                                            aggfunc=np.sum).sol
                    if XovOpt.get("OrbRep") == 'lin':
                        # print("isol",isol)
                        # print(prev.pert_cloop.columns)
                        # print(df_sol.columns.values)
                        regex = re.compile(".*d[A,C,R]0$")
                        df_sol.columns = [x[:-1] if x in list(filter(regex.match, df_sol.columns)) else x for x in
                                          df_sol.columns.values]
                        # if idx == 0:
                        #     df_sol = df_sol.drop(df_sol.filter(regex=".*[A,C,R]1").columns,axis=1)
                        # print(df_sol.columns)

                    # prev.pert_cloop.drop(['dA1', 'dC1', 'dR1'], axis='columns', inplace=True)
                    # prev.pert_cloop.drop(['dRl', 'dPt'], axis='columns', inplace=True)

                    df_sol.columns = prev.pert_cloop.columns
                    _ = prev.pert_cloop.astype(float)
                    # _.columns = ['/' + k for k in _.columns]
                    #########################
                    if XovOpt.get("local"):
                        fig, ax1 = plt.subplots(nrows=1)
                        _.hist(ax=ax1,bins=np.arange(-120,120,10))
                        fig.savefig(XovOpt.get("tmpdir") + 'test_residuals_' + str(idx) + '.png')
                    ##########################
                    iters_orbres.append((_ ** 2).median(axis=0) ** 0.5)
                    iters_orbres_mean.append(_.median(axis=0))

        if XovOpt.get("sol4_glo")!=[None]:
            filter_string_glo = ["/" + x.split('/')[-1] for x in XovOpt.get("sol4_glo")] #["/dRA","/dDEC","/dPM","/dL","/dh2"]
            sol_glb = []
            std_glb = []
            for filt in filter_string_glo:
                filtered_dict = {k:v for (k,v) in prev.sol_dict['sol'].items() if filt in k}
                filtered_dict = list(filtered_dict.values())
                if len(filtered_dict)>0:
                    sol_glb.append(np.array(filtered_dict)[0])
                    filtered_dict_std = {k:v for (k,v) in prev.sol_dict['std'].items() if filt in k}
                    filtered_dict_std = list(filtered_dict_std.values())
                    std_glb.append(np.array(filtered_dict_std)[0])
                else:
                    sol_glb.append("0")
            iters_glocorr.append(np.hstack([tst_id,sol_glb]))

    if simulated_data:
        fig, [ax1,ax2,ax3,ax4,ax5] = plt.subplots(nrows=5,sharex=True)
    else:
        fig, [ax1,ax2,ax4] = plt.subplots(nrows=3,sharex=True)

    print("Total RMS for iters (iter,ltpl,m0,degf,%change): ")
    iters_rms = np.array(iters_rms)
    iters_rms = iters_rms[np.argsort(iters_rms[:,0])]
    #iters_rms[:,1:] = iters_rms[:,1:].round(3)
    wrmse = iters_rms[:,2].astype(float)
    perc_rms_change = np.concatenate([[None],np.abs((np.diff(wrmse)/wrmse[1:]*100.)).round(2)])
    print(np.concatenate([iters_rms,perc_rms_change[:,np.newaxis]],axis=1))

    ax1.plot(iters_rms[:,1].astype('float'),'-r')
    ax1.set_ylabel('rms (m)')
    ax1a = ax1.twinx()
    ax1a.plot(iters_rms[:,3].astype('float'),'.k')
    ax1a.set_ylabel('num of obs (post-screening)')

    if filter_string_orb != [None]:

        print("Total RMS/avg for solutions (orbpar): ")
        printout_list = [iters_orbcorr,iters_orbcorr_avg,iters_orbcorr_it,iters_orbcorr_avg_it]
        for idx,printout in enumerate(printout_list):
            printout = pd.DataFrame(printout,columns=np.hstack(['tst_id',filter_string_orb]))
            printout = printout.sort_values(by='tst_id').reset_index(drop=True).drop(columns='tst_id').astype('float') #.round(2)
            # printout[["/dRl","/dPt"]] *= 1e4
            if idx is 2:
                print("Total RMS/avg for single iter (orbpar): ")
            print(printout)
            printout_list[idx] = printout

            if idx is 0:
                printout.plot(ax=ax2)
                ax2.set_ylabel('rms (tot orb sol)')

        if XovOpt.get("OrbRep") == 'lin':
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
            # iters_orbres[["/dRl","/dPt"]] *= 1e5
            print(iters_orbres.dropna(axis=1))
            iters_orbres_mean = pd.DataFrame(iters_orbres_mean,columns=np.hstack(['tst_id',filter_string_orb]))
            iters_orbres_mean = iters_orbres_mean.sort_values(by='tst_id').reset_index(drop=True).drop(columns='tst_id').astype('float') #.round(2)
            # iters_orbres[["/dRl","/dPt"]] *= 1e5
            print(iters_orbres_mean.dropna(axis=1))

        if simulated_data and len(iters_orbres)>0:
            iters_orbres.plot(ax=ax3)
            ax3.get_legend().remove()
            ax3.set_ylabel('rms (orb res)')

    if XovOpt.get("sol4_glo")!=[None]:

        print("Cumulated solution (glopar): ")
        iters_glocorr = pd.DataFrame(iters_glocorr,columns=np.hstack(['tst_id',filter_string_glo]))
        iters_glocorr = iters_glocorr.sort_values(by='tst_id').reset_index(drop=True).drop(columns='tst_id').astype('float') #.round(2)
        iters_glocorr.columns = [x.split('/')[1] for x in iters_glocorr.columns.values]
        print(iters_glocorr)
        print("Iter improvement")
        print(iters_glocorr.diff())
        print("A posteriori error on global pars")
        m_X_iters = pd.DataFrame.from_dict(m_X_iters)
        m_X_iters.columns = [x.split('/')[1] for x in prev.parNames.keys()] #sol4_pars]
        print(m_X_iters[iters_glocorr.columns])
        tmp = ['dR/'+x for x in list(iters_glocorr.columns)]
        print("From least squares std:")
        print(tmp)
        print(np.array([prev.sol_dict_iter['std'][x] for x in tmp]).astype('float'))
        # print(prev.sol_dict['std'][iters_glocorr.columns])
        # print("Iter improvement (relative to formal error): ")
        # print((iters_glocorr.diff().abs()).div(m_X_iters[iters_glocorr.columns]))
        #

        if simulated_data and len(XovOpt.get("pert_cloop")['glo'])>0:
            print("Real residuals (glopar): ")
            #pert_cloop_glo = {'dRA': np.linalg.norm([0., 0.001, 0.000]),
            #                                     'dDEC':np.linalg.norm([-0., 0.0013, 0.000]),
            #                                     'dPM':np.linalg.norm([0, 0.001, 0.000]),
            #                                     'dL':0.03*np.linalg.norm([0.00993822,-0.00104581,-0.00010280,-0.00002364,-0.00000532]),
            #                     'dh2': 0.}
            pert_cloop_glo = XovOpt.get("pert_cloop")['glo']
            # pert_cloop_glo = [np.linalg.norm(x) for x in list(pert_cloop_glo.values())]
            #iters_glocorr.columns = [x[1:] for x in iters_glocorr.columns.values]
            pert_cloop_glo = { key:value for (key,value) in pert_cloop_glo.items() if key in iters_glocorr.columns.values}
            pert_cloop_glo = [np.sum(x) for x in list(pert_cloop_glo.values())]
            iters_glores = iters_glocorr.add(pert_cloop_glo, axis='columns').abs()
            iters_glores.columns = ['/'+x for x in iters_glores.columns.values]
            iters_glores = pd.DataFrame(iters_glores.reset_index(),columns=np.hstack(['tst_id',filter_string_glo]))
            iters_glores = iters_glores.sort_values(by='tst_id').reset_index(drop=True).drop(columns='tst_id').astype('float') #.round(2)
            iters_glores.columns = [x.split('/')[1] for x in iters_glores.columns.values]
            # get percentage of total perturbation still to be recovered
            print(iters_glores)
            print("Residual relative to formal errors")
            # print(iters_glores.abs())
            # print(m_X_iters)
            m_X_select = np.reshape([row["dR/"+x] for x in list(iters_glores.columns.values) for row in m_X_iters],(-1,len(m_X_iters))).T
            print((iters_glores.abs()).div(m_X_select))

        iters_glocorr.plot(ax=ax4)
        ax4.set_ylabel('sol (glo sol)')

        if simulated_data and len(XovOpt.get("pert_cloop")['glo'])>0:
            iters_glores.plot(logy=True,ax=ax5)
            ax5.get_legend().remove()
            ax5.set_ylabel('resid (as,/day)')

            print('to_be_recovered (sim mode, dRl, dPt, dRA, dDEC, dL in arcsec; dPM in arcsec/Julian year)')
            print(pert_cloop_glo)

        # print("Latest solution:")
        # last_sol = iters_glocorr[iters_glocorr.columns].iloc[-1]
        # last_std = m_X_iters[iters_glocorr.columns].iloc[-1]
        # if simulated_data and len(pert_cloop['glo'])>0:
        #   last_err = iters_glores.iloc[-1]
        #   last_sol = pd.concat([last_sol,last_std, last_err],axis=1)
        #   last_sol.columns = ['sol','std','err']
        # else:
        #   last_sol = pd.concat([last_sol,last_std],axis=1)
        #   last_sol.columns = ['sol','std']
        #
        # print(last_sol)

    # exit()
    # ax2.set_ylabel('rms (m)')
    # ax1a = ax1.twinx()
    # ax1a.plot(iters_rms[:, 2].astype('float'), '.k')
    # ax1a.set_ylabel('num of obs (post-screening)')
    #
    plt.savefig(XovOpt.get("tmpdir") + 'rms_iters_' + sol + '.png')
    plt.close()

    # print(iters_track_rms)
    plot_tracks_histo(iters_track_rms, filename=XovOpt.get("tmpdir") + '/histo_tracks_eval' + sol + '.png')

    # exit()

def add_xov_separation(tmp):
    tmp.xov.xovers['dist_max'] = tmp.xov.xovers.filter(regex='^dist_[A,B].*$').max(axis=1)
    tmp.xov.xovers['dist_min'] = tmp.xov.xovers.filter(regex='^dist_[A,B].*$').min(axis=1)
    tmp.xov.xovers['dist_minA'] = tmp.xov.xovers.filter(regex='^dist_A.*$').min(axis=1)
    tmp.xov.xovers['dist_minB'] = tmp.xov.xovers.filter(regex='^dist_B.*$').min(axis=1)
    tmp.xov.xovers['dist_min_mean'] = tmp.xov.xovers.filter(regex='^dist_min.*$').mean(axis=1)


if __name__ == '__main__':


    simulated_data = False #True

    # analyze_sol(sols=['KX1_0'], ref_sol='', subexp = '0res_1amp')
    # analyze_sol(sols=['AGTb_0','AGTb_1','AGTb_2','AGTb_3'], ref_sol='', subexp = '0res_1amp')
    # analyze_sol(sols=['KX2_0','KX2_1','KX2_2','KX2_3','KX2_4','KX2_5','KX2_6','KX2_7'], ref_sol='', subexp = '0res_1amp')
#    analyze_sol(sols=['KX3_0','KX3_1','KX3_2','KX3_3'], ref_sol='', subexp = '0res_1amp') # 'AGTP_0','AGS_0',
    #analyze_sol(sol='tp9_0', ref_sol='tp9_0', subexp = '3res_20amp')

    # check_iters(sol='tp4_0',subexp='3res_20amp')
    check_iters(sol='BS0_0',subexp='0res_1amp')

